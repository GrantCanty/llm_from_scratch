import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset

from decoder.decoder import Decoder
from decoder.sampling import ResumableBatchSampler


class CountingDataset(Dataset):
    def __init__(self, length=11):
        self.length = length
        self.reads = []

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self.reads.append(index)
        token = index % 10 + 1
        inputs = torch.tensor([token, token + 1, token + 2])
        targets = torch.tensor([token + 1, token + 2, -100])
        if index in (2, 3):
            targets.fill_(-100)
        return inputs, targets


class SamplingTests(unittest.TestCase):
    def make_decoder(self):
        decoder = Decoder.__new__(Decoder)
        decoder.seed = 42
        decoder.device = torch.device('cpu')
        decoder.GPT_CONFIG = dict(vocab_size=16, context_length=3, emb_dim=8,
                                  n_heads=2, n_layers=1, drop_rate=.2, qkv_bias=True)
        decoder.reset_weights()
        return decoder

    def loader(self, *, length=11, batch_size=2, shuffle=True, drop_last=False):
        dataset = CountingDataset(length)
        sampler = ResumableBatchSampler(dataset, batch_size, shuffle=shuffle,
                                        seed=17, drop_last=drop_last)
        return DataLoader(dataset, batch_sampler=sampler,
                          generator=torch.Generator().manual_seed(19))

    def validation(self):
        return [tuple(t.unsqueeze(0) for t in CountingDataset()[i]) for i in (0, 1)]

    def options(self, optimizer_type='adamw'):
        return dict(num_epochs=2, optimizer_type=optimizer_type, drop_rate=.2,
                    accumulation_steps=2, warmup_tokens=70, adaptive_clip=True,
                    clip_window=3, clip_min_history=1, clip_multiplier=.7,
                    eval_freq=1, eval_iter=2, update_norm_interval=1)

    def test_requested_indices_are_not_committed_until_success(self):
        sampler = ResumableBatchSampler(list(range(10)), 2, shuffle=False)
        iterator = iter(sampler)
        self.assertEqual(next(iterator), [0, 1])
        self.assertEqual(next(iterator), [2, 3])
        self.assertEqual(next(iterator), [4, 5])  # Simulated prefetch ahead of training.
        self.assertEqual(sampler.state_dict()['position'], 0)
        sampler.commit_batches(1)
        self.assertEqual(sampler.position, 2)
        restored = ResumableBatchSampler(list(range(10)), 2, shuffle=False)
        restored.load_state_dict(sampler.state_dict())
        self.assertEqual(next(iter(restored)), [2, 3])
        with self.assertRaises(ValueError):
            restored.commit_batches(5)

    def test_evaluation_is_read_only_and_partial_drop_last_policies(self):
        for drop_last in (False, True):
            sampler = ResumableBatchSampler(list(range(5)), 2, shuffle=False, drop_last=drop_last)
            iterator = iter(sampler)
            next(iterator)
            sampler.commit_batches(1)
            with sampler.evaluation():
                self.assertEqual(list(sampler), [[0, 1], [2, 3]] + ([] if drop_last else [[4]]))
            self.assertEqual(sampler.position, 2)
            self.assertEqual(list(iterator), [[2, 3]] + ([] if drop_last else [[4]]))
            sampler.commit_batches(1 if drop_last else 2)
            self.assertEqual(sampler.position, 4 if drop_last else 5)
            self.assertEqual(list(sampler), [])
            sampler.set_epoch(1)
            self.assertEqual(sampler.position, 0)

    def test_resume_reads_only_unconsumed_suffix(self):
        decoder = self.make_decoder()
        train = self.loader(batch_size=1, shuffle=False)
        # Use a fixed validation list and avoid train-loss evaluation reads in this access test.
        validation = self.validation()
        evaluation = (1., torch.ones(3), 1., torch.ones(3), 2., .5)
        with patch('decoder.model.evaluate_model', return_value=evaluation):
            decoder.train(train, validation, **{**self.options(), 'num_epochs': 1}, max_updates=2)
        committed = train.batch_sampler.position
        self.assertGreater(committed, 4)  # Empty-target examples were also consumed.
        self.assertEqual(decoder.training_state['examples_consumed_epoch'], committed)
        self.assertEqual(train.dataset.reads, list(range(committed)))
        with tempfile.TemporaryDirectory() as directory:
            path = decoder.save_checkpoint(Path(directory) / 'position')
            payload = torch.load(path, weights_only=True)
            self.assertEqual(payload['training_state']['sampler_state']['position'], committed)
            restored = self.make_decoder()
            next_loader = self.loader(batch_size=1, shuffle=False)
            with patch('decoder.model.evaluate_model', return_value=evaluation):
                restored.resume(path, next_loader, validation)
        self.assertEqual(next_loader.dataset.reads, list(range(committed, 11)))
        self.assertEqual(restored.training_state['examples_consumed'], 11)
        self.assertEqual(next_loader.batch_sampler.position, 11)

    def test_direct_resume_matches_training_with_shuffle_dropout_and_both_optimizers(self):
        for kind in ('adamw', 'muon'):
            if kind == 'muon' and not hasattr(torch.optim, 'Muon'):
                continue
            baseline = self.make_decoder()
            baseline_loader = self.loader()
            validation = self.validation()
            baseline.train(baseline_loader, validation, **self.options(kind))
            paused = self.make_decoder()
            loader = self.loader()
            paused.train(loader, validation, **self.options(kind), max_updates=1)
            committed = loader.batch_sampler.position
            order = loader.batch_sampler.order.clone()
            with tempfile.TemporaryDirectory() as directory:
                path = paused.save_checkpoint(Path(directory) / 'run')
                restored = self.make_decoder()
                resumed_loader = self.loader()
                restored.resume(path, resumed_loader, validation)
            for name, value in baseline.decoder_model.state_dict().items():
                torch.testing.assert_close(value, restored.decoder_model.state_dict()[name], rtol=0, atol=0)
            for a, b in zip(baseline.training_history, restored.training_history):
                for field in ('update', 'scored_tokens', 'loss', 'grad_norm', 'clip_threshold', 'learning_rates'):
                    self.assertEqual(a[field], b[field])
            self.assertEqual(baseline.training_results[0], restored.training_results[0])
            self.assertEqual(restored.training_state['examples_consumed'], 22)
            # Evaluation should not change the training sampler's position.
            self.assertEqual(resumed_loader.batch_sampler.position, 11)
            self.assertGreater(committed, 0)

    def test_failed_update_leaves_committed_position_unchanged(self):
        decoder = self.make_decoder()
        train = self.loader(shuffle=False)
        with patch('decoder.optimizers.torch.optim.AdamW.step', side_effect=RuntimeError('failed step')):
            with self.assertRaisesRegex(RuntimeError, 'failed step'):
                decoder.train(train, self.validation(), **self.options())
        self.assertEqual(train.batch_sampler.position, 0)
        self.assertFalse(decoder.training_state['checkpoint_safe'])

    def test_order_and_batch_mismatch_rejected(self):
        sampler = ResumableBatchSampler(list(range(5)), 2)
        sampler.set_epoch(0)
        state = sampler.state_dict()
        wrong = dict(state, order=torch.zeros(5, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            sampler.load_state_dict(wrong)
        different = ResumableBatchSampler(list(range(5)), 1)
        with self.assertRaisesRegex(ValueError, 'configuration'):
            different.load_state_dict(state)
