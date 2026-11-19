import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset, IterableDataset

from decoder.decoder import Decoder


class ResumeTests(unittest.TestCase):
    def make_decoder(self, device='cpu'):
        decoder = Decoder.__new__(Decoder)
        decoder.seed = 42
        decoder.device = torch.device(device)
        decoder.GPT_CONFIG = dict(vocab_size=16, context_length=3, emb_dim=8,
                                  n_heads=2, n_layers=1, drop_rate=.2, qkv_bias=True,
                                  flash_attn_func=None)
        decoder.reset_weights()
        return decoder

    def loaders(self, workers=0, batch_size=1):
        inputs = torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
                               [5, 6, 7], [6, 7, 8], [7, 8, 9]])
        targets = torch.tensor([[2, -100, -100], [3, 4, 5], [4, 5, -100], [-100]*3,
                                [6, 7, 8], [7, -100, -100], [8, 9, -100]])
        data = TensorDataset(inputs, targets)
        train = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=workers,
                           generator=torch.Generator().manual_seed(17))
        val = DataLoader(data, batch_size=2, shuffle=True,
                         generator=torch.Generator().manual_seed(19))
        return train, val

    def options(self, kind):
        return dict(num_epochs=2, optimizer_type=kind, drop_rate=.2,
                    accumulation_steps=2, warmup_tokens=50, adaptive_clip=True,
                    clip_window=3, clip_min_history=1, clip_multiplier=.7,
                    eval_freq=1, eval_iter=2, update_norm_interval=1)

    def assert_equivalent(self, expected, actual):
        for name, value in expected.decoder_model.state_dict().items():
            torch.testing.assert_close(value, actual.decoder_model.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(len(expected.training_history), len(actual.training_history))
        for a, b in zip(expected.training_history, actual.training_history):
            for key in ('update', 'scored_tokens', 'microbatches_seen', 'learning_rates',
                        'grad_norm', 'clip_threshold', 'clip_factor', 'loss'):
                self.assertEqual(a[key], b[key], key)
        for key in ('updates', 'scored_tokens', 'adaptive_clipping', 'epoch', 'epoch_complete'):
            self.assertEqual(expected.training_state[key], actual.training_state[key], key)
        # Includes validation/train-evaluation outputs with dropout and shuffled loaders.
        for index in (0, 2, 4, 5, 6):
            self.assertEqual(expected.training_results[index], actual.training_results[index])
        for key in ('loader', 'sampler'):
            torch.testing.assert_close(expected.training_state['loader_generators'][key],
                                       actual.training_state['loader_generators'][key], rtol=0, atol=0)

    def test_mid_epoch_file_resume_matches_uninterrupted_with_dropout_and_shuffle(self):
        for kind in ('adamw', 'muon'):
            if kind == 'muon' and not hasattr(torch.optim, 'Muon'):
                continue
            baseline = self.make_decoder()
            baseline.train(*self.loaders(), **self.options(kind))
            paused = self.make_decoder()
            paused.train(*self.loaders(), **self.options(kind), max_updates=2)
            self.assertEqual(paused.training_state['status'], 'stopped')
            self.assertFalse(paused.training_state['epoch_complete'])
            with tempfile.TemporaryDirectory() as directory:
                path = paused.save_checkpoint(Path(directory) / 'run', metadata={'dataset': 'fixed'})
                restored = self.make_decoder()
                # Load into a deliberately different vocabulary/config: file config is authoritative.
                restored.GPT_CONFIG['vocab_size'] = 32
                restored.resume(path, *self.loaders(), expected_metadata={'dataset': 'fixed'})
                self.assert_equivalent(baseline, restored)
                next_path = restored.save_checkpoint(Path(directory) / 'again')
                self.assertEqual(torch.load(next_path, weights_only=True)['metadata'], {'dataset': 'fixed'})

    def test_epoch_boundary_extension_and_in_memory_continuation(self):
        baseline = self.make_decoder()
        baseline.train(*self.loaders(), **self.options('adamw'))
        paused = self.make_decoder()
        options = self.options('adamw')
        options['num_epochs'] = 1
        train, val = self.loaders()
        paused.train(train, val, **options)
        optimizer = paused.optimizer
        torch.rand(20)  # Notebook activity must not perturb retained training RNG.
        paused.resume(None, train, val, num_epochs=2)
        self.assertIs(paused.optimizer, optimizer)
        self.assertEqual(paused.training_config['num_epochs'], 2)
        self.assert_equivalent(baseline, paused)
        with self.assertRaisesRegex(ValueError, 'already complete'):
            paused.resume(None, train, val)

    def test_file_epoch_boundary_extension(self):
        baseline = self.make_decoder()
        baseline.train(*self.loaders(), **self.options('adamw'))
        paused = self.make_decoder()
        options = self.options('adamw')
        options['num_epochs'] = 1
        paused.train(*self.loaders(), **options)
        with tempfile.TemporaryDirectory() as directory:
            path = paused.save_checkpoint(Path(directory) / 'epoch')
            restored = self.make_decoder()
            restored.resume(path, *self.loaders(), num_epochs=2)
        self.assert_equivalent(baseline, restored)

    def test_repeated_save_resume_at_last_batch_boundary(self):
        baseline = self.make_decoder()
        baseline.train(*self.loaders(), **self.options('adamw'))
        paused = self.make_decoder()
        paused.train(*self.loaders(), **self.options('adamw'), max_updates=1)
        with tempfile.TemporaryDirectory() as directory:
            for _ in range(2):
                path = paused.save_checkpoint(Path(directory) / 'run')
                restored = self.make_decoder()
                restored.resume(path, *self.loaders(), max_updates=1)
                paused = restored
            path = paused.save_checkpoint(Path(directory) / 'run')
            restored = self.make_decoder()
            restored.resume(path, *self.loaders())
        self.assert_equivalent(baseline, restored)

    def test_random_sampler_without_explicit_generator_and_step_warmup(self):
        def loaders():
            train, val = self.loaders()
            return (DataLoader(train.dataset, batch_size=1, shuffle=True),
                    DataLoader(val.dataset, batch_size=2, shuffle=False))
        options = self.options('adamw')
        options['warmup_tokens'] = 0
        options['warmup_steps'] = 5
        baseline = self.make_decoder()
        baseline.train(*loaders(), **options)
        paused = self.make_decoder()
        paused.train(*loaders(), **options, max_updates=1)
        with tempfile.TemporaryDirectory() as directory:
            path = paused.save_checkpoint(Path(directory) / 'run')
            restored = self.make_decoder()
            restored.resume(path, *loaders())
        for name, value in baseline.decoder_model.state_dict().items():
            torch.testing.assert_close(value, restored.decoder_model.state_dict()[name], rtol=0, atol=0)
        self.assertEqual([r['learning_rates'] for r in baseline.training_history],
                         [r['learning_rates'] for r in restored.training_history])

    def test_fixed_batch_lists_and_legacy_epoch_checkpoint(self):
        def batches():
            train, val = self.loaders()
            return list(train), list(val)
        options = self.options('adamw')
        baseline = self.make_decoder()
        train, val = batches()
        baseline.train(train, val, **options)
        paused = self.make_decoder()
        train, val = batches()
        paused.train(train, val, **{**options, 'num_epochs': 1})
        with tempfile.TemporaryDirectory() as directory:
            path = paused.save_checkpoint(Path(directory) / 'legacy')
            payload = torch.load(path, weights_only=True)
            payload['training_state'].pop('loader_specs')
            payload['training_state'].pop('evaluation_results')
            payload['training_state'].pop('val_loader_generators')
            torch.save(payload, path)
            restored = self.make_decoder()
            train, val = batches()
            restored.resume(path, train, val, num_epochs=2)
        for name, value in baseline.decoder_model.state_dict().items():
            torch.testing.assert_close(value, restored.decoder_model.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(baseline.training_results[6], restored.training_results[6])

    def test_resumed_evaluation_failure_can_be_checkpointed_again(self):
        paused = self.make_decoder()
        paused.train(*self.loaders(), **self.options('adamw'), max_updates=1)
        with tempfile.TemporaryDirectory() as directory:
            path = paused.save_checkpoint(Path(directory) / 'run')
            restored = self.make_decoder()
            with patch('decoder.model.evaluate_model', side_effect=RuntimeError('evaluation failed')):
                with self.assertRaisesRegex(RuntimeError, 'evaluation failed'):
                    restored.resume(path, *self.loaders())
            self.assertEqual(restored.training_state['updates'], 2)
            self.assertTrue(restored.training_state['checkpoint_safe'])
            self.assertIsNotNone(restored.training_results)
            saved = restored.save_checkpoint(Path(directory) / 'recoverable')
            payload = torch.load(saved, weights_only=True)
            self.assertEqual(payload['training_state']['updates'], 2)
            self.assertTrue(payload['optimizer_state_dict']['state'])

    @unittest.skipUnless(torch.cuda.is_available(), 'Requires CUDA')
    def test_cuda_file_resume(self):
        for kind in ('adamw', 'muon'):
            if kind == 'muon' and not hasattr(torch.optim, 'Muon'):
                continue
            baseline = self.make_decoder('cuda')
            baseline.train(*self.loaders(), **self.options(kind))
            paused = self.make_decoder('cuda')
            paused.train(*self.loaders(), **self.options(kind), max_updates=1)
            with tempfile.TemporaryDirectory() as directory:
                path = paused.save_checkpoint(Path(directory) / 'gpu')
                restored = self.make_decoder('cuda')
                restored.resume(path, *self.loaders())
            for name, value in baseline.decoder_model.state_dict().items():
                torch.testing.assert_close(value, restored.decoder_model.state_dict()[name],
                                           rtol=1e-5, atol=1e-6)

    def test_rejects_unsupported_or_changed_data_and_weights_files_without_mutation(self):
        paused = self.make_decoder()
        paused.train(*self.loaders(), **self.options('adamw'), max_updates=1)
        with tempfile.TemporaryDirectory() as directory:
            path = paused.save_checkpoint(Path(directory) / 'run', metadata={'tokenizer': 'v1'})
            restored = self.make_decoder()
            before = copy.deepcopy(restored.decoder_model.state_dict())
            for train, val, options in ((*self.loaders(batch_size=2), {}),
                                        (*self.loaders(workers=1), {}),
                                        (*self.loaders(), {'expected_metadata': {'tokenizer': 'v2'}})):
                with self.assertRaises(ValueError):
                    restored.resume(path, train, val, **options)
            for name, value in before.items():
                torch.testing.assert_close(value, restored.decoder_model.state_dict()[name], rtol=0, atol=0)
            self.assertIsNone(restored.optimizer)
            weights = paused.save_checkpoint(Path(directory) / 'weights', checkpoint=False)
            with self.assertRaisesRegex(ValueError, 'full checkpoint'):
                restored.resume(weights, *self.loaders())
            with self.assertRaisesRegex(RuntimeError, 'retained'):
                paused.resume(path, *self.loaders())
