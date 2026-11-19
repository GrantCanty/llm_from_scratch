import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from decoder.decoder import Decoder
from decoder.model import GPTModel
from decoder.optimizers import build_optimizer


class CheckpointTests(unittest.TestCase):
    def make_decoder(self):
        decoder = Decoder.__new__(Decoder)
        decoder.seed = 42
        decoder.device = torch.device('cpu')
        decoder.GPT_CONFIG = dict(vocab_size=16, context_length=3, emb_dim=8,
                                  n_heads=2, n_layers=1, drop_rate=0., qkv_bias=True,
                                  flash_attn_func=None)
        decoder.reset_weights()
        return decoder

    def train(self, decoder, kind='adamw'):
        generator = torch.Generator().manual_seed(17)
        dataset = torch.utils.data.TensorDataset(
            torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]]),
            torch.tensor([[2, -100, -100], [3, 4, 5], [4, 5, -100]]))
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, generator=generator)
        return decoder.train(loader, list(loader), 1, drop_rate=0., optimizer_type=kind,
                             warmup_tokens=12, accumulation_steps=2, adaptive_clip=True,
                             clip_window=3, clip_min_history=1, clip_multiplier=.5)

    def test_full_files_capture_both_optimizers_and_exact_training_state(self):
        for kind in ('adamw', 'muon'):
            if kind == 'muon' and not hasattr(torch.optim, 'Muon'):
                continue
            decoder = self.make_decoder()
            results = self.train(decoder, kind)
            optimizer = decoder.optimizer
            with tempfile.TemporaryDirectory() as directory:
                path = decoder.save_checkpoint(Path(directory) / 'run', metadata={'tokenizer': 'test-v1'})
                payload = torch.load(path, weights_only=True)
            self.assertEqual(payload['format_version'], 1)
            self.assertEqual(payload['metadata']['tokenizer'], 'test-v1')
            progress = payload['training_state']
            self.assertEqual(progress['updates'], 2)
            self.assertEqual(progress['scored_tokens'], 6)
            self.assertEqual(progress['microbatches_seen'], 3)
            self.assertEqual(progress['batches_consumed_epoch'], 3)
            self.assertTrue(progress['epoch_complete'])
            self.assertTrue(progress['checkpoint_safe'])
            self.assertEqual(progress['adaptive_clipping']['updates'], 2)
            self.assertEqual(len(progress['adaptive_clipping']['norms']), 2)
            self.assertEqual(progress['warmup']['tokens'], 12)
            self.assertIn('loader', progress['epoch_start_loader_generators'])
            self.assertIn('sampler', progress['loader_generators'])
            self.assertEqual(payload['training_config']['optimizer_type'], kind)
            self.assertEqual(len(payload['training_history']), 2)
            self.assertEqual(payload['training_results'][6], results[6])
            self.assertIs(decoder.optimizer, optimizer)  # Saving never cleans it up.
            self.assertTrue(all(p.grad is None for p in decoder.decoder_model.parameters()))
            # Independently load child state dicts and check the next update.
            # This verifies the file, without implementing a resume API.
            restored = GPTModel(decoder.GPT_CONFIG)
            restored.load_state_dict(payload['model_state_dict'])
            other = build_optimizer(restored, optimizer_type=kind)
            other.load_state_dict(payload['optimizer_state_dict'])
            for p, q in zip(decoder.decoder_model.parameters(), restored.parameters()):
                p.grad = torch.ones_like(p)
                q.grad = torch.ones_like(q)
            decoder.optimizer.step()
            other.step()
            for p, q in zip(decoder.decoder_model.parameters(), restored.parameters()):
                torch.testing.assert_close(p, q, rtol=0, atol=0)
            decoder.cleanup_memory(empty_cache=False)

    def test_weights_only_before_training_and_after_cleanup(self):
        decoder = self.make_decoder()
        with tempfile.TemporaryDirectory() as directory:
            path = decoder.save_checkpoint(Path(directory) / 'weights', checkpoint=False)
            expected = torch.load(path, weights_only=True)
            self.assertEqual(set(expected), set(decoder.decoder_model.state_dict()))
            self.assertTrue(all(t.device.type == 'cpu' for t in expected.values()))
            other = self.make_decoder()
            other.load_weights(path)
            self.train(decoder)
            full_path = decoder.save_checkpoint(Path(directory) / 'full')
            with self.assertRaisesRegex(ValueError, 'full checkpoint'):
                other.load_weights(full_path)
            decoder.cleanup_memory(empty_cache=False)
            decoder.save_checkpoint(path, checkpoint=False)
            with self.assertRaisesRegex(RuntimeError, 'No retained'):
                decoder.save_checkpoint(full_path)

    def test_only_cleanup_or_reset_discards_retained_training_state(self):
        decoder = self.make_decoder()
        self.train(decoder)
        optimizer = decoder.optimizer
        with self.assertRaisesRegex(RuntimeError, 'retained'):
            self.train(decoder)
        self.assertIs(decoder.optimizer, optimizer)
        decoder.reset_weights()
        self.assertIsNone(decoder.optimizer)
        self.assertFalse(optimizer.state)
        self.assertIsNone(decoder.training_state)

    def test_failed_microbatch_disallows_full_checkpoint_but_retains_state(self):
        decoder = self.make_decoder()
        with patch('decoder.model.utils.calc_loss_batch', side_effect=RuntimeError('failed forward')):
            with self.assertRaisesRegex(RuntimeError, 'failed forward'):
                self.train(decoder)
        self.assertIsNotNone(decoder.optimizer)
        self.assertFalse(decoder.training_state['checkpoint_safe'])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'boundary'):
                decoder.save_checkpoint(Path(directory) / 'failed')
            decoder.save_checkpoint(Path(directory) / 'weights', checkpoint=False)

    def test_completed_update_survives_evaluation_failure_and_skips_runtime_callable(self):
        decoder = self.make_decoder()
        decoder.GPT_CONFIG['flash_attn_func'] = lambda: None
        with patch('decoder.model.evaluate_model', side_effect=RuntimeError('evaluation failed')):
            with self.assertRaisesRegex(RuntimeError, 'evaluation failed'):
                self.train(decoder)
        self.assertTrue(decoder.training_state['checkpoint_safe'])
        self.assertEqual(decoder.training_state['updates'], 1)
        with tempfile.TemporaryDirectory() as directory:
            path = decoder.save_checkpoint(Path(directory) / 'safe_boundary')
            payload = torch.load(path, weights_only=True)
        self.assertNotIn('flash_attn_func', payload['model_config'])
        self.assertIn('name', payload['attention_kernel'])
        self.assertEqual(payload['training_state']['status'], 'interrupted_or_failed')

    def test_atomic_failure_preserves_previous_file_and_live_state(self):
        decoder = self.make_decoder()
        self.train(decoder)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pth'
            decoder.save_checkpoint(path)
            original = path.read_bytes()
            def fail(payload, handle):
                handle.write(b'incomplete')
                raise OSError('disk full')
            with patch('decoder.checkpoint.torch.save', side_effect=fail):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    decoder.save_checkpoint(path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            self.assertTrue(decoder.optimizer.state)

    def test_rng_capture_is_loadable_and_does_not_advance_rng(self):
        decoder = self.make_decoder()
        self.train(decoder)
        random.seed(5)
        np.random.seed(6)
        torch.manual_seed(7)
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        with tempfile.TemporaryDirectory() as directory:
            path = decoder.save_checkpoint(Path(directory) / 'rng')
            payload = torch.load(path, weights_only=True)
        self.assertEqual(payload['rng_state']['python'], python_state)
        self.assertEqual(payload['rng_state']['numpy'][1], numpy_state[1].tolist())
        torch.testing.assert_close(payload['rng_state']['torch_cpu'], torch_state)
        self.assertEqual(random.getstate(), python_state)
        self.assertEqual(np.random.get_state()[1].tolist(), numpy_state[1].tolist())
        torch.testing.assert_close(torch.get_rng_state(), torch_state)
