import unittest

import torch

from decoder.capacity import _validate, profile_training_configs


class CapacityTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(vocab_size=32, context_length=16, emb_dim=16,
                        n_heads=2, n_layers=1, drop_rate=.1, qkv_bias=True)

    def test_invalid_dimensions(self):
        with self.assertRaisesRegex(ValueError, 'divisible'):
            _validate({**self.cfg, 'emb_dim': 15}, 2)
        with self.assertRaisesRegex(ValueError, 'batch_size'):
            _validate(self.cfg, 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'Requires CUDA')
    def test_full_training_probe_releases_tensors_and_preserves_rng(self):
        before = torch.random.get_rng_state().clone()
        baseline = torch.cuda.memory_allocated()
        results = profile_training_configs(
            self.cfg, [{'batch_size': 2}, {'batch_size': 1, 'context_length': 8}],
            warmup_steps=1, measured_steps=1,
        )
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        for row in results:
            self.assertEqual(row['status'], 'ok', row)
            self.assertGreater(row['tokens_per_second'], 0)
            self.assertLessEqual(row['allocated_after_gib'] * 2**30, baseline + 1024**2)
