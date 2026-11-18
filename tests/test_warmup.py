import unittest
from unittest.mock import patch

import torch

from decoder.model import GPTModel, train_model_simple_function


class RecordingAdamW(torch.optim.AdamW):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.used_lrs = []

    def step(self, closure=None):
        self.used_lrs.append([group['lr'] for group in self.param_groups])
        return super().step(closure)


class WarmupTests(unittest.TestCase):
    def run_training(self, **kwargs):
        torch.manual_seed(42)
        model = GPTModel(dict(vocab_size=16, context_length=3, emb_dim=8,
                              n_heads=2, n_layers=1, drop_rate=0., qkv_bias=True))
        params = list(model.parameters())
        optimizer = RecordingAdamW([{'params': params[:1], 'lr': .004},
                                    {'params': params[1:], 'lr': .002}])
        # Unequal scored-target counts and an empty batch exercise token accounting.
        batches = [(torch.tensor([[1, 2, 3]]), torch.tensor([[2, -100, -100]])),
                   (torch.tensor([[1]]), torch.tensor([[-100]])),
                   (torch.tensor([[3, 4, 5]]), torch.tensor([[4, 5, 6]]))]
        history = []
        results = train_model_simple_function(model, batches, batches, optimizer, 'cpu',
                                              2, 1, 3, update_history=history, **kwargs)
        return optimizer, history, results

    def test_update_warmup_continues_across_epochs_and_evaluations(self):
        optimizer, history, results = self.run_training(warmup_steps=3)
        expected = [[.004/3, .002/3], [.008/3, .004/3], [.004, .002], [.004, .002]]
        torch.testing.assert_close(torch.tensor(optimizer.used_lrs), torch.tensor(expected))
        self.assertEqual([row['scored_tokens'] for row in history], [1, 4, 5, 8])
        self.assertEqual([row['update'] for row in history], [1, 2, 3, 4])
        self.assertEqual([row['learning_rates'] for row in history], optimizer.used_lrs)
        self.assertEqual(results[6], [1, 4, 5, 8])
        self.assertTrue(all(torch.isfinite(torch.tensor(row['grad_norm'])) for row in history))
        self.assertTrue(all(state['step'].item() == 4 for state in optimizer.state.values()))

    def test_token_warmup_uses_actual_scored_targets(self):
        optimizer, _, _ = self.run_training(warmup_tokens=5)
        expected = [[.0008, .0004], [.0032, .0016], [.004, .002], [.004, .002]]
        torch.testing.assert_close(torch.tensor(optimizer.used_lrs), torch.tensor(expected))

    def test_update_measurements_count_scored_targets_and_skip_empty_batches(self):
        # Four updates, each taking two seconds. Evaluation and empty batches
        # must not consume timer samples or contribute scored tokens.
        with patch('decoder.model.time.perf_counter', side_effect=range(0, 16, 2)):
            _, history, _ = self.run_training()
        self.assertEqual([row['step_seconds'] for row in history], [2.] * 4)
        self.assertEqual([row['tokens_per_second'] for row in history], [.5, 1.5, .5, 1.5])
        for row in history:
            self.assertIsNone(row['peak_memory_allocated_bytes'])
            self.assertIsNone(row['peak_memory_reserved_bytes'])

    def test_disabled_and_single_update_warmup_use_peak_lr(self):
        for kwargs in ({}, {'warmup_steps': 1}):
            optimizer, _, _ = self.run_training(**kwargs)
            self.assertEqual(optimizer.used_lrs, [[.004, .002]] * 4)

    def test_invalid_warmup_is_rejected(self):
        for kwargs in ({'warmup_steps': -1}, {'warmup_tokens': 1.5},
                       {'warmup_steps': True}, {'warmup_steps': 2, 'warmup_tokens': 10}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_training(**kwargs)
