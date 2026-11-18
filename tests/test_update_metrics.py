import unittest
from unittest.mock import patch

import torch

from decoder.model import train_model_simple_function
from decoder.update_metrics import snapshot_parameters, measure_updates


class UpdateMetricsTests(unittest.TestCase):
    def test_actual_adamw_update_includes_decay_and_uses_previous_weights(self):
        model = torch.nn.Linear(2, 1)
        with torch.no_grad():
            model.weight.fill_(2.)
            model.bias.zero_()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.1, weight_decay=.2)
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        snapshot = snapshot_parameters(model, optimizer)
        optimizer.step()
        metrics = measure_updates(snapshot, optimizer)
        weight = metrics['parameters']['weight']
        # Adam's first adaptive update is .1, plus .04 decoupled decay.
        self.assertAlmostEqual(weight['update_to_weight_ratio'], .07, places=6)
        self.assertAlmostEqual(weight['weight_norm'], 2 * 2 ** .5, places=6)
        self.assertEqual(weight['optimizer'], 'AdamW')
        self.assertEqual(metrics['parameters']['bias']['parameter_kind'], 'bias_or_norm')
        self.assertTrue(torch.isfinite(torch.tensor(metrics['parameters']['bias']['update_to_weight_ratio'])))
        self.assertAlmostEqual(metrics['update_norm'], (.14 ** 2 * 2 + .1 ** 2) ** .5, places=6)

    def run_loop(self, interval):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        batches = [(torch.ones(1, 1), torch.ones(1, 1, dtype=torch.long))] * 4
        history = []
        evaluation = (1., torch.ones(1), 1., torch.ones(1), 2., .5)
        with patch('decoder.model.utils.calc_loss_batch', side_effect=lambda *a, **k: model(torch.ones(1, 1)).square().sum()), \
             patch('decoder.model.evaluate_model', return_value=evaluation):
            result = train_model_simple_function(model, batches, batches, optimizer, 'cpu',
                                                1, 100, 1, update_history=history,
                                                update_norm_interval=interval)
        self.assertEqual(len(result), 8)
        return history

    def test_sampling_and_disable(self):
        history = self.run_loop(3)
        self.assertEqual([r['update'] for r in history if r['update_norms'] is not None], [1, 3])
        self.assertTrue(all(r['update_norms'] is None for r in self.run_loop(0)))

    def test_invalid_interval(self):
        for value in (-1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.run_loop(value)
