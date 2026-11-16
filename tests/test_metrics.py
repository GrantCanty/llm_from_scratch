import unittest

import torch

from decoder.utils import calc_metrics_loader


class MetricsTests(unittest.TestCase):
    def test_padding_is_ignored_and_batches_are_weighted_by_token_count(self):
        class FixedModel(torch.nn.Module):
            def forward(self, ids):
                return torch.nn.functional.one_hot(ids, num_classes=3).float() * 2

        batches = [
            (torch.tensor([[0, 1, 2]]), torch.tensor([[0, 1, -100]])),
            (torch.tensor([[0]]), torch.tensor([[2]])),
        ]
        result = calc_metrics_loader(batches, FixedModel(), 'cpu')
        logits = FixedModel()(torch.tensor([[0, 1, 0]]))
        expected_loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), torch.tensor([0, 1, 2])
        ).item()
        self.assertAlmostEqual(result['accuracy'], 2 / 3)
        self.assertAlmostEqual(result['loss'], expected_loss, places=6)
        limited = calc_metrics_loader(batches, FixedModel(), 'cpu', num_batches=1)
        self.assertEqual(limited['accuracy'], 1.0)
