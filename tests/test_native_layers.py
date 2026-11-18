import math
import unittest

import torch

from decoder.model import GELU, LayerNorm, GPTModel


class NativeLayerTests(unittest.TestCase):
    def test_forward_and_backward_match_baseline(self):
        torch.manual_seed(12)
        for layer in (GELU().double(), LayerNorm(8).double()):
            x = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
            reference_x = x.detach().clone().requires_grad_()
            if isinstance(layer, torch.nn.GELU):
                expected = .5 * reference_x * (1 + torch.tanh(
                    math.sqrt(2 / math.pi) * (reference_x + .044715 * reference_x ** 3)))
            else:
                with torch.no_grad():
                    layer.weight.uniform_(.5, 1.5)
                    layer.bias.uniform_(-.2, .2)
                expected = ((reference_x - reference_x.mean(-1, keepdim=True)) /
                            torch.sqrt(reference_x.var(-1, unbiased=False, keepdim=True) + 1e-8))
                expected = expected * layer.weight + layer.bias
            actual = layer(x)
            torch.testing.assert_close(actual, expected)
            upstream = torch.randn_like(actual)
            actual_grad = torch.autograd.grad(actual, (x, *layer.parameters()), upstream)
            expected_grad = torch.autograd.grad(expected, (reference_x, *layer.parameters()), upstream)
            for actual_value, expected_value in zip(actual_grad, expected_grad):
                torch.testing.assert_close(actual_value, expected_value)

    def test_full_model_loads_legacy_and_native_weights_strictly(self):
        cfg = dict(vocab_size=16, context_length=4, emb_dim=8,
                   n_heads=2, n_layers=1, drop_rate=0., qkv_bias=True)
        model = GPTModel(cfg).eval()
        weights = model.state_dict()
        legacy = {key.replace('.weight', '.scale').replace('.bias', '.shift')
                  if ('norm1.' in key or 'norm2.' in key or 'final_norm.' in key)
                  else key: value.clone() for key, value in weights.items()}
        restored = GPTModel(cfg).eval()
        restored.load_state_dict(legacy, strict=True)
        ids = torch.tensor([[1, 2, 3]])
        torch.testing.assert_close(restored(ids), model(ids), rtol=0, atol=0)
        restored.load_state_dict(weights, strict=True)
        self.assertEqual(sum(p.numel() for p in model.parameters()),
                         sum(p.numel() for p in restored.parameters()))
