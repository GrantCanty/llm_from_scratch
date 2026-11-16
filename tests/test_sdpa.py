import copy
import unittest

import torch

from decoder.self_attention import MultiHeadedAttention


def manual_attention(module, x, mask):
    b, length, _ = x.shape
    def project(layer):
        return layer(x).view(b, length, module.num_heads, module.head_dim).transpose(1, 2)
    q, k, v = [project(layer) for layer in (module.W_query, module.W_key, module.W_value)]
    allowed = module.mask[:length, :length].bool()[None, None]
    if mask is not None:
        allowed = allowed & mask[:, None, None, :].bool()
    scores = (q @ k.transpose(-2, -1)) / module.head_dim ** 0.5
    weights = scores.masked_fill(~allowed, -torch.inf).softmax(dim=-1)
    result = (weights @ v).transpose(1, 2).contiguous().view(b, length, module.d_out)
    return module.out_proj(result)


class SDPAtests(unittest.TestCase):
    def test_outputs_and_gradients_match_manual_attention(self):
        for mask in (None, torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])):
            with self.subTest(padded=mask is not None):
                module = MultiHeadedAttention(8, 8, 8, 0., 2, True)
                reference = copy.deepcopy(module)
                x = torch.randn(2, 4, 8, requires_grad=True)
                other = x.detach().clone().requires_grad_()
                actual = module(x, mask)
                expected = manual_attention(reference, other, mask)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                actual.square().sum().backward()
                expected.square().sum().backward()
                torch.testing.assert_close(x.grad, other.grad, atol=1e-6, rtol=1e-5)
                for p, q in zip(module.parameters(), reference.parameters()):
                    torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5)

    def test_evaluation_disables_dropout_and_preserves_causality(self):
        module = MultiHeadedAttention(8, 8, 8, .8, 2).eval()
        x = torch.randn(2, 4, 8)
        torch.testing.assert_close(module(x), module(x))
        torch.testing.assert_close(module(x), manual_attention(module, x, None))
        changed = x.clone()
        changed[:, 2:] += 10
        torch.testing.assert_close(module(x)[:, :2], module(changed)[:, :2])
        mask = torch.tensor([[1, 0, 1, 1], [1, 0, 1, 1]])
        changed = x.clone()
        changed[:, 1] += 10
        # Unchanged queries cannot attend to the modified masked key/value.
        torch.testing.assert_close(module(x, mask)[:, 2:], module(changed, mask)[:, 2:])
