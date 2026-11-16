import copy
import importlib
import unittest

import torch
import torch.nn.functional as F

from decoder.model import GPTModel
from decoder.self_attention import MultiHeadedAttention


class FA4IntegrationTests(unittest.TestCase):
    def test_adapter_layout_padding_gradients_and_dropout_argument(self):
        calls = []
        def fake_fa4(q, k, v, *, causal, dropout_p):
            calls.append((q.shape, causal, dropout_p))
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                is_causal=causal, dropout_p=dropout_p,
            ).transpose(1, 2)
            return out, None

        module = MultiHeadedAttention(8, 8, 4, .1, 2,
                                    attention_backend='fa4', flash_attn_func=fake_fa4)
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
        tensors = [torch.randn(2, 4, 2, 4, requires_grad=True) for _ in range(3)]
        reference = [t.detach().clone().requires_grad_() for t in tensors]
        actual = module._fa4_attention(*tensors, mask, 0.)
        allowed = torch.ones(4, 4, dtype=torch.bool).tril()[None, None] & mask[:, None, None, :]
        expected = F.scaled_dot_product_attention(
            *[t.transpose(1, 2) for t in reference], attn_mask=allowed,
        ).transpose(1, 2)
        torch.testing.assert_close(actual[mask], expected[mask])
        actual[mask].square().sum().backward()
        expected[mask].square().sum().backward()
        for a, b in zip(tensors, reference):
            torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-5)
        self.assertEqual(calls[0], (torch.Size([2, 4, 2, 4]), True, 0.))
        module._fa4_attention(*tensors, mask, .1)
        self.assertEqual(calls[-1][-1], .1)
        with self.assertRaisesRegex(ValueError, 'right-padded'):
            module._fa4_attention(*tensors, torch.tensor([[1, 0, 1, 1]] * 2).bool(), 0.)

    def test_configuration_and_device_errors(self):
        with self.assertRaisesRegex(ValueError, 'Pass flash_attn_func'):
            MultiHeadedAttention(8, 8, 4, 0., 2, attention_backend='fa4')
        module = MultiHeadedAttention(8, 8, 4, 0., 2,
                                    attention_backend='fa4', flash_attn_func=lambda: None)
        with self.assertRaisesRegex(ValueError, 'CUDA'):
            module(torch.randn(1, 4, 8))

    @unittest.skipUnless(torch.cuda.is_available(), 'Requires cloud CUDA GPU and FA4 installation')
    def test_real_fa4_model_forward_backward(self):
        fa4 = importlib.import_module('torch-cuda')
        cfg = dict(vocab_size=32, context_length=256, emb_dim=128,
                   n_heads=2, n_layers=1, drop_rate=.1, qkv_bias=True,
                   attention_backend='fa4', flash_attn_func=fa4.flash_attn_func)
        model = GPTModel(cfg).cuda()
        ids = torch.randint(0, 32, (2, 256), device='cuda')
        mask = torch.ones_like(ids, dtype=torch.bool)
        mask[1, 128:] = False
        targets = ids.clone().masked_fill(~mask, -100)
        logits = model(ids, mask)
        F.cross_entropy(logits.float().flatten(0, 1), targets.flatten()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        model.eval()
        torch.testing.assert_close(model(ids, mask), model(ids, mask))
