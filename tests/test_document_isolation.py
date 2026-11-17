import copy
import unittest
import torch
import torch.nn.functional as F
from decoder.model import GPTModel, train_model_simple_function
from decoder.packing import CausalLMCollator, iter_packed_windows
from decoder.self_attention import MultiHeadedAttention
from decoder.utils import calc_loss_batch


class DocumentIsolationTests(unittest.TestCase):
    def test_logits_and_gradients_match_independent_documents(self):
        torch.manual_seed(42)
        cfg = dict(vocab_size=32, context_length=12, emb_dim=16,
                   n_heads=2, n_layers=3, drop_rate=0., qkv_bias=True)
        model = GPTModel(cfg).eval()
        reference = copy.deepcopy(model)
        batch = CausalLMCollator(12, 31)(list(iter_packed_windows([[1, 31, 3], [4, 5]], 12, 31)))
        kwargs = {k: batch[k] for k in ('attention_mask', 'document_ids', 'position_ids')}
        logits = model(batch['input_ids'], **kwargs)
        changed = batch['input_ids'].clone()
        changed[0, :4] = 8
        torch.testing.assert_close(logits[:, 4:], model(changed, **kwargs)[:, 4:])
        independent = torch.cat([reference(batch['input_ids'][:, :4]),
                                 reference(batch['input_ids'][:, 4:])], dim=1)
        torch.testing.assert_close(logits, independent, atol=1e-6, rtol=1e-5)
        F.cross_entropy(logits.flatten(0, 1), batch['labels'].flatten()).backward()
        F.cross_entropy(independent.flatten(0, 1), batch['labels'].flatten()).backward()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, atol=1e-6, rtol=1e-5)
        self.assertEqual(batch['position_ids'].tolist(), [[0, 1, 2, 3, 0, 1]])
        self.assertEqual(batch['labels'].tolist(), [[31, 3, 31, -100, 5, 31]])

    def test_all_eligible_pairs_and_overlap_metadata(self):
        docs = [[1], [], [2, 31, 3], [4]*10, [5]]
        expected = [(a, b) for doc in docs if doc for a, b in zip(doc, doc[1:]+[31])]
        for context in (1, 2, 4, 8):
            actual = []
            for row in iter_packed_windows(docs, context, 31):
                batch = CausalLMCollator(context, 31)([row])
                actual.extend((a, b) for a, b in zip(batch['input_ids'][0].tolist(), batch['labels'][0].tolist()) if b != -100)
            self.assertEqual(actual, expected)
        with self.assertRaisesRegex(ValueError, 'document_ids'):
            CausalLMCollator(4, 31)([{'input_ids': [1, 31]}])

    def test_zero_target_batch_does_not_update_weights(self):
        cfg = dict(vocab_size=32, context_length=1, emb_dim=8,
                   n_heads=2, n_layers=1, drop_rate=0., qkv_bias=True)
        model = GPTModel(cfg)
        batch = CausalLMCollator(1, 31)([{'input_ids': [31, 2], 'document_ids': [0, 1]}])
        loss = calc_loss_batch(batch['input_ids'], batch['labels'], model, 'cpu',
                               batch['attention_mask'], document_ids=batch['document_ids'], position_ids=batch['position_ids'])
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        before = copy.deepcopy(model.state_dict())
        train_model_simple_function(model, [batch], [batch], torch.optim.AdamW(model.parameters()), 'cpu', 1, 1, 1)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key])

    def test_fa4_segment_calls_match_masked_sdpa_gradients(self):
        calls = []
        def kernel(q, k, v, **kwargs):
            calls.append(q.shape[1])
            return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2), None
        module = MultiHeadedAttention(8, 8, 6, 0., 2, attention_backend='fa4', flash_attn_func=kernel)
        tensors = [torch.randn(2, 6, 2, 4, dtype=torch.float64, requires_grad=True) for _ in range(3)]
        refs = [t.detach().clone().requires_grad_() for t in tensors]
        docs = torch.tensor([[0, 0, 1, 1, 1, 2], [0, 1, 1, -1, -1, -1]])
        valid = docs >= 0
        actual = module._fa4_attention(*tensors, valid, 0., docs)
        allowed = (docs[:, None, :, None] == docs[:, None, None, :]) & torch.ones(6, 6, dtype=torch.bool).tril() & valid[:, None, None, :] & valid[:, None, :, None]
        expected = F.scaled_dot_product_attention(*[t.transpose(1, 2) for t in refs], attn_mask=allowed).transpose(1, 2)
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        expected.square().sum().backward()
        for a, b in zip(tensors, refs):
            torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-5)
        self.assertEqual(calls, [2, 3, 1, 1, 2])

    def test_padding_positions_and_metrics(self):
        from decoder.utils import calc_metrics_loader
        cfg = dict(vocab_size=32, context_length=8, emb_dim=8,
                   n_heads=2, n_layers=2, drop_rate=0., qkv_bias=True)
        model = GPTModel(cfg).eval()
        rows = list(iter_packed_windows([[1], [2, 3], [4]*6], 8, 31))
        batch = CausalLMCollator(8, 31)(rows)
        loss, sums, counts, logits = calc_loss_batch(
            batch['input_ids'], batch['labels'], model, 'cpu', batch['attention_mask'],
            return_logits=True, document_ids=batch['document_ids'], position_ids=batch['position_ids'])
        self.assertTrue(torch.isfinite(logits).all())
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        valid = batch['labels'] != -100
        expected_counts = torch.bincount(batch['position_ids'][valid], minlength=8)
        self.assertEqual(counts.tolist(), expected_counts.tolist())
        metrics = calc_metrics_loader([batch], model, 'cpu')
        self.assertEqual(metrics['position_counts'].tolist(), counts.tolist())
        self.assertEqual(metrics['scored_tokens'], valid.sum().item())
        with self.assertRaisesRegex(ValueError, 'position_ids'):
            model(batch['input_ids'], batch['attention_mask'], batch['document_ids'],
                  torch.ones_like(batch['position_ids']))
