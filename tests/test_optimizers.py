import copy
import unittest
from unittest.mock import patch

import torch

from decoder.model import GPTModel, train_model_simple_function
from decoder.optimizers import build_optimizer
from decoder.update_metrics import snapshot_parameters, measure_updates


@unittest.skipUnless(hasattr(torch.optim, 'Muon'), 'Native Muon unavailable')
class MuonTests(unittest.TestCase):
    def make_model(self):
        torch.manual_seed(17)
        return GPTModel(dict(vocab_size=16, context_length=3, emb_dim=8,
                             n_heads=2, n_layers=1, drop_rate=0., qkv_bias=True))

    def test_partition_protects_embeddings_and_ties_and_excludes_frozen(self):
        model = self.make_model()
        model.out_head.weight = model.tok_emb.weight
        model.trf_blocks[0].att.W_query.weight.requires_grad_(False)
        optimizer = build_optimizer(model, optimizer_type='muon')
        adam = {id(p) for p in optimizer.optimizers['AdamW'].param_groups[0]['params']}
        muon = {id(p) for p in optimizer.optimizers['Muon'].param_groups[0]['params']}
        self.assertFalse(adam & muon)
        self.assertEqual(adam | muon, {id(p) for p in model.parameters() if p.requires_grad})
        self.assertIn(id(model.tok_emb.weight), adam)
        self.assertIn(id(model.pos_emb.weight), adam)
        self.assertIn(id(model.final_norm.weight), adam)
        self.assertIn(id(model.trf_blocks[0].ff.layers[0].weight), muon)
        self.assertIn(id(model.trf_blocks[0].att.W_key.weight), muon)
        self.assertTrue(all(p.ndim == 2 for group in optimizer.optimizers['Muon'].param_groups
                            for p in group['params']))

    def test_real_updates_metrics_and_state_roundtrip(self):
        model = self.make_model()
        optimizer = build_optimizer(model, optimizer_type='muon')
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        snapshot = snapshot_parameters(model, optimizer)
        optimizer.step()
        metrics = measure_updates(snapshot, optimizer)['parameters']
        self.assertTrue(all(value['update_norm'] > 0 for value in metrics.values()))
        self.assertEqual(metrics['trf_blocks.0.ff.layers.0.weight']['optimizer'], 'Muon')
        self.assertEqual(metrics['out_head.weight']['optimizer'], 'AdamW')
        restored = self.make_model()
        restored.load_state_dict(model.state_dict())
        other = build_optimizer(restored, optimizer_type='muon')
        other.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        torch.manual_seed(42)
        for p, q in zip(model.parameters(), restored.parameters()):
            p.grad = torch.randn_like(p)
            q.grad = p.grad.clone()
        optimizer.step()
        other.step()
        for p, q in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        optimizer.zero_grad()
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), 'Requires CUDA')
    def test_cuda_training_smoke(self):
        model = self.make_model().cuda()
        optimizer = build_optimizer(model, optimizer_type='muon')
        batch = (torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]]))
        history = []
        train_model_simple_function(model, [batch, batch], [batch], optimizer,
                                    'cuda', 1, 100, 1, update_history=history,
                                    accumulation_steps=2, adaptive_clip=True, clip_min_history=1,
                                    update_norm_interval=1)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["microbatches"], 2)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_loop_warms_both_groups_and_skips_empty_batches(self):
        model = self.make_model()
        optimizer = build_optimizer(model, optimizer_type='muon', learning_rate=.001,
                                    muon_learning_rate=.002)
        batch = (torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]]))
        empty = (batch[0], torch.full_like(batch[1], -100))
        history = []
        train_model_simple_function(model, [batch, empty, batch], [batch], optimizer,
                                    'cpu', 1, 100, 1, warmup_steps=2,
                                    update_history=history, update_norm_interval=1)
        self.assertEqual([row['learning_rates'] for row in history],
                         [[.0005, .001], [.001, .002]])
        self.assertEqual([row['scored_tokens'] for row in history], [3, 6])
        self.assertEqual([g['optimizer'] for g in history[0]['optimizer_groups']], ['AdamW', 'Muon'])
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))


class OptimizerConfigTests(unittest.TestCase):
    def test_missing_muon_fails_without_changing_adamw_baseline(self):
        model = torch.nn.Linear(2, 2)
        with patch.object(torch.optim, 'Muon', create=True) as available:
            # Delete the patched attribute to emulate an older cloud build.
            del torch.optim.Muon
            with self.assertRaisesRegex(RuntimeError, 'PyTorch'):
                build_optimizer(model, optimizer_type='muon')
            self.assertIsInstance(build_optimizer(model), torch.optim.AdamW)

    def test_invalid_optimizer(self):
        with self.assertRaises(ValueError):
            build_optimizer(torch.nn.Linear(2, 2), optimizer_type='typo')
