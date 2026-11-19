import copy
import unittest
from unittest.mock import patch

import torch

from decoder.model import GPTModel, train_model_simple_function
from decoder.optimizers import build_optimizer
from decoder.training_controls import AdaptiveGradientClipper


class AccumulationTests(unittest.TestCase):
    def make_model(self):
        torch.manual_seed(42)
        return GPTModel(dict(vocab_size=16, context_length=3, emb_dim=8,
                             n_heads=2, n_layers=1, drop_rate=0., qkv_bias=True)).double()

    def test_unequal_tokens_partial_tail_and_joint_optimizer_equivalence(self):
        inputs = torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        targets = torch.tensor([[2, -100, -100], [3, 4, 5], [4, 5, -100]])
        empty = (inputs[:1], torch.full_like(targets[:1], -100))
        for optimizer_type in ('adamw', 'muon'):
            if optimizer_type == 'muon' and not hasattr(torch.optim, 'Muon'):
                continue
            reference, accumulated = self.make_model(), self.make_model()
            opt_ref = build_optimizer(reference, optimizer_type=optimizer_type)
            opt_acc = build_optimizer(accumulated, optimizer_type=optimizer_type)
            # Record actual normalized gradients entering the optimizer, before
            # Adam/Muon can hide a scalar normalization error.
            gradients = [[], []]
            def record(optimizer, model, destination):
                original = optimizer.step
                def step():
                    destination.append([p.grad.clone() for p in model.parameters()])
                    return original()
                return step
            opt_ref.step = record(opt_ref, reference, gradients[0])
            opt_acc.step = record(opt_acc, accumulated, gradients[1])
            history_ref, history_acc = [], []
            eval_result = (1., torch.ones(3), 1., torch.ones(3), 2., .5)
            with patch('decoder.model.evaluate_model', return_value=eval_result):
                train_model_simple_function(reference,
                    [(inputs[:2], targets[:2]), (inputs[2:], targets[2:])], [],
                    opt_ref, 'cpu', 1, 100, 1, update_history=history_ref, warmup_tokens=12,
                    adaptive_clip=True, clip_window=2, clip_min_history=1, clip_multiplier=.5)
                train_model_simple_function(accumulated,
                    [(inputs[:1], targets[:1]), empty,
                     (inputs[1:2], targets[1:2]), (inputs[2:], targets[2:])], [],
                    opt_acc, 'cpu', 1, 100, 1, update_history=history_acc,
                    accumulation_steps=2, warmup_tokens=12,
                    adaptive_clip=True, clip_window=2, clip_min_history=1, clip_multiplier=.5)
            self.assertTrue(history_acc[1]['was_clipped'])
            self.assertEqual([r['microbatches'] for r in history_acc], [2, 1])
            self.assertEqual([r['scored_tokens_per_update'] for r in history_acc], [4, 2])
            self.assertEqual([r['scored_tokens'] for r in history_acc], [4, 6])
            self.assertEqual([r['learning_rates'] for r in history_acc],
                             [r['learning_rates'] for r in history_ref])
            for left, right in zip(gradients[0], gradients[1]):
                for a, b in zip(left, right):
                    torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-8)
            for a, b in zip(reference.parameters(), accumulated.parameters()):
                torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-5)
            for a, b in zip(history_ref, history_acc):
                self.assertAlmostEqual(a['loss'], b['loss'], places=8)

    def test_all_empty_and_epoch_tails(self):
        model = self.make_model()
        optimizer = build_optimizer(model)
        batch = (torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]]))
        history = []
        eval_result = (1., torch.ones(3), 1., torch.ones(3), 2., .5)
        with patch('decoder.model.evaluate_model', return_value=eval_result):
            train_model_simple_function(model, [batch], [], optimizer, 'cpu', 2, 100, 1,
                                        accumulation_steps=4, update_history=history)
        self.assertEqual([r['microbatches_seen'] for r in history], [1, 2])
        before = copy.deepcopy(model.state_dict())
        empty_history = []
        train_model_simple_function(model, [(batch[0], torch.full_like(batch[1], -100))], [],
                                    optimizer, 'cpu', 1, 1, 1, accumulation_steps=4,
                                    update_history=empty_history)
        self.assertEqual(empty_history, [])
        for name, p in model.state_dict().items():
            torch.testing.assert_close(p, before[name], rtol=0, atol=0)

    def test_nonfinite_loss_aborts_before_step(self):
        model = self.make_model()
        optimizer = build_optimizer(model)
        with patch.object(optimizer, 'step') as step, \
             patch('decoder.model.utils.calc_loss_batch', return_value=torch.tensor(float('nan'))):
            with self.assertRaises(FloatingPointError):
                train_model_simple_function(model, [(torch.ones(1), torch.ones(1))], [],
                                            optimizer, 'cpu', 1, 1, 1)
            step.assert_not_called()


class AdaptiveClippingTests(unittest.TestCase):
    def test_previous_raw_window_threshold_and_shared_direction(self):
        clipper = AdaptiveGradientClipper(True, window=2, multiplier=3., min_history=2)
        params = [torch.nn.Parameter(torch.zeros(1)) for _ in range(2)]
        for norm in (1., 1., 10., 1.):
            params[0].grad = torch.tensor([norm * .6])
            params[1].grad = torch.tensor([norm * .8])
            stats = clipper.apply(params)
            if norm == 10.:
                self.assertAlmostEqual(stats['clip_threshold'], 3.)
                self.assertAlmostEqual(stats['grad_norm_after_clipping'], 3., places=5)
                self.assertAlmostEqual(params[0].grad.item() / params[1].grad.item(), .75, places=6)
                self.assertTrue(stats['was_clipped'])
        # Last baseline uses raw 10 rather than the clipped norm 3.
        self.assertAlmostEqual(stats['clip_threshold'], 16.5)
        self.assertAlmostEqual(stats['clipping_fraction'], .25)

    def test_disabled_warm_start_and_nonfinite(self):
        parameter = torch.nn.Parameter(torch.zeros(1))
        for enabled in (False, True):
            clipper = AdaptiveGradientClipper(enabled, window=3, min_history=2)
            parameter.grad = torch.tensor([100.])
            stats = clipper.apply([parameter])
            self.assertIsNone(stats['clip_threshold'])
            self.assertEqual(stats['clip_factor'], 1.)
        parameter.grad = torch.tensor([float('inf')])
        with self.assertRaises(RuntimeError):
            clipper.apply([parameter])

    def test_invalid_config(self):
        for kwargs in (dict(window=0), dict(min_history=101), dict(multiplier=float('nan')),
                       dict(multiplier=0), dict(enabled='yes')):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AdaptiveGradientClipper(**kwargs)
