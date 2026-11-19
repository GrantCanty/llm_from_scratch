import gc
import unittest
import weakref
from unittest.mock import patch

import torch

from decoder.decoder import Decoder
from decoder.model import GPTModel, train_model_simple_function
from decoder.optimizers import build_optimizer


class MemoryLifecycleTests(unittest.TestCase):
    def make_decoder(self, device='cpu'):
        decoder = Decoder.__new__(Decoder)
        decoder.seed = 42
        decoder.device = torch.device(device)
        decoder.GPT_CONFIG = dict(vocab_size=16, context_length=3, emb_dim=8,
                                  n_heads=2, n_layers=1, drop_rate=0., qkv_bias=True)
        decoder.reset_weights()
        return decoder

    def batch(self):
        return (torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]]))

    def assert_cpu_metrics(self, value):
        if isinstance(value, torch.Tensor):
            self.assertEqual(value.device.type, 'cpu')
            self.assertIsNone(value.grad_fn)
        elif isinstance(value, dict):
            for item in value.values():
                self.assert_cpu_metrics(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                self.assert_cpu_metrics(item)

    def test_repeat_runs_retain_state_until_explicit_cleanup(self):
        for kind in ('adamw', 'muon'):
            if kind == 'muon' and not hasattr(torch.optim, 'Muon'):
                continue
            decoder = self.make_decoder()
            owned = []
            def builder(*args, **kwargs):
                optimizer = build_optimizer(*args, **kwargs)
                owned.append(optimizer)
                return optimizer
            previous = decoder.decoder_model.tok_emb.weight.detach().clone()
            with patch('decoder.decoder.build_optimizer', side_effect=builder):
                for _ in range(2):
                    result = decoder.train([self.batch()], [self.batch()], 1,
                                           optimizer_type=kind, drop_rate=0., update_norm_interval=1)
                    self.assertFalse(torch.equal(previous, decoder.decoder_model.tok_emb.weight))
                    previous = decoder.decoder_model.tok_emb.weight.detach().clone()
                    self.assertTrue(all(p.grad is None for p in decoder.decoder_model.parameters()))
                    self.assertEqual(decoder.training_memory['after']['gradient_bytes'], 0)
                    self.assert_cpu_metrics((result, decoder.training_history, decoder.training_memory))
                    self.assertIs(decoder.optimizer, owned[-1])
                    children = getattr(decoder.optimizer, 'optimizers', {'single': decoder.optimizer})
                    self.assertTrue(all(child.state for child in children.values()))
                    decoder.cleanup_memory(empty_cache=False)
                    self.assertIsNone(decoder.optimizer)
                    self.assertTrue(all(not child.state for child in children.values()))
            before = {name: p.clone() for name, p in decoder.decoder_model.state_dict().items()}
            history = decoder.training_history
            for p in decoder.decoder_model.parameters():
                p.grad = torch.ones_like(p)
            report = decoder.cleanup_memory()
            self.assertGreater(report['before']['gradient_bytes'], 0)
            self.assertEqual(report['after']['gradient_bytes'], 0)
            self.assertIs(history, decoder.training_history)
            for name, value in decoder.decoder_model.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
            # Generation remains usable with the same trained weights.
            class Tokenizer:
                def encode(self, text, **kwargs):
                    return [1, 2]
                def decode(self, tokens):
                    return ','.join(map(str, tokens))
            self.assertIsInstance(decoder.generate('test', Tokenizer(), max_tokens=1, top_k=4), str)

    def test_error_or_interrupt_retains_optimizer_until_explicit_cleanup(self):
        for error in (RuntimeError('simulated training failure'), KeyboardInterrupt()):
            decoder = self.make_decoder()
            held = []
            def fail(model, train_loader, val_loader, optimizer, *args, **kwargs):
                for p in model.parameters():
                    p.grad = torch.ones_like(p)
                optimizer.step()  # Allocate real Adam state before interruption.
                held.append(optimizer)
                raise error
            with patch('decoder.decoder.model.train_model_simple_function', side_effect=fail):
                try:
                    decoder.train([self.batch()], [self.batch()], 1)
                except BaseException as exc:
                    retained_traceback = exc.__traceback__
            self.assertIsNotNone(retained_traceback)
            self.assertTrue(held[0].state)
            self.assertIs(decoder.optimizer, held[0])
            self.assertTrue(all(p.grad is None for p in decoder.decoder_model.parameters()))
            decoder.cleanup_memory(empty_cache=False)
            self.assertFalse(held[0].state)

    def test_reset_releases_old_model_before_constructing_replacement(self):
        decoder = self.make_decoder()
        old_model = weakref.ref(decoder.decoder_model)
        old_parameter = weakref.ref(decoder.decoder_model.tok_emb.weight)
        def constructor(cfg):
            self.assertIsNone(old_model())
            self.assertIsNone(old_parameter())
            return GPTModel(cfg)
        with patch('decoder.decoder.model.GPTModel', side_effect=constructor):
            decoder.reset_weights()

    def test_failed_update_releases_norm_snapshots_but_keeps_external_optimizer(self):
        decoder = self.make_decoder()
        optimizer = build_optimizer(decoder.decoder_model)
        original_step = optimizer.step
        def fail_after_step():
            original_step()
            raise RuntimeError('failed after optimizer update')
        from decoder.update_metrics import snapshot_parameters
        snapshots = []
        def capture(*args):
            result = snapshot_parameters(*args)
            snapshots.extend(weakref.ref(item[2]) for item in result)
            return result
        with patch.object(optimizer, 'step', side_effect=fail_after_step), \
             patch('decoder.model.snapshot_parameters', side_effect=capture):
            try:
                train_model_simple_function(decoder.decoder_model, [self.batch()], [], optimizer,
                                            'cpu', 1, 1, 1, update_history=[])
            except RuntimeError as exc:
                retained_traceback = exc.__traceback__
        gc.collect()
        self.assertIsNotNone(retained_traceback)
        self.assertTrue(snapshots)
        self.assertTrue(all(ref() is None for ref in snapshots))
        # The lower-level caller still owns and can continue its optimizer.
        self.assertTrue(optimizer.state)

    @unittest.skipUnless(torch.cuda.is_available(), 'Requires CUDA')
    def test_cuda_live_allocation_returns_to_model_baseline_across_runs(self):
        decoder = self.make_decoder('cuda')
        baseline = decoder.memory_report()['allocated_bytes']
        for kind in ('adamw', 'muon'):
            if kind == 'muon' and not hasattr(torch.optim, 'Muon'):
                continue
            for _ in range(3):
                decoder.train([self.batch()] * 2, [self.batch()], 1, optimizer_type=kind,
                              accumulation_steps=2, update_norm_interval=1, drop_rate=0.)
                after = decoder.cleanup_memory()['after']
                self.assertEqual(after['gradient_bytes'], 0)
                self.assertLessEqual(after['allocated_bytes'], baseline + 1024 ** 2)
