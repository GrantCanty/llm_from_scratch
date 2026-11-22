import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from decoder.decoder import Decoder
from decoder.sampling import ResumableBatchSampler
from decoder.schedules import LearningRateSchedule


class ScheduleTests(unittest.TestCase):
    def test_step_warmup_cosine_boundaries_and_floor(self):
        schedule = LearningRateSchedule(warmup_steps=2, lr_decay='cosine', decay_steps=6, min_lr_ratio=.1)
        expected = [.5, 1., .1 + .9 * .5 * (1 + math.cos(math.pi/4)), .55,
                    .1 + .9 * .5 * (1 + math.cos(3*math.pi/4)), .1, .1]
        for update, value in enumerate(expected, 1):
            self.assertAlmostEqual(schedule.at(updates=update, tokens=999)[0], value)
        self.assertEqual(schedule.at(updates=7, tokens=0)[1], 'minimum')

    def test_token_schedule_uses_actual_progress_and_no_warmup(self):
        schedule = LearningRateSchedule(warmup_tokens=6, lr_decay='cosine', decay_tokens=18)
        for tokens, expected in ((2, 1/3), (6, 1), (12, .55), (18, .1), (25, .1)):
            self.assertAlmostEqual(schedule.at(updates=999, tokens=tokens)[0], expected)
        no_warmup = LearningRateSchedule(lr_decay='cosine', decay_steps=4, min_lr_ratio=0.)
        self.assertAlmostEqual(no_warmup.at(updates=2, tokens=0)[0], .5)
        self.assertEqual(no_warmup.at(updates=4, tokens=0)[0], 0.)

    def test_invalid_or_ambiguous_budgets(self):
        invalid = [dict(lr_decay='bad'), dict(decay_tokens=100),
                   dict(lr_decay='cosine'), dict(lr_decay='cosine', decay_steps=1, decay_tokens=1),
                   dict(lr_decay='cosine', warmup_tokens=10, decay_steps=100),
                   dict(lr_decay='cosine', warmup_steps=10, decay_tokens=100),
                   dict(lr_decay='cosine', lr_sustain_steps=2, decay_tokens=100),
                   dict(lr_sustain_steps=2),
                   dict(lr_decay='cosine', decay_steps=1.5), dict(decay_steps=True),
                   dict(min_lr_ratio=-1), dict(min_lr_ratio=float('nan')), dict(min_lr_ratio=2)]
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                LearningRateSchedule(**config)

    def test_step_sustain_holds_peak_and_keeps_the_endpoint(self):
        schedule = LearningRateSchedule(warmup_steps=2, lr_sustain_steps=2,
                                        lr_decay='cosine', decay_steps=6)
        self.assertEqual(schedule.at(updates=2, tokens=999), (1., 'warmup'))
        for update in (3, 4):
            self.assertEqual(schedule.at(updates=update, tokens=999), (1., 'sustain'))
        self.assertEqual(schedule.at(updates=5, tokens=999), (.55, 'decay'))
        self.assertEqual(schedule.at(updates=6, tokens=999), (.1, 'minimum'))

    def test_old_schedule_state_defaults_to_zero_sustain(self):
        schedule = LearningRateSchedule(lr_decay='cosine', decay_steps=10)
        legacy = schedule.state_dict()
        del legacy['lr_sustain_steps'], legacy['lr_sustain_tokens']
        self.assertTrue(schedule.matches_state(legacy))
        self.assertFalse(LearningRateSchedule(**legacy, lr_sustain_steps=2).matches_state(legacy))

    def test_endpoint_during_warmup_or_sustain_preserves_both_phases(self):
        for endpoint in (1, 2, 4, 5):
            with self.subTest(endpoint=endpoint):
                schedule = LearningRateSchedule(warmup_steps=2, lr_sustain_steps=3,
                                                lr_decay='cosine', decay_steps=endpoint)
                self.assertEqual(schedule.at(updates=2, tokens=0), (1., 'warmup'))
                self.assertEqual(schedule.at(updates=5, tokens=0), (1., 'sustain'))
                self.assertEqual(schedule.at(updates=6, tokens=0), (.1, 'minimum'))


class DecayResumeTests(unittest.TestCase):
    def make_decoder(self):
        decoder = Decoder.__new__(Decoder)
        decoder.seed = 42
        decoder.device = torch.device('cpu')
        decoder.GPT_CONFIG = dict(vocab_size=16, context_length=3, emb_dim=8,
                                  n_heads=2, n_layers=1, drop_rate=.2, qkv_bias=True)
        decoder.reset_weights()
        return decoder

    def loaders(self):
        inputs = torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
        targets = torch.tensor([[2, -100, -100], [-100]*3, [4, 5, 6], [5, 6, -100]])
        data = TensorDataset(inputs, targets)
        train = DataLoader(data, batch_sampler=ResumableBatchSampler(data, 1, seed=17),
                           generator=torch.Generator().manual_seed(19))
        val = [(inputs, targets)]
        return train, val

    def test_decay_resume_preserves_peak_lrs_both_optimizers_and_token_progress(self):
        for kind in ('adamw', 'muon'):
            if kind == 'muon' and not hasattr(torch.optim, 'Muon'):
                continue
            options = dict(num_epochs=2, optimizer_type=kind, learning_rate=.004,
                           muon_learning_rate=.002, drop_rate=.2, accumulation_steps=2,
                           warmup_tokens=2, lr_sustain_tokens=4,
                           lr_decay='cosine', decay_tokens=12,
                           min_lr_ratio=.1, eval_freq=1, eval_iter=1,
                           adaptive_clip=True, clip_min_history=1, clip_window=3)
            baseline = self.make_decoder()
            baseline.train(*self.loaders(), **options)
            paused = self.make_decoder()
            paused.train(*self.loaders(), **options, max_updates=1)
            with tempfile.TemporaryDirectory() as directory:
                path = paused.save_checkpoint(Path(directory) / 'decay')
                restored = self.make_decoder()
                restored.resume(path, *self.loaders())
            for name, expected in baseline.decoder_model.state_dict().items():
                torch.testing.assert_close(expected, restored.decoder_model.state_dict()[name], rtol=0, atol=0)
            self.assertEqual([r['learning_rates'] for r in baseline.training_history],
                             [r['learning_rates'] for r in restored.training_history])
            schedule = LearningRateSchedule(warmup_tokens=2, lr_sustain_tokens=4,
                                            lr_decay='cosine', decay_tokens=12)
            for row in restored.training_history:
                factor, phase = schedule.at(updates=row['update'], tokens=row['scored_tokens'])
                self.assertEqual(row['lr_phase'], phase)
                self.assertEqual(row['lr_factor'], factor)
                for actual, peak in zip(row['learning_rates'], [.004, .002]):
                    self.assertAlmostEqual(actual, peak * factor)
            self.assertEqual(restored.training_history[-1]['lr_phase'], 'minimum')
            self.assertIn('sustain', [row['lr_phase'] for row in restored.training_history])
            self.assertEqual(restored.training_state['lr_schedule']['decay_tokens'], 12)
            # Extending training does not stretch/restart the saved horizon.
            restored.resume(None, *self.loaders(), num_epochs=3)
            self.assertEqual(restored.training_history[-1]['lr_phase'], 'minimum')

    def test_legacy_constant_checkpoint_without_schedule_fields(self):
        original = self.make_decoder()
        original.train(*self.loaders(), num_epochs=1, warmup_steps=3, max_updates=1)
        with tempfile.TemporaryDirectory() as directory:
            path = original.save_checkpoint(Path(directory) / 'legacy')
            payload = torch.load(path, weights_only=True)
            for key in ('lr_decay', 'decay_steps', 'decay_tokens', 'min_lr_ratio',
                        'lr_sustain_steps', 'lr_sustain_tokens'):
                payload['training_config'].pop(key)
            payload['training_state'].pop('lr_schedule')
            torch.save(payload, path)
            restored = self.make_decoder()
            restored.resume(path, *self.loaders())
        self.assertEqual(restored.training_state['lr_schedule']['lr_decay'], 'constant')
        self.assertEqual(restored.training_state['updates'], 3)
