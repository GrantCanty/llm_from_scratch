"""Token-weighted microbatch groups and history-based global clipping."""
from collections import deque
import math
from numbers import Integral, Real

import torch

from .utils import unpack_batch


def validate_accumulation_steps(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError('accumulation_steps must be a positive integer')


def scored_microbatch_groups(loader, accumulation_steps):
    """Buffer input batches only, never activation graphs; flush each epoch tail.

    Count valid targets before forward so each backward can use its exact share
    of the group's token-mean objective. Empty batches do not fill a group.
    """
    group = []
    for batch in loader:
        unpacked = unpack_batch(batch)
        count = (unpacked[1] != -100).sum().item()
        if not count:
            continue
        group.append((unpacked, count))
        if len(group) == accumulation_steps:
            yield group
            group = []
    if group:
        yield group


class AdaptiveGradientClipper:
    """Threshold = multiplier * mean(previous window of raw global norms)."""
    def __init__(self, enabled=False, window=100, multiplier=3., min_history=20):
        if not isinstance(enabled, bool):
            raise ValueError('adaptive_clip must be a boolean')
        for name, value in (('clip_window', window), ('clip_min_history', min_history)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if min_history > window:
            raise ValueError('clip_min_history cannot exceed clip_window')
        if (isinstance(multiplier, bool) or not isinstance(multiplier, Real)
                or not math.isfinite(multiplier) or multiplier <= 0):
            raise ValueError('clip_multiplier must be finite and positive')
        self.enabled, self.multiplier, self.min_history = enabled, multiplier, min_history
        self.norms = deque(maxlen=window)
        self.updates = self.clipped_updates = 0

    @torch.no_grad()
    def apply(self, parameters):
        parameters = list(parameters)
        # Includes every optimizer's parameters, after token normalization.
        norm = torch.nn.utils.clip_grad_norm_(parameters, float('inf'), error_if_nonfinite=True).item()
        baseline = sum(self.norms) / len(self.norms) if self.norms else None
        threshold = (max(1e-12, self.multiplier * baseline)
                     if self.enabled and len(self.norms) >= self.min_history else None)
        factor = min(1., threshold / (norm + 1e-6)) if threshold is not None else 1.
        if factor < 1.:
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(factor)
            self.clipped_updates += 1
        self.updates += 1
        # Record RAW norm after computing this update's threshold: no self-influence.
        self.norms.append(norm)
        return dict(grad_norm=norm, grad_norm_after_clipping=norm * factor,
                    clip_threshold=threshold, clip_baseline=baseline,
                    clip_factor=factor, was_clipped=factor < 1.,
                    clipping_fraction=self.clipped_updates / self.updates)
