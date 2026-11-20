"""Stateless LR factors based on completed update/token progress."""
import math
from numbers import Integral, Real


class LearningRateSchedule:
    def __init__(self, *, warmup_steps=0, warmup_tokens=0, lr_decay='constant',
                 decay_steps=0, decay_tokens=0, min_lr_ratio=.1):
        for name, value in (('warmup_steps', warmup_steps), ('warmup_tokens', warmup_tokens),
                            ('decay_steps', decay_steps), ('decay_tokens', decay_tokens)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if warmup_steps and warmup_tokens:
            raise ValueError('Choose warmup_steps or warmup_tokens, not both')
        if lr_decay not in ('constant', 'cosine'):
            raise ValueError("lr_decay must be 'constant' or 'cosine'")
        if (isinstance(min_lr_ratio, bool) or not isinstance(min_lr_ratio, Real)
                or not math.isfinite(min_lr_ratio) or not 0 <= min_lr_ratio <= 1):
            raise ValueError('min_lr_ratio must be finite and in [0, 1]')
        if lr_decay == 'constant' and (decay_steps or decay_tokens):
            raise ValueError("Set lr_decay='cosine' when specifying a decay horizon")
        if lr_decay == 'cosine':
            if bool(decay_steps) == bool(decay_tokens):
                raise ValueError('Cosine decay requires exactly one of decay_steps or decay_tokens')
            if (decay_steps and warmup_tokens) or (decay_tokens and warmup_steps):
                raise ValueError('Warmup and decay must use the same unit (steps or tokens)')
            if (decay_steps or decay_tokens) <= (warmup_steps or warmup_tokens):
                raise ValueError('Decay horizon must be greater than the warmup budget')
        self.config = dict(warmup_steps=warmup_steps, warmup_tokens=warmup_tokens,
                           lr_decay=lr_decay, decay_steps=decay_steps,
                           decay_tokens=decay_tokens, min_lr_ratio=min_lr_ratio)

    def state_dict(self):
        return dict(self.config)

    def at(self, *, updates, tokens):
        cfg = self.config
        warmup = cfg['warmup_steps'] or cfg['warmup_tokens']
        progress = tokens if cfg['warmup_tokens'] or cfg['decay_tokens'] else updates
        if warmup and progress <= warmup:
            return progress / warmup, 'warmup'
        if cfg['lr_decay'] == 'constant':
            return 1., 'constant'
        horizon = cfg['decay_steps'] or cfg['decay_tokens']
        fraction = min(1., max(0., (progress - warmup) / (horizon - warmup)))
        floor = cfg['min_lr_ratio']
        factor = floor + .5 * (1. - floor) * (1. + math.cos(math.pi * fraction))
        return factor, 'minimum' if fraction == 1. else 'decay'
