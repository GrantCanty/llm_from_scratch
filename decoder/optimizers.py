"""Shared optimizer construction for training and capacity probes."""
import torch


class OptimizerPair:
    """Coordinate two disjoint optimizers at one training-update boundary.

    This small adapter supports this project's loop, not arbitrary torch
    schedulers or GradScaler. Both child states are included in state_dict.
    """
    def __init__(self, adamw, muon):
        self.optimizers = {'AdamW': adamw, 'Muon': muon}

    @property
    def param_groups(self):
        return [group for optimizer in self.optimizers.values() for group in optimizer.param_groups]

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        for optimizer in self.optimizers.values():
            optimizer.step()

    def state_dict(self):
        return {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()}

    def load_state_dict(self, state):
        if set(state) != set(self.optimizers):
            raise ValueError('Expected AdamW and Muon optimizer states')
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state[name])


def build_optimizer(model, *, optimizer_type='adamw', learning_rate=4e-4,
                    weight_decay=.1, adamw_betas=(.9, .999),
                    muon_learning_rate=None, muon_weight_decay=.1,
                    muon_momentum=.95, muon_nesterov=True,
                    muon_adjust_lr_fn='match_rms_adamw', muon_ns_steps=5):
    """AdamW baseline or Muon hidden Linear matrices plus AdamW remainder.

    Embeddings/output head stay in AdamW even if shared with a hidden matrix.
    Frozen parameters are excluded; each trainable parameter is owned once.
    """
    if optimizer_type not in ('adamw', 'muon'):
        raise ValueError("optimizer_type must be 'adamw' or 'muon'")
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    adam_kwargs = dict(lr=learning_rate, weight_decay=weight_decay, betas=adamw_betas)
    if optimizer_type == 'adamw':
        return torch.optim.AdamW([{'params': [p for _, p in named],
                                  'optimizer_name': 'AdamW'}], **adam_kwargs)
    if not hasattr(torch.optim, 'Muon'):
        raise RuntimeError('Muon requires a PyTorch build exposing torch.optim.Muon; '
                           'keep the working FA4 environment until a compatible build is verified')
    if muon_adjust_lr_fn not in ('original', 'match_rms_adamw', 'spectral_unclamped'):
        raise ValueError('Unsupported muon_adjust_lr_fn')
    if muon_learning_rate is None:
        if muon_adjust_lr_fn != 'match_rms_adamw':
            raise ValueError('Set muon_learning_rate explicitly for this scaling convention')
        muon_learning_rate = learning_rate
    protected = {id(p) for module in model.modules() if isinstance(module, torch.nn.Embedding)
                 for p in module.parameters()}
    protected.update(id(p) for p in model.out_head.parameters())
    hidden = {id(module.weight) for module in model.trf_blocks.modules()
              if isinstance(module, torch.nn.Linear) and module.weight.ndim == 2}
    muon_params = [p for _, p in named if id(p) in hidden - protected]
    adam_params = [p for _, p in named if id(p) not in hidden - protected]
    if not muon_params:
        raise ValueError('No trainable hidden matrices for Muon')
    adam_ids, muon_ids = {id(p) for p in adam_params}, {id(p) for p in muon_params}
    if adam_ids & muon_ids or adam_ids | muon_ids != {id(p) for _, p in named}:
        raise RuntimeError('Optimizer parameter partition must be disjoint and exhaustive')
    adamw = torch.optim.AdamW([{'params': adam_params, 'optimizer_name': 'AdamW'}], **adam_kwargs)
    muon = torch.optim.Muon([{'params': muon_params, 'optimizer_name': 'Muon'}],
                            lr=muon_learning_rate, weight_decay=muon_weight_decay,
                            momentum=muon_momentum, nesterov=muon_nesterov,
                            adjust_lr_fn=muon_adjust_lr_fn, ns_steps=muon_ns_steps)
    return OptimizerPair(adamw, muon)


def optimizer_group_metadata(optimizer):
    """CPU configuration suitable for history and capacity records."""
    keys = ('lr', 'weight_decay', 'betas', 'eps', 'momentum', 'nesterov',
            'adjust_lr_fn', 'ns_steps', 'ns_coefficients')
    return [dict(optimizer=group.get('optimizer_name', type(optimizer).__name__),
                 param_group=index, parameter_count=sum(p.numel() for p in group['params']),
                 **{key: group[key] for key in keys if key in group})
            for index, group in enumerate(optimizer.param_groups)]
