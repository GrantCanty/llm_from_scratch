"""Optimizer-independent measurements of a single, actual parameter update."""
import math

import torch


@torch.no_grad()
def snapshot_parameters(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    return [
        (names[id(p)], p, p.detach().float().clone(), index,
         group['lr'], group.get('weight_decay', 0.0))
        for index, group in enumerate(optimizer.param_groups)
        for p in group['params']
        if id(p) in names and p.requires_grad and p.grad is not None
    ]


@torch.no_grad()
def measure_updates(snapshot, optimizer):
    """Return CPU scalars; denominator is the pre-step norm, including for decay."""
    parameters = {}
    weight_sq = update_sq = 0.0
    for name, parameter, before, group, lr, decay in snapshot:
        weight_norm = torch.linalg.vector_norm(before).item()
        update_norm = torch.linalg.vector_norm(parameter.detach().float() - before).item()
        parameters[name] = {
            'weight_norm': weight_norm,
            'update_norm': update_norm,
            'update_to_weight_ratio': update_norm / (weight_norm + 1e-12),
            'parameter_kind': 'matrix' if parameter.ndim >= 2 else 'bias_or_norm',
            'optimizer': type(optimizer).__name__,
            'param_group': group,
            'learning_rate': lr,
            'weight_decay': decay,
        }
        weight_sq += weight_norm ** 2
        update_sq += update_norm ** 2
    return {
        'weight_norm': math.sqrt(weight_sq),
        'update_norm': math.sqrt(update_sq),
        'update_to_weight_ratio': math.sqrt(update_sq) / (math.sqrt(weight_sq) + 1e-12),
        'parameters': parameters,
    }
