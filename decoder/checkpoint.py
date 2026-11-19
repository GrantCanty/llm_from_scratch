"""Versioned checkpoint snapshots; loading/resume is intentionally separate."""
import copy
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


def cpu_snapshot(value):
    """Detach/copy tensors without duplicating them on the accelerator."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device='cpu', copy=True)
    if isinstance(value, dict):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


def rng_state(device):
    numpy_state = np.random.get_state()
    state = {'python': random.getstate(), 'torch_cpu': torch.get_rng_state(),
             'numpy': (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:])}
    device = torch.device(device)
    if device.type == 'cuda':
        index = torch.cuda.current_device() if device.index is None else device.index
        state['torch_cuda'] = {index: torch.cuda.get_rng_state(index).cpu()}
    elif device.type == 'mps':
        state['torch_mps'] = torch.mps.get_rng_state().cpu()
    return state


def loader_generator_state(loader):
    """Capture separate DataLoader/sampler generators when exposed.

    This is not a generic cursor for arbitrary iterable datasets or prefetchers.
    """
    states = {}
    for name, owner in (('loader', loader), ('sampler', getattr(loader, 'sampler', None))):
        generator = getattr(owner, 'generator', None)
        if isinstance(generator, torch.Generator):
            states[name] = generator.get_state().cpu()
    return states


def atomic_save(payload, path):
    """Replace the destination only after a complete, flushed save beside it."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return str(path)
