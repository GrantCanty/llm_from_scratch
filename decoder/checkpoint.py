"""Versioned snapshots, RNG restoration, and deterministic loader replay."""
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


def restore_rng_state(state, device):
    random.setstate(state['python'])
    name, keys, position, has_gauss, cached = state['numpy']
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached))
    torch.set_rng_state(state['torch_cpu'].cpu())
    device = torch.device(device)
    if device.type == 'cuda':
        saved = state.get('torch_cuda')
        if not saved:
            raise ValueError('Checkpoint has no CUDA RNG state for CUDA continuation')
        index = torch.cuda.current_device() if device.index is None else device.index
        if len(saved) != 1:
            raise ValueError('Only single-device CUDA checkpoints are supported')
        torch.cuda.set_rng_state(next(iter(saved.values())).cpu(), index)
    elif device.type == 'mps':
        if 'torch_mps' not in state:
            raise ValueError('Checkpoint has no MPS RNG state for MPS continuation')
        torch.mps.set_rng_state(state['torch_mps'].cpu())


def restore_loader_generators(loader, state):
    for name, owner in (('loader', loader), ('sampler', getattr(loader, 'sampler', None))):
        generator = getattr(owner, 'generator', None)
        if name in state:
            if not isinstance(generator, torch.Generator):
                raise ValueError(f'Recreate the {name} with an explicit torch.Generator')
            generator.set_state(state[name].cpu())
        elif isinstance(generator, torch.Generator):
            raise ValueError(f'Checkpoint did not use an explicit {name} generator')


def loader_spec(loader):
    from dataclasses import asdict, is_dataclass
    from torch.utils.data import DataLoader
    if isinstance(loader, (list, tuple)):
        return dict(kind='batches', length=len(loader))
    if not isinstance(loader, DataLoader):
        return dict(kind='unsupported', type=type(loader).__qualname__)
    collator = loader.collate_fn
    def safe_len(value):
        try:
            return len(value)
        except (TypeError, NotImplementedError):
            return None
    return dict(kind='dataloader', length=safe_len(loader),
                dataset_type=f'{type(loader.dataset).__module__}.{type(loader.dataset).__qualname__}',
                dataset_length=safe_len(loader.dataset),
                dataset_fingerprint=getattr(loader.dataset, '_fingerprint', None),
                batch_size=loader.batch_size, drop_last=loader.drop_last,
                num_workers=loader.num_workers,
                sampler=type(loader.sampler).__qualname__,
                sampler_replacement=getattr(loader.sampler, 'replacement', None),
                sampler_num_samples=getattr(loader.sampler, 'num_samples', None),
                collator=(asdict(collator) if is_dataclass(collator) and not isinstance(collator, type)
                          else f'{getattr(collator, "__module__", type(collator).__module__)}.'
                               f'{getattr(collator, "__qualname__", type(collator).__qualname__)}'))


def validate_resume_loader(loader, saved_spec=None):
    from torch.utils.data import (DataLoader, IterableDataset, RandomSampler,
                                  SequentialSampler, BatchSampler)
    if isinstance(loader, (list, tuple)):
        pass
    elif isinstance(loader, DataLoader):
        if loader.num_workers != 0 or isinstance(loader.dataset, IterableDataset):
            raise ValueError('Resume requires a map-style DataLoader with num_workers=0; '
                             'streaming/worker-prefetch restoration is not supported')
        if type(loader.sampler) not in (RandomSampler, SequentialSampler) or type(loader.batch_sampler) is not BatchSampler:
            raise ValueError('Resume does not support custom samplers/batch samplers')
    else:
        raise ValueError('Resume requires a map-style DataLoader or a list/tuple of fixed batches')
    if saved_spec is not None and loader_spec(loader) != saved_spec:
        raise ValueError('Loader/dataset configuration differs from the checkpoint')


def replay_epoch(loader, state, device):
    """Rebuild a deterministic iterator without consuming training RNG twice."""
    current_rng = rng_state(device)
    current_generators = loader_generator_state(loader)
    restore_rng_state(state['epoch_start_rng'], device)
    restore_loader_generators(loader, state.get('epoch_start_loader_generators', {}))
    try:
        iterator = iter(loader)
        for _ in range(state['batches_consumed_epoch']):
            try:
                next(iterator)
            except StopIteration as exc:
                raise ValueError('Checkpoint cursor exceeds the recreated loader') from exc
    finally:
        restore_rng_state(current_rng, device)
        restore_loader_generators(loader, current_generators)
    return iterator


def validate_optimizer_snapshot(optimizer, model, payload):
    """Preflight membership and buffer shapes without retaining CPU state frames."""
    names = {id(p): name for name, p in model.named_parameters()}
    membership = [[names[id(p)] for p in group['params']] for group in optimizer.param_groups]
    if membership != payload['optimizer_parameter_names']:
        raise ValueError('Optimizer parameter membership differs from checkpoint')
    optimizer.load_state_dict(payload['optimizer_state_dict'])
    children = getattr(optimizer, 'optimizers', {'single': optimizer})
    for child in children.values():
        for parameter, values in child.state.items():
            for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq', 'momentum_buffer'):
                if key in values and values[key].shape != parameter.shape:
                    raise ValueError('Optimizer buffer shape differs from parameter')
