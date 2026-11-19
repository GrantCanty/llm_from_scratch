"""Release owned training state and inspect live versus cached device memory."""
import gc

import torch


def release_optimizer(optimizer):
    """Discard a finished, internally owned optimizer, including a Muon pair.

    Do not call on an optimizer intended for continued/resumed training.
    Clearing in place also releases state if a notebook retains the object.
    """
    optimizer.zero_grad(set_to_none=True)
    children = getattr(optimizer, 'optimizers', None)
    for child in children.values() if children is not None else (optimizer,):
        child.state.clear()


def memory_report(model, device):
    """CPU-only byte counts; allocator totals include all models on this device.

    Peaks are historical (training resets them per update). CPU allocator counts
    are unavailable; this deliberately does not present process RSS as VRAM.
    """
    device = torch.device(device)
    parameters = list(model.parameters())
    report = {
        'device': str(device),
        'parameter_bytes': sum(p.numel() * p.element_size() for p in parameters),
        'buffer_bytes': sum(b.numel() * b.element_size() for b in model.buffers()),
        'gradient_bytes': sum(p.grad.numel() * p.grad.element_size()
                              for p in parameters if p.grad is not None),
        'allocated_bytes': None, 'reserved_bytes': None,
        'peak_allocated_bytes': None, 'peak_reserved_bytes': None,
    }
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        report.update(allocated_bytes=torch.cuda.memory_allocated(device),
                      reserved_bytes=torch.cuda.memory_reserved(device),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    elif device.type == 'mps':
        torch.mps.synchronize()
        report.update(allocated_bytes=torch.mps.current_allocated_memory(),
                      driver_allocated_bytes=torch.mps.driver_allocated_memory())
    return report


def collect_device_cache(device, *, empty_cache):
    """Collect unreachable Python objects, optionally release unused device cache."""
    gc.collect()
    device = torch.device(device)
    if empty_cache:
        if device.type == 'cuda':
            with torch.cuda.device(device):
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.synchronize()
            torch.mps.empty_cache()
