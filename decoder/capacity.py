"""Probe full-model training capacity on synthetic, full-length batches."""
import gc
import time

import torch

from .model import GPTModel
from .utils import calc_loss_batch


def _validate(cfg, batch_size):
    for key in ('vocab_size', 'context_length', 'emb_dim', 'n_heads', 'n_layers'):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    if cfg['emb_dim'] % cfg['n_heads']:
        raise ValueError('emb_dim must be divisible by n_heads')


def _run_trial(cfg, batch_size, device, warmup_steps, measured_steps, learning_rate, embedding_dropout):
    # Everything owning tensors is local; do not return tensors or exceptions.
    model = GPTModel(cfg).to(device).train()
    model.drop_emb = torch.nn.Dropout(embedding_dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    tokens = torch.randint(cfg['vocab_size'], (batch_size, cfg['context_length'] + 1), device=device)
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    mask = torch.ones_like(inputs, dtype=torch.bool)
    # Exercise the packed-document path rather than measuring unrestricted attention.
    segment_length = cfg.get('probe_document_length', max(1, cfg['context_length'] // 2))
    if not isinstance(segment_length, int) or segment_length < 1:
        raise ValueError('probe_document_length must be a positive integer')
    offsets = torch.arange(cfg['context_length'] + 1, device=device)
    docs = (offsets // segment_length).expand(batch_size, -1).clone()
    docs[:, -1] = docs[:, -2]  # Final synthetic target acts as a within-document EOT.
    document_ids = docs[:, :-1]
    position_ids = (offsets[:-1] % segment_length).expand_as(inputs)
    targets = targets.masked_fill(docs[:, :-1] != docs[:, 1:], -100)
    parameters = sum(p.numel() for p in model.parameters())
    # Include model allocation, first-use optimizer state, and warmup in peaks.
    for step in range(warmup_steps + measured_steps):
        if step == warmup_steps:
            torch.cuda.synchronize(device)
            started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        # Compare backends at the same precision, including the current loss helper.
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = calc_loss_batch(inputs, targets, model, device, mask,
                                   document_ids=document_ids, position_ids=position_ids)
        loss.backward()
        optimizer.step()
        del loss
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        'status': 'ok',
        'parameters': parameters,
        'seconds_per_step': elapsed / measured_steps,
        'tokens_per_second': (targets != -100).sum().item() * measured_steps / elapsed,
    }


def _try_trial(*args, **kwargs):
    # Leave the exception handler before cleanup to release its traceback frames.
    try:
        return _run_trial(*args, **kwargs)
    except torch.cuda.OutOfMemoryError:
        return {'status': 'oom'}
    except Exception as exc:
        return {'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}


def profile_training_configs(base_cfg, candidates, *, flash_attn_func=None,
                             device='cuda', warmup_steps=2, measured_steps=3,
                             learning_rate=4e-4, embedding_dropout=.25,
                             headroom_fraction=.1):
    """Return CPU-only records for candidate dicts containing batch_size and cfg overrides.

    Uses full-length, unpadded batches and the current training loss helper. This
    changes no existing model weights. Close unused GPU models first for an
    accurate capacity estimate. JIT kernels/context may remain resident afterward.
    """
    if warmup_steps < 1 or measured_steps < 1:
        raise ValueError('Use at least one warmup and one measured step')
    if not 0 <= headroom_fraction < 1:
        raise ValueError('headroom_fraction must be in [0, 1)')
    device = torch.device(device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('This capacity probe requires a CUDA GPU')
    device = torch.device('cuda', torch.cuda.current_device() if device.index is None else device.index)
    records = []
    with torch.cuda.device(device), torch.random.fork_rng(devices=[device.index]):
        for candidate in candidates:
            overrides = dict(candidate)
            batch_size = overrides.pop('batch_size')
            cfg = {**base_cfg, **overrides}
            if flash_attn_func is not None:
                cfg['flash_attn_func'] = flash_attn_func
            _validate(cfg, batch_size)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            free_before, total = torch.cuda.mem_get_info(device)
            reserved_before = torch.cuda.memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device)
            torch.manual_seed(123)
            record = {
                'batch_size': batch_size, 'context_length': cfg['context_length'],
                'emb_dim': cfg['emb_dim'], 'n_heads': cfg['n_heads'],
                'head_dim': cfg['emb_dim'] // cfg['n_heads'], 'n_layers': cfg['n_layers'],
                'backend': cfg.get('attention_backend', 'sdpa'),
                'document_isolation': True,
                'probe_document_length': cfg.get('probe_document_length', max(1, cfg['context_length'] // 2)),
            }
            record.update(_try_trial(cfg, batch_size, device, warmup_steps,
                                     measured_steps, learning_rate, embedding_dropout))
            peak_reserved = torch.cuda.max_memory_reserved(device)
            estimated_free_at_peak = free_before - max(0, peak_reserved - reserved_before)
            record.update(
                peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                peak_reserved_gib=peak_reserved / 2**30,
                within_headroom=(record['status'] == 'ok' and
                                 estimated_free_at_peak >= total * headroom_fraction),
            )
            gc.collect()
            torch.cuda.empty_cache()
            record['allocated_after_gib'] = torch.cuda.memory_allocated(device) / 2**30
            records.append(record)
            print(record, flush=True)
    return records
