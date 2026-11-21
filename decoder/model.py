import torch
from . import transformer
from . import utils
from .optimizers import optimizer_group_metadata
from .checkpoint import (rng_state, loader_generator_state, loader_spec, replay_epoch, direct_epoch, cpu_snapshot)
from .training_controls import AdaptiveGradientClipper, scored_microbatch_groups, validate_accumulation_steps
from .schedules import LearningRateSchedule
from .sampling import resumable_sampler, evaluation_view, commit_loader_position
from .update_metrics import snapshot_parameters, measure_updates
import numpy as np
import math
import time
from numbers import Integral


class GPTModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention_backend = cfg.get("attention_backend", "sdpa")
        self.tok_emb = torch.nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = torch.nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = torch.nn.Dropout(0)
        self.trf_blocks = torch.nn.Sequential(
            *[transformer.TransformerBlock(cfg) for _ in range(cfg['n_layers'])])
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = torch.nn.Linear(
            cfg['emb_dim'], cfg['vocab_size'], bias=False
        )
    
    def forward(self, in_idx, attention_mask=None, document_ids=None, position_ids=None):
        # Keep master parameters in FP32; autocast covers training, eval and generation.
        if self.attention_backend == "fa4":
            if in_idx.device.type != "cuda":
                raise ValueError("The fa4 backend requires a CUDA device")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return self._forward(in_idx, attention_mask, document_ids, position_ids)
        return self._forward(in_idx, attention_mask, document_ids, position_ids)

    def _forward(self, in_idx, attention_mask=None, document_ids=None, position_ids=None):
        batch_size, seq_len = in_idx.shape
        if seq_len > self.pos_emb.num_embeddings:
            raise ValueError("Input sequence exceeds the model context length")
        if attention_mask is not None:
            if attention_mask.shape != in_idx.shape:
                raise ValueError("attention_mask must have the same shape as input IDs")
            attention_mask = attention_mask.to(device=in_idx.device, dtype=torch.bool)
            if not attention_mask[:, 0].all():
                raise ValueError("Use nonempty, right-padded sequences")
        if document_ids is not None:
            document_ids = document_ids.to(in_idx.device)
            if document_ids.shape != in_idx.shape:
                raise ValueError("document_ids must have the same shape as input IDs")
            valid = attention_mask if attention_mask is not None else torch.ones_like(in_idx, dtype=torch.bool)
            if (document_ids[valid] < 0).any() or ((document_ids[:, 1:] < document_ids[:, :-1]) & valid[:, 1:]).any():
                raise ValueError("Valid document_ids must be nonnegative and nondecreasing")
            offsets = torch.arange(seq_len, device=in_idx.device).expand_as(in_idx)
            starts = torch.cat((torch.ones_like(in_idx[:, :1], dtype=torch.bool),
                                document_ids[:, 1:] != document_ids[:, :-1]), dim=1)
            expected = offsets - torch.where(starts, offsets, 0).cummax(1).values
            expected = expected.masked_fill(~valid, 0)
            if position_ids is not None and not torch.equal(position_ids.to(in_idx.device), expected):
                raise ValueError("position_ids must reset at each visible document segment")
            position_ids = expected
        elif position_ids is None:
            position_ids = torch.arange(seq_len, device=in_idx.device)
        position_ids = position_ids.to(in_idx.device)
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(position_ids)
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        for block in self.trf_blocks:
            x = block(x, attention_mask=attention_mask, document_ids=document_ids)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    

class LayerNorm(torch.nn.LayerNorm):
    """Native LayerNorm with the baseline epsilon and legacy weight loading."""

    def __init__(self, emb_dim):
        super().__init__(emb_dim, eps=1e-8)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Earlier weight exports used scale/shift instead of weight/bias.
        for old, new in (("scale", "weight"), ("shift", "bias")):
            old_key, new_key = prefix + old, prefix + new
            if old_key in state_dict:
                if new_key not in state_dict:
                    state_dict[new_key] = state_dict[old_key]
                del state_dict[old_key]
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class GELU(torch.nn.GELU):
    """Native GELU using the baseline tanh approximation."""

    def __init__(self):
        super().__init__(approximate="tanh")


class FeedForward(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            GELU(),
            torch.nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim'])
        )
    
    def forward(self, x):
        return self.layers(x)
    

def train_model_simple_function(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter,
                                *, warmup_steps=0, warmup_tokens=0, update_history=None, update_norm_interval=100,
                                accumulation_steps=1, adaptive_clip=False, clip_window=100,
                                clip_multiplier=3., clip_min_history=20, training_state=None,
                                resume_state=None, max_updates=None, lr_decay='constant',
                                decay_steps=0, decay_tokens=0, min_lr_ratio=.1):
    """Linear warmup with optional cosine decay; existing result tuple is preserved.

    Warmup uses either optimizer updates or non-ignored targets. Each update uses
    the LR at its ending progress (first update: peak / warmup_steps, or
    peak * batch_scored_tokens / warmup_tokens). Empty-target batches do not
    advance the schedule. Accumulation uses the group's scored-token mean,
    flushing partial groups at epoch end. Clipping follows normalization, once
    across both optimizers. Fresh calls reset progress; resume_state restores it.
    """
    schedule = LearningRateSchedule(warmup_steps=warmup_steps, warmup_tokens=warmup_tokens,
                                    lr_decay=lr_decay, decay_steps=decay_steps,
                                    decay_tokens=decay_tokens, min_lr_ratio=min_lr_ratio)
    if (isinstance(update_norm_interval, bool) or
            not isinstance(update_norm_interval, Integral) or update_norm_interval < 0):
        raise ValueError("update_norm_interval must be a nonnegative integer")
    validate_accumulation_steps(accumulation_steps)
    clipper = AdaptiveGradientClipper(adaptive_clip, clip_window, clip_multiplier, clip_min_history)
    if max_updates is not None and (isinstance(max_updates, bool) or
            not isinstance(max_updates, Integral) or max_updates < 1):
        raise ValueError('max_updates must be a positive integer or None')
    if training_state is None and resumable_sampler(train_loader) is not None:
        training_state = {}
    prior = cpu_snapshot(resume_state) if resume_state is not None else None
    microbatches_seen = 0
    peak_lrs = [group['lr'] for group in optimizer.param_groups]
    train_losses, val_losses, val_perplexities, track_tokens_seen = [], [], [], []
    train_loss_by_pos_list, val_loss_by_pos_list = [], []
    val_accuracies = []
    peak_reserves = []
    tokens_seen, global_step = 0, -1
    cuda_device = torch.device(device) if torch.device(device).type == 'cuda' else None
    start_epoch = 0
    if prior is not None:
        if not prior.get('checkpoint_safe'):
            raise ValueError('Cannot resume an incomplete optimizer update')
        if (prior['warmup']['steps'], prior['warmup']['tokens']) != (warmup_steps, warmup_tokens):
            raise ValueError('Warmup configuration differs from checkpoint')
        saved_schedule = prior.get('lr_schedule')
        if ((saved_schedule is not None and saved_schedule != schedule.state_dict()) or
                (saved_schedule is None and lr_decay != 'constant')):
            raise ValueError('Learning-rate schedule differs from checkpoint')
        clipper.load_state_dict(prior['adaptive_clipping'])
        microbatches_seen = prior['microbatches_seen']
        tokens_seen, global_step = prior['scored_tokens'], prior['updates'] - 1
        peak_lrs = prior['warmup']['peak_lrs']
        start_epoch = prior['epoch'] + int(prior['epoch_complete'])
        previous_results = prior.get('evaluation_results')
        if previous_results is not None:
            (train_losses, train_loss_by_pos_list, val_losses, val_loss_by_pos_list,
             val_perplexities, val_accuracies, track_tokens_seen, previous_peak) = cpu_snapshot(previous_results)
            if previous_peak is not None:
                peak_reserves.append(previous_peak)
    stop_update = global_step + 1 + max_updates if max_updates is not None else None

    def results():
        values = (train_losses, train_loss_by_pos_list, val_losses, val_loss_by_pos_list,
                  val_perplexities, val_accuracies, track_tokens_seen,
                  max((peak for peak in peak_reserves if peak is not None), default=None))
        if training_state is not None:
            training_state['evaluation_results'] = values
        return values

    if training_state is not None:
        training_state.update(status='running', checkpoint_safe=True, updates=global_step + 1,
                              scored_tokens=tokens_seen, microbatches_seen=microbatches_seen,
                              epoch=start_epoch, epoch_complete=False,
                              batches_consumed_epoch=(prior['batches_consumed_epoch']
                                  if prior and not prior['epoch_complete'] else 0),
                              warmup=dict(steps=warmup_steps, tokens=warmup_tokens,
                                          peak_lrs=peak_lrs),
                              lr_schedule=schedule.state_dict(),
                              adaptive_clipping=clipper.state_dict(),
                              loader_specs=dict(train=loader_spec(train_loader), val=loader_spec(val_loader)),
                              examples_consumed=prior.get('examples_consumed', 0) if prior else 0)
        results()

    # Release local snapshots/graphs even when a notebook retains a traceback.
    # External optimizer ownership is unchanged for this lower-level API.
    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            replaying = prior is not None and epoch == start_epoch and not prior['epoch_complete']
            if replaying:
                epoch_loader = (direct_epoch(train_loader, prior['sampler_state'], device)
                                if prior.get('sampler_state') is not None
                                else replay_epoch(train_loader, prior, device))
                if training_state is not None:
                    training_state.update(epoch=epoch, epoch_start_rng=prior['epoch_start_rng'],
                        epoch_start_loader_generators=prior.get('epoch_start_loader_generators', {}))
            else:
                sampler = resumable_sampler(train_loader)
                if sampler is not None:
                    sampler.set_epoch(epoch, reset=(prior is None and epoch == 0))
                epoch_loader = train_loader
                if training_state is not None:
                    training_state.update(epoch=epoch, epoch_complete=False, batches_consumed_epoch=0,
                                          epoch_start_rng=rng_state(device),
                                          epoch_start_loader_generators=loader_generator_state(train_loader))
            for microbatches in scored_microbatch_groups(epoch_loader, accumulation_steps, training_state):
                scored_tokens = sum(count for _, count in microbatches)
                # Measure complete updates, excluding loader/evaluation time. Resetting
                # peaks retains live allocations (weights, gradients, optimizer state).
                if cuda_device is not None:
                    torch.cuda.synchronize(cuda_device)
                    torch.cuda.reset_peak_memory_stats(cuda_device)
                step_start = time.perf_counter()
                optimizer.zero_grad()
                loss_value = 0.0
                for unpacked, count in microbatches:
                    input_batch, target_batch, attention_mask, document_ids, position_ids = unpacked
                    loss = utils.calc_loss_batch(input_batch, target_batch, model, device, attention_mask,
                                                 document_ids=document_ids, position_ids=position_ids)
                    if not torch.isfinite(loss).item():
                        raise FloatingPointError('Non-finite training loss; optimizer update aborted')
                    weight = count / scored_tokens
                    (loss * weight).backward()
                    loss_value += loss.detach().item() * weight
                    del loss
                clipping = clipper.apply(model.parameters())
                grad_norm = clipping['grad_norm']

                factor, lr_phase = schedule.at(updates=global_step + 2,
                                               tokens=tokens_seen + scored_tokens)
                for group, peak_lr in zip(optimizer.param_groups, peak_lrs):
                    group['lr'] = peak_lr * factor
                sample_update = (update_history is not None and update_norm_interval > 0
                                 and ((global_step + 2) == 1 or
                                      (global_step + 2) % update_norm_interval == 0))
                snapshot = snapshot_parameters(model, optimizer) if sample_update else None
                optimizer.step()
                commit_loader_position(train_loader, training_state)
                if training_state is not None:
                    training_state.update(checkpoint_safe=True, updates=global_step + 2,
                                          scored_tokens=tokens_seen + scored_tokens,
                                          microbatches_seen=microbatches_seen + len(microbatches),
                                          adaptive_clipping=clipper.state_dict(),
                                          loader_generators=loader_generator_state(train_loader))
                if cuda_device is not None:
                    torch.cuda.synchronize(cuda_device)
                step_seconds = time.perf_counter() - step_start
                tokens_per_second = scored_tokens / step_seconds
                peak_allocated = (torch.cuda.max_memory_allocated(cuda_device)
                                  if cuda_device is not None else None)
                peak_reserved = (torch.cuda.max_memory_reserved(cuda_device)
                                 if cuda_device is not None else None)
                update_norms = measure_updates(snapshot, optimizer) if snapshot is not None else None
                del snapshot
                microbatches_seen += len(microbatches)
                tokens_seen += scored_tokens
                global_step += 1
                if update_history is not None:
                    update_history.append({
                        'update_norms': update_norms,
                        'update': global_step + 1, 'scored_tokens': tokens_seen,
                        'optimizer_groups': optimizer_group_metadata(optimizer),
                        'learning_rates': [group['lr'] for group in optimizer.param_groups],
                        'lr_factor': factor, 'lr_phase': lr_phase,
                        **clipping, 'loss': loss_value,
                        'microbatches': len(microbatches), 'microbatches_seen': microbatches_seen,
                        'scored_tokens_per_update': scored_tokens,
                        'examples_consumed': (training_state.get('examples_consumed')
                                              if training_state is not None else None),
                        'examples_consumed_epoch': (training_state.get('examples_consumed_epoch')
                                                    if training_state is not None else None),
                        'accumulation_steps': accumulation_steps,
                        'adaptive_clip': adaptive_clip, 'clip_window': clip_window,
                        'clip_multiplier': clip_multiplier, 'clip_min_history': clip_min_history,
                        'step_seconds': step_seconds,
                        'tokens_per_second': tokens_per_second,
                        'peak_memory_allocated_bytes': peak_allocated,
                        'peak_memory_reserved_bytes': peak_reserved,
                    })

                if global_step % eval_freq == 0:
                    with evaluation_view(train_loader), evaluation_view(val_loader):
                        train_loss, train_loss_by_pos, val_loss, val_loss_by_pos, val_perplexity, val_accuracy = evaluate_model(model, train_loader, val_loader, device, eval_iter)

                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    val_perplexities.append(val_perplexity)
                    val_accuracies.append(val_accuracy)

                    track_tokens_seen.append(tokens_seen)

                    train_loss_by_pos_list.append(train_loss_by_pos.detach().cpu())

                    val_loss_by_pos_list.append(val_loss_by_pos.detach().cpu())

                    peak_reserves.append(peak_reserved)

                    lr_summary = ', '.join(
                        f"{group.get('optimizer_name', type(optimizer).__name__)}[{index}]={group['lr']:.3g}"
                        for index, group in enumerate(optimizer.param_groups))
                    print(f'epoch {epoch+1} (step {global_step:06d} - tokens seen {tokens_seen})\ntrain_loss: {train_loss:.3f} val_loss: {val_loss:.3f} val_perplexity: {val_perplexity:.2f} val_accuracy: {val_accuracy:.2%} grad_norm: {grad_norm:.2f} clip_factor: {clipping["clip_factor"]:.3f} lr: {lr_summary}')
                    memory = (f'{peak_allocated / 2**30:.2f} GiB allocated, '
                              f'{peak_reserved / 2**30:.2f} GiB reserved'
                              if peak_allocated is not None else 'N/A (non-CUDA)')
                    print(f'last update: {tokens_per_second:,.0f} scored tokens/s | peak memory: {memory}')
                    results()

                if stop_update is not None and global_step + 1 >= stop_update:
                    if training_state is not None:
                        training_state['status'] = 'stopped'
                    return results()

            commit_loader_position(train_loader, training_state)
            if training_state is not None:
                training_state.update(epoch_complete=True, checkpoint_safe=True,
                                      loader_generators=loader_generator_state(train_loader))
        if training_state is not None:
            training_state['status'] = 'complete'
        return results()
    finally:
        if training_state is not None:
            training_state['loader_generators'] = loader_generator_state(train_loader)
            training_state['val_loader_generators'] = loader_generator_state(val_loader)
            val_sampler = resumable_sampler(val_loader)
            if val_sampler is not None:
                training_state['val_sampler_state'] = val_sampler.state_dict()
        loss = snapshot = microbatches = unpacked = epoch_loader = None
        input_batch = target_batch = attention_mask = document_ids = position_ids = None
        train_loss_by_pos = val_loss_by_pos = None


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = utils.calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_metrics = utils.calc_metrics_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()

    train_loss_by_pos = train_loss['loss_by_pos']
    train_loss = train_loss['loss']

    val_loss = val_metrics["loss"]
    val_loss_by_pos = val_metrics['loss_by_pos']
    val_perplexity = math.exp(val_loss)
    val_accuracy = val_metrics["accuracy"]

    return train_loss, train_loss_by_pos, val_loss, val_loss_by_pos, val_perplexity, val_accuracy

def generate_and_print_sample(model, tokenizer, device, start_context, max_new_tokens=15, top_k=25, temperature=1):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = utils.text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = utils.generate_text_simple(
            model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
            top_k=top_k,
            temperature=temperature,
            eos_id=getattr(tokenizer, "eot_token", None),
        )
        decoded_text = utils.token_ids_to_text(token_ids, tokenizer)
        #print(decoded_text.replace('\n', ' '))
    model.train()
    t = decoded_text.replace('\n', ' ')
    return t

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])
        gpt.trf_blocks[b].norm1.weight = assign(
            gpt.trf_blocks[b].norm1.weight,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.bias = assign(
            gpt.trf_blocks[b].norm1.bias,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.weight = assign(
            gpt.trf_blocks[b].norm2.weight,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.bias = assign(
            gpt.trf_blocks[b].norm2.bias,
            params["blocks"][b]["ln_2"]["b"])
    gpt.final_norm.weight = assign(gpt.final_norm.weight, params["g"])
    gpt.final_norm.bias = assign(gpt.final_norm.bias, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
