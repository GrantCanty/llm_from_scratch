# Project Details  
This project lets you create an LLM based off of the GPT2 architecture. This is a decoder only LLM. This package prioritizes ease of use in creating a custom decoder.  

## Functionality  
The project lets you create a decoder with the ability to change the following parameters:  
* Number of layers  
* Number of heads  
* Embedding dimensions  

The project has the following functions:  
* `train(epoch_count, optional: eval_freq, optional: eval_iter, optional: start_context, optional: drop_rate, optional: content_length)`  
* `generate(text, optional: max_tokens, optional: temperature, optional: top_k)`  
* `save_checkpoint(name=None, checkpoint=True)`
* `load_weights(path)`  

Create an instance of the package like this:  

```
import decoder

model = decoder.Decoder()
model.train(5, drop_rate=.25)
model.generate('the black cat ate')
```

## Context packing

Split original documents into training/validation before packing. The packer
concatenates tokenized documents with EOT separators and emits windows of
`context_length + 1` tokens. The collator shifts these into at most
`context_length` inputs and targets. Long documents continue across windows;
short documents fill spare space. Only the final window may need padding.

```python
import torch
from torch.utils.data import DataLoader
from decoder.packing import pack_dataset, CausalLMCollator

context_length = model.GPT_CONFIG["context_length"]
splits = full_data.train_test_split(train_size=0.8, seed=123)

# Use text_column="content" for the original Ultra-FineWeb notebook dataset.
# Use text_column="text" for HuggingFaceFW/fineweb.
tokenized_train = pack_dataset(
    splits["train"].shuffle(seed=123), tokenizer, context_length,
    text_column="text",
)
tokenized_val = pack_dataset(
    splits["test"], tokenizer, context_length, text_column="text",
)
collator = CausalLMCollator(context_length, tokenizer.eot_token)
train_loader = DataLoader(
    tokenized_train, batch_size=8, shuffle=True, collate_fn=collator,
    generator=torch.Generator().manual_seed(456),
)
val_loader = DataLoader(
    tokenized_val, batch_size=8, shuffle=False, collate_fn=collator,
)
```

If importing via a parent package, use
`from llm_from_scratch.decoder.packing import ...` instead. No changes to
`decoder/__init__.py` are necessary.

The `tokenized_train`/`tokenized_val` rows are packed windows, not original
documents; rebuild them and both loaders after changing context length. Packing
uses the provided document order and is fixed across epochs; loader shuffling
changes window order, not which documents share a window. Seeded document
shuffling before packing controls that grouping.

Tokenization is batched, but the packing buffer spans every tokenization batch.
Empty documents are skipped. Windows share a boundary token so each eligible
within-document pair is scored once. A single final shared token has no remaining
target and is not emitted again.

Packed rows carry source `document_ids`, including through window overlap. The
collator requires this metadata: regenerate legacy packed datasets and loaders.
It returns `document_ids` and reset `position_ids` alongside inputs, labels, and
padding masks. Training and evaluation pass these through every transformer layer.
Attention is causal and restricted to the same document. Appended EOT belongs to
the document it ends; source tokens with the same token ID do not create boundaries.
Last-token-to-EOT targets are scored; EOT-to-next-document targets are ignored
(`-100`). Without BOS, the first token of each document is unscored. Positions
restart at zero for each visible document segment, including long-document
continuations in a new window. Positional metrics use these reset positions.

SDPA combines document, causal, and padding masks. FA4 calls the existing dense
kernel separately for each visible document segment; this preserves isolation but
can add launch overhead for many short documents. Real cloud kernel compatibility
and performance must be verified on the installed patched build. Ordinary calls
without document metadata represent a single sequence, as used in generation;
do not strip metadata from packed training batches.

`pack_dataset` takes an already materialized Hugging Face `Dataset` and a
tiktoken-compatible tokenizer. It writes packed examples to the Hugging Face
cache rather than building the full token stream in RAM. Use `cache_dir=...` to
choose a writable or persistent location. Reuse the resulting datasets or save
them with `save_to_disk()` to preserve the exact packed baseline.

## Learning-rate warmup

Set `learning_rate` to the peak LR and choose **one** warmup budget:

```python
train_results = model.train(
    train_loader, val_loader,
    num_epochs=2, learning_rate=0.0004, drop_rate=0.25,
    warmup_tokens=10_000_000,  # Example budget; choose for your experiment.
)
# Alternatively: warmup_steps=500 for 500 optimizer updates.
learning_rates = [row['learning_rates'][0] for row in model.training_history]
```

LR increases linearly, then stays at the peak by default; optional cosine decay
is described below. Defaults
(`warmup_steps=0`, `warmup_tokens=0`) preserve constant-LR training. Token warmup
counts actual non-ignored targets, excluding padding and cross-document targets.
Each update uses its ending progress: with `warmup_steps=500`, update 1 uses
`peak_lr / 500`, update 500 uses the peak. With token warmup, an update that
reaches or crosses the budget uses the peak. Empty-target batches do not step
or advance warmup. Evaluation and epoch boundaries do not restart warmup.

Evaluation output includes LR. `model.training_history` records each completed
update's learning rates, cumulative scored tokens, batch loss, and gradient norm;
the existing `train_results` tuple is unchanged. History is kept in CPU memory.
`Decoder.train()` retains its optimizer and warmup/clipping progress after
return. Save with `save_checkpoint()` before explicit cleanup/reset. A new
`train()` call is rejected while state is retained; use `resume()` to continue
without recreating optimizer moments, warmup, or clipping history.

## Training baseline measurements

Training logs show the last update's **scored tokens/second** and CUDA peak
allocated/reserved memory in GiB. Every completed update also records
`step_seconds`, `tokens_per_second`, `peak_memory_allocated_bytes`, and
`peak_memory_reserved_bytes` in `model.training_history`. Throughput counts
non-ignored targets, excluding padding and cross-document targets.

Timing covers zeroing gradients, forward, backward, clipping, and the optimizer
step, with CUDA synchronization at both boundaries. It excludes data loading,
evaluation, and logging. Peaks reset before each update and include live model
and optimizer allocations; the first update includes optimizer-state creation.
Reserved memory includes the CUDA allocator cache, which may retain allocations
from earlier evaluation. These counters cover PyTorch-managed GPU memory, not
total device usage. On CPU/MPS, timing is host wall time and CUDA memory fields
are `None` (displayed as N/A).

For comparisons, use the same batch, context, packing, and evaluation settings.
Exclude initial warmup updates from throughput comparisons, use
`sum(scored_tokens_per_update) / sum(step_seconds)` rather than averaging rates,
and retain the maximum per-update memory peak, including the first update.
`scored_tokens` in history is cumulative, so take consecutive differences to
obtain per-update counts. Evaluation memory is excluded from these peaks.

Update-to-weight tracking is enabled by default: `update_norm_interval=100`
records update 1 and every 100th optimizer update; set it to `0` to disable or
`1` to measure every update. Empty-target batches do not count. Sampled history
rows have an `update_norms` dictionary; other rows have `None`:

```python
samples = [row for row in model.training_history if row['update_norms'] is not None]
ratios = [row['update_norms']['parameters']['tok_emb.weight']['update_to_weight_ratio']
          for row in samples]
tokens = [row['scored_tokens'] for row in samples]
```

Each sample contains global and named-parameter `weight_norm`, `update_norm`,
and `update_to_weight_ratio`, using the pre-step weight norm and the actual
single-step change, including weight decay. Ratios use a `1e-12` denominator
epsilon; interpret initially zero weights using their absolute update norm.
Parameter records include optimizer class, parameter-group index, LR, decay,
and `parameter_kind` (`matrix` versus `bias_or_norm`). The global ratio combines
squared norms over participating parameters, rather than averaging ratios.
Parameters without gradients or outside the optimizer are excluded.

Snapshots temporarily copy participating parameters in FP32 on their device.
Sampled step timing and memory include snapshot creation/storage; post-step
norm reductions and CPU metric conversion occur outside the timed section.
Exclude sampled rows when comparing throughput, and account for their memory
cost. The measurement uses actual weight differences rather than Adam internals,
so it can also measure future Muon updates. The training API defaults to AdamW; optional mixed Muon/AdamW is described below. Compare optimizer baselines with matched initialization, data order, token
budget, and evaluation settings; log validation loss alongside these diagnostics.

## Native activation and normalization

GELU and LayerNorm now use PyTorch's native implementations. GELU retains
`approximate='tanh'`, and LayerNorm retains `eps=1e-8`, affine scale/bias,
and population variance. The parameter count is unchanged; fused kernels can
introduce small floating-point differences. Normalization parameter names in
new exports and update metrics are `weight`/`bias`. Older `scale`/`shift`
weight exports still load automatically, and the GPT-2 import helper uses the
new names. New exports require this updated code.

Visualize the history after a run (or the updates recorded so far):

```python
from decoder.plotting import plot_training_history

fig, axes = plot_training_history(model.training_history)
fig.savefig('training_diagnostics.png', dpi=150, bbox_inches='tight')
# In a notebook, display the figure with plt.show() or `fig`.
```

The dashboard shows training batch loss, pre-clipping gradient norm, each
parameter group's LR, global and per-parameter update ratios, global norms,
and throughput. The default x-axis is millions of scored tokens; use
`x_axis='updates'` for update counts. By default it compares all matrix parameters across every block, including
attention, MLP, embeddings, and the output head. The ratio legend appears below
the dashboard, and figure height expands to fit it. Choose specific tensors with
`parameters=['tok_emb.weight', 'trf_blocks.0.att.W_query.weight']`.
Ratio/norm panels use only sampled steps. Zero-weight ratios and nonpositive
values on log axes are omitted. Biases and norm tensors can be selected
explicitly. Throughput distinguishes sampled steps because copying weights
adds overhead. Loss is the scored-token-weighted training update loss, not held-out validation loss.


## Optional Muon + AdamW

```python
train_results = model.train(
    train_loader, val_loader,
    optimizer_type='muon',
    learning_rate=4e-4,          # AdamW peak LR
    muon_learning_rate=4e-4,    # Muon peak LR, tune for this scaling convention
    muon_adjust_lr_fn='match_rms_adamw',
    weight_decay=.1,            # AdamW decay
    muon_weight_decay=.1,
    muon_momentum=.95,
    warmup_tokens=10_000_000,
    update_norm_interval=100,
)
```

Default `optimizer_type='adamw'` retains the full-model baseline. Mixed mode
uses native `torch.optim.Muon` for trainable 2D attention/MLP Linear weights
inside transformer blocks. Embeddings, output head, biases, normalization, and
other parameters use AdamW. Shared parameter identities are assigned once;
embedding/output identities take priority over hidden-matrix membership. No
architecture changes are made.

`match_rms_adamw` applies PyTorch's shape scaling to make update RMS comparable
to AdamW. If `muon_learning_rate=None`, this convention starts with the AdamW
LR; that is a starting point, not a guarantee of optimal tuning. `original` and
`spectral_unclamped` scaling require an explicit Muon LR. Also configurable:
`adamw_betas=(.9, .999)`, `muon_nesterov=True`, and `muon_ns_steps=5`.
Both optimizers step once per nonempty training batch, share the same warmup
factor, and participate in global gradient-norm measurement. Optional accumulation and adaptive clipping are described below; both are disabled
by default.

`learning_rates` and `optimizer_groups` in history list AdamW first, Muon second.
Group metadata records the optimizer, parameter count, LR, decay, momentum/betas,
and Muon scaling/iterations. Sampled parameter metrics identify the owning
optimizer. Both LRs appear in evaluation logs; existing plotting works unchanged.

Capacity probes use the same builder:

```python
records = profile_training_configs(
    model.GPT_CONFIG, candidates,
    optimizer_options=dict(optimizer_type='muon', muon_learning_rate=4e-4,
                           muon_adjust_lr_fn='match_rms_adamw'),
)
```

A PyTorch build exposing native Muon with these options is required. Missing
Muon fails explicitly; no automatic installation or optimizer fallback occurs.
Preserve and verify the patched FA4 environment before changing PyTorch. Local
CPU tests verify actual updates, partitioning, shared weights, warmup, and both
optimizer states round-tripping. They do not establish CUDA/FA4 compatibility
or throughput. Run a short finite cloud baseline comparison before a long run.

Full `save_checkpoint()` files include both child optimizer states. Plain
weights-only exports use `checkpoint=False`. Compare fresh runs at matched token
budgets/data order, and check validation loss and wall time. `resume()` restores
both saved optimizer states rather than starting them over.


## Gradient accumulation and adaptive clipping

```python
train_results = model.train(
    train_loader, val_loader,
    optimizer_type='muon', learning_rate=LR,
    muon_adjust_lr_fn='match_rms_adamw',
    accumulation_steps=4,
    adaptive_clip=True,
    clip_multiplier=3.0,
    clip_window=100,
    clip_min_history=20,
    warmup_steps=WARMUP_STEPS,
)
```

Reduce the DataLoader's batch size to save activation memory. For example,
physical batch 4 with `accumulation_steps=4` approximates physical batch 16
with one update, when stochastic behavior is disabled. Each microbatch runs
backward separately and releases its graph. Only input batches (normally CPU
DataLoader tensors) are buffered for a group to count its total scored targets;
activation graphs are never buffered. Gradients are weighted by actual scored
tokens, not an average of microbatch means. Empty-target batches are ignored.
The final partial group is stepped at each epoch end; no samples are dropped.
Defaults `accumulation_steps=1, adaptive_clip=False` preserve prior behavior.

After accumulation, the threshold is `clip_multiplier * mean(previous raw norms)`
over the last `clip_window` optimizer updates. The current norm is excluded.
The first `clip_min_history` updates collect history without clipping. A tiny
`1e-12` threshold floor handles zero baselines. All parameters across AdamW and
Muon share one scale factor, preserving the combined gradient direction. Raw
norms enter the history after the decision, even if clipped; repeated spikes
can therefore raise subsequent thresholds. This is a relative-limit heuristic,
not a calibrated statistical outlier test. Non-finite losses/gradients abort
before either optimizer steps. FP16 GradScaler is not supported by this loop;
its current BF16 path does not require unscaling.

History now records `microbatches`, cumulative `microbatches_seen`, actual
`scored_tokens_per_update`, accumulation/clipping configuration, `clip_baseline`,
`clip_threshold` (`None` while inactive), `clip_factor`, `was_clipped`,
`clipping_fraction` since this training call, and `grad_norm_after_clipping`.
`grad_norm` remains the pre-clipping global norm; `loss` is the group token mean.
Plots overlay the threshold and mark clipped updates. LR warmup, norm sampling,
and `eval_freq` count optimizer updates, not microbatches. Token warmup counts
all scored targets in the group, using its ending progress. Keep effective
batch size and warmup exposure comparable when changing accumulation.

Timing/throughput cover the complete accumulation group and exclude loading
buffered inputs and evaluation. Group throughput counts actual scored targets;
CUDA peaks include all its microbatches. Fresh training after explicit
cleanup/reset starts new warmup/clipping state. Full checkpoint saving captures
that state; `resume()` restores it for supported deterministic loaders.

Capacity probes accept the same `accumulation_steps`, `adaptive_clip`,
`clip_window`, `clip_multiplier`, and `clip_min_history` keyword arguments,
alongside `optimizer_options`. Synthetic microbatches repeat the same data and
are a capacity measurement, not a learning-quality test. Set enough probe
updates to exercise adaptive clipping after its history collection phase.


## Memory between notebook runs

`Decoder.train()` retains the optimizer and training progress after success or
interruption, so full state can be saved afterward. It sets gradients to `None`
and releases temporary references; it does not discard optimizer moments or
momentum. Weights and CPU history/results remain available for generation,
saving, and plotting. The lower-level `train_model_simple_function` also leaves
its caller-owned optimizer intact. Local norm snapshots and loss references are
released when the loop exits, including on failure.

```python
print(model.memory_report())            # Current bytes; allocator totals include other models.
print(model.training_memory['before'])  # At entry to the latest train call.
print(model.training_memory['after'])   # After training; optimizer retained, gradients cleared.

report = model.cleanup_memory()        # Explicitly DISCARD optimizer/progress; save first!
after_gib = {key: round(value / 2**30, 3)
             for key, value in report['after'].items()
             if key.endswith('_bytes') and value is not None}
print(after_gib)
```

Reports include parameter, buffer, and gradient bytes, plus CUDA current/peak
allocated and reserved bytes. On MPS, current and driver allocation are reported;
CUDA-only fields are `None`. On CPU, device allocator totals are `None`. Peak
values are historical, while `allocated_bytes`/`reserved_bytes` are current.
Cleanup does not reset peaks or rewrite training performance measurements.

Model weights, buffers, and optimizer state should remain allocated after
training. `cleanup_memory()` explicitly discards optimizer/progress and gradients
but preserves weights/history/results; save a full checkpoint before using it.
It also releases unused cache by default (`empty_cache=False` retains cache).
Unused cache may keep CUDA reserved memory high after live optimizer state is
freed. Emptying cache cannot free a tensor that still has a live owner.

For fresh weights with the same architecture, use `model.reset_weights(seed)`.
It releases its old model before allocating the replacement, avoiding their
previous overlap on the device. It intentionally discards current weights,
optimizer, and progress; save first if needed. External model aliases, parameters, optimizers,
GPU-resident datasets, notebook output history, and retained exception tracebacks
can still keep allocations alive. Remove obsolete references when no longer
needed. Constructing `model = Decoder(...)` can temporarily keep the previous
`model` alive while the right-hand side is evaluated; prefer `reset_weights`
for same-architecture experiments. Library cleanup does not inspect/delete
notebook globals or erase traceback information.

CPU tests cover AdamW/Muon retention and explicit cleanup, interruptions with
retained tracebacks, snapshot release, reset ordering, unchanged weights after cleanup,
and generation afterward. A CUDA-only lifecycle test checks repeated-run live
allocations against the model baseline; actual FA4/GPU memory behavior must be
verified in the preserved cloud environment.


## Saving checkpoints or weights only

```python
train_results = model.train(train_loader, val_loader, num_epochs=1)

# Full training snapshot (default): save BEFORE cleanup/reset.
model.save_checkpoint('run_checkpoint.pth', checkpoint=True,
                      metadata={'tokenizer': 'artifact/hash',
                                'dataset': 'dataset/revision/split/seed'})

# Plain model state dict, compatible with existing load_weights().
model.save_checkpoint('run_weights.pth', checkpoint=False)

# Explicitly discard optimizer/progress, keeping trained weights and plots.
model.cleanup_memory()
```

`save_checkpoint` replaces `save_weights`. Full snapshots contain a versioned
format identifier, CPU model/optimizer tensors, model/training configuration,
ordered optimizer parameter names, warmup peak LRs/budgets/counters, the raw
adaptive-clipping window/counters, epoch/batch/token progress, metric history
and completed-run results, Python/NumPy/PyTorch RNG state, exposed DataLoader/
sampler generator states, seed, and optional metadata. Both Muon and AdamW are
included in mixed mode. Runtime FA4 callables are recorded by descriptive name,
not pickled; the compatible kernel/environment must be supplied again later.

Saving makes CPU snapshots and atomically replaces the destination only after
writing/flushing a temporary file beside it. It does not discard any live state.
Parent directories must already exist. Returned paths use `.pth`. Full saving
before training or after cleanup/reset raises an error; weights-only saving is
still available. Full loading and continuation use `resume()`.
`load_weights()` accepts only weights-only files, and requires cleanup/reset
before replacing weights if an optimizer is retained.

Full snapshots require a completed optimizer-update boundary. An interruption
inside accumulation or an optimizer step is marked unsafe and full saving is
rejected; weights-only export remains possible, but is not resumable training
state. State at a completed boundary remains retained even if evaluation fails.
`train()` will not silently replace an existing optimizer; save and explicitly
cleanup/reset before requesting fresh training, or use `resume(None, ...)` to continue.

Data progress and generator states support deterministic replay as described
below. Record tokenizer/data identity, split, shuffle seed, and packing recipe
in `metadata`. Periodic/best-checkpoint policies remain separate work; optional LR decay is described below.

## Resuming training

After completing one epoch, continue to a **total target of two epochs**:

```python
model.save_checkpoint('run.pth', metadata={'tokenizer': 'artifact/hash',
                                          'dataset': 'revision/split/packing-hash'})
model.cleanup_memory()  # Explicitly release retained state before loading a file.

train_results = model.resume(
    'run.pth', train_loader, val_loader,
    num_epochs=2,  # TOTAL epochs, including those already completed.
    expected_metadata={'tokenizer': 'artifact/hash',
                       'dataset': 'revision/split/packing-hash'},
)
```

Alternatively continue the retained optimizer in the same process without
saving/loading: `model.resume(None, train_loader, val_loader, num_epochs=2)`.
A file cannot silently replace an already-retained optimizer; cleanup/reset
first, or use the in-memory continuation path. Model configuration is restored
from the file, including vocabulary size. For an FA4 checkpoint, pass the
compatible `flash_attn_func=...` or reuse the callable already configured on the
Decoder. The same device type is required (CPU, CUDA, or MPS); CUDA device index
may differ. CPU parity tests do not guarantee bitwise GPU/FA4 determinism.

For deliberate mid-epoch stopping at a completed update boundary:

```python
model.train(train_loader, val_loader, num_epochs=2, max_updates=500)
model.save_checkpoint('partial.pth')
model.cleanup_memory()
model.resume('partial.pth', train_loader, val_loader)  # Finish original 2-epoch target.
```

`max_updates` limits optimizer updates in THIS call, including in `resume()`;
it does not change the planned epoch target. `num_epochs` in `resume()` defaults
to the saved target. If that target is already complete, specify a larger total.
Warmup uses restored peak LRs and cumulative progress; adaptive clipping restores
its raw-norm window and counters. Update/token counts, evaluation cadence,
history, and evaluation result lists continue rather than restart. Optimizer
parameter membership is checked before loading, and saving works again after
resume. Saved metadata is preserved unless explicitly replaced on the next save.

Supported data contract:

- Recreate the same deterministic map-style DataLoaders with `num_workers=0`,
  standard SequentialSampler/RandomSampler and standard BatchSampler (or the
  supported ResumableBatchSampler described below), or use
  unchanged lists/tuples of fixed batches. Datasets, transforms, and collation
  must be reproducible and must not depend on untracked mutable state.
- Recreate explicit loader/sampler generators where the original used them;
  saved states replace their seeds. Validation-loader generator state is also
  restored. RandomSampler without a separate generator is covered by global RNG
  restoration for the supported single-process case.
- Ordinary samplers and fixed batch lists retain the legacy epoch-prefix replay
  path: consumed batches are fetched again but not trained again. For direct
  continuation without fetching that prefix, use ResumableBatchSampler below.
  Empty-target batches count toward the cursor; accumulation and partial tails
  retain their policy.
- Saved loader shapes/options, collator configuration (when a dataclass), and
  dataset fingerprints (when exposed) are checked. They cannot prove that an
  arbitrary dataset with the same length contains identical text. Preserve the
  source data and verify its identity using `expected_metadata`.
- Streaming/IterableDataset, worker prefetch (`num_workers>0`), and custom
  samplers/batch samplers other than ResumableBatchSampler are rejected.
  No automatic kernel downloads or environment changes occur.

Earlier version-1 files without loader specifications can be read using the
same supported loaders; metadata/data identity must be verified by the caller.
Files lacking completed-run evaluation results cannot reconstruct metrics that
were never saved. Resume does not recover a partially applied update: unsafe
checkpoints are still rejected. Keep periodic checkpoints for interruption
recovery; automatic checkpoint scheduling remains a future step.


## Direct resume from committed dataset position

After a completed epoch, continue from a full training checkpoint on a loader
built exclusively from unseen data:

```python
model.resume(checkpoint_path, new_train_loader, val_loader, new_data=True)
```

This adds one pass over the new loader and carries forward the optimizer,
learning-rate schedule, and cumulative training counters. Select unseen source
documents before tokenizing and packing. The checkpoint records positions in
the previous packed dataset and cannot identify duplicate source text in the
new loader. Keep the validation loader configuration the same. To resume a
checkpoint saved partway through this new pass, use ordinary `resume()` with
the same new loader and omit `new_data`.

Use the checkpointable batch sampler for new runs on packed, map-style datasets:

```python
from torch.utils.data import DataLoader
from decoder.sampling import ResumableBatchSampler

train_batch_sampler = ResumableBatchSampler(
    tokenized_train, batch_size=4, shuffle=True, seed=456, drop_last=False,
)
train_loader = DataLoader(
    tokenized_train,
    batch_sampler=train_batch_sampler,
    collate_fn=gpt2_collate_fn,
    num_workers=0,
    generator=torch.Generator().manual_seed(789),
)
```

Do not also pass DataLoader `batch_size`, `shuffle`, `sampler`, or `drop_last`;
those settings belong to the batch sampler. Recreate this same loader when
calling `resume()`. The existing save/resume APIs need no extra arguments.
The main notebook uses this sampler; ordinary loaders/older checkpoints remain
supported through their existing replay path. Switching an old checkpoint to a
different sampler configuration is rejected; start a new run to use direct resume.

The full `.pth` checkpoint now includes
`training_state['sampler_state']`: epoch, full shuffled index order, committed
`position` within that order, dataset length, batch size, shuffle seed/policy,
drop-last policy, and sampler generator state. This is a position in the saved
order, not the numeric dataset index. Order storage costs about 8 bytes per
dataset example, on CPU/disk. Each new epoch uses a deterministic seed derived
from `seed + epoch`; intentional repeat exposure in later epochs is unchanged.

Indices requested by the loader are tracked separately from the committed
cursor. Merely iterating/previewing/prefetching batches does not commit them.
The training loop commits all input batches in an accumulation group only after
both optimizer steps succeed. Zero-target batches are included in that consumed
prefix, committed with the next successful group or at normal epoch end when
there are no pending gradients. Failed updates leave the cursor unchanged and
full checkpoint saving remains blocked while the update is unsafe.

Partial final batches retain their actual size. With `drop_last=True`, the
intentionally dropped tail is not marked consumed/trained. `len(train_loader)`
continues to describe the full epoch, not the remaining suffix. The sampler's
`position` and `training_state['examples_consumed_epoch']` expose the committed
count for this epoch; `examples_consumed` counts committed example exposure
across epochs. History includes these example counters on optimizer updates.

On direct resume, the saved order/cursor are restored and the loader begins at
the next unconsumed example. Earlier examples are neither fetched for replay nor
trained again. Training-loss evaluation uses a read-only view from the epoch's
start and cannot advance the training cursor; evaluation reads are deliberately
separate from consumed training exposure. Read-only train-loss evaluation now
uses the same epoch prefix rather than samples from a newly shuffled iterator.

Exact data/augmentation identity and deterministic single-process loading are
still required. Dataset fingerprints and sampler settings are checked; use
checkpoint metadata for source/split/packing hashes as before. Although issued
and committed positions are separated for prefetch safety, multi-worker RNG and
prefetch restoration are not yet supported: keep `num_workers=0`. Custom
samplers other than this explicit supported class remain rejected on resume.


## Learning-rate decay

Cosine decay is optional; `lr_decay='constant'` preserves existing training.
Use an explicit total scored-token horizon (including warmup):

```python
train_results = model.train(
    train_loader, val_loader, num_epochs=1,
    optimizer_type='muon', learning_rate=LR,
    warmup_tokens=6_500_000,
    lr_decay='cosine', decay_tokens=79_000_000,
    min_lr_ratio=0.1,
    accumulation_steps=4,
)
```

This warms up linearly through 6.5M actual scored tokens, then cosine-decays from
peak LR to 10% of peak at 79M total scored tokens. `decay_tokens` is the endpoint,
not an additional duration after warmup. Set it to the planned exposure across
ALL epochs; repeated exposure counts again, ignored labels do not. The horizon
does not stop training: afterward the LR stays at its minimum. Epoch count and
`max_updates` retain their separate meanings.

Alternatively pair `warmup_steps=400` with `lr_decay='cosine', decay_steps=4830`.
Steps mean completed optimizer updates, not microbatches. Exactly one positive
decay horizon is required for cosine; it must exceed warmup and use the same
unit. Mixed step/token warmup/decay is rejected. Without warmup, decay starts
immediately. `min_lr_ratio` must be in [0, 1]; 0 allows a zero endpoint, while
0.1 is a configurable starting value rather than an established optimum.

For progress p, warmup W, and endpoint D, the post-warmup factor is
`min_lr_ratio + (1-min_lr_ratio) * (1+cos(pi*(p-W)/(D-W))) / 2`, with progress
clamped at the endpoint. Each update uses its ENDING progress, consistent with
existing warmup. Crossing the token endpoint in a group therefore applies the
minimum LR to that update. Both AdamW and Muon use the same factor on their own
saved peak LRs; relative group LR differences are preserved.

History records `lr_factor` and `lr_phase` (`warmup`, `constant`, `decay`, or
`minimum`), and the existing LR plot shows the curve automatically. Full
checkpoints preserve schedule configuration, peak LRs, and progress. Resume
continues the curve without restarting warmup or rescaling an already-decayed
LR. Extending the epoch target does not stretch the original decay horizon;
beyond it, training stays at the saved floor. Older constant-LR checkpoints
without decay fields still resume as before. No automatic LR-decay retuning or
horizon changes are made during resume.


## Two concurrent training trials on one GPU

Use `run_sweep.py` on the cloud machine, from the checkout root. It starts one
fresh Python process per trial and never runs the marimo notebook, its setup,
or the download script. It requires CUDA and previously packed datasets; no
model/dataset downloads or dependency installation are performed by the runner.

First persist the prepared packed datasets once in the cloud notebook:

```python
tokenized_train.save_to_disk('/marimo/data/packed_train')
tokenized_val.save_to_disk('/marimo/data/packed_val')
```

Edit `examples/two_trials.json` to match those paths, model configuration, padding
ID, and the existing patched FA4 source/dependency paths. Its default is two
identical 177M-style trials with physical batch 16, accumulation 1, 500 updates
maximum, 6.5M-token warmup, and constant peak LR. These are short throughput
probes, not a full training budget or LR-quality comparison. Validation uses a
fixed batch size and fixed examples across trials.

From a terminal, inspect the plan without importing Torch or launching workers:

```bash
python run_sweep.py --config examples/two_trials.json --output /marimo/sweeps/parallel --parallel 2 --dry-run
```

Then run the two workers on the same visible GPU:

```bash
python run_sweep.py --config examples/two_trials.json --output /marimo/sweeps/parallel --parallel 2
```

Both workers inherit the current CUDA visibility; no MIG/MPS setup is changed.
The optional runtime `cuda_device` selects a logical CUDA index, default 0.
They reuse the installed environment; runtime `python_paths` exposes the already
patched FA4 build after importing the environment's Torch/NumPy. Prepare and
verify that environment first. Each process has independent model, optimizer,
RNG, dataset cursor, and allocator. Existing notebook models still occupy VRAM;
save and call `cleanup_memory()` on finished runs to release optimizer/cache
before testing concurrent workers. Their model weights remain allocated.

Each trial directory receives `config.json`, `trial.log`, `checkpoint.pth`,
`history.json`, and `metrics.json`. The launcher writes `summary.json` and
`summary.csv`, reports child failures, and refuses to overwrite existing trial
directories. A worker failure does not cancel the other trial. No test dataset
is loaded or used to choose settings.

For a sequential control, run the SAME two-trial configuration with
`--parallel 1` and a different output directory. Compare:

- `aggregate_wall_tokens_per_second`: total successful tokens divided by total
  sweep wall time, including worker startup, initialization, final evaluation,
  and checkpoint saving. Do not count a failed sweep as a successful comparison.
- Per-trial `wall_tokens_per_second`: training-call throughput including loading,
  intermediate evaluation, logging, and compilation during the call.
- `steady_step_tokens_per_second`: update-only throughput after the configured
  initial timing warmup, excluding norm-snapshot steps. It excludes data loading
  and evaluation, matching the underlying step timer's scope.
- Peak training allocated/reserved bytes, validation loss/accuracy, actual
  scored-token counts, and update counts. Training memory peaks exclude separate
  evaluation peaks and non-PyTorch allocations; retain device headroom.

Start with two workers; extra VRAM alone does not establish additional compute
capacity. Concurrent trials may slow individually; aggregate work per elapsed
second is the comparison. Use enough updates that FA4 compilation/startup does
not dominate the result.

To make a sweep after checking throughput, each `trials` entry can override
`batch_size` and/or the `training` dictionary:

```json
{"name": "lr_0p001", "training": {"learning_rate": 0.001}}
```

Keep initial model/data seeds, tokenizer, data artifacts, validation batch size,
and effective batch fixed for schedule comparisons. For physical batch trials,
compare (4, accumulation 4), (8, accumulation 2), and (16, accumulation 1).
Confirm actual scored-token counts match: `max_updates` is an update limit,
not a token budget. Set warmup/decay horizons for the chosen experimental budget;
short runs may not meaningfully distinguish quality or reach a decay phase.

### Minimal MPS smoke test

`train.py` is independent of the project training pipeline. It downloads nothing,
uses a small four-layer transformer and a fixed synthetic batch on CUDA, and
prints timed tokens/second and peak PyTorch memory after 20 warmup steps.
The repeated batch is only a compute test, not a model-quality experiment.

On the GPU host, start the MPS daemon once (if your environment supports it):

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
```

Measure one client first, then the same workload with two clients:

```bash
CUDA_MPS_PINNED_DEVICE_MEM_LIMIT="0=20G" python train.py --name single

CUDA_MPS_PINNED_DEVICE_MEM_LIMIT="0=20G" python train.py --name a --lr 1e-3 > mps-a.log 2>&1 &
pid_a=$!
CUDA_MPS_PINNED_DEVICE_MEM_LIMIT="0=20G" python train.py --name b --lr 5e-4 > mps-b.log 2>&1 &
pid_b=$!
wait "$pid_a"; status_a=$?
wait "$pid_b"; status_b=$?
echo "Exit codes: $status_a $status_b"
tail -n 2 mps-a.log mps-b.log
```

While both are training, use another terminal with the same pipe-directory export
to run `echo get_server_list | nvidia-cuda-mps-control`, then
`echo "get_client_list SERVER_PID" | nvidia-cuda-mps-control` with the returned
server PID. Verify both clients appear; the script printing the environment
variable does not establish an MPS connection by itself. See NVIDIA's
[MPS tools reference](https://docs.nvidia.com/deploy/mps/latest/appendix-tools-and-interface-reference.html).

Compare the sum of both clients' tokens/second against the single-client result.
Increase `--steps` if the runs are too short to overlap substantially. These
results apply to this synthetic workload; benchmark the actual model afterward.
The 20G setting is a per-client allocation ceiling, not a reservation or compute
partition. No active-thread limit is set. This script does not start/stop MPS,
change GPU configuration, or save checkpoints. Local verification covers CLI and
syntax only; CUDA/MPS execution requires the GPU host.
