# Training roadmap

Status: implementation has begun with document packing, explicitly requested by the user. Other phases remain planned; this document does not authorize implementing everything immediately.

## Objective and current baseline

Build and evaluate a small multilingual causal language model, initially around the current 175M-parameter architecture. Target languages: English, French, Spanish, German, and potentially Italian.

The project has shifted next-token targets, padding-aware loss, positional loss metrics, SDPA and optional FA4 attention, and capacity-probing work. FA4's SM120 dependency/source patches passed the user's cloud tests. Preserve that working environment and record its versions and source revision before changing it. Local CPU tests cannot establish cloud kernel compatibility.

Current reference context is 512 tokens. Larger contexts and model dimensions are experiments, not predetermined requirements. Validate the actual live model configuration for each run.

## Decisions

- Pack documents together to fill context windows, with EOT at real document boundaries and mandatory causal attention isolation between documents. Packing shares storage and compute, not conditioning context.
- Retain overflow tokens by continuing into subsequent windows; do not select only documents that fit or truncate long documents.
- Use document-isolated SDPA as the correctness reference; enable FA4 only through a verified document-isolated path. The current dense FA4 call does not enforce document boundaries. Sparse/compressed attention and Engram are deferred research experiments.
- Add optional Muon for hidden weight matrices, retaining AdamW for other parameters and as a full-model baseline.
- Control both total training tokens and tokens per optimizer update when comparing runs.
- Implement resumable checkpoints before committing to long cloud runs.
- Keep architecture, optimizer, and data experiments separately identifiable.

## Concepts and conventions

### Gradient accumulation

A microbatch is one forward/backward pass that fits in GPU memory. An optimizer update can combine gradients from several microbatches before changing weights.

For full packed batches on one GPU:

    tokens_per_update = microbatch_size * context_length * accumulation_steps

Example: batch 4, context 1024, accumulation 8 gives 32,768 scored targets per update. Batch 2, context 2048, accumulation 8 gives the same token count. Context length still changes the learning problem; matching token counts does not make the runs identical.

Accumulation does not hold all microbatch activation graphs at once. Backpropagate each microbatch and release its graph. It is a memory tool, not an automatic speed improvement; a larger physical batch can be more efficient when it fits.

Normalize by the total number of non-ignored targets in the entire accumulation group. Dividing each microbatch mean by a fixed accumulation count is correct only when the valid token counts are equal and the group is complete. Handle padding, smaller last batches, and partial final accumulation groups explicitly. Accumulation should match a combined-batch gradient up to floating-point differences when stochastic layers are disabled; dropout can prevent exact equality in real runs.

### Gradient clipping

After accumulating and token-normalizing all gradients, compute their global L2 norm across all trainable parameters. If it exceeds a configured limit, scale the gradients together to cap that norm. This limits unusually large raw gradients; it does not directly bound the parameter update produced by AdamW or Muon.

Order of operations:

1. Zero gradients at the beginning of the accumulation group.
2. Run forward/backward for each microbatch.
3. Normalize accumulated gradients to the group's valid-token mean.
4. If using FP16 loss scaling, unscale gradients first.
5. Clip once across the complete model, including parameters in both optimizers.
6. Step both optimizers, then advance the schedule once per completed update.

Start with a configurable global norm limit of 1.0 as an experiment, not a guaranteed optimum. Log the pre-clipping norm and how often clipping occurs. Fail visibly on non-finite loss/gradients; clipping is not a repair for NaNs.

### Two token budgets

- Total scored training tokens per run: controls data exposure for comparisons. Repeat epochs count again; padding does not. Record distinct data volume separately from repeated exposure.
- Scored tokens per optimizer update: controls effective batch size and update frequency independently of physical batch size and context length.

Log microbatch count, optimizer-update count, actual cumulative tokens, and actual tokens per update separately. Schedule warmup, decay, evaluation, and checkpoints against documented token thresholds. Execute updates/checkpoints at update boundaries. Record any overshoot or final partial update policy rather than silently discarding tokens.

Equal token budgets are useful for learning comparisons; also compare time-to-quality or equal elapsed time for practical efficiency. Equal tokens do not imply equal compute across architectures.

## Implementation sequence

### Phase 1 — Freeze data, tokenizer, environment, and evaluation

- Decide the shared tokenizer before substantial multilingual training; evaluate a roughly 32K byte-level BPE candidate against the existing tokenizer on held-out text per language.
- Record vocabulary, special-token IDs, tokenizer artifact/hash, dataset identity/revision, and language mixture. Do not assume EOT ID remains unchanged after replacing the tokenizer.
- Seed and persist the document-level train/validation split. Tokenize/pack the two splits independently.
- Check document duplicates and train/validation overlap, including overlap between quality tiers.
- Use fixed validation examples and deterministic evaluation order. Do not consume the training shuffle generator during training-loss evaluation; use a separate loader.
- Record software versions, FA4 revision and applied patches, hardware, model config, run seed, and data order seed. Seeds alone do not guarantee identical kernels or dropout across backends.
- Track per-language validation loss/perplexity/accuracy and positional loss with observation counts. Tokenizer changes require care when comparing perplexity; consider bits per byte later.

Acceptance: repeated preprocessing produces the same membership, windows, and scored-token counts; evaluation does not perturb training data order. Preserve a baseline run and its initialization.

### Phase 2 — Pack training documents

Implemented so far: `decoder/packing.py` provides disk-backed stream packing and a reusable collator; `main.ipynb` and README usage are updated. Existing tests cover stream preservation, tokenization batches, real tokenization, padding, and small training runs. **Document attention isolation is now implemented:** packing schema v2 carries source document IDs; the collator masks cross-document labels and resets positions; all layers receive boundary metadata. SDPA combines document, causal, and padding masks; FA4 uses separate dense calls per visible document segment with the existing patched API. CPU tests cover independent logits/gradients, source EOT tokens, overlap, padding, zero-target batches, and reset-position metrics. Real FA4 cloud parity and throughput remain unverified; a CUDA-only test covers isolated forward/backward parity with SDPA. Regenerate legacy packed datasets/loaders.

- Shuffle training documents reproducibly, append EOT once at actual document endings, and concatenate into a token stream.
- Emit context_length + 1 tokens per window with stride context_length. Preserve overlap metadata so every eligible within-document next-token pair is scored once; cross-document transitions are deliberately excluded from loss.
- Continue long documents and fill spare window space from subsequent documents. Carry leftovers across preprocessing batches/shards instead of dropping them at every map boundary.
- Keep the final partial window if it has at least two tokens, using current padding/ignore rules. Define handling of an isolated final token explicitly.
- Carry a document/segment ID alongside every token through preprocessing, window overlap, collation, training, and evaluation. Derive boundaries from source documents, not token-value scans: an EOT-valued token in source text must not silently create a new document. Assign the appended EOT to the document it terminates.
- Enforce block-diagonal causal attention in every transformer layer: query i may attend to key j only when both are valid, j <= i, and their document IDs match. Combine this with padding validity. EOT alone is not an attention barrier. Isolate all documents, including documents in the same language; language mixing remains a data-sampling choice.
- Mask the shifted target with ignore_index=-100 whenever input and target belong to different documents. Keep last-content-token -> EOT supervision; exclude EOT -> next-document-first-token supervision. Without an explicit BOS, a document's first token has no within-document predecessor and is unscored. Adding BOS would be a separate tokenizer/objective decision.
- Supply explicit position IDs, starting at zero for each document segment visible in a window. A long document continued into a new window starts a new visible segment at position zero, retaining the current bounded learned-position table. Resetting positions does not replace attention masking, and EOT retains the preceding document's position sequence. Define positional metrics using these position IDs, not physical packing offsets.
- For SDPA, use an explicit combined document/causal/padding mask as the correctness reference. Ensure padded queries cannot cause non-finite outputs or gradients, and handle batches with zero scored targets without dividing by zero or taking a spurious optimizer step.
- For FA4, inspect the installed, patched API before selecting an implementation. Prefer a supported variable-length causal path with cumulative lengths for each visible document segment; otherwise use independent per-segment calls or an explicitly selected SDPA fallback. A dense causal call over a packed row is forbidden. Unsupported configurations must fail clearly rather than silently dropping isolation. Verify outputs and gradients against SDPA, then benchmark the actual cloud build before adopting it.
- Version the packed dataset schema/cache to include boundary metadata and record isolation, position, and loss policies in run/checkpoint config. Reject legacy packed rows lacking required metadata for isolated training. Update notebook, loader, evaluation, and capacity-probe plumbing together so no entry point silently uses stream semantics.
- Use fixed packing for validation. Decide and record whether training document order and packing are fixed or regenerated each epoch.

Acceptance:

- Reconstruct eligible within-document next-token pairs and match labels/counts exactly, including single-token documents, long documents, exact boundaries, final leftovers, and tokenization-batch boundaries. Test boundaries at the shared overlap token, EOT-valued source tokens, and padding. No training/validation crossing or artificial EOT at window boundaries.
- With dropout disabled, changing document A's tokens must leave document B's logits unchanged when they share a window. Compare each visible segment's outputs and parameter gradients against running that segment independently with the same positions and token-weighted objective. Test both same-language and mixed-language neighbors, multiple layers, and different packing offsets.
- Confirm attention cannot cross document boundaries, future tokens remain invisible within each document, position IDs reset as specified, and cross-document labels are ignored while EOT targets remain scored. Check finite forward/backward behavior with padding and zero-valid-target batches.
- Run CPU SDPA tests locally and real FA4 forward/backward parity tests on the preserved cloud environment; a mock kernel does not establish FA4 compatibility. Report packing utilization separately from scored-token utilization, since boundary targets are excluded. Use actual non-ignored targets for budgets and accumulation.

### Phase 3 — Training loop and resume infrastructure

- Add token-weighted gradient accumulation, global gradient clipping, token accounting, and configurable learning-rate warmup/decay.
- Keep physical batch size, accumulation count/target token budget, context length, and total-run token budget independently configurable.
- Remove duplicate cross-entropy work; collect positional statistics during evaluation rather than every training step.
- Replace fragile positional result indexing with named metrics/history fields, updating notebook plots explicitly.
- Save checkpoints on persistent storage with atomic replacement at completed optimizer-update boundaries.
- Include model state, both optimizer states where applicable, scheduler state, optional scaler state, configs, counters, metric history, tokenizer identity, data/split identity, RNG states, and sampler/packing cursor needed to recreate subsequent batches.
- Keep rolling latest checkpoints and a best-validation checkpoint; distinguish resumable training checkpoints from inference-only weight exports.
- Loading a resume checkpoint must restore the optimizer rather than recreate it inside every train() call. Loading weights alone remains a separate supported operation.

Acceptance: with stochastic behavior disabled, accumulated gradients match a combined batch for unequal valid-token counts and a partial group. Clipping happens once per update. Save/reload produces the same next batch, schedule position, counters, and comparable continuation; exact GPU equality depends on backend determinism. Handle non-finite steps explicitly.

### Phase 4 — Optional Muon + AdamW

- Prefer the supported implementation available in the recorded PyTorch environment; verify its API and update-scaling convention before choosing defaults.
- Muon group: 2D attention and feed-forward hidden weight matrices.
- AdamW group: token/position embeddings, output vocabulary head, biases, normalization scales/shifts, and remaining parameters.
- Assert disjoint, exhaustive parameter membership; handle shared/tied weights by parameter identity.
- Expose optimizer learning rates, weight decay policy, momentum/betas, and Muon scaling convention in run config. Avoid blindly transferring learning rates between conventions.
- Integrate both optimizers with accumulation, clipping, scheduling, checkpointing, and capacity tests.

Acceptance: all intended parameters update, optimizer state round-trips, and a short cloud run is finite. Compare tuned AdamW and Muon+AdamW under matched data exposure/effective token batch and also measure elapsed time.

### Phase 5 — Capacity and architecture experiments

- Make capacity probes use the actual optimizer, precision, clipping, accumulation, dropout, and loss configuration.
- Include complete forward/backward/update steps, first-use optimizer allocations, warmup, evaluation, and peak allocated/reserved GPU memory. Leave headroom and report non-OOM kernel failures separately.
- Prefer process-isolated trials where practical so model references, error tracebacks, JIT state, or allocator state do not contaminate successive trials.
- Stage 1 sweep: physical batch, accumulation, context 512/1024/2048 using a fixed architecture.
- Stage 2 sweep: a small depth/width comparison with compatible head dimensions; record parameter counts and compute/time rather than treating all configurations as equivalent.
- Evaluate finalists on fixed held-out data and repeat promising comparisons across seeds.
- Later, one change at a time: tied embeddings/output weights, positional encoding, normalization, feed-forward type, and possibly alternative attention or conditional memory.
- Sparse attention, token compression, Engram, and large MoE changes remain deferred until a measured bottleneck or research question justifies them.

Acceptance: find configurations that fit with headroom, then separately establish which learn best under the chosen budget. A successful capacity trial is not a quality result.

## Next implementation session

Document-isolated packing, labels, positions, SDPA, and the per-segment FA4 adapter are implemented end to end. Next run the CUDA-only FA4 parity tests and benchmark the isolated path in the preserved cloud environment. Capacity probes now exercise document isolation and report scored-token throughput; configure probe_document_length to represent expected packing density. Confirm the tokenizer/data baseline decision and establish resumable loop infrastructure before long runs. Muon, accumulation/clipping/scheduling, and architecture experiments remain planned.

## References

- Gradient clipping: https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
- Accumulation and mixed precision: https://docs.pytorch.org/docs/stable/notes/amp_examples.html
- Token-weighted accumulation: https://huggingface.co/docs/accelerate/main/usage_guides/gradient_accumulation
- Muon parameter selection and scaling: https://docs.pytorch.org/docs/main/generated/torch.optim.Muon.html
