import torch
from . import model
from .optimizers import build_optimizer
from .sampling import resumable_sampler
from .memory import release_optimizer, memory_report, collect_device_cache
from pathlib import Path
from .checkpoint import (cpu_snapshot, atomic_save, rng_state, restore_rng_state,
                         restore_loader_generators, validate_resume_loader, loader_generator_state,
                         validate_optimizer_snapshot)


class Decoder():
    def __init__(self, n_layers=12, emb_dim = 768, n_heads=12, context_length = 256, seed=123,
                 attention_backend="sdpa", flash_attn_func=None):
        self.seed = seed

        self.GPT_CONFIG = {
            "vocab_size": 50257,
            "context_length": context_length,
            "emb_dim": emb_dim,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "drop_rate": 0.1,
            "qkv_bias": True,
            "attention_backend": attention_backend,
            "flash_attn_func": flash_attn_func
        }

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
        self.reset_weights()

    def reset_weights(self, seed=None):
        """Rebuild the model from scratch, using the constructor seed by default.

        This replaces decoder_model; external optimizers must be recreated.
        train() already creates a fresh optimizer for each call. Old owned
        parameters are released before allocating the replacement; external
        references can still retain the previous model.
        """
        self._discard_training_state()
        # Release our ownership before constructing/moving the replacement.
        # External model/parameter/optimizer references can still retain it.
        if getattr(self, 'decoder_model', None) is not None:
            self.decoder_model.zero_grad(set_to_none=True)
            self.decoder_model = None
            collect_device_cache(self.device, empty_cache=False)
        torch.manual_seed(self.seed if seed is None else seed)
        self.decoder_model = model.GPTModel(self.GPT_CONFIG).to(self.device)
    
    def generate(self, text, tokenizer, max_tokens = 50, temperature = 1.2, top_k = 35):
        return model.generate_and_print_sample(self.decoder_model, tokenizer, self.device, text, max_tokens, top_k, temperature)


    def train(self, train_loader, val_loader, num_epochs=10, eval_freq=5, eval_iter=1, drop_rate=0.1, learning_rate=0.0004,
              *, warmup_steps=0, warmup_tokens=0, update_norm_interval=100,
              optimizer_type='adamw', weight_decay=.1, adamw_betas=(.9, .999),
              muon_learning_rate=None, muon_weight_decay=.1, muon_momentum=.95,
              muon_nesterov=True, muon_adjust_lr_fn='match_rms_adamw', muon_ns_steps=5,
              accumulation_steps=1, adaptive_clip=False, clip_window=100,
              clip_multiplier=3., clip_min_history=20, max_updates=None):
        """Train with optional linear warmup, then hold learning_rate constant.

        optimizer_type='muon' uses native Muon for hidden Linear matrices and
        AdamW elsewhere. Default match_rms_adamw scaling uses learning_rate when
        muon_learning_rate is omitted; other scaling requires an explicit LR.
        accumulation_steps groups nonempty input batches; token-weighted
        backward runs separately and epoch tails are flushed. adaptive_clip
        uses clip_multiplier times the previous clip_window raw norms after
        clip_min_history updates. Both controls default to baseline behavior.
        Select updates (warmup_steps) or scored targets (warmup_tokens).
        A fresh call requires no retained optimizer (explicit cleanup/reset).
        On return/interruption, only gradients/temporaries are freed; optimizer,
        warmup/clipping progress, and history remain for save_checkpoint().
        training_memory records current bytes before and after training.
        Per-update LR, token count, loss, gradient norm, scored tokens/second,
        elapsed seconds and CUDA peak allocated/reserved bytes are in
        training_history. Actual per-parameter update norms are sampled on update
        1 and every update_norm_interval updates (0 disables). Timing excludes loading/evaluation; memory is None
        on non-CUDA devices. The return tuple is unchanged.
        """
        if getattr(self, 'optimizer', None) is not None:
            raise RuntimeError('Training state is retained. Save a checkpoint, then call '
                               'cleanup_memory() or reset_weights() before a fresh train(); '
                               'use resume(None, train_loader, val_loader, ...) to continue.')
        self.decoder_model.drop_emb = torch.nn.Dropout(drop_rate)

        for name, loader in (
            ("train_loader", train_loader),
            ("val_loader", val_loader),
        ):
            if not hasattr(loader, '__iter__'):
                raise TypeError(f"{name} must be an iterable")

        optimizer = build_optimizer(
            self.decoder_model, optimizer_type=optimizer_type, learning_rate=learning_rate,
            weight_decay=weight_decay, adamw_betas=adamw_betas,
            muon_learning_rate=muon_learning_rate, muon_weight_decay=muon_weight_decay,
            muon_momentum=muon_momentum, muon_nesterov=muon_nesterov,
            muon_adjust_lr_fn=muon_adjust_lr_fn, muon_ns_steps=muon_ns_steps)
        self.optimizer = optimizer
        self.training_history = []
        self.training_state = {}
        self.training_results = None
        self.training_config = dict(
            num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iter,
            drop_rate=drop_rate, learning_rate=learning_rate, optimizer_type=optimizer_type,
            weight_decay=weight_decay, adamw_betas=adamw_betas,
            muon_learning_rate=muon_learning_rate, muon_weight_decay=muon_weight_decay,
            muon_momentum=muon_momentum, muon_nesterov=muon_nesterov,
            muon_adjust_lr_fn=muon_adjust_lr_fn, muon_ns_steps=muon_ns_steps,
            warmup_steps=warmup_steps, warmup_tokens=warmup_tokens,
            update_norm_interval=update_norm_interval, accumulation_steps=accumulation_steps,
            adaptive_clip=adaptive_clip, clip_window=clip_window,
            clip_multiplier=clip_multiplier, clip_min_history=clip_min_history)
        self.training_memory = {'before': self.memory_report()}
        try:
            self.training_results = model.train_model_simple_function(self.decoder_model, train_loader, val_loader, optimizer, self.device, num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iter,
                                                     warmup_steps=warmup_steps, warmup_tokens=warmup_tokens,
                                                     update_history=self.training_history,
                                                     update_norm_interval=update_norm_interval,
                                                     accumulation_steps=accumulation_steps,
                                                     adaptive_clip=adaptive_clip, clip_window=clip_window,
                                                     clip_multiplier=clip_multiplier, clip_min_history=clip_min_history,
                                                     training_state=self.training_state, max_updates=max_updates)
            return self.training_results
        except BaseException:
            self.training_state['status'] = 'interrupted_or_failed'
            raise
        finally:
            # Keep optimizer/training state available for an explicit checkpoint.
            self.decoder_model.zero_grad(set_to_none=True)
            collect_device_cache(self.device, empty_cache=False)
            self.training_memory['after'] = self.memory_report()
            self.training_state['continuation_rng'] = rng_state(self.device)
            self.training_results = self.training_state.get('evaluation_results', self.training_results)

    def resume(self, checkpoint_path, train_loader, val_loader, *, num_epochs=None,
               flash_attn_func=None, expected_metadata=None, max_updates=None):
        """Continue a full checkpoint, or retained state when checkpoint_path=None.

        num_epochs is the TOTAL target, including epochs already completed.
        max_updates optionally stops this call at a completed update boundary.
        Deterministic map-style loaders with num_workers=0 support standard
        samplers (prefix replay) or ResumableBatchSampler (direct committed
        position). Fixed batch lists/tuples also work. Preserve data identity.
        """
        if checkpoint_path is None:
            if getattr(self, 'optimizer', None) is None or not self.training_state:
                raise RuntimeError('No retained training state; provide a full checkpoint path')
            payload = dict(training_state=self.training_state, training_config=self.training_config,
                           training_history=self.training_history,
                           training_results=getattr(self, 'training_results', None),
                           metadata=getattr(self, 'checkpoint_metadata', {}))
        else:
            if getattr(self, 'optimizer', None) is not None:
                raise RuntimeError('Training state is retained; explicitly cleanup/reset before loading another checkpoint')
            payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            if payload.get('format') != 'llm_by_scratch.training_checkpoint' or payload.get('format_version') != 1:
                raise ValueError('resume() requires a supported full checkpoint, not weights only')
        state = cpu_snapshot(payload['training_state'])
        config = dict(payload['training_config'])
        if not state.get('checkpoint_safe'):
            raise ValueError('Cannot resume an incomplete optimizer update')
        from numbers import Integral
        target = config['num_epochs'] if num_epochs is None else num_epochs
        if isinstance(target, bool) or not isinstance(target, Integral) or target < 1:
            raise ValueError('num_epochs must be a positive TOTAL epoch target')
        next_epoch = state['epoch'] + int(state['epoch_complete'])
        if target <= next_epoch:
            raise ValueError('Epoch target is already complete; increase num_epochs to continue training')
        if max_updates is not None and (isinstance(max_updates, bool) or
                not isinstance(max_updates, Integral) or max_updates < 1):
            raise ValueError('max_updates must be a positive integer or None')
        specs = state.get('loader_specs', {})
        validate_resume_loader(train_loader, specs.get('train'))
        validate_resume_loader(val_loader, specs.get('val'))
        for loader, key in ((train_loader, 'sampler_state'), (val_loader, 'val_sampler_state')):
            saved_sampler = state.get(key)
            sampler = resumable_sampler(loader)
            if saved_sampler is not None:
                if sampler is None:
                    raise ValueError('Recreate the loader with ResumableBatchSampler')
                sampler.validate_state_dict(saved_sampler)
            elif sampler is not None and state['updates']:
                raise ValueError('Checkpoint has no committed sampler state for this loader')
        metadata = payload.get('metadata', {})
        if expected_metadata is not None and any(metadata.get(k) != v for k, v in expected_metadata.items()):
            raise ValueError('Checkpoint tokenizer/dataset metadata does not match expected_metadata')
        # Validate explicit generator presence before touching the model or RNG.
        for loader, generators in ((train_loader, state.get('loader_generators', {})),
                                   (val_loader, state.get('val_loader_generators', {}))):
            if set(loader_generator_state(loader)) != set(generators):
                raise ValueError('Recreate loader/sampler generators as in the checkpoint')
        if checkpoint_path is not None:
            saved_device = torch.device(payload.get('environment', {}).get('device', str(self.device)))
            if saved_device.type != self.device.type:
                raise ValueError('Resume currently requires the same device type as the saved run')
            cfg = dict(payload['model_config'])
            kernel = flash_attn_func or self.GPT_CONFIG.get('flash_attn_func')
            cfg['flash_attn_func'] = kernel if cfg.get('attention_backend', 'sdpa') == 'fa4' else None
            if cfg.get('attention_backend') == 'fa4' and not callable(kernel):
                raise ValueError('Supply the compatible flash_attn_func for this FA4 checkpoint')
            # CPU preflight leaves the current model untouched on bad files.
            before_rng = rng_state(self.device)
            try:
                restored_model = model.GPTModel(cfg)
                restored_model.drop_emb = torch.nn.Dropout(config['drop_rate'])
                restored_model.load_state_dict(payload['model_state_dict'], strict=True)
                options = {key: config[key] for key in (
                    'optimizer_type', 'learning_rate', 'weight_decay', 'adamw_betas',
                    'muon_learning_rate', 'muon_weight_decay', 'muon_momentum',
                    'muon_nesterov', 'muon_adjust_lr_fn', 'muon_ns_steps')}
                prototype = build_optimizer(restored_model, **options)
                validate_optimizer_snapshot(prototype, restored_model, payload)
            finally:
                restore_rng_state(before_rng, self.device)
            del prototype
            self.decoder_model = None
            collect_device_cache(self.device, empty_cache=False)
            self.GPT_CONFIG = cfg
            self.seed = payload['seed']
            self.decoder_model = restored_model.to(self.device)
            self.optimizer = build_optimizer(self.decoder_model, **options)
            self.optimizer.load_state_dict(payload['optimizer_state_dict'])
            self.training_history = cpu_snapshot(payload.get('training_history', []))
            self.training_results = cpu_snapshot(payload.get('training_results'))
            self.checkpoint_metadata = cpu_snapshot(metadata)
            self.training_state = state
            self.training_config = config
            if 'evaluation_results' not in state and payload.get('training_results') is not None:
                state['evaluation_results'] = cpu_snapshot(payload['training_results'])
            restore_rng_state(payload['rng_state'], self.device)
            del payload  # Do not retain a CPU copy of all weights/state throughout resumed training.
        elif 'continuation_rng' in state:
            restore_rng_state(state['continuation_rng'], self.device)
        for loader, key in ((train_loader, 'sampler_state'), (val_loader, 'val_sampler_state')):
            if state.get(key) is not None:
                resumable_sampler(loader).load_state_dict(state[key])
        restore_loader_generators(train_loader, state.get('loader_generators', {}))
        restore_loader_generators(val_loader, state.get('val_loader_generators', {}))
        config['num_epochs'] = target
        self.training_config = config
        self.training_memory = {'before': self.memory_report()}
        try:
            self.training_results = model.train_model_simple_function(
                self.decoder_model, train_loader, val_loader, self.optimizer, self.device,
                num_epochs=target, eval_freq=config['eval_freq'], eval_iter=config['eval_iter'],
                warmup_steps=config['warmup_steps'], warmup_tokens=config['warmup_tokens'],
                update_history=self.training_history, update_norm_interval=config['update_norm_interval'],
                accumulation_steps=config['accumulation_steps'], adaptive_clip=config['adaptive_clip'],
                clip_window=config['clip_window'], clip_multiplier=config['clip_multiplier'],
                clip_min_history=config['clip_min_history'], training_state=self.training_state,
                resume_state=state, max_updates=max_updates)
            return self.training_results
        except BaseException:
            self.training_state['status'] = 'interrupted_or_failed'
            raise
        finally:
            self.decoder_model.zero_grad(set_to_none=True)
            collect_device_cache(self.device, empty_cache=False)
            self.training_memory['after'] = self.memory_report()
            self.training_state['continuation_rng'] = rng_state(self.device)
            self.training_results = self.training_state.get('evaluation_results', self.training_results)

    def memory_report(self):
        """Current live/cache byte counts, plus this model's weights/gradients."""
        return memory_report(self.decoder_model, self.device)

    def cleanup_memory(self, *, empty_cache=True):
        """Explicitly discard optimizer/progress, gradients, and optional cache.

        Save a full checkpoint first if needed. Keeps current weights, history,
        and device intact. Cannot free tensors owned by other notebook variables.
        """
        before = self.memory_report()
        self._discard_training_state()
        self.decoder_model.zero_grad(set_to_none=True)
        collect_device_cache(self.device, empty_cache=empty_cache)
        return {'before': before, 'after': self.memory_report()}

    def _discard_training_state(self):
        optimizer = getattr(self, 'optimizer', None)
        if optimizer is not None:
            release_optimizer(optimizer)
        self.optimizer = None
        self.training_state = None
        self.training_config = None
        self.checkpoint_metadata = {}
        # CPU histories/results remain available for plotting.

    def save_checkpoint(self, name=None, *, checkpoint=True, metadata=None):
        """Atomically save full training state, or a plain weights-only state dict.

        Full snapshots require retained state at a completed update boundary.
        Saving never cleans up memory. Load and continue full files with resume().
        metadata can identify the tokenizer, dataset/split, and data-order recipe.
        """
        if not isinstance(checkpoint, bool):
            raise ValueError('checkpoint must be True or False')
        if name is None:
            suffix = 'checkpoint' if checkpoint else 'weights'
            name = (f"GPT_Model_embdim_{self.GPT_CONFIG['emb_dim']}_"
                    f"nlayers_{self.GPT_CONFIG['n_layers']}_nheads_{self.GPT_CONFIG['n_heads']}_{suffix}.pth")
        name = safe_file_name(name)
        if not checkpoint:
            return atomic_save(cpu_snapshot(self.decoder_model.state_dict()), name)
        if getattr(self, 'optimizer', None) is None or not getattr(self, 'training_state', None):
            raise RuntimeError('No retained training state. Train first and save before cleanup/reset, '
                               'or use checkpoint=False for weights only.')
        if not self.training_state.get('checkpoint_safe', False):
            raise RuntimeError('Training stopped outside a completed optimizer-update boundary; '
                               'a full checkpoint would not be safe to resume.')
        names = {id(p): n for n, p in self.decoder_model.named_parameters()}
        kernel = self.GPT_CONFIG.get('flash_attn_func')
        payload = dict(
            format='llm_by_scratch.training_checkpoint', format_version=1,
            model_state_dict=self.decoder_model.state_dict(),
            model_config={key: value for key, value in self.GPT_CONFIG.items()
                          if key != 'flash_attn_func'},
            attention_kernel=(dict(module=getattr(kernel, '__module__', None),
                                   name=getattr(kernel, '__qualname__', None)) if kernel else None),
            optimizer_state_dict=self.optimizer.state_dict(),
            optimizer_parameter_names=[[names[id(p)] for p in group['params']]
                                       for group in self.optimizer.param_groups],
            training_config=self.training_config, training_state=self.training_state,
            training_history=self.training_history, training_results=self.training_results,
            rng_state=rng_state(self.device), seed=self.seed,
            model_training=self.decoder_model.training,
            environment=dict(torch_version=str(torch.__version__), device=str(self.device)),
            metadata=(metadata if metadata is not None else getattr(self, 'checkpoint_metadata', {})),
            data_resume_note='Resume supports deterministic map-style loaders with num_workers=0 '
                             'and standard/resumable samplers or fixed batch lists. Preserve data identity; '
                             'streaming, worker prefetch, and custom sampler restoration are unsupported.',
        )
        path = atomic_save(cpu_snapshot(payload), name)
        if metadata is not None:
            self.checkpoint_metadata = cpu_snapshot(metadata)
        return path

    def load_weights(self, path):
        if getattr(self, 'optimizer', None) is not None:
            raise RuntimeError('Call cleanup_memory() or reset_weights() before replacing weights.')
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state.get('format') == 'llm_by_scratch.training_checkpoint':
            raise ValueError('This is a full checkpoint; use resume() to load and continue full checkpoints.')
        self.decoder_model.load_state_dict(state)

def safe_file_name(name):
    if name is not None:
        name = str(Path(name).with_suffix(".pth"))
    return name
