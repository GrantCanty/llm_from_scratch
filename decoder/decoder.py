import torch
from . import model
from .optimizers import build_optimizer
from pathlib import Path


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
        train() already creates a fresh optimizer for each call.
        """
        torch.manual_seed(self.seed if seed is None else seed)
        self.decoder_model = model.GPTModel(self.GPT_CONFIG).to(self.device)
    
    def generate(self, text, tokenizer, max_tokens = 50, temperature = 1.2, top_k = 35):
        return model.generate_and_print_sample(self.decoder_model, tokenizer, self.device, text, max_tokens, top_k, temperature)


    def train(self, train_loader, val_loader, num_epochs=10, eval_freq=5, eval_iter=1, drop_rate=0.1, learning_rate=0.0004,
              *, warmup_steps=0, warmup_tokens=0, update_norm_interval=100,
              optimizer_type='adamw', weight_decay=.1, adamw_betas=(.9, .999),
              muon_learning_rate=None, muon_weight_decay=.1, muon_momentum=.95,
              muon_nesterov=True, muon_adjust_lr_fn='match_rms_adamw', muon_ns_steps=5):
        """Train with optional linear warmup, then hold learning_rate constant.

        optimizer_type='muon' uses native Muon for hidden Linear matrices and
        AdamW elsewhere. Default match_rms_adamw scaling uses learning_rate when
        muon_learning_rate is omitted; other scaling requires an explicit LR.
        Select updates (warmup_steps) or scored targets (warmup_tokens).
        Each call creates a fresh optimizer and restarts the schedule/history.
        Per-update LR, token count, loss, gradient norm, scored tokens/second,
        elapsed seconds and CUDA peak allocated/reserved bytes are in
        training_history. Actual per-parameter update norms are sampled on update
        1 and every update_norm_interval updates (0 disables). Timing excludes loading/evaluation; memory is None
        on non-CUDA devices. The return tuple is unchanged.
        """
        self.decoder_model.drop_emb =  torch.nn.Dropout(drop_rate)

        for name, loader in (
            ("train_loader", train_loader),
            ("val_loader", val_loader),
        ):
            if not hasattr(loader, '__iter__'):
                raise TypeError(f"{name} must be an iterable")

        self.training_history = []
        optimizer = build_optimizer(
            self.decoder_model, optimizer_type=optimizer_type, learning_rate=learning_rate,
            weight_decay=weight_decay, adamw_betas=adamw_betas,
            muon_learning_rate=muon_learning_rate, muon_weight_decay=muon_weight_decay,
            muon_momentum=muon_momentum, muon_nesterov=muon_nesterov,
            muon_adjust_lr_fn=muon_adjust_lr_fn, muon_ns_steps=muon_ns_steps)
        return model.train_model_simple_function(self.decoder_model, train_loader, val_loader, optimizer, self.device, num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iter,
                                                 warmup_steps=warmup_steps, warmup_tokens=warmup_tokens,
                                                 update_history=self.training_history,
                                                 update_norm_interval=update_norm_interval)

    def save_weights(self, name=None):
        name = f"GPT_Model_embdim_{self.GPT_CONFIG['emb_dim']}_nlayers_{self.GPT_CONFIG['n_layers']}_nheads_{self.GPT_CONFIG['n_heads']}.pth" if name == None else safe_file_name(name)
        torch.save(self.decoder_model.state_dict(), name)

    def load_weights(self, path):
        self.decoder_model.load_state_dict(torch.load(path, weights_only=True))

def safe_file_name(name):
    if name is not None:
        name = str(Path(name).with_suffix(".pth"))
    return name
