import torch
from . import model
from pathlib import Path


class Decoder():
    def __init__(self, n_layers=12, emb_dim = 768, n_heads=12, context_length = 256, seed=123):
        self.seed = seed

        self.GPT_CONFIG = {
            "vocab_size": 50257,
            "context_length": context_length,
            "emb_dim": emb_dim,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "drop_rate": 0.1,
            "qkv_bias": True
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


    def train(self, train_loader, val_loader, num_epochs=10, eval_freq=5, eval_iter=1, drop_rate=0.1, learning_rate=0.0004):
        self.decoder_model.drop_emb =  torch.nn.Dropout(drop_rate)

        for name, loader in (
            ("train_loader", train_loader),
            ("val_loader", val_loader),
        ):
            if not hasattr(loader, '__iter__'):
                raise TypeError(f"{name} must be an iterable")

        optimizer = torch.optim.AdamW(self.decoder_model.parameters(), lr=learning_rate, weight_decay=0.1)
        return model.train_model_simple_function(self.decoder_model, train_loader, val_loader, optimizer, self.device, num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iter)

    def save_weights(self, name=None):
        name = f"GPT_Model_embdim_{self.GPT_CONFIG['emb_dim']}_nlayers_{self.GPT_CONFIG['n_layers']}_nheads_{self.GPT_CONFIG['n_heads']}.pth" if name == None else safe_file_name(name)
        torch.save(self.decoder_model.state_dict(), name)

    def load_weights(self, path):
        self.decoder_model.load_state_dict(torch.load(path, weights_only=True))

def safe_file_name(name):
    if name is not None:
        name = str(Path(name).with_suffix(".pth"))
    return name