import torch
import model
import config
import tiktoken
import embeddings
from pathlib import Path


class Decoder():
    def __init__(self, n_layers=12, emb_dim = 768, n_heads=12, tokenizer = tiktoken.get_encoding('gpt2')):
        torch.manual_seed(123)

        self.GPT_CONFIG = {
            "vocab_size": 50257,
            "context_length": 1024,
            "emb_dim": emb_dim,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "drop_rate": 0.1,
            "qkv_bias": True
        }

        self.decoder_model = model.GPTModel(self.GPT_CONFIG)
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decoder_model.to(self.device)
    
    def generate(self, text, max_tokens = 50, temperature = 1.2, top_k = 35):
        return model.generate_and_print_sample(self.decoder_model, self.tokenizer, self.device, text, max_tokens, top_k, temperature)


    def train(self, num_epochs=10, eval_freq=5, eval_iter=1, start_context="Every effort moves you", drop_rate=0.1):
        self.decoder_model.drop_emb =  torch.nn.Dropout(drop_rate)
        file_path = Path('data', 'the-verdict.txt')
        with open(file_path, 'r', encoding='utf-8') as file:
            text_data = file.read()
        
        train_ratio = 0.9
        split_idx = int(train_ratio * len(text_data))
        train_data = text_data[:split_idx]
        val_data = text_data[split_idx:]

        train_loader = embeddings.create_dataloader_v1(
        train_data,
            batch_size=2,
            max_length=config.GPT_CONFIG_124M_TRAIN['context_length'],
            stride=config.GPT_CONFIG_124M_TRAIN['context_length'],
            drop_last=True,
            shuffle=True,
            num_workers=0
        )

        val_loader = embeddings.create_dataloader_v1(
            val_data,
            batch_size=2,
            max_length=config.GPT_CONFIG_124M_TRAIN['context_length'],
            stride=config.GPT_CONFIG_124M_TRAIN['context_length'],
            drop_last=False,
            shuffle=False,
            num_workers=0
        )

        optimizer = torch.optim.AdamW(self.decoder_model.parameters(), lr=0.0004, weight_decay=0.1)
        return model.train_model_simple_function(self.decoder_model, train_loader, val_loader, optimizer, self.device, num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iter, start_context=start_context, tokenizer=self.tokenizer)

    def save_weights(self):
        torch.save(self.decoder_model.state_dict(), f"GPT_Model_embdim_{self.GPT_CONFIG['emb_dim']}_nlayers_{self.GPT_CONFIG['n_layers']}_nheads_{self.GPT_CONFIG['n_heads']}.pth")

    def load_weights(self, path):
        self.decoder_model.load_state_dict(torch.load(path, weights_only=True))