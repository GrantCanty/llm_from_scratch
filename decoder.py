import torch
import model
import config
import tiktoken
from gpt_download import download_and_load_gpt2
import embeddings


'''tokenizer = tiktoken.get_encoding('gpt2')
torch.manual_seed(123)
settings, params = download_and_load_gpt2(model_size='124M', models_dir='gpt2')

gpt_model = model.GPTModel(config.GPT_CONFIG_124M_INFER)
model.load_weights(gpt_model, params)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpt_model.to(device)
model.generate_and_print_sample(gpt_model, tokenizer, device, "Every step moves you", max_new_tokens=50, temperature=1.4, top_k=35)'''

#settings, params = download_and_load_gpt2(model_size='124M', models_dir='gpt2')

class Decoder():
    def __init__(self, param_count: int = 0, layer_count: int = 0, transformer_count: int = 0, tokenizer = tiktoken.get_encoding('gpt2')):
        torch.manual_seed(123)

        self.decoder_model = model.GPTModel(config.GPT_CONFIG_124M_INFER)
        self.tokenizer = tokenizer
        
        
        #model.load_weights(self.decoder_model, params)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decoder_model.to(self.device)
    
    def generate(self, text, max_tokens = 50, temperature = 1.2, top_k = 35):
        return model.generate_and_print_sample(self.decoder_model, self.tokenizer, self.device, text, max_tokens, top_k, temperature)


    def train(self, num_epochs=10, eval_freq=5, eval_iter=1, start_context="Every effort moves you"):
        file_path = 'data/the-verdict.txt'
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
