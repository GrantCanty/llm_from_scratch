import torch
import model
import config
import tiktoken
from gpt_download import download_and_load_gpt2


'''tokenizer = tiktoken.get_encoding('gpt2')
torch.manual_seed(123)
settings, params = download_and_load_gpt2(model_size='124M', models_dir='gpt2')

gpt_model = model.GPTModel(config.GPT_CONFIG_124M_INFER)
model.load_weights(gpt_model, params)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpt_model.to(device)
model.generate_and_print_sample(gpt_model, tokenizer, device, "Every step moves you", max_new_tokens=50, temperature=1.4, top_k=35)'''

settings, params = download_and_load_gpt2(model_size='124M', models_dir='gpt2')

class Decoder():
    def __init__(self, param_count: int = 0, layer_count: int = 0, transformer_count: int = 0, tokenizer = tiktoken.get_encoding('gpt2')):
        torch.manual_seed(123)

        self.decoder_model = model.GPTModel(config.GPT_CONFIG_124M_INFER)
        self.tokenizer = tokenizer
        
        
        model.load_weights(self.decoder_model, params)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decoder_model.to(self.device)
    
    def generate(self, text, max_tokens = 50, temperature = 1.2, top_k = 35):
        return model.generate_and_print_sample(self.decoder_model, self.tokenizer, self.device, text, max_tokens, top_k, temperature)
