import embeddings
import torch
import model
import config
import tiktoken
import matplotlib.pyplot as plt


tokenizer = tiktoken.get_encoding('gpt2')

file_path = 'data/the-verdict.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text_data = file.read()

print(len(text_data))
print(len(tokenizer.encode(text_data)))

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

torch.manual_seed(123)
gpt_model = model.GPTModel(config.GPT_CONFIG_124M_TRAIN)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpt_model.to(device)
optimizer = torch.optim.AdamW(gpt_model.parameters(), lr=0.0004, weight_decay=0.1)
num_epochs = 10

model.train_model_simple_function(gpt_model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, eval_freq=5, eval_iter=1, start_context="Every effort moves you", tokenizer=tokenizer)
torch.save({'model_state_dict': gpt_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, 'model.pth')

