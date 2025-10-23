import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

test_str: str = "this is the string that i want to test on to gain a better understanding of how embedding works"

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_tokens = []
        self.target_tokens = []
    
        tokens = tokenizer.encode(txt)

        for i in range(0, len(tokens) - max_length, stride):
            input_chunk = tokens[i:i+max_length]
            target_chunk = tokens[i+1:i+1+max_length]

            self.input_tokens.append(torch.tensor(input_chunk))
            self.target_tokens.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_tokens)
    
    def __getitem__(self, idx):
        return self.input_tokens[idx], self.target_tokens[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0
    )

    return dataloader

def embed_text(txt: str, max_length=4, batch_size=8, output_dim=256, stride=1):
    vocab_size = 50257
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    dataloader = create_dataloader_v1(
        txt, batch_size=batch_size, max_length=max_length, stride=stride, shuffle=False)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    #print("Token IDs:\n", inputs)
    #print("Target IDs:\n", targets)
    #print("\nInputs shape:\n", inputs.shape)

    input_token_embeddings = token_embedding_layer(inputs)
    target_token_embeddings = token_embedding_layer(targets)

    #print(input_token_embeddings)

    context_length = max_length

    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))

    input_embeddings = pos_embeddings + input_token_embeddings
    target_embeddings = pos_embeddings + target_token_embeddings

    return input_embeddings, target_embeddings