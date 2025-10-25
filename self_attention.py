import torch

class SelfAttentionV1(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_out = d_out
        self.W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        self.W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        self.W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec
    

class SelfAttentionV2(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_out = d_out
        self.W_query = torch.nn.Linear(d_in, d_out, bias='qkv_bias')
        self.W_key = torch.nn.Linear(d_in, d_out, bias='qkv_bias')
        self.W_value = torch.nn.Linear(d_in, d_out, bias='qkv_bias')
    
    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(-2, -1)
        attn_weights = torch.softmax(attn_scores / attn_scores.shape[-1]**.5, dim=-1)
        context_vec = attn_weights @ values

        return context_vec
    

class CausalSelfAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout):
        super().__init__()
        torch.manual_seed(123)
        self.d_out = d_out
        self.W_query = torch.nn.Linear(d_in, d_out, bias=False)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=False)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=False)
        self.context_len = context_len
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(self.context_len, self.context_len), diagonal=0)
        )

    def forward(self, x):
        
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        masked = attn_scores.masked_fill_(~self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(masked / keys.shape[-1]**.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        print(attn_weights)
        context_vec = attn_weights @ values

        print(context_vec)
        return context_vec
