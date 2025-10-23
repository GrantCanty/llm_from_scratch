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