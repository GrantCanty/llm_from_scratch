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
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(context_len, context_len), diagonal=0)
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

        context_vec = attn_weights @ values
        return context_vec


class MultiHeadedAttentionWrapper(torch.nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout, num_heads):
        super().__init__()
        self.heads = [CausalSelfAttention(d_in, d_out, context_len, dropout) for _ in range(num_heads)]
    
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout, num_heads, qkv_bias=False,
                 attention_backend="sdpa", flash_attn_func=None):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        if attention_backend not in ("sdpa", "fa4"):
            raise ValueError("attention_backend must be 'sdpa' or 'fa4'")
        if attention_backend == "fa4" and not callable(flash_attn_func):
            raise ValueError("Pass flash_attn_func=fa4.flash_attn_func for the fa4 backend")
        self.attention_backend = attention_backend
        self.flash_attn_func = flash_attn_func
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = torch.nn.Linear(d_out, d_out)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(context_len, context_len), diagonal=0)
        )
    
    def forward(self, x, attention_mask=None):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        if num_tokens > self.mask.size(0):
            raise ValueError("Input sequence exceeds the attention context length")
        valid_keys = None
        if attention_mask is not None:
            if attention_mask.shape != (b, num_tokens):
                raise ValueError("attention_mask must have shape [batch, tokens]")
            valid_keys = attention_mask.to(device=x.device, dtype=torch.bool)

        dropout_p = self.dropout.p if self.training else 0.0
        if self.attention_backend == "fa4":
            if queries.device.type != "cuda" or queries.dtype not in (torch.float16, torch.bfloat16):
                raise ValueError("FA4 requires CUDA and FP16/BF16 queries; use BF16 autocast")
            context_vec = self._fa4_attention(queries, keys, values, valid_keys, dropout_p)
        else:
            allowed = None
            if valid_keys is not None:
                causal = self.mask[:num_tokens, :num_tokens].bool()
                allowed = causal[None, None, :, :] & valid_keys[:, None, None, :]
            context_vec = torch.nn.functional.scaled_dot_product_attention(
                queries.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2),
                attn_mask=allowed,
                is_causal=valid_keys is None,
                dropout_p=dropout_p,
            ).transpose(1, 2)

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec

    def _fa4_attention(self, queries, keys, values, valid_keys, dropout_p):
        # Dense causal attention is equivalent at valid positions for right padding.
        # Padded query outputs are discarded through labels=-100 in the loss.
        if valid_keys is not None:
            right_padded = valid_keys[:, 0].all() & ~(~valid_keys[:, :-1] & valid_keys[:, 1:]).any()
            if not right_padded:
                raise ValueError("FA4 supports only nonempty, right-padded masks here")
        kwargs = {}
        if dropout_p > 0:
            # Some SM120 builds generate seeds with an overflowing high=2**63.
            # Supply a fresh representable seed, also reproducible via manual_seed.
            kwargs["dropout_seed"] = torch.randint(
                0, torch.iinfo(torch.int64).max, (1,),
                dtype=torch.int64, device="cpu",
            ).item()
        output, _ = self.flash_attn_func(
            queries.contiguous(), keys.contiguous(), values.contiguous(),
            causal=True, dropout_p=dropout_p, **kwargs,
        )
        return output
