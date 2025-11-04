import torch
import model
import self_attention


class TransformerBlock(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = self_attention.MultiHeadedAttention(
            d_in = cfg['emb_dim'],
            d_out = cfg['emb_dim'],
            context_len=cfg['context_length'],
            num_heads=cfg['n_heads'],
            dropout=cfg['drop_rate'],
            qkv_bias=cfg['qkv_bias'])
        self.ff = model.FeedForward(cfg)
        self.norm1 = model.LayerNorm(cfg['emb_dim'])
        self.norm2 = model.LayerNorm(cfg['emb_dim'])
        self.drop_shortcut = torch.nn.Dropout(cfg['drop_rate'])
    
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

        
