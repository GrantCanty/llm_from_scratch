GPT_CONFIG_124M_INFER = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,      # Context length
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True
}

GPT_CONFIG_124M_TRAIN = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 256,      # Context length
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True
}