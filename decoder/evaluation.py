"""Likelihood metrics for raw held-out text."""

import math

import torch
from torch.nn import functional as F


@torch.inference_mode()
def evaluate_bpb(model, texts, tokenizer):
    """Return corpus bits per UTF-8 byte on the tokenizer's canonical encoding.

    Use EOT as a start token and score only text targets. Long documents use
    overlapping windows with a stride of half the context length.
    """
    context_length = model.pos_emb.num_embeddings
    device = model.pos_emb.weight.device
    stride = max(1, context_length // 2)
    encode = getattr(tokenizer, "encode_ordinary", tokenizer.encode)
    total_bytes = 0
    total_nll = 0.0
    was_training = model.training
    model.eval()
    try:
        for text in texts:
            if not text:
                continue
            total_bytes += len(text.encode("utf-8"))
            tokens = encode(text)
            sequence = [tokenizer.eot_token, *tokens]
            scored = 0
            while scored < len(tokens):
                end = min(len(tokens), context_length if scored == 0 else scored + stride)
                start = max(0, end - context_length)
                inputs = torch.tensor(sequence[start:end], device=device).unsqueeze(0)
                logits = model(inputs)[0, scored - start:]
                targets = torch.tensor(tokens[scored:end], device=device)
                total_nll += F.cross_entropy(logits.float(), targets, reduction="sum").item()
                scored = end
    finally:
        model.train(was_training)
    if total_bytes == 0:
        raise ValueError("BPB evaluation requires at least one UTF-8 byte of text")
    return total_nll / (math.log(2) * total_bytes)
