"""Standalone CUDA/MPS smoke benchmark using synthetic tokens; no downloads.

This deliberately uses a small PyTorch transformer, not the project model or FA4.
It tests concurrent forward/backward/AdamW steps, not language-model quality.
"""
import argparse
import json
import math
import os
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--name', default='trial')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seq-len', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--warmup', type=int, default=20,
                        help='Untimed training steps before measurement')
    parser.add_argument('--log-every', type=int, default=50)
    args = parser.parse_args()
    if min(args.batch_size, args.seq_len, args.steps, args.log_every) < 1:
        parser.error('Batch size, sequence length, steps and log interval must be positive')
    if args.warmup < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('Warmup must be nonnegative and LR must be finite and positive')

    # Lazy import keeps --help usable without PyTorch/CUDA installed.
    import torch
    from torch import nn
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        parser.error('This benchmark requires CUDA; it will not run on CPU')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    device = torch.device('cuda:0')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=dtype == torch.float16)
    vocab_size, width = 4096, 512

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, width)
            self.blocks = nn.ModuleList([
                nn.TransformerEncoderLayer(width, 8, 4 * width, dropout=0,
                                           batch_first=True, norm_first=True)
                for _ in range(4)
            ])
            self.norm = nn.LayerNorm(width)
            self.head = nn.Linear(width, vocab_size, bias=False)

        def forward(self, tokens, mask):
            x = self.embedding(tokens)
            for block in self.blocks:
                x = block(x, src_mask=mask, is_causal=True)
            return self.head(self.norm(x))

    model = TinyLM().to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # One fixed GPU-resident batch removes data loading and disk I/O from this test.
    tokens = torch.randint(vocab_size, (args.batch_size, args.seq_len + 1), device=device)
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    mask = torch.ones(args.seq_len, args.seq_len, device=device, dtype=torch.bool).triu(1)
    print(json.dumps(dict(name=args.name, pid=os.getpid(), gpu=torch.cuda.get_device_name(),
                          dtype=str(dtype), parameters=sum(p.numel() for p in model.parameters()),
                          mps_pipe=os.environ.get('CUDA_MPS_PIPE_DIRECTORY'),
                          mps_memory_limit=os.environ.get('CUDA_MPS_PINNED_DEVICE_MEM_LIMIT'))),
          flush=True)

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=dtype):
            logits = model(inputs, mask)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss.detach()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for index in range(1, args.steps + 1):
        loss = step()
        if index % args.log_every == 0 or index == args.steps:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                raise RuntimeError('Nonfinite loss during training')
            print(f'{args.name}: step={index} loss={loss_value:.4f} '
                  f'tokens/s={index * args.batch_size * args.seq_len / elapsed:,.0f}', flush=True)
    elapsed = time.perf_counter() - started
    print(json.dumps(dict(name=args.name, status='completed', steps=args.steps,
                          seconds=elapsed, final_loss=loss_value,
                          tokens_per_second=args.steps * args.batch_size * args.seq_len / elapsed,
                          peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)), flush=True)


if __name__ == '__main__':
    main()
