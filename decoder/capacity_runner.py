from itertools import product
from decoder.capacity import profile_training_configs

candidates = [
    {"batch_size": batch, "context_length": context}
    for context, batch in product(
        [512, 1024, 2048],
        [2, 4, 8],
    )
]

def run(base_cfg, candidates, flash_attn_func, headroom_fraction=0.10, optimizer_options=None,
        accumulation_steps=1, adaptive_clip=False, clip_window=100,
        clip_multiplier=3., clip_min_history=20):
    result = profile_training_configs(
        base_cfg, candidates, flash_attn_func=flash_attn_func,
        headroom_fraction=headroom_fraction, optimizer_options=optimizer_options,
        accumulation_steps=accumulation_steps, adaptive_clip=adaptive_clip,
        clip_window=clip_window, clip_multiplier=clip_multiplier,
        clip_min_history=clip_min_history,
    )
    print(result)