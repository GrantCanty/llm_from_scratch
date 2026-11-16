from itertools import product
from decoder.capacity import profile_training_configs

candidates = [
    {"batch_size": batch, "context_length": context}
    for context, batch in product(
        [512, 1024, 2048],
        [2, 4, 8],
    )
]

def run(base_cfg, candidates, flash_attn_func, headroom_fraction=0.10):
    result = profile_training_configs(
        base_cfg, candidates, flash_attn_func, headroom_fraction,
    )
    print(result)