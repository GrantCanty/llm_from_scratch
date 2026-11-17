# Project Details  
This project lets you create an LLM based off of the GPT2 architecture. This is a decoder only LLM. This package prioritizes ease of use in creating a custom decoder.  

## Functionality  
The project lets you create a decoder with the ability to change the following parameters:  
* Number of layers  
* Number of heads  
* Embedding dimensions  

The project has the following functions:  
* `train(epoch_count, optional: eval_freq, optional: eval_iter, optional: start_context, optional: drop_rate, optional: content_length)`  
* `generate(text, optional: max_tokens, optional: temperature, optional: top_k)`  
* `save_weights`  
* `load_weights(path)`  

Create an instance of the package like this:  

```
import decoder

model = decoder.Decoder()
model.train(5, drop_rate=.25)
model.generate('the black cat ate')
```

## Context packing

Split original documents into training/validation before packing. The packer
concatenates tokenized documents with EOT separators and emits windows of
`context_length + 1` tokens. The collator shifts these into at most
`context_length` inputs and targets. Long documents continue across windows;
short documents fill spare space. Only the final window may need padding.

```python
import torch
from torch.utils.data import DataLoader
from decoder.packing import pack_dataset, CausalLMCollator

context_length = model.GPT_CONFIG["context_length"]
splits = full_data.train_test_split(train_size=0.8, seed=123)

# Use text_column="content" for the original Ultra-FineWeb notebook dataset.
# Use text_column="text" for HuggingFaceFW/fineweb.
tokenized_train = pack_dataset(
    splits["train"].shuffle(seed=123), tokenizer, context_length,
    text_column="text",
)
tokenized_val = pack_dataset(
    splits["test"], tokenizer, context_length, text_column="text",
)
collator = CausalLMCollator(context_length, tokenizer.eot_token)
train_loader = DataLoader(
    tokenized_train, batch_size=8, shuffle=True, collate_fn=collator,
    generator=torch.Generator().manual_seed(456),
)
val_loader = DataLoader(
    tokenized_val, batch_size=8, shuffle=False, collate_fn=collator,
)
```

If importing via a parent package, use
`from llm_from_scratch.decoder.packing import ...` instead. No changes to
`decoder/__init__.py` are necessary.

The `tokenized_train`/`tokenized_val` rows are packed windows, not original
documents; rebuild them and both loaders after changing context length. Packing
uses the provided document order and is fixed across epochs; loader shuffling
changes window order, not which documents share a window. Seeded document
shuffling before packing controls that grouping.

Tokenization is batched, but the packing buffer spans every tokenization batch.
Empty documents are skipped. Windows share a boundary token so each eligible
within-document pair is scored once. A single final shared token has no remaining
target and is not emitted again.

Packed rows carry source `document_ids`, including through window overlap. The
collator requires this metadata: regenerate legacy packed datasets and loaders.
It returns `document_ids` and reset `position_ids` alongside inputs, labels, and
padding masks. Training and evaluation pass these through every transformer layer.
Attention is causal and restricted to the same document. Appended EOT belongs to
the document it ends; source tokens with the same token ID do not create boundaries.
Last-token-to-EOT targets are scored; EOT-to-next-document targets are ignored
(`-100`). Without BOS, the first token of each document is unscored. Positions
restart at zero for each visible document segment, including long-document
continuations in a new window. Positional metrics use these reset positions.

SDPA combines document, causal, and padding masks. FA4 calls the existing dense
kernel separately for each visible document segment; this preserves isolation but
can add launch overhead for many short documents. Real cloud kernel compatibility
and performance must be verified on the installed patched build. Ordinary calls
without document metadata represent a single sequence, as used in generation;
do not strip metadata from packed training batches.

`pack_dataset` takes an already materialized Hugging Face `Dataset` and a
tiktoken-compatible tokenizer. It writes packed examples to the Hugging Face
cache rather than building the full token stream in RAM. Use `cache_dir=...` to
choose a writable or persistent location. Reuse the resulting datasets or save
them with `save_to_disk()` to preserve the exact packed baseline.
