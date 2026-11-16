"""Pack document token streams into shifted causal-language-model windows."""
from dataclasses import dataclass
from numbers import Integral

import torch
from torch.nn.utils.rnn import pad_sequence


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def iter_packed_windows(documents, context_length, eot_id):
    """Yield input_ids windows from unshifted, non-EOT-terminated token lists.

    Empty documents are skipped. Nonempty documents each receive one appended
    EOT, even when a window ends inside a document. Adjacent windows share one
    token, so every adjacent pair in the concatenated stream is scored once.
    Only the final window can be short. A lone shared carry token has no new
    target and is not emitted. Documents are consumed in their supplied order.
    """
    _positive_integer(context_length, "context_length")
    window_size = context_length + 1
    buffer = []
    for document in documents:
        if len(document) == 0:
            continue
        tokens = list(document) + [eot_id]
        offset = 0
        while offset < len(tokens):
            take = min(window_size - len(buffer), len(tokens) - offset)
            buffer.extend(tokens[offset:offset + take])
            offset += take
            if len(buffer) == window_size:
                yield {"input_ids": buffer}
                buffer = buffer[-1:]
    if len(buffer) >= 2:
        yield {"input_ids": buffer}


def _generate_packed_examples(dataset, tokenizer, context_length, eot_id,
                              text_column, tokenization_batch_size):
    def tokenized_documents():
        for batch in dataset.iter(batch_size=tokenization_batch_size):
            yield from tokenizer.encode_batch(batch[text_column], allowed_special="all")

    # One packer spans all tokenization batches. No per-map-batch tail is lost.
    yield from iter_packed_windows(tokenized_documents(), context_length, eot_id)


def pack_dataset(dataset, tokenizer, context_length, *, text_column="text",
                 eot_id=None, tokenization_batch_size=1000, cache_dir=None):
    """Tokenize and pack one already-separated Hugging Face Dataset split.

    Uses a tiktoken-compatible encode_batch API. Call independently for training
    and validation. Shuffle documents with dataset.shuffle(seed=...) BEFORE
    calling if desired; this function does not introduce random ordering. The
    resulting map-style Dataset is disk-backed and works with ordinary loaders.
    Packing is sequential across tokenization batches, not a stateful map().
    EOT does not impose a document-level attention barrier.
    """
    from datasets import Dataset, Features, Sequence, Value

    _positive_integer(context_length, "context_length")
    _positive_integer(tokenization_batch_size, "tokenization_batch_size")
    if text_column not in dataset.column_names:
        raise ValueError(f"Dataset has no text column {text_column!r}")
    if eot_id is None:
        eot_id = tokenizer.eot_token
    features = Features({"input_ids": Sequence(Value("int64"))})
    if len(dataset) == 0:
        return Dataset.from_dict({"input_ids": []}, features=features)
    try:
        return Dataset.from_generator(
            _generate_packed_examples,
            features=features,
            cache_dir=cache_dir,
            gen_kwargs={
                "dataset": dataset, "tokenizer": tokenizer,
                "context_length": int(context_length), "eot_id": int(eot_id),
                "text_column": text_column,
                "tokenization_batch_size": int(tokenization_batch_size),
            },
        )
    except ValueError as exc:
        # Arrow's reader rejects a generated split with zero rows, even with
        # explicit features (e.g. a split consisting entirely of empty text).
        if 'corresponds to no data!' not in str(exc):
            raise
        return Dataset.from_dict({"input_ids": []}, features=features)


@dataclass(frozen=True)
class CausalLMCollator:
    """Shift packed windows once and right-pad only the final partial examples."""
    context_length: int
    pad_token_id: int

    def __post_init__(self):
        _positive_integer(self.context_length, "context_length")

    def __call__(self, batch):
        seqs = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
        if not seqs or any(not 2 <= len(seq) <= self.context_length + 1 for seq in seqs):
            raise ValueError("Each window must contain 2 through context_length + 1 tokens")
        inputs = [seq[:-1] for seq in seqs]
        targets = [seq[1:] for seq in seqs]
        input_ids = pad_sequence(inputs, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(targets, batch_first=True, padding_value=-100)
        lengths = torch.tensor([len(seq) for seq in inputs])
        attention_mask = torch.arange(input_ids.size(1))[None, :] < lengths[:, None]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
