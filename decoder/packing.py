"""Pack documents with explicit boundaries for isolated causal attention."""
from dataclasses import dataclass
from numbers import Integral

import torch


PACKING_SCHEMA_VERSION = 2
from torch.nn.utils.rnn import pad_sequence


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def iter_packed_windows(documents, context_length, eot_id):
    """Yield input_ids windows from unshifted, non-EOT-terminated token lists.

    Empty documents are skipped. Nonempty documents each receive one appended
    EOT, even when a window ends inside a document. Adjacent windows share one
    token and its source document ID. The collator excludes cross-document
    transitions from supervision.
    Only the final window can be short. A lone shared carry token has no new
    target and is not emitted. Documents are consumed in their supplied order.
    """
    _positive_integer(context_length, "context_length")
    window_size = context_length + 1
    buffer = []
    segments = []
    for document_id, document in enumerate(documents):
        if len(document) == 0:
            continue
        tokens = list(document) + [eot_id]
        offset = 0
        while offset < len(tokens):
            take = min(window_size - len(buffer), len(tokens) - offset)
            buffer.extend(tokens[offset:offset + take])
            segments.extend([document_id] * take)
            offset += take
            if len(buffer) == window_size:
                yield {"input_ids": buffer, "document_ids": segments}
                buffer = buffer[-1:]
                segments = segments[-1:]
    if len(buffer) >= 2:
        yield {"input_ids": buffer, "document_ids": segments}


def _generate_packed_examples(dataset, tokenizer, context_length, eot_id,
                              text_column, tokenization_batch_size, packing_schema_version):
    if packing_schema_version != PACKING_SCHEMA_VERSION:
        raise ValueError("Unsupported packing schema")
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
    Document IDs are required by the collator to enforce attention isolation.
    """
    from datasets import Dataset, Features, Sequence, Value

    _positive_integer(context_length, "context_length")
    _positive_integer(tokenization_batch_size, "tokenization_batch_size")
    if text_column not in dataset.column_names:
        raise ValueError(f"Dataset has no text column {text_column!r}")
    if eot_id is None:
        eot_id = tokenizer.eot_token
    features = Features({name: Sequence(Value("int64"))
                         for name in ("input_ids", "document_ids")})
    if len(dataset) == 0:
        return Dataset.from_dict({name: [] for name in features}, features=features)
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
                "packing_schema_version": PACKING_SCHEMA_VERSION,
            },
        )
    except ValueError as exc:
        # Arrow's reader rejects a generated split with zero rows, even with
        # explicit features (e.g. a split consisting entirely of empty text).
        if 'corresponds to no data!' not in str(exc):
            raise
        return Dataset.from_dict({name: [] for name in features}, features=features)


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
        if any("document_ids" not in item for item in batch):
            raise ValueError("Packed rows require document_ids; regenerate legacy packed datasets")
        docs = [torch.tensor(item["document_ids"], dtype=torch.long) for item in batch]
        if any(d.shape != seq.shape or (d < 0).any() or (d[1:] < d[:-1]).any()
               for d, seq in zip(docs, seqs)):
            raise ValueError("document_ids must match tokens and be nonnegative and nondecreasing")
        inputs = [seq[:-1] for seq in seqs]
        targets = [seq[1:] for seq in seqs]
        input_ids = pad_sequence(inputs, batch_first=True, padding_value=self.pad_token_id)
        targets = [t.masked_fill(d[:-1] != d[1:], -100) for t, d in zip(targets, docs)]
        labels = pad_sequence(targets, batch_first=True, padding_value=-100)
        document_ids = pad_sequence([d[:-1] for d in docs], batch_first=True, padding_value=-1)
        positions = []
        for d in docs:
            pos = torch.arange(len(d) - 1)
            starts = torch.cat((torch.tensor([True]), d[1:-1] != d[:-2]))
            positions.append(pos - torch.where(starts, pos, 0).cummax(0).values)
        position_ids = pad_sequence(positions, batch_first=True, padding_value=0)
        lengths = torch.tensor([len(seq) for seq in inputs])
        attention_mask = torch.arange(input_ids.size(1))[None, :] < lengths[:, None]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
                "document_ids": document_ids, "position_ids": position_ids}
