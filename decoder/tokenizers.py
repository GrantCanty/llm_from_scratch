"""Untrained byte tokenizers for the decoder's integer token interface."""

import re
from dataclasses import dataclass


_END_OF_TEXT = "<|endoftext|>"
_UPPERCASE_BYTES = re.compile(rb"[A-Z]")
_LOWERCASE = bytes.maketrans(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ", b"abcdefghijklmnopqrstuvwxyz")


@dataclass(frozen=True)
class ByteTokenizer:
    """Encode UTF-8 bytes, optionally factoring out ASCII uppercase.

    With `encoding="ascii"`, reject non-ASCII input and use a smaller
    vocabulary. UTF-8 preserves Unicode, including punctuation in English text.
    """

    uppercase_marker: bool = False
    encoding: str = "utf-8"

    def __post_init__(self):
        if self.encoding not in {"ascii", "utf-8"}:
            raise ValueError("encoding must be 'ascii' or 'utf-8'")

    @property
    def eot_token(self):
        return 128 if self.encoding == "ascii" else 256

    @property
    def n_vocab(self):
        return self.eot_token + 1 + self.uppercase_marker

    def _encode_text(self, text):
        data = text.encode(self.encoding)
        if self.uppercase_marker:
            marker = self.eot_token + 1
            tokens = []
            start = 0
            for match in _UPPERCASE_BYTES.finditer(data):
                index = match.start()
                tokens.extend(data[start:index])
                tokens.extend((marker, _LOWERCASE[data[index]]))
                start = index + 1
            tokens.extend(data[start:])
            return tokens
        return list(data)

    def encode(self, text, *, allowed_special=()):
        """Return integer IDs, recognizing EOT only when explicitly allowed."""
        if allowed_special != "all" and _END_OF_TEXT not in allowed_special:
            return self._encode_text(text)
        tokens = []
        for index, part in enumerate(text.split(_END_OF_TEXT)):
            if index:
                tokens.append(self.eot_token)
            tokens.extend(self._encode_text(part))
        return tokens

    def encode_batch(self, texts, *, allowed_special=()):
        return [self.encode(text, allowed_special=allowed_special) for text in texts]

    def decode(self, tokens, *, errors="replace"):
        """Decode IDs. Ignore an uppercase marker without a lowercase follower."""
        chunks = []
        data = bytearray()
        uppercase = False
        for token in tokens:
            if token == self.eot_token:
                chunks.extend((data.decode(self.encoding, errors=errors), _END_OF_TEXT))
                data.clear()
                uppercase = False
            elif self.uppercase_marker and token == self.eot_token + 1:
                uppercase = True
            elif 0 <= token < self.eot_token:
                data.append(token - 32 if uppercase and 97 <= token <= 122 else token)
                uppercase = False
            else:
                raise ValueError(f"Invalid token ID {token}")
        chunks.append(data.decode(self.encoding, errors=errors))
        return "".join(chunks)
