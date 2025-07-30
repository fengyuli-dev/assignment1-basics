import pickle as pkl
from typing import Iterable, Iterator

import regex

from cs336_basics.train_tokenizer import PRETOKEN_REGEX


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=None):
        self.vocab = vocab
        self.token_id_dict = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab = pkl.load(vocab_filepath)
        merges = pkl.load(merges_filepath)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped = [regex.escape(tok) for tok in sorted_tokens]
            pattern = "(" + "|".join(escaped) + ")"
            segments = regex.split(pattern, text)
        else:
            segments = [text]

        token_ids = []
        for seg in segments:
            # Handle special tokens
            if self.special_tokens and seg in self.special_tokens:
                tok = seg.encode('utf-8')
                token_ids.append(self.token_id_dict[tok])
                continue
            # Pre-tokenize and apply merges iteratively
            for m in regex.finditer(PRETOKEN_REGEX, seg):
                pre_token = m.group(0).encode("utf-8")
                merged = [bytes([b]) for b in pre_token]
                for a, b in self.merges:
                    i = 0
                    out = []
                    while i < len(merged):
                        if i < len(merged) - 1 and merged[i] == a and merged[i + 1] == b:
                            out.append(a + b)
                            i += 2
                        else:
                            out.append(merged[i])
                            i += 1
                    merged = out
                for piece in merged:
                    token_ids.append(self.token_id_dict[piece])

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for id in self.encode(text):
                yield id

    def decode(self, ids: list[int]) -> str:
        tokens = []
        for id in ids:
            tokens.append(self.vocab[id])
        return b''.join(tokens).decode("utf-8", errors="replace")
