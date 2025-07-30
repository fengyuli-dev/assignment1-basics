import os
from collections import Counter, defaultdict

import regex

PRETOKEN_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

PreTokenDictType = dict[tuple[bytes, ...], int]
BytesPairDictType = dict[tuple[bytes, bytes], int]
BytesPairOriginDictType = dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]


def pre_tokenize(text: str):
    pre_tokens: PreTokenDictType = Counter()
    for match in regex.finditer(PRETOKEN_REGEX, text):
        word_bytes = match.group(0).encode("utf-8")
        word_bytes = tuple(bytes([b]) for b in word_bytes)
        pre_tokens[word_bytes] += 1
    return pre_tokens


def train_tokenizer(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]):

    with open(input_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Pre-tokenize each splitted doc
    split_regex = "|".join(special_tokens)
    text_list = regex.split(split_regex, text)
    pre_token_dicts: list[PreTokenDictType] = [pre_tokenize(text) for text in text_list]
    pre_token_dict = sum((Counter(d) for d in pre_token_dicts), Counter())

    vocab = {}
    merges = []
    for i in range(256):
        vocab[i] = i.to_bytes(length=1)
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    while len(vocab) < vocab_size:
        bytes_pairs: BytesPairDictType = Counter()
        bytes_pair_origins: BytesPairOriginDictType = defaultdict(set)

        # Merge most frequent bytes pair
        for pre_token in pre_token_dict.keys():
            if len(pre_token) > 1:
                for i in range(len(pre_token) - 1):
                    bytes_pair = (pre_token[i], pre_token[i + 1])
                    bytes_pairs[bytes_pair] += pre_token_dict[pre_token]
                    bytes_pair_origins[bytes_pair].add(pre_token)
        pair_to_merge = max(bytes_pairs.keys(), key=lambda pair: (bytes_pairs[pair], pair))
        merges.append(pair_to_merge)
        new_token = pair_to_merge[0] + pair_to_merge[1]
        vocab[len(vocab)] = new_token

        # Update pre-tokens
        new_pre_tokens: PreTokenDictType = Counter()
        for pre_token, count in pre_token_dict.items():
            if pre_token not in bytes_pair_origins[pair_to_merge]:
                new_pre_tokens[pre_token] = count
                continue
            merged_list: list[bytes] = []
            i = 0
            while i < len(pre_token):
                if i < len(pre_token) - 1 and (pre_token[i], pre_token[i + 1]) == pair_to_merge:
                    merged_list.append(new_token)
                    i += 2
                else:
                    merged_list.append(pre_token[i])
                    i += 1
            new_pre_tokens[tuple(merged_list)] = count
        pre_token_dict = new_pre_tokens

    return vocab, merges
