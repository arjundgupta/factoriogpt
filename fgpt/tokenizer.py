"""Byte-pair-encoding (BPE) tokenizer for FactorioGPT replies.

Token ids 0..41 are the 42 single characters of spec.CHARS (letters, digits, space, .?!, USER, BOT),
so a typed prompt is fed one character per token and the keyboard is unchanged. Ids 42.. are merged
tokens learned from the bot's replies, GPT-2 style: text is first split into word chunks
(" BELT", "!", ...), and merges never cross a chunk boundary. Every merged token is at most LMAX
characters long, which bounds how many letters the in-game detokenizer writes per step.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from . import spec

LMAX = 8
CHUNK = re.compile(r" ?[A-Z0-9]+| ?[.?!]+| +")


def chunks(text: str):
    return CHUNK.findall(text)


class Tokenizer:
    def __init__(self, merges: list[tuple[int, int]]):
        self.merges = [tuple(m) for m in merges]
        self.tokens = list(spec.CHARS)                     # id -> string
        for a, b in self.merges:
            self.tokens.append(self.tokens[a] + self.tokens[b])
        self.rank = {m: i for i, m in enumerate(self.merges)}
        self.V = len(self.tokens)
        self._cache: dict[str, list[int]] = {}

    # -------------------------------------------------------------- encode / decode
    def _encode_chunk(self, s: str) -> list[int]:
        if s in self._cache:
            return self._cache[s]
        ids = [spec.VOCAB[c] for c in s]
        while len(ids) > 1:
            best, bi = None, -1
            for i in range(len(ids) - 1):
                r = self.rank.get((ids[i], ids[i + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if best is None:
                break
            ids[bi:bi + 2] = [spec.V + best]
        self._cache[s] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        out = []
        for ch in chunks(text):
            out += self._encode_chunk(ch)
        assert self.decode(out) == text, text
        return out

    def decode(self, ids) -> str:
        return "".join(self.tokens[i] for i in ids)

    def encode_dialogue(self, turns) -> list[int]:
        """USER + prompt (one char per token) + BOT + reply (BPE) ... + USER."""
        ids = []
        for t in turns:
            ids += [spec.VOCAB[spec.USER]] + spec.encode(t["user"]) + [spec.VOCAB[spec.BOT]] + self.encode(t["bot"])
        return ids + [spec.VOCAB[spec.USER]]

    # -------------------------------------------------------------- persistence
    def save(self, path):
        Path(path).write_text(json.dumps({"lmax": LMAX, "merges": self.merges, "tokens": self.tokens}))

    @staticmethod
    def load(path) -> "Tokenizer":
        return Tokenizer(json.loads(Path(path).read_text())["merges"])


def train_bpe(texts, vocab: int, lmax: int = LMAX) -> Tokenizer:
    """Classic BPE on chunk frequencies: repeatedly merge the most frequent adjacent pair."""
    freq = Counter()
    for t in texts:
        freq.update(chunks(t))
    words = {w: [spec.VOCAB[c] for c in w] for w in freq}
    lens = list(spec.CHARS)
    lens = [len(s) for s in lens]
    merges = []
    while spec.V + len(merges) < vocab:
        pairs = Counter()
        for w, ids in words.items():
            f = freq[w]
            for a, b in zip(ids, ids[1:]):
                if lens[a] + lens[b] <= lmax:
                    pairs[(a, b)] += f
        if not pairs:
            break
        (a, b), _ = pairs.most_common(1)[0]
        new = spec.V + len(merges)
        merges.append((a, b))
        lens.append(lens[a] + lens[b])
        for w, ids in words.items():
            if len(ids) < 2:
                continue
            i, out = 0, []
            while i < len(ids):
                if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
                    out.append(new)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            words[w] = out
    return Tokenizer(merges)
