"""Model configs, vocabulary, and the mapping from model quantities to Factorio signals."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- vocabulary
USER, BOT = "<", ">"   # turn markers (single characters in the text stream)
CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .?!") + [USER, BOT]
VOCAB = {c: i for i, c in enumerate(CHARS)}
V = len(CHARS)  # 42

# Each token is a real Factorio signal, so the keyboard and screen show letters directly.
_SPECIAL = {" ": "shape-horizontal", ".": "signal-letter-dot", "?": "signal-question-mark",
            "!": "signal-exclamation-mark", USER: "signal-input", BOT: "signal-output"}
VOCAB_SIGNALS = [("virtual", _SPECIAL.get(c, f"signal-{c}")) for c in CHARS]

# Control signals live on control wires only; they never appear in data vectors.
CONTROL = {k: ("virtual", v) for k, v in {
    "TICK": "signal-T", "T0": "signal-clock", "PHASE": "signal-hourglass", "POS": "signal-P",
    "POS1": "signal-plus", "RUN": "signal-check", "RESET": "signal-deny", "DONE": "signal-white-flag",
    "N": "signal-N", "W": "signal-W", "S2": "signal-S", "MS": "signal-M", "INV": "signal-I",
    "MAX": "signal-X", "Z": "signal-Z", "ONE": "signal-1",
    # tokenizer builds: character cursor, token length, letter code, target screen cell
    "CUR": "signal-C", "LEN": "signal-L", "CODE": "signal-K", "TGT": "signal-G",
}.items()}

_ALL = [tuple(s) for s in json.loads((ROOT / "fgpt" / "signals.json").read_text())]
_RESERVED = set(VOCAB_SIGNALS) | set(CONTROL.values())
FREE = [s for s in _ALL if s not in _RESERVED]          # 1,210 minus reserved


def dim_signals(n: int):
    """Signals carrying vector dimensions 0..n-1 (residual, q/k/v, MLP hidden)."""
    assert n <= len(FREE), n
    return FREE[:n]


def pos_signals(t: int):
    """Signals carrying positions 0..t-1 (attention scores, position one-hot). Disjoint from the vocab."""
    return FREE[:t]


TOKEN_BASE = 256   # merged BPE tokens use FREE[256:], so they never collide with position signals (T <= 256)


def token_signals(v: int):
    """Signals for token ids 0..v-1: the 42 characters keep their letter signals, merged tokens follow."""
    assert v - V <= len(FREE) - TOKEN_BASE, v
    return VOCAB_SIGNALS + FREE[TOKEN_BASE:TOKEN_BASE + v - V]


def bus_signals(n: int):
    """Signals addressing screen cells 0..n-1 on the detokenizer bus."""
    assert n <= len(FREE), n
    return FREE[:n]


@dataclass
class Config:
    name: str = "v0"
    vocab: int = V
    T: int = 64          # context length
    d: int = 32          # model width
    L: int = 1           # layers
    h: int = 2           # heads
    ff: int = 128        # MLP width
    tok: int = 0         # BPE vocabulary size (0 = one character per token)
    screen: int = 0      # screen cells for tokenizer builds (letters shown)

    @property
    def dh(self):
        return self.d // self.h

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @staticmethod
    def load(path):
        return Config(**json.loads(Path(path).read_text()))


CONFIGS = {
    "v0": Config("v0", T=64, d=32, L=1, h=2, ff=128),
    "v1lite": Config("v1lite", T=64, d=64, L=2, h=4, ff=256),
    "v1": Config("v1", T=128, d=128, L=4, h=4, ff=512),
    "v3": Config("v3", T=256, d=288, L=8, h=8, ff=1152),
    "train_tiny": Config("train_tiny", T=16, d=16, L=1, h=2, ff=64),
    # tokenizer builds: replies use 768 BPE tokens (about 3 letters each), prompts stay one letter per token
    "v0t": Config("v0t", vocab=768, T=64, d=32, L=1, h=2, ff=128, tok=768, screen=192),
    "v1t": Config("v1t", vocab=768, T=128, d=128, L=4, h=4, ff=512, tok=768, screen=384),
    "v3t": Config("v3t", vocab=768, T=256, d=288, L=8, h=8, ff=1152, tok=768, screen=768),
}


def tokenizer(cfg: Config):
    """The BPE tokenizer for a config (None for character-level builds)."""
    if not cfg.tok:
        return None
    from .tokenizer import Tokenizer
    return Tokenizer.load(ROOT / "fgpt" / f"tokens_{cfg.tok}.json")

# ---------------------------------------------------------------- fixed point
ACT = 8            # activations are Q.8: 1.0 == 256
NORM_PRE = 4       # x >> 4 before squaring in RMSNorm
INV_SHIFT = 12     # n = (x * INV) >> 12
QK_SHIFT = 8       # scores = (q . k) >> 8
EXP_ONE = 1024     # exp LUT output for a score equal to the max
EXP_STEP = 32      # LUT bucket width in Q.8 units (0.125)
EXP_LEVELS = 64    # buckets: covers scores down to max - 8.0
EXP_R = EXP_STEP * EXP_LEVELS
SCORE_BIG = 1 << 28   # added to raw q.k so no valid score is ever exactly 0 (absent)
LOGIT_BIG = 1 << 24   # same for logits before argmax
TIE_SHIFT = 10        # tokenizer builds: argmax over (logit << 10) + LOGIT_BIG_T + (V-1-id), so ties go to the lowest id
LOGIT_BIG_T = 1 << 29


def encode(text: str) -> list[int]:
    return [VOCAB[c] for c in text]


def decode(ids) -> str:
    return "".join(CHARS[i] for i in ids)
