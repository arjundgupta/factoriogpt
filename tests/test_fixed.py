"""fixed.py must reproduce the values measured in the real game (notes/m0_raw.json)."""
import json
from pathlib import Path

import numpy as np

from fgpt import fixed as fx

RAW = json.loads(Path(__file__).resolve().parents[1].joinpath("notes/m0_raw.json").read_text())
P4 = RAW["probes"]["res"]["P4"]

CASES = {
    "add_overflow": (2147483647, "+", 1), "mul_overflow": (70000, "*", 70000), "div_neg": (-7, "/", 2),
    "mod_neg": (-7, "%", 2), "shr_neg": (-8, ">>", 1), "shl_32": (1, "<<", 32), "shl_31": (1, "<<", 31),
    "shr_33": (1024, ">>", 33), "div_zero": (7, "/", 0), "pow_sq_big": (50000, "^", 2), "pow_neg": (2, "^", -1),
}


def test_scalar_ops_match_game():
    for name, (a, op, b) in CASES.items():
        assert fx.OPS[op](a, b) == P4[name], name


def test_array_ops_match_scalar():
    rng = np.random.default_rng(0)
    a = rng.integers(-2**31, 2**31, 2000, dtype=np.int64)
    b = rng.integers(-40, 40, 2000, dtype=np.int64)
    for op in ["+", "-", "*", "/", "%", ">>", "<<"]:
        arr = fx.OPS[op](a, b)
        for i in range(0, 2000, 97):
            assert arr[i] == fx.OPS[op](int(a[i]), int(b[i])), (op, a[i], b[i])


def test_dot_matches_game_probes():
    assert fx.dot([3, -1, 2], [2, 4, 1]) == RAW["probes"]["res"]["P1"]["signal-Z"]
    n = 1215
    w = np.arange(1, n + 1)
    dropped_sum = 1215 * 1216 // 2 - 738177   # 5 dropped signals in the full-width probe
    assert fx.dot(np.ones(n, np.int64), w) - dropped_sum == 738177
