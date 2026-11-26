"""Lookup tables built from decider-combinator banks.

A bank is: base value + sum over k of delta_k * [x >= th_k] (comparator ">=") or [x > th_k] (">").
The integer simulator, the QAT model and the circuit generator all use these exact tables.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import spec


@dataclass
class Bank:
    base: int
    th: list[int]
    delta: list[int]
    cmp: str  # ">=" or ">"

    def __call__(self, x):
        x = np.asarray(x, np.int64)
        out = np.full(x.shape, self.base, np.int64)
        for t, d in zip(self.th, self.delta):
            out += d * ((x >= t) if self.cmp == ">=" else (x > t))
        return out


def rsqrt_bank() -> Bank:
    """INV = round(2^16 / sqrt(ms)), piecewise constant on geometric buckets of ms (ratio 2^(1/4))."""
    ths, vals = [], []
    for k in range(0, 65):
        t = math.ceil(2 ** (k / 4))
        if ths and t <= ths[-1]:
            continue
        ths.append(t)
        mid = 2 ** ((k + 0.5) / 4)
        vals.append(round(2 ** 16 / math.sqrt(mid)))
    base = round(2 ** 16 / math.sqrt(0.5))   # ms == 0 (all-zero vector)
    deltas, prev = [], base
    for v in vals:
        deltas.append(v - prev)
        prev = v
    return Bank(base, ths, deltas, ">=")


def exp_bank() -> Bank:
    """e(shifted) for shifted = score - max + R (Q.8). e = E[b] on buckets of width EXP_STEP, 0 below range."""
    S, K, R = spec.EXP_STEP, spec.EXP_LEVELS, spec.EXP_R
    E = [round(spec.EXP_ONE * math.exp(-(k * S) / 256)) for k in range(K)] + [0]
    th = [R - S * (k + 1) for k in range(K)]
    delta = [E[k] - E[k + 1] for k in range(K)]
    # Drop zero deltas (deep tail): those deciders would do nothing.
    keep = [(t, d) for t, d in zip(th, delta) if d != 0]
    return Bank(0, [t for t, _ in keep], [d for _, d in keep], ">")


RSQRT = rsqrt_bank()
EXP = exp_bank()
