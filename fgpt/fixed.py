"""Bit-exact copies of Factorio combinator integer arithmetic (measured on 2.1.24, notes/game_facts.md).

Every value on a wire is a signed 32-bit integer and every operation wraps mod 2^32.
These functions are the single source of truth for the integer simulator, the tick
simulator and the QAT fake-quant.
"""
from __future__ import annotations

import numpy as np

I32 = np.int32
M32 = 1 << 32


def wrap(x):
    """Wrap any integer (or int64 array) to signed int32."""
    if isinstance(x, np.ndarray):
        return x.astype(np.int64).astype(np.uint32).astype(np.int32)
    x = int(x) & 0xFFFFFFFF
    return x - M32 if x >= 1 << 31 else x


def add(a, b):
    return wrap(np.asarray(a, np.int64) + np.asarray(b, np.int64)) if _arr(a, b) else wrap(int(a) + int(b))


def sub(a, b):
    return wrap(np.asarray(a, np.int64) - np.asarray(b, np.int64)) if _arr(a, b) else wrap(int(a) - int(b))


def mul(a, b):
    if _arr(a, b):
        # The int64 product is exact up to 2^62; the wrap then keeps the low 32 bits, which matches the game.
        return wrap(np.asarray(a, np.int64) * np.asarray(b, np.int64))
    return wrap(int(a) * int(b))


def div(a, b):
    """Truncates toward zero; x / 0 = 0."""
    if _arr(a, b):
        a64, b64 = np.asarray(a, np.int64), np.asarray(b, np.int64)
        safe = np.where(b64 == 0, 1, b64)
        q = np.abs(a64) // np.abs(safe) * np.sign(a64) * np.sign(safe)
        return wrap(np.where(b64 == 0, 0, q))
    a, b = int(a), int(b)
    if b == 0:
        return 0
    q = abs(a) // abs(b)
    return wrap(q if (a >= 0) == (b > 0) else -q)


def mod(a, b):
    """Sign of the dividend (C semantics); x % 0 = 0 (assumed, matching /)."""
    if _arr(a, b):
        return sub(a, mul(div(a, b), b))
    a, b = int(a), int(b)
    if b == 0:
        return 0
    return wrap(a - div(a, b) * b)


def shr(a, s):
    """Arithmetic right shift; shift amount is taken mod 32."""
    if _arr(a, s):
        return wrap(np.asarray(a, np.int64) >> (np.asarray(s, np.int64) & 31))
    return wrap(int(a) >> (int(s) & 31))


def shl(a, s):
    if _arr(a, s):
        return wrap(np.asarray(a, np.int64) << (np.asarray(s, np.int64) & 31))
    return wrap(int(a) << (int(s) & 31))


def pow_(a, b):
    """Integer power with wrap; negative exponent gives 0."""
    a, b = int(a), int(b)
    if b < 0:
        return 0
    return wrap(pow(a, b, M32))


def dot(x, w):
    """each[red] * each[green] -> single signal: wrapped sum of wrapped products."""
    return wrap(int(np.sum(mul(np.asarray(x), np.asarray(w)).astype(np.int64))))


def _arr(*xs):
    return any(isinstance(x, np.ndarray) for x in xs)


OPS = {"+": add, "-": sub, "*": mul, "/": div, "%": mod, ">>": shr, "<<": shl, "^": pow_}
