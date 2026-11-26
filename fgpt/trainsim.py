"""Integer reference for IN-GAME training of a tiny GPT (forward + backward + SGD), one op = one combinator.

Model (train_tiny): 1 transformer block, 1 head, no norm, ReLU MLP, learned token+position embeddings.
All T positions of one training sequence are computed in parallel (unrolled) each step.

Scales:
  master weights  : int, w_float * 2^14    (memory cells, updated by SGD)
  forward copies  : master >> 4  (2^10)    embedding copy: master >> 6 (Q.8)
  activations     : Q.8   (x, q, k, v, o, h, logits)
  probabilities   : 2^10  (attention A, output softmax)
  activation grads: 2^10
  weight grads    : 2^18 (= grad_float * 2^18)   -> update  = grad / -(2^(4 + LR))
  embedding grads : 2^10                         -> update  = grad * 2^4 / -(2^LR)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import fixed as fx
from . import spec
from .luts import EXP

WORDS = ["IRON", "COPPER", "STEEL", "COAL", "STONE", "GEAR", "BELT", "PIPE", "PUMP", "LAMP", "WIRE", "CABLE",
         "CHIP", "ROBOT", "DRONE", "TRAIN", "WAGON", "RAIL", "TANK", "CAR", "OIL", "WATER", "STEAM", "BOILER",
         "ENGINE", "TURRET", "WALL", "GATE", "ROCKET", "BITER", "SPIDER", "LASER", "RADAR", "SOLAR", "NUCLEAR",
         "PLASTIC", "SULFUR", "BATTERY", "LOGIC", "SCIENCE"]
CHARS = ["."] + [chr(ord("A") + i) for i in range(26)]
TV = {c: i for i, c in enumerate(CHARS)}
VT = len(CHARS)       # 27
T, D, FF = 8, 16, 32
MS, FS, ES = 14, 4, 6  # master scale bits, forward-copy shift, embedding forward shift
LR = 6                 # learning rate = 2^-LR


def word_seq(w: str) -> list[int]:
    s = ("." + w + "." * (T + 1))[: T + 1]
    return [TV[c] for c in s]


SEQS = [word_seq(w) for w in WORDS if len(w) <= T - 1]


def tsignals():
    """Vocab signals (letters + dot) for the trainer; positions use spec.pos_signals."""
    names = [("virtual", "signal-letter-dot")] + [("virtual", f"signal-{c}") for c in CHARS[1:]]
    return names


@dataclass
class Params:
    E: np.ndarray    # [D, VT + T]  rows = output dims, columns = (vocab one-hot, position one-hot)
    Wq: np.ndarray   # [D, D]
    Wk: np.ndarray
    Wv: np.ndarray
    Wo: np.ndarray
    W1: np.ndarray   # [FF, D]
    W2: np.ndarray   # [D, FF]
    Wh: np.ndarray   # [VT, D]

    def items(self):
        return [("E", self.E), ("Wq", self.Wq), ("Wk", self.Wk), ("Wv", self.Wv), ("Wo", self.Wo),
                ("W1", self.W1), ("W2", self.W2), ("Wh", self.Wh)]


def init(seed=0) -> Params:
    rng = np.random.default_rng(seed)
    def m(o, i, scale):
        return np.round(rng.normal(0, scale, (o, i)) * 2 ** MS).astype(np.int64)
    return Params(E=m(D, VT + T, 0.5), Wq=m(D, D, 1 / math.sqrt(D)), Wk=m(D, D, 1 / math.sqrt(D)),
                  Wv=m(D, D, 1 / math.sqrt(D)), Wo=m(D, D, 0.5 / math.sqrt(D)), W1=m(FF, D, 1 / math.sqrt(D)),
                  W2=m(D, FF, 0.5 / math.sqrt(FF)), Wh=m(VT, D, 1 / math.sqrt(D)))


def W32(x):
    return fx.wrap(np.asarray(x, np.int64)).astype(np.int64)


def mv(Wf, X, s):
    """Rows of Wf times each row of X (positions): [P, out]; one arithmetic combinator per (position, out)."""
    return W32(fx.shr(W32(np.asarray(X, np.int64) @ np.asarray(Wf, np.int64).T), s))


def mvT(Wf, G, s):
    """Transpose product: sum_j G[p, j] * Wf[j, :]  (scalar x vector combinators summed on a wire)."""
    return W32(fx.shr(W32(np.asarray(G, np.int64) @ np.asarray(Wf, np.int64)), s))


def outer(G, X):
    """Weight gradient rows: dW[j, :] = sum_p G[p, j] * X[p, :]."""
    return W32(np.asarray(G, np.int64).T @ np.asarray(X, np.int64))


def softmax_rows(S, valid):
    """Exp-LUT softmax on rows of S (Q.8 scores), probabilities scaled 2^10 (trunc division)."""
    big = S + 2 ** 20
    m = np.where(valid, big, -2 ** 40).max(axis=1, keepdims=True)
    e = EXP(np.where(valid, big - (m - spec.EXP_R), 0))
    e = np.where(valid, e, 0)
    Z = e.sum(axis=1, keepdims=True)
    return W32(fx.div(W32(e * 1024), np.broadcast_to(Z, e.shape)))


def step(P: Params, seq: list[int], train: bool = True):
    """One training step on one sequence. Returns (predictions, correct-prob sum, grads dict)."""
    onehot = np.zeros((T, VT + T), np.int64)
    for p in range(T):
        onehot[p, seq[p]] = 1
        onehot[p, VT + p] = 1
    tgt = np.array(seq[1:T + 1])
    F = {k: fx.shr(v, FS) for k, v in P.items() if k != "E"}
    Ef = fx.shr(P.E, ES)
    x = mv(Ef, onehot, 0)                                      # Q.8
    q, k, v = mv(F["Wq"], x, 10), mv(F["Wk"], x, 10), mv(F["Wv"], x, 10)
    valid = np.tril(np.ones((T, T), bool))
    S = W32(fx.shr(W32(q @ k.T), 10))                          # (q.k)/4 in Q.8
    A = softmax_rows(S, valid)                                 # 2^10
    o = W32(fx.shr(W32(A @ v), 10))
    x1 = W32(x + mv(F["Wo"], o, 10))
    hpre = mv(F["W1"], x1, 10)
    h = np.where(hpre > 0, hpre, 0)
    x2 = W32(x1 + mv(F["W2"], h, 10))
    logits = mv(F["Wh"], x2, 10)
    probs = softmax_rows(logits, np.ones((T, VT), bool))
    pred = probs.argmax(axis=1)
    correct = int(probs[np.arange(T), tgt].sum())
    if not train:
        return pred, correct, None
    Y = np.zeros((T, VT), np.int64)
    Y[np.arange(T), tgt] = 1024
    dl = W32(probs - Y)                                        # 2^10
    # backward
    g = {}
    g["Wh"] = outer(dl, x2)                                    # 2^18
    dx2 = mvT(F["Wh"], dl, 10)                                 # 2^10
    g["W2"] = outer(dx2, h)
    dh = mvT(F["W2"], dx2, 10)
    dhp = np.where(hpre > 0, dh, 0)
    g["W1"] = outer(dhp, x1)
    dx1 = W32(dx2 + mvT(F["W1"], dhp, 10))
    g["Wo"] = outer(dx1, o)
    do = mvT(F["Wo"], dx1, 10)                                 # 2^10
    dv = W32(fx.shr(W32(A.T @ do), 10))                        # sum_p A[p,p'] do_p
    dA = W32(fx.shr(W32(do @ v.T), 8))                         # 2^10
    dA = np.where(valid, dA, 0)
    c = W32(fx.shr(W32((A * dA).sum(axis=1, keepdims=True)), 10))
    dS = W32(fx.shr(W32(A * W32(dA - c)), 10))
    dS = np.where(valid, dS, 0)
    dq = W32(fx.shr(W32(dS @ k), 10))
    dk = W32(fx.shr(W32(dS.T @ q), 10))
    g["Wq"], g["Wk"], g["Wv"] = outer(dq, x), outer(dk, x), outer(dv, x)
    raw = W32(np.asarray(dq, np.int64) @ F["Wq"] + np.asarray(dk, np.int64) @ F["Wk"] + np.asarray(dv, np.int64) @ F["Wv"])
    dx = W32(dx1 + fx.shr(raw, 10))
    g["E"] = outer(dx, onehot)                                 # 2^10
    return pred, correct, g


def update(P: Params, g: dict):
    for name, W in P.items():
        if name == "E":
            delta = fx.div(g[name], np.int64(-(2 ** (LR - 4))))
        else:
            delta = fx.div(g[name], np.int64(-(2 ** (4 + LR))))
        W[...] = W32(W + delta)


def evaluate(P: Params):
    tot, acc, n = 0, 0, 0
    for s in SEQS:
        pred, correct, _ = step(P, s, train=False)
        L = s.index(TV["."], 1) if TV["."] in s[1:] else T
        acc += int((pred[:L] == np.array(s[1:L + 1])).sum())
        n += L
        tot += correct
    return acc / n, tot / (len(SEQS) * T * 1024)


def train(steps=3000, seed=0, log=print):
    P = init(seed)
    hist = []
    for i in range(steps):
        s = SEQS[i % len(SEQS)]
        _, _, g = step(P, s)
        update(P, g)
        if i % 200 == 0 or i == steps - 1:
            acc, pc = evaluate(P)
            hist.append((i, acc, pc))
            log(f"step {i}: char accuracy {acc:.1%}, mean p(correct) {pc:.3f}")
    return P, hist
