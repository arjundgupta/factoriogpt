"""Bit-exact integer reference of the FactorioGPT circuit.

Every operation mirrors one combinator (or one wire sum) of the generated circuit, using
fgpt.fixed semantics (int32 wraparound, truncating /, arithmetic >>). The game must
match this simulator exactly.

Integer parameters (dict, produced by train/export.py):
  E [V,d] Q.8 token embedding     P [T,d] Q.8 position embedding
  layers[l]: wq wk wv wo w1 w2 (int8 matrices, [out, in]) and shifts sq sk sv so s1 s2
  head [V,d] int8, shead
Norm gains and 1/sqrt(dh) are already folded into the following matrices.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import fixed as fx
from . import spec
from .luts import EXP, RSQRT


def matvec(W, x, s):
    """One arithmetic combinator per output: (each[x] * each[W_row]) -> sum, then a vector >> s."""
    W = np.asarray(W, np.int64)
    x = np.asarray(x, np.int64)
    raw = fx.wrap((W * x[None, :]).sum(axis=1)) if W.size else np.zeros(0, np.int64)
    # Each product wraps first, then the sum wraps. Wrapping is modular, so wrapping the total once is identical.
    return fx.shr(raw.astype(np.int64), s).astype(np.int64)


def rmsnorm(x, d):
    xs = fx.shr(np.asarray(x, np.int64), spec.NORM_PRE).astype(np.int64)
    S = fx.wrap(int((xs * xs).sum()))
    ms = fx.div(S, d)
    inv = int(RSQRT(np.array([ms]))[0])
    return fx.shr(fx.mul(np.asarray(x, np.int64), np.int64(inv)).astype(np.int64), spec.INV_SHIFT).astype(np.int64), ms, inv


@dataclass
class State:
    K: list = field(default_factory=list)   # K[l][p] -> int64[d]
    Vc: list = field(default_factory=list)


def step(P: dict, cfg: spec.Config, st: State, tok: int, pos: int, trace: dict | None = None):
    """Forward one token at position pos (writes its K/V into the cache). Returns logits (int64[V])."""
    d, dh = cfg.d, cfg.dh
    x = fx.wrap(np.asarray(P["E"][tok], np.int64) + np.asarray(P["P"][pos], np.int64)).astype(np.int64)
    if trace is not None:
        trace["x0"] = x.copy()
    for l, Ly in enumerate(P["layers"]):
        n, ms, inv = rmsnorm(x, d)
        q = matvec(Ly["wq"], n, Ly["sq"])
        k = matvec(Ly["wk"], n, Ly["sk"])
        v = matvec(Ly["wv"], n, Ly["sv"])
        st.K[l][pos] = k
        st.Vc[l][pos] = v
        att = np.zeros(d, np.int64)
        for h in range(cfg.h):
            hs = slice(h * dh, (h + 1) * dh)
            Ks = np.stack([st.K[l][p] for p in range(pos + 1)])
            raw = fx.wrap((Ks[:, hs] * q[None, hs]).sum(axis=1)).astype(np.int64)
            s = fx.shr(fx.add(raw, np.int64(spec.SCORE_BIG)).astype(np.int64), spec.QK_SHIFT).astype(np.int64)
            mx = int(s.max())
            shifted = fx.sub(s, np.int64(mx - spec.EXP_R)).astype(np.int64)
            e = EXP(shifted)
            Z = fx.wrap(int(e.sum()))
            Vs = np.stack([st.Vc[l][p] for p in range(pos + 1)])
            o = fx.wrap((e[:, None] * Vs).sum(axis=0)).astype(np.int64)
            oh = fx.div(o, np.int64(Z)).astype(np.int64)
            att[hs] = oh[hs]
            if trace is not None:
                trace[f"attn{l}.{h}"] = (e / max(Z, 1)).tolist()   # attention weights over positions 0..pos
        a = matvec(Ly["wo"], att, Ly["so"])
        x = fx.wrap(x + a).astype(np.int64)
        n2, _, _ = rmsnorm(x, d)
        hdn = matvec(Ly["w1"], n2, Ly["s1"])
        hdn = np.where(hdn > 0, hdn, 0)
        m = matvec(Ly["w2"], hdn, Ly["s2"])
        x = fx.wrap(x + m).astype(np.int64)
        if trace is not None:
            trace[f"x{l + 1}"] = x.copy()
    nf, _, _ = rmsnorm(x, d)
    logits = matvec(P["head"], nf, P["shead"])
    if trace is not None:
        trace["logits"] = logits.copy()
    return logits


def new_state(cfg):
    return State(K=[[None] * cfg.T for _ in range(cfg.L)], Vc=[[None] * cfg.T for _ in range(cfg.L)])


def argmax(logits, tok: bool = False):
    """Selector 'max' on logits + LOGIT_BIG; first index on ties (tie-break verified separately).
    Tokenizer builds make every value unique first: (logit << TIE_SHIFT) + LOGIT_BIG_T + (V-1-id)."""
    lg = np.asarray(logits, np.int64)
    if not tok:
        return int(np.argmax(fx.wrap(lg + spec.LOGIT_BIG)))
    V = len(lg)
    u = fx.wrap(fx.wrap(lg * (1 << spec.TIE_SHIFT)) + spec.LOGIT_BIG_T + (V - 1 - np.arange(V)))
    return int(np.argmax(u))


def generate(P: dict, cfg: spec.Config, prompt: str, traces: list | None = None) -> str:
    if cfg.tok:
        return generate_tok(P, cfg, prompt, traces)
    """Emulates the in-game controller: position 0 = USER marker, then the prompt,
    then the BOT marker, then generated tokens until the model emits USER or the context is full."""
    ids = spec.encode(prompt)
    n = len(ids)
    st = new_state(cfg)
    tok = spec.VOCAB[spec.USER]
    out = [tok]
    for pos in range(cfg.T - 1):
        tr = {} if traces is not None else None
        logits = step(P, cfg, st, tok, pos, tr)
        pred = argmax(logits)
        if traces is not None:
            tr["pred"] = pred
            traces.append(tr)
        if pos + 1 <= n:
            tok = ids[pos]
        elif pos + 1 == n + 1:
            tok = spec.VOCAB[spec.BOT]
        else:
            tok = pred
        out.append(tok)
        if pos + 1 >= n + 2 and tok == spec.VOCAB[spec.USER]:
            break
    return spec.decode(out)


def generate_tok(P: dict, cfg: spec.Config, prompt: str, traces: list | None = None, tk=None) -> str:
    """Tokenizer builds: the prompt is fed one character per token; replies are BPE tokens.
    The screen shows letters: each committed token writes its letters at the character cursor CUR.
    Generation also stops when the screen has no room for another token (CUR > screen - LMAX)."""
    from .tokenizer import LMAX
    tk = tk or spec.tokenizer(cfg)
    ids = spec.encode(prompt)
    n = len(ids)
    st = new_state(cfg)
    tok = spec.VOCAB[spec.USER]
    out = [tok]
    cur = 0
    for pos in range(cfg.T - 1):
        if cur > cfg.screen - LMAX:
            break
        tr = {} if traces is not None else None
        logits = step(P, cfg, st, tok, pos, tr)
        pred = argmax(logits, tok=True)
        if traces is not None:
            tr["pred"] = pred
            traces.append(tr)
        if pos + 1 <= n:
            tok = ids[pos]
        elif pos + 1 == n + 1:
            tok = spec.VOCAB[spec.BOT]
        else:
            tok = pred
        out.append(tok)
        cur += len(tk.tokens[tok])
        if pos + 1 >= n + 2 and tok == spec.VOCAB[spec.USER]:
            break
    return tk.decode(out)


if __name__ == "__main__":   # python -m fgpt.intsim "TELL ME A JOKE": the trained model's reply, as the game writes it
    import gzip
    import pickle
    import sys
    from pathlib import Path
    out = Path(__file__).resolve().parents[1] / "train" / "out" / "v3t"
    cfg, P = spec.Config.load(out / "config.json"), pickle.load(gzip.open(out / "int_params.pkl.gz", "rb"))
    for q in sys.argv[1:] or ["TELL ME A JOKE"]:
        print(generate(P, cfg, q))
