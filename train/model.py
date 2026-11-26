"""FactorioGPT in PyTorch: float training plus quantization-aware training (QAT) that mirrors fgpt.intsim."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fgpt import spec  # noqa: E402
from fgpt.luts import EXP, RSQRT  # noqa: E402


def ste(x, q):
    return x + (q - x).detach()


def floor_ste(x):
    return ste(x, torch.floor(x))


def trunc_ste(x):
    return ste(x, torch.trunc(x))


class Tables:
    def __init__(self, device):
        self.r_th = torch.tensor(RSQRT.th, dtype=torch.float32, device=device)
        vals, v = [], RSQRT.base
        for dlt in RSQRT.delta:
            v += dlt
            vals.append(v)
        self.r_vals = torch.tensor([RSQRT.base] + vals, dtype=torch.float32, device=device)
        K, S = spec.EXP_LEVELS, spec.EXP_STEP
        E = [round(spec.EXP_ONE * math.exp(-(k * S) / 256)) for k in range(K)] + [0]
        self.e_vals = torch.tensor(E, dtype=torch.float32, device=device)

    def rsqrt(self, ms):
        idx = torch.bucketize(ms, self.r_th, right=True)
        q = self.r_vals[idx]
        true = 2 ** 16 / torch.sqrt(torch.clamp(ms, min=0.5))
        return ste(true, q)

    def exp(self, shifted):
        """shifted <= R; returns E[floor((R - shifted)/S)] (0 beyond the table) with an exp-shaped gradient."""
        b = torch.floor((spec.EXP_R - shifted) / spec.EXP_STEP).clamp(0, spec.EXP_LEVELS).long()
        q = self.e_vals[b]
        true = spec.EXP_ONE * torch.exp((shifted - spec.EXP_R) / 256)
        return ste(true, q)


class GPT(nn.Module):
    def __init__(self, cfg: spec.Config):
        super().__init__()
        self.cfg = cfg
        d, ff = cfg.d, cfg.ff
        self.E = nn.Parameter(torch.randn(cfg.vocab, d) * 0.3)
        self.P = nn.Parameter(torch.randn(cfg.T, d) * 0.1)
        self.layers = nn.ModuleList()
        for _ in range(cfg.L):
            m = nn.Module()
            m.g1 = nn.Parameter(torch.ones(d))
            m.wq = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
            m.wk = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
            m.wv = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
            m.wo = nn.Parameter(torch.randn(d, d) / math.sqrt(d) / math.sqrt(2 * cfg.L))
            m.g2 = nn.Parameter(torch.ones(d))
            m.w1 = nn.Parameter(torch.randn(ff, d) / math.sqrt(d))
            m.w2 = nn.Parameter(torch.randn(d, ff) / math.sqrt(ff) / math.sqrt(2 * cfg.L))
            self.layers.append(m)
        self.gf = nn.Parameter(torch.ones(d))
        self.head = nn.Parameter(torch.randn(cfg.vocab, d) / math.sqrt(d))
        self.shifts = None   # frozen per-matrix shifts for QAT
        self.tables = None

    # ------------------------------------------------------------------ folded weights
    def folded(self):
        """Float matrices exactly as they go into the circuit: norm gains and 1/sqrt(dh) folded in."""
        out = []
        sc = 1 / math.sqrt(self.cfg.dh)
        for m in self.layers:
            out.append({"wq": m.wq * m.g1[None, :] * sc, "wk": m.wk * m.g1[None, :], "wv": m.wv * m.g1[None, :],
                        "wo": m.wo, "w1": m.w1 * m.g2[None, :], "w2": m.w2})
        return out, self.head * self.gf[None, :]

    def freeze_shifts(self):
        layers, head = self.folded()
        def sh(w):
            mx = float(w.detach().abs().max())
            return int(math.floor(math.log2(127 / max(mx, 1e-8))))
        self.shifts = {"layers": [{k: sh(w) for k, w in L.items()} for L in layers], "head": sh(head)}
        return self.shifts

    # ------------------------------------------------------------------ float forward
    def forward_float(self, idx):
        B, T = idx.shape
        cfg = self.cfg
        x = self.E[idx] + self.P[:T][None]
        mask = torch.triu(torch.ones(T, T, device=idx.device, dtype=torch.bool), 1)
        layers, head = self.folded()
        for W in layers:
            n = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6)
            q = (n @ W["wq"].T).view(B, T, cfg.h, cfg.dh).transpose(1, 2)
            k = (n @ W["wk"].T).view(B, T, cfg.h, cfg.dh).transpose(1, 2)
            v = (n @ W["wv"].T).view(B, T, cfg.h, cfg.dh).transpose(1, 2)
            s = (q @ k.transpose(-1, -2)).masked_fill(mask, float("-inf"))
            a = (torch.softmax(s, -1) @ v).transpose(1, 2).reshape(B, T, cfg.d)
            x = x + a @ W["wo"].T
            n2 = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6)
            x = x + torch.relu(n2 @ W["w1"].T) @ W["w2"].T
        nf = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6)
        return nf @ head.T

    # ------------------------------------------------------------------ QAT forward (mirrors intsim)
    def qmat(self, w, s):
        return ste(w * 2 ** s, torch.clamp(torch.round(w * 2 ** s), -127, 127))

    def qmv(self, x, w, s):
        return floor_ste((x @ self.qmat(w, s).T) / 2 ** s)

    def qrms(self, x):
        xs = floor_ste(x / 2 ** spec.NORM_PRE)
        ms = floor_ste((xs * xs).sum(-1, keepdim=True) / self.cfg.d)
        inv = self.tables.rsqrt(ms)
        return floor_ste(x * inv / 2 ** spec.INV_SHIFT)

    def forward_qat(self, idx):
        if self.tables is None or self.tables.r_th.device != idx.device:
            self.tables = Tables(idx.device)
        B, T = idx.shape
        cfg = self.cfg
        x = ste(self.E * 256, torch.round(self.E * 256))[idx] + ste(self.P * 256, torch.round(self.P * 256))[:T][None]
        mask = torch.triu(torch.ones(T, T, device=idx.device, dtype=torch.bool), 1)
        layers, head = self.folded()
        for W, S in zip(layers, self.shifts["layers"]):
            n = self.qrms(x)
            q = self.qmv(n, W["wq"], S["wq"]).view(B, T, cfg.h, cfg.dh).transpose(1, 2)
            k = self.qmv(n, W["wk"], S["wk"]).view(B, T, cfg.h, cfg.dh).transpose(1, 2)
            v = self.qmv(n, W["wv"], S["wv"]).view(B, T, cfg.h, cfg.dh).transpose(1, 2)
            s = floor_ste((q @ k.transpose(-1, -2)) / 2 ** spec.QK_SHIFT)
            s = s.masked_fill(mask, -1e9)
            mx = s.max(-1, keepdim=True).values.detach()
            e = self.tables.exp(s - mx + spec.EXP_R).masked_fill(mask, 0.0)
            Z = e.sum(-1, keepdim=True)
            o = trunc_ste((e @ v) / Z)
            a = o.transpose(1, 2).reshape(B, T, cfg.d)
            x = x + self.qmv(a, W["wo"], S["wo"])
            n2 = self.qrms(x)
            hdn = torch.relu(self.qmv(n2, W["w1"], S["w1"]))
            x = x + self.qmv(hdn, W["w2"], S["w2"])
        nf = self.qrms(x)
        return self.qmv(nf, head, self.shifts["head"]) / 256.0

    def forward(self, idx, qat=False):
        return self.forward_qat(idx) if qat else self.forward_float(idx)

    # ------------------------------------------------------------------ export
    @torch.no_grad()
    def export_int(self) -> dict:
        assert self.shifts is not None
        layers, head = self.folded()
        def q(w, s):
            return np.clip(np.round(w.detach().cpu().double().numpy() * 2 ** s), -127, 127).astype(np.int64)
        P = {"E": np.round(self.E.detach().cpu().double().numpy() * 256).astype(np.int64),
             "P": np.round(self.P.detach().cpu().double().numpy() * 256).astype(np.int64),
             "layers": [], "head": q(head, self.shifts["head"]), "shead": self.shifts["head"]}
        for W, S in zip(layers, self.shifts["layers"]):
            P["layers"].append({k: q(W[k], S[k]) for k in W} | {"s" + k[1:]: S[k] for k in W})
        return P
