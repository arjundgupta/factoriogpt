"""Generate the IN-GAME TRAINING circuit: a tiny GPT that learns with backprop + SGD, built from vanilla combinators.

Mirrors fgpt.trainsim exactly. Every training step (one word from a ROM of constant combinators):
  forward over all T positions (unrolled) -> softmax cross-entropy gradient -> backward -> weight gradients
  -> one UPD pulse adds   delta = grad / -(2^shift)   into each weight-row memory cell.
User controls: INIT (hold ON: load the initial random weights), TRAIN (ON: training runs).
Display: target word, the model's predicted next letters, and an accuracy lamp bar.
"""
from __future__ import annotations

import numpy as np

from . import spec
from . import trainsim as ts
from .circuit import ANYTHING, EACH, EVERYTHING, Circuit
from .gpt_circuit import Seq, _s
from .luts import EXP

C = spec.CONTROL
BIG_S = 1 << 30   # added to raw q.k (before >> 10) so valid scores are never 0
BIG_V = 1 << 20   # added to shifted values (logits, dA) for the same reason


def build_trainer(P: ts.Params, x0=0, y0=0):
    T, D, FF, VT = ts.T, ts.D, ts.FF, ts.VT
    H = 176
    c = Circuit(H, x0, y0, lane=6)
    E, EV = _s(EACH), _s(EVERYTHING)
    S = {k: _s(v) for k, v in C.items()}
    S.update({"INIT": ("virtual", "signal-info"), "TRAIN": ("virtual", "signal-lightning"), "UPD": ("virtual", "signal-W"),
              "SEQ": ("virtual", "signal-Q"), "C": ("virtual", "signal-C"), "CORR": ("virtual", "signal-A")})
    TS = [_s(s) for s in ts.tsignals()]
    SP = [_s(s) for s in spec.pos_signals(T + 1)]
    SD = [_s(s) for s in spec.dim_signals(max(D, FF, VT + T))]
    IN_SIG = TS + SP[:T]          # embedding input columns: vocab one-hot then position one-hot
    info = {"T": T}

    def net(color, name=""):
        return c.net(color, name)

    CTRL, TICKN, UPDN = net("green", "CTRL"), net("green", "TICKN"), net("green", "UPDN")
    T0D, T0C, PHN = net("red"), net("red", "T0C"), net("red")
    SEQC, SEQ1, SEQD = net("red", "SEQC"), net("red"), net("red")
    ROMBUS, VCONST = net("green", "ROMBUS"), net("red", "VCONST")

    # ------------------------------------------------------------ controller
    m = c.module("controller")
    q = Seq(H)
    def put(e, side=False):
        m.put(e, *q(), side=side)
        return e
    init = c.const([(S["INIT"], 1)], CTRL, tag="INIT")
    train = c.const([(S["TRAIN"], 1)], CTRL, tag="TRAIN")
    put(c.selector({"operation": "time", "game_tick_signal": S["TICK"]}, {}, {"g": TICKN}, ctrl=True))
    m.put(init, 0, 0, side=True)
    put(c.arith(S["TICK"], "+", 0, S["T0"], {"g": TICKN}, {"r": T0D}, ctrl=True))
    m.put(train, 0, 1, side=True)
    put(c.decider([(S["INIT"], ">", 0, "g")], [(EV, True, 0, "r")], {"r": T0D, "g": CTRL}, {"r": T0C}, mem=True))
    put(c.decider([(S["INIT"], "=", 0, "g")], [(EV, True, 0, "r")], {"r": T0C, "g": CTRL}, {"r": T0C}, mem=True))
    put(c.arith(S["TICK"], "-", S["T0"], S["PHASE"], {"g": TICKN, "r": T0C}, {"r": PHN}, fn="g", sn="r", ctrl=True))
    ph_mod = put(c.arith(S["PHASE"], "%", 1, S["PHASE"], {"r": PHN}, {"g": CTRL}, ctrl=True))
    put(c.decider([(S["TRAIN"], ">", 0, "g"), (S["INIT"], "=", 0, "g")], [(S["RUN"], False, 1, "rg")], {"g": CTRL}, {"g": CTRL}, ctrl=True))
    upd = put(c.decider([(S["PHASE"], "=", 0, "g"), (S["RUN"], ">", 0, "g")], [(S["UPD"], False, 1, "rg")],
                        {"g": CTRL}, {"g": UPDN}, ctrl=True))
    # SEQ register: next = (SEQ + 1) % N on each UPD
    put(c.arith(S["SEQ"], "+", 1, S["SEQ"], {"r": SEQC}, {"r": SEQ1}, ctrl=True))
    put(c.arith(S["SEQ"], "%", len(ts.SEQS), S["SEQ"], {"r": SEQ1}, {"r": SEQD}, ctrl=True))
    put(c.decider([(S["UPD"], ">", 0, "g")], [(EV, True, 0, "r")], {"r": SEQD, "g": UPDN}, {"r": SEQC}, mem=True))
    put(c.decider([(S["UPD"], "=", 0, "g")], [(EV, True, 0, "r")], {"r": SEQC, "g": UPDN}, {"r": SEQC}, mem=True))
    put(c.arith(S["SEQ"], "+", 0, S["SEQ"], {"r": SEQC}, {"g": CTRL}, ctrl=True))
    m.close()
    info.update(init=init.id, train=train.id)

    # ------------------------------------------------------------ ROM + one-hots
    m = c.module("rom")
    q = Seq(H)
    for s, seq in enumerate(ts.SEQS):
        wn = net("red")
        col, row = q()
        m.put(c.const([(SP[p], seq[p] + 1) for p in range(T + 1)], wn), col, row, side=True)
        m.put(c.decider([(S["SEQ"], "=", s, "g")], [(EV, True, 0, "r")], {"r": wn, "g": CTRL}, {"g": ROMBUS}, ctrl=True), col, row)
    ONEHOT = [net("red", f"ONEHOT{p}") for p in range(T)]
    TGT = [net("green", f"TGT{p}") for p in range(T)]
    col, row = q()
    m.put(c.const([(TS[v], v + 1) for v in range(VT)], VCONST), col, row, side=True)
    for p in range(T):
        col, row = q()
        m.put(c.decider([(E, "=", SP[p], "r", "g")], [(E, False, 1, "r")], {"r": VCONST, "g": ROMBUS}, {"r": ONEHOT[p]}, ctrl=True), col, row)
        m.put(c.const([(SP[p], 1)], ONEHOT[p]), col, row, side=True)
        m.put(c.decider([(E, "=", SP[p + 1], "r", "g")], [(E, False, 1, "r")], {"r": VCONST, "g": ROMBUS}, {"g": TGT[p]}, ctrl=True), *q())
    m.close()

    # ------------------------------------------------------------ helpers
    def vecs(name, color):
        return [net(color, f"{name}{p}") for p in range(T)]

    def matrix(name, W0, sig_in, sig_out, X, Y_raw, Gr, Gg, DX_raw, fwd_shift, upd_div, mm, r_off):
        """Weight rows as memory cells + forward, transpose-backward and outer-product combinators.
        Columns per position p: 3p forward, 3p+1 backward (W^T g), 3p+2 outer product; then the row tail.
        X[p]: inputs (red). Y_raw[p]: forward raw outputs. Gr/Gg: output gradient (red / green copy).
        DX_raw[p]: raw input-gradient outputs (or None). Returns the next free row."""
        rows = W0.shape[0]
        for j in range(rows):
            jj = r_off + j
            MW, FWn, DW, DEL, INC = net("red", f"{name}.MW{j}"), net("green"), net("red"), net("red"), net("red")
            for p in range(T):
                mm.put(c.arith(E, "*", E, sig_out[j], {"r": X[p], "g": FWn}, {"r": Y_raw[p]}, fn="r", sn="g"), 3 * p, jj)
                if DX_raw:
                    mm.put(c.arith(E, "*", sig_out[j], E, {"r": Gr[p], "g": FWn}, {"r": DX_raw[p]}, fn="g", sn="r"), 3 * p + 1, jj)
                mm.put(c.arith(E, "*", sig_out[j], E, {"r": X[p], "g": Gg[p]}, {"r": DW}, fn="r", sn="g"), 3 * p + 2, jj)
            tc = 3 * T
            mm.put(c.arith(E, "/", -upd_div, E, {"r": DW}, {"r": DEL}), tc, jj)
            mm.put(c.arith(E, ">>", fwd_shift, E, {"r": MW}, {"g": FWn}, tag=f"{name}.fwdcopy"),
                   (tc + 1) if DX_raw else (3 * T - 2), jj)
            mm.put(c.decider([(S["UPD"], ">", 0, "g")], [(EV, True, 0, "r")], {"r": DEL, "g": UPDN}, {"r": MW}, mem=True), tc + 2, jj)
            mm.put(c.decider([(S["INIT"], ">", 0, "g")], [(EV, True, 0, "r")], {"r": INC, "g": CTRL}, {"r": MW}, mem=True), tc + 3, jj)
            mm.put(c.const([(sig_in[i], int(W0[j, i])) for i in range(W0.shape[1])], INC), tc + 3, jj, side=True)
            mm.put(c.decider([(S["INIT"], "=", 0, "g")], [(EV, True, 0, "r")], {"r": MW, "g": CTRL}, {"r": MW}, mem=True), tc + 4, jj)
        return r_off + rows

    LUTN = len(EXP.th)

    def softmax(mod, col0, SC, valid_sig, out_r, out_g=None, big=None):
        """Exp-LUT softmax of vector net SC (values already offset so present = valid). Writes (e*1024)/Z to out_r."""
        qq = Seq(H, col0)
        MXS, MXN, SH, EB, ZN, E1K = net("red"), net("green"), net("red"), net("green"), net("green"), net("red")
        mod.put(c.selector({"operation": "select", "select_max": True, "index_constant": 0}, {"r": SC}, {"r": MXS}), *qq())
        cr = qq()
        mod.put(c.arith(E, "+", 0, S["MAX"], {"r": MXS}, {"g": MXN}), *cr)
        mod.put(c.const([(S["MAX"], -spec.EXP_R)], MXN), *cr, side=True)
        mod.put(c.arith(E, "-", S["MAX"], E, {"r": SC, "g": MXN}, {"r": SH}, fn="r", sn="g"), *qq())
        for t, dl in zip(EXP.th, EXP.delta):
            mod.put(c.decider([(E, ">", t, "r")], [(E, False, dl, "g")], {"r": SH}, {"g": EB}), *qq())
        mod.put(c.arith(E, "+", 0, S["Z"], {"g": EB}, {"g": ZN}, fn="g"), *qq())
        mod.put(c.arith(E, "*", 1024, E, {"g": EB}, {"r": E1K}, fn="g"), *qq())
        mod.put(c.arith(E, "/", S["Z"], E, {"r": E1K, "g": ZN}, {"r": out_r}, fn="r", sn="g"), *qq())
        if out_g is not None:
            mod.put(c.arith(E, "+", 0, E, {"r": out_r}, {"g": out_g}, fn="r"), *qq())
        return qq

    def shift(mod, colrow, src, k, dst, side_const=None):
        e = c.arith(E, ">>", k, E, {"r": src}, {dst.color[0]: dst}, fn="r")
        mod.put(e, *colrow)
        return e

    def ident(mod, colrow, src, dst):
        mod.put(c.arith(E, "+", 0, E, {src.color[0]: src}, {dst.color[0]: dst}, fn=src.color[0]), *colrow)

    # ------------------------------------------------------------ nets per position
    X = vecs("X", "red")
    QR, KR, VR = vecs("QR", "red"), vecs("KR", "red"), vecs("VR", "red")
    QS, KS, VS = vecs("QS", "red"), vecs("KS", "green"), vecs("VS", "green")
    OR, OS = vecs("OR", "red"), vecs("OS", "red")
    X1R, X1 = vecs("X1R", "red"), vecs("X1", "red")
    HR, HPRE, Hh = vecs("HR", "red"), vecs("HPRE", "red"), vecs("H", "red")
    X2R, X2 = vecs("X2R", "red"), vecs("X2", "red")
    LR, LOG = vecs("LR", "red"), vecs("LOG", "red")
    PROB, DL, DLg = vecs("PROB", "red"), vecs("DL", "red"), vecs("DLg", "green")
    DX2R, DX2, DX2g = vecs("DX2R", "red"), vecs("DX2", "red"), vecs("DX2g", "green")
    DHR, DH = vecs("DHR", "red"), vecs("DH", "green")
    DHP, DHPg = vecs("DHP", "red"), vecs("DHPg", "green")
    DX1R, DX1, DX1g = vecs("DX1R", "red"), vecs("DX1", "red"), vecs("DX1g", "green")
    DOR, DO = vecs("DOR", "red"), vecs("DO", "red")
    DQR, DQ, DQg = vecs("DQR", "red"), vecs("DQ", "red"), vecs("DQg", "green")
    DKR, DK, DKg = vecs("DKR", "red"), vecs("DK", "red"), vecs("DKg", "green")
    DVR, DV, DVg = vecs("DVR", "red"), vecs("DV", "red"), vecs("DVg", "green")
    DXR, DX, DXg = vecs("DXR", "red"), vecs("DX", "red"), vecs("DXg", "green")

    F = {k: v for k, v in P.items()}
    ldiv = 2 ** (4 + ts.LR)
    edl = 2 ** (ts.LR - 4)

    # ---- attention: column p holds cells (p, p') for p' <= p (rows 6p'..6p'+5), then p's softmax + softmax backward
    SR, SS = vecs("SR", "red"), vecs("SS", "red")
    AR, AG = vecs("AR", "red"), vecs("AG", "green")
    DAR = vecs("DAR", "red")
    DSr, DSg = vecs("DSr", "red"), vecs("DSg", "green")
    ma = c.module("attention")
    for p in range(T):
        cA, cB = 2 * p, 2 * p + 1
        for pp in range(p + 1):
            r = 6 * pp
            ma.put(c.arith(E, "*", E, SP[pp], {"r": QS[p], "g": KS[pp]}, {"r": SR[p]}, fn="r", sn="g"), cA, r)
            ma.put(c.arith(E, "*", SP[pp], E, {"r": AR[p], "g": VS[pp]}, {"r": OR[p]}, fn="g", sn="r"), cA, r + 1)
            ma.put(c.arith(E, "*", SP[pp], E, {"r": DO[p], "g": AG[p]}, {"r": DVR[pp]}, fn="r", sn="g"), cA, r + 2)
            ma.put(c.arith(E, "*", E, SP[pp], {"r": DO[p], "g": VS[pp]}, {"r": DAR[p]}, fn="r", sn="g"), cA, r + 3)
            ma.put(c.arith(E, "*", SP[pp], E, {"r": DSr[p], "g": KS[pp]}, {"r": DQR[p]}, fn="g", sn="r"), cA, r + 4)
            ma.put(c.arith(E, "*", SP[pp], E, {"r": QS[p], "g": DSg[p]}, {"r": DKR[pp]}, fn="r", sn="g"), cA, r + 5)
        # column B: score shift, max, exp LUT, Z, e*1024
        MXS, MXN, SH, EB, ZN, E1K = net("red"), net("green"), net("red"), net("green"), net("green"), net("red")
        qb = Seq(H, cB)
        cr = qb()
        ma.put(c.arith(E, ">>", 10, E, {"r": SR[p]}, {"r": SS[p]}, fn="r"), *cr)
        ma.put(c.const([(SP[pp], BIG_S) for pp in range(p + 1)], SR[p]), *cr, side=True)
        ma.put(c.selector({"operation": "select", "select_max": True, "index_constant": 0}, {"r": SS[p]}, {"r": MXS}), *qb())
        cr = qb()
        ma.put(c.arith(E, "+", 0, S["MAX"], {"r": MXS}, {"g": MXN}), *cr)
        ma.put(c.const([(S["MAX"], -spec.EXP_R)], MXN), *cr, side=True)
        ma.put(c.arith(E, "-", S["MAX"], E, {"r": SS[p], "g": MXN}, {"r": SH}, fn="r", sn="g"), *qb())
        for t, dl in zip(EXP.th, EXP.delta):
            ma.put(c.decider([(E, ">", t, "r")], [(E, False, dl, "g")], {"r": SH}, {"g": EB}), *qb())
        ma.put(c.arith(E, "+", 0, S["Z"], {"g": EB}, {"g": ZN}, fn="g"), *qb())
        ma.put(c.arith(E, "*", 1024, E, {"g": EB}, {"r": E1K}, fn="g"), *qb())
        # column A below the cells: A = (e*1024)/Z, softmax backward
        DA0, DAB, CR, CB, DMC, DSRw = net("red"), net("red"), net("red"), net("green"), net("red"), net("red")
        qa = Seq(H, cA)
        qa.i = 6 * T
        ma.put(c.arith(E, "/", S["Z"], E, {"r": E1K, "g": ZN}, {"r": AR[p]}, fn="r", sn="g"), *qa())
        ma.put(c.arith(E, "+", 0, E, {"r": AR[p]}, {"g": AG[p]}, fn="r"), *qa())
        ma.put(c.arith(E, ">>", 8, E, {"r": DAR[p]}, {"r": DA0}, fn="r"), *qa())
        ma.put(c.arith(E, "*", E, S["C"], {"r": DA0, "g": AG[p]}, {"r": CR}, fn="r", sn="g"), *qa())
        cr = qa()
        ma.put(c.arith(S["C"], ">>", 10, S["C"], {"r": CR}, {"g": CB}, fn="r"), *cr)
        ma.put(c.const([(S["C"], BIG_V)], CB), *cr, side=True)
        cr = qa()
        ma.put(c.arith(E, "+", 0, E, {"r": DA0}, {"r": DAB}, fn="r"), *cr)
        ma.put(c.const([(SP[pp], BIG_V) for pp in range(p + 1)], DAB), *cr, side=True)
        ma.put(c.arith(E, "-", S["C"], E, {"r": DAB, "g": CB}, {"r": DMC}, fn="r", sn="g"), *qa())
        ma.put(c.arith(E, "*", E, E, {"r": DMC, "g": AG[p]}, {"r": DSRw}, fn="r", sn="g"), *qa())
        ma.put(c.arith(E, ">>", 10, E, {"r": DSRw}, {"r": DSr[p]}, fn="r"), *qa())
        ma.put(c.arith(E, "+", 0, E, {"r": DSr[p]}, {"g": DSg[p]}, fn="r"), *qa())
    ma.close()

    WM = c.module("weights")
    # ---- embedding E: X = Ef . onehot
    r = matrix("E", F["E"], IN_SIG, SD, ONEHOT, X, None, DXg, None, ts.ES, edl, WM, 0)
    # ---- q, k, v stacked in one module (shared per-position input X and input-gradient DXR columns)
    mm = WM
    for name, Wm, raw, G, Gg in (("Wq", F["Wq"], QR, DQ, DQg), ("Wk", F["Wk"], KR, DK, DKg), ("Wv", F["Wv"], VR, DV, DVg)):
        r = matrix(name, Wm, SD, SD, X, raw, G, Gg, DXR, ts.FS, ldiv, mm, r)
    r0 = r
    for p in range(T):
        shift(mm, (3 * p, r0), QR[p], 10, QS[p])
        shift(mm, (3 * p, r0 + 1), KR[p], 10, KS[p])
        shift(mm, (3 * p, r0 + 2), VR[p], 10, VS[p])
        shift(mm, (3 * p + 1, r0), DQR[p], 10, DQ[p])
        shift(mm, (3 * p + 1, r0 + 1), DKR[p], 10, DK[p])
        shift(mm, (3 * p + 1, r0 + 2), DVR[p], 10, DV[p])
        ident(mm, (3 * p + 2, r0), DQ[p], DQg[p])
        ident(mm, (3 * p + 2, r0 + 1), DK[p], DKg[p])
        ident(mm, (3 * p + 2, r0 + 2), DV[p], DVg[p])
        shift(mm, (3 * p + 1, r0 + 3), DXR[p], 10, DX[p])
        ident(mm, (3 * p, r0 + 3), DX1[p], DX[p])
        ident(mm, (3 * p + 2, r0 + 3), DX[p], DXg[p])
    r = r0 + 4

    # ---- output projection Wo, residual
    mo = WM
    r0 = matrix("Wo", F["Wo"], SD, SD, OS, X1R, DX1, DX1g, DOR, ts.FS, ldiv, mo, r)
    for p in range(T):
        shift(mo, (3 * p, r0), OR[p], 10, OS[p])
        shift(mo, (3 * p, r0 + 1), X1R[p], 10, X1[p])
        ident(mo, (3 * p, r0 + 2), X[p], X1[p])
        shift(mo, (3 * p + 1, r0), DOR[p], 10, DO[p])
        ident(mo, (3 * p + 2, r0), DX1[p], DX1g[p])
        ident(mo, (3 * p + 1, r0 + 1), DX2[p], DX1[p])
    r = r0 + 3
    # ---- MLP
    m1 = WM
    r0 = matrix("W1", F["W1"], SD, SD, X1, HR, DHP, DHPg, DX1R, ts.FS, ldiv, m1, r)
    for p in range(T):
        shift(m1, (3 * p, r0), HR[p], 10, HPRE[p])
        m1.put(c.decider([(E, ">", 0, "r")], [(E, True, 0, "r")], {"r": HPRE[p]}, {"r": Hh[p]}), 3 * p, r0 + 1)
        m1.put(c.decider([(E, ">", 0, "r")], [(E, True, 0, "g")], {"r": HPRE[p], "g": DH[p]}, {"r": DHP[p]}), 3 * p + 1, r0)
        ident(m1, (3 * p + 2, r0), DHP[p], DHPg[p])
        shift(m1, (3 * p + 1, r0 + 1), DX1R[p], 10, DX1[p])
    r = r0 + 2
    m2 = WM
    r0 = matrix("W2", F["W2"], SD, SD, Hh, X2R, DX2, DX2g, DHR, ts.FS, ldiv, m2, r)
    for p in range(T):
        shift(m2, (3 * p, r0), X2R[p], 10, X2[p])
        ident(m2, (3 * p, r0 + 1), X1[p], X2[p])
        shift(m2, (3 * p + 1, r0), DHR[p], 10, DH[p])
        ident(m2, (3 * p + 2, r0), DX2[p], DX2g[p])
    r = r0 + 2
    # ---- head
    mh = WM
    r0 = matrix("Wh", F["Wh"], SD, TS, X2, LR, DL, DLg, DX2R, ts.FS, ldiv, mh, r)
    for p in range(T):
        cr = (3 * p, r0)
        shift(mh, cr, LR[p], 10, LOG[p])
        mh.put(c.const([(TS[v], BIG_V) for v in range(VT)], LOG[p]), *cr, side=True)
        shift(mh, (3 * p + 1, r0), DX2R[p], 10, DX2[p])
        ident(mh, (3 * p + 2, r0), DL[p], DLg[p])
    r = r0 + 1
    WM.close()

    # ---- output softmax, loss gradient, display
    mso = c.module("out_softmax")
    PREDN = [net("red", f"PRED{p}") for p in range(T)]
    CORR = net("red", "CORR")
    for p in range(T):
        qq = Seq(H, p)
        qq = _softmax_seq(c, mso, qq, LOG[p], PROB[p], None, net, S)
        mso.put(c.arith(E, "+", 0, E, {"r": PROB[p]}, {"r": DL[p]}, fn="r"), *qq())
        mso.put(c.arith(E, "*", -1024, E, {"g": TGT[p]}, {"r": DL[p]}, fn="g"), *qq())
        mso.put(c.arith(E, "*", E, S["CORR"], {"r": PROB[p], "g": TGT[p]}, {"r": CORR}, fn="r", sn="g"), *qq())
        PS = net("red")
        mso.put(c.selector({"operation": "select", "select_max": True, "index_constant": 0}, {"r": LOG[p]}, {"r": PS}), *qq())
        mso.put(c.decider([(E, "!=", 0, "r")], [(E, False, 1, "r")], {"r": PS}, {"r": PREDN[p]}), *qq())
    mso.close()

    # display: target letters (top row), predicted letters (bottom row), accuracy bar
    info["pred_panels"], info["tgt_panels"], info["lamps"] = [], [], []
    def disp_item(k, x, y):
        if k < T:
            pt = c.panel(TGT[k], tag=f"TGTP{k}")
            c.place_free(pt, x + 0.5, y + 0.5, tag=f"TGTP{k}")
            pp = c.panel(PREDN[k], tag=f"PREDP{k}")
            c.place_free(pp, x + 0.5, y + 2.5, tag=f"PREDP{k}")
            info["tgt_panels"].append(pt.id)
            info["pred_panels"].append(pp.id)
        else:
            i = k - T
            lamp = c._ent("lamp", {"condition": {"first_signal": S["CORR"], "comparator": ">=",
                                                 "constant": int((i + 1) * T * 1024 / 9)}}, {"r": CORR}, tag=f"ACC{i}")
            c.place_free(lamp, x + 0.5, y + 4.5, tag=f"ACC{i}")
            info["lamps"].append(lamp.id)
    gb = c.grid_bbox()
    c.text_region(x0, gb[3] + 1, T + 8, 16, 6, disp_item, "display")

    # ------------------------------------------------------------ timing
    period = _train_timing(c)
    c.ents[upd.id].params["conditions"][0]["constant"] = period - 1
    c.ents[ph_mod.id].params["second_constant"] = period
    info["period"] = period
    c.add_power()
    info["route"] = c.route()
    return c, info


def _softmax_seq(c, mod, qq, SC, out_r, out_g, net, S):
    E = _s(EACH)
    MXS, MXN, SH, EB, ZN, E1K = net("red"), net("green"), net("red"), net("green"), net("green"), net("red")
    mod.put(c.selector({"operation": "select", "select_max": True, "index_constant": 0}, {"r": SC}, {"r": MXS}), *qq())
    cr = qq()
    mod.put(c.arith(E, "+", 0, S["MAX"], {"r": MXS}, {"g": MXN}), *cr)
    mod.put(c.const([(S["MAX"], -spec.EXP_R)], MXN), *cr, side=True)
    mod.put(c.arith(E, "-", S["MAX"], E, {"r": SC, "g": MXN}, {"r": SH}, fn="r", sn="g"), *qq())
    for t, dl in zip(EXP.th, EXP.delta):
        mod.put(c.decider([(E, ">", t, "r")], [(E, False, dl, "g")], {"r": SH}, {"g": EB}), *qq())
    mod.put(c.arith(E, "+", 0, S["Z"], {"g": EB}, {"g": ZN}, fn="g"), *qq())
    mod.put(c.arith(E, "*", 1024, E, {"g": EB}, {"r": E1K}, fn="g"), *qq())
    mod.put(c.arith(E, "/", S["Z"], E, {"r": E1K, "g": ZN}, {"r": out_r}, fn="r", sn="g"), *qq())
    if out_g is not None:
        mod.put(c.arith(E, "+", 0, E, {"r": out_r}, {"g": out_g}, fn="r"), *qq())
    return qq


def _train_timing(c: Circuit):
    producers = {}
    for e in c.ents:
        for conn, nid in e.conns.items():
            if conn.startswith("out") or e.kind == "const":
                producers.setdefault(nid, []).append(e)
    memo = {}

    def settle(nid, stack=()):
        if nid in memo:
            return memo[nid]
        assert nid not in stack, f"loop through {c.nets[nid].name}"
        best = 0
        for e in producers.get(nid, []):
            if e.kind == "const":
                continue
            if e.mem:
                best = max(best, 1)
                continue
            if e.ctrl:
                best = max(best, 6)
                continue
            ins = [n for k, n in e.conns.items() if k.startswith("in")]
            best = max(best, 1 + max((settle(n, stack + (nid,)) for n in ins), default=0))
        memo[nid] = best
        return best

    worst = 0
    for e in c.ents:
        if e.kind == "decider" and e.mem and e.params["conditions"][0]["first_signal"] == ("virtual", "signal-W"):
            worst = max(worst, settle(e.conns["in_r"]))
    return worst + 4
