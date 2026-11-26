"""Generate the full FactorioGPT circuit (vanilla combinators) from integer parameters.

Datapath = intsim.step, one combinator per intsim operation. Controller = intsim.generate:
  clock (selector game tick) -> PHASE = (TICK - T0) % P ; T0 latched while RESET is held
  commit every P ticks: token register <- next token, POS <- POS + 1, screen cell <- token
  KV cache cells for layer l are written at PHASE == t_l (after K/V settle)
User interface: a RESET constant combinator (hold ON while typing), keyboard constant
combinators (one character each), display panels showing the conversation.
"""
from __future__ import annotations

from . import spec
from .circuit import ANYTHING, EACH, EVERYTHING, Circuit
from .luts import EXP, RSQRT

C = spec.CONTROL


def _sig(s):
    return {"type": s[0], "name": s[1]}


def build_gpt(P: dict, cfg: spec.Config, kb: int | None = None, x0=0, y0=0, ui: str = "text"):
    T, d, L, h, dh, ff, V = cfg.T, cfg.d, cfg.L, cfg.h, cfg.dh, cfg.ff, cfg.vocab
    H = max(64, ((T + 15) // 16) * 16)
    kb = kb or (min(32, T // 2) if ui == "columns" else min(31, T // 2))
    c = Circuit(H, x0, y0)
    tk = spec.tokenizer(cfg)
    SV = [_s(s) for s in (spec.token_signals(V) if tk else spec.VOCAB_SIGNALS)]
    SP = [_s(s) for s in spec.pos_signals(T)]
    SD = [_s(s) for s in spec.dim_signals(max(d, ff))]
    S = {k: _s(v) for k, v in C.items()}
    E, EV = _s(EACH), _s(EVERYTHING)
    info = {"cfg": cfg, "kb": kb}

    # ---------------------------------------------------------------- nets
    CTRL = c.net("green", "CTRL")
    CMT = c.net("green", "CMT")
    TICKN = c.net("green", "TICKN")
    T0D, T0C, PHN = c.net("red", "T0D"), c.net("red", "T0C"), c.net("red", "PHN")
    POSC, POSD1, POSD = c.net("red", "POSC"), c.net("red", "POSD1"), c.net("red", "POSD")
    TOKC, NXT, PRED = c.net("red", "TOKC"), c.net("red", "NXT"), c.net("red", "PRED")
    DONEC, DOND = c.net("red", "DONEC"), c.net("red", "DOND")
    EMBIN, PCONST, MASK = c.net("red", "EMBIN"), c.net("red", "PCONST"), c.net("green", "MASK")
    SCRD = c.net("red", "SCRD")
    LENN = c.net("green", "LEN")   # tokenizer builds: letters in the token being committed

    # ---------------------------------------------------------------- controller
    m = c.module("controller")
    rows = iter(range(H))
    def put(e, side=False, col=0):
        m.put(e, col, next(rows), side)
        return e
    reset = c.const([(S["RESET"], 1)], CTRL, tag="RESET")
    put(c.selector({"operation": "time", "game_tick_signal": S["TICK"]}, {}, {"g": TICKN}, ctrl=True))
    if ui == "columns":
        m.put(reset, 0, 0, side=True)
    put(c.arith(S["TICK"], "+", 0, S["T0"], {"g": TICKN}, {"r": T0D}, ctrl=True))
    put(c.decider([(S["RESET"], ">", 0, "g")], [(EV, True, 0, "r")], {"r": T0D, "g": CTRL}, {"r": T0C}, mem=True))
    put(c.decider([(S["RESET"], "=", 0, "g")], [(EV, True, 0, "r")], {"r": T0C, "g": CTRL}, {"r": T0C}, mem=True))
    put(c.arith(S["TICK"], "-", S["T0"], S["PHASE"], {"g": TICKN, "r": T0C}, {"r": PHN}, fn="g", sn="r", ctrl=True))
    ph_mod = put(c.arith(S["PHASE"], "%", 1, S["PHASE"], {"r": PHN}, {"g": CTRL}, ctrl=True))
    # POS register
    put(c.arith(S["POS"], "+", 1, S["POS"], {"r": POSC}, {"r": POSD1}, ctrl=True))
    put(c.decider([(S["RESET"], "=", 0, "g")], [(EV, True, 0, "r")], {"r": POSD1, "g": CTRL}, {"r": POSD}, ctrl=True))
    _cell(c, put, POSD, POSC, CMT)
    put(c.arith(S["POS"], "+", 0, S["POS"], {"r": POSC}, {"g": CTRL}, ctrl=True))
    put(c.arith(S["POS"], "+", 1, S["POS1"], {"r": POSC}, {"g": CTRL}, ctrl=True))
    # DONE register
    put(c.decider([(_s(spec.VOCAB_SIGNALS[spec.VOCAB[spec.USER]]), ">", 0, "r"), (S["POS"], ">", S["N"], "g", "g"),
                   (S["RESET"], "=", 0, "g")], [(S["DONE"], False, 1, "rg")], {"r": PRED, "g": CTRL}, {"r": DOND}, ctrl=True))
    _cell(c, put, DOND, DONEC, CMT)
    put(c.arith(S["DONE"], "+", 0, S["DONE"], {"r": DONEC}, {"g": CTRL}, ctrl=True))
    run_conds = [(S["RESET"], "=", 0, "g"), (S["DONE"], "=", 0, "g"), (S["POS"], "<", T - 1, "g")]
    if tk:
        # CUR register: letters on screen so far. Stop when there is no room for another token.
        from .tokenizer import LMAX
        CURC, CURD1, CURD = c.net("red", "CURC"), c.net("red", "CURD1"), c.net("red", "CURD")
        put(c.arith(S["CUR"], "+", S["LEN"], S["CUR"], {"r": CURC, "g": LENN}, {"r": CURD1}, fn="r", sn="g", ctrl=True))
        put(c.decider([(S["RESET"], "=", 0, "g")], [(EV, True, 0, "r")], {"r": CURD1, "g": CTRL}, {"r": CURD}, ctrl=True))
        _cell(c, put, CURD, CURC, CMT)
        put(c.arith(S["CUR"], "+", 0, S["CUR"], {"r": CURC}, {"g": CTRL}, ctrl=True))
        run_conds.append((S["CUR"], "<=", cfg.screen - LMAX, "g"))
    put(c.decider(run_conds, [(S["RUN"], False, 1, "rg")], {"g": CTRL}, {"g": CTRL}, ctrl=True))
    commit = put(c.decider([(S["PHASE"], "=", 0, "g"), (S["RUN"], ">", 0, "g")], [(S["W"], False, 1, "rg")],
                           {"g": CTRL}, {"g": CMT}, ctrl=True))
    put(c.decider([(S["RESET"], ">", 0, "g")], [(S["W"], False, 1, "rg")], {"g": CTRL}, {"g": CMT}, ctrl=True))
    # next token logic
    put(c.decider([(S["RESET"], ">", 0, "g")], [(SV[spec.VOCAB[spec.USER]], False, 1, "rg")], {"g": CTRL}, {"r": NXT}, ctrl=True))
    put(c.decider([(S["POS"], "=", S["N"], "g", "g"), (S["RESET"], "=", 0, "g")], [(SV[spec.VOCAB[spec.BOT]], False, 1, "rg")],
                  {"g": CTRL}, {"r": NXT}, ctrl=True))
    put(c.decider([(S["POS"], ">", S["N"], "g", "g"), (S["RESET"], "=", 0, "g")], [(EV, True, 0, "r")],
                  {"r": PRED, "g": CTRL}, {"r": NXT}, ctrl=True))
    _cell(c, put, NXT, TOKC, CMT)
    put(c.decider([(S["RESET"], "=", 0, "g")], [(EV, True, 0, "r")], {"r": NXT, "g": CTRL}, {"r": SCRD}, ctrl=True))
    # embedding input = token one-hot + position one-hot; attention valid mask
    put(c.arith(E, "+", 0, E, {"r": TOKC}, {"r": EMBIN}, ctrl=True))
    put(c.const([(SP[p], p + 1) for p in range(T)], PCONST), side=True)
    put(c.decider([(E, "=", S["POS1"], "r", "g")], [(E, False, 1, "r")], {"r": PCONST, "g": CTRL}, {"r": EMBIN}, ctrl=True))
    put(c.decider([(E, "<=", S["POS1"], "r", "g")], [(E, False, 1, "r")], {"r": PCONST, "g": CTRL}, {"g": MASK}, ctrl=True))
    m.close()
    info["reset"], info["commit"], info["ph_mod"] = reset.id, commit.id, ph_mod.id

    # ---------------------------------------------------------------- keyboard
    keys = []
    def key_parts(k):
        kn = c.net("red", f"KEY{k}")
        key = c.const([], kn, tag=f"KEY{k}")
        nd = c.decider([(_s(ANYTHING), "!=", 0, "r")], [(S["N"], False, 1, "rg")], {"r": kn}, {"g": CTRL}, ctrl=True)
        gt = c.decider([(S["POS"], "=", k - 1, "g"), (S["RESET"], "=", 0, "g")], [(EV, True, 0, "r")],
                       {"r": kn, "g": CTRL}, {"r": NXT}, ctrl=True)
        keys.append(key.id)
        return key, nd, gt
    if ui == "columns":
        m = c.module("keyboard")
        for k in range(1, kb + 1):
            col, row = divmod(k - 1, H // 2)
            key, nd, gt = key_parts(k)
            m.put(key, col, 2 * row, side=True)
            m.put(nd, col, 2 * row)
            m.put(gt, col, 2 * row + 1)
        m.close()
    else:
        def kb_item(k, x, y):
            if k == 0:
                c.place_free(reset, x + 0.5, y + 0.5, tag="RESET")
                return
            key, nd, gt = key_parts(k)
            c.place_free(key, x + 0.5, y + 0.5, tag=f"KEY{k}")
            c.place_free(nd, x + 0.5, y + 2.0, "north", "keyboard")
            c.place_free(gt, x + 0.5, y + 4.0, "north", "keyboard")
        c.text_region(x0, y0 - 8, kb + 1, 32, 6, kb_item, "keyboard")
    info["keys"] = keys

    # ---------------------------------------------------------------- embedding
    X = c.net("red", "X0")
    m = c.module("embedding")
    for i in range(d):
        w = c.net("green")
        sig = [(SV[v], P["E"][v][i]) for v in range(V)] + [(SP[p], P["P"][p][i]) for p in range(T)]
        col, row = divmod(i, H)
        m.put(c.const(sig, w), col, row, side=True)
        m.put(c.arith(E, "*", E, SD[i], {"r": EMBIN, "g": w}, {"r": X}, fn="r", sn="g"), col, row)
    m.close()

    # ---------------------------------------------------------------- layers
    kv_deciders = []
    for l, Ly in enumerate(P["layers"]):
        N1 = _rmsnorm(c, X, d, f"L{l}.norm1", S)
        # q, k, v
        QR = [c.net("red", f"L{l}.QR{hh}") for hh in range(h)]
        Q = [c.net("green", f"L{l}.Q{hh}") for hh in range(h)]
        KR, KD = c.net("red", f"L{l}.KR"), c.net("red", f"L{l}.KD")
        VR, VD = c.net("red", f"L{l}.VR"), c.net("red", f"L{l}.VD")
        m = c.module(f"L{l}.qkv")
        j = 0
        for which, W, outs in (("q", Ly["wq"], None), ("k", Ly["wk"], KR), ("v", Ly["wv"], VR)):
            for i in range(d):
                out = QR[i // dh] if which == "q" else outs
                _neuron(c, m, j, H, W[i], SD, SD[i], N1, out)
                j += 1
        col = (j + H - 1) // H
        r = 0
        for hh in range(h):
            m.put(c.arith(E, ">>", Ly["sq"], E, {"r": QR[hh]}, {"g": Q[hh]}), col, r); r += 1
        m.put(c.arith(E, ">>", Ly["sk"], E, {"r": KR}, {"r": KD}), col, r); r += 1
        m.put(c.arith(E, ">>", Ly["sv"], E, {"r": VR}, {"r": VD}), col, r); r += 1
        m.close()

        # attention matrix: one row per position
        SR = [c.net("red", f"L{l}.SR{hh}") for hh in range(h)]
        EB = [c.net("green", f"L{l}.EB{hh}") for hh in range(h)]
        OR = [c.net("red", f"L{l}.OR{hh}") for hh in range(h)]
        m = c.module(f"L{l}.attn")
        for p in range(T):
            KC, VC = c.net("red", f"L{l}.KC{p}"), c.net("red", f"L{l}.VC{p}")
            WKn, WVn = c.net("green"), c.net("green")
            for wnet, col in ((WKn, 0), (WVn, 3 + h)):
                kd = c.decider([(S["PHASE"], "=", 0, "g"), (S["POS"], "=", p, "g"), (S["RUN"], ">", 0, "g")],
                               [(S["W"], False, 1, "rg")], {"g": CTRL}, {"g": wnet}, ctrl=True)
                kv_deciders.append((l, kd.id))
                m.put(kd, col, p)
            m.put(c.decider([(S["W"], ">", 0, "g")], [(EV, True, 0, "r")], {"r": KD, "g": WKn}, {"r": KC}, mem=True), 1, p)
            m.put(c.decider([(S["W"], "=", 0, "g")], [(EV, True, 0, "r")], {"r": KC, "g": WKn}, {"r": KC}, mem=True), 2, p)
            for hh in range(h):
                m.put(c.arith(E, "*", E, SP[p], {"r": KC, "g": Q[hh]}, {"r": SR[hh]}, fn="r", sn="g"), 3 + hh, p)
            m.put(c.decider([(S["W"], ">", 0, "g")], [(EV, True, 0, "r")], {"r": VD, "g": WVn}, {"r": VC}, mem=True), 4 + h, p)
            m.put(c.decider([(S["W"], "=", 0, "g")], [(EV, True, 0, "r")], {"r": VC, "g": WVn}, {"r": VC}, mem=True), 5 + h, p)
            for hh in range(h):
                m.put(c.arith(E, "*", SP[p], E, {"r": VC, "g": EB[hh]}, {"r": OR[hh]}, fn="r", sn="g"), 6 + h + hh, p)
        m.close()

        # softmax + weighted average, one column per head
        ATT = c.net("red", f"L{l}.ATT")
        m = c.module(f"L{l}.softmax")
        per_head = (len(EXP.th) + 8 + H - 1) // H
        for hh in range(h):
            q = Seq(H, hh * per_head)
            SS, SM, MXS = c.net("red"), c.net("red"), c.net("red")
            MXN, SH, ZN, OD = c.net("green"), c.net("red"), c.net("green"), c.net("red")
            HM = c.net("green")
            cr = q()
            m.put(c.arith(E, ">>", spec.QK_SHIFT, E, {"r": SR[hh]}, {"r": SS}), *cr)
            m.put(c.const([(SP[p], spec.SCORE_BIG) for p in range(T)], SR[hh]), *cr, side=True)
            m.put(c.decider([(E, ">", 0, "g")], [(E, True, 0, "r")], {"r": SS, "g": MASK}, {"r": SM}), *q())
            m.put(c.selector({"operation": "select", "select_max": True, "index_constant": 0}, {"r": SM}, {"r": MXS}), *q())
            cr = q()
            m.put(c.arith(E, "+", 0, S["MAX"], {"r": MXS}, {"g": MXN}), *cr)
            m.put(c.const([(S["MAX"], -spec.EXP_R)], MXN), *cr, side=True)
            m.put(c.arith(E, "-", S["MAX"], E, {"r": SM, "g": MXN}, {"r": SH}, fn="r", sn="g"), *q())
            for t, dl in zip(EXP.th, EXP.delta):
                m.put(c.decider([(E, ">", t, "r")], [(E, False, dl, "g")], {"r": SH}, {"g": EB[hh]}), *q())
            m.put(c.arith(E, "+", 0, S["Z"], {"g": EB[hh]}, {"g": ZN}, fn="g"), *q())
            m.put(c.arith(E, "/", S["Z"], E, {"r": OR[hh], "g": ZN}, {"r": OD}, fn="r", sn="g"), *q())
            cr = q()
            m.put(c.arith(E, "*", E, E, {"r": OD, "g": HM}, {"r": ATT}, fn="r", sn="g"), *cr)
            m.put(c.const([(SD[i], 1) for i in range(hh * dh, (hh + 1) * dh)], HM), *cr, side=True)
        m.close()

        # output projection + residual
        X1 = c.net("red", f"L{l}.X1")
        AOR = c.net("red")
        m = c.module(f"L{l}.wo")
        for i in range(d):
            _neuron(c, m, i, H, Ly["wo"][i], SD, SD[i], ATT, AOR)
        col = (d + H - 1) // H
        m.put(c.arith(E, ">>", Ly["so"], E, {"r": AOR}, {"r": X1}), col, 0)
        m.put(c.arith(E, "+", 0, E, {"r": X}, {"r": X1}), col, 1)
        m.close()

        # MLP
        N2 = _rmsnorm(c, X1, d, f"L{l}.norm2", S)
        HR, HS, HRL = c.net("red"), c.net("red"), c.net("red", f"L{l}.H")
        m = c.module(f"L{l}.w1")
        for i in range(ff):
            _neuron(c, m, i, H, Ly["w1"][i], SD, SD[i], N2, HR)
        col = (ff + H - 1) // H
        m.put(c.arith(E, ">>", Ly["s1"], E, {"r": HR}, {"r": HS}), col, 0)
        m.put(c.decider([(E, ">", 0, "r")], [(E, True, 0, "r")], {"r": HS}, {"r": HRL}), col, 1)
        m.close()
        X2, MR = c.net("red", f"L{l}.X2"), c.net("red")
        m = c.module(f"L{l}.w2")
        for i in range(d):
            _neuron(c, m, i, H, Ly["w2"][i], SD, SD[i], HRL, MR)
        col = (d + H - 1) // H
        m.put(c.arith(E, ">>", Ly["s2"], E, {"r": MR}, {"r": X2}), col, 0)
        m.put(c.arith(E, "+", 0, E, {"r": X1}, {"r": X2}), col, 1)
        m.close()
        X = X2

    # ---------------------------------------------------------------- head
    NF = _rmsnorm(c, X, d, "final_norm", S)
    LR, LS, PS = c.net("red", "LR"), c.net("red", "LS"), c.net("red", "PS")
    m = c.module("head")
    for v in range(V):
        _neuron(c, m, v, H, P["head"][v], SD, SV[v], NF, LR)
    col = (V + H - 1) // H
    if tk:
        # unique values so the max is never tied: (logit << TIE_SHIFT) + LOGIT_BIG_T + (V-1-id)
        LS1 = c.net("red")
        m.put(c.arith(E, ">>", P["shead"], E, {"r": LR}, {"r": LS1}), col, 0)
        m.put(c.arith(E, "<<", spec.TIE_SHIFT, E, {"r": LS1}, {"r": LS}), col, 1)
        m.put(c.const([(SV[v], spec.LOGIT_BIG_T + V - 1 - v) for v in range(V)], LS), col, 1, side=True)
        m.put(c.selector({"operation": "select", "select_max": True, "index_constant": 0}, {"r": LS}, {"r": PS}), col, 2)
        m.put(c.decider([(E, "!=", 0, "r")], [(E, False, 1, "r")], {"r": PS}, {"r": PRED}), col, 3)
    else:
        m.put(c.arith(E, ">>", P["shead"], E, {"r": LR}, {"r": LS}), col, 0)
        m.put(c.const([(SV[v], spec.LOGIT_BIG) for v in range(V)], LS), col, 0, side=True)
        m.put(c.selector({"operation": "select", "select_max": True, "index_constant": 0}, {"r": LS}, {"r": PS}), col, 1)
        m.put(c.decider([(E, "!=", 0, "r")], [(E, False, 1, "r")], {"r": PS}, {"r": PRED}), col, 2)
    m.close()

    # ---------------------------------------------------------------- detokenizer (tokenizer builds)
    if tk:
        from .tokenizer import LMAX
        NS = cfg.screen
        SB = [_s(s) for s in spec.bus_signals(NS)]
        BUS, POSCODE = c.net("red", "BUS"), c.net("red", "POSCODE")
        m = c.module("detok")
        q = Seq(H)
        # LEN = letters in the committed token (SCRD holds it as a one-hot token signal)
        ROML = c.net("green")
        cr = q()
        m.put(c.arith(E, "*", E, S["LEN"], {"r": SCRD, "g": ROML}, {"g": LENN}, fn="r", sn="g"), *cr)
        m.put(c.const([(SV[t], len(tk.tokens[t])) for t in range(V)], ROML), *cr, side=True)
        for j in range(LMAX):
            # letter j of the token as a code 1..42, sent to screen cell CUR + j on the bus
            ROMj, CODEj, TGTj, OHj = c.net("green"), c.net("green"), c.net("green"), c.net("red")
            cr = q()
            m.put(c.arith(E, "*", E, S["CODE"], {"r": SCRD, "g": ROMj}, {"g": CODEj}, fn="r", sn="g"), *cr)
            m.put(c.const([(SV[t], spec.VOCAB[tk.tokens[t][j]] + 1) for t in range(V) if len(tk.tokens[t]) > j], ROMj),
                  *cr, side=True)
            cr = q()
            m.put(c.arith(S["CUR"], "+", j + 1, S["TGT"], {"g": CTRL}, {"g": TGTj}), *cr)
            if j == 0:
                m.put(c.const([(SB[k], k + 1) for k in range(NS)], POSCODE), *cr, side=True)
            m.put(c.decider([(E, "=", S["TGT"], "r", "g")], [(E, False, 1, "r")], {"r": POSCODE, "g": TGTj}, {"r": OHj}), *q())
            m.put(c.arith(E, "*", S["CODE"], E, {"r": OHj, "g": CODEj}, {"r": BUS}, fn="r", sn="g"), *q())
        m.close()

    # ---------------------------------------------------------------- screen
    screen, screen_w = [], []
    def screen_parts(p):
        SC, WS = c.net("red", f"SC{p}"), c.net("green")
        wd = c.decider([(S["PHASE"], "=", 0, "g"), (S["RUN"], ">", 0, "g"), (S["POS"], "=", p - 1, "g"),
                        (S["RESET"], ">", 0, "g", "rg", "or")], [(S["W"], False, 1, "rg")], {"g": CTRL}, {"g": WS}, ctrl=True)
        g = c.decider([(S["W"], ">", 0, "g")], [(EV, True, 0, "r")], {"r": SCRD, "g": WS}, {"r": SC}, mem=True)
        hh = c.decider([(S["W"], "=", 0, "g")], [(EV, True, 0, "r")], {"r": SC, "g": WS}, {"r": SC}, mem=True)
        pn = c.panel(SC, tag=f"SCREEN{p}")
        screen.append(pn.id)
        screen_w.append(wd.id)
        return wd, g, hh, pn
    if ui == "columns":
        m = c.module("screen")
        for p in range(1, T):
            col, row = divmod(p - 1, H)
            col *= 4
            wd, g, hh, pn = screen_parts(p)
            m.put(wd, col, row)
            m.put(g, col + 1, row)
            m.put(hh, col + 2, row)
            m.put(pn, col + 3, row, side=True)
        m.close()
    info["screen"] = screen

    if tk:
        def letter_cell(k):
            SC, WS = c.net("red", f"SC{k}"), c.net("green")
            wd = c.decider([(S["PHASE"], "=", 0, "g"), (S["RUN"], ">", 0, "g"), (SB[k], "!=", 0, "r"),
                            (S["RESET"], ">", 0, "g", "rg", "or")], [(S["W"], False, 1, "rg")],
                           {"r": BUS, "g": CTRL}, {"g": WS}, ctrl=True)
            g = c.decider([(S["W"], ">", 0, "g")], [(SB[k], True, 0, "r")], {"r": BUS, "g": WS}, {"r": SC}, mem=True)
            hh = c.decider([(S["W"], "=", 0, "g")], [(EV, True, 0, "r")], {"r": SC, "g": WS}, {"r": SC}, mem=True)
            recs = [{"condition": {"first_signal": SB[k], "comparator": "=", "constant": i + 1},
                     "icon": _s(spec.VOCAB_SIGNALS[i]), "text": ""} for i in range(spec.V)]
            pn = c.panel(SC, records=recs, tag=f"SCREEN{k}")
            screen.append(pn.id)
            screen_w.append(wd.id)
            return wd, g, hh, pn
        gb = c.grid_bbox()
        def tsc_item(k, x, y):
            wd, g, hh, pn = letter_cell(k)
            c.place_free(pn, x + 0.5, y + 0.5, tag=f"SCREEN{k}")
            c.place_free(wd, x + 0.5, y + 2.0, "north", "screen")
            c.place_free(g, x + 0.5, y + 4.0, "north", "screen")
            c.place_free(hh, x + 0.5, y + 6.0, "north", "screen")
        per_line = 64 if T >= 256 else 48 if T >= 128 else 32
        c.text_region(x0, gb[3] + 1, NS, per_line, 8, tsc_item, "screen")
        info["screen"] = screen
        info["tok"] = cfg.tok
    elif ui != "columns":
        gb = c.grid_bbox()
        def sc_item(k, x, y):
            wd, g, hh, pn = screen_parts(k + 1)
            c.place_free(pn, x + 0.5, y + 0.5, tag=f"SCREEN{k + 1}")
            c.place_free(wd, x + 0.5, y + 2.0, "north", "screen")
            c.place_free(g, x + 0.5, y + 4.0, "north", "screen")
            c.place_free(hh, x + 0.5, y + 6.0, "north", "screen")
        c.text_region(x0, gb[3] + 1, T - 1, 32, 8, sc_item, "screen")
        info["screen"] = screen

    # ---------------------------------------------------------------- timing
    t_kv, period = _timing(c, cfg, extra=10 if tk else 0)
    for l, eid in kv_deciders:
        c.ents[eid].params["conditions"][0]["constant"] = t_kv[l]
    for eid in screen_w:
        c.ents[eid].params["conditions"][0]["constant"] = period - 1
    c.ents[commit.id].params["conditions"][0]["constant"] = period - 1
    c.ents[ph_mod.id].params["second_constant"] = period
    info["t_kv"], info["period"] = t_kv, period
    c.add_power()
    info["route"] = c.route()
    return c, info


def _s(s):
    return tuple(s)


class Seq:
    """Sequential (col, row) positions inside a module, spilling into the next column."""
    def __init__(self, H, col0=0):
        self.H, self.i, self.col0 = H, 0, col0

    def __call__(self):
        col, row = divmod(self.i, self.H)
        self.i += 1
        return self.col0 + col, row


def _cell(c, put, data, out, wnet):
    W = _s(C["W"])
    put(c.decider([(W, ">", 0, "g")], [(_s(EVERYTHING), True, 0, "r")], {"r": data, "g": wnet}, {"r": out}, mem=True))
    put(c.decider([(W, "=", 0, "g")], [(_s(EVERYTHING), True, 0, "r")], {"r": out, "g": wnet}, {"r": out}, mem=True))


def _neuron(c, m, j, H, wrow, SIN, sig_out, inp, out):
    w = c.net("green")
    col, row = divmod(j, H)
    m.put(c.const([(SIN[i], int(x)) for i, x in enumerate(wrow)], w), col, row, side=True)
    m.put(c.arith(_s(EACH), "*", _s(EACH), sig_out, {"r": inp, "g": w}, {"r": out}, fn="r", sn="g"), col, row)


def _rmsnorm(c, X, d, name, S):
    E = _s(EACH)
    XS, SQ, MSN, INVN, NM, N = (c.net("red"), c.net("red"), c.net("red"), c.net("green"), c.net("red"),
                                c.net("red", name + ".out"))
    m = c.module(name)
    q = Seq(c.H)
    m.put(c.arith(E, ">>", spec.NORM_PRE, E, {"r": X}, {"r": XS}), *q())
    m.put(c.arith(E, "^", 2, S["S2"], {"r": XS}, {"r": SQ}), *q())
    cr = q()
    m.put(c.arith(S["S2"], "/", d, S["MS"], {"r": SQ}, {"r": MSN}), *cr)
    m.put(c.const([(S["INV"], RSQRT.base)], INVN), *cr, side=True)
    for t, dl in zip(RSQRT.th, RSQRT.delta):
        m.put(c.decider([(S["MS"], ">=", t, "r")], [(S["INV"], False, dl, "rg")], {"r": MSN}, {"g": INVN}), *q())
    m.put(c.arith(E, "*", S["INV"], E, {"r": X, "g": INVN}, {"r": NM}, fn="r", sn="g"), *q())
    m.put(c.arith(E, ">>", spec.INV_SHIFT, E, {"r": NM}, {"r": N}), *q())
    m.close()
    return N


def _timing(c: Circuit, cfg, extra: int = 0):
    """Settle time (ticks after the commit pulse) of each net, by longest path through the
    combinational datapath. Memory cells and controller combinators are fixed-time sources."""
    producers = {}
    for e in c.ents:
        for conn, nid in e.conns.items():
            if conn.startswith("out") or (e.kind == "const"):
                producers.setdefault(nid, []).append(e)
    named = {n.name: n.id for n in c.nets if n.name}
    fixed = {named["CTRL"]: 4, named["TOKC"]: 1, named["POSC"]: 1, named["DONEC"]: 1}
    t_kv = {}
    memo = {}

    def settle(nid, stack=()):
        if nid in memo:
            return memo[nid]
        if nid in fixed:
            return fixed[nid]
        assert nid not in stack, f"combinational loop through net {c.nets[nid].name}"
        best = 0
        for e in producers.get(nid, []):
            if e.kind == "const":
                continue
            if e.mem:
                net = c.nets[nid].name
                if ".KC" in net or ".VC" in net:
                    l = int(net.split(".")[0][1:])
                    best = max(best, t_kv[l] + 2)
                continue
            if e.ctrl:
                best = max(best, 5)
                continue
            ins = [n for k, n in e.conns.items() if k.startswith("in")]
            best = max(best, 1 + max((settle(n, stack + (nid,)) for n in ins), default=0))
        memo[nid] = best
        return best

    for l in range(cfg.L):
        t = max(settle(named[f"L{l}.KD"]), settle(named[f"L{l}.VD"])) - 1
        t_kv[l] = max(t, 6)
    # extra: tokenizer builds run the detokenizer and CUR update off the next token (NXT is PRED + 1 tick):
    # SCRD +1, ROM lookups +1, bus / CUR+LEN +1, CUR gate +1, then the same 3-tick commit margin, plus slack.
    period = max(settle(named["PRED"]) + 4 + extra, 12)
    return t_kv, period
