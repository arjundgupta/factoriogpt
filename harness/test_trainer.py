"""Build the in-game trainer on the bench server and check it against fgpt.trainsim, bit for bit.

  python harness/test_trainer.py [K]
Loads the initial weights (INIT), trains K steps (TRAIN), then compares every weight row and the
model's predictions with the integer reference after the same K SGD steps.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))

from cbuild import CONN, build  # noqa: E402
from fio import Factorio  # noqa: E402

from fgpt import spec  # noqa: E402
from fgpt import trainsim as ts  # noqa: E402
from fgpt.train_circuit import build_trainer  # noqa: E402


def set_const(f, eid, on):
    return f.lua(f"storage.fg[{eid}].get_or_create_control_behavior().enabled = {str(on).lower()}; return true")


def net_reader(circ, name):
    n = circ.named[name]
    for eid, conn in n.members:
        return eid, CONN[conn]


def read_nets(f, circ, names):
    req = [[nm, *net_reader(circ, nm)] for nm in names]
    r = f.lua(f"""local data = helpers.json_to_table([==[{json.dumps(req)}]==])
      local out = {{}}
      for _, it in ipairs(data) do
        local n = storage.fg[it[2]].get_circuit_network(defines.wire_connector_id[it[3]])
        local m = {{}}
        if n and n.signals then for _, s in ipairs(n.signals) do m[(s.signal.type or "item") .. ":" .. s.signal.name] = s.count end end
        out[it[1]] = m
      end
      return out""")
    return r["res"]


def weights_from_game(f, circ, Pref: ts.Params):
    TS = ts.tsignals()
    SP = spec.pos_signals(ts.T + 1)
    SD = spec.dim_signals(max(ts.D, ts.FF, ts.VT + ts.T))
    in_sig = {"E": TS + SP[: ts.T]}
    names = []
    for name, W in Pref.items():
        names += [f"{name}.MW{j}" for j in range(W.shape[0])]
    vals = {}
    for i in range(0, len(names), 60):
        vals.update(read_nets(f, circ, names[i:i + 60]))
    out = {}
    for name, W in Pref.items():
        sig = in_sig.get(name, SD)
        M = np.zeros_like(W)
        for j in range(W.shape[0]):
            row = vals.get(f"{name}.MW{j}") or {}
            for i in range(W.shape[1]):
                M[j, i] = row.get(f"{sig[i][0]}:{sig[i][1]}", 0)
        out[name] = M
    return out


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    P0 = ts.init(0)
    circ, info = build_trainer(P0)
    period = info["period"]
    print(f"trainer: {len(circ.ents)} entities, period {period}, route {info['route']}")
    f = Factorio("bench", rcon_port=27116, game_port=34298)
    f.stop(save=False)
    f.start("bench", fresh=True, lab=True)
    f.lua("game.tick_paused = true; return true")
    build(f, circ, 0, 0)
    set_const(f, info["train"], False)
    set_const(f, info["init"], True)
    f.run_ticks(20)
    set_const(f, info["init"], False)
    f.run_ticks(5)
    W = weights_from_game(f, circ, P0)
    bad = {k: int((W[k] != v).sum()) for k, v in P0.items()}
    print("after INIT, mismatching weights per matrix:", bad)
    set_const(f, info["train"], True)
    seq = lambda: read_nets(f, circ, ["SEQC"])["SEQC"].get("virtual:signal-Q", 0)
    t0 = time.time()
    done, last = 0, seq()
    while done < K:
        f.run_ticks(period // 2)
        cur = seq()
        if cur != last:
            done += 1
            last = cur
    print(f"{K} steps in game ({time.time() - t0:.1f}s)")
    Wg = weights_from_game(f, circ, P0)
    Pref = ts.init(0)
    for i in range(K):
        _, _, g = ts.step(Pref, ts.SEQS[i % len(ts.SEQS)])
        ts.update(Pref, g)
    bad = {k: int((Wg[k] != v).sum()) for k, v in Pref.items()}
    print("after training, mismatching weights per matrix:", bad)
    if any(bad.values()):
        for k, v in Pref.items():
            if bad[k]:
                idx = np.argwhere(Wg[k] != v)[:3]
                print(" ", k, [(tuple(i), int(Wg[k][tuple(i)]), int(v[tuple(i)])) for i in idx])
    return f, circ, info


if __name__ == "__main__":
    main()
