"""Build a trained FactorioGPT on the bench server and check the game against the integer simulator.

  python harness/test_gpt.py v3t "HI" "WHAT IS YOUR NAME?"
"""
from __future__ import annotations

import gzip
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))

from cbuild import build, read_screen, set_reset, type_prompt  # noqa: E402
from fio import Factorio  # noqa: E402

from fgpt import intsim, spec  # noqa: E402
from fgpt.gpt_circuit import build_gpt  # noqa: E402


def load(name):
    out = ROOT / "train" / "out" / name
    p = out / "int_params.pkl"   # (train.py writes it; the repo ships it gzipped)
    P = pickle.load(open(p, "rb")) if p.exists() else pickle.load(gzip.open(str(p) + ".gz", "rb"))
    return spec.Config.load(out / "config.json"), P


def run_prompt(f, info, cfg, prompt, speed=100, log=print):
    set_reset(f, info, True)
    type_prompt(f, info, prompt)
    f.run_ticks(10)
    set_reset(f, info, False)
    f.lua(f"game.speed = {speed}; game.tick_paused = false; return true")
    # Wait until DONE, or until the context is full.
    t0 = time.time()
    limit = cfg.T * info["period"] + 200
    start = f.tick()
    while f.tick() - start < limit:
        time.sleep(0.25)
        txt = read_screen(f, info)
        if len(txt) > len(prompt) + 3 and txt.endswith(spec.USER):
            break
    f.lua("game.speed = 1; game.tick_paused = true; return true")
    txt = read_screen(f, info)
    log(f"  game ({f.tick() - start} ticks, {time.time() - t0:.1f}s): {txt}")
    return txt


def main():
    name = sys.argv[1]
    prompts = sys.argv[2:] or ["HI"]
    cfg, P = load(name)
    circ, info = build_gpt(P, cfg)
    print(f"{name}: {len(circ.ents)} entities, period {info['period']}, t_kv {info['t_kv']}, route {info['route']}")
    f = Factorio("bench", rcon_port=27116, game_port=34298)
    f.stop(save=False)
    f.start("bench", fresh=True, lab=True)
    f.lua("game.tick_paused = true; return true")
    build(f, circ, 0, 0)
    f.run_ticks(30)
    ok = 0
    for prompt in prompts:
        want = intsim.generate(P, cfg, prompt)
        print(f"prompt {prompt!r}\n  sim : {want}")
        got = run_prompt(f, info, cfg, prompt)
        ok += got == want
        print("  MATCH" if got == want else "  MISMATCH")
    print(f"{ok}/{len(prompts)} prompts match")
    return f, circ, info


if __name__ == "__main__":
    main()
