"""Export a model's blueprint from a bench build, paste it into a fresh bench world, and check chat output.

  python harness/paste_test.py v3t "HI" "TELL ME A JOKE"
Writes game/factoriogpt.txt (other models: game/factoriogpt_<model>.txt).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))

from cbuild import attach, build, export_blueprint, paste_and_locate  # noqa: E402
from fio import Factorio  # noqa: E402
from test_gpt import load, run_prompt  # noqa: E402

from fgpt import intsim, spec  # noqa: E402
from fgpt.gpt_circuit import build_gpt  # noqa: E402


def main():
    model, prompts = sys.argv[1], [a for a in sys.argv[2:] if not a.startswith("--")] or ["HI"]
    cfg, P = load(model)
    circ, info = build_gpt(P, cfg)
    f = Factorio("bench", rcon_port=27116, game_port=34298)
    bp = ROOT / "game" / ("factoriogpt.txt" if model == "v3t" else f"factoriogpt_{model}.txt")
    x0, y0, x1, y1 = circ.bbox()
    if "--paste-only" not in sys.argv:
        f.stop(save=False)
        f.start("bench", fresh=True, lab=True)
        f.lua("game.tick_paused = true; return true")
        build(f, circ, 0, 0)
        t = time.time()
        r = export_blueprint(f, (x0 - 5, y0 - 2, x1 + 2, y1 + 2), 0, 0, f"FactorioGPT {model}", str(bp))
        print(f"exported {r['res']} to {bp.name} ({bp.stat().st_size / 1e6:.1f} MB, {time.time() - t:.0f}s)", flush=True)
    f.stop(save=False)
    f.start("bench", fresh=True, lab=True)
    # big builds reach past the lab's generated area: generate chunks around the whole footprint first
    cx, cy = round((x0 + x1) / 2), round((y0 + y1) / 2)
    rad = int(max(x1 - x0, y1 - y0) / 64) + 2
    f.lua(f"game.tick_paused = true; S.request_to_generate_chunks({{{cx}, {cy}}}, {rad}); S.force_generate_chunk_requests(); return true")
    t = time.time()
    built, ox, oy = paste_and_locate(f, str(bp), circ, at=(cx, cy))
    missing = attach(f, circ, ox, oy)
    print(f"pasted {built} entities, offset ({ox}, {oy}), unmatched circuit entities: {missing} ({time.time() - t:.0f}s)", flush=True)
    f.run_ticks(30)
    ok = 0
    for q in prompts:
        want = intsim.generate(P, cfg, q)
        got = run_prompt(f, info, cfg, q)
        ok += got == want
        print(f"  {q!r}: {'MATCH' if got == want else 'MISMATCH'}  sim: {want}", flush=True)
    print(f"{ok}/{len(prompts)} prompts match after paste")
    f.stop(save=False)


if __name__ == "__main__":
    main()
