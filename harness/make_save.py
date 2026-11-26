"""Make the playable save, game/factoriogpt.zip, from the two blueprints in game/.

  python harness/make_save.py

A flat lab world with the GPT and the in-game trainer pasted from their blueprints and powered. The GPT holds RESET,
ready for a prompt, and the player spawns at its keyboard. The save is then checked: the GPT's replies against the
integer simulator, and the trainer's weights after a few SGD steps against the integer reference.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))

from cbuild import attach, ent_lua, paste_and_locate, set_reset  # noqa: E402
from fio import Factorio  # noqa: E402
from test_gpt import load, run_prompt  # noqa: E402
from test_trainer import read_nets, set_const, weights_from_game  # noqa: E402

from fgpt import intsim  # noqa: E402
from fgpt import trainsim as ts  # noqa: E402
from fgpt.gpt_circuit import build_gpt  # noqa: E402
from fgpt.train_circuit import build_trainer  # noqa: E402

GAME = ROOT / "game"
GAP = 80   # tiles between the GPT and the trainer


def place(f, bp, circ, at):
    """Paste a blueprint centred at `at` and map the circuit's entities onto it. Returns the offset.
    (The build is found by its clock: the one nearest where this build's clock lands, not the other build's.)"""
    x0, y0, x1, y1 = circ.bbox()
    clock = next(e for e in circ.ents if e.kind == "selector" and e.params.get("operation") == "time")
    near = (at[0] + clock.pos[0] - (x0 + x1) / 2, at[1] + clock.pos[1] - (y0 + y1) / 2)
    built, ox, oy = paste_and_locate(f, str(bp), circ, at=at, near=near)
    missing = attach(f, circ, ox, oy)
    print(f"  {bp.name}: {built} entities, {missing} circuit entities unmatched", flush=True)
    if missing:
        raise RuntimeError("the blueprint does not match the generated circuit")
    return ox, oy


def main():
    cfg, P = load("v3t")
    gpt, ginfo = build_gpt(P, cfg)
    P0 = ts.init(0)
    trn, tinfo = build_trainer(P0)
    gx0, gy0, gx1, gy1 = gpt.bbox()
    tx0, ty0, tx1, ty1 = trn.bbox()
    g_at = (0, 0)
    t_at = (0, round((gy1 - gy0) / 2 + GAP + (ty1 - ty0) / 2))

    f = Factorio("bench", rcon_port=27116, game_port=34298)
    f.stop(save=False)
    f.start("bench", fresh=True, lab=True)
    f.lua("game.tick_paused = true; return true")
    for (x0, y0, x1, y1), (cx, cy) in (((gx0, gy0, gx1, gy1), g_at), ((tx0, ty0, tx1, ty1), t_at)):
        rad = int(max(x1 - x0, y1 - y0) / 64) + 2
        f.lua(f"S.request_to_generate_chunks({{{cx}, {cy}}}, {rad}); S.force_generate_chunk_requests(); return true")

    print("pasting", flush=True)
    to = place(f, GAME / "factoriogpt_trainer.txt", trn, t_at)
    # the trainer's blueprint has no power source of its own: an energy interface beside one of its substations
    sub = next(e for e in trn.ents if e.kind == "substation")
    sx, sy = sub.pos[0] + to[0], sub.pos[1] + to[1]
    f.lua(f"""local p = S.find_non_colliding_position("electric-energy-interface", {{{sx}, {sy}}}, 8, 1)
      local e = S.create_entity{{name="electric-energy-interface", position=p, force=F}}
      e.power_production = 1e12; e.electric_buffer_size = 1e12; return true""")
    go = place(f, GAME / "factoriogpt.txt", gpt, g_at)

    # the GPT waits for a prompt (RESET held), and the player starts at its keyboard
    set_reset(f, ginfo, True)
    f.lua(f"""local k = {ent_lua(ginfo['keys'][0])}.position
      F.set_spawn_position({{k.x + 8, k.y - 6}}, S); return true""")
    f.lua("game.tick_paused = false; return true")
    f.run_ticks(120)
    f.lua("storage.fg = nil; storage.bp = nil; game.speed = 1; game.tick_paused = false; return true")
    f.save("factoriogpt")
    print("saved", flush=True)

    # check the save as it was written
    attach(f, gpt, *go)
    ok = 0
    prompts = ["TELL ME A JOKE", "HI"]
    for q in prompts:
        want = intsim.generate(P, cfg, q)
        got = run_prompt(f, ginfo, cfg, q)
        ok += got == want
        print(f"  {q!r}: {'MATCH' if got == want else 'MISMATCH'}  sim: {want}", flush=True)

    attach(f, trn, *to)
    K = 12
    set_const(f, tinfo["train"], False)
    set_const(f, tinfo["init"], True)
    f.run_ticks(20)
    set_const(f, tinfo["init"], False)
    f.run_ticks(5)
    set_const(f, tinfo["train"], True)
    seq = lambda: read_nets(f, trn, ["SEQC"])["SEQC"].get("virtual:signal-Q", 0)
    done, last = 0, seq()
    while done < K:
        f.run_ticks(tinfo["period"] // 2)
        cur = seq()
        if cur != last:
            done, last = done + 1, cur
    Wg = weights_from_game(f, trn, P0)
    Pref = ts.init(0)
    for i in range(K):
        _, _, g = ts.step(Pref, ts.SEQS[i % len(ts.SEQS)])
        ts.update(Pref, g)
    bad = sum(int((Wg[k] != v).sum()) for k, v in Pref.items())
    print(f"  trainer: {bad} of {sum(v.size for _, v in Pref.items())} weights differ after {K} steps", flush=True)
    f.stop(save=False)

    if ok == len(prompts) and bad == 0:
        shutil.copy(f.saves / "factoriogpt.zip", GAME / "factoriogpt.zip")
        print(f"wrote game/factoriogpt.zip ({(GAME / 'factoriogpt.zip').stat().st_size / 1e6:.1f} MB)")
    else:
        sys.exit("checks failed: game/factoriogpt.zip not written")


if __name__ == "__main__":
    main()
