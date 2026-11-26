"""Build a generated circuit (fgpt.circuit.Circuit) in a Factorio world over RCON, and drive it.

Lua is used only to place, wire and inspect entities in the test world; the circuit itself is vanilla.
"""
from __future__ import annotations

import json
import time

from fio import Factorio

NAMES = {"arith": "arithmetic-combinator", "decider": "decider-combinator", "selector": "selector-combinator",
         "const": "constant-combinator", "panel": "display-panel", "lamp": "small-lamp", "pole": "big-electric-pole",
         "substation": "substation", "eei": "electric-energy-interface"}
CONN = {"in_r": "combinator_input_red", "in_g": "combinator_input_green", "out_r": "combinator_output_red",
        "out_g": "combinator_output_green", "r": "circuit_red", "g": "circuit_green"}


def _conv(x):
    if isinstance(x, tuple) and len(x) == 2 and all(isinstance(v, str) for v in x):
        return {"type": x[0], "name": x[1]}
    if isinstance(x, dict):
        return {k: _conv(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_conv(v) for v in x]
    return x


BUILD_LUA = r"""
storage.fg = storage.fg or {}
local data = helpers.json_to_table([==[%s]==])
local ox, oy = %f, %f
local made, errs = 0, {}
for _, e in ipairs(data) do
  local name = e.n
  local opts = {name=name, position={e.x + ox, e.y + oy}, force=F}
  if e.k == "arith" or e.k == "decider" or e.k == "selector" then opts.direction = defines.direction[e.d or "east"] end
  local ent = S.create_entity(opts)
  if not ent then
    errs[#errs+1] = "place " .. name .. " " .. e.x .. "," .. e.y
  else
    storage.fg[e.id] = ent
    made = made + 1
    local ok, err = pcall(function()
      if e.k == "const" then
        local cb = ent.get_or_create_control_behavior()
        for i, sv in ipairs(e.p.signals or {}) do
          local si = math.floor((i - 1) / 1000) + 1
          local sec = cb.get_section(si) or cb.add_section()
          sec.set_slot((i - 1) %% 1000 + 1, {value={type=sv[1].type, name=sv[1].name, quality="normal"}, min=sv[2]})
        end
        if e.t and string.sub(e.t, 1, 5) == "RESET" then cb.enabled = true end
      elseif e.k == "panel" then
        local cb = ent.get_or_create_control_behavior()
        for i, r in ipairs(e.p.records) do if i <= cb.records_count then cb.set_record(i, r) else cb.add_record(r) end end
      elseif e.k == "lamp" then
        local cb = ent.get_or_create_control_behavior()
        cb.circuit_enable_disable = true
        cb.circuit_condition = e.p.condition
      elseif e.k == "eei" then
        ent.power_production = 1e12; ent.electric_buffer_size = 1e12
      elseif e.k == "arith" or e.k == "decider" or e.k == "selector" then
        ent.get_or_create_control_behavior().parameters = e.p
      end
    end)
    if not ok then errs[#errs+1] = "params " .. e.id .. " " .. tostring(err) end
  end
end
return {made=made, errs=errs}
"""

WIRE_LUA = r"""
local data = helpers.json_to_table([==[%s]==])
local ok_n, bad = 0, {}
for _, w in ipairs(data) do
  local a, b = storage.fg[w[1]], storage.fg[w[3]]
  local ok = false
  if a and b then
    ok = a.get_wire_connector(defines.wire_connector_id[w[2]], true).connect_to(b.get_wire_connector(defines.wire_connector_id[w[4]], true), false)
  end
  if ok then ok_n = ok_n + 1 elseif #bad < 20 then
    bad[#bad+1] = {w[1], w[2], w[3], w[4], a and a.name, b and b.name, a and a.position, b and b.position}
  end
end
return {ok=ok_n, bad=bad}
"""


def build(f: Factorio, circ, ox: float, oy: float, batch: int = 250, power: bool = True, log=print):
    ents = []
    for e in circ.ents:
        ents.append({"id": e.id, "k": e.kind, "n": NAMES[e.kind], "x": e.pos[0], "y": e.pos[1], "p": _conv(e.params), "t": e.tag,
                     "d": e.dir})
    if power:
        ents.append({"id": -1, "k": "eei", "n": NAMES["eei"], "x": circ.x0 - 3.0, "y": circ.y0 + 1.0, "p": {}, "t": "power"})
    t0 = time.time()
    f.lua("storage.fg = {}; return true")
    made, errs = 0, []
    for i in range(0, len(ents), batch):
        r = f.lua(BUILD_LUA % (json.dumps(ents[i:i + batch]), ox, oy))
        if not r.get("ok"):
            raise RuntimeError(r)
        made += r["res"]["made"]
        errs += r["res"]["errs"] or []
    log(f"placed {made}/{len(ents)} entities in {time.time() - t0:.1f}s; errors: {errs[:5]}")
    wires = [[a, CONN[ca], b, CONN[cb]] for a, ca, b, cb in circ.wires]
    ok, bad = 0, []
    for i in range(0, len(wires), 1000):
        r = f.lua(WIRE_LUA % json.dumps(wires[i:i + 1000]))
        if not r.get("ok"):
            raise RuntimeError(r)
        ok += r["res"]["ok"]
        bad += r["res"]["bad"] or []
    log(f"wired {ok}/{len(wires)}; failures: {bad[:5]}")
    return {"made": made, "errs": errs, "wired": ok, "bad": bad}


def ent_lua(eid: int) -> str:
    return f"storage.fg[{eid}]"


def set_reset(f: Factorio, info, on: bool):
    return f.lua(f"{ent_lua(info['reset'])}.get_or_create_control_behavior().enabled = {str(on).lower()}; return true")


def type_prompt(f: Factorio, info, text: str):
    from fgpt import spec
    assert len(text) <= len(info["keys"])
    cmds = []
    for k, eid in enumerate(info["keys"]):
        if k < len(text):
            t, n = spec.VOCAB_SIGNALS[spec.VOCAB[text[k]]]
            cmds.append(f"do local sec = {ent_lua(eid)}.get_or_create_control_behavior().get_section(1); "
                        f"for j = sec.filters_count, 1, -1 do sec.clear_slot(j) end; "
                        f"sec.set_slot(1, {{value={{type='{t}', name='{n}', quality='normal'}}, min=1}}) end")
        else:
            cmds.append(f"do local sec = {ent_lua(eid)}.get_or_create_control_behavior().get_section(1); "
                        f"for j = sec.filters_count, 1, -1 do sec.clear_slot(j) end end")
    return f.lua("\n".join(cmds) + "\nreturn true")


def read_screen(f: Factorio, info) -> str:
    from fgpt import spec
    rev = {n: spec.CHARS[i] for i, (_, n) in enumerate(spec.VOCAB_SIGNALS)}
    ids = "{" + ",".join(str(i) for i in info["screen"]) + "}"
    r = f.lua(f"""local out = {{}}
      for i, id in ipairs({ids}) do
        local n = storage.fg[id].get_circuit_network(defines.wire_connector_id.circuit_red)
        local s = ""
        if n and n.signals then for _, x in ipairs(n.signals) do if x.count > 0 then s = s .. x.signal.name .. "|" end end end
        out[i] = s
      end
      return out""")
    if info.get("tok"):
        # tokenizer builds: each cell holds its own bus signal with a letter code 1..42
        r = f.lua(f"""local out = {{}}
          for i, id in ipairs({ids}) do
            local n = storage.fg[id].get_circuit_network(defines.wire_connector_id.circuit_red)
            local v = 0
            if n and n.signals then for _, x in ipairs(n.signals) do v = (v == 0) and x.count or -1 end end
            out[i] = v
          end
          return out""")
        return spec.USER + "".join(spec.CHARS[v - 1] if 1 <= v <= spec.V else ("" if v == 0 else "*") for v in r["res"])
    chars = []
    for s in r["res"]:
        names = [x for x in s.split("|") if x]
        chars.append(rev.get(names[0], "#") if len(names) == 1 else ("" if not names else "*"))
    return spec.USER + "".join(chars)


def read_net(f: Factorio, eid: int, conn: str):
    return f.lua(f"""local n = storage.fg[{eid}].get_circuit_network(defines.wire_connector_id.{CONN[conn]})
      local r = {{}} if n and n.signals then for _, x in ipairs(n.signals) do r[x.signal.name] = x.count end end return r""")["res"]


def export_blueprint(f: Factorio, bbox, ox: float, oy: float, label: str, out_path: str):
    """Export the built circuit area as a blueprint string file (written server-side, then copied)."""
    import shutil
    from pathlib import Path
    x0, y0, x1, y1 = bbox
    fname = f"fgpt_{label}.txt"
    r = f.lua(f"""
      local inv = game.create_inventory(1)
      local st = inv[1]
      st.set_stack{{name="blueprint"}}
      st.create_blueprint{{surface=S, force=F, area={{{{{x0 + ox},{y0 + oy}}},{{{x1 + ox},{y1 + oy}}}}}}}
      st.label = {json.dumps(label)}
      local n = st.get_blueprint_entity_count()
      local s = st.export_stack()
      helpers.write_file({json.dumps(fname)}, s, false)
      inv.destroy()
      return {{entities=n, chars=#s}}""")
    src = Path(f.data) / "script-output" / fname
    for _ in range(50):
        if src.exists() and src.stat().st_size > 0:
            break
        time.sleep(0.1)
    shutil.copy(src, out_path)
    return r


def attach(f: Factorio, circ, ox: float, oy: float, batch: int = 2000):
    """Map circuit entity ids to already-built entities (e.g. pasted from a blueprint) by position."""
    f.lua("storage.fg = {}; return true")
    items = [[e.id, NAMES[e.kind], e.pos[0] + ox, e.pos[1] + oy] for e in circ.ents]
    missing = 0
    for i in range(0, len(items), batch):
        r = f.lua(f"""local data = helpers.json_to_table([==[{json.dumps(items[i:i + batch])}]==])
          local miss = 0
          for _, it in ipairs(data) do
            local e = S.find_entity(it[2], {{it[3], it[4]}})
            if e then storage.fg[it[1]] = e else miss = miss + 1 end
          end
          return miss""")
        missing += r["res"]
    return missing


def paste_and_locate(f: Factorio, bp_path: str, circ, at=(0.0, 0.0), near=None):
    """Paste a blueprint file, then find the offset that maps circuit positions onto the pasted entities.
    The pasted build is located by its clock (time selector) nearest to `near` (default: the paste point)."""
    near = near or at
    from pathlib import Path
    bp = Path(bp_path).read_text().strip()
    f.lua("storage.bp = {}; return true")
    # 4 MB pieces. Commands are forwarded to connected players at a few KB/s, so upload with nobody connected.
    for i in range(0, len(bp), 4_000_000):
        f.lua(f"storage.bp[#storage.bp + 1] = [==[{bp[i:i + 4_000_000]}]==]; return #storage.bp")
    r = f.lua(f"""
      local inv = game.create_inventory(1); local st = inv[1]
      st.import_stack(table.concat(storage.bp)); storage.bp = nil
      local ghosts = st.build_blueprint{{surface=S, force=F, position={{{at[0]},{at[1]}}}, build_mode=defines.build_mode.forced}}
      local n = 0
      for _, g in ipairs(ghosts) do if g.valid and g.type == "entity-ghost" then local _, e = g.silent_revive(); if e then n = n + 1 end end end
      inv.destroy()
      local sel = S.find_entities_filtered{{name="selector-combinator"}}
      local best, bd = nil, 1e18
      for _, s in ipairs(sel) do
        if s.get_control_behavior().parameters.operation == "time" then
          local d = (s.position.x - {near[0]})^2 + (s.position.y - {near[1]})^2
          if d < bd then best, bd = s, d end
        end
      end
      return {{built=n, clock=best and best.position}}""")
    if "res" not in r:
        raise RuntimeError(f"blueprint paste failed: {r}")
    clock = next(e for e in circ.ents if e.kind == "selector" and e.params.get("operation") == "time")
    p = r["res"]["clock"]
    ox, oy = p["x"] - clock.pos[0], p["y"] - clock.pos[1]
    return r["res"]["built"], ox, oy
