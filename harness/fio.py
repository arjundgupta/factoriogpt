"""Core Factorio I/O for the test harness, used by the build and test scripts.

Everything here talks to the *test server* over RCON. Lua runs only in that test
world: to build, inspect and step circuits. It never does model math.
The player can join the same world from the normal game client (see join_info()).
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import signal
import socket
import subprocess
import time
import zlib
from pathlib import Path

from rcon import FACTORIO, GAME_PORT, PASSWORD, RCON_PORT, Rcon

ROOT = Path(__file__).resolve().parent
SERVER = ROOT / "server"
DATA = SERVER / "data"
SAVES = DATA / "saves"
PRISTINE = SERVER / "sandbox.zip"
CLIENT_SCRIPT_OUTPUT = Path.home() / "Library/Application Support/factorio/script-output"
LOG = SERVER / "server.log"

# ----------------------------------------------------------------------------------
# Lua prelude: helpers available to every command. Targets are either {x,y} positions
# or "#<unit_number>".
# ----------------------------------------------------------------------------------
PRELUDE = r"""
local S = (storage.fgsurf and game.surfaces[storage.fgsurf]) or game.surfaces[1]
local F = game.forces.player
local W = defines.wire_connector_id
local function sigid(s)
  if type(s) == "table" then return {type=s.type or "virtual", name=s.name, quality=s.quality or "normal"} end
  local t, n = string.match(s, "^([%w%-]+):(.+)$")
  if t then return {type=t, name=n, quality="normal"} end
  if prototypes.virtual_signal[s] then return {type="virtual", name=s, quality="normal"} end
  if prototypes.item[s] then return {type="item", name=s, quality="normal"} end
  if prototypes.fluid[s] then return {type="fluid", name=s, quality="normal"} end
  error("unknown signal " .. tostring(s))
end
local function signame(sig)
  local t = sig.type or "item"
  if t == "virtual" or t == "item" then return sig.name end
  return t .. ":" .. sig.name
end
local function target(t)
  if type(t) == "string" and string.sub(t, 1, 1) == "#" then
    local e = game.get_entity_by_unit_number(tonumber(string.sub(t, 2)))
    if not e then error("no entity " .. t) end
    return e
  end
  local x, y = t[1] or t.x, t[2] or t.y
  -- Prefer an entity whose box contains the point, else the nearest entity within 1 tile.
  local best, bd = nil, 1e9
  for _, e in ipairs(S.find_entities_filtered{position={x, y}, radius=1.0}) do
    if e.type ~= "character" and e.type ~= "entity-ghost" then
      local bb = e.bounding_box
      local inside = x >= bb.left_top.x and x <= bb.right_bottom.x and y >= bb.left_top.y and y <= bb.right_bottom.y
      local d = (e.position.x - x)^2 + (e.position.y - y)^2 - (inside and 100 or 0)
      if d < bd then best, bd = e, d end
    end
  end
  if not best then error("no entity at " .. x .. "," .. y) end
  return best
end
local CONN = {
  red=W.circuit_red, green=W.circuit_green,
  input_red=W.combinator_input_red, input_green=W.combinator_input_green,
  output_red=W.combinator_output_red, output_green=W.combinator_output_green,
}
local function is_combinator(e)
  return e.type == "arithmetic-combinator" or e.type == "decider-combinator" or e.type == "selector-combinator"
end
local function conn(e, c)
  if c == "auto_red" or c == nil then c = is_combinator(e) and "output_red" or "red" end
  if c == "auto_green" then c = is_combinator(e) and "output_green" or "green" end
  local id = CONN[c]
  if not id then error("bad connector " .. tostring(c)) end
  return id, c
end
local function net_signals(e, id)
  local n = e.get_circuit_network(id)
  if not n then return nil end
  local r = {}
  for _, s in ipairs(n.signals or {}) do r[signame(s.signal)] = s.count end
  return r
end
local function all_signals(e)
  local r = {}
  local names = is_combinator(e) and {"input_red","input_green","output_red","output_green"} or {"red","green"}
  for _, c in ipairs(names) do
    local ok, v = pcall(net_signals, e, CONN[c])
    if ok and v then r[c] = v end
  end
  return r
end
local function describe(e, deep)
  local d = {unit=e.unit_number, name=e.name, type=e.type, x=e.position.x, y=e.position.y,
             direction=e.direction}
  if deep then
    local cb = e.get_control_behavior()
    if cb then
      if e.type == "constant-combinator" then
        local secs = {}
        for i, sec in ipairs(cb.sections) do
          local fl = {}
          for j, f in pairs(sec.filters) do if f.value then fl[#fl+1] = {slot=j, signal=signame(f.value), value=f.min} end end
          secs[i] = {active=sec.active, group=sec.group, filters=fl}
        end
        d.enabled = cb.enabled; d.sections = secs
      elseif e.type == "arithmetic-combinator" or e.type == "decider-combinator" or e.type == "selector-combinator" then
        d.parameters = cb.parameters
      elseif e.type == "lamp" then
        d.circuit_condition = cb.circuit_condition; d.use_colors = cb.use_colors
      end
    end
    local wires = {}
    for _, c in pairs({"red","green","input_red","input_green","output_red","output_green"}) do
      local ok, wc = pcall(function() return e.get_wire_connector(CONN[c], false) end)
      if ok and wc then
        for _, cn in ipairs(wc.connections) do
          local o = cn.target.owner
          wires[#wires+1] = {from=c, to_unit=o.unit_number, to_name=o.name, to_x=o.position.x, to_y=o.position.y,
                             to_connector=cn.target.wire_connector_id}
        end
      end
    end
    d.wires = wires
    d.signals = all_signals(e)
    if e.energy ~= nil then d.energy = e.energy end
    d.status = e.status
  end
  return d
end
"""


def _wrap(body: str) -> str:
    return (PRELUDE + "\nlocal ok, res = pcall(function()\n" + body + "\nend)\n"
            "if ok then rcon.print(helpers.table_to_json({ok=true, res=res})) "
            "else rcon.print(helpers.table_to_json({ok=false, err=tostring(res)})) end")


def lua_str(v) -> str:
    """Python value → Lua literal."""
    if v is None:
        return "nil"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)  # raw UTF-8 is fine in Lua; avoid \u escapes
    if isinstance(v, (list, tuple)):
        return "{" + ",".join(lua_str(x) for x in v) + "}"
    if isinstance(v, dict):
        return "{" + ",".join(f"[{lua_str(k)}]={lua_str(x)}" for k, x in v.items()) + "}"
    raise TypeError(type(v))


def tgt(t) -> str:
    """Target spec: (x, y) | [x, y] | {"x":..,"y":..} | "#unit" | "x,y"."""
    if isinstance(t, str):
        if t.startswith("#"):
            return lua_str(t)
        x, y = (float(p) for p in t.split(","))
        return lua_str([x, y])
    if isinstance(t, dict):
        return lua_str([t["x"], t["y"]])
    return lua_str(list(t))


# ----------------------------------------------------------------------------------
# Blueprint strings (offline): "0" + base64(zlib(json))
# ----------------------------------------------------------------------------------
def bp_decode(s: str) -> dict:
    s = s.strip()
    return json.loads(zlib.decompress(base64.b64decode(s[1:])))


def bp_encode(obj: dict) -> str:
    return "0" + base64.b64encode(zlib.compress(json.dumps(obj, separators=(",", ":")).encode(), 9)).decode()


# ----------------------------------------------------------------------------------
class Factorio:
    def __init__(self, instance: str = "main", rcon_port: int = RCON_PORT, game_port: int = GAME_PORT):
        """instance="main" is the world the user joins. Other instances (e.g. "bench") get their own
        data dir, config, log and ports, so they can run alongside it."""
        self.instance = instance
        self.rcon_port, self.game_port = rcon_port, game_port
        if instance == "main":
            self.data, self.config, self.log = DATA, SERVER / "config.ini", LOG
        else:
            self.data = SERVER / f"data_{instance}"
            self.config = SERVER / f"config_{instance}.ini"
            self.log = SERVER / f"server_{instance}.log"
            self.data.mkdir(parents=True, exist_ok=True)
            self.config.write_text(f"[path]\nread-data=__PATH__system-read-data__\nwrite-data={self.data}\n")
        self.saves = self.data / "saves"
        self.proc: subprocess.Popen | None = None
        self._rcon: Rcon | None = None
        self.world: str | None = None

    # ---------------- connection ----------------
    def rcon(self) -> Rcon:
        if self._rcon is None:
            self._rcon = Rcon(port=self.rcon_port)
            self._rcon.lua("local _=0")  # unlocks /sc on this connection
        return self._rcon

    def _reset_rcon(self):
        try:
            if self._rcon:
                self._rcon.close()
        except OSError:
            pass
        self._rcon = None

    def running(self) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", self.rcon_port), timeout=1):
                return True
        except OSError:
            return False

    def lua(self, body: str):
        """Run Lua (in a function body, with PRELUDE helpers). Returns the decoded JSON {ok,res|err}."""
        for attempt in range(2):
            try:
                raw = self.rcon().lua(_wrap(body))
                break
            except (OSError, ConnectionError):
                self._reset_rcon()
                if attempt:
                    raise
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"ok": False, "raw": raw}

    def lua_raw(self, code: str) -> str:
        """Run raw Lua exactly as given (no prelude, no wrapping). Use rcon.print() for output."""
        return self.rcon().lua(code)

    def command(self, cmd: str) -> str:
        """Run a raw server console command, e.g. '/server-save' or '/players'."""
        return self.rcon().cmd(cmd)

    # ---------------- lifecycle ----------------
    def start(self, world: str = "main", fresh: bool = False, lab: bool = False) -> dict:
        """lab=False gives an exactly-default freeplay world (grass, trees, biters, crash site), like single player.
        lab=True gives a flat checkerboard lab world."""
        if self.running():
            return {"status": "already running", "world": self.world, **self.status()}
        self.saves.mkdir(parents=True, exist_ok=True)
        save = self.saves / f"{world}.zip"
        created = False
        if fresh or not save.exists():
            shutil.copy(PRISTINE, save)
            created = True
        log = open(self.log, "w")
        # The headless server quits when stdin closes. Feed it from `tail -f /dev/null` in its own
        # session, so it keeps running after this Python process exits.
        self.proc = subprocess.Popen(
            ["/bin/sh", "-c", 'tail -f /dev/null | "$@"', "sh",
             FACTORIO, "--config", str(self.config),
             "--start-server", str(save),
             "--server-settings", str(SERVER / "server-settings.json"),
             "--port", str(self.game_port),
             "--rcon-port", str(self.rcon_port), "--rcon-password", PASSWORD],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
        for _ in range(240):
            if self.proc.poll() is not None:
                raise RuntimeError("server exited early; see log:\n" + self.log_tail(30))
            if self.running():
                break
            time.sleep(0.25)
        else:
            raise TimeoutError("server did not open RCON")
        self.world = world
        self._reset_rcon()
        if created:
            if lab:
                self.prepare_lab()
            else:
                self.prepare_normal()
            self.save()
        return {"status": "started", "world": world, "created": created, **self.status(), **self.join_info()}

    def stop(self, save: bool = True) -> dict:
        if not self.running():
            return {"status": "not running"}
        if save:
            self.save()
        self._reset_rcon()
        if self.proc:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)   # sh, tail and factorio share this group
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc = None
        else:   # started by an earlier process: kill every process group running this config
            pids = subprocess.run(["pgrep", "-f", str(self.config)], capture_output=True, text=True).stdout.split()
            for pgid in {os.getpgid(int(p)) for p in pids if int(p) != os.getpid()}:
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for _ in range(240):   # big maps take a while to save on shutdown
            if not self.running() and subprocess.run(["pgrep", "-f", str(self.config)], capture_output=True).returncode != 0:
                break
            time.sleep(0.5)
        return {"status": "stopped", "saved": save}

    def save(self, name: str | None = None) -> str:
        return self.command(f"/server-save {name}" if name else "/server-save")

    def list_worlds(self) -> list[str]:
        return sorted(p.stem for p in self.saves.glob("*.zip"))

    def log_tail(self, n: int = 40) -> str:
        if not self.log.exists():
            return ""
        return "\n".join(self.log.read_text(errors="replace").splitlines()[-n:])

    def join_info(self) -> dict:
        return {"join": f"In Factorio: Multiplayer → Connect to address → 127.0.0.1:{self.game_port}",
                "tip": "Once in: open the map editor with /editor (test world only)."}

    def status(self) -> dict:
        r = self.lua("""
          local players = {}
          for _, p in pairs(game.connected_players) do players[#players+1] = {name=p.name, x=p.position.x, y=p.position.y, controller=p.controller_type} end
          return {tick=game.tick, paused=game.tick_paused, speed=game.speed, players=players,
                  entities=#S.find_entities_filtered{force=F}, version=script.active_mods["base"]}
        """)
        return r.get("res", r)

    # ---------------- world ----------------
    def prepare_normal(self) -> dict:
        """Exactly a default freeplay world: same map generation and settings as single player, no changes."""
        return self.lua("game.tick_paused = false; return {tile=S.get_tile(0,0).name}")

    def prepare_lab(self, radius_chunks: int = 8) -> dict:
        """Flat lab-tile world: no enemies, no resources, always day, everything researched and charted."""
        self.lua("""
          game.tick_paused = false
          S.generate_with_lab_tiles = true
          S.always_day = true
          S.peaceful_mode = true
          game.map_settings.enemy_expansion.enabled = false
          game.map_settings.pollution.enabled = false
          S.clear(true)
          return true
        """)
        time.sleep(0.5)  # clearing completes at the end of the tick
        self.lua(f"""
          S.request_to_generate_chunks({{0, 0}}, {radius_chunks})
          S.force_generate_chunk_requests()
          F.research_all_technologies()
          return true
        """)
        r = 32 * radius_chunks
        self.lua(f"F.chart(S, {{{{-{r}, -{r}}}, {{{r}, {r}}}}}); return true")
        time.sleep(1.0)
        return self.lua("return {generated=S.is_chunk_generated({1,1}), charted=F.is_chunk_charted(S,{1,1}), tile=S.get_tile(40,40).name}")

    def clear_area(self, x1, y1, x2, y2, keep_power: bool = False) -> dict:
        keep = "true" if keep_power else "false"
        return self.lua(f"""
          local n = 0
          for _, e in ipairs(S.find_entities_filtered{{area={{{{{x1},{y1}}},{{{x2},{y2}}}}}}}) do
            if e.type ~= "character" and not ({keep} and (e.type == "electric-energy-interface" or e.type == "electric-pole")) then
              e.destroy(); n = n + 1
            end
          end
          return n
        """)

    def add_power(self, x1, y1, x2, y2) -> dict:
        """Infinite power over an area: one editor energy interface plus a substation grid."""
        return self.lua(f"""
          local n = 0
          local eei = S.create_entity{{name="electric-energy-interface", position={{{x1}-3, {y1}}}, force=F}}
          if eei then eei.power_production = 1e12; eei.electric_buffer_size = 1e12 end
          for x = {x1}, {x2}, 18 do for y = {y1}, {y2}, 18 do
            if S.create_entity{{name="substation", position={{x, y}}, force=F}} then n = n + 1 end
          end end
          return {{substations=n}}
        """)

    # ---------------- time ----------------
    def pause(self, paused: bool = True):
        return self.lua(f"game.tick_paused = {lua_str(paused)}; return game.tick")

    def set_speed(self, speed: float):
        return self.lua(f"game.speed = {speed}; return game.speed")

    def tick(self) -> int:
        return self.lua("return game.tick")["res"]

    def run_ticks(self, n: int, timeout: float = 120) -> dict:
        """Pause, then advance exactly n ticks and stay paused."""
        start = self.lua(f"game.tick_paused = true; game.ticks_to_run = {int(n)}; return game.tick")["res"]
        t0 = time.time()
        while time.time() - t0 < timeout:
            r = self.lua("return {tick=game.tick, left=game.ticks_to_run}")["res"]
            if r["left"] == 0 and r["tick"] >= start + n:
                return {"from": start, "to": r["tick"]}
            time.sleep(0.05)
        raise TimeoutError("ticks did not finish")

    def benchmark(self, ticks: int = 600) -> dict:
        """Measure max UPS: run unpaused at a huge game.speed and time it."""
        self.lua("game.tick_paused = false; game.speed = 1000")
        t0 = time.time()
        start = self.tick()
        while self.tick() < start + ticks:
            time.sleep(0.02)
        dt = time.time() - t0
        end = self.tick()
        self.lua("game.speed = 1")
        return {"ticks": end - start, "seconds": round(dt, 3), "ups": round((end - start) / dt, 1)}

    # ---------------- entities ----------------
    def place(self, name: str, x: float, y: float, direction: str | None = None) -> dict:
        d = f", direction=defines.direction.{direction}" if direction else ""
        return self.lua(f"""
          local e = S.create_entity{{name={lua_str(name)}, position={{{x},{y}}}, force=F{d}}}
          if not e then error("cannot place {name} at {x},{y}") end
          return describe(e, false)
        """)

    def remove(self, target) -> dict:
        return self.lua(f"local e = target({tgt(target)}); local d = describe(e,false); e.destroy(); return d")

    def find(self, x1, y1, x2, y2, name: str | None = None, type_: str | None = None, deep: bool = False) -> dict:
        f = f"area={{{{{x1},{y1}}},{{{x2},{y2}}}}}"
        if name:
            f += f", name={lua_str(name)}"
        if type_:
            f += f", type={lua_str(type_)}"
        return self.lua(f"""
          local out = {{}}
          for _, e in ipairs(S.find_entities_filtered{{{f}}}) do
            if e.type ~= "character" then out[#out+1] = describe(e, {lua_str(deep)}) end
          end
          return out
        """)

    def inspect(self, target) -> dict:
        return self.lua(f"return describe(target({tgt(target)}), true)")

    def set_constant(self, target, signals: dict, section: int = 1, replace: bool = True, enabled: bool = True) -> dict:
        """signals: {"signal-A": 5, "iron-plate": -3, "fluid:water": 1}"""
        items = list(signals.items())
        return self.lua(f"""
          local e = target({tgt(target)})
          local cb = e.get_or_create_control_behavior()
          cb.enabled = {lua_str(enabled)}
          while #cb.sections < {section} do cb.add_section() end
          local sec = cb.get_section({section})
          if {lua_str(replace)} then for j = sec.filters_count, 1, -1 do sec.clear_slot(j) end end
          local base = {lua_str(replace)} and 0 or sec.filters_count
          local pairs_ = {lua_str([[k, v] for k, v in items])}
          for i, kv in ipairs(pairs_) do sec.set_slot(base + i, {{value=sigid(kv[1]), min=kv[2]}}) end
          return describe(e, true)
        """)

    def set_combinator(self, target, params: dict) -> dict:
        """Set arithmetic/decider/selector parameters. Signal fields may be strings like "signal-each"."""
        return self.lua(f"""
          local e = target({tgt(target)})
          local p = {lua_str(params)}
          local function fix(t)
            if type(t) ~= "table" then return t end
            for k, v in pairs(t) do
              if (string.find(k, "signal") and not string.find(k, "networks")) and type(v) == "string" then t[k] = sigid(v)
              elseif type(v) == "table" then fix(v) end
            end
            return t
          end
          e.get_or_create_control_behavior().parameters = fix(p)
          return describe(e, true)
        """)

    def connect(self, a, a_conn: str, b, b_conn: str) -> dict:
        """Connectors: red|green (poles, constants, lamps…), input_red|input_green|output_red|output_green (combinators)."""
        return self.lua(f"""
          local A, B = target({tgt(a)}), target({tgt(b)})
          local ok = A.get_wire_connector(CONN[{lua_str(a_conn)}], true).connect_to(B.get_wire_connector(CONN[{lua_str(b_conn)}], true), false)
          return {{connected=ok}}
        """)

    def disconnect(self, a, a_conn: str, b, b_conn: str) -> dict:
        return self.lua(f"""
          local A, B = target({tgt(a)}), target({tgt(b)})
          return {{disconnected=A.get_wire_connector(CONN[{lua_str(a_conn)}], true).disconnect_from(B.get_wire_connector(CONN[{lua_str(b_conn)}], true))}}
        """)

    def read(self, target, connector: str | None = None) -> dict:
        """Signals on one connector, or all of the entity's connectors if connector is None."""
        if connector:
            return self.lua(f"local e = target({tgt(target)}); local id = conn(e, {lua_str(connector)}); return net_signals(e, id) or {{}}")
        return self.lua(f"return all_signals(target({tgt(target)}))")

    def watch(self, targets: list, ticks: int, connector: str | None = None, signals: list[str] | None = None) -> dict:
        """Advance `ticks` ticks (paused stepping) and record the signals of each target every tick."""
        tl = "{" + ",".join(tgt(t) for t in targets) + "}"
        filt = lua_str(signals) if signals else "nil"
        rows = []
        self.lua("game.tick_paused = true")
        for _ in range(int(ticks)):
            self.run_ticks(1)
            r = self.lua(f"""
              local filt = {filt}
              local row = {{tick=game.tick, values={{}}}}
              for i, t in ipairs({tl}) do
                local e = target(t)
                local s = net_signals(e, conn(e, {lua_str(connector)})) or {{}}
                if filt then local f = {{}} for _, n in ipairs(filt) do f[n] = s[n] or 0 end s = f end
                row.values[i] = s
              end
              return row
            """)
            rows.append(r.get("res", r))
        return {"rows": rows}

    # ---------------- blueprints ----------------
    def paste_blueprint(self, bp: str, x: float = 0, y: float = 0) -> dict:
        """Build a blueprint string at (x, y) instantly (ghosts are revived)."""
        if os.path.exists(bp):
            bp = Path(bp).read_text().strip()
        return self.lua(f"""
          local inv = game.create_inventory(1)
          local st = inv[1]
          local rc = st.import_stack({lua_str(bp)})
          if rc ~= 0 and rc ~= 1 then inv.destroy(); error("import_stack failed: " .. rc) end
          if not st.is_blueprint then inv.destroy(); error("not a single blueprint (books not supported)") end
          local ghosts = st.build_blueprint{{surface=S, force=F, position={{{x},{y}}}, build_mode=defines.build_mode.forced, skip_fog_of_war=false}}
          local built, failed = 0, 0
          for _, g in ipairs(ghosts) do
            if g.valid and g.type == "entity-ghost" then
              local _, ent = g.silent_revive()
              if ent then built = built + 1 else failed = failed + 1 end
            elseif g.valid then built = built + 1 end
          end
          inv.destroy()
          return {{built=built, failed=failed, import_rc=rc}}
        """)

    def export_blueprint(self, x1, y1, x2, y2, label: str = "fgpt", out_file: str | None = None) -> dict:
        r = self.lua(f"""
          local inv = game.create_inventory(1)
          local st = inv[1]
          st.set_stack{{name="blueprint"}}
          st.create_blueprint{{surface=S, force=F, area={{{{{x1},{y1}}},{{{x2},{y2}}}}}, always_include_tiles=false}}
          st.label = {lua_str(label)}
          local s = st.export_stack()
          local n = st.get_blueprint_entity_count()
          inv.destroy()
          return {{entities=n, blueprint=s}}
        """)
        if out_file and r.get("ok"):
            Path(out_file).write_text(r["res"]["blueprint"])
            r["res"]["file"] = out_file
        return r

    # ---------------- players & visuals ----------------
    def players(self) -> dict:
        return self.lua("""
          local out = {}
          for _, p in pairs(game.players) do out[#out+1] = {index=p.index, name=p.name, connected=p.connected, admin=p.admin,
            x=p.position.x, y=p.position.y, controller=p.controller_type} end
          return out
        """)

    def chat(self, msg: str) -> dict:
        return self.lua(f"game.print({lua_str('[harness] ' + msg)}); return true")

    def teleport_player(self, x, y, player: str | int | None = None) -> dict:
        sel = lua_str(player) if player is not None else "nil"
        return self.lua(f"""
          local p = {sel} and game.get_player({sel}) or game.connected_players[1]
          if not p then error("no connected player") end
          p.teleport({{{x},{y}}}, S)
          return {{name=p.name}}
        """)

    def promote(self, player: str) -> str:
        return self.command(f"/promote {player}")

    def screenshot(self, x, y, zoom: float = 1.0, width: int = 1600, height: int = 1000,
                   alt_mode: bool = True, player: str | int | None = None, name: str | None = None) -> dict:
        """Render the world at (x,y). Needs a connected player: the image is rendered by their client
        and written to their script-output folder."""
        name = name or f"fgpt_{int(time.time() * 1000)}.png"
        sel = lua_str(player) if player is not None else "nil"
        r = self.lua(f"""
          local p = {sel} and game.get_player({sel}) or game.connected_players[1]
          if not p then error("screenshots need a connected player (join the server from your game)") end
          game.take_screenshot{{player=p, by_player=p, surface=S, position={{{x},{y}}}, resolution={{{width},{height}}},
            zoom={zoom}, path={lua_str('fgpt/' + name)}, show_entity_info={'true' if alt_mode else 'false'}, show_gui=false,
            daytime=0, water_tick=0, allow_in_replay=false}}
          return {{player=p.name}}
        """)
        if not r.get("ok"):
            return r
        # take_screenshot happens at the end of the tick; make sure ticks advance if paused
        st = self.lua("return game.tick_paused")["res"]
        if st:
            self.run_ticks(1)
        path = CLIENT_SCRIPT_OUTPUT / "fgpt" / name
        for _ in range(1200):   # very large renders can take a while
            if path.exists() and path.stat().st_size > 0:
                time.sleep(0.2)
                return {"ok": True, "res": {"path": str(path)}}
            time.sleep(0.1)
        return {"ok": False, "err": f"screenshot not found at {path}"}

    def ascii_map(self, x1, y1, x2, y2) -> str:
        """Text map of an area: one char per tile. Works without a connected player."""
        r = self.lua(f"""
          local CH = {{["constant-combinator"]="C", ["arithmetic-combinator"]="A", ["decider-combinator"]="D",
            ["selector-combinator"]="S", ["lamp"]="o", ["display-panel"]="P", ["electric-pole"]="+", ["electric-energy-interface"]="E",
            ["power-switch"]="W"}}
          local grid = {{}}
          for _, e in ipairs(S.find_entities_filtered{{area={{{{{x1},{y1}}},{{{x2},{y2}}}}}}}) do
            if e.type ~= "character" then
              local c = CH[e.type] or "#"
              local bb = e.bounding_box
              for tx = math.floor(bb.left_top.x + 0.01), math.ceil(bb.right_bottom.x - 0.01) - 1 do
                for ty = math.floor(bb.left_top.y + 0.01), math.ceil(bb.right_bottom.y - 0.01) - 1 do
                  grid[ty] = grid[ty] or {{}}; grid[ty][tx] = c
                end
              end
            end
          end
          local lines = {{}}
          for ty = {int(y1)}, {int(y2) - 1} do
            local row = {{}}
            for tx = {int(x1)}, {int(x2) - 1} do row[#row+1] = (grid[ty] and grid[ty][tx]) or "." end
            lines[#lines+1] = table.concat(row)
          end
          return table.concat(lines, "\\n")
        """)
        if not r.get("ok"):
            return json.dumps(r)
        legend = "Legend: C constant  A arithmetic  D decider  S selector  o lamp  P display  + pole  E power  # other"
        return f"x {x1}..{x2}, y {y1}..{y2}\n{r['res']}\n{legend}"
