"""Combinator netlist, layout and wire routing.

A Circuit holds entities (combinators, constants, panels, poles) and nets (one wire color each).
Layout: modules are placed left to right as columns of slots inside a grid of 4x16-slot blocks,
separated by 2-tile lanes. Substations sit at lane crossings (power + circuit relays), and big
poles in the lanes relay nets whose members are farther apart than combinator wire reach.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

EACH = ("virtual", "signal-each")
EVERYTHING = ("virtual", "signal-everything")
ANYTHING = ("virtual", "signal-anything")

COLOR = {"in_r": "red", "in_g": "green", "out_r": "red", "out_g": "green", "r": "red", "g": "green"}
REACH = 8.5        # edge-to-edge distance for combinator wires (game limit 9, measured between bounding boxes)
POLE_REACH = 29.0  # big pole <-> big pole edge-to-edge (game limit 32)
HALF = {"arith": (1.0, 0.5), "decider": (1.0, 0.5), "selector": (1.0, 0.5), "const": (0.5, 0.5), "panel": (0.5, 0.5),
        "lamp": (0.5, 0.5), "pole": (1.0, 1.0), "substation": (1.0, 1.0)}


def half(e):
    hx, hy = HALF.get(e.kind, (0.5, 0.5))
    return (hy, hx) if e.dir in ("north", "south") else (hx, hy)
BCOLS, BROWS, LANE = 4, 16, 2
SLOT_W = 3
BPX = BCOLS * SLOT_W + LANE   # 14
BPY = BROWS + LANE            # 18


@dataclass
class Ent:
    id: int
    kind: str                 # arith | decider | selector | const | panel | lamp | pole | substation | eei
    params: dict = field(default_factory=dict)
    conns: dict = field(default_factory=dict)   # connector -> net id
    pos: tuple | None = None
    tag: str = ""
    mem: bool = False          # memory-cell combinator (timing boundary)
    ctrl: bool = False         # controller combinator (fixed timing)
    dir: str = "east"


@dataclass
class Net:
    id: int
    color: str
    name: str
    members: list = field(default_factory=list)  # (ent id, connector)


class Module:
    def __init__(self, circ: "Circuit", name: str):
        self.c, self.name = circ, name
        self.col0 = circ.next_col
        self.ncols = 0

    def put(self, ent: Ent, col: int, row: int, side: bool = False):
        assert 0 <= row < self.c.H, (self.name, row)
        self.ncols = max(self.ncols, col + 1)
        gcol = self.col0 + col
        key = (gcol, row, "side" if side else "main")
        assert key not in self.c.occupied, (self.name, key)
        self.c.occupied[key] = ent.id
        ent.pos = self.c.slot_xy(gcol, row, side)
        ent.tag = ent.tag or self.name
        return ent

    def close(self):
        self.c.next_col = self.col0 + max(self.ncols, 1)


class Circuit:
    def __init__(self, H: int, x0: float = 0, y0: float = 0, lane: int = LANE):
        assert H % BROWS == 0
        self.H = H
        self.LANE = lane
        self.BPX = BCOLS * SLOT_W + lane
        self.BPY = BROWS + lane
        self.x0, self.y0 = x0, y0
        self.ents: list[Ent] = []
        self.nets: list[Net] = []
        self.occupied: dict = {}
        self.next_col = 0
        self.wires: list = []     # (ent a, conn a, ent b, conn b)
        self.named: dict = {}
        self.extra_spots: list = []   # big-pole spots outside the block grid (free-form regions)
        self.regions: list = []       # (x0, y0, x1, y1) of free-form regions

    # ------------------------------------------------------------ geometry
    def slot_xy(self, col, row, side):
        bx, cx = divmod(col, BCOLS)
        by, ry = divmod(row, BROWS)
        x = self.x0 + self.LANE + bx * self.BPX + cx * SLOT_W
        y = self.y0 + self.LANE + by * self.BPY + ry
        return (x + 0.5, y + 0.5) if side else (x + 2.0, y + 0.5)

    @property
    def nblocks_x(self):
        return math.ceil(self.next_col / BCOLS)

    @property
    def nblocks_y(self):
        return self.H // BROWS

    def grid_bbox(self):
        return (self.x0, self.y0, self.x0 + self.LANE + self.nblocks_x * self.BPX, self.y0 + self.LANE + self.nblocks_y * self.BPY)

    def bbox(self):
        x0, y0, x1, y1 = self.grid_bbox()
        for a, b, c, d in self.regions:
            x0, y0, x1, y1 = min(x0, a), min(y0, b), max(x1, c), max(y1, d)
        return (x0, y0, x1, y1)

    def place_free(self, ent: Ent, x: float, y: float, dir: str = "east", tag: str = ""):
        ent.pos, ent.dir = (x, y), dir
        ent.tag = ent.tag or tag
        return ent

    def text_region(self, x0, y0, n, per_line, rows_per_item, build_item, tag):
        """Free-form region: items in lines of `per_line`, grouped by 8 with 2-tile gaps holding substations
        and relay poles. build_item(k, x, y) places item k with its top-left tile at (x, y), 1 tile wide."""
        groups = (per_line + 7) // 8
        width = groups * 8 + (groups + 1) * 2
        lines = (n + per_line - 1) // per_line
        height = lines * rows_per_item
        for k in range(n):
            line, j = divmod(k, per_line)
            g, jj = divmod(j, 8)
            x = x0 + 2 + g * 10 + jj
            y = y0 + line * rows_per_item
            build_item(k, x, y)
        # substations every 18 rows in every gap column, relay pole spots in the remaining gap tiles
        for g in range(groups + 1):
            gx = x0 + g * 10
            for yy in range(0, height + 18, 18):
                self._ent("substation", {}, {}, pos=(gx + 1.0, y0 + min(yy, max(height - 2, 0)) + 1.0), tag="power")
            used_rows = {min(yy, max(height - 2, 0)) for yy in range(0, height + 18, 18)}
            for yy in range(0, height - 1, 2):
                if any(abs(yy - u) < 2 for u in used_rows):
                    continue
                self.extra_spots.append((gx + 1.0, y0 + yy + 1.0))
        self.regions.append((x0, y0, x0 + width, y0 + height))
        return (x0, y0, x0 + width, y0 + height)

    # ------------------------------------------------------------ construction
    def net(self, color: str, name: str = "") -> Net:
        n = Net(len(self.nets), color, name)
        self.nets.append(n)
        if name:
            self.named[name] = n
        return n

    def _ent(self, kind, params, conns, **kw) -> Ent:
        e = Ent(len(self.ents), kind, params, {}, **kw)
        self.ents.append(e)
        for conn, net in conns.items():
            if net is None:
                continue
            assert COLOR[conn] == net.color, (kind, conn, net.name, net.color)
            e.conns[conn] = net.id
            net.members.append((e.id, conn))
        return e

    def arith(self, first, op, second, out, ins: dict, outs: dict, fn="rg", sn="rg", **kw):
        p = {"operation": op, "output_signal": out,
             "first_signal_networks": _nets(fn), "second_signal_networks": _nets(sn)}
        if isinstance(first, int):
            p["first_constant"] = first
        else:
            p["first_signal"] = first
        if isinstance(second, int):
            p["second_constant"] = second
        else:
            p["second_signal"] = second
        return self._ent("arith", p, _conns(ins, outs), **kw)

    def decider(self, conds: list, outs_spec: list, ins: dict, outs: dict, **kw):
        cs = []
        for i, c in enumerate(conds):
            d = {"first_signal": c[0], "comparator": c[1], "first_signal_networks": _nets(c[3] if len(c) > 3 else "rg"),
                 "second_signal_networks": _nets(c[4] if len(c) > 4 else "rg")}
            if isinstance(c[2], int):
                d["constant"] = c[2]
            else:
                d["second_signal"] = c[2]
            if i:
                d["compare_type"] = c[5] if len(c) > 5 else "and"
            cs.append(d)
        os_ = []
        for o in outs_spec:
            sig, copy = o[0], o[1]
            d = {"signal": sig, "copy_count_from_input": copy, "networks": _nets(o[3] if len(o) > 3 else "rg")}
            if not copy:
                d["constant"] = o[2] if len(o) > 2 else 1
            os_.append(d)
        return self._ent("decider", {"conditions": cs, "outputs": os_}, _conns(ins, outs), **kw)

    def const(self, signals, out: Net, **kw):
        sig = [(s, int(v)) for s, v in (signals.items() if isinstance(signals, dict) else signals) if int(v) != 0]
        return self._ent("const", {"signals": sig}, {("r" if out.color == "red" else "g"): out}, **kw)

    def selector(self, params, ins: dict, outs: dict, **kw):
        return self._ent("selector", params, _conns(ins, outs), **kw)

    def panel(self, inp: Net, records=None, **kw):
        rec = {"condition": {"first_signal": ANYTHING, "comparator": ">", "constant": 0}, "icon": ANYTHING, "text": ""}
        return self._ent("panel", {"records": records or [rec]}, {("r" if inp.color == "red" else "g"): inp}, **kw)

    def module(self, name) -> Module:
        return Module(self, name)

    # ------------------------------------------------------------ power
    def add_power(self):
        """Substations at every lane crossing (they also act as circuit relays: reach 18)."""
        subs = []
        for bx in range(self.nblocks_x + 1):
            for by in range(self.nblocks_y + 1):
                x = self.x0 + bx * self.BPX + 1.0
                y = self.y0 + by * self.BPY + 1.0
                subs.append(self._ent("substation", {}, {}, pos=(x, y), tag="power"))
                # wide lanes: crossings are more than 18 tiles apart, so add one mid-lane substation per segment
                if self.BPY > 18 and by < self.nblocks_y:
                    subs.append(self._ent("substation", {}, {}, pos=(x, y + 2 * ((self.BPY // 2) // 2)), tag="power"))
                if self.BPX > 18 and bx < self.nblocks_x:
                    subs.append(self._ent("substation", {}, {}, pos=(x + 2 * ((self.BPX // 2) // 2), y), tag="power"))
        return subs

    # ------------------------------------------------------------ routing
    def lane_spots(self):
        spots = []
        nbx, nby, L = self.nblocks_x, self.nblocks_y, self.LANE
        for bx in range(nbx + 1):          # vertical lanes (L/2 columns of 2x2 spots)
            for lc in range(L // 2):
                for by in range(nby):
                    for k in range(BROWS // 2):
                        spots.append((self.x0 + bx * self.BPX + 2 * lc + 1.0, self.y0 + L + by * self.BPY + 2 * k + 1.0))
        for by in range(nby + 1):          # horizontal lanes
            for lr in range(L // 2):
                for bx in range(nbx):
                    for k in range((BCOLS * SLOT_W) // 2):
                        spots.append((self.x0 + L + bx * self.BPX + 2 * k + 1.0, self.y0 + by * self.BPY + 2 * lr + 1.0))
        # spare 2x2 cells inside lane crossings (the substation takes the first one)
        for bx in range(nbx + 1):
            for by in range(nby + 1):
                for a in range(L // 2):
                    for b in range(L // 2):
                        if a or b:
                            spots.append((self.x0 + bx * self.BPX + 2 * a + 1.0, self.y0 + by * self.BPY + 2 * b + 1.0))
        spots += list(self.extra_spots)
        taken = {(round(e.pos[0], 1), round(e.pos[1], 1)) for e in self.ents if e.kind == "substation"}
        return np.array([s for s in spots if (round(s[0], 1), round(s[1], 1)) not in taken])

    def route(self):
        """Connect each net: MST over members; long edges are bridged with big poles in the lanes."""
        spots = self.lane_spots()
        used = {}  # spot index -> {"red": net id | None, "green": ...}
        pole_ent = {}
        stats = {"wires": 0, "poles": 0, "long_nets": 0}
        pos = {e.id: e.pos for e in self.ents}
        hal = {e.id: half(e) for e in self.ents}
        for net in self.nets:
            mem = net.members
            if len(mem) < 2:
                continue
            P = np.array([pos[e] for e, _ in mem])
            Hm = np.array([hal[e] for e, _ in mem])
            edges = _mst(P, Hm)
            comps = _components(len(mem), [(i, j) for i, j, d in edges if d <= REACH])
            for i, j, d in edges:
                if d <= REACH:
                    self.wires.append((mem[i][0], mem[i][1], mem[j][0], mem[j][1]))
                    stats["wires"] += 1
            if len(comps) == 1:
                continue
            stats["long_nets"] += 1
            # one pole per component, near one of its members
            cpoles = []
            for comp in comps:
                best = None
                for mi in comp:
                    dd = np.hypot(np.maximum(np.abs(spots[:, 0] - P[mi, 0]) - 1 - Hm[mi, 0], 0),
                                  np.maximum(np.abs(spots[:, 1] - P[mi, 1]) - 1 - Hm[mi, 1], 0))
                    order = np.argsort(dd)
                    for si in order[:400]:
                        if dd[si] > REACH:
                            break
                        if used.get(si, {}).get(net.color) is None:
                            if best is None or dd[si] < best[0]:
                                best = (dd[si], si, mi)
                            break
                if best is None:
                    raise RuntimeError(f"no pole spot for net {net.name} member {mem[comp[0]]}")
                _, si, mi = best
                pe = self._pole(si, spots, used, pole_ent, net, stats)
                self.wires.append((mem[mi][0], mem[mi][1], pe, COLOR_CONN[net.color]))
                stats["wires"] += 1
                cpoles.append(si)
            # connect the component poles with an MST; bridge long gaps with intermediate poles
            PP = spots[cpoles]
            def pd(u, v):
                d = np.maximum(np.abs(spots[u] - spots[v]) - 2, 0)
                return float(np.hypot(*d))
            for i, j, d in _mst(PP, np.ones_like(PP)):
                a, b = cpoles[i], cpoles[j]
                chain = [a]
                while pd(chain[-1], b) > POLE_REACH:
                    cur = spots[chain[-1]]
                    tgt = spots[b]
                    step = cur + (tgt - cur) / np.hypot(*(tgt - cur)) * (POLE_REACH - 2)
                    dd = np.hypot(spots[:, 0] - step[0], spots[:, 1] - step[1])
                    dcur = np.hypot(np.maximum(np.abs(spots[:, 0] - cur[0]) - 2, 0), np.maximum(np.abs(spots[:, 1] - cur[1]) - 2, 0))
                    ok = [s for s in np.argsort(dd)[:200]
                          if used.get(s, {}).get(net.color) is None and dcur[s] <= POLE_REACH and s not in chain]
                    if not ok:
                        raise RuntimeError(f"cannot bridge net {net.name}")
                    self._pole(ok[0], spots, used, pole_ent, net, stats)
                    chain.append(ok[0])
                chain.append(b)
                for u, v in zip(chain, chain[1:]):
                    self.wires.append((pole_ent[u], COLOR_CONN[net.color], pole_ent[v], COLOR_CONN[net.color]))
                    stats["wires"] += 1
        return stats

    def _pole(self, si, spots, used, pole_ent, net, stats):
        slot = used.setdefault(si, {"red": None, "green": None})
        if slot[net.color] is not None and slot[net.color] != net.id:
            raise RuntimeError("pole color clash")
        slot[net.color] = net.id
        if si not in pole_ent:
            e = self._ent("pole", {}, {}, pos=tuple(spots[si]), tag="relay")
            pole_ent[si] = e.id
            stats["poles"] += 1
        return pole_ent[si]

    # ------------------------------------------------------------ export
    def to_json(self):
        return {"ents": [{"id": e.id, "kind": e.kind, "params": e.params, "pos": e.pos, "tag": e.tag, "dir": e.dir} for e in self.ents],
                "wires": self.wires, "bbox": self.bbox()}


COLOR_CONN = {"red": "r", "green": "g"}


def _nets(s):
    return {"red": "r" in s, "green": "g" in s}


def _conns(ins, outs):
    d = {}
    for k, v in ins.items():
        d["in_" + k] = v
    for k, v in outs.items():
        d["out_" + k] = v
    return d


def _bd(P, Hh, j):
    dx = np.maximum(np.abs(P[:, 0] - P[j, 0]) - Hh[:, 0] - Hh[j, 0], 0)
    dy = np.maximum(np.abs(P[:, 1] - P[j, 1]) - Hh[:, 1] - Hh[j, 1], 0)
    return np.hypot(dx, dy)


def _mst(P, Hh=None):
    """Prim's MST on boxes (edge-to-edge distance); returns (i, j, dist) edges."""
    n = len(P)
    if n < 2:
        return []
    if Hh is None:
        Hh = np.zeros_like(P)
    in_tree = np.zeros(n, bool)
    in_tree[0] = True
    dist = _bd(P, Hh, 0)
    par = np.zeros(n, int)
    edges = []
    for _ in range(n - 1):
        d = np.where(in_tree, np.inf, dist)
        j = int(np.argmin(d))
        edges.append((int(par[j]), j, float(d[j])))
        in_tree[j] = True
        nd = _bd(P, Hh, j)
        upd = (~in_tree) & (nd < dist)
        dist[upd] = nd[upd]
        par[upd] = j
    return edges


def _components(n, edges):
    parent = list(range(n))

    def f(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in edges:
        parent[f(a)] = f(b)
    groups = {}
    for i in range(n):
        groups.setdefault(f(i), []).append(i)
    return list(groups.values())
