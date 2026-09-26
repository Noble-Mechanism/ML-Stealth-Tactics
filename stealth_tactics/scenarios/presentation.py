"""Spec 4 Red presentations (docs/specs/04-red-presentations.md).

A *presentation* is one Red flight (``n_red`` jets, default 6) placed relative
to a fixed Blue start: formation, start range / azimuth / altitude, flight
aggressiveness ``a`` and doctrine, and exactly one pre-planned maneuver with
its range trigger. It stores **every drawn value** (plus the Blue block and the
absolute Red start states), so a replay never re-runs the sampler (decision M).

Seeding (M):
- ``sample_presentation(seed)`` draws from named, independent sub-streams
  ``default_rng(SeedSequence([seed, PRESENTATION_SALT, crc32(name)]))``, so
  adding a field later does not shift the others. Nothing else is touched.
- ``sim_seed`` (sensor / datalink / RWR / Pk streams) is derived from the
  presentation seed, never from a genome's index: every genome facing the same
  presentation gets the same dice.
- ``build_eval_set(master_seed, gen, N)`` gives N stratified presentations
  (maneuver x aggressiveness band x doctrine cells, decision N);
  ``build_benchmark_set`` the fixed benchmark set.

Menus (formations, maneuvers) are data: ``presentation_menus.yaml``.
"""

from __future__ import annotations

import ast
import itertools
import json
import math
import operator
import zlib
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from stealth_tactics.sim.aircraft import Aircraft, AircraftState, _wrap_pi
from stealth_tactics.sim.world import SHOOT_ASSESS_SHOOT, SHOOT_SHOOT_ASSESS

NM_M = 1852.0
SAMPLER_VERSION = "spec4-v1"
PRESENTATION_SALT = 0x5EC4
SIM_SALT = 0x51A5
EVAL_SALT = 0xE7A1
BENCH_SALT = 0xBE9C
BANDS = ("conservative", "middle", "aggressive")
BAND_RANGES = {"conservative": (0.0, 1.0 / 3.0), "middle": (1.0 / 3.0, 2.0 / 3.0),
               "aggressive": (2.0 / 3.0, 1.0)}
DOCTRINES = (SHOOT_ASSESS_SHOOT, SHOOT_SHOOT_ASSESS)
MENUS_PATH = Path(__file__).with_name("presentation_menus.yaml")
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BLUE_SCENARIO = ROOT / "scenarios" / "default_4v3.yaml"
SLOT_SHAPES = {"single": [[0, 0]], "pair": [[0, 0], ["elem", 0]]}


# --------------------------------------------------------------- config --
@dataclass(frozen=True)
class PresentationConfig:
    """All sampling ranges (decisions B, D, F, G, I) and run settings."""
    n_red: int = 6                                  # B (Rusty: always 6; 8 later)
    range_nm: Tuple[float, float] = (40.0, 60.0)    # D: Blue lead -> Red lead
    azimuth_deg: Tuple[float, float] = (-40.0, 40.0)  # D: off Blue lead's nose
    base_alt_m: Tuple[float, float] = (6000.0, 12000.0)
    alt_clip_m: Tuple[float, float] = (1000.0, 13500.0)
    alt_jitter_m: float = 150.0
    red_speed_mps: float = 250.0                    # D: fixed (speed is cosmetic)
    band_weights: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)   # F
    ssa_probability: float = 0.5                    # G
    maneuver_weights: Optional[Tuple[Tuple[str, float], ...]] = None  # None = menu weights
    trigger_margin_nm: float = 5.0                  # I
    breakup_range_nm: float = 20.0                  # E
    max_time_s: float = 360.0                       # Rusty Q1: 240 -> 360 s
    dt: float = 0.5
    blue_scenario: str = str(DEFAULT_BLUE_SCENARIO)
    menus_path: str = str(MENUS_PATH)


DEFAULT_PRESENTATION_CONFIG = PresentationConfig()


# ---------------------------------------------------------------- menus --
@lru_cache(maxsize=8)
def load_menus(path: str = str(MENUS_PATH)) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for name, f in data["formations"].items():
        f.setdefault("params", {})
        f.setdefault("mirror", False)
        f.setdefault("description", name)
    for name, m in data["maneuvers"].items():
        m.setdefault("params", {})
        m.setdefault("weight", 1.0)
        m.setdefault("divide", "all")
    return data


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv}
_FUNCS = {"sind": lambda d: math.sin(math.radians(d)),
          "cosd": lambda d: math.cos(math.radians(d))}


def eval_expr(expr: Any, params: Dict[str, Any]) -> float:
    """Number or arithmetic expression over *params* (+ - * /, sind, cosd)."""
    if isinstance(expr, (int, float)):
        return float(expr)

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.Name):
            return float(params[n.id])
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id in _FUNCS and len(n.args) == 1):
            return _FUNCS[n.func.id](ev(n.args[0]))
        raise ValueError(f"bad formation expression {expr!r}")
    return ev(ast.parse(str(expr), mode="eval"))


def param_corners(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every combination of range endpoints / choices (for the box-bound test)."""
    keys = sorted(spec)
    opts = [v["choice"] if isinstance(v, dict) else [v[0], v[1]] for v in
            (spec[k] for k in keys)]
    return [dict(zip(keys, combo)) for combo in itertools.product(*opts)]


def draw_params(spec: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    out = {}
    for k in sorted(spec):                       # sorted: stable draw order
        v = spec[k]
        if isinstance(v, dict):
            ch = v["choice"]
            c = ch[int(rng.integers(len(ch)))]
            out[k] = c if isinstance(c, str) else (float(c) if isinstance(c, float) else int(c))
        else:
            out[k] = float(rng.uniform(v[0], v[1]))
    return out


def formation_size(fdef: dict) -> int:
    n = 0
    for g in fdef["groups"]:
        slots = g["slots"]
        n += len(SLOT_SHAPES[slots] if isinstance(slots, str) else slots)
    return n


def formations_for(n_red: int, menus: dict) -> List[str]:
    return sorted(k for k, f in menus["formations"].items() if formation_size(f) == n_red)


def realize_formation(fdef: dict, params: Dict[str, Any], mirror: int = 1) -> List[dict]:
    """Slot list: {slot, group, right_nm, back_nm, up_m}, lead at the origin."""
    out = []
    for g in fdef["groups"]:
        at = list(g["at"]) + [0] * (3 - len(g["at"]))
        ar, ab, au = (eval_expr(v, params) for v in at)
        slots = SLOT_SHAPES[g["slots"]] if isinstance(g["slots"], str) else g["slots"]
        for k, s in enumerate(slots, start=1):
            s = list(s) + [0] * (3 - len(s))
            r = ar + eval_expr(s[0], params)
            out.append({"slot": len(out) + 1, "group": g["name"], "ref": f"{g['name']}.{k}",
                        "right_nm": mirror * r, "back_nm": ab + eval_expr(s[1], params),
                        "up_m": au + eval_expr(s[2], params)})
    r0, b0, u0 = out[0]["right_nm"], out[0]["back_nm"], out[0]["up_m"]
    for o in out:                                 # lead (slot 1) at the origin
        o["right_nm"] -= r0
        o["back_nm"] -= b0
        o["up_m"] -= u0
        o["right_nm"] = o["right_nm"] + 0.0      # no -0.0 in JSON
    return out


def halves_of(fdef: dict, slots: List[dict]) -> List[List[int]]:
    """Two lists of slot numbers; the half with the smaller mean `right` first
    (it turns left in a split)."""
    halves = []
    for h in fdef["halves"]:
        ids = []
        for ref in h:
            ids += [s["slot"] for s in slots
                    if (s["ref"] == ref if "." in str(ref) else s["group"] == ref)]
        halves.append(sorted(ids))
    mean = [np.mean([slots[i - 1]["right_nm"] for i in h]) for h in halves]
    if mean[1] < mean[0] - 1e-9:
        halves = [halves[1], halves[0]]
    return halves


# ---------------------------------------------------------- presentation --
@dataclass
class Presentation:
    sampler_version: str
    presentation_seed: int
    sim_seed: int
    n_red: int
    formation: str
    formation_params: Dict[str, Any]
    mirror: int
    range_nm: float
    azimuth_deg: float
    base_alt_m: float
    band: str
    aggressiveness: float
    doctrine: str
    maneuver: Dict[str, Any]       # {type, kind, divide, trigger_range_nm, params, halves}
    red_jets: List[Dict[str, Any]]  # {id, name, slot, group, x, y, alt, heading_rad,
    #                                  speed_mps, hot_alt_m, right_nm, back_nm}
    blue: List[Dict[str, Any]]      # Blue block (scenario YAML entries)
    blue_scenario: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @staticmethod
    def from_dict(d: dict) -> "Presentation":
        return Presentation(**d)

    @staticmethod
    def from_json(s: str) -> "Presentation":
        return Presentation.from_dict(json.loads(s))

    def summary(self) -> str:
        m = self.maneuver
        mp = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                       for k, v in sorted(m["params"].items()))
        return (f"seed {self.presentation_seed}: {self.n_red} Red, {self.formation}"
                f"{' (mirrored)' if self.mirror < 0 else ''}, {self.range_nm:.1f} NM at "
                f"{self.azimuth_deg:+.1f} deg, base alt {self.base_alt_m:.0f} m; "
                f"a={self.aggressiveness:.2f} ({self.band}), {self.doctrine}; "
                f"maneuver {m['type']} at {m['trigger_range_nm']:.1f} NM ({mp})")


def _stream(seed: int, name: str) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), PRESENTATION_SALT, zlib.crc32(name.encode())]))


def derive_sim_seed(presentation_seed: int) -> int:
    return int(np.random.SeedSequence([int(presentation_seed), SIM_SALT])
               .generate_state(1, dtype=np.uint32)[0])


@lru_cache(maxsize=8)
def _blue_block(path: str) -> Tuple[dict, ...]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return tuple(dict(b) for b in data.get("blue", []))


def _weighted_choice(rng, names: Sequence[str], weights: Sequence[float]) -> str:
    w = np.asarray(weights, dtype=float)
    return names[int(rng.choice(len(names), p=w / w.sum()))]


def sample_presentation(seed: int, cfg: PresentationConfig = DEFAULT_PRESENTATION_CONFIG,
                        stratum: Optional[Tuple[str, str, str]] = None) -> Presentation:
    """Draw one presentation from *seed*. ``stratum`` = (maneuver, band,
    doctrine) fixes those three (evaluation-set cells)."""
    menus = load_menus(cfg.menus_path)
    seed = int(seed)
    # --- formation (C) ---
    rng = _stream(seed, "formation")
    fnames = formations_for(cfg.n_red, menus)
    if not fnames:
        raise ValueError(f"no formation with {cfg.n_red} slots in {cfg.menus_path}")
    formation = fnames[int(rng.integers(len(fnames)))]
    fdef = menus["formations"][formation]
    fparams = draw_params(fdef["params"], rng)
    mirror = (-1 if rng.random() < 0.5 else 1) if fdef["mirror"] else 1
    slots = realize_formation(fdef, fparams, mirror)
    halves = halves_of(fdef, slots)
    # --- geometry (D) ---
    rng = _stream(seed, "geometry")
    range_nm = float(rng.uniform(*cfg.range_nm))
    az_deg = float(rng.uniform(*cfg.azimuth_deg))
    base_alt = float(rng.uniform(*cfg.base_alt_m))
    jitter = [float(rng.uniform(-cfg.alt_jitter_m, cfg.alt_jitter_m)) for _ in slots]
    # --- aggressiveness (F) ---
    rng = _stream(seed, "aggressiveness")
    band = _weighted_choice(rng, BANDS, cfg.band_weights)
    u = float(rng.random())
    if stratum is not None:
        band = stratum[1]
    lo, hi = BAND_RANGES[band]
    a = lo + u * (hi - lo)
    if band != "aggressive":
        a = min(a, hi - 1e-9)                  # stay strictly inside the band
    # --- doctrine (G) ---
    rng = _stream(seed, "doctrine")
    doctrine = SHOOT_SHOOT_ASSESS if rng.random() < cfg.ssa_probability else SHOOT_ASSESS_SHOOT
    if stratum is not None:
        doctrine = stratum[2]
    # --- maneuver (H, I) ---
    rng = _stream(seed, "maneuver")
    mnames = sorted(menus["maneuvers"])
    wmap = dict(cfg.maneuver_weights) if cfg.maneuver_weights else {
        k: float(menus["maneuvers"][k]["weight"]) for k in mnames}
    mtype = _weighted_choice(rng, mnames, [wmap.get(k, 0.0) for k in mnames])
    if stratum is not None:
        mtype = stratum[0]
    mdef = menus["maneuvers"][mtype]
    trig_u = float(rng.random())
    mparams = draw_params(mdef["params"], rng)
    t_lo, t_hi = mdef["trigger_nm"]
    t_hi = min(t_hi, range_nm - cfg.trigger_margin_nm)
    t_lo = min(t_lo, t_hi)
    trigger = t_lo + trig_u * (t_hi - t_lo)

    # --- absolute start states ---
    blue = [dict(b) for b in _blue_block(cfg.blue_scenario)]
    b0 = blue[0]
    bx, by = float(b0.get("x", 0.0)), float(b0.get("y", -40000.0))
    bh = float(b0.get("heading_rad", 0.0))
    brg = bh + math.radians(az_deg)
    lx, ly = bx + range_nm * NM_M * math.sin(brg), by + range_nm * NM_M * math.cos(brg)
    h = _wrap_pi(math.atan2(bx - lx, by - ly))       # Red heading: at Blue lead
    fx, fy = math.sin(h), math.cos(h)
    rx, ry = math.cos(h), -math.sin(h)
    lo_alt, hi_alt = cfg.alt_clip_m
    red = []
    for s, j in zip(slots, jitter):
        r_m, b_m = s["right_nm"] * NM_M, s["back_nm"] * NM_M
        alt = float(min(hi_alt, max(lo_alt, base_alt + s["up_m"] + j)))
        red.append({"id": f"R{s['slot']}", "name": f"RedFighter-{s['slot']}",
                    "slot": s["slot"], "group": s["group"],
                    "x": lx + rx * r_m - fx * b_m, "y": ly + ry * r_m - fy * b_m,
                    "alt": alt, "heading_rad": h, "speed_mps": cfg.red_speed_mps,
                    "hot_alt_m": alt, "right_nm": s["right_nm"], "back_nm": s["back_nm"]})
    return Presentation(
        sampler_version=SAMPLER_VERSION, presentation_seed=seed,
        sim_seed=derive_sim_seed(seed), n_red=cfg.n_red, formation=formation,
        formation_params=fparams, mirror=mirror, range_nm=range_nm, azimuth_deg=az_deg,
        base_alt_m=base_alt, band=band, aggressiveness=a, doctrine=doctrine,
        maneuver={"type": mtype, "kind": mdef["kind"], "divide": mdef["divide"],
                  "trigger_range_nm": trigger, "params": mparams, "halves": halves},
        red_jets=red, blue=blue, blue_scenario=Path(cfg.blue_scenario).name)


def build_presentation_aircraft(p: Presentation) -> List[Aircraft]:
    """Blue block + Red flight as Aircraft (YAML path is untouched)."""
    out: List[Aircraft] = []
    for i, spec in enumerate(p.blue):
        st = AircraftState(x=float(spec.get("x", 0)), y=float(spec.get("y", -40000 + i * 500)),
                           alt=float(spec.get("alt", 9000)),
                           heading_rad=float(spec.get("heading_rad", 0.0)),
                           speed_mps=float(spec.get("speed_mps", 260)))
        out.append(Aircraft.make_blue(spec.get("id", f"B{i + 1}"),
                                      spec.get("name", f"F-35-{i + 1}"), st))
    for r in p.red_jets:
        st = AircraftState(x=r["x"], y=r["y"], alt=r["alt"], heading_rad=r["heading_rad"],
                           speed_mps=r["speed_mps"])
        ac = Aircraft.make_red(r["id"], r["name"], st)
        ac.firing_doctrine = p.doctrine                  # G: per jet
        out.append(ac)
    return out


# ---------------------------------------------------------- eval sets (N) --
def strata(cfg: PresentationConfig = DEFAULT_PRESENTATION_CONFIG) -> List[Tuple[str, str, str]]:
    menus = load_menus(cfg.menus_path)
    return list(itertools.product(sorted(menus["maneuvers"]), BANDS, DOCTRINES))


def _set(master_seed: int, salt: int, idx: int, n: int,
         cfg: PresentationConfig) -> List[Presentation]:
    ss = np.random.SeedSequence([int(master_seed), salt, int(idx)])
    seeds = [int(s) for s in ss.generate_state(n, dtype=np.uint64)]
    cells = strata(cfg)
    order: List[Tuple[str, str, str]] = []
    rng = np.random.default_rng(np.random.SeedSequence([int(master_seed), salt, int(idx), 1]))
    while len(order) < n:                      # full crossings; a partial last one
        perm = rng.permutation(len(cells))     # is a random subset of cells
        order += [cells[i] for i in perm]
    return [sample_presentation(s, cfg, stratum=c) for s, c in zip(seeds, order[:n])]


def build_eval_set(master_seed: int, gen: int, n: int = 24,
                   cfg: PresentationConfig = DEFAULT_PRESENTATION_CONFIG) -> List[Presentation]:
    """N presentations for generation *gen* (same for every genome; resampled
    per generation). N = 24 is one full maneuver x band x doctrine crossing."""
    return _set(master_seed, EVAL_SALT, gen, n, cfg)


def build_benchmark_set(master_seed: int, n: int = 64,
                        cfg: PresentationConfig = DEFAULT_PRESENTATION_CONFIG
                        ) -> List[Presentation]:
    """Fixed benchmark set (scores each generation's best for charts and
    champion selection)."""
    return _set(master_seed, BENCH_SALT, 0, n, cfg)
