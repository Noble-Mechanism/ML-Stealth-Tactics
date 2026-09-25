"""Spec 2 scripted replays: A 'launch on remote', B 'passive triangulation'.

Replay A: lead F-35 (B1) flies at one Red Su-27 and holds it on radar FC. The
wingman (B2) starts 20 NM abeam (and 15 NM forward, so it reaches missile
range while the lead still holds FC), flies a heading that keeps Red 65 deg off its
nose (outside its +/-60 deg radar/IRST field of regard). When it holds the
lead's FC track over the link and Red is within 0.95 x table Rmax for the
shot geometry (Red 50 deg off the nose; the table has a launch off-nose axis),
it turns in just enough to put Red 50 deg off the nose (inside the 60 deg launch limit
that still applies), fires ONE missile on the remote track (its own radar needs
>= 3 s to build FC, so the shot is necessarily on the lead's track), then turns
cold. The lead keeps supporting until the missile is within 15 NM. Blue
missiles only; Red weapons off.

Replay B: two F-35s start 4 NM apart 42 NM behind a Red flying away (hot tail
visible to IRST at long range), splay out to 20 NM lateral, then fly parallel,
closing on Red. Their IRST bearings (own + received over the link) are
triangulated; the fix is drawn in TacView as a separate 'TRI est' waypoint
object. Radars stay on (Spec 1), so fusion usually prefers the radar track: the
fused source is logged over time.

Replay C (lead-trail): two F-35s 15 NM in trail, both hot on Red. The lead fires
at the earliest valid own-FC shot and turns cold; the trail may take over
midcourse support via handoff. Variants: 'no-support' (shooter-only support)
and 'delayed' (the lead waits for the trail's FC before shooting).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from stealth_tactics.acmi.exporter import ACMIExporter
from stealth_tactics.sim.aircraft import Aircraft, AircraftState, _wrap_pi, bearing_to
from stealth_tactics.sim.sensor_config import NM_M
from stealth_tactics.sim.missile_envelope import off_nose_signed
from stealth_tactics.sim.sensors import distance_3d
from stealth_tactics.sim.world import SimConfig, World
from stealth_tactics.sim.weapons import OUTCOMES

TIMELINE_WEAPON_EVENTS = ("launch", "support_handoff", "autonomous", "support_lost",
                          "support_regained", "support_dropped", "burnout") + OUTCOMES

RED_TYPE_NAME = "Su-27"
DEG = math.pi / 180.0


# ======================================================== Replay A ==========
@dataclass
class ReplayA:
    world: World
    seed: int
    fired_t: Optional[float] = None
    outcome: str = ""
    shot_range_m: Optional[float] = None     # B2-R1 at launch
    shot_rmax_m: Optional[float] = None      # table Rmax at launch (shot geometry)
    shot_off_nose_deg: Optional[float] = None
    mach_end: Optional[float] = None         # missile Mach at fuze / defeat
    pk_final: Optional[float] = None
    timeline: List[dict] = field(default_factory=list)


WING_OFF_DEG = 65.0     # Red kept this far off B2's nose (outside the 60 deg FOR)
SHOT_OFF_DEG = 50.0     # turn-in for the shot (inside the 60 deg launch limit)


def _shot_geometry_rmax(w: World, shooter: Aircraft, target: Aircraft,
                        off_deg: float) -> float:
    """Table Rmax for the shot B2 is about to take: same position and speed,
    nose turned to put the target ``off_deg`` off (the envelope table has a
    launch off-nose axis, 3a fix), rather than the current wide heading."""
    import copy
    hyp = copy.copy(shooter.state)
    hyp.heading_rad = _wrap_pi(bearing_to(shooter.state, target.state) - off_deg * DEG)
    return w.weapons.envelope.for_states(hyp, target.state)[0]


def run_launch_on_remote(seed: int = 1, record: bool = True, shot_frac: float = 0.95,
                         start_nm: float = 60.0, offset_nm: float = 20.0,
                         ahead_nm: float = 15.0) -> ReplayA:
    b1 = Aircraft.make_blue("B1", "F-35-1", AircraftState(0.0, 0.0, 9000.0, 0.0, 260.0))
    b2 = Aircraft.make_blue("B2", "F-35-2",
                            AircraftState(offset_nm * NM_M, ahead_nm * NM_M, 9000.0,
                                          0.0, 260.0))
    r1 = Aircraft.make_red("R1", "Red-1",
                           AircraftState(0.0, start_nm * NM_M, 8500.0, math.pi, 255.0))
    b2.state.heading_rad = _wrap_pi(bearing_to(b2.state, r1.state) - WING_OFF_DEG * DEG)
    b2.cmd_heading_rad = b2.state.heading_rad
    r1.type_name = RED_TYPE_NAME
    st = {"phase": "wide", "fired_t": None}
    notes: List[dict] = []

    def note(w: World, jet: str, text: str, rng: float) -> None:
        ev = {"t": w.time_s, "type": "note", "observer": jet, "target": "R1",
              "range_m": rng, "text": text}
        notes.append(ev)
        w._frame_event(ev)

    def blue_ctl(w: World) -> None:
        # Lead: pure pursuit on Red at cruise
        if b1.state.alive:
            b1.cmd_heading_rad = bearing_to(b1.state, r1.state)
            b1.cmd_speed_mps, b1.cmd_alt_m = 260.0, 9000.0
        if not b2.state.alive:
            return
        brg = bearing_to(b2.state, r1.state)
        rng = distance_3d(b2.state, r1.state)
        b2.cmd_speed_mps, b2.cmd_alt_m = 260.0, 9000.0
        if st["phase"] == "wide":
            b2.cmd_heading_rad = _wrap_pi(brg - WING_OFF_DEG * DEG)  # Red outside FOR
            src = w.fire_control_source(b2, r1.id)
            # shot_frac (0.95) x table Rmax for the shot geometry (Red 50 deg off
            # the nose after the turn-in; 3a fix: the table has an off-nose axis)
            if (src is not None and src != b2.id
                    and rng <= shot_frac * _shot_geometry_rmax(w, b2, r1, SHOT_OFF_DEG)):
                st["phase"] = "shoot"
                note(w, "B2", f"F-35-2 turns in for shot on {src}'s FC track "
                              f"(Red to {SHOT_OFF_DEG:.0f} deg off nose)", rng)
        if st["phase"] == "shoot":
            if b2.ammo < b2.params.ammo:          # missile away -> turn cold
                st["phase"], st["fired_t"] = "cold", w.time_s
                note(w, "B2", "F-35-2 turns cold after launch", rng)
            else:
                b2.cmd_heading_rad = _wrap_pi(brg - SHOT_OFF_DEG * DEG)
                b2.cmd_fire, b2.fire_target = True, r1.id
                # geometry at the moment of the (possible) launch this step
                st["pre"] = (rng, w.weapons.rmax_m(b2, r1),
                             off_nose_signed(b2.state, r1.state))
        if st["phase"] == "cold":
            b2.cmd_heading_rad = _wrap_pi(brg + math.pi)
            b2.cmd_speed_mps = b2.params.max_speed_mps

    def red_ctl(w: World) -> None:  # fly at the lead; weapons off
        tgt = b1 if b1.state.alive else b2
        r1.cmd_heading_rad = bearing_to(r1.state, tgt.state)
        r1.cmd_speed_mps, r1.cmd_alt_m = 255.0, 8500.0

    w = World([b1, b2, r1], SimConfig(dt=0.5, max_time_s=240.0, seed=seed),
              blue_ctl, red_ctl, record=record)
    res = w.run()
    rep = ReplayA(w, seed, st["fired_t"])
    if w.missiles:
        m0 = w.missiles[0]
        rep.shot_range_m = m0.launch_range_m
        rep.mach_end, rep.pk_final = m0.mach_end, m0.pk_final
        if st.get("pre") is not None:
            rep.shot_rmax_m, rep.shot_off_nose_deg = st["pre"][1], st["pre"][2]
    outs = [e["type"] for e in res.events if e["type"] in OUTCOMES]
    rep.outcome = outs[0] if outs else ("no_shot" if not any(
        e["type"] == "launch" for e in res.events) else "unresolved")
    keep_sensor = {("B1", "radar_detect"), ("B1", "fc_track"), ("B1", "fc_lost"),
                   ("B2", "link_track"), ("B2", "radar_detect"), ("B2", "fc_track"),
                   ("B2", "rwr_detect"), ("R1", "radar_detect"), ("R1", "fc_track")}
    seen = set()
    tl = []
    for e in res.sensor_events:
        k = (e["observer"], e["type"])
        if k in keep_sensor and k not in seen:
            seen.add(k)
            tl.append(e)
    for e in res.events:
        if e["type"] in TIMELINE_WEAPON_EVENTS:
            tl.append(e)
    tl += notes
    rep.timeline = sorted(tl, key=lambda e: e["t"])
    return rep


def describe_a(rep: ReplayA) -> str:
    w = rep.world
    by = {a.id: a for a in w.aircraft}
    lines = [f"Replay A: launch on remote (seed {rep.seed}) -> outcome: {rep.outcome.upper()}",
             "  B1 lead flies at Red; B2 20 NM abeam (15 NM forward) keeps Red 65 deg"
             " off its nose;"
             " Red (Su-27) flies at B1, weapons off; F-35 260 m/s, Red 255 m/s"]
    for e in rep.timeline:
        et = e["type"]
        if et == "launch":
            src = e.get("remote_source") or "own"
            txt = f"{e['shooter']} LAUNCH {e['missile']} (FC source: {src})"
        else:
            txt = e.get("text", et)
        rng = e.get("range_m")
        rs = f"  [{rng / NM_M:5.1f} NM]" if rng is not None and et not in ("hit", "miss") else ""
        lines.append(f"  t={e['t']:6.1f}s  {txt}{rs}")
    return "\n".join(lines)


def hit_rate_a(seeds: int = 100, shot_frac: float = 0.95) -> dict:
    out: Dict[str, int] = {}
    fired_ranges, handoffs = [], 0
    mach_all, mach_fz, pk_fz, off_deg, rmax_l = [], [], [], [], []
    for s in range(seeds):
        rep = run_launch_on_remote(seed=s, record=False, shot_frac=shot_frac)
        out[rep.outcome] = out.get(rep.outcome, 0) + 1
        if rep.shot_range_m is not None:
            fired_ranges.append(rep.shot_range_m / NM_M)
        if rep.mach_end is not None:
            mach_all.append(rep.mach_end)
        if rep.pk_final is not None:
            mach_fz.append(rep.mach_end)
            pk_fz.append(rep.pk_final)
        if rep.shot_off_nose_deg is not None:
            off_deg.append(rep.shot_off_nose_deg)
            rmax_l.append(rep.shot_rmax_m / NM_M)
        ev = rep.world.events
        L = [e for e in ev if e["type"] == "launch"]
        if L:
            if L[0].get("remote_source") == "B1":
                handoffs += any(e["type"] == "support_handoff" and e["observer"] == "B1"
                                for e in ev)
    med = (lambda xs: float(np.median(xs)) if xs else None)
    return {"seeds": seeds, "outcomes": out, "remote_launch_with_handoff": handoffs,
            "median_shot_nm": med(fired_ranges), "median_mach_end": med(mach_all),
            "median_mach_fuze": med(mach_fz), "median_pk_final": med(pk_fz),
            "median_off_nose_deg": med(off_deg), "median_rmax_nm": med(rmax_l)}


def shot_frac_table(fracs, seeds: int = 100) -> str:
    """Replay A hit rate vs shot range (fraction of the table Rmax)."""
    lines = [f"Replay A (launch on remote + handoff) vs shot range, {seeds} seeds each "
             "(Spec 3a missile, 3a fix: off-nose Rmax table, 40 s coast, 180 s cap).",
             "shot_frac = fraction of the table Rmax for the shot geometry (Red 50 deg off "
             "B2's nose after the turn-in) at which F-35-2 fires on B1's FC track.",
             "Old missile (Spec 2/2b): shot at 0.95 x 45 km = 23.1 NM, hit 55/100. "
             "Spec 3a before the fix: 0.95 -> 0 % (29.0 NM, 4-axis table ignored the "
             "50 deg off-nose), 0.85 -> 29 %, 0.75 -> 35 %, 0.65 -> 39 %.",
             f"{'shot_frac':>9} {'hit':>5} {'med shot NM':>11} {'off deg':>7} "
             f"{'med M@end':>9} {'med Pk':>6}  outcomes"]
    for fr in fracs:
        hr = hit_rate_a(seeds, shot_frac=fr)
        n = hr["seeds"]
        f2 = (lambda v, fmt: fmt.format(v) if v is not None else "-")
        lines.append(
            f"{fr:9.2f} {hr['outcomes'].get('hit', 0) / n * 100:4.0f}% "
            f"{f2(hr['median_shot_nm'], '{:.1f}'):>11} "
            f"{f2(hr['median_off_nose_deg'], '{:.0f}'):>7} "
            f"{f2(hr['median_mach_end'], '{:.2f}'):>9} "
            f"{f2(hr['median_pk_final'], '{:.2f}'):>6}  "
            + ", ".join(f"{k} {v}" for k, v in sorted(hr["outcomes"].items(),
                                                      key=lambda kv: -kv[1])))
    lines.append("med M@end = missile Mach at fuze or defeat (all shots); med Pk = final "
                 "Pk over fuzed shots.")
    return "\n".join(lines) + "\n"


# ======================================================== Replay C ==========
@dataclass
class ReplayC:
    world: World
    seed: int
    variant: str
    fired_t: Optional[float] = None
    shot_range_m: Optional[float] = None
    outcome: str = ""
    lapse_t: Optional[float] = None          # first time support lapsed (missile coasts)
    lapse_missile_rng_m: Optional[float] = None
    trail_fc_t: Optional[float] = None       # first time trail holds its own FC track
    trail_qual_t: Optional[float] = None     # first time trail could support (FC + 90 deg)
    handoff_to_trail_t: Optional[float] = None   # handoff or regain by B2
    autonomous_t: Optional[float] = None
    pk_factor: Optional[float] = None        # at the autonomous point
    t_unsupported: float = 0.0               # cumulative unsupported time (the gap)
    aim_err_m: Optional[float] = None
    mach_end: Optional[float] = None         # Spec 3a: missile Mach at fuze / defeat
    pk_final: Optional[float] = None         # Spec 3a: final Pk at the fuze
    timeline: List[dict] = field(default_factory=list)
    samples: Dict[float, dict] = field(default_factory=dict)

    @property
    def support_gap(self) -> bool:
        return self.lapse_t is not None

    @property
    def gap_s(self) -> Optional[float]:
        """Cumulative time the missile flew unsupported before autonomy."""
        return self.t_unsupported if self.lapse_t is not None else None


LT_VARIANTS = {
    "support": "trail supports via handoff (spec 2 Blue policy)",
    "no-support": "nobody supports after the lead turns out (trail does not support)",
    "delayed": "lead waits until trail has its own FC track, then shoots and turns out",
}


def run_lead_trail(seed: int = 1, record: bool = True, variant: str = "support",
                   start_nm: float = 60.0, trail_nm: float = 15.0) -> ReplayC:
    """Two F-35s in 15 NM lead-trail, both hot on a Red flying at the lead.

    Lead (B1) fires ONE missile on its own FC track at the earliest valid shot
    (existing launch rules: own FC, range <= table Rmax, <= 60 deg off nose),
    then turns cold (180, max speed). Trail (B2) stays hot on Red.
    """
    assert variant in LT_VARIANTS
    b1 = Aircraft.make_blue("B1", "F-35-1", AircraftState(0.0, 0.0, 9000.0, 0.0, 260.0))
    b2 = Aircraft.make_blue("B2", "F-35-2",
                            AircraftState(0.0, -trail_nm * NM_M, 9000.0, 0.0, 260.0))
    r1 = Aircraft.make_red("R1", "Red-1",
                           AircraftState(0.0, start_nm * NM_M, 8500.0, math.pi, 255.0))
    r1.type_name = RED_TYPE_NAME
    st = {"phase": "hot", "notes": [], "lead_out_logged": False}
    rep_box: Dict[str, ReplayC] = {}

    def note(w: World, jet: str, text: str, rng: float, etype: str = "note") -> None:
        ev = {"t": w.time_s, "type": etype, "observer": jet, "target": "R1",
              "range_m": rng, "text": text}
        st["notes"].append(ev)
        w._frame_event(ev)

    def blue_ctl(w: World) -> None:
        rep = rep_box["rep"]
        t = w.time_s
        m = w.missiles[0] if w.missiles else None
        rep.samples[round(t, 2)] = {
            "b1": distance_3d(b1.state, r1.state), "b2": distance_3d(b2.state, r1.state),
            "m": (distance_3d(m.pos, r1.state) if m is not None else None)}
        st2 = w.tracks.get("B2")
        if rep.trail_fc_t is None and st2 is not None and st2.is_fire_control("R1"):
            rep.trail_fc_t = t
            note(w, "B2", "F-35-2 (trail) now holds its OWN fire-control track on R1",
                 distance_3d(b2.state, r1.state))
        if (m is not None and rep.trail_qual_t is None
                and w.weapons.has_midcourse_support(b2, r1, w.tracks)):
            rep.trail_qual_t = t
        # Trail: always hot on Red
        b2.cmd_heading_rad = bearing_to(b2.state, r1.state)
        b2.cmd_speed_mps, b2.cmd_alt_m = 260.0, 9000.0
        # Lead
        brg = bearing_to(b1.state, r1.state)
        rng = distance_3d(b1.state, r1.state)
        b1.cmd_speed_mps, b1.cmd_alt_m = 260.0, 9000.0
        if st["phase"] == "hot":
            b1.cmd_heading_rad = brg
            if b1.ammo < b1.params.ammo:
                st["phase"] = "cold"
            else:
                own_fc = w.tracks["B1"].is_fire_control("R1")
                ready = own_fc and (variant != "delayed" or rep.trail_fc_t is not None)
                if ready:
                    b1.cmd_fire, b1.fire_target = True, r1.id
        if st["phase"] == "cold":
            if not st["lead_out_logged"]:
                st["lead_out_logged"] = True
                note(w, "B1", "F-35-1 (lead) turns out: 180 to cold, max speed", rng)
            b1.cmd_heading_rad = _wrap_pi(brg + math.pi)
            b1.cmd_speed_mps = b1.params.max_speed_mps

    def red_ctl(w: World) -> None:  # fly at the lead; weapons off
        r1.cmd_heading_rad = bearing_to(r1.state, b1.state)
        r1.cmd_speed_mps, r1.cmd_alt_m = 255.0, 8500.0

    w = World([b1, b2, r1], SimConfig(dt=0.5, max_time_s=240.0, seed=seed),
              blue_ctl, red_ctl, record=record)
    if variant == "no-support":
        # Replay-level override (no sim change): nobody supports once the lead
        # has turned out (before that only the shooter, as usual).
        def nobody_after_turn(m, target, by_id, tracks):
            if st["phase"] == "cold":
                return None
            return m.shooter_id if w.weapons._qualifies(by_id.get(m.shooter_id), target,
                                                        m, tracks) else None
        w.weapons.pick_supporter = nobody_after_turn
    rep = ReplayC(w, seed, variant)
    rep_box["rep"] = rep
    res = w.run()
    for e in res.events:
        et = e["type"]
        if et == "launch" and rep.fired_t is None:
            rep.fired_t = e["t"]
            smp = rep.samples.get(round(e["t"], 2))
            rep.shot_range_m = smp["b1"] if smp else None
        elif et == "support_lost" and rep.lapse_t is None:
            rep.lapse_t, rep.lapse_missile_rng_m = e["t"], e.get("range_m")
        elif et in ("support_handoff", "support_regained") and e.get("observer") == "B2":
            rep.handoff_to_trail_t = rep.handoff_to_trail_t or e["t"]
        elif et == "autonomous":
            rep.autonomous_t, rep.pk_factor = e["t"], e.get("pk_factor")
    if w.missiles:
        rep.t_unsupported = w.missiles[0].t_unsupported
        rep.aim_err_m = w.missiles[0].aim_err_at_auto_m
        rep.mach_end = w.missiles[0].mach_end
        rep.pk_final = w.missiles[0].pk_final
    outs = [e["type"] for e in res.events if e["type"] in OUTCOMES]
    rep.outcome = outs[0] if outs else ("no_shot" if rep.fired_t is None else "unresolved")
    keep_sensor = {("B1", "radar_detect"), ("B1", "fc_track"), ("B2", "radar_detect"),
                   ("B2", "link_track"), ("R1", "radar_detect")}
    seen, tl = set(), []
    for e in res.sensor_events:
        k = (e["observer"], e["type"])
        if k in keep_sensor and k not in seen:
            seen.add(k)
            tl.append(e)
    for e in res.events:
        if e["type"] in TIMELINE_WEAPON_EVENTS:
            tl.append(e)
    tl += st["notes"]
    rep.timeline = sorted(tl, key=lambda e: e["t"])
    return rep


def _near_sample(rep: ReplayC, t: float) -> Optional[dict]:
    if not rep.samples:
        return None
    k = min(rep.samples, key=lambda x: abs(x - t))
    return rep.samples[k]


def describe_c(rep: ReplayC) -> str:
    lines = [f"Replay C: lead-trail launch + handoff (seed {rep.seed}, variant "
             f"'{rep.variant}': {LT_VARIANTS[rep.variant]}) -> outcome: "
             f"{rep.outcome.upper()}",
             "  B1 lead and B2 trail (15 NM directly behind, same track) hot at Red "
             "(Su-27, flies at B1, weapons off); F-35 260 m/s, Red 255 m/s",
             f"  {'t (s)':>6}  {'event':<96} {'B1-R1':>6} {'B2-R1':>6} {'M-R1':>6}  (NM)"]
    for e in rep.timeline:
        et = e["type"]
        if et == "launch":
            src = e.get("remote_source") or "own FC"
            txt = f"{e['shooter']} LAUNCH {e['missile']} ({src})"
        elif et == "support_handoff" and e.get("previous") is None:
            txt = e.get("text", et) + " (at launch)"
        else:
            txt = e.get("text", et)
        smp = _near_sample(rep, e["t"])
        if smp and et == "launch" and smp["m"] is None:
            smp = dict(smp, m=smp["b1"])       # missile leaves the shooter's rail
        f = (lambda v: f"{v / NM_M:6.1f}" if v is not None else f"{'-':>6}")
        rs = (f"{f(smp['b1'])} {f(smp['b2'])} {f(smp['m'])}" if smp else "")
        lines.append(f"  {e['t']:6.1f}  {txt[:96]:<96} {rs}")
    if rep.support_gap:
        lines.append(
            f"  SUPPORT GAP: support lapsed at t={rep.lapse_t:.1f}s (missile "
            f"{(rep.lapse_missile_rng_m or 0) / NM_M:.1f} NM from R1); missile coasted on "
            f"the extrapolated aim point, cumulative unsupported {rep.t_unsupported:.1f} s; "
            + (f"trail took over at t={rep.handoff_to_trail_t:.1f}s; "
               if rep.handoff_to_trail_t else "trail never took over; ")
            + (f"Pk factor at autonomy {rep.pk_factor:.2f}" if rep.pk_factor is not None
               else f"outcome {rep.outcome}")
            + (f", aim error at autonomy {rep.aim_err_m:.0f} m" if rep.aim_err_m is not None
               else "") + ".")
    return "\n".join(lines)


def stats_lead_trail(seeds: int = 100, variant: str = "support") -> dict:
    outs: Dict[str, int] = {}
    gaps, gap_n, shot_r, handoffs, pkf, trail_fc, aim = [], 0, [], 0, [], [], []
    pk_fin, mach_end = [], []
    for s in range(seeds):
        rep = run_lead_trail(seed=s, record=False, variant=variant)
        outs[rep.outcome] = outs.get(rep.outcome, 0) + 1
        if rep.shot_range_m is not None:
            shot_r.append(rep.shot_range_m / NM_M)
        if rep.support_gap:
            gap_n += 1
            gaps.append(rep.t_unsupported)
        handoffs += rep.handoff_to_trail_t is not None
        if rep.pk_factor is not None:
            pkf.append(rep.pk_factor)
        if rep.aim_err_m is not None:
            aim.append(rep.aim_err_m)
        if rep.pk_final is not None:
            pk_fin.append(rep.pk_final)
            mach_end.append(rep.mach_end)
        if rep.trail_fc_t is not None:
            smp = _near_sample(rep, rep.trail_fc_t)
            trail_fc.append((rep.trail_fc_t, smp["b2"] / NM_M))
    med = (lambda xs: float(np.median(xs)) if xs else None)
    return {"variant": variant, "seeds": seeds, "outcomes": outs,
            "hit_rate": outs.get("hit", 0) / seeds,
            "gap_frac": gap_n / seeds, "median_gap_s": med(gaps),
            "lost_timeout_frac": outs.get("lost_coast_timeout", 0) / seeds,
            "lost_basket_frac": outs.get("lost_basket", 0) / seeds,
            "handoff_to_trail_frac": handoffs / seeds,
            "median_pk_factor": med(pkf), "median_aim_err_m": med(aim),
            "median_pk_final": med(pk_fin), "median_mach_end": med(mach_end),
            "median_shot_nm": med(shot_r),
            "median_trail_fc_t": med([a for a, _ in trail_fc]),
            "median_trail_fc_nm": med([b for _, b in trail_fc])}


def comparison_table(stats: List[dict]) -> str:
    hdr = (f"{'variant':<11} {'hit':>5} {'handoff->B2':>11} {'gap':>5} {'med gap s':>9} "
           f"{'lost:timeout':>12} {'lost:basket':>11} {'med Pk fac':>10} {'med shot NM':>11} "
           f"{'trail FC t/NM':>13} {'med M@fuze':>10} {'med Pk':>6}  outcomes")
    out = [hdr, "-" * len(hdr)]
    f1 = (lambda v, fmt: fmt.format(v) if v is not None else "-")
    for s in stats:
        oc = ", ".join(f"{k} {v}" for k, v in sorted(s["outcomes"].items()))
        tf = (f"{s['median_trail_fc_t']:.0f}s/{s['median_trail_fc_nm']:.1f}"
              if s.get("median_trail_fc_t") is not None else "-")
        out.append(
            f"{s['variant']:<11} {s['hit_rate'] * 100:4.0f}% "
            f"{s['handoff_to_trail_frac'] * 100:10.0f}% {s['gap_frac'] * 100:4.0f}% "
            f"{f1(s['median_gap_s'], '{:.1f}'):>9} {s['lost_timeout_frac'] * 100:11.0f}% "
            f"{s['lost_basket_frac'] * 100:10.0f}% {f1(s['median_pk_factor'], '{:.2f}'):>10} "
            f"{f1(s['median_shot_nm'], '{:.1f}'):>11} {tf:>13} "
            f"{f1(s.get('median_mach_end'), '{:.2f}'):>10} "
            f"{f1(s.get('median_pk_final'), '{:.2f}'):>6}  {oc}")
    return "\n".join(out)


# ======================================================== Replay B ==========
@dataclass
class ReplayB:
    world: World
    seed: int
    rows: List[dict] = field(default_factory=list)
    desc: str = ""


def run_triangulation(seed: int = 1, record: bool = True, start_nm: float = 36.0,
                      chase_mps: float = 300.0, splay_deg: float = 10.0,
                      final_sep_nm: float = 24.0, duration_s: float = 600.0) -> ReplayB:
    # Red flies away (north) at cruise 255 m/s (hot tail); F-35s chase slightly faster so
    # range stays ~26-36 NM (inside tail-aspect IRST range) while the lateral split grows
    y0 = 0.0
    half0 = 2.0 * NM_M          # start 4 NM apart
    half_final = final_sep_nm / 2 * NM_M    # splay to final lateral separation
    b1 = Aircraft.make_blue("B1", "F-35-1", AircraftState(-half0, y0, 9000.0, 0.0, chase_mps))
    b2 = Aircraft.make_blue("B2", "F-35-2", AircraftState(half0, y0, 9000.0, 0.0, chase_mps))
    r1 = Aircraft.make_red("R1", "Red-1",
                           AircraftState(0.0, start_nm * NM_M, 8500.0, 0.0, 255.0))
    r1.type_name = RED_TYPE_NAME
    splay = splay_deg * DEG

    def blue_ctl(w: World) -> None:
        for ac, sign in ((b1, -1.0), (b2, 1.0)):
            ac.cmd_speed_mps, ac.cmd_alt_m = chase_mps, 9000.0
            if abs(ac.state.x) < half_final:
                ac.cmd_heading_rad = sign * splay
            else:
                ac.cmd_heading_rad = 0.0

    def red_ctl(w: World) -> None:
        r1.cmd_heading_rad, r1.cmd_speed_mps, r1.cmd_alt_m = 0.0, 255.0, 8500.0

    w = World([b1, b2, r1], SimConfig(dt=0.5, max_time_s=duration_s, seed=seed),
              blue_ctl, red_ctl, record=record)
    rep = ReplayB(w, seed)
    rep.desc = (f"Two F-35s start 4 NM apart {start_nm:.0f} NM behind a Red flying away "
                f"(255 m/s, hot tail), chase at {chase_mps:.0f} m/s, splay +/-{splay_deg:.0f} "
                f"deg to {final_sep_nm:.0f} NM lateral, then fly parallel ({duration_s:.0f} s).")
    state = {"had_fix": False, "last_status": -1e9}

    def hook(world: World, frame: dict) -> None:
        store = world.tracks.get("B1")
        f = store.fused.get("R1") if store else None
        t = world.time_s
        truth = r1.state.position()
        rng_true = distance_3d(b1.state, r1.state)
        tri = f.tri if f else None
        u1 = r1.state.x - b1.state.x, r1.state.y - b1.state.y
        u2 = r1.state.x - b2.state.x, r1.state.y - b2.state.y
        true_ang = math.degrees(abs(_wrap_pi(math.atan2(*u1) - math.atan2(*u2))))
        tr = store.get("R1") if store else None
        own_irst = tr is not None and "irst" in tr.components
        b2_irst = tr is not None and any("irst" in c for c in tr.remote.values())
        row = {"t": t, "true_range_m": rng_true, "true_angle_deg": true_ang,
               "fused_sensor": f.sensor if f else "-",
               "fused_src": f.source_jet if f else "-",
               "fused_sigma_m": f.sigma_m if f else float("nan"),
               "fused_err_m": float(np.linalg.norm(f.est_pos - truth)) if f else float("nan"),
               "radar_track": tr is not None and "radar" in tr.components,
               "own_irst": own_irst, "b2_irst_rx": b2_irst,
               "tri": None}
        if tri is not None:
            est = tri["pos"]
            row["tri"] = {"err_m": float(np.linalg.norm(est - truth)),
                          "range_err_m": float(np.linalg.norm(est - b1.state.position())
                                               - rng_true),
                          "sigma_m": tri["sigma"], "angle_deg": tri["angle_deg"],
                          "age_s": tri["age"]}
            frame.setdefault("markers", []).append({
                "id": "TRI_EST", "name": "TRI est", "type": "Navaid+Static+Waypoint",
                "color": "Yellow", "x": est[0], "y": est[1], "alt": est[2],
                "label": f"TRI est sigma {tri['sigma'] / NM_M:.2f} NM, "
                         f"LOS angle {tri['angle_deg']:.0f} deg"})
            if not state["had_fix"] or t - state["last_status"] >= 30.0:
                kind = "first" if not state["had_fix"] else "update"
                frame.setdefault("events", []).append({
                    "t": t, "type": "tri_fix", "observer": "B1", "target": "R1",
                    "range_m": rng_true,
                    "text": (f"B1 passive IRST triangulation ({kind}): LOS angle "
                             f"{tri['angle_deg']:.1f} deg, error "
                             f"{row['tri']['err_m'] / NM_M:.2f} NM, sigma "
                             f"{tri['sigma'] / NM_M:.2f} NM, fused source "
                             f"{row['fused_sensor']}/{row['fused_src']}")})
                state["had_fix"], state["last_status"] = True, t
        rep.rows.append(row)

    w.frame_hook = hook
    w.run()
    return rep


def triangulation_table(rep: ReplayB, every_s: float = 5.0) -> str:
    rows = [r for r in rep.rows if abs((r["t"] / every_s) - round(r["t"] / every_s)) < 1e-6]
    hdr = (f"{'t (s)':>6} {'true rng NM':>11} {'LOS ang deg':>11} {'IRST own/B2':>11} "
           f"{'TRI err NM':>10} {'TRI rng err NM':>14} {'TRI sigma NM':>12} "
           f"{'radar trk':>9} {'fused src':>14} {'fused err NM':>12}")
    out = [
        f"Replay B passive triangulation (seed {rep.seed}): B1's view of R1, rows every "
        f"{every_s:.0f} s",
        getattr(rep, "desc", ""),
        "LOS ang = true angle between the two jets' lines of sight to Red. TRI = passive "
        "fix from B1's own IRST bearing + B2's bearing received over the link "
        "(>= 10 deg required; never fire-control).",
        "TRI err = 3D position error of the fix; TRI rng err = (range B1->fix) - true range;"
        " sigma = propagated 1-sigma incl. 50 m/s x age.",
        "", hdr, "-" * len(hdr)]
    for r in rows:
        tri = r["tri"]
        irst = f"{'Y' if r['own_irst'] else '-'}/{'Y' if r['b2_irst_rx'] else '-'}"
        if tri:
            ts = (f"{tri['err_m'] / NM_M:10.2f} {tri['range_err_m'] / NM_M:14.2f} "
                  f"{tri['sigma_m'] / NM_M:12.2f}")
        else:
            ts = f"{'-':>10} {'-':>14} {'-':>12}"
        fe = r["fused_err_m"] / NM_M if r["fused_err_m"] == r["fused_err_m"] else float("nan")
        out.append(f"{r['t']:6.0f} {r['true_range_m'] / NM_M:11.1f} "
                   f"{r['true_angle_deg']:11.1f} {irst:>11} {ts} "
                   f"{'Y' if r['radar_track'] else '-':>9} "
                   f"{r['fused_sensor'] + '/' + r['fused_src']:>14} {fe:12.3f}")
    # summary by angle bins
    fixes = [r for r in rep.rows if r["tri"]]
    out += ["", "Summary (all 0.5 s samples with a fix), binned by LOS angle:"]
    bins = [(10, 20), (20, 30), (30, 40), (40, 60), (60, 90), (90, 180)]
    for lo, hi in bins:
        sel = [r for r in fixes if lo <= r["tri"]["angle_deg"] < hi]
        if not sel:
            continue
        e = np.array([r["tri"]["err_m"] for r in sel]) / NM_M
        sg = np.array([r["tri"]["sigma_m"] for r in sel]) / NM_M
        rr = np.array([r["true_range_m"] for r in sel]) / NM_M
        out.append(f"  {lo:3d}-{hi:3d} deg: n={len(sel):4d}  true range {rr.min():5.1f}-"
                   f"{rr.max():5.1f} NM  median err {np.median(e):5.2f} NM  "
                   f"RMS err {math.sqrt(np.mean(e ** 2)):5.2f} NM  median sigma "
                   f"{np.median(sg):5.2f} NM")
    no_fix_irst = [r for r in rep.rows if r["own_irst"] and r["b2_irst_rx"] and not r["tri"]]
    out.append(f"  samples with both IRST bearings but NO fix (angle < 10 deg or stale): "
               f"{len(no_fix_irst)}")
    srcs: Dict[str, int] = {}
    for r in rep.rows:
        k = f"{r['fused_sensor']}/{r['fused_src']}"
        srcs[k] = srcs.get(k, 0) + 1
    out.append("  fused source share over the replay (B1): " + ", ".join(
        f"{k}: {v / len(rep.rows) * 100:.0f}%" for k, v in sorted(srcs.items())))
    first = next((r for r in rep.rows if r["tri"]), None)
    if first:
        out.append(f"  first triangulation fix: t={first['t']:.1f} s, true range "
                   f"{first['true_range_m'] / NM_M:.1f} NM, LOS angle "
                   f"{first['tri']['angle_deg']:.1f} deg, error "
                   f"{first['tri']['err_m'] / NM_M:.2f} NM")
    return "\n".join(out) + "\n"


def export(world: World, path: Path, title: str) -> Path:
    return ACMIExporter(title=title).export(world.frames, path)
