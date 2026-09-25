"""3a fix: launch off-nose axis of the Rmax table vs the full sim path.

Each case: sim-path Rmax/Rne (``missile_sweep.sim_rmax_nm``: WeaponModel,
0.5 s world step, shooter nose ``off`` deg off the line of sight at launch,
then pointing at the target) vs the interpolated startup table.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Tuple

import numpy as np

from stealth_tactics.analysis.missile_sweep import FT, sim_rmax_nm
from stealth_tactics.sim.sensor_config import DEFAULT_SENSOR_CONFIG, NM_M
from stealth_tactics.sim.weapons import WeaponModel

# (alt ft, shooter Mach, behavior, aspect deg or None, target Mach, off-nose deg)
CASES: List[Tuple[float, float, str, Optional[float], float, float]] = [
    (40000.0, 0.9, "hot", 0.0, 0.9, 0.0),
    (40000.0, 0.9, "hot", 0.0, 0.9, 30.0),
    (40000.0, 0.9, "hot", 0.0, 0.9, 50.0),
    (40000.0, 0.9, "beam", 90.0, 0.9, 50.0),
    (40000.0, 0.9, "beam", 90.0, 0.9, -50.0),
    (40000.0, 0.9, "beam", 45.0, 0.9, 50.0),
    (40000.0, 0.9, "beam", 45.0, 0.9, -50.0),
    (40000.0, 0.9, "beam", 45.0, 0.9, 22.0),
    (40000.0, 0.9, "cold", 180.0, 0.9, 30.0),
    (40000.0, 0.9, "turncold", 0.0, 0.9, 50.0),
    (30000.0, 1.0, "beam", 60.0, 0.9, 50.0),
    (30000.0, 1.0, "beam", 60.0, 0.9, -40.0),
    (25000.0, 0.8, "hot", 0.0, 0.8, 55.0),
    (29500.0, 0.78, "beam", 50.0, 0.75, 50.0),   # ~launch-on-remote geometry
    (40000.0, 0.9, "beam", 90.0, 0.9, -45.0),
    (40000.0, 0.9, "beam", 70.0, 0.9, 35.0),
    (15000.0, 0.9, "hot", 0.0, 0.9, 45.0),
    (45000.0, 1.2, "beam", 110.0, 1.1, -35.0),
]
# Lag shots past ~55 deg (and any shot past ~55-60 deg at a cold target) fall
# off a cliff (the turn bleeds the missile below Mach 1.2): between the bins
# the table is only approximate there. Reported separately.
CLIFF_CASES = [
    (40000.0, 0.9, "beam", 135.0, 0.9, -57.0),
    (40000.0, 0.9, "cold", 180.0, 0.9, 57.0),
    (40000.0, 0.9, "cold", 180.0, 0.9, 56.0),
]


def _one(case) -> Tuple:
    alt, ms, beh, asp, mt, off = case
    sim = sim_rmax_nm(alt, ms, beh, mt, lo=1.0, hi=80.0, off_nose_deg=off, aspect_deg=asp)
    env = WeaponModel(np.random.default_rng(0), DEFAULT_SENSOR_CONFIG).envelope
    f = env.rne_m if beh == "turncold" else env.rmax_m
    tab = f(alt * FT, ms, asp, mt, off) / NM_M
    tab0 = f(alt * FT, ms, asp, mt, 0.0) / NM_M
    return case, sim, tab, tab0


def run(workers: int = 8) -> str:
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_one, CASES))
        cliff = list(ex.map(_one, CLIFF_CASES))
    lines = ["Rmax table (with launch off-nose axis) vs sim path, NM. off = launch angle off "
             "the shooter's nose (+ lead: nose toward the side the target moves; - lag).",
             "turncold rows = Rne (target turns cold at 3 g at launch). 'tab off=0' = table "
             "value nose-on (what a table without the off-nose axis gives for this geometry).",
             f"{'alt ft':>6} {'M_s':>4} {'target':<8} {'asp':>4} {'M_t':>4} {'off':>5} | "
             f"{'sim':>6} {'table':>6} {'err':>5} | {'tab off=0':>9}"]
    lines.append("-" * len(lines[-1]))
    errs = []
    for (alt, ms, beh, asp, mt, off), sim, tab, tab0 in res:
        errs.append(tab - sim)
        lines.append(f"{alt:6.0f} {ms:4.2f} {beh:<8} {asp:4.0f} {mt:4.2f} {off:5.0f} | "
                     f"{sim:6.1f} {tab:6.1f} {tab - sim:+5.2f} | {tab0:9.1f}")
    lines.append(f"max |table - sim| = {max(abs(e) for e in errs):.2f} NM over {len(errs)} cases")
    lines.append("")
    lines.append("Cliff region (between the 55 and 60 deg bins, lag or cold target): "
                 "approximate. A lookup with >= half its interpolation weight on no-shot "
                 "(0) cells returns 0; otherwise the plain blend (pulled down toward 0).")
    for (alt, ms, beh, asp, mt, off), sim, tab, tab0 in cliff:
        lines.append(f"{alt:6.0f} {ms:4.2f} {beh:<8} {asp:4.0f} {mt:4.2f} {off:5.0f} | "
                     f"{sim:6.1f} {tab:6.1f} {tab - sim:+5.2f} | {tab0:9.1f}")
    return "\n".join(lines) + "\n"


def random_check(n: int = 1000, seed: int = 1) -> str:
    """Table vs the batch engine (same physics as the sim path, checked by
    ``test_batch_engine_agrees_with_sim_path``) at random points in the grid."""
    from stealth_tactics.sim.missile_envelope import bisect_rmax
    kc = DEFAULT_SENSOR_CONFIG.missile_kinematics
    env = WeaponModel(np.random.default_rng(0), DEFAULT_SENSOR_CONFIG).envelope
    rng = np.random.default_rng(seed)
    alt = rng.uniform(1000.0, 15000.0, n)
    ms = rng.uniform(0.5, 1.3, n)
    asp = rng.uniform(0.0, 180.0, n)
    mt = rng.uniform(0.5, 1.3, n)
    off = rng.uniform(-60.0, 60.0, n)
    tc = rng.random(n) < 0.25
    exact = bisect_rmax(kc, alt, ms, asp, mt, tc, off_nose_deg=off) / NM_M
    # Rne is defined as min(turn cold, hold course) (see missile_envelope)
    straight = bisect_rmax(kc, alt[tc], ms[tc], asp[tc], mt[tc], False,
                           off_nose_deg=off[tc]) / NM_M
    exact[tc] = np.minimum(exact[tc], straight)
    tab = np.array([(env.rne_m if t else env.rmax_m)(a, m, s_, u, o) for a, m, s_, u, o, t
                    in zip(alt, ms, asp, mt, off, tc)]) / NM_M
    both = (exact > 0) & (tab > 0)
    err = np.abs(tab - exact)[both]
    over = int(np.sum((tab > 0) & (exact == 0)))
    under = int(np.sum((tab == 0) & (exact > 0)))
    return (f"Random check, {n} points (alt 1-15 km, M_s 0.5-1.3, aspect 0-180, M_t 0.5-1.3, "
            f"off-nose -60..60, 25 % Rne = min(turn cold, straight)), table vs batch engine:\n"
            f"  both > 0: {both.sum()}; |err| median {np.median(err):.2f}, p90 "
            f"{np.percentile(err, 90):.2f}, p99 {np.percentile(err, 99):.2f}, max "
            f"{err.max():.2f} NM; within 0.5 NM: {np.mean(err <= 0.5) * 100:.0f} %\n"
            f"  optimistic by > 1 NM: {int(np.sum((tab - exact)[both] > 1.0))}; table > 0 but "
            f"no range hits: {over}; table 0 but a hit exists: {under}\n")


if __name__ == "__main__":
    print(run())
    print(random_check())
