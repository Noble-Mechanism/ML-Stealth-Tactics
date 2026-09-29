"""Build / refresh the default missile envelope table (Spec 3a / 8).

Run: ``python -m stealth_tactics.analysis.build_envelope``
Using ``-m`` (not a loose script) so ProcessPoolExecutor spawn workers do not
re-enter the caller.
"""
from __future__ import annotations


def main() -> int:
    import os
    import time

    os.environ.setdefault("STEALTH_TACTICS_CACHE_DIR",
                          str(__import__("pathlib").Path.home() / ".cache" / "stealth_tactics"))
    from stealth_tactics.sim.missile_envelope import (
        BUILD_INFO, OUTCOME_NAMES, batch_shots, get_envelope)
    from stealth_tactics.sim.sensor_config import DEFAULT_SENSOR_CONFIG as CFG, NM_M

    print(f"workers={os.environ.get('STEALTH_TACTICS_ENV_WORKERS', 'auto')} "
          f"cache={os.environ.get('STEALTH_TACTICS_CACHE_DIR')}", flush=True)
    t0 = time.time()
    env = get_envelope(CFG.missile_kinematics)
    print(f"done in {time.time() - t0:.1f}s info={BUILD_INFO}", flush=True)
    alt = 40_000 * 0.3048
    print("hot", round(env.rmax_m(alt, 0.9, 0.0, 0.9, 0.0) / NM_M, 2), flush=True)
    for off in (0.0, 50.0, -50.0):
        tab = env.rmax_m(alt, 0.9, 90.0, 0.9, off) / NM_M
        res = batch_shots(CFG.missile_kinematics, alt, 0.9, 90.0, 0.9,
                          (tab - 0.3) * NM_M, False, off_nose_deg=off)
        o = int(res["outcome"].reshape(-1)[0])
        print(f"beam off={off}: table={tab:.2f}, batch@tab-0.3 -> {OUTCOME_NAMES[o]}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
