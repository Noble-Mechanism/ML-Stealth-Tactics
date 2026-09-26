"""Spec 5: ``obs-dump`` — print one Blue jet's named observation at a decision
time (HandBlue flies the fight up to that time)."""

from __future__ import annotations

from stealth_tactics.presentation_runner import run_presentation
from stealth_tactics.scenarios.presentation import sample_presentation

from .adapters import HandBlue
from .controller import NetworkBlueController
from .observation import OBS_SPEC


class _Capture(NetworkBlueController):
    def __init__(self, ids, t_dump: float, jet: str) -> None:
        super().__init__(HandBlue(), ids)
        self.t_dump, self.jet, self.hit = t_dump, jet, None

    def actions(self, world, jets, obs, infos):
        if self.hit is None and world.time_s >= self.t_dump - 1e-9:
            for i, j in enumerate(jets):
                if j.id == self.jet:
                    self.hit = (world.time_s, obs[i].copy(), infos[i])
        return super().actions(world, jets, obs, infos)


def obs_dump(seed: int, t: float, jet: str, stats_index: bool = False) -> str:
    if stats_index:
        from stealth_tactics.analysis.presentation_stats import stats_seeds
        seed = stats_seeds(max(100, seed + 1), 2026)[seed]
    cap = {}

    def fac(ids):
        cap["c"] = _Capture(ids, t, jet)
        return cap["c"]
    run_presentation(sample_presentation(seed), blue_factory=fac)
    hit = cap["c"].hit
    if hit is None:
        return f"{jet} was not alive at a decision at or after t = {t} s (seed {seed})"
    tt, ob, info = hit
    lo, hi = OBS_SPEC.bounds()
    L = [f"Observation of {jet} at t = {tt:.1f} s, presentation seed {seed} "
         f"({OBS_SPEC.size} inputs; HandBlue flew until then)",
         f"Contact slots: {list(info.keys)}"]
    for i, (name, v) in enumerate(zip(OBS_SPEC.names(), ob)):
        L.append(f"{i:3d} {name:28s} {v:+.4f}   [{lo[i]:+.1f}, {hi[i]:+.1f}]")
    return "\n".join(L)
