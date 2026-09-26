"""Spec 6 test plan 2: a no-sim toy task for the GA machinery. Fitness =
-MSE between the network's 13 outputs and HandBlue's actions on a fixed set of
logged observations; the behaviour descriptor is the mean of 8 output
channels. Same NeuroGA loop, selection, novelty, mutation and checkpoints."""

from __future__ import annotations

from typing import List

import numpy as np

from .mlp import MLPPolicy
from .neuroga import MemberEval
from .novelty import BD_NAMES


class ToyTask:
    def __init__(self, obs: np.ndarray, act: np.ndarray) -> None:
        self.obs, self.act = obs, act

    def eval_set(self, gen: int) -> List[dict]:
        return [{"presentation_seed": 0, "toy": True}]

    def bench_set(self) -> List[dict]:
        return [{"presentation_seed": 0, "toy": True}]

    def test_set(self) -> List[dict]:
        return []

    def evaluate(self, ga, genomes, pres, tag) -> List[MemberEval]:
        out = []
        for g in genomes:
            y = MLPPolicy(g.weights, ga.arch)(self.obs)
            f = -float(((y - self.act) ** 2).mean())
            cols = [0, 1, 2, 10, 11, 12, 9, 3][:len(BD_NAMES)]
            bd = np.clip((y[:, cols].mean(0) + 1.0) / 2.0, 0.0, 1.5)
            out.append(MemberEval(f, [f], bd, {"launch_r_rmax": None}, [{}]))
        return out
