"""Spec 5: the network interface (observation, action, controller, radar
on/off, restricted Blue view, adapters). docs/specs/05-network-interface.md"""

from __future__ import annotations

import dataclasses
import filecmp
import math
from fractions import Fraction

import numpy as np
import pytest

from stealth_tactics.analysis.interface_adapter import (
    SilentAudit, layer1_job, run_variant, summarize)
from stealth_tactics.analysis.presentation_stats import stats_seeds
from stealth_tactics.policy import (ACTION_SPEC, OBS_SPEC, NetworkBlueController,
                                    build_observation, decode_action, encode_commands)
from stealth_tactics.policy.action import (ALT_MAX_M, ALT_MIN_M, SPD_MAX_MPS, SPD_MIN_MPS,
                                           decode_alt, decode_heading, decode_speed,
                                           select_target)
from stealth_tactics.policy.adapters import HandBlue, RandomMLPPolicy, ScriptedViaInterface
from stealth_tactics.policy.observation import (CONTACT_NAMES, MATE_NAMES, MATE_ORDER,
                                                MISSION_AXIS_RAD, OWN_NAMES, ObsInfo,
                                                order_contacts)
from stealth_tactics.policy.view import (BlueView, ContactView, CueView, MateView,
                                         MissileView, OwnView, build_blue_view)
from stealth_tactics.presentation_runner import export_presentation_acmi, run_presentation
from stealth_tactics.scenarios.presentation import sample_presentation
from stealth_tactics.sim.aircraft import Aircraft, AircraftState, Coalition
from stealth_tactics.sim.missile_envelope import MissileEnvelope, get_envelope
from stealth_tactics.sim.sensor_config import DEFAULT_SENSOR_CONFIG
from stealth_tactics.sim.tracks import SensorComponent, Track
from stealth_tactics.sim.world import SHOOT_ASSESS_SHOOT, SHOOT_SHOOT_ASSESS, World
from stealth_tactics.tactics.genome import TacticsGenome
from stealth_tactics.tactics.interpreter import TacticsController

SEEDS = stats_seeds(100, 2026)
S, A = OBS_SPEC, ACTION_SPEC
LO, HI = S.bounds()


class Probe:
    """Blue controller wrapper: calls ``fn(world)`` at the chosen times (after
    sensors/RWR are updated, before Blue acts), then the inner controller."""

    def __init__(self, inner, times, fn) -> None:
        self.inner, self.times, self.fn = inner, sorted(times), fn

    def __call__(self, world) -> None:
        while self.times and world.time_s >= self.times[0] - 1e-9:
            self.times.pop(0)
            self.fn(world)
        self.inner(world)


def probe_run(seed, times, fn, inner_factory=None, prepare=None, record=False):
    inner_factory = inner_factory or (lambda ids: TacticsController(TacticsGenome(), ids))
    return run_presentation(sample_presentation(seed), record=record, prepare=prepare,
                            blue_factory=lambda ids: Probe(inner_factory(ids), times, fn))


def obs_of(world, ac, target=None, mates=None):
    return build_observation(build_blue_view(world, ac), target, mates or {})


def blues(world):
    return [a for a in world.aircraft if a.coalition == Coalition.BLUE and a.state.alive]


# ------------------------------------------------------------ layout ------
def test_obs_layout_exact_cover():
    assert (len(OWN_NAMES), len(CONTACT_NAMES), len(MATE_NAMES)) == (21, 28, 14)
    assert S.size == 231 and S.k == 6 and S.n_mates == 3
    cover = np.zeros(S.size, int)
    cover[S.own_slice] += 1
    for i in range(S.k):
        cover[S.contact_slice(i)] += 1
    for j in range(S.n_mates):
        cover[S.mate_slice(j)] += 1
    assert (cover == 1).all()
    names = S.names()
    assert len(names) == len(set(names)) == 231
    assert S.index("c0.present") == 21 and S.index("w0.alive") == 21 + 6 * 28
    lo, hi = S.bounds()
    assert lo.shape == hi.shape == (231,) and (lo < hi).all()
    assert set(zip(lo.tolist(), hi.tolist())) <= {(0.0, 1.0), (-1.0, 1.0), (0.0, 1.5),
                                                  (-1.5, 1.5)}
    assert A.size == 13 and A.target_slice == slice(3, 10)
    assert (A.fire, A.radar, A.pair) == (10, 11, 12)


def test_mate_order_element_relative():
    assert MATE_ORDER == {0: (1, 2, 3), 1: (0, 2, 3), 2: (3, 0, 1), 3: (2, 0, 1)}


def _check_obs(ob, view=None):
    assert ob.shape == (231,)
    assert np.isfinite(ob).all()
    assert (ob >= LO - 1e-12).all() and (ob <= HI + 1e-12).all(), \
        [(S.names()[i], ob[i]) for i in np.where((ob < LO) | (ob > HI))[0]]
    for i in range(S.k):
        f = ob[S.contact_slice(i)]
        if f[0] == 0:
            assert (f == 0).all()                       # empty slot all zeros
            continue
        s, c = f[CONTACT_NAMES.index("brg_sin")], f[CONTACT_NAMES.index("brg_cos")]
        assert abs(s * s + c * c - 1) < 1e-9
        if f[1] == 0:                                   # bearing-only
            for n in ("range", "dalt", "kin_valid", "aspect_sin", "aspect_cos", "closure",
                      "tgt_speed", "rmax", "shot_ready"):
                assert f[CONTACT_NAMES.index(n)] == 0, n
        if f[CONTACT_NAMES.index("kin_valid")] == 0:
            for n in ("aspect_sin", "aspect_cos", "closure", "tgt_speed", "rmax"):
                assert f[CONTACT_NAMES.index(n)] == 0, n
        else:
            s, c = f[CONTACT_NAMES.index("aspect_sin")], f[CONTACT_NAMES.index("aspect_cos")]
            assert abs(s * s + c * c - 1) < 1e-9
    for j in range(S.n_mates):
        w = ob[S.mate_slice(j)]
        if w[0] == 0:
            assert (w == 0).all()
    own = ob[S.own_slice]
    assert own[:4].sum() == 1.0
    assert abs(own[6] ** 2 + own[7] ** 2 - 1) < 1e-9


def test_obs_shape_bounds_masking_over_fights():
    """Every decision step, every live jet, 8 presentations x 2 policies
    (the 100-presentation sweep runs in interface-adapter-test)."""
    n = [0]

    class Checked(NetworkBlueController):
        def actions(self, world, jets, obs, infos):
            for ob in obs:
                _check_obs(ob)
                n[0] += 1
            return super().actions(world, jets, obs, infos)

    for seed in SEEDS[:4]:
        for pol in (HandBlue(), RandomMLPPolicy(3)):
            run_presentation(sample_presentation(seed),
                             blue_factory=lambda ids: Checked(pol, ids))
    assert n[0] > 3000


def test_dead_mate_zero_and_slot_onehot():
    seen = {"dead": 0}

    def fn(world):
        bl = [a for a in world.aircraft if a.coalition == Coalition.BLUE]
        for ac in bl:
            if not ac.state.alive:
                continue
            ob, _ = obs_of(world, ac)
            slot = int(ac.id[1:]) - 1
            assert ob[slot] == 1.0 and ob[:4].sum() == 1.0
            for j, s in enumerate(MATE_ORDER[slot]):
                w = ob[S.mate_slice(j)]
                mate = bl[s]
                assert mate.id == f"B{s + 1}"
                if mate.state.alive:
                    assert w[0] == 1.0
                    d = math.dist((mate.state.x, mate.state.y, mate.state.alt),
                                  (ac.state.x, ac.state.y, ac.state.alt))
                    assert w[1] == pytest.approx(min(1.5, d / S.mate_range_m))
                else:
                    assert (w == 0).all()
                    seen["dead"] += 1
    probe_run(SEEDS[0], list(range(0, 360, 5)), fn)
    assert seen["dead"] > 0


# ------------------------------------------------------------ ordering ----
def _cv(key, pos=None, sigma=100.0, mode=0, first_t=0.0, ranged=True, vel=None):
    return ContactView(key=key, ranged=ranged, est_pos=pos if ranged else None,
                       sigma_m=sigma if ranged else None, sensor="radar" if ranged else "rwr",
                       own=True, age_s=0.0, vel=vel, rwr_bearing=None if ranged else 0.3,
                       own_fc=False, remote_fc=False, may_fire=True, rwr_mode=mode,
                       first_t=first_t)


def _view(contacts, missiles=(), mates=()):
    own = OwnView("B1", 0, 0.0, 0.0, 8000.0, 0.0, 250.0, 4, True, SHOOT_ASSESS_SHOOT)
    return BlueView(t=10.0, own=own, mates=tuple(mates), contacts=tuple(contacts),
                    missiles=tuple(missiles), cues=(), red_kills=0, pair_open=False,
                    envelope=get_envelope(DEFAULT_SENSOR_CONFIG.missile_kinematics), active_range_m=27780.0)


def test_ordering_pin_range_ties_bearing_only():
    cs = [_cv("R3", (0, 30000, 8000)), _cv("R1", (0, 20000, 8000), sigma=500),
          _cv("R2", (0, 20000, 8000), sigma=100), _cv("R9", (0, 20000, 8000), sigma=100),
          _cv("R5", ranged=False, mode=1, first_t=1), _cv("R4", ranged=False, mode=3,
                                                          first_t=5),
          _cv("R6", ranged=False, mode=3, first_t=2)]
    keys = [c.key for c in order_contacts(_view(cs), None)]
    assert keys == ["R2", "R9", "R1", "R3", "R6", "R4", "R5"]
    keys = [c.key for c in order_contacts(_view(cs), "R3")]
    assert keys == ["R3", "R2", "R9", "R1", "R6", "R4", "R5"]
    keys = [c.key for c in order_contacts(_view(cs), "R6")]          # pin bearing-only
    assert keys[0] == "R6"
    keys = [c.key for c in order_contacts(_view(cs), "R77")]          # gone -> nearest
    assert keys[0] == "R2"
    # a contact dropping out shifts the rest in order
    keys2 = [c.key for c in order_contacts(_view([c for c in cs if c.key != "R9"]), "R3")]
    assert keys2 == ["R3", "R2", "R1", "R6", "R4", "R5"]
    ob, info = build_observation(_view(cs), "R3")
    assert info.keys == ["R3", "R2", "R9", "R1", "R6", "R4"]          # K = 6 truncation
    # bearing-only slots: range fields 0, range_valid 0, kin fields 0
    f = ob[S.contact_slice(4)]
    assert f[0] == 1 and f[1] == 0 and f[2] == 0 and f[6] == 0
    # radar tracks without a velocity: kinematics fields 0
    f = ob[S.contact_slice(0)]
    assert f[1] == 1 and f[6] == 0 and (f[7:11] == 0).all()


def test_rmax_rne_match_envelope_on_perceived_states():
    env = get_envelope(DEFAULT_SENSOR_CONFIG.missile_kinematics)
    pos, vel = (8000.0, 40000.0, 9000.0), (-50.0, -240.0, 0.0)
    c = dataclasses.replace(_cv("R1", pos, vel=vel), own_fc=True)
    v = _view([c])
    ob, info = build_observation(v, None)
    own = AircraftState(0.0, 0.0, 8000.0, 0.0, 250.0)
    tgt = AircraftState(pos[0], pos[1], pos[2], math.atan2(vel[0], vel[1]),
                        math.hypot(vel[0], vel[1]))
    rmax, rne = env.for_states(own, tgt)
    r = math.dist(pos, (0.0, 0.0, 8000.0))
    f = ob[S.contact_slice(0)]
    idx = CONTACT_NAMES.index
    assert f[idx("rmax")] == pytest.approx(rmax / S.contact_range_m)
    assert f[idx("r_rmax")] == pytest.approx(min(r / rmax, 2.0) / 2.0)
    assert f[idx("r_rne")] == pytest.approx(min(r / rne, 2.0) / 2.0)
    assert f[idx("shot_ready")] == float(r <= rmax)
    # signed off-nose: target right of the nose -> positive sine
    assert f[idx("brg_sin")] > 0 and f[idx("brg_sin")] == pytest.approx(
        math.sin(math.atan2(pos[0], pos[1])))
    assert info.bearings[0] == pytest.approx(math.atan2(pos[0], pos[1]))


def test_shot_ready_agrees_with_world_gates():
    """A fire request at a shot_ready slot launches that step (the world's own
    gates agree) except at the very edge of the envelope, where the perceived
    range / Rmax (noisy estimate) and the World's true-range gate can disagree;
    nothing launches without a request."""
    req = []

    class ShootReady(NetworkBlueController):
        def actions(self, world, jets, obs, infos):
            acts = []
            for ac, ob, info in zip(jets, obs, infos):
                out = np.zeros(A.size)
                out[A.target_slice] = -1.0
                slot = next((i for i, r in enumerate(info.shot_ready) if r), None)
                if slot is None:                          # fly at the nearest contact
                    slot0 = 0 if info.keys[0] else None
                    out[A.target0 + (slot0 if slot0 is not None else A.k)] = 1.0
                    out[A.fire] = -1.0
                else:
                    out[A.target0 + slot] = 1.0
                    out[A.fire] = 1.0
                    req.append((world.time_s, ac.id,
                                float(ob[S.contact_slice(slot)][CONTACT_NAMES.index("r_rmax")]) * 2))
                out[A.radar], out[A.pair] = 1.0, -1.0
                acts.append(decode_action(out, info))
            return acts

    res = run_presentation(sample_presentation(SEEDS[1]),
                           blue_factory=lambda ids: ShootReady(None, ids,
                                                               decision_period_s=0.5))
    launches = {(e["t"], e["shooter"]) for e in res.events
                if e["type"] == "launch" and e["shooter"].startswith("B")}
    keys = {(t, j) for t, j, _ in req}
    assert req and launches
    assert launches <= keys
    denied = [x for x in req if (x[0], x[1]) not in launches]
    assert all(rr >= 0.9 for _, _, rr in denied), denied


# ------------------------------------------------------------ truth leakage
def test_no_truth_leak_move_red_and_poison_true_range():
    checked = [0]

    def fn(world):
        for ac in blues(world):
            ob1, _ = obs_of(world, ac)
            saved = []
            for r in world.aircraft:
                if r.coalition == Coalition.RED:
                    saved.append((r, r.state.x, r.state.y, r.state.alt, r.state.heading_rad,
                                  r.state.speed_mps))
                    r.state.x += 7777.0
                    r.state.y -= 5555.0
                    r.state.alt += 999.0
                    r.state.heading_rad += 1.0
                    r.state.speed_mps += 33.0
            poisoned = []
            for st in world.tracks.values():
                for tr in st.tracks.values():
                    for c in list(tr.components.values()) + [
                            c for d in tr.remote.values() for c in d.values()]:
                        poisoned.append((c, c.true_range_m))
                        c.true_range_m = float("nan")
            ob2, _ = obs_of(world, ac)
            for r, x, y, z, h, v in saved:
                r.state.x, r.state.y, r.state.alt, r.state.heading_rad, r.state.speed_mps = \
                    x, y, z, h, v
            for c, tr in poisoned:
                c.true_range_m = tr
            assert np.array_equal(ob1, ob2)
            assert np.isfinite(ob2).all()
            checked[0] += 1
    probe_run(SEEDS[2], [20, 60, 90, 120, 150, 200], fn)
    assert checked[0] >= 12


FORBIDDEN_TYPES = (Aircraft, Track, SensorComponent, World)


def _walk(obj, path="view", seen=None):
    seen = seen if seen is not None else set()
    if id(obj) in seen:
        return
    seen.add(id(obj))
    assert not isinstance(obj, FORBIDDEN_TYPES), path
    tn = type(obj).__name__
    assert tn not in ("Missile", "TrackStore", "FusedTrack", "RwrCue"), path
    if isinstance(obj, MissileEnvelope):
        return                                          # pure lookup table (by reference)
    if dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            assert "true" not in f.name and not f.name.startswith("pk"), f"{path}.{f.name}"
            assert f.name not in ("primary_fc_target", "defense_state"), f"{path}.{f.name}"
            _walk(getattr(obj, f.name), f"{path}.{f.name}", seen)
    elif isinstance(obj, (tuple, list)):
        for i, x in enumerate(obj):
            _walk(x, f"{path}[{i}]", seen)
    else:
        assert isinstance(obj, (str, int, float, bool, type(None))), (path, type(obj))


class _Strict:
    def __init__(self, obj, allowed=("coalition",)):
        object.__setattr__(self, "_o", obj)
        object.__setattr__(self, "_allowed", set(allowed))

    def __getattr__(self, name):
        if name in self._allowed:
            return getattr(self._o, name)
        raise AssertionError(f"truth access: Red {type(self._o).__name__}.{name}")


class _StrictDict(dict):
    def __init__(self, d, red_ids):
        super().__init__(d)
        self.red = set(red_ids)

    def get(self, k, default=None):
        assert k not in self.red, f"truth access: Red store {k}"
        return super().get(k, default)

    def __getitem__(self, k):
        assert k not in self.red, f"truth access: Red store {k}"
        return super().__getitem__(k)


class StrictWorld:
    """Proxy: Red aircraft / missiles / stores / RWR raise on any access except
    the coalition flag; primary_fc_target and truth lookups raise. The declared
    world gates (may_fire_at, ssa_pair_open; spec 5 O/K) pass through."""

    def __init__(self, w):
        self._w = w
        red = {a.id for a in w.aircraft if a.coalition == Coalition.RED}
        self.aircraft = [a if a.coalition == Coalition.BLUE else _Strict(a) for a in w.aircraft]
        self.missiles = [m if m.coalition == Coalition.BLUE.value else _Strict(m)
                         for m in w.missiles]
        self.tracks = _StrictDict(w.tracks, red)
        self.rwr = _StrictDict(w.rwr, red)
        self.time_s, self.events, self.sensor_cfg = w.time_s, w.events, w.sensor_cfg
        self.weapons = type("W", (), {"envelope": w.weapons.envelope, "kcfg": w.weapons.kcfg})
        self.may_fire_at, self.ssa_pair_open = w.may_fire_at, w.ssa_pair_open
        self.doctrine_of = w.doctrine_of

    def __getattr__(self, name):
        raise AssertionError(f"blue_view touched World.{name}")


def test_view_strict_proxy_and_structure():
    checked = [0]

    def fn(world):
        sw = StrictWorld(world)
        for ac in blues(world):
            v1 = build_blue_view(sw, ac)
            v2 = build_blue_view(world, ac)
            assert v1 == v2
            _walk(v2)
            checked[0] += 1
    probe_run(SEEDS[3], [0, 30, 60, 100, 140, 180, 240], fn)
    assert checked[0] >= 20


def test_identical_blue_picture_different_red_truth_same_obs():
    """(d) Two worlds with the same Blue picture but different Red truth:
    re-running the SAME world after moving Red and restoring the Blue picture
    is covered by the move test; here, a world whose Red jets are replaced by
    far-away copies (not in any Blue sensor) gives the same obs."""
    def fn(world):
        import copy
        for ac in blues(world)[:1]:
            ob1, _ = obs_of(world, ac)
            w2 = copy.copy(world)
            w2.aircraft = [copy.deepcopy(a) if a.coalition == Coalition.RED else a
                           for a in world.aircraft]
            for r in w2.aircraft:
                if r.coalition == Coalition.RED:
                    r.state.x, r.state.y = 9e6, 9e6
            ob2, _ = obs_of(w2, ac)
            assert np.array_equal(ob1, ob2)
    probe_run(SEEDS[4], [40, 110, 170], fn)


# ------------------------------------------------------------ action ------
def test_decode_ranges_and_target_masking():
    for h in (-1.0, -0.5, 0.0, 0.5, 1.0, 7.0, -9.0):
        v = decode_heading(max(-1, min(1, h)), 0.3)
        assert -math.pi - 1e-12 <= v <= math.pi + 1e-12
    assert decode_heading(0.0, 0.3) == 0.3
    assert decode_heading(0.5, 0.0) == pytest.approx(math.pi / 2)
    assert decode_alt(-1.0) == ALT_MIN_M and decode_alt(1.0) == ALT_MAX_M
    assert decode_speed(-1.0) == SPD_MIN_MPS and decode_speed(1.0) == SPD_MAX_MPS
    assert decode_alt(0.0) == pytest.approx((ALT_MIN_M + ALT_MAX_M) / 2)
    info = ObsInfo(keys=["R1", None, "R3", None, None, None],
                   bearings=[0.2, None, -0.7, None, None, None])
    lg = np.array([0, 9, 1, 0, 0, 0, 0.5])              # slot 1 empty (masked)
    assert select_target(lg, info) == 2
    lg = np.array([0, 9, 0, 0, 0, 0, 0.5])
    assert select_target(lg, info) is None               # "none" wins among present
    out = np.zeros(A.size)
    out[A.target_slice] = [0, 9, 0, 0, 0, 0, 0.5]
    out[A.fire], out[A.heading] = 1.0, 0.25
    a = decode_action(out, info)
    assert a.target_key is None and not a.fire           # none -> no fire
    assert a.heading_rad == pytest.approx(MISSION_AXIS_RAD + math.pi / 4)
    out[A.target_slice] = [0, 9, 3, 0, 0, 0, 0.5]
    a = decode_action(out, info)
    assert a.target_key == "R3" and a.fire
    assert a.heading_rad == pytest.approx(-0.7 + math.pi / 4)
    out[A.heading] = float("nan")
    out[A.alt] = 5.0
    a = decode_action(out, info)
    assert math.isfinite(a.heading_rad) and a.alt_m == ALT_MAX_M


def test_encode_decode_round_trip():
    rng = np.random.default_rng(1)
    info = ObsInfo(keys=["R1", "R2", None, None, None, None],
                   bearings=[0.4, -1.1, None, None, None, None])
    n_exact = 0
    N = 2000
    for _ in range(N):
        h = float(rng.uniform(-math.pi, math.pi))
        z = float(rng.uniform(ALT_MIN_M, ALT_MAX_M))
        v = float(rng.uniform(SPD_MIN_MPS, SPD_MAX_MPS))
        fire = bool(rng.random() < 0.3)
        tk = "R2" if fire else None
        out, exact = encode_commands(h, z, v, tk, fire, info, radar=False, pair=True)
        a = decode_action(out, info)
        n_exact += exact
        assert a.fire == fire and (a.target_key == "R2" if fire else True)
        assert not a.radar and a.pair
        dh = (a.heading_rad - h + math.pi) % (2 * math.pi) - math.pi
        assert abs(dh) <= 8 * np.spacing(math.pi)
        assert abs(a.alt_m - z) <= 8 * np.spacing(ALT_MAX_M)
        assert abs(a.speed_mps - v) <= 8 * np.spacing(SPD_MAX_MPS)
        if exact:
            assert (a.heading_rad, a.alt_m, a.speed_mps) == (h, z, v)
    # the approved affine maps cannot hit every double; most are exact
    assert n_exact / N > 0.6


def test_fire_needs_world_gates():
    """A fire request at a contact without FC (or blocked by doctrine) never
    launches: gates stay in the World."""
    class FireAtAll(NetworkBlueController):
        def actions(self, world, jets, obs, infos):
            acts = []
            for ob, info in zip(obs, infos):
                out = np.full(A.size, -1.0)
                out[A.target0 + (0 if info.keys[0] else A.k)] = 1.0
                out[A.fire], out[A.radar] = 1.0, 1.0
                acts.append(decode_action(out, info))
            return acts

    fired_without_fc = []

    class Watch(FireAtAll):
        def __call__(self, world):
            super().__call__(world)
            for ac in blues(world):
                if ac.cmd_fire and ac.fire_target and \
                        world.fire_control_source(ac, ac.fire_target) is None:
                    fired_without_fc.append((world.time_s, ac.id, ac.fire_target))

    res = run_presentation(sample_presentation(SEEDS[5]),
                           blue_factory=lambda ids: Watch(None, ids))
    assert fired_without_fc                                # requests were made
    blocked = {(t, a) for t, a, _ in fired_without_fc}
    launches = {(e["t"], e["shooter"]) for e in res.events if e["type"] == "launch"}
    assert not (blocked & launches)


# ------------------------------------------------------------ controller --
def test_controller_decision_times_dwell_pair_and_determinism():
    times = []

    class Flipper:
        """Radar flips every decision; pair bit alternates."""
        def __init__(self):
            self.k = 0

        def __call__(self, obs):
            self.k += 1
            out = np.full((obs.shape[0], A.size), -1.0)
            out[:, A.target0 + A.k] = 1.0
            out[:, A.radar] = 1.0 if self.k % 2 else -1.0
            out[:, A.pair] = 1.0 if self.k % 3 == 0 else -1.0
            return out

    class Rec(NetworkBlueController):
        def actions(self, world, jets, obs, infos):
            times.append(world.time_s)
            return super().actions(world, jets, obs, infos)

    ctl = {}

    def fac(ids):
        ctl["c"] = Rec(Flipper(), ids)
        return ctl["c"]
    res = run_presentation(sample_presentation(SEEDS[6]), blue_factory=fac, record=True)
    assert all(abs(t - round(t)) < 1e-9 for t in times)
    assert times[:5] == [0.0, 1.0, 2.0, 3.0, 4.0]
    radar = [e for e in res.events if e["type"] == "radar"]
    by_jet = {}
    for e in radar:
        by_jet.setdefault(e["observer"], []).append(e["t"])
    assert by_jet
    for ts in by_jet.values():                             # 5 s minimum dwell
        assert all(b - a >= 5.0 - 1e-9 for a, b in zip(ts, ts[1:]))
    assert sum(m.radar_denied for m in ctl["c"].mem.values()) > 0
    # determinism: same presentation + same policy -> identical fight and ACMI
    res2 = run_presentation(sample_presentation(SEEDS[6]),
                            blue_factory=lambda ids: Rec(Flipper(), ids), record=True)
    assert res.events == res2.events and res.shots == res2.shots
    # radar events appear in the ACMI; RadarMode written
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        p1, p2 = pathlib.Path(td) / "a.acmi", pathlib.Path(td) / "b.acmi"
        export_presentation_acmi(res, p1)
        export_presentation_acmi(res2, p2)
        assert filecmp.cmp(p1, p2, shallow=False)
        txt = p1.read_text()
        assert "RadarMode=0" in txt and "RADAR OFF" in txt


def test_pair_bit_only_when_no_pair_open():
    log = []

    class PairWatch(NetworkBlueController):
        def _apply_decision(self, world, ac, a):
            before = world.doctrine_of(ac)
            was_open = world.ssa_pair_open(ac)
            super()._apply_decision(world, ac, a)
            log.append((was_open, before, world.doctrine_of(ac), a.pair))

    class PairShooter(HandBlue):
        def __call__(self, obs):
            out = super().__call__(obs)
            out[:, A.pair] = 1.0 if (int(obs[0, S.index("own.time")] * 360) // 7) % 2 else -1.0
            return out

    run_presentation(sample_presentation(SEEDS[7]),
                     blue_factory=lambda ids: PairWatch(PairShooter(), ids))
    for was_open, before, after, pair in log:
        if was_open:
            assert after == before
        else:
            assert after == (SHOOT_SHOOT_ASSESS if pair else SHOOT_ASSESS_SHOOT)
    assert any(o for o, *_ in log)
    assert any(b != a for _, b, a, _ in log)


def test_dead_jets_skipped_and_batch_order_independent():
    seen_sizes = set()

    class Rec(RandomMLPPolicy):
        def __call__(self, obs):
            seen_sizes.add(obs.shape[0])
            return super().__call__(obs)

    res = run_presentation(sample_presentation(SEEDS[0]),
                           blue_factory=lambda ids: NetworkBlueController(Rec(5), ids))
    assert 4 in seen_sizes
    if res.red_kills:
        assert min(seen_sizes) < 4
    # row-wise policy: rows are independent, so batch order does not matter
    obs = np.random.default_rng(0).uniform(-1, 1, (4, 231))
    pol = RandomMLPPolicy(5)
    assert np.allclose(pol(obs)[::-1], pol(obs[::-1]))


# ------------------------------------------------------------ radar off ---
def test_radar_off_sim_gates():
    """B1 radar off all fight: no own radar hits, no own FC, no Red RWR cue or
    spec 1 RWR component from B1; B1 still launches on remote FC and support
    passes to a flightmate."""
    r, ex = run_variant(SEEDS[0], "silent")
    d = summarize(SEEDS[0], "silent", r, ex)
    a = ex["audit"]
    assert a["red_cues"] == 0 and a["red_rwr_components"] == 0
    assert a["own_radar_components"] == 0 and a["own_fc"] == 0 and a["emitting_steps"] == 0
    assert d["silent_remote_launches"] > 0 and d["silent_own_launches"] == 0
    assert d["support_handoffs"] > 0


def test_radar_off_drops_fc_within_gap_rule():
    """Radar on until a Blue holds own FC, then off: own FC is gone within
    fc_max_gap_s + one scan, and the Red RWR modes from it stop at once."""
    state = {}

    def fn(world):
        if "off" not in state:
            for ac in blues(world):
                st = world.tracks[ac.id]
                if any(tr.fire_control for tr in st.tracks.values()):
                    ac.radar_emitting = False
                    state["off"] = (world.time_s, ac.id)
                    return
        else:
            t0, bid = state["off"]
            ac = world.get(bid)
            if ac is None or not ac.state.alive:
                return
            for red in world.aircraft:
                if red.coalition == Coalition.RED:
                    assert all(c.emitter_id != bid or c.mode == "missile_active"
                               for c in world.rwr.get(red.id, []))
            if world.time_s > t0 + world.sensor_cfg.track.fc_max_gap_s + 1.0 + 1e-9:
                st = world.tracks[bid]
                assert not any(tr.fire_control for tr in st.tracks.values())
                assert not any("radar" in tr.components and
                               tr.components["radar"].last_t > t0 for tr in st.tracks.values())
                state["checked"] = state.get("checked", 0) + 1

    probe_run(SEEDS[1], [x * 0.5 for x in range(0, 500)], fn)
    assert state.get("checked", 0) > 10


def test_default_radar_on_unchanged():
    a = sample_presentation(SEEDS[8])
    r1 = run_presentation(a, record=True)
    for f in r1.frames:
        for st in f["aircraft"].values() if isinstance(f.get("aircraft"), dict) else []:
            assert st.get("radar", True) is True
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "x.acmi"
        export_presentation_acmi(r1, p)
        assert "RadarMode" not in p.read_text()


# ------------------------------------------------------------ adapters ----
@pytest.mark.parametrize("i", [0, 1, 10])
def test_layer1_scripted_through_interface_identical(i):
    """Layer 1: 0.5 s encode -> decode of the script reproduces the direct
    fight (ACMI bytes, SimResult, events) with 0 fire-target misses."""
    d = layer1_job(SEEDS[i])
    assert d["n_fire_miss"] == 0
    assert d["result_identical"]
    assert d["acmi_identical"]


def test_hand_policy_uses_obs_only():
    pol = HandBlue()
    obs = np.zeros((2, 231))
    obs[0, 0] = obs[1, 1] = 1.0
    out = pol(obs)
    assert out.shape == (2, 13) and np.isfinite(out).all()
    assert (out[:, A.target0 + A.k] == 1.0).all()          # nothing seen -> none


def test_acmi_negative_zero_normalized():
    """Spec 5 follow-up (Rusty 2026-09-26): -0.0 is written as 0.0 in every
    numeric field; object removals and ordinary negatives are untouched.
    Seed index 10 above is the fight whose ACMI differed only by -0.0."""
    from stealth_tactics.acmi.exporter import _no_neg_zero as f
    assert f("1B,T=-115.045491|35.087479|2656.0|0.0|-0.0|158.1") == \
        "1B,T=-115.045491|35.087479|2656.0|0.0|0.0|158.1"
    assert f("-0A01") == "-0A01" and f("-1B19E4") == "-1B19E4"
    assert f("a=-0.00,b=-0.5,c=-0,d=-0.01,e=-01") == "a=0.00,b=-0.5,c=0,d=-0.01,e=-01"
    assert f("0,ReferenceTime=2026-01-01T00:00:00Z") == "0,ReferenceTime=2026-01-01T00:00:00Z"
