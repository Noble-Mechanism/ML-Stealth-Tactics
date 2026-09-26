"""Discrete-time air combat world loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

import numpy as np

from .aircraft import Aircraft, Coalition, integrate_aircraft
from .sensors import SensorModel
from .sensor_config import SensorConfig, DEFAULT_SENSOR_CONFIG
from .tracks import TrackStore
from .datalink import Datalink
from .weapons import WeaponModel, Missile
from .rwr import RwrModel, RwrCue

# Spec 3 firing doctrines (approved change C), per jet, both sides.
SHOOT_ASSESS_SHOOT = "shoot_assess_shoot"   # max 1 own missile in flight per target
SHOOT_SHOOT_ASSESS = "shoot_shoot_assess"   # 2 at the same target 3 s apart, then
                                            # nothing more at THAT target until both
                                            # resolve (other targets: new pair at once)
LEGACY = "legacy"                           # pre-spec-3: fire whenever gates allow
DOCTRINES = (SHOOT_ASSESS_SHOOT, SHOOT_SHOOT_ASSESS, LEGACY)


@dataclass
class SimConfig:
    dt: float = 0.5  # seconds — coarse for GA speed
    max_time_s: float = 300.0
    seed: int = 42
    # None => sensor_config.DEFAULT_SENSOR_CONFIG
    sensor_config: Optional[SensorConfig] = None
    # Spec 3 firing doctrine per coalition (Aircraft.firing_doctrine overrides
    # per jet). Red default shoot_assess_shoot (spec 4 randomizes); Blue is a
    # setting (spec 5 may make it a network output). "legacy" = old behaviour.
    blue_doctrine: str = SHOOT_ASSESS_SHOOT
    red_doctrine: str = SHOOT_ASSESS_SHOOT
    ssa_interval_s: float = 3.0
    # D9: end once every live Red jet has departed and no missile is in flight
    early_end_on_departure: bool = True


@dataclass
class SimResult:
    time_s: float
    blue_kills: int
    red_kills: int  # = blue losses
    blue_alive: int
    red_alive: int
    blue_ammo_remaining: int
    winner: str  # "blue", "red", "draw"
    blue_shots: int = 0  # Blue missile launches (engagement signal)
    events: List[dict] = field(default_factory=list)
    # Sensor events (radar/irst/rwr detect + lost, fc_track/fc_lost, track_lost)
    sensor_events: List[dict] = field(default_factory=list)
    # Snapshot history for ACMI: list of (t, states dict)
    frames: List[dict] = field(default_factory=list)
    # Spec 3: one entry per missile (shooter, target, launch t / range, off-nose,
    # a-pole, f-pole, outcome, Mach at end, Pk, target defense state at launch
    # and at the end)
    shots: List[dict] = field(default_factory=list)
    red_shots: int = 0
    ended_early: bool = False


def _rng(a: Aircraft, b: Aircraft) -> float:
    return float(np.linalg.norm(a.state.position() - b.state.position()))


class World:
    """Fast point-mass engagement simulation."""

    def __init__(
        self,
        aircraft: List[Aircraft],
        config: SimConfig,
        blue_controller: Callable[[World], None],
        red_controller: Callable[[World], None],
        record: bool = True,
    ) -> None:
        self.aircraft = aircraft
        self.config = config
        self.blue_controller = blue_controller
        self.red_controller = red_controller
        self.record = record
        self.rng = np.random.default_rng(config.seed)
        # Sensors draw from an independent, seed-derived stream so adding sensor
        # randomness does not perturb the weapon Pk stream (still deterministic).
        cfg = config.sensor_config or DEFAULT_SENSOR_CONFIG
        self.sensor_rng = np.random.default_rng([config.seed, cfg.rng_salt])
        self.sensors = SensorModel(self.sensor_rng, cfg)
        self.sensor_cfg = cfg
        # Spec 2 datalink: own seed-derived RNG stream (message loss draws)
        self.datalink = Datalink(
            np.random.default_rng([config.seed, cfg.datalink.rng_salt]), cfg)
        self.weapons = WeaponModel(self.rng, cfg)
        # Optional hook(world, frame) run after each sensor update (replay markers)
        self.frame_hook: Optional[Callable[["World", dict], None]] = None
        self.missiles: List[Missile] = []
        self.time_s = 0.0
        self.events: List[dict] = []
        self.frames: List[dict] = []
        self._by_id: Dict[str, Aircraft] = {a.id: a for a in aircraft}
        # observer id -> TrackStore (same objects as self.sensors.stores)
        self.tracks: Dict[str, TrackStore] = self.sensors.stores
        self.sensor_events: List[dict] = []
        self.blue_shots = 0
        self.red_shots = 0
        self._missile_end_recorded: set = set()
        # Spec 3 RWR modes (own RNG stream so spec 1-3a draws do not shift)
        self.rwr_model = RwrModel(
            np.random.default_rng([config.seed, cfg.rwr.mode_rng_salt]), cfg)
        self.rwr: Dict[str, List[RwrCue]] = self.rwr_model.cues
        self.rwr_events: List[dict] = []
        # Spec 3 shoot-shoot-assess open pair per jet (only the first shot fired):
        # (target, t_first, first missile id)
        self._salvo: Dict[str, tuple] = {}
        self._shot_info: Dict[str, dict] = {}
        self.ended_early = False

    def get(self, uid: str) -> Optional[Aircraft]:
        return self._by_id.get(uid)

    def alive(self, coalition: Optional[Coalition] = None) -> List[Aircraft]:
        out = [a for a in self.aircraft if a.state.alive]
        if coalition is not None:
            out = [a for a in out if a.coalition == coalition]
        return out

    def run(self) -> SimResult:
        self._record_frame()
        dt = self.config.dt
        while self.time_s < self.config.max_time_s:
            if not self.alive(Coalition.BLUE) or not self.alive(Coalition.RED):
                break
            if self.config.early_end_on_departure and self._all_red_departed():
                self.ended_early = True
                break

            self._update_sensors()
            self._update_rwr()

            # Controllers set cmd_* and cmd_fire
            self.blue_controller(self)
            self.red_controller(self)

            # Process fire commands
            for ac in self.aircraft:
                if ac.cmd_fire and ac.fire_target and ac.state.alive and not ac.departed:
                    tgt_id = self._doctrine_target(ac, ac.fire_target)
                    tgt = self._by_id.get(tgt_id) if tgt_id else None
                    fc_src = self.fire_control_source(ac, tgt.id) if tgt else None
                    if tgt and fc_src is not None:
                        remote = None if fc_src == ac.id else fc_src
                        m = self.weapons.launch(ac, tgt, remote_source=remote)
                        if m:
                            self.missiles.append(m)
                            m.launch_t = self.time_s
                            self._after_launch(ac, tgt, m)
                            if ac.coalition == Coalition.BLUE:
                                self.blue_shots += 1
                            else:
                                self.red_shots += 1
                            self.events.append({
                                "t": self.time_s,
                                "type": "launch",
                                "shooter": ac.id,
                                "target": tgt.id,
                                "missile": m.id,
                                "remote_source": remote,
                            })
                            if remote:
                                self._frame_event({
                                    "t": self.time_s, "type": "launch_remote",
                                    "observer": ac.id, "target": tgt.id,
                                    "missile": m.id, "range_m": _rng(ac, tgt),
                                    "text": f"{ac.name}: {m.id} LAUNCH ON REMOTE "
                                            f"(FC track from {remote})"})
                            else:
                                self._frame_event({
                                    "t": self.time_s, "type": "launch",
                                    "observer": ac.id, "target": tgt.id,
                                    "missile": m.id, "range_m": _rng(ac, tgt),
                                    "text": f"{ac.name}: {m.id} launch (own FC)"})
                ac.cmd_fire = False
                ac.fire_target = None

            # Integrate aircraft
            for ac in self.aircraft:
                integrate_aircraft(ac, dt)

            # Missiles (datalink support uses current tracks)
            killed = self.weapons.step(
                self.missiles, self._by_id, dt, tracks=self.tracks, t=self.time_s
            )
            for ev in self.weapons.events:
                self.events.append(ev)
                self._frame_event(ev)
            self.weapons.events = []
            for kid in killed:
                self.events.append({"t": self.time_s, "type": "kill", "target": kid})
            for m in self.missiles:
                info = self._shot_info.get(m.id)
                if info is not None and m.end_t is not None and "target_state_end" not in info:
                    tg = self._by_id.get(m.target_id)
                    info["target_state_end"] = tg.defense_state if tg else None

            self.time_s += dt
            self._record_frame()

        return self._make_result()

    # ------------------------------------------------------------ spec 3 --
    def doctrine_of(self, ac: Aircraft) -> str:
        if ac.firing_doctrine:
            return ac.firing_doctrine
        return (self.config.blue_doctrine if ac.coalition == Coalition.BLUE
                else self.config.red_doctrine)

    def own_missiles_in_flight(self, ac: Aircraft) -> List[Missile]:
        return [m for m in self.missiles if m.alive and m.shooter_id == ac.id]

    def _doctrine_target(self, ac: Aircraft, requested: str) -> Optional[str]:
        """Target the doctrine allows *ac* to fire at now (None = hold fire).

        Both doctrines are per contact (the network decides when and how many
        shots come off; doctrine only limits missiles per contact).
        shoot_assess_shoot: at most one own missile in flight per target.
        shoot_shoot_assess (per contact): a pair at one target, the second shot
        ``ssa_interval_s`` after the first (a due second shot is redirected to
        the pair's target). While the pair is open (second not yet fired, first
        still flying) the jet fires at nothing else. After that, a target is
        blocked only while one of the jet's own missiles aimed at it is still
        in flight; other targets can be engaged with a new pair at once."""
        doc = self.doctrine_of(ac)
        if doc == LEGACY:
            return requested
        live = self.own_missiles_in_flight(ac)
        if doc == SHOOT_ASSESS_SHOOT:
            # per contact: at most one own missile in flight at this target
            return None if any(m.target_id == requested for m in live) else requested
        if doc == SHOOT_SHOOT_ASSESS:
            salvo = self._salvo.get(ac.id)
            if salvo is not None:
                tgt_id, t1, mid1 = salvo
                first_alive = any(m.id == mid1 for m in live)
                tgt = self._by_id.get(tgt_id)
                if first_alive and tgt is not None and tgt.state.alive:
                    if self.time_s >= t1 + self.config.ssa_interval_s - 1e-9:
                        return tgt_id          # second shot of the open pair
                    return None                # pair open: wait for shot 2
                self._salvo.pop(ac.id, None)   # pair lapsed (first resolved early)
            if any(m.target_id == requested for m in live):
                return None                    # this contact still has a pair out
            return requested
        raise ValueError(f"unknown firing doctrine {doc!r}")

    def may_fire_at(self, ac: Aircraft, tgt_id: str) -> bool:
        """True if the firing doctrine lets *ac* launch at *tgt_id* this step
        (launch gates such as FC, range and ammo are checked separately)."""
        return self._doctrine_target(ac, tgt_id) == tgt_id

    def _after_launch(self, ac: Aircraft, tgt: Aircraft, m: Missile) -> None:
        if self.doctrine_of(ac) == SHOOT_SHOOT_ASSESS:
            salvo = self._salvo.get(ac.id)
            if salvo is None:
                self._salvo[ac.id] = (tgt.id, self.time_s, m.id)
            else:
                self._salvo.pop(ac.id, None)   # pair complete: per-contact block now
        self._shot_info[m.id] = {"target_state_launch": tgt.defense_state}

    def _all_red_departed(self) -> bool:
        reds = self.alive(Coalition.RED)
        return (bool(reds) and all(r.departed for r in reds)
                and not any(m.alive for m in self.missiles))

    def log_event(self, ev: dict) -> None:
        """Controller events (defend / recommit / press / depart ...): event
        log + this frame's ACMI events."""
        self.events.append(ev)
        self._frame_event(ev)

    def _update_rwr(self) -> None:
        evts = self.rwr_model.update(self.aircraft, self.missiles, self.tracks, self.time_s)
        self.rwr = self.rwr_model.cues
        if evts:
            self.rwr_events.extend(evts)
            if self.record and self.frames and abs(self.frames[-1]["t"] - self.time_s) < 1e-9:
                self.frames[-1].setdefault("events", []).extend(evts)

    def _shots_table(self) -> List[dict]:
        out = []
        for m in self.missiles:
            tgt = self._by_id.get(m.target_id)
            info = self._shot_info.get(m.id, {})
            out.append({
                "missile": m.id, "shooter": m.shooter_id, "target": m.target_id,
                "coalition": m.coalition, "launch_t": m.launch_t,
                "launch_range_m": m.launch_range_m, "off_nose_deg": m.off_nose_deg,
                "a_pole_m": m.a_pole_m, "f_pole_m": m.f_pole_m,
                "outcome": m.outcome if not m.alive else "in_flight",
                "mach_end": m.mach_end, "pk": m.pk_final,
                "target_state_launch": info.get("target_state_launch"),
                "target_state_end": info.get("target_state_end",
                                             tgt.defense_state if tgt is not None else None),
                "end_t": m.end_t,
            })
        return out

    def fire_control_source(self, ac: Aircraft, target_id: str) -> Optional[str]:
        """Who provides fire-control quality on target for a launch by *ac*:
        ac.id (own FC), a flightmate id (remote FC over the link, Spec 2 policy),
        or None."""
        store = self.tracks.get(ac.id)
        if store is None:
            return None
        if store.is_fire_control(target_id):
            return ac.id
        dl = self.sensor_cfg.datalink
        coal = ac.coalition.value
        if dl.weapons_policy_shared(coal):
            return store.remote_fc_source(target_id, self.time_s, dl.remote_fc_max_age_s(coal))
        return None

    def _frame_event(self, ev: dict) -> None:
        if self.record and self.frames and abs(self.frames[-1]["t"] - self.time_s) < 1e-9:
            self.frames[-1].setdefault("events", []).append(ev)

    def _update_sensors(self) -> None:
        """Datalink delivery -> own sensors (+ maintenance + fusion) -> datalink send.
        Annotates the frame for this time."""
        evts = self.datalink.deliver(self.aircraft, self.sensors.stores, self.time_s)
        evts += self.sensors.update(self.aircraft, self.time_s)
        evts += self.datalink.send(self.aircraft, self.sensors.stores, self.time_s)
        self.sensor_events.extend(evts)
        if self.record and self.frames and abs(self.frames[-1]["t"] - self.time_s) < 1e-9:
            fr = self.frames[-1]
            fr.setdefault("sensor_events", []).extend(evts)
            for ac in self.aircraft:
                if ac.id in fr["aircraft"]:
                    fr["aircraft"][ac.id]["locked_target"] = self.sensors.primary_fc_target(ac.id)
            if self.frame_hook is not None:
                self.frame_hook(self, fr)

    def _record_frame(self) -> None:
        if not self.record:
            return
        states = {}
        for ac in self.aircraft:
            states[ac.id] = {
                "x": ac.state.x,
                "y": ac.state.y,
                "alt": ac.state.alt,
                "heading": ac.state.heading_rad,
                "speed": ac.state.speed_mps,
                "alive": ac.state.alive,
                "name": ac.name,  # callsign → ACMI Pilot=
                "type_name": ac.type_name,  # TacView Name=
                "type": ac.ac_type.value,
                "coalition": ac.coalition.value,
                "ammo": ac.ammo,
                # closest fire-control track (ACMI LockedTarget); refreshed after
                # the sensor update at this same time
                "locked_target": self.sensors.primary_fc_target(ac.id),
            }
        mstates = []
        for m in self.missiles:
            # Dead missiles are recorded once more (alive=False) so the ACMI
            # exporter removes the object at the end of its flight.
            if m.alive or m.id not in self._missile_end_recorded:
                if not m.alive:
                    self._missile_end_recorded.add(m.id)
                mstates.append({
                    "id": m.id,
                    "x": m.pos.x,
                    "y": m.pos.y,
                    "alt": m.pos.alt,
                    "alive": m.alive,
                    "coalition": m.coalition,
                    "hit": m.hit,
                    "mach": m.mach,
                    "speed": m.pos.speed_mps,
                    "heading": m.pos.heading_rad,
                    "pitch": m.pos.pitch_rad,
                })
        self.frames.append({"t": self.time_s, "aircraft": states, "missiles": mstates})

    def _make_result(self) -> SimResult:
        blue_alive = len(self.alive(Coalition.BLUE))
        red_alive = len(self.alive(Coalition.RED))
        n_blue = sum(1 for a in self.aircraft if a.coalition == Coalition.BLUE)
        n_red = sum(1 for a in self.aircraft if a.coalition == Coalition.RED)
        blue_kills = n_red - red_alive
        red_kills = n_blue - blue_alive
        blue_ammo = sum(a.ammo for a in self.aircraft if a.coalition == Coalition.BLUE)

        if red_alive == 0 and blue_alive > 0:
            winner = "blue"
        elif blue_alive == 0 and red_alive > 0:
            winner = "red"
        else:
            winner = "draw"

        return SimResult(
            time_s=self.time_s,
            blue_kills=blue_kills,
            red_kills=red_kills,
            blue_alive=blue_alive,
            red_alive=red_alive,
            blue_ammo_remaining=blue_ammo,
            winner=winner,
            blue_shots=self.blue_shots,
            events=self.events,
            sensor_events=self.sensor_events,
            frames=self.frames,
            shots=self._shots_table(),
            red_shots=self.red_shots,
            ended_early=self.ended_early,
        )
