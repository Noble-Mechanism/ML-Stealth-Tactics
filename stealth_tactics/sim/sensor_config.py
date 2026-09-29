"""Single source of truth for every sensor number (Spec 1).

ALL values are UNCLASSIFIED PLACEHOLDERS chosen for plausible game balance.
They are NOT real F-35 / threat radar, IRST, RWR, or signature data.

Tune here (or build a modified ``SensorConfig`` and pass it via
``SimConfig.sensor_config``). ``aircraft.py`` pulls ``radar_range_m`` and the
base ``rcs_factor`` from this module so there is exactly one place to edit.

Keys of the per-type dicts are ``AircraftType`` values ("F-35", "RedFighter").
See docs/specs/01-sensors.md for the formulas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

F35_KEY = "F-35"
RED_KEY = "RedFighter"
NM_M = 1852.0


# ---------------------------------------------------------------- radar ----
@dataclass(frozen=True)
class RadarConfig:
    # Reference range vs a unit-RCS (1.0) target: R50 = ref_range * rcs**0.25.
    ref_range_m: Dict[str, float] = field(default_factory=lambda: {
        F35_KEY: 90_000.0,
        RED_KEY: 70_000.0,
    })
    # Per-scan Pd curve: Pd(R) = 1 / (1 + (R/R50)**k), with
    #   k = 40 / (snr_slope_db * ln 10)
    # i.e. a logistic in SNR(dB) with SNR_dB = 40*log10(R50/R) (radar eq R^-4).
    # snr_slope_db = logistic scale in dB (Swerling-1-like steepness ~2 dB).
    snr_slope_db: float = 2.0
    # Hard instrumented-range cap (fraction of ref_range_m): no detections past it.
    instrumented_range_factor: float = 1.5
    # Field of regard (half-angles, deg from the nose; az and el independently).
    for_az_deg: float = 60.0
    for_el_deg: float = 60.0
    # Measurement noise (1-sigma) used for track position estimates.
    range_sigma_m: float = 50.0
    angle_sigma_deg: float = 0.5
    velocity_sigma_mps: float = 10.0
    # Scan period (s): radar updates once per second.
    scan_period_s: float = 1.0

    @property
    def pd_exponent(self) -> float:
        return 40.0 / (self.snr_slope_db * math.log(10.0))


# ------------------------------------------------------------ signature ----
@dataclass(frozen=True)
class SignatureConfig:
    # Azimuth-only aspect table: (aspect_deg, rcs_factor). Aspect 0 = observer on
    # the target's nose, 90 = beam, 180 = tail. Interpolated with a piecewise
    # cosine ease (C1-smooth, zero slope at each knot, monotone per segment).
    tables: Dict[str, Tuple[Tuple[float, float], ...]] = field(default_factory=lambda: {
        # Spec 3 (approved change A): flat best RCS within 20 deg of the nose,
        # modest penalty for a 35-45 deg crank (Red detection range x1.18 at
        # 35 deg, x1.24 at 45 deg vs nose-on), steep rise to the beam. Was
        # (0, .05) (30, .10) (60, .45) (90, .90) (135, .55) (180, .30).
        F35_KEY: (
            (0.0, 0.05),
            (20.0, 0.05),
            (45.0, 0.12),
            (70.0, 0.50),
            (90.0, 0.90),
            (135.0, 0.55),
            (180.0, 0.30),
        ),
    })
    # Types without a table are isotropic with this rcs_factor.
    isotropic_rcs: Dict[str, float] = field(default_factory=lambda: {
        RED_KEY: 1.0,
    })
    default_rcs: float = 1.0


# ----------------------------------------------------------------- IRST ----
@dataclass(frozen=True)
class IRSTConfig:
    # 50%-Pd range vs a reference target at reference speed, by OBSERVER type:
    # nose-on (cold) and tail-on (hot, exhaust visible).
    nose_r50_m: Dict[str, float] = field(default_factory=lambda: {
        F35_KEY: 25_000.0,
        RED_KEY: 30_000.0,   # Red gets the better IRST: its counter to stealth
    })
    tail_r50_m: Dict[str, float] = field(default_factory=lambda: {
        F35_KEY: 50_000.0,
        RED_KEY: 60_000.0,
    })
    # Target IR signature scale by TARGET type (1.0 = no IR reduction).
    target_ir_scale: Dict[str, float] = field(default_factory=lambda: {
        F35_KEY: 1.0,
        RED_KEY: 1.0,
    })
    # Speed (engine power proxy): R50 *= clip((v/ref_speed)**speed_exponent, lo, hi)
    ref_speed_mps: float = 250.0
    speed_exponent: float = 1.0
    speed_factor_min: float = 0.5
    speed_factor_max: float = 1.5
    # Pd(R) = 1/(1+(R/R50)**k), k = 20/(snr_slope_db*ln10): IR point-source
    # signal ~ R^-2 (20 log10). Atmospheric extinction steepens the real falloff,
    # folded into a 1.0 dB logistic slope -> k ~ 8.7 (same steepness as radar).
    snr_slope_db: float = 1.0
    max_range_factor: float = 2.0   # skip Pd draw beyond this * R50 (Pd ~ 0)
    for_az_deg: float = 60.0
    for_el_deg: float = 60.0
    bearing_sigma_deg: float = 0.5      # accurate angles (az and el)
    range_error_frac: float = 0.30      # poor range: 1-sigma multiplicative error
    scan_period_s: float = 1.0

    @property
    def pd_exponent(self) -> float:
        return 20.0 / (self.snr_slope_db * math.log(10.0))


# ------------------------------------------------------------------ RWR ----
@dataclass(frozen=True)
class RWRConfig:
    # Enemy RWR detects an EMITTER of this type out to factor * emitter ref range,
    # provided the RWR's aircraft is inside that emitter's radar field of regard.
    emitter_detect_factor: Dict[str, float] = field(default_factory=lambda: {
        F35_KEY: 0.60,  # LPI radar: hard to intercept
        RED_KEY: 1.50,  # conventional radar: intercepted well beyond its own range
    })
    bearing_sigma_deg: float = 5.0  # coarse bearing-only contact
    update_period_s: float = 0.0    # 0 => every sim step (0.5 s)
    # Spec 3 RWR modes (sim/rwr.py): per-mode range factor on the intercept
    # range above (1.0 = same gate for search / lock / support; e.g. < 1 makes
    # an LPI lock harder to see). Missile-active warnings have no range gate.
    mode_range_factor: Dict[str, float] = field(default_factory=lambda: {
        "search": 1.0, "lock": 1.0, "support": 1.0,
    })
    # Own RNG stream for RWR-mode bearing noise so spec 1-3a draws do not shift.
    mode_rng_salt: int = 0x3D3F


# --------------------------------------------------------------- tracks ----
@dataclass(frozen=True)
class TrackConfig:
    coast_s: float = 10.0            # drop a sensor contact / track 10 s after last hit
    # Fire-control quality: radar-derived, range <= fc_range_frac * R50,
    # held continuously >= fc_min_hold_s. "Continuous" tolerates one missed
    # scan: the streak breaks if no radar hit for > fc_max_gap_s.
    fc_range_frac: float = 0.70
    fc_min_hold_s: float = 3.0
    fc_max_gap_s: float = 2.0
    # Spec 2b: per-observer-type FC range gate. If a type is listed, its FC gate is
    # fc_range_ref_m[type] * rcs_eff^0.25 (ref = vs a unit-RCS target) instead of
    # fc_range_frac * R50. Blue F-35: 50 NM (92.6 km) vs RCS 1.0. Red: unchanged.
    fc_range_ref_m: Dict[str, float] = field(default_factory=lambda: {F35_KEY: 92_600.0})
    # Track position sigma shrinks with hold time: sigma = meas_sigma/sqrt(n_eff),
    # n_eff = min(1 + held_s/scan_period, max_integration). Coast growth adds
    # coast_growth_mps * time_since_last_hit.
    max_integration: float = 10.0
    coast_growth_mps: float = 50.0


# ------------------------------------------------------ datalink (Spec 2) ---
BLUE = "Blue"
RED = "Red"


@dataclass(frozen=True)
class DatalinkConfig:
    """Intra-flight track sharing. Keys are Coalition values ("Blue"/"Red")."""
    # Each sender broadcasts its OWN radar/IRST tracks every update_period_s;
    # the message arrives latency_s later.
    update_period_s: Dict[str, float] = field(default_factory=lambda: {BLUE: 1.0, RED: 2.0})
    latency_s: Dict[str, float] = field(default_factory=lambda: {BLUE: 1.0, RED: 2.0})
    # One message per sender per update; lost (for all recipients) with this prob.
    loss_prob: Dict[str, float] = field(default_factory=lambda: {BLUE: 0.05, RED: 0.05})
    # Link only between flightmates within this range (checked at send time).
    max_range_m: float = 150.0 * NM_M
    shared_sources: Tuple[str, ...] = ("radar", "irst")   # RWR is NOT shared
    # A remote fire-control report is usable for launch while its snapshot age
    # <= latency + remote_fc_periods * update_period (Blue: 1 + 2*1 = 3 s,
    # i.e. tolerates one lost message).
    remote_fc_periods: float = 2.0
    # IRST triangulation (passive ranging from two jets' IRST bearings)
    triangulation: Dict[str, bool] = field(default_factory=lambda: {BLUE: True, RED: False})
    tri_min_angle_deg: float = 10.0      # no fix when LOS differ by less than this
    tri_max_angle_deg: float = 170.0     # (near-opposite LOS is degenerate too)
    tri_max_age_s: float = 5.0           # only use IRST bearings this fresh
    # Weapons policy (Spec 2 rule 5): remote launch on a flightmate's FC track +
    # midcourse support handoff to any flightmate holding its own FC track.
    remote_launch_and_handoff: Dict[str, bool] = field(
        default_factory=lambda: {BLUE: True, RED: False})
    # Single switch: True => Red gets the same weapons policy as Blue (symmetric).
    symmetric_weapons_policy: bool = False
    # Supporting jet must be within this range of the missile (weapon link).
    weapon_link_range_m: float = 150.0 * NM_M
    rng_salt: int = 0xD17A

    def remote_fc_max_age_s(self, coalition: str) -> float:
        return (self.latency_s[coalition]
                + self.remote_fc_periods * self.update_period_s[coalition])

    def weapons_policy_shared(self, coalition: str) -> bool:
        if self.symmetric_weapons_policy:
            return bool(self.remote_launch_and_handoff.get(BLUE, True))
        return bool(self.remote_launch_and_handoff.get(coalition, False))


@dataclass(frozen=True)
class MissileCoastConfig:
    """Spec 2b (+ Spec 3a Pk formula): missile behaviour when midcourse support
    lapses before the seeker goes active."""
    # Coasting missile flies at the target's last supported position extrapolated
    # with the last known velocity. Support may resume any time.
    # Spec 3a coast Pk factor at the active point (replaces exp(-t/12 s)):
    #   f_coast = exp(-(e / coast_err_scale_m)**2)
    #             * (1 - coast_time_penalty * min(t_gap / coast_time_ref_s, 1))
    # e = aim-point error vs the TRUE target, t_gap = time since last update.
    coast_err_scale_m: float = 2000.0
    coast_time_penalty: float = 0.15
    # 3a fix: time scale 40 s (was 20 s) so the aim-point error term sets most
    # of the penalty; the time term alone costs at most 15 % at the 40 s timeout.
    coast_time_ref_s: float = 40.0
    # Lost (dud) if unsupported CONTINUOUSLY for longer than this before autonomy
    # (3a fix: 40 s, was 20 s; long shots need > 20 s to reach the active point).
    coast_timeout_s: float = 40.0
    # At the active point (within active_range_m of the TRUE target): lost if the
    # extrapolated aim point is farther than this from the true target.
    seeker_basket_m: float = 5000.0

    def coast_pk_factor(self, err_m: float, t_gap_s: float) -> float:
        return (math.exp(-(err_m / self.coast_err_scale_m) ** 2)
                * (1.0 - self.coast_time_penalty
                   * min(t_gap_s / self.coast_time_ref_s, 1.0)))



# ------------------------------------------------ aircraft energy (Spec 8) ---
@dataclass(frozen=True)
class AircraftEnergyConfig:
    """Spec 8 / 8b / 8c point-mass jet energy model. Unclassified placeholders,
    NOT real F-35 / adversary performance. Separate Blue / Red sets (A-b);
    single n_max (A-c).

    Spec 8b: available load factor = min(n_max, q S CLmax / W); Cd0 has a
    transonic rise (``cd0_factor``).
    Spec 8c (thrust retune): thrust = T_sl x sigma^thrust_density_exp below the
    tropopause, x (rho / rho_tropo)^thrust_density_exp_strat above it, x
    (1 + thrust_ram_gain x max(0, M - thrust_ram_ref_mach)). The Cd0 rise is
    x^cd0_rise_power (slow start, steep near the peak).
    """
    mass_kg: float = 18000.0
    S_m2: float = 40.0
    Cd0: float = 0.025
    k_induced: float = 0.16
    n_max: float = 7.0
    T_sl_N: float = 200000.0
    thrust_density_exp: float = 0.8
    # Lapse exponent above the tropopause (None = same as below)
    thrust_density_exp_strat: Optional[float] = 1.0
    thrust_tropo_rho: float = 0.36391780335853485    # ISA density at 11 km
    max_mach: float = 1.2
    max_alt_m: float = 15000.0
    min_speed_mps: float = 90.0
    # Spec 8b lift limit
    CLmax: float = 1.3
    # Transonic drag rise: Cd0 x 1 up to cd0_rise_mach, rising as x^cd0_rise_power
    # (x = fraction of the way to the peak) to cd0_peak_factor at cd0_peak_mach,
    # then declining cd0_decline_per_mach (factor per Mach) to cd0_supersonic_floor.
    cd0_rise_mach: float = 0.85
    cd0_peak_mach: float = 1.10
    cd0_peak_factor: float = 3.08
    cd0_rise_power: float = 3.0
    cd0_decline_per_mach: float = 4.0
    cd0_supersonic_floor: float = 1.5
    # Thrust vs Mach (ram recovery above the reference Mach); 0 in Spec 8c
    thrust_ram_gain: float = 0.0
    thrust_ram_ref_mach: float = 0.9

    def cd0_factor(self, mach: float) -> float:
        m0, mp = self.cd0_rise_mach, self.cd0_peak_mach
        if mach <= m0:
            return 1.0
        if mach <= mp:
            x = (mach - m0) / (mp - m0)
            return 1.0 + (self.cd0_peak_factor - 1.0) * x ** self.cd0_rise_power
        return max(self.cd0_supersonic_floor,
                   self.cd0_peak_factor - self.cd0_decline_per_mach * (mach - mp))

    def density_lapse(self, rho: float) -> float:
        rt = self.thrust_tropo_rho
        a2 = self.thrust_density_exp_strat
        if a2 is None or rho >= rt:
            return (rho / 1.225) ** self.thrust_density_exp
        return (rt / 1.225) ** self.thrust_density_exp * (rho / rt) ** a2

    def thrust_n(self, rho: float, mach: float) -> float:
        ram = 1.0 + self.thrust_ram_gain * max(0.0, mach - self.thrust_ram_ref_mach)
        return self.T_sl_N * self.density_lapse(rho) * ram


# Spec 8c calibration (see docs/specs/08c-thrust-retune.md). Red is ~25% better:
# ~25% more specific excess power and sustained g (lower k, more thrust), a
# 15% higher lift limit (CLmax 1.5), n_max 8, Mach ceiling 1.4.
BLUE_ENERGY = AircraftEnergyConfig(
    mass_kg=18000.0, S_m2=40.0, Cd0=0.025, k_induced=0.16, n_max=7.0,
    T_sl_N=200000.0, thrust_density_exp=0.8, thrust_density_exp_strat=1.0,
    max_mach=1.2, max_alt_m=15000.0, min_speed_mps=90.0, CLmax=1.3,
    cd0_peak_mach=1.10, cd0_peak_factor=3.08, cd0_rise_power=3.0,
    cd0_decline_per_mach=4.0, thrust_ram_gain=0.0,
)
RED_ENERGY = AircraftEnergyConfig(
    mass_kg=18000.0, S_m2=40.0, Cd0=0.028, k_induced=0.12, n_max=8.0,
    T_sl_N=232000.0, thrust_density_exp=0.8, thrust_density_exp_strat=1.0,
    max_mach=1.4, max_alt_m=15000.0, min_speed_mps=85.0, CLmax=1.5,
    cd0_peak_mach=1.10, cd0_peak_factor=3.20, cd0_rise_power=3.0,
    cd0_decline_per_mach=4.0, thrust_ram_gain=0.0,
)


# ------------------------------------------------ missile kinematics (3a) ---
@dataclass(frozen=True)
class MissileKinematicsConfig:
    """Spec 3a point-mass missile fly-out + shared (both sides) Pk model.

    Prototype calibration values, NOT sourced AIM-120 data (only launch mass and
    diameter come from public references). See docs/specs/03a-missile-kinematics.md.
    """
    dt_s: float = 0.05                    # internal integration step
    # Motor / mass
    launch_mass_kg: float = 161.5
    propellant_kg: float = 50.0
    burn_time_s: float = 6.0
    isp_s: float = 250.0
    diameter_m: float = 0.178
    # Zero-lift drag Cd0(M): cd_subsonic below mach_rise_start, linear rise to
    # cd_peak at mach_cd_peak, then cd_peak * (mach_cd_peak / M)**cd_super_exp.
    cd_subsonic: float = 0.35
    cd_peak: float = 0.70
    mach_rise_start: float = 0.8
    mach_cd_peak: float = 1.1
    cd_super_exp: float = 0.6
    cd_scale: float = 1.10                # calibration knob (prototype 1.25 w/o gravity)
    # Induced drag k * (n m g)^2 / (q S); turn limit min(g_max, q S CLmax / (m g))
    k_induced: float = 0.08
    cl_max: float = 12.0
    g_max: float = 40.0
    # Lift carries the missile's weight (holds altitude unless guidance climbs/dives)
    gravity: bool = True
    # Proportional navigation gain
    pn_gain: float = 4.0
    # Spec 8 B: simple loft-bias midcourse (default on). Bias aim upward while
    # range-to-aim > loft_handoff_m and tf past loft_settle_s; linear decay to 0
    # at handoff. After handoff / once inside active range, pure PN.
    loft_enabled: bool = True
    loft_angle_deg: float = 20.0
    loft_handoff_m: float = 25.0 * NM_M
    loft_settle_s: float = 1.0
    # Spec 8 B-d: defeat_opening only after this many continuous seconds of
    # opening, and never while loft bias is still active.
    opening_grace_s: float = 3.0
    # Seeker goes active this far from the TRUE target (midcourse -> terminal)
    active_range_m: float = 15.0 * NM_M
    # Kinematic defeat (after burnout): below min Mach (immediate), or opening
    # from the target after opening_grace_s of continuous opening (Spec 8 B-d;
    # never while loft bias is active). An opening missile closer than
    # overshoot_range_m is miss_overshoot (immediate).
    defeat_min_mach: float = 1.2
    defeat_on_opening: bool = True
    overshoot_range_m: float = 1000.0
    # safety cap (outcome "timeout"); 3a fix: 180 s (was 120 s) so energy, not
    # the cap, sets Rmax everywhere in the table
    max_flight_time_s: float = 180.0
    proximity_fuze_m: float = 50.0
    # Launch gates (range gate = Rmax lookup table)
    min_launch_range_m: float = 500.0
    max_off_boresight_deg: float = 60.0
    # Pk: same missile for Blue and Red
    base_pk: float = 0.60
    # Endgame energy f_E: 1.0 at >= endgame_mach_full, linear to endgame_pk_floor
    # at endgame_mach_floor (clipped below).
    endgame_mach_full: float = 2.0
    endgame_mach_floor: float = 1.2
    endgame_pk_floor: float = 0.6
    # Supported to active, then support dropped before impact: x0.90 once.
    # Gaps shorter than support_gap_tol_s count as continuous.
    support_drop_factor: float = 0.90
    support_gap_tol_s: float = 2.0
    # Rmax lookup table grid (built from this model at startup, cached on disk).
    # Axes: altitude (m, mean of shooter/target), shooter Mach, target aspect (deg,
    # 0 = hot / nose-on to the shooter, 180 = cold), target Mach, launch angle off
    # the shooter's nose (deg, signed: + = nose toward the side the target is
    # moving across the LOS ("lead"), - = "lag"). Co-altitude, straight-and-level
    # target; Rne = target turns cold at env_turn_g at launch.
    env_alt_m: Tuple[float, ...] = (1000.0, 3000.0, 5000.0, 7000.0, 9000.0, 10000.0,
                                    11000.0, 12000.0, 13000.0, 14000.0, 15000.0)
    env_shooter_mach: Tuple[float, ...] = (0.5, 0.7, 0.9, 1.1, 1.3)
    # 3a fix: 20 deg aspect steps to 120 (was 45): Rmax is curved in aspect,
    # 45 deg steps interpolated up to ~1.9 NM wrong between bins.
    # Spec 8: 10 deg steps to 120. Loft puts a cliff in Rmax vs aspect (full-
    # loft shots reach ~60+ NM at <=60 deg aspect, ~30-40 NM past it); 20 deg
    # bins over-read beam / lag Rmax by up to ~11 NM near 40 kft.
    env_aspect_deg: Tuple[float, ...] = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0,
                                         80.0, 90.0, 100.0, 110.0, 120.0, 150.0, 180.0)
    env_target_mach: Tuple[float, ...] = (0.5, 0.9, 1.3)
    # signed off-nose bins: 15 deg near the nose (Rmax is flat there), 5-10 deg
    # toward the 60 deg launch limit, denser on the lag side where Rmax bends
    # and then falls off a cliff (the turn bleeds the missile below Mach 1.2:
    # ~55-60 deg lag vs a Mach 0.9 target, ~40-45 deg vs Mach 1.3); 75 only
    # bounds interpolation past 60.
    env_off_nose_deg: Tuple[float, ...] = (-75.0, -60.0, -55.0, -50.0, -45.0, -40.0,
                                           -35.0, -30.0, -15.0, 0.0, 15.0, 30.0,
                                           40.0, 50.0, 55.0, 60.0, 75.0)
    env_turn_g: float = 3.0
    env_range_lo_nm: float = 1.0
    env_range_hi_nm: float = 140.0
    env_coarse_nm: float = 2.5
    env_tol_nm: float = 0.05

    def endgame_factor(self, mach: float) -> float:
        span = self.endgame_mach_full - self.endgame_mach_floor
        frac = min(1.0, max(0.0, (mach - self.endgame_mach_floor) / span))
        return self.endgame_pk_floor + (1.0 - self.endgame_pk_floor) * frac


@dataclass(frozen=True)
class SensorConfig:
    radar: RadarConfig = field(default_factory=RadarConfig)
    signature: SignatureConfig = field(default_factory=SignatureConfig)
    irst: IRSTConfig = field(default_factory=IRSTConfig)
    rwr: RWRConfig = field(default_factory=RWRConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    datalink: DatalinkConfig = field(default_factory=DatalinkConfig)
    missile: MissileCoastConfig = field(default_factory=MissileCoastConfig)
    missile_kinematics: MissileKinematicsConfig = field(
        default_factory=MissileKinematicsConfig)
    # Spec 8 E: per-coalition missile hook. None => Red uses missile_kinematics
    # (same missile). Do not tune a "superior" Red missile until A-D prove
    # insufficient.
    missile_kinematics_red: Optional[MissileKinematicsConfig] = None
    # Spec 8 A: jet energy params keyed by AircraftType value.
    aircraft_energy: Dict[str, AircraftEnergyConfig] = field(default_factory=lambda: {
        F35_KEY: BLUE_ENERGY,
        RED_KEY: RED_ENERGY,
    })
    # Independent RNG stream salt so sensor draws don't perturb weapon Pk draws.
    rng_salt: int = 0x5E45

    def kinematics_for(self, coalition: str) -> MissileKinematicsConfig:
        if coalition == RED and self.missile_kinematics_red is not None:
            return self.missile_kinematics_red
        return self.missile_kinematics


DEFAULT_SENSOR_CONFIG = SensorConfig()
