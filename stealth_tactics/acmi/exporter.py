"""TacView ACMI 2.2 UTF-8 text exporter.

Spec: https://raia-software-inc.gitbook.io/tacview/technical-documentation/acmi-telemetry-file-format
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


from stealth_tactics.sim.geo import enu_to_llh, heading_rad_to_yaw_deg


def _stable_hex_id(uid: str, prefix: int = 0x100) -> str:
    """Deterministic hex object ID from string uid."""
    h = 0
    for ch in uid:
        h = (h * 131 + ord(ch)) & 0xFFFFFF
    return f"{prefix + h:X}"


def _escape(text: str) -> str:
    """ACMI text: escape commas; pipes separate event fields so replace them."""
    return text.replace("|", "/").replace(",", "\\,")


# Sensor event types emitted as TacView events (Spec 1). The first occurrence of
# BOOKMARK_TYPES per observer/target pair is a Bookmark (timeline highlight);
# everything else is a Message.
SENSOR_EVENT_TYPES = {
    "radar_detect", "irst_detect", "rwr_detect", "fc_track", "fc_lost",
    "radar_lost", "irst_lost", "rwr_lost", "track_lost",
}
BOOKMARK_TYPES = {"radar_detect", "fc_track", "rwr_detect", "irst_detect"}
# Spec 2: datalink / weapon events (frame["events"] and frame["sensor_events"])
LINK_WEAPON_TYPES = {"link_track", "launch", "launch_remote", "support_handoff",
                     "autonomous", "support_lost", "hit", "miss", "timeout", "tri_fix",
                     "support_regained", "lost_coast_timeout", "lost_basket",
                     "note",
                     # Spec 3a missile kinematics
                     "burnout", "support_dropped", "defeat_speed", "defeat_opening",
                     "miss_overshoot",
                     # Spec 3 missile defense
                     "ground", "rwr_mode", "defend", "threat_cleared", "recommit",
                     "press", "depart", "blue_defend", "blue_recommit",
                     # Spec 4 presentations
                     "preplanned_start", "preplanned_skip", "preplanned_abort",
                     "preplanned_end", "breakup", "winchester"}
ALWAYS_BOOKMARK = {"launch", "launch_remote", "support_handoff", "autonomous",
                   "support_lost", "hit", "miss", "timeout", "support_regained",
                   "lost_coast_timeout", "lost_basket", "support_dropped",
                   "defeat_speed", "defeat_opening", "miss_overshoot",
                   "ground", "defend", "recommit", "press", "depart",
                   "blue_defend", "blue_recommit",
                   "preplanned_start", "preplanned_abort", "winchester"}
NM_M = 1852.0


class ACMIExporter:
    """Export simulation frames to ACMI 2.2 .txt.acmi.

    Spec 1 additions (both optional, frames without them export as before):
    - ``frame["aircraft"][id]["locked_target"]``: closest fire-control track ->
      ``LockedTargetMode=1,LockedTarget=<hex id>`` (cleared with
      ``LockedTargetMode=0,LockedTarget=``).
    - ``frame["sensor_events"]``: ``0,Event=Message|<obs>|<tgt>|text`` (first
      detection per pair of each sensor / first fire-control track is a
      ``Bookmark``).

    Spec 3a: missiles with ``heading`` / ``pitch`` / ``mach`` / ``speed`` export
    orientation plus ``Mach=`` and ``TAS=`` (m/s) every frame; kinematic defeats
    (``defeat_speed``, ``defeat_opening``, ``miss_overshoot``) and
    ``support_dropped`` are bookmarks, ``burnout`` a message.

    Spec 3: ``defend`` / ``recommit`` / ``press`` / ``depart`` (and the Blue
    test reaction's ``blue_defend`` / ``blue_recommit``) and missile ``ground``
    are bookmarks; ``rwr_mode`` changes and ``threat_cleared`` are messages.
    """

    def __init__(
        self,
        reference_time: Optional[datetime] = None,
        title: str = "ML Stealth Tactics Engagement",
        sensor_events: bool = True,
        comments: Optional[str] = None,
    ) -> None:
        self.reference_time = reference_time or datetime(
            2026, 9, 14, 12, 0, 0, tzinfo=timezone.utc
        )
        self.title = title
        self.sensor_events = sensor_events
        # Spec 4: optional global Comments= (e.g. the presentation summary)
        self.comments = comments
        self._id_map: Dict[str, str] = {}

    def object_id(self, uid: str, kind: str = "ac") -> str:
        if uid not in self._id_map:
            prefix = {"ac": 0xA00, "missile": 0xC00, "marker": 0xE00}.get(kind, 0xC00)
            self._id_map[uid] = _stable_hex_id(uid, prefix=prefix)
        return self._id_map[uid]

    def export(self, frames: List[dict], out_path: Path | str) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        lines: List[str] = [
            "FileType=text/acmi/tacview",
            "FileVersion=2.2",
            f"0,ReferenceTime={self.reference_time.strftime('%Y-%m-%dT%H:%M:%SZ')}",
            f"0,Title={self.title}",
            "0,DataSource=ML-Stealth-Tactics",
            "0,Author=stealth_tactics",
        ]
        if self.comments:
            lines.append(f"0,Comments={_escape(self.comments)}")

        introduced: set = set()
        last_alive: Dict[str, bool] = {}
        last_lock: Dict[str, Optional[str]] = {}
        bookmarked: set = set()
        markers_live: set = set()

        for frame in frames:
            t = frame["t"]
            lines.append(f"#{t:.2f}")

            for uid, st in frame["aircraft"].items():
                oid = self.object_id(uid, kind="ac")
                alive = bool(st["alive"])

                if not alive:
                    if uid in introduced and last_alive.get(uid, True):
                        lines.append(f"-{oid}")
                        last_alive[uid] = False
                    continue

                lon, lat, alt = enu_to_llh(st["x"], st["y"], st["alt"])
                yaw = heading_rad_to_yaw_deg(st["heading"])
                # ACMI 2.2: T=lon|lat|alt|roll|pitch|yaw|...
                t_str = f"T={lon:.6f}|{lat:.6f}|{alt:.1f}|||{yaw:.1f}"

                lock_str = ""
                lock = st.get("locked_target")
                if lock != last_lock.get(uid):
                    if lock:
                        lock_str = (f",LockedTargetMode=1,"
                                    f"LockedTarget={self.object_id(lock, kind='ac')}")
                    elif uid in last_lock:
                        lock_str = ",LockedTargetMode=0,LockedTarget="
                    last_lock[uid] = lock

                if uid not in introduced:
                    coalition = st["coalition"]
                    color = "Blue" if coalition == "Blue" else "Red"
                    # Name= must match TacView DB (F-35A); Pilot= holds ship callsign
                    tac_name = st.get("type_name") or st["name"]
                    pilot = st["name"]
                    short = ""
                    if tac_name == "F-35A":
                        short = ",ShortName=F-35"
                    lines.append(
                        f"{oid},Name={tac_name},Type=Air+FixedWing,{t_str},"
                        f"Coalition={coalition},Color={color},Pilot={pilot}{short}"
                        f"{lock_str}"
                    )
                    introduced.add(uid)
                else:
                    lines.append(f"{oid},{t_str}{lock_str}")
                last_alive[uid] = True

            for m in frame.get("missiles", []):
                mid = m["id"]
                oid = self.object_id(mid, kind="missile")
                if not m["alive"]:
                    if mid in introduced and last_alive.get(mid, True):
                        lines.append(f"-{oid}")
                        last_alive[mid] = False
                    continue
                lon, lat, alt = enu_to_llh(m["x"], m["y"], m["alt"])
                if "heading" in m:     # Spec 3a: orientation + Mach / TAS
                    yaw = heading_rad_to_yaw_deg(m["heading"])
                    pitch = math.degrees(m.get("pitch", 0.0))
                    t_str = (f"T={lon:.6f}|{lat:.6f}|{alt:.1f}|0.0|{pitch:.1f}|{yaw:.1f},"
                             f"Mach={m['mach']:.2f},TAS={m['speed']:.0f}")
                else:
                    t_str = f"T={lon:.6f}|{lat:.6f}|{alt:.1f}"
                if mid not in introduced:
                    color = "Blue" if m["coalition"] == "Blue" else "Red"
                    lines.append(
                        f"{oid},Name=Missile,Type=Weapon+Missile,{t_str},"
                        f"Coalition={m['coalition']},Color={color}"
                    )
                    introduced.add(mid)
                else:
                    lines.append(f"{oid},{t_str}")
                last_alive[mid] = True

            # Spec 2 marker objects (e.g. triangulated position estimate)
            present = set()
            for mk in frame.get("markers", []):
                mid = mk["id"]
                oid = self.object_id(mid, kind="marker")
                present.add(mid)
                lon, lat, alt = enu_to_llh(mk["x"], mk["y"], mk["alt"])
                t_str = f"T={lon:.6f}|{lat:.6f}|{alt:.1f}"
                label = f",Label={_escape(mk['label'])}" if mk.get("label") else ""
                if mid not in markers_live:
                    lines.append(
                        f"{oid},Name={_escape(mk.get('name', mid))},"
                        f"Type={mk.get('type', 'Navaid+Static+Waypoint')},{t_str},"
                        f"Color={mk.get('color', 'Yellow')}{label}")
                    markers_live.add(mid)
                else:
                    lines.append(f"{oid},{t_str}{label}")
            for mid in sorted(markers_live - present):
                lines.append(f"-{self.object_id(mid, kind='marker')}")
                markers_live.discard(mid)

            if self.sensor_events:
                evs = list(frame.get("sensor_events", [])) + list(frame.get("events", []))
                for ev in evs:
                    etype = ev.get("type")
                    if etype not in SENSOR_EVENT_TYPES and etype not in LINK_WEAPON_TYPES:
                        continue
                    if "observer" not in ev or "target" not in ev:
                        continue
                    obs = self.object_id(ev["observer"], kind="ac")
                    tgt = self.object_id(ev["target"], kind="ac")
                    ids = f"{obs}|{tgt}"
                    if ev.get("missile"):
                        ids += f"|{self.object_id(ev['missile'], kind='missile')}"
                    text = ev.get("text", etype)
                    if "range_m" in ev and etype not in ("hit", "miss"):
                        rng = float(ev["range_m"])
                        text += f" @ {rng / NM_M:.1f} NM ({rng / 1000.0:.1f} km)"
                    key = (etype, ev["observer"], ev["target"])
                    kind = "Message"
                    if etype in ALWAYS_BOOKMARK or (
                            etype in BOOKMARK_TYPES and key not in bookmarked):
                        kind = "Bookmark"
                        bookmarked.add(key)
                    lines.append(f"0,Event={kind}|{ids}|{_escape(text)}")

        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out_path

    @staticmethod
    def validate_header(path: Path | str) -> bool:
        text = Path(path).read_text(encoding="utf-8")
        lines = text.splitlines()
        return (
            len(lines) >= 3
            and lines[0].strip() == "FileType=text/acmi/tacview"
            and lines[1].strip() == "FileVersion=2.2"
            and lines[2].startswith("0,ReferenceTime=")
        )
