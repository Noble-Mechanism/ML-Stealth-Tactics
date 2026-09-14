"""TacView ACMI 2.2 UTF-8 text exporter.

Spec: https://raia-software-inc.gitbook.io/tacview/technical-documentation/acmi-telemetry-file-format
"""

from __future__ import annotations

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


class ACMIExporter:
    """Export simulation frames to ACMI 2.2 .txt.acmi."""

    def __init__(
        self,
        reference_time: Optional[datetime] = None,
        title: str = "ML Stealth Tactics Engagement",
    ) -> None:
        self.reference_time = reference_time or datetime(
            2026, 9, 14, 12, 0, 0, tzinfo=timezone.utc
        )
        self.title = title
        self._id_map: Dict[str, str] = {}

    def object_id(self, uid: str, kind: str = "ac") -> str:
        if uid not in self._id_map:
            prefix = 0xA00 if kind == "ac" else 0xC00
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

        introduced: set = set()
        last_alive: Dict[str, bool] = {}

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

                if uid not in introduced:
                    coalition = st["coalition"]
                    color = "Blue" if coalition == "Blue" else "Red"
                    lines.append(
                        f"{oid},Name={st['name']},Type=Air+FixedWing,{t_str},"
                        f"Coalition={coalition},Color={color},Pilot={st['name']}"
                    )
                    introduced.add(uid)
                else:
                    lines.append(f"{oid},{t_str}")
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
