"""Spec 6 O: deterministic npz (fixed zip timestamps) and atomic writes, so a
checkpoint written by a resumed run is byte-identical to an uninterrupted one."""

from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path
from typing import Dict

import numpy as np

_EPOCH = (1980, 1, 1, 0, 0, 0)


def save_npz(path, arrays: Dict[str, np.ndarray]) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arrays[name]),
                                      allow_pickle=False)
            zi = zipfile.ZipInfo(name + ".npy", date_time=_EPOCH)
            zi.external_attr = 0o644 << 16
            zf.writestr(zi, buf.getvalue())
    os.replace(tmp, path)


def load_npz(path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def write_json(path, obj) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
