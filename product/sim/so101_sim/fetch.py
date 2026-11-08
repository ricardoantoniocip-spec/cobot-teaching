"""Download the official SO-101 MuJoCo model (MJCF + meshes) into product/sim/so101_model/.

Same source and commit as hardware/so101/so101_new_calib.urdf, so the sim and the
retarget kinematics use the same robot. About 16 MB, not stored in git.
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

COMMIT = "a758567c3978dfeefe282ede0500085a48fe8f78"
BASE = f"https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/{COMMIT}/Simulation/SO101"
MODEL_DIR = Path(__file__).resolve().parents[1] / "so101_model"
ARM_XML = MODEL_DIR / "so101_new_calib.xml"


def _get(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as r:
        dest.write_bytes(r.read())


def fetch(force: bool = False) -> Path:
    """Download the arm model if missing. Returns the path of so101_new_calib.xml."""
    if not ARM_XML.exists() or force:
        print(f"Downloading {BASE}/so101_new_calib.xml")
        _get(f"{BASE}/so101_new_calib.xml", ARM_XML)
    meshes = re.findall(r'<mesh file="([^"]+)"', ARM_XML.read_text())
    for name in meshes:
        dest = MODEL_DIR / "assets" / name
        if dest.exists() and not force:
            continue
        print(f"  mesh {name}")
        _get(f"{BASE}/assets/{name}", dest)
    print(f"Model ready: {ARM_XML} ({len(meshes)} meshes)")
    return ARM_XML


def ensure_model() -> Path:
    return ARM_XML if ARM_XML.exists() else fetch()
