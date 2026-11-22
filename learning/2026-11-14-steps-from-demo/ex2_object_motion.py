"""Exercise 2: find grasp and release from the OBJECT, not the hand.

Run (from the repo root, venv active). Use your original take 1 video, or tracked.mp4 from
the project files (video-app/take1). About 20 s on a laptop CPU:
    python learning\\2026-11-14-steps-from-demo\\ex2_object_motion.py C:\\path\\to\\take1.mp4

Calls find_object_motion (product/teach/retarget/object_motion.py) and prints its two
"looks like" curves as text, so you can see how it decides. Writes ex2_object_motion.jpg
in the current folder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "product" / "teach"))

import numpy as np  # noqa: E402

from retarget.object_motion import NoObjectMotion, debug_image, find_object_motion, read_small  # noqa: E402


def bar(v: float) -> str:
    return ("#" * int(round(max(v, 0) * 20))).ljust(20)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float)
    a = p.parse_args()

    frames, fps, scale = read_small(a.video, a.start, a.end)
    print(f"{len(frames)} frames at {fps:.1f} fps, shrunk to {frames.shape[2]} x {frames.shape[1]} px")
    try:
        m = find_object_motion(frames, fps, scale)
    except NoObjectMotion as e:
        print(f"No object motion found: {e}")
        return 1
    print(f"object at pick ({m.pick_px[0]:.0f}, {m.pick_px[1]:.0f}) px, at place ({m.place_px[0]:.0f}, {m.place_px[1]:.0f}) px,"
          f" about {m.size_px[0]:.0f} x {m.size_px[1]:.0f} px, look-alike score {m.match:.2f}")
    g, r = m.t_grasp + a.start, m.t_release + a.start  # video time
    print("\n t (s)  A still shows the object   B already shows the object")
    step = max(1, int(round(fps / 2)))
    for i in range(0, len(frames), step):
        t = a.start + i / fps
        mark = ""
        if abs(t - g) <= 0.25:
            mark = "  <- grasp"
        elif abs(t - r) <= 0.25:
            mark = "  <- release"
        print(f" {t:5.1f}  {m.like_start[i]:+.2f} {bar(m.like_start[i])}  {m.like_end[i]:+.2f} {bar(m.like_end[i])}{mark}")
    print(f"\ngrasp at {g:.2f} s, release at {r:.2f} s  ->  --grips {g:.2f}-{r:.2f}")
    for n in m.notes:
        print("note:", n)
    m.t_grasp, m.t_release = g, r
    debug_image(a.video, m, Path("ex2_object_motion.jpg"))
    print(f"wrote {Path('ex2_object_motion.jpg').resolve()}")
    print("\nCompare: take 1 steps.json was made with grips typed by hand: grasp 3.07 s, release 8.53 s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
