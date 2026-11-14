"""Exercise 2: the grip ratio, its two thresholds, and why take 1 found no grip.

Run (from the repo root, venv active). No video and no MediaPipe needed:
    python learning\\2026-11-14-vision-front-end\\ex2_grip_ratio.py

Uses the made-up operator from product/teach/retarget/examples.py and the real functions
from product/teach/retarget/pose.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "product" / "teach"))

import numpy as np  # noqa: E402

from retarget import pose  # noqa: E402
from retarget.examples import synthetic_operator_frames  # noqa: E402
from retarget.kinematics import SerialChain  # noqa: E402


def bar(r: float) -> str:
    return "#" * int(round(r * 20))


# --- Part A: a pinch grasp. Grip ratio over time, every second.
frames = synthetic_operator_frames()  # 9 s at 30 Hz: rest, reach, pinch, carry, release, rest
print("A. Pinch grasp (made-up operator)")
print("   t (s)  ratio  state")
ratios = np.array([pose.grip_ratio(f.hand_world) for f in frames])
states = pose.grip_states(ratios, close_ratio=0.5, open_ratio=0.7)
for i in range(0, len(frames), 30):
    print(f"   {frames[i].t:4.1f}   {ratios[i]:.2f}   {'CLOSED' if states[i] else 'open  '}  {bar(ratios[i])}")

# --- Part B: why two thresholds? A real hand closes slowly (about 1 s) and the tracker
#     jitters. Ratio goes 1.0 -> 0.35 -> 1.0, plus noise. Count how often the grip flips.
t = np.arange(0, 6, 1 / 30)
slow = np.interp(t, [0, 1.5, 2.5, 3.5, 4.5, 6], [1.0, 1.0, 0.35, 0.35, 1.0, 1.0])
NOISE_SD = 0.06  # try 0.12
rng = np.random.default_rng(1)
noisy = slow + rng.normal(0, NOISE_SD, len(t))
one = (noisy < 0.6).astype(int)  # one threshold in the middle
two = pose.grip_states(noisy, 0.5, 0.7).astype(int)
print(f"\nB. Slow close and open, with tracker noise (sd {NOISE_SD})")
print(f"   one threshold (0.6):       grip changes {np.count_nonzero(np.diff(one))} times")
print(f"   two thresholds (0.5/0.7):  grip changes {np.count_nonzero(np.diff(two))} times  (one pick = 2)")

# --- Part C: a whole-hand grasp, like take 1. Thumb and index wrap round a 70 mm carton,
#     so the gap between their tips stays wide.
chain = SerialChain.from_urdf()
whole = synthetic_operator_frames()
for f in whole:
    hw = f.hand_world
    mid = (hw[pose.H_THUMB_TIP] + hw[pose.H_INDEX_TIP]) / 2
    d = (hw[pose.H_INDEX_TIP] - hw[pose.H_THUMB_TIP])
    d /= np.linalg.norm(d)
    hw[pose.H_THUMB_TIP], hw[pose.H_INDEX_TIP] = mid - 0.04 * d, mid + 0.04 * d  # 80 mm apart, all the time
r_whole = [pose.grip_ratio(f.hand_world) for f in whole]
res = pose.frames_to_demo(whole, chain)
steps = pose.segment_steps(res.demo)
print("\nC. Whole-hand grasp (thumb-index gap fixed at 80 mm)")
print(f"   grip ratio min {min(r_whole):.2f}, max {max(r_whole):.2f}: never below 0.5")
print(f"   steps found: {' > '.join(s['step'] for s in steps)}")
for w in res.warnings:
    print(f"   warning: {w}")

# --- Part D: the manual fix, --grips 3.0-6.5 (seconds of the video)
cfg = pose.PoseConfig(grip_events=[(3.0, 6.5)])
res2 = pose.frames_to_demo(whole, chain, cfg)
steps2 = pose.segment_steps(res2.demo)
print("\nD. Same video with --grips 3.0-6.5")
print(f"   steps found: {' > '.join(s['step'] for s in steps2)}")
