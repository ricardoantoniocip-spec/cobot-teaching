"""Exercise 1: what MediaPipe sees in ONE frame of your video.

Run (from the repo root, venv active):
    python learning\\2026-11-14-vision-front-end\\ex1_landmarks.py C:\\path\\to\\take1.mp4 --t 4.0

Prints the arm landmarks in pixels and in metres, how sure the model is, which hand it
picks for your working arm, and the grip ratio. Writes ex1_frame.jpg in the current folder.
Uses the same model files and the same rules as product/teach/retarget/pose.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "product" / "teach"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from retarget import pose  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("--t", type=float, default=4.0, help="video time in seconds")
    p.add_argument("--arm", default="right", choices=["right", "left"])
    p.add_argument("--out", type=Path, default=Path("ex1_frame.jpg"))
    a = p.parse_args()

    import mediapipe as mp
    from mediapipe.tasks.python import vision

    cap = cv2.VideoCapture(str(a.video))
    cap.set(cv2.CAP_PROP_POS_MSEC, a.t * 1000)
    ok, img = cap.read()
    cap.release()
    if not ok:
        print(f"Could not read a frame at {a.t} s from {a.video}")
        return 1
    h, w = img.shape[:2]
    print(f"frame at {a.t:.1f} s, {w} x {h} pixels")

    models = pose.ensure_models()
    base = mp.tasks.BaseOptions
    pose_det = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(base_options=base(model_asset_path=str(models["pose_landmarker_full.task"]))))
    hand_det = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(base_options=base(model_asset_path=str(models["hand_landmarker.task"])), num_hands=2))
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    pr = pose_det.detect(mp_img)
    hr = hand_det.detect(mp_img)

    f = pose.ArmFrame(t=a.t)
    j = pose.POSE[a.arm]
    print("\n-- body (pose model, 33 points; we use 3) --")
    if not pr.pose_landmarks:
        print("no person found")
    else:
        lm, wl = pr.pose_landmarks[0], pr.pose_world_landmarks[0]
        for name, k in j.items():
            print(f"{name:8} #{k:2}  image ({lm[k].x:.3f}, {lm[k].y:.3f}) = ({lm[k].x * w:4.0f}, {lm[k].y * h:4.0f}) px"
                  f"   world ({wl[k].x:+.3f}, {wl[k].y:+.3f}, {wl[k].z:+.3f}) m   visibility {lm[k].visibility:.2f}")
        vis = min(lm[k].visibility for k in j.values())
        print(f"lowest visibility {vis:.2f} -> {'USED' if vis >= pose.MIN_VISIBILITY else 'DROPPED'} (pose.py needs {pose.MIN_VISIBILITY})")
        if vis >= pose.MIN_VISIBILITY:
            f.pose_px = np.array([[p_.x * w, p_.y * h] for p_ in lm])
            f.pose_world = np.array([[p_.x, p_.y, p_.z] for p_ in wl])
            ang = pose.elbow_angle_deg(f.pose_world[j["shoulder"]], f.pose_world[j["elbow"]], f.pose_world[j["wrist"]])
            print(f"elbow angle {ang:.0f} deg (180 = straight arm)")

    print("\n-- hands (hand model, 21 points each) --")
    hands_px = [np.array([[p_.x * w, p_.y * h] for p_ in hd]) for hd in hr.hand_landmarks]
    if not hands_px:
        print("no hand found")
    for k, hp in enumerate(hands_px):
        label = hr.handedness[k][0].category_name
        msg = f"hand {k}: model calls it '{label}', wrist at ({hp[0][0]:.0f}, {hp[0][1]:.0f}) px"
        if f.pose_px is not None:
            wrist, elbow = f.pose_px[j["wrist"]], f.pose_px[j["elbow"]]
            forearm = max(np.linalg.norm(wrist - elbow), 1.0)
            d = np.linalg.norm(hp[0] - wrist)
            msg += f", {d:.0f} px from your {a.arm} wrist (limit {0.6 * forearm:.0f} px = 0.6 x forearm)"
        print(msg)
    k = pose._pick_hand(hands_px, f, a.arm) if hands_px else None
    if k is None:
        print(f"-> no hand accepted for the {a.arm} arm in this frame")
    else:
        hw = np.array([[p_.x, p_.y, p_.z] for p_ in hr.hand_world_landmarks[k]])
        hp = hands_px[k]
        ratio = pose.grip_ratio(hw)
        pinch = (hp[pose.H_THUMB_TIP] + hp[pose.H_INDEX_TIP]) / 2
        gap_mm = np.linalg.norm(hw[pose.H_THUMB_TIP] - hw[pose.H_INDEX_TIP]) * 1000
        palm_mm = np.linalg.norm(hw[pose.H_INDEX_MCP] - hw[pose.H_WRIST]) * 1000
        print(f"-> hand {k} accepted. thumb-index gap {gap_mm:.0f} mm, palm {palm_mm:.0f} mm, grip ratio {ratio:.2f}")
        print(f"   pinch point ({pinch[0]:.0f}, {pinch[1]:.0f}) px. Below 0.5 closes the grip, above 0.7 opens it: "
              f"{'closed' if ratio < 0.5 else 'open' if ratio > 0.7 else 'in between (keeps the last state)'}")
        f.hand_px, f.hand_world = hp, hw

    cv2.imwrite(str(a.out), pose.draw_overlay(img, f, a.arm))
    print(f"\nwrote {a.out.resolve()}")
    pose_det.close()
    hand_det.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
