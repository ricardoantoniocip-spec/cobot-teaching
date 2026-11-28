"""Track the operator's arm in a plain video (no markers on the body) and turn it into a hand demo.

    video -> MediaPipe pose + hand landmarks -> arm joints, grasp point, hand open or closed -> HandDemo (demo.csv)

Two ways to put the hand into the robot's frame:

  body mode (default, any video)
      Your shoulder is mapped onto the robot's shoulder and your arm is scaled down to the
      robot's reach. The lowest hand height in the video is taken as the table. Good for
      learning the tools and for training in the twin. Positions are NOT true millimetres.

  table-marker mode (--markers, for the real cell)
      A printed ArUco marker on the table at a measured place gives the camera position.
      The hand's distance from the camera comes from its size in the image. True scale,
      expect errors of a few centimetres with one camera.

The robot follows the grasp point (between the thumb tip and the fingertips), not the arm angles:
a human arm and a 5-joint SO-101 do not share joints. Shoulder, elbow and wrist are still
tracked and saved (arm_track.csv, tracked.mp4) so you can see and check the tracking.
"""

from __future__ import annotations

import csv
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .demo import HandDemo
from .kinematics import SerialChain, make_transform, tool_down

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_URLS = {
    "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "pose_landmarker_heavy.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
}

# MediaPipe pose landmark numbers (the person's own left and right).
POSE = {
    "left": {"shoulder": 11, "elbow": 13, "wrist": 15},
    "right": {"shoulder": 12, "elbow": 14, "wrist": 16},
}
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 11, 12, 23, 24
MIN_VISIBILITY = 0.5  # how sure the detector must be to call a body point "seen" (drawn in white)
# The robot follows the wrist and hand. The shoulder only matters as a fixed reference, and it is
# the point a laptop camera loses first (edge of the picture, hidden behind the arm, a dark top).
# Requiring shoulder, elbow AND wrist at 0.5 threw away every picture where the shoulder was unsure,
# hand included, so the robot never moved. Now: a picture counts when the wrist is seen; the shoulder
# is needed only during calibration, and less surely (MediaPipe still estimates its position well).
WRIST_MIN_VIS = 0.4
SHOULDER_MIN_VIS = 0.3  # during calibration only
# MediaPipe hand landmark numbers.
H_WRIST, H_THUMB_TIP, H_INDEX_MCP, H_INDEX_TIP, H_MIDDLE_MCP, H_PINKY_MCP = 0, 4, 5, 8, 9, 17
HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]  # fmt: skip


@dataclass
class ArmFrame:
    """What the detectors saw in one video frame. Camera axes: x right, y down, z away from the camera."""

    t: float
    pose_px: np.ndarray | None = None  # (33, 2) pixels
    pose_world: np.ndarray | None = None  # (33, 3) metres, origin between the hips
    pose_vis: np.ndarray | None = None  # (33,) 0..1 how sure the detector is
    hand_px: np.ndarray | None = None  # (21, 2) pixels
    hand_world: np.ndarray | None = None  # (21, 3) metres, origin in the hand
    cam_to_table: np.ndarray | None = None  # 4x4, only in table-marker mode


@dataclass
class PoseConfig:
    arm: str = "right"  # which of YOUR arms does the task
    close_ratio: float = 0.78  # whole-hand openness below this = hand closed = holding the part
    open_ratio: float = 0.84  # ... above this = hand open = released (the gap in between avoids flicker)
    min_hold_s: float = 0.4  # a closed hand shorter than this is noise, not a grasp
    clear_below: float = 0.70  # a grasp only counts if the hand is clearly closed: typical openness while closed under this
    smooth_s: float = 0.2  # moving average on the hand path
    max_gap_s: float = 0.5  # longer gaps without a hand are reported
    out_hz: float = 30.0
    # Manual overrides, in seconds of the video (the time is printed on tracked.mp4)
    grip_events: list[tuple[float, float]] | None = None  # [(t_close, t_open), ...] replaces the automatic grip
    t_start: float | None = None  # use only this part of the video
    t_end: float | None = None
    # body mode
    robot_reach_m: float = 0.33  # your fully stretched arm maps to this distance from the robot's shoulder
    max_radius_m: float = 0.27  # the hand path is shrunk to stay within this distance of the robot base
    table_z_m: float = 0.008  # tool height when your hand is at its lowest (fingers just above the table)
    # table-marker mode
    hand_width_mm: float | None = None  # your knuckle width, index to little finger. Sets the depth scale.


@dataclass
class PoseResult:
    demo: HandDemo
    arm_rows: list[dict] = field(default_factory=list)  # one row per tracked frame, for arm_track.csv
    warnings: list[str] = field(default_factory=list)
    mode: str = "body"
    scale: float = 1.0  # robot metres per human metre (body mode)
    video_t0: float = 0.0  # video time (s) of the demo's t = 0: demo and step times start at the first tracked frame
    tracked: tuple | None = None  # (frames, camera) from track_video, so the grip can be redone without tracking again


# ---------------------------------------------------------------- detectors


def ensure_models(names: tuple[str, ...] = ("pose_landmarker_full.task", "hand_landmarker.task")) -> dict[str, Path]:
    """Download the MediaPipe model files once (not stored in git). Only the ones asked for."""
    paths = {}
    for name in names:
        url = MODEL_URLS[name]
        dest = MODEL_DIR / name
        if not dest.exists():
            print(f"Downloading {name}")
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=120) as r:
                dest.write_bytes(r.read())
        paths[name] = dest
    return paths


def track_video(video: Path, arm: str = "right", markers=None, camera=None, overlay: Path | None = None, progress: bool = True):
    """Run pose and hand detection on every frame. Returns (frames, camera model, fps).

    markers: a video.MarkerSetup to also find the table marker. overlay: write an mp4 with the
    tracking drawn on top, so you can see what was found.
    """
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import vision

    from .video import CameraModel, detect_poses, make_detector

    models = ensure_models()
    base = mp.tasks.BaseOptions
    pose_det = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(base_options=base(model_asset_path=str(models["pose_landmarker_full.task"])), running_mode=vision.RunningMode.VIDEO)
    )
    hand_det = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(base_options=base(model_asset_path=str(models["hand_landmarker.task"])), running_mode=vision.RunningMode.VIDEO, num_hands=2)
    )
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    aruco = make_detector(markers.dictionary) if markers is not None else None
    writer = None
    frames: list[ArmFrame] = []
    cam = camera
    cam_to_table = None
    idx = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        h, w = img.shape[:2]
        if cam is None:
            cam = CameraModel.guess(w, h)
        f = ArmFrame(t=idx / fps)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        ts_ms = int(round(idx * 1000 / fps))
        pr = pose_det.detect_for_video(mp_img, ts_ms)
        if pr.pose_landmarks and arm_seen(np.array([p.visibility for p in pr.pose_landmarks[0]]), arm):
            f.pose_px = np.array([[p.x * w, p.y * h] for p in pr.pose_landmarks[0]])
            f.pose_vis = np.array([p.visibility for p in pr.pose_landmarks[0]])
            f.pose_world = np.array([[p.x, p.y, p.z] for p in pr.pose_world_landmarks[0]])
        hr = hand_det.detect_for_video(mp_img, ts_ms)
        if hr.hand_landmarks:
            hands_px = [np.array([[p.x * w, p.y * h] for p in hand]) for hand in hr.hand_landmarks]
            k = _pick_hand(hands_px, f, arm)
            if k is not None:
                f.hand_px = hands_px[k]
                f.hand_world = np.array([[p.x, p.y, p.z] for p in hr.hand_world_landmarks[k]])
        if aruco is not None:
            found = detect_poses(img, markers, cam, aruco)
            if markers.table_id in found:
                cam_to_table = found[markers.table_id]  # the camera is fixed: keep the latest view
            f.cam_to_table = cam_to_table
        frames.append(f)
        if overlay is not None:
            if writer is None:
                writer = cv2.VideoWriter(str(overlay), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(draw_overlay(img, f, arm))
        idx += 1
        if progress and idx % 100 == 0:
            print(f"  frame {idx}/{total}")
    cap.release()
    if writer is not None:
        writer.release()
    pose_det.close()
    hand_det.close()
    if not frames:
        raise RuntimeError(f"{video} has no frames")
    return frames, cam, fps


def _pick_hand(hands_px: list[np.ndarray], f: ArmFrame, arm: str) -> int | None:
    """The hand whose wrist is closest to the tracked arm's wrist. (Left/right labels from the
    hand model flip with mirrored videos, so distance is safer.)"""
    if f.pose_px is None:
        return 0 if len(hands_px) == 1 else None
    j = POSE[arm]
    wrist, elbow = f.pose_px[j["wrist"]], f.pose_px[j["elbow"]]
    forearm = max(np.linalg.norm(wrist - elbow), 1.0)
    d = [np.linalg.norm(hp[H_WRIST] - wrist) for hp in hands_px]
    k = int(np.argmin(d))
    return k if d[k] < 0.6 * forearm else None


def draw_overlay(img: np.ndarray, f: ArmFrame, arm: str) -> np.ndarray:
    import cv2

    out = img.copy()
    s = max(1, round(img.shape[1] / 480))
    cv2.putText(out, f"t = {f.t:.1f} s", (8 * s, 20 * s), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, (0, 255, 255), 2 * s)
    if f.pose_px is not None:
        j = POSE[arm]
        pts = [tuple(int(v) for v in f.pose_px[j[n]]) for n in ("shoulder", "elbow", "wrist")]
        cv2.line(out, pts[0], pts[1], (0, 200, 255), 2 * s)
        cv2.line(out, pts[1], pts[2], (0, 200, 255), 2 * s)
        for p in pts:
            cv2.circle(out, p, 4 * s, (0, 120, 255), -1)
        ang = elbow_angle_deg(f.pose_world[j["shoulder"]], f.pose_world[j["elbow"]], f.pose_world[j["wrist"]])
        cv2.putText(out, f"elbow {ang:.0f} deg", (pts[1][0] + 6 * s, pts[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, (0, 200, 255), s)
    if f.hand_px is not None:
        for a, b in HAND_BONES:
            cv2.line(out, tuple(int(v) for v in f.hand_px[a]), tuple(int(v) for v in f.hand_px[b]), (80, 220, 80), s)
        openness = hand_openness(f.hand_world)
        closed, half = openness < 0.70, openness < 0.78  # same numbers as PoseConfig.clear_below and close_ratio
        word = "CLOSED" if closed else "half closed" if half else "open"
        dot = tuple(int(v) for v in grasp_point(f.hand_px))
        cv2.circle(out, dot, 5 * s, (0, 0, 255) if closed else (255, 255, 255), -1)
        cv2.putText(out, f"hand {word} {openness:.2f}", (dot[0] + 6 * s, dot[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, (0, 0, 255) if closed else (255, 255, 255), s)
    return out


# ---------------------------------------------------------------- geometry (no detector needed, unit tested)


def arm_seen(vis: np.ndarray, arm: str) -> bool:
    """Is the working arm usable in this picture? Only the wrist has to be seen (see WRIST_MIN_VIS)."""
    return bool(vis[POSE[arm]["wrist"]] >= WRIST_MIN_VIS)


def hand_frame(hand_world: np.ndarray) -> np.ndarray:
    """3x3, columns: the hand's forward (wrist to middle knuckle), across the knuckles, and the palm normal.

    Same axes as hand_world (camera axes). Right-handed for a left or a right hand alike, so a change
    of this frame between two pictures is a real rotation of the hand.
    """
    h = hand_world
    fwd = h[H_MIDDLE_MCP] - h[H_WRIST]
    fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
    across = h[H_INDEX_MCP] - h[H_PINKY_MCP]
    across = across - np.dot(across, fwd) * fwd
    across = across / (np.linalg.norm(across) + 1e-12)
    return np.column_stack([fwd, across, np.cross(fwd, across)])


def wrist_angles(hand_now: np.ndarray, hand_ref: np.ndarray, axes: np.ndarray) -> tuple[float, float]:
    """How much the hand has tilted and rolled since a reference picture, radians.

    pitch: how much further the fingers point down (positive) or up (negative), measured against
           your own up direction, so it is the hand's direction in space, whatever the elbow does.
    roll:  turn of the palm about the hand's own forward axis (twist part of the rotation), positive
           by the right-hand rule about forward. Like turning a key.
    axes:  body_axes(), to measure pitch against your up.
    """
    a, b = hand_frame(hand_ref), hand_frame(hand_now)
    up = axes[:, 2]
    pitch = float(np.arcsin(np.clip(-b[:, 0] @ up, -1, 1)) - np.arcsin(np.clip(-a[:, 0] @ up, -1, 1)))
    r = a.T @ b  # rotation from the reference hand to the hand now, in the reference hand's axes
    roll = float(2 * np.arctan2(r[2, 1] - r[1, 2], 1 + np.trace(r)))  # swing-twist: the twist about forward
    roll = float((roll + np.pi) % (2 * np.pi) - np.pi)
    return pitch, roll


def elbow_angle_deg(shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> float:
    """Angle between upper arm and forearm: 180 = straight arm, 90 = right angle."""
    a, b = shoulder - elbow, wrist - elbow
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


FINGERS = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))  # knuckle, two joints, tip: index to little finger


def hand_openness(hand_world: np.ndarray) -> float:
    """How open the whole hand is, 0 to 1, from the four fingers (not the thumb).

    For each finger: distance from the wrist to the fingertip, divided by the length of the finger
    laid out straight. A flat open hand gives about 0.9, a fist about 0.45 (measured on photos).
    The hand is "closed" when the fingers curl around a part, whichever fingers do the gripping.
    """
    h = hand_world
    vals = []
    for m, p, d, t in FINGERS:
        straight = np.linalg.norm(h[m] - h[H_WRIST]) + np.linalg.norm(h[p] - h[m]) + np.linalg.norm(h[d] - h[p]) + np.linalg.norm(h[t] - h[d])
        vals.append(np.linalg.norm(h[t] - h[H_WRIST]) / (straight + 1e-9))
    return float(np.mean(vals))


def grasp_point(hand: np.ndarray) -> np.ndarray:
    """The point the robot's gripper follows: halfway between the thumb tip and the index and middle fingertips."""
    return (hand[H_THUMB_TIP] + (hand[H_INDEX_TIP] + hand[12]) / 2) / 2


def grip_states(openness: np.ndarray, close_below: float, open_above: float, min_hold: int = 1) -> np.ndarray:
    """0/1 per sample: 1 while the hand is closed. Two thresholds so it does not flicker, and a
    closed stretch shorter than min_hold samples is dropped."""
    out, closed = np.zeros(len(openness)), False
    for i, r in enumerate(openness):
        if closed and r > open_above:
            closed = False
        elif not closed and r < close_below:
            closed = True
        out[i] = float(closed)
    edges = np.flatnonzero(np.diff(np.concatenate([[0.0], out, [0.0]])))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a < min_hold:
            out[a:b] = 0.0
    return out


def body_axes(frames: list[ArmFrame]) -> np.ndarray:
    """3x3 matrix whose columns are the operator's forward, left and up directions in camera axes.

    Left comes from the shoulder line, up from hips to shoulders (or the image's up when the hips
    are hidden). Medians over the whole video, so one fixed frame: the robot base does not move either.
    """
    world = np.array([f.pose_world for f in frames if f.pose_world is not None])
    vis = np.array([f.pose_vis for f in frames if f.pose_world is not None])
    med = np.median(world, axis=0)
    left = med[L_SHOULDER] - med[R_SHOULDER]
    if np.median(vis[:, [L_HIP, R_HIP]]) > 0.5:
        up = (med[L_SHOULDER] + med[R_SHOULDER]) / 2 - (med[L_HIP] + med[R_HIP]) / 2
    else:
        up = np.array([0.0, -1.0, 0.0])
    up = up / np.linalg.norm(up)
    left = left - np.dot(left, up) * up
    left = left / np.linalg.norm(left)
    return np.column_stack([np.cross(left, up), left, up])


def hand_in_camera(hand_world: np.ndarray, hand_px: np.ndarray, cam, scale: float = 1.0) -> np.ndarray:
    """Pose of the hand model in the camera (4x4). The hand's size in the image gives its distance,
    the same way a marker of known size does."""
    import cv2

    ok, rvec, tvec = cv2.solvePnP((hand_world * scale).astype(np.float64), hand_px.astype(np.float64), cam.matrix, cam.dist, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        raise RuntimeError("solvePnP failed")
    r, _ = cv2.Rodrigues(rvec)
    return make_transform(r, tvec.ravel())


def _interp_gaps(t: np.ndarray, x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(t, t[valid], x[valid, i]) for i in range(x.shape[1])])


def _moving_average(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1 or len(x) < n:
        return x.copy()
    pad = n // 2
    p = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    return np.column_stack([np.convolve(p[:, i], np.ones(n) / n, mode="valid")[: len(x)] for i in range(x.shape[1])])


def frames_to_demo(frames: list[ArmFrame], chain: SerialChain, cfg: PoseConfig | None = None, markers=None, cam=None) -> PoseResult:
    """Tracked frames -> hand demo in the robot base frame. markers + cam switch on table-marker mode."""
    cfg = cfg or PoseConfig()
    j = POSE[cfg.arm]
    warnings: list[str] = []
    frames = [f for f in frames if (cfg.t_start is None or f.t >= cfg.t_start) and (cfg.t_end is None or f.t <= cfg.t_end)]
    n = len(frames)
    t = np.array([f.t for f in frames])
    has_pose = np.array([f.pose_world is not None for f in frames])
    has_hand = np.array([f.hand_world is not None and f.pose_world is not None for f in frames])
    metric = markers is not None
    if has_pose.sum() < 5:
        raise RuntimeError("No person found in the video. Film so that both shoulders and the working arm are visible.")
    if has_hand.sum() < 5:
        raise RuntimeError("No hand found near the tracked arm. Check --arm left/right and that the hand is in view.")

    # --- arm joints relative to the shoulder, in the operator's own axes (forward, left, up)
    axes = body_axes(frames)
    med_shoulder = np.median([f.pose_world[j["shoulder"]] for f in frames if f.pose_world is not None], axis=0)
    elbow_b, wrist_b, pinch_b = (np.full((n, 3), np.nan) for _ in range(3))
    ratios = np.full(n, np.nan)
    elbow_deg = np.full(n, np.nan)
    for i, f in enumerate(frames):
        if f.pose_world is None:
            continue
        elbow_b[i] = (f.pose_world[j["elbow"]] - med_shoulder) @ axes
        wrist_b[i] = (f.pose_world[j["wrist"]] - med_shoulder) @ axes
        elbow_deg[i] = elbow_angle_deg(f.pose_world[j["shoulder"]], f.pose_world[j["elbow"]], f.pose_world[j["wrist"]])
        if f.hand_world is not None:
            hw = f.hand_world
            pinch_b[i] = wrist_b[i] + (grasp_point(hw) - hw[H_WRIST]) @ axes
            ratios[i] = hand_openness(hw)

    # --- pinch point in the robot base frame
    scale = 1.0
    if metric:
        hand_scale = 1.0
        if cfg.hand_width_mm:
            model_w = np.median([np.linalg.norm(f.hand_world[H_INDEX_MCP] - f.hand_world[H_PINKY_MCP]) for f in frames if f.hand_world is not None])
            hand_scale = cfg.hand_width_mm / 1000 / model_w
        else:
            warnings.append("no --hand-width-mm given: using the model's average hand size, so distances can be 10-15% off")
        pos = np.full((n, 3), np.nan)
        for i, f in enumerate(frames):
            if f.hand_world is None or f.cam_to_table is None:
                continue
            cam_hand = hand_in_camera(f.hand_world, f.hand_px, cam, hand_scale)
            pinch_cam = cam_hand @ np.append(grasp_point(f.hand_world) * hand_scale, 1.0)
            pos[i] = (markers.base_to_table @ np.linalg.inv(f.cam_to_table) @ pinch_cam)[:3]
        if np.isfinite(pos[:, 0]).sum() < 5:
            raise RuntimeError("The table marker was not seen together with the hand. Check the marker id, size and lighting.")
    else:
        upper = np.nanmedian(np.linalg.norm(elbow_b, axis=1))
        fore = np.nanmedian(np.linalg.norm(wrist_b - elbow_b, axis=1))
        hand_len = np.nanmedian(np.linalg.norm(pinch_b - wrist_b, axis=1))
        scale = cfg.robot_reach_m / (upper + fore + hand_len)
        shoulder_robot = chain._fk_frames(np.zeros(len(chain.active)))[2][1]  # position of the shoulder_lift joint
        for _ in range(6):  # shrink further if the path would leave the area the arm reaches with its gripper pointing down
            pos = shoulder_robot + scale * pinch_b
            far = np.nanmax(np.linalg.norm(pos[:, :2], axis=1))
            if far <= cfg.max_radius_m:
                break
            scale *= 0.98 * cfg.max_radius_m / far
        pos[:, 2] = scale * (pinch_b[:, 2] - np.nanpercentile(pinch_b[:, 2], 2)) + cfg.table_z_m

    valid = np.isfinite(pos[:, 0])
    first, last = np.flatnonzero(valid)[[0, -1]]
    if valid.sum() < n:
        warnings.append(f"hand not tracked in {n - valid.sum()} of {n} frames (filled in from neighbours)")
    tv = t[valid]
    for k in np.flatnonzero(np.diff(tv) > cfg.max_gap_s):
        warnings.append(f"hand lost for {tv[k + 1] - tv[k]:.1f} s at t={tv[k]:.1f} s: the path is a straight line across it")
    pos = _interp_gaps(t, pos, valid)
    dt = float(np.median(np.diff(t))) if n > 1 else 1 / 30
    pos = _moving_average(pos, max(1, int(round(cfg.smooth_s / dt)) | 1))

    # --- grip: fill gaps with the last seen value, median filter, two thresholds
    r = ratios.copy()
    seen = np.isfinite(r)
    r = r[np.maximum.accumulate(np.where(seen, np.arange(n), np.flatnonzero(seen)[0]))]
    r_f = np.array([np.median(r[max(0, i - 2) : i + 3]) for i in range(n)])
    if cfg.grip_events:
        grip = np.array([float(any(c <= ti < o for c, o in cfg.grip_events)) for ti in t])
    else:
        grip = grip_states(r_f, cfg.close_ratio, cfg.open_ratio, max(1, int(round(cfg.min_hold_s / dt))))
        # A half-closed hand is not a grasp. Fingers around a wide part stay almost straight and hover
        # at the threshold, which gave grasps and releases in mid-air. Count only clearly closed hands.
        edges = np.flatnonzero(np.diff(np.concatenate([[0.0], grip, [0.0]])))
        for a, b in zip(edges[::2], edges[1::2]):
            typical = float(np.median(r_f[a:b]))
            if typical > cfg.clear_below:
                grip[a:b] = 0.0
                warnings.append(f"hand half closed from {t[a]:.1f} to {t[b - 1]:.1f} s (openness {typical:.2f}, a clear grasp is under {cfg.clear_below}): not counted as a grasp")

    # --- cut to the tracked part, resample at a fixed rate, tool pointing down
    t_out = np.arange(t[first], t[last] + 1e-9, 1 / cfg.out_hz)
    p_out = np.column_stack([np.interp(t_out, t, pos[:, i]) for i in range(3)])
    g_out = grip[np.clip(np.searchsorted(t, t_out, side="right") - 1, 0, n - 1)]
    poses = np.array([make_transform(tool_down(chain, np.arctan2(p[1], p[0])), p) for p in p_out])
    demo = HandDemo(t_out - t_out[0], poses, g_out)
    if not grip.any():
        warnings.append(f"no grasp found from the hand: it never clearly closed (lowest openness {np.nanmin(ratios):.2f}; flat hand = 0.9, fist = 0.45). A wide part keeps the fingers almost straight.")

    rows = []
    for i in range(n):
        if not np.isfinite(elbow_b[i, 0]):
            continue
        row = {"t": round(float(t[i]), 4), "elbow_angle_deg": round(float(elbow_deg[i]), 1)}
        for name, arr in (("elbow", elbow_b), ("wrist", wrist_b), ("grasp", pinch_b)):
            for a, v in zip("xyz", arr[i]):
                row[f"{name}_{a}"] = round(float(v), 4) if np.isfinite(v) else ""
        row["hand_openness"] = round(float(ratios[i]), 3) if np.isfinite(ratios[i]) else ""
        row["grip"] = int(grip[i])
        row.update({f"robot_{a}": round(float(v), 4) for a, v in zip("xyz", pos[i])})
        rows.append(row)
    return PoseResult(demo=demo, arm_rows=rows, warnings=warnings, mode="table-marker" if metric else "body", scale=float(scale),
                      video_t0=float(t_out[0]))


# ---------------------------------------------------------------- steps (what the operator did, in order)


def segment_steps(demo: HandDemo) -> list[dict]:
    """Cut the demo into steps at every grip change: reach, grasp, move, release, retreat.

    This list is the hand-over point for later work: a vision model can add a label to each step
    ("pick the bracket"), and a program generator can write one robot instruction per step.
    """
    g = demo.grip >= 0.5
    cuts = [0, *(np.flatnonzero(np.diff(g.astype(int)) != 0) + 1).tolist(), len(demo) - 1]
    steps = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b <= a:
            continue
        holding = bool(g[a])
        name = "move" if holding else ("retreat" if a > 0 and b == len(demo) - 1 else "reach")
        steps.append({"step": name, "t_start": round(float(demo.t[a]), 3), "t_end": round(float(demo.t[b]), 3),
                      "from_xyz": np.round(demo.poses[a][:3, 3], 4).tolist(), "to_xyz": np.round(demo.poses[b][:3, 3], 4).tolist(),
                      "holding": holding, "label": None})  # fmt: skip
        if b < len(demo) - 1:
            steps.append({"step": "release" if holding else "grasp", "t_start": round(float(demo.t[b]), 3), "t_end": round(float(demo.t[b]), 3),
                          "from_xyz": np.round(demo.poses[b][:3, 3], 4).tolist(), "to_xyz": np.round(demo.poses[b][:3, 3], 4).tolist(),
                          "holding": not holding, "label": None})  # fmt: skip
    return steps


def pick_and_place_points(steps: list[dict]) -> tuple[np.ndarray, np.ndarray] | None:
    """Tool position at the first grasp and at the release after it, or None if there is no full grasp."""
    grasp = next((s for s in steps if s["step"] == "grasp"), None)
    if grasp is None:
        return None
    release = next((s for s in steps if s["step"] == "release" and s["t_start"] > grasp["t_start"]), None)
    if release is None:
        return None
    return np.array(grasp["to_xyz"]), np.array(release["to_xyz"])


# ---------------------------------------------------------------- one call for the command line


def video_to_demo(video: Path, out_dir: Path, chain: SerialChain, cfg: PoseConfig | None = None, markers=None, camera=None, overlay: bool = True,
                  tracked: tuple | None = None) -> tuple[PoseResult, list[dict]]:  # fmt: skip
    """Track a video and write demo.csv, arm_track.csv, steps.json and tracked.mp4 into out_dir.
    Pass `tracked` (from an earlier result) to redo the steps with another grip without tracking again."""
    from .demo import write_demo

    cfg = cfg or PoseConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    if tracked is None:
        frames, cam, _ = track_video(video, cfg.arm, markers, camera, out_dir / "tracked.mp4" if overlay else None)
    else:
        frames, cam = tracked
    res = frames_to_demo(frames, chain, cfg, markers, cam)
    res.tracked = (frames, cam)
    steps = segment_steps(res.demo)
    write_demo(out_dir / "demo.csv", res.demo)
    with open(out_dir / "arm_track.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res.arm_rows[0].keys()))
        w.writeheader()
        w.writerows(res.arm_rows)
    (out_dir / "steps.json").write_text(json.dumps({"video": video.name, "mode": res.mode, "arm": cfg.arm,
                                                      "video_t0": round(res.video_t0, 3), "steps": steps}, indent=2))
    return res, steps
