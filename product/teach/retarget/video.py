"""Extract a hand demonstration from a video, using ArUco markers.

Setup (one fixed camera):
- a "table" marker at a known pose in the robot base frame (measure it once),
- a "hand" marker on the back of the operator's hand (the sleeve's hand pod later).
Each frame gives T_cam_table and T_cam_hand; the hand pose in the robot frame is
    T_base_hand = T_base_table @ inv(T_cam_table) @ T_cam_hand
Grip comes from a grip-events CSV for now (t_close, t_open per row);
the sleeve's EMG/FSR replaces this later.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .demo import HandDemo
from .kinematics import make_transform, rpy_to_matrix


@dataclass
class CameraModel:
    matrix: np.ndarray  # 3x3
    dist: np.ndarray  # distortion coefficients

    @classmethod
    def from_json(cls, path: Path) -> "CameraModel":
        d = json.loads(Path(path).read_text())
        m = np.array([[d["fx"], 0, d["cx"]], [0, d["fy"], d["cy"]], [0, 0, 1]], dtype=float)
        return cls(m, np.array(d.get("dist", [0, 0, 0, 0, 0]), dtype=float))

    @classmethod
    def guess(cls, width: int, height: int, hfov_deg: float = 60.0) -> "CameraModel":
        """Rough model from the field of view. Calibrate the camera for real numbers."""
        f = width / 2 / np.tan(np.radians(hfov_deg) / 2)
        m = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=float)
        return cls(m, np.zeros(5))


@dataclass
class MarkerSetup:
    table_id: int
    table_size_m: float
    hand_id: int
    hand_size_m: float
    base_to_table: np.ndarray  # 4x4 table marker pose in the robot base frame
    dictionary: str = "DICT_4X4_50"

    @classmethod
    def from_json(cls, path: Path) -> "MarkerSetup":
        d = json.loads(Path(path).read_text())
        bt = d["base_to_table"]
        return cls(
            table_id=d["table_id"],
            table_size_m=d["table_size_m"],
            hand_id=d["hand_id"],
            hand_size_m=d["hand_size_m"],
            base_to_table=make_transform(rpy_to_matrix(*bt["rpy"]), np.array(bt["xyz"], dtype=float)),
            dictionary=d.get("dictionary", "DICT_4X4_50"),
        )


def marker_object_points(size: float) -> np.ndarray:
    """Corner order used by OpenCV ArUco and SOLVEPNP_IPPE_SQUARE."""
    h = size / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)


def marker_pose(corners: np.ndarray, size: float, cam: CameraModel) -> np.ndarray:
    import cv2

    ok, rvec, tvec = cv2.solvePnP(
        marker_object_points(size), corners.reshape(4, 2).astype(np.float32), cam.matrix, cam.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not ok:
        raise RuntimeError("solvePnP failed")
    r, _ = cv2.Rodrigues(rvec)
    return make_transform(r, tvec.ravel())


def detect_poses(frame: np.ndarray, setup: MarkerSetup, cam: CameraModel, detector) -> dict[int, np.ndarray]:
    """Camera-frame poses of the table and hand markers found in one frame."""
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    corners, ids, _ = detector.detectMarkers(gray)
    found: dict[int, np.ndarray] = {}
    if ids is None:
        return found
    sizes = {setup.table_id: setup.table_size_m, setup.hand_id: setup.hand_size_m}
    for c, i in zip(corners, ids.ravel()):
        if int(i) in sizes:
            found[int(i)] = marker_pose(c, sizes[int(i)], cam)
    return found


def make_detector(dictionary: str):
    import cv2

    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(d, params)


def read_grip_events(path: Path | None) -> list[tuple[float, float]]:
    if path is None:
        return []
    with open(path, newline="") as f:
        return [(float(r["t_close"]), float(r["t_open"])) for r in csv.DictReader(f)]


def grip_at(t: float, events: list[tuple[float, float]]) -> float:
    return 1.0 if any(c <= t < o for c, o in events) else 0.0


def video_to_demo(
    video: Path,
    setup: MarkerSetup,
    cam: CameraModel | None = None,
    grip_events: list[tuple[float, float]] | None = None,
    max_gap_s: float = 0.3,
) -> tuple[HandDemo, list[str]]:
    """Return the hand demo and a list of warnings (missed frames, gaps)."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam = cam or CameraModel.guess(width, height)
    detector = make_detector(setup.dictionary)
    events = grip_events or []

    ts, poses, grips, warnings = [], [], [], []
    cam_to_table: np.ndarray | None = None
    frame_idx = missed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps
        frame_idx += 1
        found = detect_poses(frame, setup, cam, detector)
        if setup.table_id in found:
            cam_to_table = found[setup.table_id]  # camera is fixed; keep the latest view
        if setup.hand_id not in found or cam_to_table is None:
            missed += 1
            continue
        base_to_hand = setup.base_to_table @ np.linalg.inv(cam_to_table) @ found[setup.hand_id]
        ts.append(t)
        poses.append(base_to_hand)
        grips.append(grip_at(t, events))
    cap.release()

    if not ts:
        raise RuntimeError("No frame had both the table marker and the hand marker visible")
    if missed:
        warnings.append(f"{missed} of {frame_idx} frames skipped (hand or table marker not seen)")
    gaps = np.diff(ts)
    for i in np.flatnonzero(gaps > max_gap_s):
        warnings.append(f"gap of {gaps[i]:.2f} s at t={ts[i]:.2f} s: the arm will move straight across it")
    return HandDemo(np.array(ts), np.array(poses), np.array(grips)), warnings
