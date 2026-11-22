"""Where an object stands on the table, in true millimetres, from its box in the picture.

One printed ArUco marker lies flat on the table at a measured place (markers.json, the same file
as table-marker mode). The marker tells us where the camera is. A pixel is then a line of sight,
and where that line meets the table top is a point on the table:

    box bottom centre  -> the object's near edge on the table
    box bottom corners -> its width
    box top centre     -> its height (taking the footprint as about square)

The object's centre is half a width behind the near edge, seen from the camera. All of this is
for an object standing on the marker's plane. One camera, so treat the result as "within a
centimetre or two" until it is measured on a real cell; height and width are rough.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OnTable:
    centre_base_m: np.ndarray  # (3,) object centre on the table top, robot base frame
    width_m: float
    height_m: float
    near_edge_base_m: np.ndarray  # (3,) the point under the box's bottom centre


def _ray(px, cam) -> np.ndarray:
    """Line of sight of a pixel in the camera frame (z forward), lens distortion removed."""
    import cv2

    p = cv2.undistortPoints(np.array([[px]], dtype=np.float64), cam.matrix, cam.dist).ravel()
    return np.array([p[0], p[1], 1.0])


def pixel_on_plane(px, cam, cam_to_table: np.ndarray, height_m: float = 0.0) -> np.ndarray:
    """The point where a pixel's line of sight meets the plane `height_m` above the table marker. Marker frame, metres."""
    t = np.linalg.inv(cam_to_table)  # camera pose in the marker frame
    o, d = t[:3, 3], t[:3, :3] @ _ray(px, cam)
    if abs(d[2]) < 1e-9 or (height_m - o[2]) / d[2] <= 0:
        raise ValueError("this pixel does not look at the table (the camera must look down at it)")
    return o + d * (height_m - o[2]) / d[2]


def box_on_table(box, cam, cam_to_table: np.ndarray, base_to_table: np.ndarray) -> OnTable:
    """box = (x0, y0, x1, y1) pixels of an object standing on the table."""
    x0, y0, x1, y1 = box
    near = pixel_on_plane(((x0 + x1) / 2, y1), cam, cam_to_table)
    t = np.linalg.inv(cam_to_table)
    cam_pos = t[:3, 3]
    away = near[:2] - cam_pos[:2]
    dist = float(np.linalg.norm(away))
    away = away / dist
    d_top = t[:3, :3] @ _ray(((x0 + x1) / 2, y0), cam)
    along = float(d_top[:2] @ away)
    width = float(np.linalg.norm(pixel_on_plane((x1, y1), cam, cam_to_table) - pixel_on_plane((x0, y1), cam, cam_to_table)))
    height = 0.0
    for _ in range(3):
        # Height: the top centre of the box is the object's far top edge, one width behind the near edge.
        height = max(float(cam_pos[2] + d_top[2] * (dist + width) / along), 0.0) if along > 1e-9 else 0.0
        # Width: the box is widest at the object's top, which is nearer the camera than the table is.
        # Measure the box's width on the plane at the object's top, at the middle of the box.
        h = min(height, cam_pos[2] * 0.8)
        ym = (y0 + y1) / 2
        width = float(np.linalg.norm(pixel_on_plane((x1, ym), cam, cam_to_table, h) - pixel_on_plane((x0, ym), cam, cam_to_table, h)))
    centre = np.array([*(near[:2] + away * width / 2), 0.0])

    def to_base(p):
        return (base_to_table @ np.append(p, 1.0))[:3]

    return OnTable(to_base(centre), width, max(height, 0.0), to_base(near))


def table_from_frames(frames) -> np.ndarray | None:
    """The camera-to-marker pose from tracked frames (pose.track_video with markers). The camera is fixed: the median view."""
    seen = [f.cam_to_table for f in frames if getattr(f, "cam_to_table", None) is not None]
    if not seen:
        return None
    mid = np.median(np.array([s[:3, 3] for s in seen]), axis=0)
    return min(seen, key=lambda s: float(np.linalg.norm(s[:3, 3] - mid)))
