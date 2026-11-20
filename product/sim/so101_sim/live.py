"""Live view: your arm leads, the SO-101 follows, and you see everything the vision finds.

    python -m so101_sim live                      webcam -> body tracking -> robot in the twin
    python -m so101_sim live --objects            also find things on the table (uses the graphics card if torch can)
    python -m so101_sim live --objects "juice carton"
    python -m so101_sim live --port COM5          also drive the real SO-101 (starts paused, SPACE lets it follow)
    python -m so101_sim live --video take.mp4     a recorded video instead of the webcam

One window: the camera picture with the body, both hands and object boxes drawn on it, the
robot in the twin next to it, and a panel with the numbers (hand openness, gripper, tool point,
joint angles, IK miss, timings, which device runs what).

Keys: q quit, c calibrate again, SPACE pause or continue the robot, o objects on or off,
r put the block back, s save a picture.

Body tracking (MediaPipe) runs on every picture on the CPU. The object detector is much slower,
so it runs beside it in its own thread and the boxes are as old as the "age" shown on them.
The real arm is UNTESTED: first run with a hand on the power switch.
"""

from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import scene, scripted  # sets up the import path for retarget
from retarget import pose as rp  # noqa: E402
from retarget.live import CALIBRATING, FOLLOWING, HOLD, LiveConfig, LiveFollower, LiveState  # noqa: E402

BODY_BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
              (0, 11), (0, 12)]  # fmt: skip
CAM_W, SIDE_W, TWIN_H, CANVAS_H = 800, 480, 360, 600
MODE_COLOUR = {"waiting": (0, 200, 255), CALIBRATING: (0, 200, 255), FOLLOWING: (80, 220, 80), HOLD: (0, 0, 255)}


@dataclass
class Seen:
    """Everything the body tracker found in one picture."""

    arm: rp.ArmFrame
    body_px: np.ndarray | None = None  # (33, 2), whole body, also when the working arm is hidden
    body_vis: np.ndarray | None = None
    hands_px: list[np.ndarray] = field(default_factory=list)
    took_ms: float = 0.0


class BodyTracker:
    """MediaPipe pose + hands, one picture at a time."""

    def __init__(self, arm: str = "right"):
        import mediapipe as mp
        from mediapipe.tasks.python import vision

        models = rp.ensure_models()
        base = mp.tasks.BaseOptions
        self.mp, self.arm, self._last_ms = mp, arm, -1
        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(base_options=base(model_asset_path=str(models["pose_landmarker_full.task"])), running_mode=vision.RunningMode.VIDEO))
        self.hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(base_options=base(model_asset_path=str(models["hand_landmarker.task"])), running_mode=vision.RunningMode.VIDEO, num_hands=2))

    def step(self, img: np.ndarray, t: float) -> Seen:
        import cv2

        t0 = time.perf_counter()
        h, w = img.shape[:2]
        ts = max(int(round(t * 1000)), self._last_ms + 1)  # MediaPipe wants rising time stamps
        self._last_ms = ts
        mp_img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        f = rp.ArmFrame(t=t)
        seen = Seen(arm=f)
        pr = self.pose.detect_for_video(mp_img, ts)
        if pr.pose_landmarks:
            lm = pr.pose_landmarks[0]
            seen.body_px = np.array([[p.x * w, p.y * h] for p in lm])
            seen.body_vis = np.array([p.visibility for p in lm])
            if min(lm[k].visibility for k in rp.POSE[self.arm].values()) >= rp.MIN_VISIBILITY:
                f.pose_px, f.pose_vis = seen.body_px, seen.body_vis
                f.pose_world = np.array([[p.x, p.y, p.z] for p in pr.pose_world_landmarks[0]])
        hr = self.hands.detect_for_video(mp_img, ts)
        if hr.hand_landmarks:
            seen.hands_px = [np.array([[p.x * w, p.y * h] for p in hand]) for hand in hr.hand_landmarks]
            k = rp._pick_hand(seen.hands_px, f, self.arm)
            if k is not None and f.pose_world is not None:
                f.hand_px = seen.hands_px[k]
                f.hand_world = np.array([[p.x, p.y, p.z] for p in hr.hand_world_landmarks[k]])
        seen.took_ms = (time.perf_counter() - t0) * 1000
        return seen

    def close(self) -> None:
        self.pose.close()
        self.hands.close()


class ObjectWatcher:
    """Runs the object detector beside the camera loop: always on the newest picture, never blocking it."""

    def __init__(self, names: list[str] | None, device: str | None = None, detector=None, in_step: bool = False):
        """in_step: no thread; the loop calls look_now() and waits for it (recorded video without a window,
        where there is no hurry and the boxes should belong to the picture they are drawn on)."""
        from retarget import objects

        self._objects = objects
        self.names = names or objects.GENERIC_NAMES
        self.named = bool(names)
        self.device = device or "?"
        self.status = "loading the model"
        self.boxes: list[tuple] = []  # (Detection, [names])
        self.t_found = 0.0  # time.monotonic() of the picture the boxes belong to
        self.took_s = 0.0
        self.looks = 0
        self.enabled = True
        self._detector, self._want_device = detector, device
        self._img, self._t_img = None, 0.0
        self._lock, self._stop = threading.Lock(), False
        if in_step:
            self._load()
        else:
            threading.Thread(target=self._run, daemon=True).start()

    def _load(self) -> bool:
        try:
            if self._detector is None:
                self._detector = self._objects.Detector(device=self._want_device)
                if self._detector.device == "cpu" and not self._stop:  # leave half the processor for the body tracking
                    import os

                    self._detector.torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
            self.device = getattr(self._detector, "device", "test")
            self.status = "ready"
            return True
        except Exception as e:  # no torch, no model, no graphics card: say so in the panel and carry on without boxes
            self.status = f"off: {e}"
            return False

    def look_now(self, img: np.ndarray) -> None:
        if self.status != "ready" or not self.enabled:
            return
        t0 = time.perf_counter()
        found = self._objects.clean(self._detector.find(img, self.names), img.shape)
        if self.named:
            found = sorted(found, key=lambda b: -b[0].score)[:1]
        self.boxes, self.took_s, self.t_found = found, time.perf_counter() - t0, time.monotonic()
        self.looks += 1

    def give(self, img: np.ndarray) -> None:
        with self._lock:
            self._img, self._t_img = img, time.monotonic()

    def _run(self) -> None:
        if not self._load():
            return
        while not self._stop:
            with self._lock:
                img, t_img, self._img = self._img, self._t_img, None
            if img is None or not self.enabled:
                time.sleep(0.02)
                continue
            t0 = time.perf_counter()
            try:
                dets = self._detector.find(img, self.names)
            except Exception as e:
                self.status = f"error: {e}"
                time.sleep(1.0)
                continue
            found = self._objects.clean(dets, img.shape)
            if self.named:
                found = sorted(found, key=lambda b: -b[0].score)[:1]
            self.boxes, self.t_found, self.took_s = found, t_img, time.perf_counter() - t0
            self.looks += 1

    def close(self) -> None:
        self._stop = True


# ---------------------------------------------------------------- drawing


def draw_camera(img: np.ndarray, seen: Seen, st: LiveState, arm: str, watcher: ObjectWatcher | None, mirror: bool) -> np.ndarray:
    """The camera picture with everything found drawn on it. mirror: like looking in a mirror (webcam)."""
    import cv2

    out = cv2.flip(img, 1) if mirror else img.copy()
    w = img.shape[1]
    s = max(1, round(w / 640))

    def P(p):
        return (int(w - 1 - p[0]) if mirror else int(p[0]), int(p[1]))

    if watcher is not None and watcher.enabled:
        age = time.monotonic() - watcher.t_found
        for det, names in watcher.boxes:
            x0, y0, x1, y1 = det.box
            a, b = P((x0, y0)), P((x1, y1))
            col = (255, 200, 0) if age < 2.0 else (150, 130, 90)
            cv2.rectangle(out, (min(a[0], b[0]), a[1]), (max(a[0], b[0]), b[1]), col, 2 * s)
            cv2.putText(out, f"{'/'.join(names[:2])} {det.score:.2f}  age {age:.1f} s", (min(a[0], b[0]), max(a[1] - 6 * s, 12 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, col, s + 1)  # fmt: skip
    if seen.body_px is not None:
        ok = seen.body_vis > 0.5
        for a, b in BODY_BONES:
            if ok[a] and ok[b]:
                cv2.line(out, P(seen.body_px[a]), P(seen.body_px[b]), (200, 200, 200), s)
        for k in range(11, 29):
            if ok[k]:
                cv2.circle(out, P(seen.body_px[k]), 3 * s, (230, 230, 230), -1)
    for hp in seen.hands_px:
        for a, b in rp.HAND_BONES:
            cv2.line(out, P(hp[a]), P(hp[b]), (120, 160, 120), s)
    f = seen.arm
    if f.pose_px is not None:
        j = rp.POSE[arm]
        pts = [P(f.pose_px[j[n]]) for n in ("shoulder", "elbow", "wrist")]
        cv2.line(out, pts[0], pts[1], (0, 160, 255), 3 * s)
        cv2.line(out, pts[1], pts[2], (0, 160, 255), 3 * s)
        for p in pts:
            cv2.circle(out, p, 5 * s, (0, 100, 255), -1)
        if st.elbow_deg is not None:
            cv2.putText(out, f"elbow {st.elbow_deg:.0f} deg", (pts[1][0] + 8 * s, pts[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (0, 160, 255), s + 1)
    if f.hand_px is not None:
        for a, b in rp.HAND_BONES:
            cv2.line(out, P(f.hand_px[a]), P(f.hand_px[b]), (80, 255, 80), 2 * s)
        dot = P(rp.grasp_point(f.hand_px))
        closed = st.openness is not None and st.openness < 0.70
        col = (0, 0, 255) if closed else (255, 255, 255)
        cv2.circle(out, dot, 7 * s, col, -1)
        cv2.circle(out, dot, 7 * s, (0, 0, 0), s)
        if st.openness is not None:
            cv2.putText(out, f"hand {st.openness:.2f} {'CLOSED' if closed else 'open'}", (dot[0] + 10 * s, dot[1] + 4 * s), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, col, s + 1)
    if st.mode == CALIBRATING:
        h = img.shape[0]
        cv2.rectangle(out, (w // 4, h - 30 * s), (3 * w // 4, h - 14 * s), (60, 60, 60), -1)
        cv2.rectangle(out, (w // 4, h - 30 * s), (w // 4 + int(w / 2 * st.calib_progress), h - 14 * s), (0, 200, 255), -1)
    return out


def draw_panel(st: LiveState, seen: Seen, lines_extra: list[str], width: int, height: int) -> np.ndarray:
    import cv2

    img = np.full((height, width, 3), 28, np.uint8)
    col = MODE_COLOUR.get(st.mode, (255, 255, 255))
    title = st.mode.upper() + (f" {st.calib_progress:.0%}" if st.mode == CALIBRATING else "")
    cv2.putText(img, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)
    cv2.putText(img, st.message[:52], (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
    f = seen.arm
    lines = [
        f"arm {'seen' if f.pose_world is not None else 'NOT SEEN'}   hand {'seen' if f.hand_world is not None else 'NOT SEEN'}   tracking {seen.took_ms:.0f} ms",
        ("hand openness -      " if st.openness is None else f"hand openness {st.openness:.2f}") + f"   gripper {st.gripper_open * 100:.0f}% open",
        "tool point  -" if st.target is None else "tool point  x {:.0f}  y {:.0f}  z {:.0f} mm".format(*(st.target * 1000)),
        "joints  " + "  ".join(f"{v:.0f}" for v in np.degrees(st.q)) + " deg",
        f"IK miss {st.ik_error_mm:.0f} mm   behind {st.lag_deg:.0f} deg" + ("   AT EDGE OF REACH" if st.limited else ""),
        *lines_extra,
    ]
    for i, text in enumerate(lines):
        cv2.putText(img, text[:64], (10, 68 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (225, 225, 225), 1)
    return img


def compose(cam: np.ndarray, twin: np.ndarray | None, panel_lines: list[str], st: LiveState, seen: Seen) -> np.ndarray:
    import cv2

    canvas = np.full((CANVAS_H, CAM_W + SIDE_W, 3), 28, np.uint8)
    scale = min(CAM_W / cam.shape[1], CANVAS_H / cam.shape[0])
    small = cv2.resize(cam, (int(cam.shape[1] * scale), int(cam.shape[0] * scale)))
    y0, x0 = (CANVAS_H - small.shape[0]) // 2, (CAM_W - small.shape[1]) // 2
    canvas[y0 : y0 + small.shape[0], x0 : x0 + small.shape[1]] = small
    top = 0
    if twin is not None:
        canvas[:TWIN_H, CAM_W:] = cv2.cvtColor(twin, cv2.COLOR_RGB2BGR)
        top = TWIN_H
    canvas[top:, CAM_W:] = draw_panel(st, seen, panel_lines, SIDE_W, CANVAS_H - top)
    return canvas


# ---------------------------------------------------------------- the twin


class Twin:
    def __init__(self, q: np.ndarray, render: bool = True):
        import mujoco

        self.mj = mujoco
        self.model, self.data = scene.load()
        scene.set_pose(self.model, self.data, q, 1.0)
        self._qpos0 = self.data.qpos.copy()
        self.renderer = mujoco.Renderer(self.model, TWIN_H, SIDE_W) if render else None

    def step(self, q: np.ndarray, gripper_open: float, dt: float) -> None:
        for name, v in zip((*scene.ARM_JOINTS, "gripper"), (*q, scene.gripper_angle(gripper_open))):
            self.data.actuator(name).ctrl[0] = v
        for _ in range(int(np.clip(round(dt / self.model.opt.timestep), 1, 100))):
            self.mj.mj_step(self.model, self.data)

    def tool_error_mm(self, tool_xyz: np.ndarray) -> float:
        return float(np.linalg.norm(self.data.site(scene.TCP_SITE).xpos - tool_xyz) * 1000)

    def reset_block(self) -> None:
        adr = self.model.joint("part").qposadr[0]
        self.data.qpos[adr : adr + 7] = self._qpos0[adr : adr + 7]
        self.data.qvel[self.model.joint("part").dofadr[0] :][:6] = 0
        self.mj.mj_forward(self.model, self.data)

    def picture(self) -> np.ndarray | None:
        if self.renderer is None:
            return None
        self.renderer.update_scene(self.data, camera="front")
        return self.renderer.render()


# ---------------------------------------------------------------- the loop


def run_live(source=0, arm: str = "right", objects: str | None = None, port: str | None = None, robot_id: str = "follower",
             max_speed: float | None = None, mirror: bool | None = None, window: bool = True, twin: bool = True, save: Path | None = None,
             log: Path | None = None, max_seconds: float | None = None, device: str | None = None, detector=None, tracker=None) -> dict:  # fmt: skip
    """objects: None = no detector, "" = common things, "juice carton" = that thing.
    Returns a short summary (used by the tests and printed at the end)."""
    import cv2

    is_file = not isinstance(source, int)
    mirror = (not is_file) if mirror is None else mirror
    cap = cv2.VideoCapture(str(source) if is_file else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {'video ' + str(source) if is_file else f'camera {source}. Try --camera 1, or close other apps that use the webcam'}")
    if not is_file:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    chain = scripted._chain()
    cfg = LiveConfig(arm=arm, q_start=scripted.home_q(), max_joint_speed_deg_s=max_speed or (45.0 if port else 90.0))
    robot = mapping = None
    q_now = None
    if port:
        from retarget import robot as rr

        robot, mapping = rr.connect(port, robot_id), rr.JointMapping.load()
        q_now = rr.current_joints(robot, mapping)
        print(f"SO-101 on {port}: connected. It starts PAUSED. Press SPACE in the window to let it follow, SPACE again to stop.")
    follower = LiveFollower(chain, cfg, q_now=q_now)
    tw = Twin(follower.q, render=window or save is not None) if twin else None
    tracker = tracker or BodyTracker(arm)
    in_step = is_file and not window  # a recorded video with nobody watching: wait for the detector, twice per video second
    watcher = ObjectWatcher([objects] if objects else None, device=device, detector=detector, in_step=in_step) if objects is not None else None
    look_every = max(1, round(0.5 * src_fps))
    paused = robot is not None
    writer = None
    log_file = log_writer = None
    if log:
        log_file = open(log, "w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow(["t", "mode", "arm_seen", "hand_seen", "hand_openness", "elbow_deg", "x_mm", "y_mm", "z_mm",
                             *(f"{j}_deg" for j in scene.ARM_JOINTS), "gripper_open", "ik_miss_mm", "behind_deg", "objects"])  # fmt: skip

    t0 = time.monotonic()
    n = n_arm = n_hand = n_follow = 0
    t_prev, fps_now, twin_err = None, 0.0, 0.0
    ik_miss: list[float] = []
    st = LiveState()
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            t = n / src_fps if is_file else time.monotonic() - t0
            if max_seconds is not None and t > max_seconds:
                break
            seen = tracker.step(img, t)
            if watcher is not None and in_step:
                if n % look_every == 0:
                    watcher.look_now(img)
            elif watcher is not None:
                watcher.give(img)
            st = follower.update(seen.arm, advance=not paused)
            dt = 1 / src_fps if t_prev is None else t - t_prev
            t_prev = t
            fps_now = 0.9 * fps_now + 0.1 / max(dt, 1e-3) if fps_now else 1 / max(dt, 1e-3)
            if tw is not None:
                tw.step(st.q, st.gripper_open, dt)
                twin_err = tw.tool_error_mm(chain.fk(st.q)[:3, 3])
            if robot is not None and not paused and st.mode in (FOLLOWING, HOLD):
                robot.send_action(mapping.to_action(st.q, st.gripper_open))
            n += 1
            n_arm += seen.arm.pose_world is not None
            n_hand += seen.arm.hand_world is not None
            if st.mode == FOLLOWING and st.target is not None:
                n_follow += 1
                ik_miss.append(st.ik_error_mm)
            if log_writer:
                xyz = ["", "", ""] if st.target is None else [round(v * 1000, 1) for v in st.target]
                names = ";".join(f"{nm[0]}:{d.score:.2f}" for d, nm in watcher.boxes) if watcher else ""
                log_writer.writerow([round(t, 3), st.mode, int(seen.arm.pose_world is not None), int(seen.arm.hand_world is not None),
                                     "" if st.openness is None else round(st.openness, 3), "" if st.elbow_deg is None else round(st.elbow_deg, 1), *xyz,
                                     *(round(float(v), 2) for v in np.degrees(st.q)), round(st.gripper_open, 2), round(st.ik_error_mm, 1), round(st.lag_deg, 1), names])  # fmt: skip
            if window or save is not None:
                out_line = "twin" if robot is None else f"twin + SO-101 on {port}" + (" PAUSED (SPACE)" if paused else " FOLLOWING (SPACE stops)")
                lines = [f"output: {out_line}", f"camera {fps_now:.0f} pictures/s   twin tool vs command {twin_err:.0f} mm" if tw else f"camera {fps_now:.0f} pictures/s"]
                if watcher is not None:
                    if not watcher.enabled:
                        lines.append("objects: off (o)")
                    elif watcher.status != "ready":
                        lines.append(f"objects: {watcher.status}"[:60])
                    else:
                        what = ", ".join(f"{nm[0]} {d.score:.2f}" for d, nm in watcher.boxes[:3]) or "nothing found"
                        lines += [f"objects on {watcher.device}: one look takes {watcher.took_s:.1f} s", f"  {what}"]
                lines.append("q quit | c calibrate | SPACE pause | o objects | r block | s pic")
                canvas = compose(draw_camera(img, seen, st, arm, watcher, mirror), tw.picture() if tw else None, lines, st, seen)
                if save is not None:
                    if writer is None:
                        writer = cv2.VideoWriter(str(save), cv2.VideoWriter_fourcc(*"mp4v"), src_fps if is_file else 20.0, canvas.shape[1::-1])
                    writer.write(canvas)
                if window:
                    cv2.imshow("Robotonic live: your arm leads, the SO-101 follows", canvas)
                    key = cv2.waitKey(max(1, int(1000 / src_fps - seen.took_ms)) if is_file else 1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("c"):
                        follower.recalibrate()
                    elif key == ord(" "):
                        paused = not paused
                    elif key == ord("o") and watcher is not None:
                        watcher.enabled = not watcher.enabled
                    elif key == ord("r") and tw is not None:
                        tw.reset_block()
                    elif key == ord("s"):
                        name = f"live_{time.strftime('%H%M%S')}.jpg"
                        cv2.imwrite(name, canvas)
                        print(f"saved {name}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if log_file:
            log_file.close()
        if watcher is not None:
            watcher.close()
        if hasattr(tracker, "close"):
            tracker.close()
        if window:
            cv2.destroyAllWindows()
        if robot is not None:
            from retarget import robot as rr

            print("Moving the arm back to its start pose, then switching it off. Keep a hand near it.")
            try:
                rr.move_to(robot, mapping, follower.q, np.asarray(q_now), st.gripper_open, np.radians(30))
            finally:
                robot.disconnect()
    summary = {
        "pictures": n, "seconds": round(n / src_fps if is_file else time.monotonic() - t0, 1), "arm_seen": n_arm, "hand_seen": n_hand,
        "following": n_follow, "ik_miss_mm_median": round(float(np.median(ik_miss)), 1) if ik_miss else None,
        "ik_miss_mm_worst": round(float(np.max(ik_miss)), 1) if ik_miss else None, "scale": round(follower.scale, 3),
        "object_looks": watcher.looks if watcher else 0, "object_device": watcher.device if watcher else None,
    }  # fmt: skip
    return summary


def print_summary(s: dict) -> None:
    n = max(s["pictures"], 1)
    print(f"{s['pictures']} pictures in {s['seconds']} s. Arm seen in {s['arm_seen'] / n:.0%}, hand in {s['hand_seen'] / n:.0%}, robot following in {s['following'] / n:.0%}.")
    if s["ik_miss_mm_median"] is not None:
        print(f"IK miss: typical {s['ik_miss_mm_median']} mm, worst {s['ik_miss_mm_worst']} mm. Scale: 1 m of your arm = {s['scale'] * 100:.0f} cm for the robot.")
    if s["object_looks"]:
        print(f"Object detector looked {s['object_looks']} times on {s['object_device']}.")


def devices() -> int:
    """What this computer will use: graphics card for the detector, renderer for the twin, camera."""
    import platform

    print(f"Python {platform.python_version()} on {platform.system()}")
    try:
        import torch

        if torch.cuda.is_available():
            gb = torch.cuda.get_device_properties(0).total_memory / 2**30
            print(f"object detector: GRAPHICS CARD ({torch.cuda.get_device_name(0)}, {gb:.0f} GB), torch {torch.__version__}")
        else:
            print(f"object detector: CPU (torch {torch.__version__} cannot use a graphics card).")
            print("  With an NVIDIA card: pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124")
    except ImportError:
        print("object detector: not installed. pip install -r ../teach/requirements-vision.txt")
    try:
        import mediapipe

        print(f"body tracking: MediaPipe {mediapipe.__version__} on the CPU (its Python package has no graphics card mode on Windows; it is fast enough)")
    except ImportError:
        print("body tracking: not installed. pip install -r ../teach/requirements-pose.txt")
    try:
        import mujoco

        model, data = scene.load()
        r = mujoco.Renderer(model, 120, 160)
        r.update_scene(data)
        r.render()
        try:
            from OpenGL import GL

            name = GL.glGetString(GL.GL_RENDERER)
            name = name.decode() if name else "unknown"
        except Exception:
            name = "OpenGL (install PyOpenGL to see the card's name)"
        print(f"twin 3D view: MuJoCo {mujoco.__version__}, drawn by: {name}")
    except Exception as e:
        print(f"twin 3D view: FAILED ({e})")
    try:
        import cv2

        cap = cv2.VideoCapture(0)
        ok, img = cap.read()
        cap.release()
        print(f"camera 0: {img.shape[1]} x {img.shape[0]}" if ok else "camera 0: not found (try --camera 1)")
    except Exception as e:
        print(f"camera: FAILED ({e})")
    return 0
