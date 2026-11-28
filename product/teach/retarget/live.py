"""Live mode: your arm is the leader, the SO-101 is the follower.

    one camera picture -> ArmFrame (retarget.pose) -> LiveFollower.update() -> joint angles + gripper, now

Same idea as body mode in pose.py, but one picture at a time, so nothing can be measured "over
the whole video". A short calibration at the start replaces that:

    Sit as you will work, rest your working hand on the table and hold still (2 s).

From those seconds the follower takes your body directions (forward, left, up), your arm length
(for the scale to the robot's reach) and the table height (where your hand rests). After that,
each picture gives a tool point for the robot, a gripper opening from how open your hand is,
a gripper tilt and roll from your wrist, and joint angles by IK.

Wrist: the hand's pose during calibration (resting on the table) is "zero". Tip your fingers down or
up from there and the gripper tilts the same way (wrist_flex); turn your palm like a key and the
gripper rolls the same way (wrist_roll). Position still comes first: near the edge of the reach the
arm may not manage the full tilt. The joints never move faster than a set speed, so a tracking jump
becomes a slow move, not a jerk.

Positions are NOT true millimetres (one camera, scaled arm). No detector is needed to run or
test this file: it works on ArmFrames.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .kinematics import (
    SerialChain,
    axis_angle_to_matrix,
    make_transform,
    solve_ik,
    tool_down,
)
from .pose import (
    H_WRIST,
    POSE,
    SHOULDER_MIN_VIS,
    ArmFrame,
    body_axes,
    elbow_angle_deg,
    grasp_point,
    hand_openness,
    wrist_angles,
)

WAITING, CALIBRATING, FOLLOWING, HOLD = "waiting", "calibrating", "following", "hold"


@dataclass
class LiveConfig:
    arm: str = "right"  # which of YOUR arms leads
    calib_s: float = 2.0  # seconds of holding still at the start
    robot_reach_m: float = 0.33  # your full arm length maps to this distance from the robot's shoulder
    min_radius_m: float = 0.15  # the tool point stays between these distances from the robot base ...
    max_radius_m: float = 0.28
    table_z_m: float = 0.008  # ... and between these heights
    max_z_m: float = 0.28
    # The SO-101 has 5 joints: with its gripper pointing straight down it only reaches about 8 cm
    # above the table. So the gripper points down near the table and tilts forward as the hand goes up.
    tilt_from_z_m: float = 0.05
    tilt_full_z_m: float = 0.24
    tilt_max_deg: float = 85.0
    smooth_s: float = 0.12  # low-pass on the tool point (higher = calmer but later)
    max_joint_speed_deg_s: float = 90.0  # no joint moves faster than this
    hand_closed_at: float = 0.62  # hand openness that fully closes the gripper (fist is about 0.45)
    hand_open_at: float = 0.84  # hand openness that fully opens it (flat hand is about 0.9)
    hold_after_s: float = 0.3  # hand not seen for this long: the robot holds its pose
    ik_iterations: int = 40
    wrist: bool = True  # your wrist tilts and rolls the gripper (False: tilt from height only, roll fixed)
    flex_gain: float = 1.0  # gripper tilt per degree of hand tilt
    roll_gain: float = 1.0  # gripper roll per degree of palm turn (-1 if it turns the wrong way)
    flex_back_max_deg: float = 30.0  # how far past straight down the gripper may tilt back towards the base
    roll_max_deg: float = 150.0
    orientation_weight: float = 0.05  # IK: position always wins, this decides how hard it tries for the tilt and roll
    q_start: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.3, 1.0, 0.0]))


@dataclass
class LiveState:
    """What the follower decided for one picture. Everything the screen shows comes from here."""

    mode: str = WAITING
    message: str = ""
    q: np.ndarray = field(default_factory=lambda: np.zeros(5))  # commanded joints, rad
    gripper_open: float = 1.0  # 0 closed .. 1 open
    target: np.ndarray | None = None  # tool point asked for, robot base frame, m
    ik_error_mm: float = 0.0  # how far IK ends from the target
    lag_deg: float = 0.0  # how far the speed limit keeps the command behind IK
    limited: bool = False  # the hand left the robot's working area and the target was pulled back
    openness: float | None = None
    elbow_deg: float | None = None
    wrist_flex_deg: float | None = None  # your hand's tilt since calibration (+ = fingers further down)
    wrist_roll_deg: float | None = None  # your palm's turn since calibration
    calib_progress: float = 0.0
    scale: float = 0.0  # robot metres per human metre


class LiveFollower:
    def __init__(self, chain: SerialChain, cfg: LiveConfig | None = None, q_now: np.ndarray | None = None):
        self.chain = chain
        self.cfg = cfg or LiveConfig()
        self.q = np.asarray(self.cfg.q_start if q_now is None else q_now, dtype=float).copy()
        self.grip = 1.0
        self.shoulder_robot = chain._fk_frames(np.zeros(len(chain.active)))[2][1]  # position of the shoulder_lift joint
        self._r_down = tool_down(chain, 0.0)
        self.recalibrate()

    def tool_rotation(self, pos: np.ndarray, flex: float = 0.0, roll: float = 0.0) -> np.ndarray:
        """Gripper direction for a tool point: down near the table, tilting forward higher up, facing away from the base.

        flex (rad, + = further down / back towards the base) and roll (rad, about the fingers) come from your wrist.
        """
        cfg = self.cfg
        tilt = -np.radians(cfg.tilt_max_deg) * np.clip((pos[2] - cfg.tilt_from_z_m) / (cfg.tilt_full_z_m - cfg.tilt_from_z_m), 0.0, 1.0)
        tilt = float(np.clip(tilt + flex, -np.radians(cfg.tilt_max_deg), np.radians(cfg.flex_back_max_deg)))
        yaw = np.arctan2(pos[1], pos[0])
        r = axis_angle_to_matrix(np.array([0.0, 0.0, 1.0]), yaw) @ axis_angle_to_matrix(np.array([0.0, 1.0, 0.0]), tilt) @ self._r_down
        return r @ axis_angle_to_matrix(np.array([0.0, 0.0, 1.0]), roll)  # tool z = along the fingers

    def recalibrate(self) -> None:
        self.mode = WAITING
        self._calib: list[ArmFrame] = []
        self._pos: np.ndarray | None = None
        self._t_prev: float | None = None
        self._t_seen: float | None = None
        self.axes = self.shoulder = self.hand_ref = None
        self._wrist = np.zeros(2)  # smoothed (pitch, roll), rad
        self.scale = self.z_table = 0.0

    # ------------------------------------------------------------ calibration

    def _finish_calibration(self) -> None:
        j = POSE[self.cfg.arm]
        frames = self._calib
        self.axes = body_axes(frames)
        self.shoulder = np.median([f.pose_world[j["shoulder"]] for f in frames], axis=0)
        upper = np.median([np.linalg.norm(f.pose_world[j["elbow"]] - f.pose_world[j["shoulder"]]) for f in frames])
        fore = np.median([np.linalg.norm(f.pose_world[j["wrist"]] - f.pose_world[j["elbow"]]) for f in frames])
        hand = np.median([np.linalg.norm(grasp_point(f.hand_world) - f.hand_world[H_WRIST]) for f in frames])
        self.scale = self.cfg.robot_reach_m / (upper + fore + hand)
        self.z_table = float(np.median([self._grasp_body(f)[2] for f in frames]))
        self.hand_ref = frames[len(frames) // 2].hand_world.copy()  # wrist "zero": the hand resting on the table
        self._wrist = np.zeros(2)
        self._calib = []
        self.mode = FOLLOWING

    def _grasp_body(self, f: ArmFrame) -> np.ndarray:
        """Grasp point relative to the shoulder, in your own axes (forward, left, up), metres."""
        j = POSE[self.cfg.arm]
        wrist = (f.pose_world[j["wrist"]] - self.shoulder) @ self.axes
        return wrist + (grasp_point(f.hand_world) - f.hand_world[H_WRIST]) @ self.axes

    # ------------------------------------------------------------ one picture

    def update(self, f: ArmFrame, advance: bool = True) -> LiveState:
        """advance=False: work everything out and show it, but keep the commanded joints where they are (robot paused)."""
        cfg = self.cfg
        seen = f.pose_world is not None and f.hand_world is not None
        dt = 0.0 if self._t_prev is None else max(f.t - self._t_prev, 0.0)
        self._t_prev = f.t
        st = LiveState(mode=self.mode, q=self.q, gripper_open=self.grip, scale=self.scale)
        if seen:
            j = POSE[cfg.arm]
            st.openness = hand_openness(f.hand_world)
            st.elbow_deg = elbow_angle_deg(f.pose_world[j["shoulder"]], f.pose_world[j["elbow"]], f.pose_world[j["wrist"]])

        if self.mode in (WAITING, CALIBRATING):
            shoulder_ok = seen and (f.pose_vis is None or f.pose_vis[POSE[cfg.arm]["shoulder"]] >= SHOULDER_MIN_VIS)
            if seen and not shoulder_ok:
                if self._calib and f.t - self._calib[-1].t > 0.5:
                    self._calib = []
                st.mode = self.mode = WAITING if not self._calib else CALIBRATING
                st.message = f"Shoulder not seen: sit further back so your {cfg.arm} shoulder is in the picture"
                return st
            if not seen:
                if self.mode == CALIBRATING and self._calib and f.t - self._calib[-1].t > 0.5:
                    self._calib = []  # lost during calibration: start it again
                st.mode = self.mode = WAITING if not self._calib else CALIBRATING
                st.message = "Show both shoulders, your working arm and your hand"
                return st
            self._calib.append(f)
            self.mode = CALIBRATING
            st.calib_progress = min(1.0, (f.t - self._calib[0].t) / cfg.calib_s)
            st.message = "Rest your hand on the table and hold still"
            if st.calib_progress >= 1.0 and len(self._calib) >= 5:
                self._finish_calibration()
                st.message = "Following"
                st.scale = self.scale
            st.mode = self.mode
            return st

        # ---- following
        if not seen:
            if self._t_seen is not None and f.t - self._t_seen > cfg.hold_after_s:
                st.mode = HOLD
                st.message = "Hand not seen: holding"
            return st
        self._t_seen = f.t
        b = self._grasp_body(f)
        pos = self.shoulder_robot + self.scale * b
        pos[2] = self.scale * (b[2] - self.z_table) + cfg.table_z_m
        target, st.limited = self._into_working_area(pos)
        if self._pos is None:
            self._pos = target
        else:
            a = 1.0 if cfg.smooth_s <= 0 else 1.0 - np.exp(-dt / cfg.smooth_s)
            self._pos = self._pos + a * (target - self._pos)
        st.target = self._pos.copy()

        flex = roll = 0.0
        if self.hand_ref is not None:
            pitch_roll = np.array(wrist_angles(f.hand_world, self.hand_ref, self.axes))
            a = 1.0 if cfg.smooth_s <= 0 or dt <= 0 else 1.0 - np.exp(-dt / cfg.smooth_s)
            self._wrist = self._wrist + a * (pitch_roll - self._wrist)
            st.wrist_flex_deg, st.wrist_roll_deg = (float(v) for v in np.degrees(self._wrist))
            if cfg.wrist:
                flex = cfg.flex_gain * self._wrist[0]
                roll = float(np.clip(cfg.roll_gain * self._wrist[1], -np.radians(cfg.roll_max_deg), np.radians(cfg.roll_max_deg)))
        goal = make_transform(self.tool_rotation(self._pos, flex, roll), self._pos)
        # position wins over gripper direction
        res = solve_ik(self.chain, goal, self.q, orientation_weight=cfg.orientation_weight, iterations=cfg.ik_iterations)
        st.ik_error_mm = res.position_error_m * 1000
        step = np.radians(cfg.max_joint_speed_deg_s) * dt
        dq = res.q - self.q
        st.lag_deg = float(np.degrees(np.abs(dq).max()))
        if advance:
            self.q = self.q + np.clip(dq, -step, step)

        want = float(np.clip((st.openness - cfg.hand_closed_at) / (cfg.hand_open_at - cfg.hand_closed_at), 0.0, 1.0))
        if advance:
            self.grip += (want - self.grip) * (1.0 if dt <= 0 else min(1.0, dt / 0.08))
        st.q, st.gripper_open, st.mode = self.q, self.grip, FOLLOWING
        st.message = "Following" + (" (at the edge of the robot's reach)" if st.limited else "")
        return st

    def _into_working_area(self, pos: np.ndarray) -> tuple[np.ndarray, bool]:
        cfg = self.cfg
        out = pos.copy()
        r = float(np.linalg.norm(out[:2]))
        if r < 1e-6:
            out[:2], r = [cfg.min_radius_m, 0.0], cfg.min_radius_m
        out[:2] *= np.clip(r, cfg.min_radius_m, cfg.max_radius_m) / r
        if out[0] < 0.02:  # never behind the base
            out[0] = 0.02
        out[2] = np.clip(out[2], cfg.table_z_m, cfg.max_z_m)
        return out, bool(np.linalg.norm(out - pos) > 0.005)
