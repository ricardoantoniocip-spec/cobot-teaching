"""Live mode: your arm is the leader, the SO-101 is the follower.

    one camera picture -> ArmFrame (retarget.pose) -> LiveFollower.update() -> joint angles + gripper, now

Same idea as body mode in pose.py, but one picture at a time, so nothing can be measured "over
the whole video". A short calibration at the start replaces that:

    Sit as you will work, rest your working hand on the table and hold still (2 s).

From those seconds the follower takes your body directions (forward, left, up), your arm length
(for the scale to the robot's reach) and the table height (where your hand rests). After that,
each picture gives a tool point for the robot, a gripper opening from how open your hand is,
a gripper tilt and roll from your wrist, and joint angles by IK.

Where the robot works: your hand's resting place during calibration becomes the middle of the
robot's working area (rest_xy_m), so it has room to move every way, also towards and away from you.

Wrist: the hand's pose during calibration (resting on the table) is "zero". Tip your fingers down or
up from there and the robot's wrist_flex turns by the same angle; turn your palm like a key and
wrist_roll turns the same way. These go straight onto the two wrist joints, after IK has placed the
tool point, so the wrist can never pull the arm into a bad pose (first real test, 28 Nov: asking IK
for the tilt jammed wrist_flex at its limit and the tool point missed by 30-90 mm). The joints never move faster than a set speed, so a tracking jump
becomes a slow move, not a jerk.

Positions are NOT true millimetres (one camera, scaled arm). No detector is needed to run or
test this file: it works on ArmFrames.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .kinematics import (
    SO101IK,
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
    smooth_s: float = 0.12  # (kept for old callers; the filters below do the smoothing now)
    # One-Euro filters (Casiez 2012): very calm when you hold still, little lag when you move fast.
    pos_min_cutoff_hz: float = 0.6  # lower = calmer when still
    pos_beta: float = 6.0  # higher = less lag on fast moves (per m/s of the tool point)
    wrist_min_cutoff_hz: float = 0.4
    wrist_beta: float = 0.6  # per rad/s
    wrist_deadband_deg: float = 6.0  # wrist changes smaller than this are ignored (tracker noise is about this big)
    clutch: bool = True  # relative mode: the robot moves by how much your hand moves, from where it is (no jumps)
    ik: str = "exact"  # "exact": closed-form SO-101 IK; "numeric": the general damped-least-squares solver
    reach: str = "elbow"  # "elbow": forward from how straight your arm is (camera in front); "depth": the tracker's depth guess
    calib_still_m: float = 0.05  # calibration restarts (at most twice) if your hand moves more than this across the picture
    max_joint_speed_deg_s: float = 90.0  # no joint moves faster than this
    hand_closed_at: float = 0.62  # hand openness that fully closes the gripper (fist is about 0.45)
    hand_open_at: float = 0.84  # hand openness that fully opens it (flat hand is about 0.9)
    hold_after_s: float = 0.3  # hand not seen for this long: the robot holds its pose
    ik_iterations: int = 40
    wrist: bool = True  # your wrist tilts and rolls the gripper (False: tilt from height only, roll fixed)
    flex_gain: float = 1.0  # gripper tilt per degree of hand tilt
    roll_gain: float = 1.0  # gripper roll per degree of palm turn (-1 if it turns the wrong way)
    roll_max_deg: float = 150.0
    flex_max_deg: float = 60.0  # largest wrist_flex change your wrist can ask for, either way
    rest_xy_m: tuple[float, float] = (0.20, 0.0)  # where your calibration hand lands: the middle of the working area
    orientation_weight: float = 0.02  # IK: position always wins over the gripper direction
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
    wrist_at_limit: bool = False  # a wrist joint is at its end stop
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
        self._q_ik = self.q.copy()  # last IK answer without the wrist added: the warm start for the next one
        self._wrist_sign = self._wrist_signs()
        self._ik = SO101IK(chain)
        self.engaged = True  # clutch: False = the robot holds while you move your hand back (key g)
        self.recalibrate()

    def _wrist_signs(self) -> np.ndarray:
        """Which way wrist_flex and wrist_roll must turn so the gripper copies the hand, read from the URDF.

        flex: + = fingers further down / back towards the base. roll: right-hand rule about the fingers.
        """
        q = np.array([0.0, 0.3, 0.3, 1.0, 0.0])
        tool, axes, _ = self.chain._fk_frames(q)
        fingers = tool[:3, 2]
        q2 = q.copy()
        q2[3] += 0.1
        back = -tool[:3, 3] / (np.linalg.norm(tool[:3, 3]) + 1e-12)  # from the tool towards the base
        flex = np.sign((self.chain.fk(q2)[:3, 2] - fingers) @ back) or 1.0
        roll = np.sign(axes[4] @ fingers) or 1.0
        return np.array([flex, roll])

    def tool_rotation(self, pos: np.ndarray) -> np.ndarray:
        """Gripper direction for a tool point: down near the table, tilting forward higher up, facing away from the base."""
        cfg = self.cfg
        tilt = -np.radians(cfg.tilt_max_deg) * np.clip((pos[2] - cfg.tilt_from_z_m) / (cfg.tilt_full_z_m - cfg.tilt_from_z_m), 0.0, 1.0)
        yaw = np.arctan2(pos[1], pos[0])
        return axis_angle_to_matrix(np.array([0.0, 0.0, 1.0]), yaw) @ axis_angle_to_matrix(np.array([0.0, 1.0, 0.0]), tilt) @ self._r_down

    def recalibrate(self) -> None:
        self.mode = WAITING
        self._calib: list[ArmFrame] = []
        self._pos: np.ndarray | None = None
        self._t_prev: float | None = None
        self._t_seen: float | None = None
        self.axes = self.shoulder = self.hand_ref = None
        self._wrist = np.zeros(2)  # smoothed (pitch, roll), rad
        self._f_pos = OneEuro(self.cfg.pos_min_cutoff_hz, self.cfg.pos_beta)
        self._f_wrist = OneEuro(self.cfg.wrist_min_cutoff_hz, self.cfg.wrist_beta)
        self.scale = self.z_table = 0.0
        self.b_rest = None
        self._calib_tries = 0
        self._anchor: dict | None = None
        self._applied = np.zeros(2)  # (flex, roll) last sent to the wrist, rad

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
        self.upper, self.fore = float(upper), float(fore)
        rest = np.median([self._grasp_body(f) for f in frames], axis=0)
        self.z_table = float(rest[2])
        self.b_rest = rest
        self.hand_ref = frames[len(frames) // 2].hand_world.copy()  # wrist "zero": the hand resting on the table
        self._wrist = np.zeros(2)
        self._calib = []
        self.mode = FOLLOWING

    def _grasp_body_raw(self, f: ArmFrame) -> np.ndarray:
        """Grasp point relative to the hip centre in camera axes (before calibration, for the stillness check)."""
        j = POSE[self.cfg.arm]
        return f.pose_world[j["wrist"]] + grasp_point(f.hand_world) - f.hand_world[H_WRIST]

    def _grasp_body(self, f: ArmFrame) -> np.ndarray:
        """Grasp point relative to the (calibration) shoulder, in your own axes (forward, left, up), metres.

        reach="elbow": how far forward the wrist is comes from how straight your elbow is, not from the
        tracker's depth guess. Shoulder-to-wrist distance follows from the elbow angle and your two bone
        lengths (law of cosines); left/right and up/down are what a camera in front of you sees well; the
        rest of that distance must be forward. First real test (28 Nov): with depth from the tracker,
        stretching the arm moved the robot down and back.
        """
        cfg = self.cfg
        j = POSE[cfg.arm]
        wrist = (f.pose_world[j["wrist"]] - self.shoulder) @ self.axes
        if cfg.reach == "elbow" and getattr(self, "upper", None):
            sh_now = (f.pose_world[j["shoulder"]] - self.shoulder) @ self.axes  # you lean: the shoulder moves
            rel = wrist - sh_now
            ang = np.radians(elbow_angle_deg(f.pose_world[j["shoulder"]], f.pose_world[j["elbow"]], f.pose_world[j["wrist"]]))
            d2 = self.upper**2 + self.fore**2 - 2 * self.upper * self.fore * np.cos(ang)
            fwd = np.sqrt(max(d2 - rel[1] ** 2 - rel[2] ** 2, 0.0))
            wrist = sh_now + np.array([fwd, rel[1], rel[2]])
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
                if self._calib and f.t - self._calib[-1].t > 1.0:
                    self._calib = []
                st.mode = self.mode = WAITING if not self._calib else CALIBRATING
                st.message = f"Shoulder not seen: sit further back so your {cfg.arm} shoulder is in the picture"
                return st
            if not seen:
                if self.mode == CALIBRATING and self._calib and f.t - self._calib[-1].t > 1.0:
                    self._calib = []  # lost during calibration: start it again
                st.mode = self.mode = WAITING if not self._calib else CALIBRATING
                st.message = "Show both shoulders, your working arm and your hand"
                return st
            self._calib.append(f)
            self.mode = CALIBRATING
            st.calib_progress = min(1.0, (f.t - self._calib[0].t) / cfg.calib_s)
            st.message = "Rest your hand on the table and hold still"
            if st.calib_progress >= 1.0 and len(self._calib) >= 5:
                # Did you move? Only across the picture (x, y): the tracker's depth is too noisy to judge,
                # and 10-90 % range so one bad picture does not count. After 3 tries it calibrates anyway.
                g = np.array([self._grasp_body_raw(c) for c in self._calib])[:, :2]
                spread = float((np.percentile(g, 90, axis=0) - np.percentile(g, 10, axis=0)).max())
                if spread > cfg.calib_still_m and self._calib_tries < 2:
                    self._calib_tries += 1
                    self._calib = [f]
                    st.calib_progress = 0.0
                    st.message = f"You moved ({spread * 100:.0f} cm): rest your hand on the table and hold still"
                    return st
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
                self._anchor = None  # when the hand comes back, carry on from where the robot is: no jump
            return st
        self._t_seen = f.t
        if cfg.clutch and not self.engaged:
            self._anchor = None
            st.mode, st.message = HOLD, "Clutch open: move your hand where you like, press g to carry on"
            return st
        b = self._grasp_body(f)
        if cfg.clutch:
            if self._anchor is None:
                self._set_anchor(f, b)
            a = self._anchor
            pos = a["pos"] + self.scale * (b - a["b"])
        else:
            pos = np.array([*cfg.rest_xy_m, cfg.table_z_m]) + self.scale * (b - self.b_rest)
        target, st.limited = self._into_working_area(pos)
        self._pos = self._f_pos(target, f.t)
        st.target = self._pos.copy()

        flex = roll = 0.0
        if self.hand_ref is not None:  # your wrist: change since calibration (or since the clutch took hold)
            ref = self._anchor["hand"] if cfg.clutch else self.hand_ref
            pitch_roll = np.array(wrist_angles(f.hand_world, ref, self.axes))
            self._wrist = self._f_wrist(pitch_roll, f.t)
            st.wrist_flex_deg, st.wrist_roll_deg = (float(v) for v in np.degrees(self._wrist))
            if cfg.wrist:
                w = _deadband(self._wrist, np.radians(cfg.wrist_deadband_deg)) * np.array([cfg.flex_gain, cfg.roll_gain])
                if cfg.clutch:
                    w = w + self._anchor["applied"]
                flex = float(np.clip(w[0], -np.radians(cfg.flex_max_deg), np.radians(cfg.flex_max_deg)))
                roll = float(np.clip(w[1], -np.radians(cfg.roll_max_deg), np.radians(cfg.roll_max_deg)))
        self._applied = np.array([flex, roll])
        q_cmd, ik_err, st.wrist_at_limit = self._solve(self._pos, flex, roll)
        st.ik_error_mm = ik_err * 1000
        step = np.radians(cfg.max_joint_speed_deg_s) * dt
        dq = q_cmd - self.q
        st.lag_deg = float(np.degrees(np.abs(dq).max()))
        if advance:
            self.q = self.q + np.clip(dq, -step, step)

        want = float(np.clip((st.openness - cfg.hand_closed_at) / (cfg.hand_open_at - cfg.hand_closed_at), 0.0, 1.0))
        if advance:
            self.grip += (want - self.grip) * (1.0 if dt <= 0 else min(1.0, dt / 0.08))
        st.q, st.gripper_open, st.mode = self.q, self.grip, FOLLOWING
        st.message = "Following" + (" (at the edge of the robot's reach)" if st.limited else "") + (" (wrist at its end stop)" if st.wrist_at_limit else "")
        if cfg.clutch and not advance:
            self._anchor = None  # paused: when it carries on, start from where the robot is
        return st

    # ------------------------------------------------------------ clutch and IK

    def set_engaged(self, on: bool) -> None:
        """Clutch: off = the robot holds while you move your hand; on = it follows again from where it is."""
        self.engaged = bool(on)
        self._anchor = None

    def reanchor(self) -> None:
        """Carry on from where the robot is now (after a pause): no jump."""
        self._anchor = None

    def _set_anchor(self, f: ArmFrame, b: np.ndarray) -> None:
        pos = self.chain.fk(self.q)[:3, 3]
        pos, _ = self._into_working_area(pos)
        self._anchor = {"b": b.copy(), "pos": pos, "hand": f.hand_world.copy(), "applied": self._applied.copy()}
        self._f_pos = OneEuro(self.cfg.pos_min_cutoff_hz, self.cfg.pos_beta)
        self._f_wrist = OneEuro(self.cfg.wrist_min_cutoff_hz, self.cfg.wrist_beta)

    def _solve(self, pos: np.ndarray, flex: float, roll: float) -> tuple[np.ndarray, float, bool]:
        """Joint angles for a tool point plus your wrist's tilt and roll. Returns (q, position miss m, wrist at end stop)."""
        cfg = self.cfg
        if cfg.ik == "numeric":
            goal = make_transform(self.tool_rotation(pos), pos)
            res = solve_ik(self.chain, goal, self._q_ik, orientation_weight=cfg.orientation_weight, iterations=cfg.ik_iterations)
            if res.position_error_m > 0.01:  # stuck in a bad pose: start again from the home pose
                retry = solve_ik(self.chain, goal, cfg.q_start, orientation_weight=cfg.orientation_weight, iterations=4 * cfg.ik_iterations)
                if retry.position_error_m < res.position_error_m:
                    res = retry
            self._q_ik = res.q.copy()
            q = res.q.copy()
            q[3] += self._wrist_sign[0] * flex
            q[4] += self._wrist_sign[1] * roll
            clipped = self.chain.clip(q)
            return clipped, res.position_error_m, bool(np.any(np.abs(clipped[3:] - q[3:]) > 1e-6))
        # exact IK: the gripper pitch is the height schedule plus your wrist tilt, so the tool point stays
        # where your hand is while the gripper tilts. If that pitch cannot be reached there, ease it back
        # towards the schedule until it can.
        tilt = -np.radians(cfg.tilt_max_deg) * np.clip((pos[2] - cfg.tilt_from_z_m) / (cfg.tilt_full_z_m - cfg.tilt_from_z_m), 0.0, 1.0)
        pitch_sched = -np.pi / 2 - tilt  # finger angle in the arm's plane (-90 deg = straight down)
        res = None
        tries = [pitch_sched - k * flex for k in (1.0, 0.75, 0.5, 0.25, 0.0)]
        tries += [pitch_sched + np.radians(d) for d in (10, 20, 30, 45, 60)]  # still not: tilt forward (reach)
        best = None
        for i, pitch in enumerate(tries):
            res = self._ik.solve(pos, pitch, roll, q_prev=self._q_ik)
            if best is None or res.position_error_m < best[1].position_error_m - 1e-4:
                best = (i, res)
            if res.position_error_m < 0.003:
                break
        k_i, res = best
        k = 1.0 if k_i == 0 else 0.0
        self._q_ik = res.q.copy()
        at_limit = bool(k < 1.0 and abs(flex) > 1e-3) or bool(np.any(np.abs(res.q[3:] - self.chain.lower[3:]) < 1e-3)
                                                             or np.any(np.abs(res.q[3:] - self.chain.upper[3:]) < 1e-3))  # fmt: skip
        return res.q.copy(), res.position_error_m, at_limit

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


def _deadband(x: np.ndarray, width: float) -> np.ndarray:
    """Zero inside +-width, then continues smoothly (no jump at the edge)."""
    return np.sign(x) * np.maximum(np.abs(x) - width, 0.0)


class OneEuro:
    """One-Euro filter for a vector (Casiez, Roussel, Vogel 2012). The cut-off frequency rises with speed:
    still = heavy smoothing (no shaking), fast = light smoothing (little lag)."""

    def __init__(self, min_cutoff_hz: float, beta: float, d_cutoff_hz: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff_hz, beta, d_cutoff_hz
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.x is None or self.t is None or t <= self.t:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = t - self.t
        dx = (x - self.x) / dt
        self.dx = self.dx + self._alpha(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self.dx))
        self.x = self.x + self._alpha(cutoff, dt) * (x - self.x)
        self.t = t
        return self.x.copy()
