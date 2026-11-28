"""Teach mode for the twin view: record waypoints like on a teach pendant, open/close the gripper, save, replay.

    python -m so101_sim view                      twin only
    python -m so101_sim view --port COM4          the real arm follows the twin (SPACE = follow / pause)
    python -m so101_sim view --path paths/box.path.json     start with a saved path loaded

Two windows open: the 3D view (jog with the joint sliders in its "Control" panel on the right) and the
Teach panel with the buttons. Keys in the Teach panel: o open, c close, w or Enter record a waypoint,
u undo, Esc stop, SPACE follow/pause (real arm). Keys in the 3D view: O, C, Enter, SPACE do the same
(MuJoCo also uses O and C to toggle two drawing options; harmless).

Freedrive (real arm only, button or f in the Teach panel): like the freedrive button on a UR. The arm motors
switch off, you move the arm by hand, the twin copies it, Record waypoint stores the pose. The gripper keeps
its motor, so o / c still work. HOLD THE ARM before you switch it on: it has no gravity hold. Press
Freedrive again and the arm holds where you left it.

Save writes three files into paths/ (next to where you started the command):
  NAME.path.json   the taught path (waypoints + gripper steps)
  NAME.csv         joint trajectory:  python -m retarget replay NAME.csv --port COM4 --max-speed-deg 20
  NAME.task.json   task file for the URScript / SO-101 generator (only if the path picks and places)

Play runs the path in the twin only (the real arm holds still). Run on robot asks first, then the real
arm follows the twin through the path, never faster than --max-speed. Stop halts both where they are.
"""

from __future__ import annotations

import json
import queue
import time
from pathlib import Path

import numpy as np

from . import scene
from .replay import SerialChain  # also puts product/teach on the import path

from retarget.demo import write_trajectory  # noqa: E402
from retarget.teach import TeachPath, safe_name, to_task, to_trajectory  # noqa: E402

# GLFW key codes seen by the MuJoCo viewer's key callback
KEY_SPACE, KEY_ENTER, KEY_KP_ENTER, KEY_O, KEY_C = 32, 257, 335, 79, 67


def gripper_fraction(angle_rad: float) -> float:
    g = (angle_rad - scene.GRIPPER_CLOSED_RAD) / (scene.GRIPPER_OPEN_RAD - scene.GRIPPER_CLOSED_RAD)
    return float(np.clip(g, 0.0, 1.0))


class TeachSession:
    """All teach logic, no windows: the GUI and the tests drive it through these methods."""

    def __init__(self, model, data, paths_dir: Path = Path("paths"), hz: float = 30.0, play_speed_deg_s: float = 20.0):
        self.model, self.data = model, data
        self.chain = SerialChain.from_urdf()
        self.paths_dir = Path(paths_dir)
        self.hz = hz
        self.play_speed = play_speed_deg_s
        self.path = TeachPath()
        self.teaching = False
        self.traj = None  # trajectory being played
        self.k = 0
        self.message = "Press Teach, jog with the sliders, record waypoints."

    # --- the twin's commanded pose (what the sliders say) ---------------------------------------------
    def target(self) -> tuple[np.ndarray, float]:
        q = np.array([self.data.actuator(j).ctrl[0] for j in scene.ARM_JOINTS])
        return q, gripper_fraction(self.data.actuator("gripper").ctrl[0])

    def set_target(self, q: np.ndarray, g: float) -> None:
        for j, v in zip(scene.ARM_JOINTS, q):
            self.data.actuator(j).ctrl[0] = v
        self.data.actuator("gripper").ctrl[0] = scene.gripper_angle(g)

    # --- teaching ---------------------------------------------------------------------------------------
    def toggle_teach(self) -> None:
        self.teaching = not self.teaching
        self.message = "TEACH ON: jog, then Record waypoint (w), o = open, c = close." if self.teaching else "Teach off."

    def record(self) -> None:
        if not self.teaching:
            self.message = "Press Teach first."
            return
        q, g = self.target()
        s = self.path.add_waypoint(q, g, self.chain)
        self.message = f"Waypoint {self.path.n_waypoints} recorded: {s.text()}"

    def gripper(self, open_: bool) -> None:
        if self.playing:
            return
        q, _ = self.target()
        self.set_target(q, 1.0 if open_ else 0.0)
        if self.teaching:
            self.path.add_gripper(open_)
            self.message = f"Step {len(self.path.steps)}: gripper {'OPEN' if open_ else 'CLOSE'} recorded."
        else:
            self.message = f"Gripper {'open' if open_ else 'closed'} (not recorded: Teach is off)."

    def undo(self) -> None:
        s = self.path.undo()
        self.message = f"Removed: {s.text()}" if s else "Nothing to undo."

    def delete(self, i: int) -> None:
        if 0 <= i < len(self.path.steps):
            st = self.path.steps.pop(i)
            self.message = f"Deleted step {i + 1}: {st.text()}"

    def goto(self, i: int) -> None:
        """Move the twin to waypoint i (a one-step MoveJ at the replay speed)."""
        if not (0 <= i < len(self.path.steps)) or self.path.steps[i].do != "movej":
            self.message = "Select a MoveJ step to go to."
            return
        q, g = self.target()
        self.traj = to_trajectory(TeachPath(steps=[self.path.steps[i]]), q_start=q, gripper_start=g,
                                  max_speed_deg_s=self.play_speed, hz=self.hz)
        self.k = 0
        self.message = f"Going to step {i + 1}."

    def clear(self) -> None:
        self.path = TeachPath(self.path.name)
        self.message = "Path cleared."

    # --- files ------------------------------------------------------------------------------------------
    def save(self, name: str) -> list[Path]:
        if not self.path.n_waypoints:
            self.message = "Nothing to save: record at least one waypoint."
            return []
        name = safe_name(name or self.path.name)
        self.path.name = name
        out = [self.path.save(self.paths_dir / f"{name}.path.json")]
        csv = self.paths_dir / f"{name}.csv"
        write_trajectory(csv, to_trajectory(self.path, max_speed_deg_s=self.play_speed, hz=self.hz))
        out.append(csv)
        task = to_task(self.path, self.chain)
        if task is not None:
            tp = self.paths_dir / f"{name}.task.json"
            tp.write_text(json.dumps(task, indent=2))
            out.append(tp)
        self.message = f"Saved {name}: " + ", ".join(p.name for p in out)
        return out

    def load(self, file: Path) -> None:
        self.path = TeachPath.load(file)
        self.message = f"Loaded {self.path.name}: {self.path.n_waypoints} waypoints, {len(self.path.steps)} steps."

    # --- replay -----------------------------------------------------------------------------------------
    @property
    def playing(self) -> bool:
        return self.traj is not None

    def play(self, speed_deg_s: float | None = None) -> bool:
        if not self.path.n_waypoints:
            self.message = "No waypoints to play."
            return False
        if speed_deg_s:
            self.play_speed = float(speed_deg_s)
        q, g = self.target()
        self.teaching = False  # do not record while playing
        self.traj = to_trajectory(self.path, q_start=q, gripper_start=g, max_speed_deg_s=self.play_speed, hz=self.hz)
        self.k = 0
        self.message = f"Playing {self.path.name} at {self.play_speed:.0f} deg/s ({self.traj.t[-1]:.1f} s)."
        return True

    def stop(self) -> None:
        if self.playing:
            self.traj = None
            self.message = "Stopped."

    def tick(self) -> None:
        """Advance playback by one control period (call at hz)."""
        if self.traj is None:
            return
        if self.k >= len(self.traj.t):
            self.traj = None
            self.message = f"Done: {self.path.name}."
            return
        self.set_target(self.traj.q[self.k], float(self.traj.gripper_open[self.k]))
        self.k += 1


class RealArm:
    """The physical SO-101 beside the twin. Three modes:
    paused     motors on, holds its last pose
    follow     motors on, follows the twin (capped speed)
    freedrive  arm motors OFF: you move the arm by hand, the twin copies it. The gripper keeps its motor (o / c).
               Leaving freedrive: the arm holds where you left it.
    """

    def __init__(self, robot, mapping, max_step_rad: float):
        self.robot, self.mapping, self.max_step = robot, mapping, max_step_rad
        self.q_cmd = self.joints()
        self.g_cmd = 1.0
        self.follow = False
        self.freedrive = False

    def joints(self) -> np.ndarray:
        return self.mapping.from_observation(self.robot.get_observation())

    def set_freedrive(self, on: bool) -> str:
        arm = list(scene.ARM_JOINTS)
        if on and not self.freedrive:
            self.follow = False
            self.robot.bus.disable_torque(arm)
            self.freedrive = True
            return "FREEDRIVE: arm motors OFF. Move the arm by hand, Record waypoint (w), o / c for the gripper."
        if not on and self.freedrive:
            q = self.joints()
            self.q_cmd = q
            # goal = where the arm is now BEFORE the motors switch on, so it does not jump
            self.robot.send_action(self.mapping.to_action(q, self.g_cmd))
            self.robot.bus.enable_torque(arm)
            self.freedrive = False
            return "Freedrive off: the arm holds here."
        return ""

    def step(self, s: TeachSession) -> None:
        """One control period. In freedrive the twin is set from the real arm, else the real arm from the twin."""
        _, g_twin = s.target()
        self.g_cmd = float(np.clip(g_twin, self.g_cmd - 0.05, self.g_cmd + 0.05))
        if self.freedrive:
            q = self.joints()
            self.q_cmd = q
            s.set_target(q, g_twin)  # the twin copies your hand-guided pose; Record takes it from here
            self.robot.bus.sync_write("Goal_Position", {"gripper": self.mapping.to_action(q, self.g_cmd)["gripper.pos"]})
            return
        if self.follow:
            from .mirror import twin_joints

            q_twin, _ = twin_joints(s.data)
            self.q_cmd = self.q_cmd + np.clip(q_twin - self.q_cmd, -self.max_step, self.max_step)
        self.robot.send_action(self.mapping.to_action(self.q_cmd, self.g_cmd))  # paused: keep holding the last pose

    def status(self) -> str:
        return "FREEDRIVE (motors off)" if self.freedrive else ("FOLLOWS the twin" if self.follow else "paused")


def handle_command(s: TeachSession, arm: RealArm | None, cmd: str, arg=None, play_speed: float = 20.0, max_speed: float = 30.0) -> None:
    """One button press or key, from the panel or the 3D view."""
    if cmd in ("follow", "freedrive", "run_robot") and arm is None:
        s.message = "Needs the real arm: start with --port COM4."
    elif cmd == "follow":
        if arm.freedrive:
            s.message = "Leave freedrive first (F)."
        else:
            arm.follow = not arm.follow
            s.message = "Real arm FOLLOWS the twin." if arm.follow else "Real arm PAUSED (holds still)."
    elif cmd == "freedrive":
        s.stop()
        on = not arm.freedrive
        msg = arm.set_freedrive(on)
        if on and not s.teaching:
            s.toggle_teach()
            msg += " Teach is ON."
        s.message = msg
    elif cmd == "teach":
        s.toggle_teach()
    elif cmd == "record":
        s.record()
    elif cmd in ("open", "close"):
        s.gripper(cmd == "open")
    elif cmd == "undo":
        s.undo()
    elif cmd == "clear":
        s.clear()
    elif cmd == "delete":
        s.delete(int(arg))
    elif cmd == "goto":  # twin only, like play
        if arm is not None and arm.freedrive:
            s.message = "Leave freedrive first (F)."
        else:
            if arm is not None:
                arm.follow = False
            s.goto(int(arg))
    elif cmd == "speed":
        s.play_speed = float(arg)
    elif cmd == "save":
        s.save(arg)
    elif cmd == "load":
        try:
            s.load(Path(arg))
        except (OSError, ValueError, KeyError) as e:
            s.message = f"Could not load {arg}: {e}"
    elif cmd in ("play", "run_robot") and arm is not None and arm.freedrive:
        s.message = "Leave freedrive first (F), then play."
    elif cmd == "play":  # twin only: the real arm holds still
        if arm is not None:
            arm.follow = False
        s.play(arg)
        if arm is not None and s.playing:
            s.message += " Twin only, real arm holds."
    elif cmd == "run_robot":
        if s.play(min(float(arg or play_speed), max_speed)):
            arm.follow = True
            s.message += " REAL ARM follows."
    elif cmd == "stop":
        s.stop()
        if arm is not None:
            arm.follow = False


def run(port: str | None = None, robot_id: str = "follower", max_speed_deg_s: float = 30.0, path_file: Path | None = None,
        paths_dir: Path = Path("paths"), play_speed_deg_s: float = 20.0, hz: float = 30.0) -> int:
    import mujoco
    import mujoco.viewer

    model, data = scene.load()
    arm = None
    if port:
        from retarget import robot as rr

        arm = RealArm(rr.connect(port, robot_id), rr.JointMapping.load(), np.radians(max_speed_deg_s) / hz)
    scene.set_pose(model, data, arm.q_cmd if arm else np.zeros(5), 1.0)  # twin starts where the real arm is

    s = TeachSession(model, data, paths_dir, hz, play_speed_deg_s)
    if path_file:
        s.load(path_file)
    cmds: queue.Queue = queue.Queue()  # keys from the viewer thread and the panel, handled in the control loop

    def on_key(key: int) -> None:
        cmd = {KEY_SPACE: "follow", KEY_ENTER: "record", KEY_KP_ENTER: "record", KEY_O: "open", KEY_C: "close"}.get(key)
        if cmd:
            cmds.put(cmd)

    physics_steps = max(1, round((1.0 / hz) / model.opt.timestep))

    def control_step(viewer) -> None:
        while not cmds.empty():
            c = cmds.get()
            c = c if isinstance(c, tuple) else (c,)
            with viewer.lock():
                handle_command(s, arm, *c, play_speed=play_speed_deg_s, max_speed=max_speed_deg_s)
            print(s.message, flush=True)
        with viewer.lock():
            s.tick()
            if arm is not None:
                arm.step(s)
            for _ in range(physics_steps):
                mujoco.mj_step(model, data)
        viewer.sync()

    print(__doc__.split("Save writes")[0])
    if arm is not None:
        print(f"SO-101 on {port}: connected. Real arm PAUSED. SPACE = follow / pause, F = freedrive (move it by hand).")
    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
            panel = _make_panel(s, cmds, arm, play_speed_deg_s, max_speed_deg_s)
            if panel is None:
                print("No Tk on this Python: keys only (in the 3D view): Enter = record, O = open, C = close. "
                      "Teach is ON. Use `python -m so101_sim path` to save/replay from files.")
                s.teaching = True
                while viewer.is_running():
                    t0 = time.monotonic()
                    control_step(viewer)
                    time.sleep(max(0.0, 1.0 / hz - (time.monotonic() - t0)))
            else:
                root, refresh = panel

                def loop() -> None:
                    if not viewer.is_running():
                        root.destroy()
                        return
                    t0 = time.monotonic()
                    control_step(viewer)
                    refresh()
                    root.after(max(1, int(1000 * (1.0 / hz - (time.monotonic() - t0)))), loop)

                root.protocol("WM_DELETE_WINDOW", lambda: (viewer.close(), root.destroy()))
                root.after(10, loop)
                root.mainloop()
    finally:
        if arm is not None:
            print("Closing: the motors switch off now, the arm goes limp.")
            arm.robot.disconnect()
    return 0


def _make_panel(s: TeachSession, cmds: queue.Queue, arm: RealArm | None, play_speed: float, max_speed: float):
    """The Teach pendant window (Tk). None if this Python has no Tk."""
    try:
        from .teach_ui import make_panel
    except ImportError:
        return None
    return make_panel(s, cmds, arm, play_speed, max_speed)


def export_path(file: Path, speed_deg_s: float = 20.0) -> int:
    """Print a saved path and write its CSV (and task file) next to it."""
    path = TeachPath.load(file)
    chain = SerialChain.from_urdf()
    print(f"{path.name}: {path.n_waypoints} waypoints, {len(path.steps)} steps")
    for ln in path.lines():
        print("  " + ln)
    stem = Path(file).name.removesuffix(".path.json").removesuffix(".json")
    traj = to_trajectory(path, max_speed_deg_s=speed_deg_s)
    csv = Path(file).with_name(f"{stem}.csv")
    write_trajectory(csv, traj)
    print(f"wrote {csv} ({traj.t[-1]:.1f} s at {speed_deg_s:.0f} deg/s)")
    task = to_task(path, chain)
    if task is not None:
        tp = Path(file).with_name(f"{stem}.task.json")
        tp.write_text(json.dumps(task, indent=2))
        print(f"wrote {tp}")
    else:
        print("no task file: the path does not close and open the gripper")
    print(f"Real arm:  python -m retarget replay {csv.resolve()} --port COM4 --max-speed-deg {speed_deg_s:.0f}   (run from product\\teach)")
    return 0
