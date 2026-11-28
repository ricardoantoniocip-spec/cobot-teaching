"""Teach mode for the twin view: record waypoints like on a teach pendant, open/close the gripper, save, replay.

    python -m so101_sim view                      twin only
    python -m so101_sim view --port COM4          the real arm follows the twin (SPACE = follow / pause)
    python -m so101_sim view --path paths/box.path.json     start with a saved path loaded

Two windows open: the 3D view (jog with the joint sliders in its "Control" panel on the right) and the
Teach panel with the buttons. Keys in the Teach panel: o open, c close, w or Enter record a waypoint,
u undo, Esc stop, SPACE follow/pause (real arm). Keys in the 3D view: O, C, Enter, SPACE do the same
(MuJoCo also uses O and C to toggle two drawing options; harmless).

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


def run(port: str | None = None, robot_id: str = "follower", max_speed_deg_s: float = 30.0, path_file: Path | None = None,
        paths_dir: Path = Path("paths"), play_speed_deg_s: float = 20.0, hz: float = 30.0) -> int:
    import mujoco
    import mujoco.viewer

    from .mirror import step_towards, twin_joints

    model, data = scene.load()
    robot = mapping = None
    q_cmd, g_cmd = np.zeros(5), 1.0
    if port:
        from retarget import robot as rr

        robot = rr.connect(port, robot_id)
        mapping = rr.JointMapping.load()
        q_cmd = rr.current_joints(robot, mapping)
    scene.set_pose(model, data, q_cmd, g_cmd)  # twin starts where the real arm is

    s = TeachSession(model, data, paths_dir, hz, play_speed_deg_s)
    if path_file:
        s.load(path_file)
    state = {"follow": False}
    cmds: queue.Queue = queue.Queue()  # keys from the viewer thread and the panel, handled in the control loop

    def on_key(key: int) -> None:
        cmd = {KEY_SPACE: "follow", KEY_ENTER: "record", KEY_KP_ENTER: "record", KEY_O: "open", KEY_C: "close"}.get(key)
        if cmd:
            cmds.put(cmd)

    def handle(cmd: str, arg=None) -> None:
        if cmd == "follow" and robot is not None:
            state["follow"] = not state["follow"]
            s.message = "Real arm FOLLOWS the twin." if state["follow"] else "Real arm PAUSED (holds still)."
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
        elif cmd == "save":
            s.save(arg)
        elif cmd == "load":
            try:
                s.load(Path(arg))
            except (OSError, ValueError, KeyError) as e:
                s.message = f"Could not load {arg}: {e}"
        elif cmd == "play":  # twin only: the real arm holds still
            if state["follow"]:
                state["follow"] = False
            s.play(arg)
            if robot is not None and s.playing:
                s.message += " Twin only, real arm holds."
        elif cmd == "run_robot" and robot is not None:
            if s.play(min(float(arg or play_speed_deg_s), max_speed_deg_s)):
                state["follow"] = True
                s.message += " REAL ARM follows."
        elif cmd == "stop":
            s.stop()
            state["follow"] = False if robot is not None else state["follow"]
        print(s.message, flush=True)

    max_step = np.radians(max_speed_deg_s) / hz
    physics_steps = max(1, round((1.0 / hz) / model.opt.timestep))

    def control_step(viewer) -> None:
        nonlocal q_cmd, g_cmd
        while not cmds.empty():
            c = cmds.get()
            with viewer.lock():
                handle(*c) if isinstance(c, tuple) else handle(c)
        with viewer.lock():
            s.tick()
            for _ in range(physics_steps):
                mujoco.mj_step(model, data)
            if robot is not None:
                if state["follow"]:
                    q_twin, g_twin = twin_joints(data)
                    q_cmd = step_towards(q_cmd, q_twin, max_step)
                    g_cmd = float(np.clip(g_twin, g_cmd - 0.05, g_cmd + 0.05))
                robot.send_action(mapping.to_action(q_cmd, g_cmd))  # paused: keep holding the last pose
        viewer.sync()

    print(__doc__.split("Save writes")[0])
    if robot is not None:
        print(f"SO-101 on {port}: connected. Real arm PAUSED. SPACE = follow / pause.")
    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
            panel = _make_panel(s, cmds, robot is not None, state, play_speed_deg_s, max_speed_deg_s)
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
        if robot is not None:
            print("Closing: the motors switch off now, the arm goes limp.")
            robot.disconnect()
    return 0


def _make_panel(s: TeachSession, cmds: queue.Queue, has_robot: bool, state: dict, play_speed: float, max_speed: float):
    """The Teach panel (Tk). None if this Python has no Tk."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except ImportError:
        return None
    try:
        root = tk.Tk()
    except tk.TclError:
        return None
    root.title("SO-101 Teach")
    root.attributes("-topmost", True)
    big = ("Segoe UI", 11)

    top = tk.Frame(root, padx=8, pady=6)
    top.pack(fill="x")
    teach_btn = tk.Button(top, text="TEACH", width=10, font=("Segoe UI", 12, "bold"), command=lambda: cmds.put("teach"))
    teach_btn.pack(side="left")
    tk.Button(top, text="Record waypoint (w)", font=big, command=lambda: cmds.put("record")).pack(side="left", padx=4)
    tk.Button(top, text="Open (o)", font=big, command=lambda: cmds.put("open")).pack(side="left", padx=2)
    tk.Button(top, text="Close (c)", font=big, command=lambda: cmds.put("close")).pack(side="left", padx=2)

    mid = tk.Frame(root, padx=8)
    mid.pack(fill="x")
    tk.Button(mid, text="Undo (u)", command=lambda: cmds.put("undo")).pack(side="left")
    tk.Button(mid, text="Clear", command=lambda: messagebox.askyesno("Clear", "Delete all steps of this path?", parent=root) and cmds.put("clear")).pack(side="left", padx=4)
    tk.Label(mid, text="Name").pack(side="left", padx=(12, 2))
    name = tk.StringVar(value=s.path.name)
    tk.Entry(mid, textvariable=name, width=18).pack(side="left")
    tk.Button(mid, text="Save", command=lambda: cmds.put(("save", name.get()))).pack(side="left", padx=4)

    def load() -> None:
        s.paths_dir.mkdir(parents=True, exist_ok=True)
        f = filedialog.askopenfilename(parent=root, initialdir=str(s.paths_dir.resolve()), title="Load a path",
                                       filetypes=[("Teach path", "*.path.json"), ("All files", "*.*")])
        if f:
            cmds.put(("load", f))

    tk.Button(mid, text="Load...", command=load).pack(side="left")

    run_f = tk.Frame(root, padx=8, pady=6)
    run_f.pack(fill="x")
    tk.Label(run_f, text="Speed deg/s").pack(side="left")
    speed = tk.DoubleVar(value=play_speed)
    tk.Spinbox(run_f, from_=5, to=90, increment=5, textvariable=speed, width=5).pack(side="left", padx=4)
    tk.Button(run_f, text="Play in twin", font=big, command=lambda: cmds.put(("play", speed.get()))).pack(side="left", padx=4)

    def run_robot() -> None:
        v = min(speed.get(), max_speed)
        if messagebox.askyesno("Run on the real arm",
                               f"Clear the workspace and keep a hand near the arm.\n\nThe real arm moves first to where the twin is, "
                               f"then through the path at {v:.0f} deg/s.\nStop = Esc or the Stop button.\n\nRun now?", parent=root):
            cmds.put(("run_robot", v))

    if has_robot:
        tk.Button(run_f, text="Run on robot", font=big, bg="#f5c542", command=run_robot).pack(side="left", padx=4)
    tk.Button(run_f, text="STOP (Esc)", font=("Segoe UI", 11, "bold"), bg="#e05050", fg="white", command=lambda: cmds.put("stop")).pack(side="left", padx=4)

    steps = tk.Listbox(root, height=12, width=70, font=("Consolas", 10))
    steps.pack(fill="both", expand=True, padx=8)
    status = tk.Label(root, anchor="w", justify="left", wraplength=520, padx=8, pady=6)
    status.pack(fill="x")

    for key, cmd in (("o", "open"), ("c", "close"), ("w", "record"), ("<Return>", "record"), ("u", "undo"), ("<space>", "follow")):
        root.bind(key, lambda e, c=cmd: None if isinstance(e.widget, (tk.Entry, tk.Spinbox)) else cmds.put(c))
    root.bind("<Escape>", lambda e: cmds.put("stop"), add="+")  # Esc stops even while typing a name

    shown: dict = {"lines": None, "name": s.path.name}

    def refresh() -> None:
        teach_btn.config(text="TEACH ON" if s.teaching else "TEACH", bg="#4caf50" if s.teaching else root.cget("bg"),
                         fg="white" if s.teaching else "black")
        lines = s.path.lines()
        if lines != shown["lines"]:
            steps.delete(0, "end")
            for ln in lines:
                steps.insert("end", ln)
            steps.see("end")
            shown["lines"] = lines
        if s.path.name != shown["name"]:  # loaded or saved under a new name
            name.set(s.path.name)
            shown["name"] = s.path.name
        arm = ("  |  real arm: FOLLOWS" if state["follow"] else "  |  real arm: paused") if has_robot else ""
        status.config(text=f"{s.message}{arm}")

    return root, refresh


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
