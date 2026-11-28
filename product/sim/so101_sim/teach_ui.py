"""The Teach pendant window: a dark, pendant-style Tk panel beside the MuJoCo view.

Only drawing and input live here. Every button puts a command on the queue; teach_panel.handle_command
does the work in the control loop, so the window never touches the twin or the robot directly.
"""

from __future__ import annotations

import queue
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

# --- look -------------------------------------------------------------------------------------------------
BG = "#14171c"
CARD = "#1d2229"
CARD_HI = "#252b34"
LINE = "#2d343e"
TEXT = "#e7eaf0"
MUTED = "#8a94a3"
ACCENT = "#2f8cff"
GREEN = "#22b573"
ORANGE = "#ff9a1f"
RED = "#e5484d"
YELLOW = "#f2c94c"
BTN = "#2a313b"

UI = "Segoe UI" if sys.platform == "win32" else "DejaVu Sans"
MONO = "Consolas" if sys.platform == "win32" else "DejaVu Sans Mono"
F_TITLE = (UI, 15, "bold")
F_BIG = (UI, 13, "bold")
F_BTN = (UI, 10, "bold")
F_TXT = (UI, 10)
F_SMALL = (UI, 8, "bold")
F_MONO = (MONO, 10)

JOINT_NAMES = ("Base", "Shoulder", "Elbow", "Wrist flex", "Wrist roll")


def _shade(hex_color: str, f: float) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    r, g, b = (max(0, min(255, int(c + (255 - c) * f if f > 0 else c * (1 + f)))) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


class Btn(tk.Label):
    """Flat button that looks the same on Windows, macOS and Linux, with hover and a disabled state."""

    def __init__(self, master, text, command, color=BTN, fg=TEXT, font=F_BTN, pady=8, **kw):
        super().__init__(master, text=text, bg=color, fg=fg, font=font, pady=pady, padx=10, cursor="hand2", **kw)
        self.command, self.color, self.fg, self.enabled = command, color, fg, True
        self.bind("<Button-1>", lambda e: self.enabled and self.command())
        self.bind("<Enter>", lambda e: self.enabled and self.config(bg=_shade(self.color, 0.12)))
        self.bind("<Leave>", lambda e: self.config(bg=self.color if self.enabled else BTN))

    def style(self, text=None, color=None, fg=None, enabled=None):
        if text is not None and text != self.cget("text"):
            self.config(text=text)
        if color is not None:
            self.color = color
        if fg is not None:
            self.fg = fg
        if enabled is not None:
            self.enabled = enabled
        bg, fg_ = (self.color, self.fg) if self.enabled else (BTN, "#5b6470")
        if self.cget("bg") != bg and not (self.enabled and self.cget("bg") == _shade(self.color, 0.12)):
            self.config(bg=bg)
        if self.cget("fg") != fg_:
            self.config(fg=fg_, cursor="hand2" if self.enabled else "arrow")


def _card(master, title: str) -> tk.Frame:
    outer = tk.Frame(master, bg=CARD, highlightbackground=LINE, highlightthickness=1)
    tk.Label(outer, text=title.upper(), bg=CARD, fg=MUTED, font=F_SMALL, anchor="w").pack(fill="x", padx=12, pady=(10, 4))
    body = tk.Frame(outer, bg=CARD)
    body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
    outer.body = body
    return outer


class RepeatBtn(Btn):
    """Jog button: one step on click, keeps stepping while held."""

    def __init__(self, master, text, command, **kw):
        super().__init__(master, text, command, **kw)
        self.unbind("<Button-1>")
        self._job = None
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<ButtonRelease-1>", self._release)

    def _press(self, _=None, first=True):
        if not self.enabled:
            return
        self.command()
        self._job = self.after(350 if first else 50, lambda: self._press(first=False))

    def _release(self, _=None):
        if self._job:
            self.after_cancel(self._job)
            self._job = None


class TwinView(tk.Label):
    """The MuJoCo twin drawn inside the pendant. Drag = turn, right-drag = move, wheel = zoom.
    Taught waypoints are drawn as dots joined by a line (orange = the gripper closes or opens there)."""

    def __init__(self, master, model, data, s, width: int = 600, height: int = 420):
        import mujoco

        super().__init__(master, bg="#0b0d10", cursor="fleur", bd=0)
        self.mj, self.model, self.data, self.s = mujoco, model, data, s
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        self.renderer = mujoco.Renderer(model, height, width)
        self.cam = mujoco.MjvCamera()
        self.home_view()
        self.img = tk.PhotoImage(width=width, height=height)
        self.config(image=self.img)
        self._last = None
        self.bind("<ButtonPress-1>", self._start)
        self.bind("<ButtonPress-3>", self._start)
        self.bind("<B1-Motion>", lambda e: self._drag(e, mujoco.mjtMouse.mjMOUSE_ROTATE_V))
        self.bind("<B3-Motion>", lambda e: self._drag(e, mujoco.mjtMouse.mjMOUSE_MOVE_V))
        self.bind("<MouseWheel>", lambda e: self._zoom(-e.delta / 120))
        self.bind("<Button-4>", lambda e: self._zoom(-1))
        self.bind("<Button-5>", lambda e: self._zoom(1))
        self.bind("<Double-Button-1>", lambda e: self.home_view())

    def home_view(self) -> None:
        self.cam.type = self.mj.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = (0.17, 0.0, 0.08)
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.62, 150.0, -34.0

    def _start(self, e) -> None:
        self._last = (e.x, e.y)

    def _drag(self, e, action) -> None:
        if self._last is None:
            self._last = (e.x, e.y)
        dx, dy = e.x - self._last[0], e.y - self._last[1]
        self._last = (e.x, e.y)
        h = self.renderer.height
        self.mj.mjv_moveCamera(self.model, action, dx / h, dy / h, self.renderer.scene, self.cam)

    def _zoom(self, steps: float) -> None:
        self.mj.mjv_moveCamera(self.model, self.mj.mjtMouse.mjMOUSE_ZOOM, 0, 0.05 * steps, self.renderer.scene, self.cam)

    def _add(self, scn, kind, size, pos, rgba, to=None, width=0.0) -> None:
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        if to is None:
            self.mj.mjv_initGeom(g, kind, np.asarray(size, float), np.asarray(pos, float), np.eye(3).flatten(), np.asarray(rgba, np.float32))
        else:
            self.mj.mjv_initGeom(g, kind, np.zeros(3), np.zeros(3), np.eye(3).flatten(), np.asarray(rgba, np.float32))
            self.mj.mjv_connector(g, kind, width, np.asarray(pos, float), np.asarray(to, float))
        scn.ngeom += 1

    def draw(self) -> None:
        r = self.renderer
        r.update_scene(self.data, camera=self.cam)
        scn = r.scene
        pts, grip_pts = [], set()
        for st_ in self.s.path.steps:
            if st_.do == "movej" and st_.tool_mm is not None:
                pts.append(np.asarray(st_.tool_mm) / 1000)
            elif st_.do == "gripper" and pts:
                grip_pts.add(len(pts) - 1)
        for a, b in zip(pts, pts[1:]):
            self._add(scn, self.mj.mjtGeom.mjGEOM_CAPSULE, None, a, (0.18, 0.55, 1.0, 0.55), to=b, width=0.0018)
        for i, p_ in enumerate(pts):
            col = (1.0, 0.6, 0.12, 1) if i in grip_pts else (0.13, 0.71, 0.45, 1)
            self._add(scn, self.mj.mjtGeom.mjGEOM_SPHERE, (0.007, 0, 0), p_, col)
        px = r.render()
        h, w = px.shape[:2]
        self.img.configure(data=b"P6 %d %d 255\n" % (w, h) + px.tobytes(), format="PPM")


def make_panel(s, cmds: queue.Queue, arm, play_speed: float, max_speed: float, embed_view: bool = True):
    try:
        root = tk.Tk()
    except tk.TclError:
        return None
    root.title("Robotonic  |  SO-101 Teach Pendant")
    root.configure(bg=BG)
    try:  # same icon as the desktop shortcut
        from pathlib import Path

        icon = Path(__file__).resolve().parents[1] / "windows" / ("teach-pendant-robot.png" if arm is not None else "teach-pendant-twin.png")
        root._icon = tk.PhotoImage(file=str(icon))
        root.iconphoto(True, root._icon)
    except Exception:  # noqa: BLE001
        pass
    root.attributes("-topmost", True)
    root.minsize(1100, 700)

    st = ttk.Style(root)
    st.theme_use("clam")
    st.configure("P.Treeview", background=CARD_HI, fieldbackground=CARD_HI, foreground=TEXT, rowheight=26, font=F_MONO,
                 borderwidth=0, relief="flat")
    st.configure("P.Treeview.Heading", background=CARD, foreground=MUTED, font=F_SMALL, relief="flat", borderwidth=0)
    st.map("P.Treeview", background=[("selected", ACCENT)], foreground=[("selected", "white")])
    st.map("P.Treeview.Heading", background=[("active", CARD)])
    st.layout("P.Treeview", [("Treeview.treearea", {"sticky": "nswe"})])  # no light border
    st.configure("P.Vertical.TScrollbar", background=BTN, troughcolor=CARD_HI, borderwidth=0, arrowcolor=MUTED,
                 bordercolor=CARD_HI, lightcolor=BTN, darkcolor=BTN, gripcount=0)
    st.map("P.Vertical.TScrollbar", background=[("active", LINE)])
    st.configure("P.Horizontal.TScale", background=ACCENT, troughcolor=BTN, bordercolor=CARD, lightcolor=ACCENT,
                 darkcolor=ACCENT, gripcount=0, sliderlength=22)
    st.map("P.Horizontal.TScale", background=[("active", _shade(ACCENT, 0.2))])

    # --- header -----------------------------------------------------------------------------------------
    head = tk.Frame(root, bg=BG)
    head.pack(fill="x", padx=16, pady=(14, 8))
    tk.Label(head, text="ROBOTONIC", bg=BG, fg=ACCENT, font=(UI, 9, "bold")).pack(anchor="w")
    title_row = tk.Frame(head, bg=BG)
    title_row.pack(fill="x")
    tk.Label(title_row, text="SO-101 Teach Pendant", bg=BG, fg=TEXT, font=F_TITLE).pack(side="left")
    mode = tk.Label(title_row, text="READY", bg=BTN, fg=TEXT, font=F_SMALL, padx=10, pady=4)
    mode.pack(side="right")
    conn = tk.Label(title_row, bg=BG, font=F_TXT, padx=12,
                    text="●  Real arm connected" if arm is not None else "●  Twin only (no robot)",
                    fg=GREEN if arm is not None else MUTED)
    conn.pack(side="right")

    body = tk.Frame(root, bg=BG)
    body.pack(fill="both", expand=True, padx=16)
    body.columnconfigure(2, weight=1)
    body.rowconfigure(0, weight=1)
    left = tk.Frame(body, bg=BG, width=290)
    left.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
    center = tk.Frame(body, bg=BG)
    center.grid(row=0, column=1, sticky="nsew", padx=(0, 12))
    right = tk.Frame(body, bg=BG)
    right.grid(row=0, column=2, sticky="nsew")

    view = None
    if embed_view:
        c = _card(center, "3D twin   ·   drag turn   ·   right-drag move   ·   wheel zoom   ·   double-click reset")
        c.pack(fill="x", pady=(0, 12))
        sw = root.winfo_screenwidth()
        vw = int(np.clip(sw - 1000, 420, 640))
        view = TwinView(c.body, s.model, s.data, s, vw, int(vw * 0.68))
        view.pack()

    # --- left: mode + teach ----------------------------------------------------------------------------
    c = _card(left, "Mode")
    c.pack(fill="x", pady=(0, 12))
    teach_btn = Btn(c.body, "TEACH", lambda: cmds.put("teach"), font=F_BIG, pady=10)
    teach_btn.pack(fill="x")
    fd_btn = follow_btn = None
    if arm is not None:
        def freedrive() -> None:
            if arm.freedrive or messagebox.askyesno(
                    "Freedrive", "HOLD THE ARM with your hand now.\n\nThe arm motors switch off so you can move it by hand. "
                    "Without your hand it drops onto the table.\nThe gripper keeps its motor (o / c).\n\nSwitch the motors off?",
                    parent=root):
                cmds.put("freedrive")

        fd_btn = Btn(c.body, "FREEDRIVE", freedrive, pady=9)
        fd_btn.pack(fill="x", pady=(8, 0))
        follow_btn = Btn(c.body, "FOLLOW TWIN", lambda: cmds.put("follow"), pady=7)
        follow_btn.pack(fill="x", pady=(8, 0))
        root.bind("f", lambda e: None if isinstance(e.widget, (tk.Entry, ttk.Entry)) else freedrive())

    c = _card(left, "Teach")
    c.pack(fill="x", pady=(0, 12))
    rec_btn = Btn(c.body, "＋  RECORD WAYPOINT", lambda: cmds.put("record"), color=ACCENT, fg="white", font=F_BIG, pady=12)
    rec_btn.pack(fill="x")
    grip = tk.Frame(c.body, bg=CARD)
    grip.pack(fill="x", pady=(8, 0))
    grip.columnconfigure((0, 1), weight=1)
    Btn(grip, "OPEN   o", lambda: cmds.put("open")).grid(row=0, column=0, sticky="ew", padx=(0, 4))
    Btn(grip, "CLOSE   c", lambda: cmds.put("close")).grid(row=0, column=1, sticky="ew", padx=(4, 0))
    Btn(c.body, "▣  SET BOX AT GRIPPER   b", lambda: cmds.put("box"), color=BTN, pady=7).pack(fill="x", pady=(8, 0))

    c = _card(left, "Jog")
    c.pack(fill="x")
    lo, hi = np.degrees(s.chain.lower), np.degrees(s.chain.upper)
    joint_rows = []  # (DoubleVar, value label, is_dragging dict)

    def jog_row(label: str, i: int, a: float, b: float, step: float, unit: str):
        r = tk.Frame(c.body, bg=CARD)
        r.pack(fill="x", pady=2)
        tk.Label(r, text=label, bg=CARD, fg=MUTED, font=F_TXT, width=9, anchor="w").pack(side="left")
        var = tk.DoubleVar(value=0.0)
        drag = {"on": False}
        to_cmd = (lambda v: np.radians(v)) if i < 5 else (lambda v: v / 100.0)

        def send(v=None):
            cmds.put(("jog", (i, to_cmd(float(var.get())))))

        def nudge(d):
            var.set(float(np.clip(var.get() + d, a, b)))
            send()

        RepeatBtn(r, "−", lambda: nudge(-step), pady=1, font=(UI, 10, "bold")).pack(side="left")
        sc = ttk.Scale(r, from_=a, to=b, variable=var, style="P.Horizontal.TScale", length=110, command=send)
        sc.pack(side="left", padx=4)
        sc.bind("<ButtonPress-1>", lambda e: drag.update(on=True))
        sc.bind("<ButtonRelease-1>", lambda e: drag.update(on=False))
        RepeatBtn(r, "+", lambda: nudge(step), pady=1, font=(UI, 10, "bold")).pack(side="left")
        val = tk.Label(r, text="", bg=CARD, fg=TEXT, font=F_MONO, width=6, anchor="e")
        val.pack(side="right")
        joint_rows.append((var, val, drag, unit))

    for i, nm in enumerate(JOINT_NAMES):
        jog_row(nm, i, float(lo[i]), float(hi[i]), 1.0, "°")
    jog_row("Gripper", 5, 0.0, 100.0, 5.0, "%")
    r = tk.Frame(c.body, bg=CARD)
    r.pack(fill="x", pady=(6, 0))
    tk.Label(r, text="Tool mm", bg=CARD, fg=MUTED, font=F_TXT, width=9, anchor="w").pack(side="left")
    tool_val = tk.Label(r, text="", bg=CARD, fg=TEXT, font=F_MONO, anchor="e")
    tool_val.pack(side="right")
    grip_val = None
    if arm is not None:
        r = tk.Frame(c.body, bg=CARD)
        r.pack(fill="x", pady=(2, 0))
        tk.Label(r, text="Grip", bg=CARD, fg=MUTED, font=F_TXT, width=9, anchor="w").pack(side="left")
        grip_val = tk.Label(r, text="", bg=CARD, fg=TEXT, font=F_MONO, anchor="e")
        grip_val.pack(side="right")

    # --- right: program -------------------------------------------------------------------------------
    c = _card(right, "Program")
    c.pack(fill="both", expand=True, pady=(0, 12))
    row = tk.Frame(c.body, bg=CARD)
    row.pack(fill="x", pady=(0, 8))
    name = tk.StringVar(value=s.path.name)
    entry = tk.Entry(row, textvariable=name, bg=CARD_HI, fg=TEXT, insertbackground=TEXT, relief="flat", font=(UI, 11),
                     highlightthickness=1, highlightbackground=LINE, highlightcolor=ACCENT)
    entry.pack(side="left", fill="x", expand=True, ipady=6)

    def load() -> None:
        s.paths_dir.mkdir(parents=True, exist_ok=True)
        f = filedialog.askopenfilename(parent=root, initialdir=str(s.paths_dir.resolve()), title="Open a program",
                                       filetypes=[("Teach path", "*.path.json"), ("All files", "*.*")])
        if f:
            cmds.put(("load", f))

    Btn(row, "SAVE", lambda: cmds.put(("save", name.get())), color=GREEN, fg="white").pack(side="left", padx=(8, 0))
    Btn(row, "OPEN…", load).pack(side="left", padx=(6, 0))

    tv_frame = tk.Frame(c.body, bg=CARD)
    tv_frame.pack(fill="both", expand=True)
    tree = ttk.Treeview(tv_frame, style="P.Treeview", columns=("n", "cmd", "detail", "tool"), show="headings", selectmode="browse")
    for col, txt, w, anchor in (("n", "#", 36, "center"), ("cmd", "COMMAND", 80, "w"), ("detail", "JOINTS  base → wrist roll", 250, "w"),
                                 ("tool", "TOOL mm", 110, "w")):
        tree.heading(col, text=txt, anchor=anchor)
        tree.column(col, width=w, minwidth=w, anchor=anchor, stretch=col == "detail")
    tree.tag_configure("odd", background="#21262e")
    tree.tag_configure("grip", foreground=YELLOW)
    sb = ttk.Scrollbar(tv_frame, orient="vertical", command=tree.yview, style="P.Vertical.TScrollbar")
    tree.configure(yscrollcommand=sb.set)
    tree.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")
    empty = tk.Label(tv_frame, text="No steps yet.\nPress TEACH, move the arm, then RECORD WAYPOINT.", bg=CARD_HI, fg=MUTED,
                     font=F_TXT, justify="center")

    def selected() -> int | None:
        sel = tree.selection()
        return int(sel[0]) if sel else None

    def on_sel(cmd: str) -> None:
        i = selected()
        if i is None:
            s.message = "Select a step in the list first."
        else:
            cmds.put((cmd, i))

    tree.bind("<Double-1>", lambda e: on_sel("goto"))
    tree.bind("<Delete>", lambda e: on_sel("delete"))
    acts = tk.Frame(c.body, bg=CARD)
    acts.pack(fill="x", pady=(8, 0))
    Btn(acts, "GO TO", lambda: on_sel("goto")).pack(side="left")
    Btn(acts, "DELETE", lambda: on_sel("delete")).pack(side="left", padx=6)
    Btn(acts, "UNDO   u", lambda: cmds.put("undo")).pack(side="left")
    Btn(acts, "CLEAR ALL", lambda: messagebox.askyesno("Clear", "Delete all steps of this program?", parent=root)
        and cmds.put("clear")).pack(side="right")

    # --- right: run -----------------------------------------------------------------------------------
    c = _card(center, "Run")
    c.pack(fill="x")
    sp = tk.Frame(c.body, bg=CARD)
    sp.pack(fill="x")
    tk.Label(sp, text="Speed", bg=CARD, fg=MUTED, font=F_TXT).pack(side="left")
    speed = tk.DoubleVar(value=play_speed)
    speed_lbl = tk.Label(sp, text="", bg=CARD, fg=TEXT, font=(UI, 11, "bold"), width=9, anchor="e")
    speed_lbl.pack(side="right")
    top_speed = 90 if arm is None else max(5.0, max_speed)

    def on_speed(_=None) -> None:
        v = int(round(speed.get() / 5) * 5) or 5
        speed_lbl.config(text=f"{v} °/s")
        cmds.put(("speed", v))

    ttk.Scale(sp, from_=5, to=top_speed, variable=speed, style="P.Horizontal.TScale", command=on_speed).pack(
        side="left", fill="x", expand=True, padx=10)
    on_speed()
    if arm is not None:
        tk.Label(c.body, text=f"Real arm limit {max_speed:.0f} °/s (start with --max-speed to change)", bg=CARD, fg=MUTED,
                 font=(UI, 8)).pack(anchor="w")
    runs = tk.Frame(c.body, bg=CARD)
    runs.pack(fill="x", pady=(10, 0))
    runs.columnconfigure((0, 1, 2), weight=1)
    play_btn = Btn(runs, "▶  PLAY IN TWIN", lambda: cmds.put(("play", speed.get())), color=ACCENT, fg="white", pady=10)
    play_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
    robot_btn = None
    if arm is not None:
        def run_robot() -> None:
            v = min(speed.get(), max_speed)
            if messagebox.askyesno("Run on the real arm",
                                   f"Clear the workspace and keep a hand near the arm.\n\nThe real arm moves first to where the "
                                   f"twin is, then through the program at {v:.0f} °/s.\nSTOP = Esc or the red button.\n\nRun now?",
                                   parent=root):
                cmds.put(("run_robot", v))

        robot_btn = Btn(runs, "▶  RUN ON ROBOT", run_robot, color=YELLOW, fg="#1a1a1a", pady=10)
        robot_btn.grid(row=0, column=1, sticky="ew", padx=4)
    Btn(runs, "■  STOP", lambda: cmds.put("stop"), color=RED, fg="white", font=F_BIG, pady=8).grid(
        row=0, column=2, sticky="ew", padx=(4, 0))

    # --- status bar -----------------------------------------------------------------------------------
    bar = tk.Frame(root, bg=CARD, highlightbackground=LINE, highlightthickness=1)
    bar.pack(fill="x", side="bottom", padx=16, pady=(12, 14))
    status = tk.Label(bar, text="", bg=CARD, fg=TEXT, font=F_TXT, anchor="w", justify="left", wraplength=1200, padx=12, pady=8)
    status.pack(fill="x")
    keys = "Keys:  w / Enter record   o open   c close   b box at gripper   u undo   Esc stop" + ("   f freedrive   Space follow" if arm else "")
    tk.Label(bar, text=keys, bg=CARD, fg=MUTED, font=(UI, 8), anchor="w", padx=12).pack(fill="x", pady=(0, 6))

    for key, cmd in (("o", "open"), ("c", "close"), ("w", "record"), ("<Return>", "record"), ("u", "undo"), ("b", "box"), ("<space>", "follow")):
        root.bind(key, lambda e, cm=cmd: None if isinstance(e.widget, (tk.Entry, ttk.Entry)) else cmds.put(cm))
    root.bind("<Escape>", lambda e: cmds.put("stop"))

    shown: dict = {"rows": None, "name": s.path.name, "q": None}

    def rows():
        out = []
        for st_ in s.path.steps:
            if st_.do == "gripper":
                out.append(("GRIPPER", "OPEN" if st_.gripper >= 0.5 else "CLOSE", "", True))
            else:
                j = " ".join(f"{int(round(v)):5d}°" for v in st_.joints_deg)
                t = ", ".join(str(int(round(v))) for v in st_.tool_mm) if st_.tool_mm else ""
                out.append(("MoveJ", j, t, False))
        return out

    def refresh() -> None:
        # mode badge
        if arm is not None and arm.freedrive:
            m, col = "FREEDRIVE", ORANGE
        elif s.playing:
            m, col = "RUNNING" + (" ON ROBOT" if arm is not None and arm.follow else " IN TWIN"), ACCENT
        elif s.teaching:
            m, col = "TEACH", GREEN
        elif arm is not None and arm.follow:
            m, col = "FOLLOW", YELLOW
        else:
            m, col = "READY", BTN
        if mode.cget("text") != m:
            mode.config(text=m, bg=col, fg="#1a1a1a" if col in (YELLOW, ORANGE) else "white")

        teach_btn.style(text="TEACH  ON" if s.teaching else "TEACH", color=GREEN if s.teaching else BTN,
                        fg="white" if s.teaching else TEXT)
        rec_btn.style(enabled=s.teaching and not s.playing)
        busy = s.playing or (arm is not None and arm.freedrive)
        play_btn.style(enabled=not busy)
        if arm is not None:
            fd_btn.style(text="FREEDRIVE  ON" if arm.freedrive else "FREEDRIVE", color=ORANGE if arm.freedrive else BTN,
                         fg="#1a1a1a" if arm.freedrive else TEXT, enabled=not s.playing)
            follow_btn.style(text="FOLLOW TWIN  ON" if arm.follow else "FOLLOW TWIN", color=YELLOW if arm.follow else BTN,
                             fg="#1a1a1a" if arm.follow else TEXT, enabled=not arm.freedrive)
            robot_btn.style(enabled=not busy)

        # program list
        r = rows()
        if r != shown["rows"]:
            tree.delete(*tree.get_children())
            for i, (cmd, det, tool, is_grip) in enumerate(r):
                tags = (("odd",) if i % 2 else ()) + (("grip",) if is_grip else ())
                tree.insert("", "end", iid=str(i), values=(i + 1, cmd, det, tool), tags=tags)
            if r:
                tree.see(str(len(r) - 1))
                empty.place_forget()
            else:
                empty.place(relx=0.5, rely=0.5, anchor="center")
            shown["rows"] = r
        if s.path.name != shown["name"]:
            name.set(s.path.name)
            shown["name"] = s.path.name

        # live position + jog sliders
        q, g = s.target()
        vals = list(np.degrees(q)) + [g * 100]
        for (var, val, drag, unit), v in zip(joint_rows, vals):
            if not drag["on"] and abs(var.get() - v) > 0.05:
                var.set(v)
            txt = f"{v:5.1f}{unit}" if unit == "°" else f"{v:4.0f} %"
            if val.cget("text") != txt:
                val.config(text=txt)
        if shown["q"] is None or np.abs(q - shown["q"]).max() > 1e-4:
            tool = s.chain.fk(q)[:3, 3] * 1000
            tool_val.config(text=f"{tool[0]:4.0f} {tool[1]:4.0f} {tool[2]:4.0f}")
            shown["q"] = q.copy()
        if grip_val is not None:
            txt = (f"part held at {arm.gripped_at * 100:.0f} %" if arm.gripped_at is not None else "no part") + f"   load {arm.last_load:.0f}"
            if grip_val.cget("text") != txt:
                grip_val.config(text=txt, fg=GREEN if arm.gripped_at is not None else MUTED)
        if view is not None:
            shown["frame"] = shown.get("frame", 0) + 1
            if shown["frame"] % 2 == 0:  # 15 frames a second is plenty
                view.draw()

        if status.cget("text") != s.message:
            status.config(text=s.message)

    refresh()
    return root, refresh
