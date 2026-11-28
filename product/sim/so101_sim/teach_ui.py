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


def make_panel(s, cmds: queue.Queue, arm, play_speed: float, max_speed: float):
    try:
        root = tk.Tk()
    except tk.TclError:
        return None
    root.title("Robotonic  |  SO-101 Teach Pendant")
    root.configure(bg=BG)
    root.attributes("-topmost", True)
    root.minsize(900, 680)
    root.geometry("1000x720")

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
    body.columnconfigure(1, weight=1)
    body.rowconfigure(0, weight=1)
    left = tk.Frame(body, bg=BG, width=270)
    left.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
    right = tk.Frame(body, bg=BG)
    right.grid(row=0, column=1, sticky="nsew")

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

    c = _card(left, "Position")
    c.pack(fill="x")
    lo, hi = np.degrees(s.chain.lower), np.degrees(s.chain.upper)
    joint_rows = []
    for i, nm in enumerate(JOINT_NAMES):
        r = tk.Frame(c.body, bg=CARD)
        r.pack(fill="x", pady=2)
        tk.Label(r, text=nm, bg=CARD, fg=MUTED, font=F_TXT, width=10, anchor="w").pack(side="left")
        bar = tk.Canvas(r, width=110, height=6, bg=BTN, highlightthickness=0)
        bar.pack(side="left", padx=6)
        fill = bar.create_rectangle(0, 0, 0, 6, fill=ACCENT, width=0)
        val = tk.Label(r, text="0.0°", bg=CARD, fg=TEXT, font=F_MONO, width=7, anchor="e")
        val.pack(side="right")
        joint_rows.append((bar, fill, val))
    r = tk.Frame(c.body, bg=CARD)
    r.pack(fill="x", pady=(6, 0))
    tk.Label(r, text="Gripper", bg=CARD, fg=MUTED, font=F_TXT, width=10, anchor="w").pack(side="left")
    grip_val = tk.Label(r, text="", bg=CARD, fg=TEXT, font=F_MONO, anchor="e")
    grip_val.pack(side="right")
    r = tk.Frame(c.body, bg=CARD)
    r.pack(fill="x", pady=(2, 0))
    tk.Label(r, text="Tool mm", bg=CARD, fg=MUTED, font=F_TXT, width=10, anchor="w").pack(side="left")
    tool_val = tk.Label(r, text="", bg=CARD, fg=TEXT, font=F_MONO, anchor="e")
    tool_val.pack(side="right")

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
    for col, txt, w, anchor in (("n", "#", 36, "center"), ("cmd", "COMMAND", 90, "w"), ("detail", "JOINTS  base → wrist roll", 300, "w"),
                                 ("tool", "TOOL x, y, z mm", 130, "w")):
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
    c = _card(right, "Run")
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
    status = tk.Label(bar, text="", bg=CARD, fg=TEXT, font=F_TXT, anchor="w", justify="left", wraplength=720, padx=12, pady=8)
    status.pack(fill="x")
    keys = "Keys:  w / Enter record   o open   c close   u undo   Esc stop" + ("   f freedrive   Space follow" if arm else "")
    tk.Label(bar, text=keys, bg=CARD, fg=MUTED, font=(UI, 8), anchor="w", padx=12).pack(fill="x", pady=(0, 6))

    for key, cmd in (("o", "open"), ("c", "close"), ("w", "record"), ("<Return>", "record"), ("u", "undo"), ("<space>", "follow")):
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

        # live position
        q, g = s.target()
        if shown["q"] is None or np.abs(q - shown["q"]).max() > 1e-3 or shown.get("g") != round(g, 2):
            qd = np.degrees(q)
            for (bar_, fill, val), v, a, b in zip(joint_rows, qd, lo, hi):
                val.config(text=f"{v:6.1f}°")
                bar_.coords(fill, 0, 0, 110 * float(np.clip((v - a) / (b - a), 0, 1)), 6)
            grip_val.config(text=f"{g * 100:3.0f} % open")
            tool = s.chain.fk(q)[:3, 3] * 1000
            tool_val.config(text=f"{tool[0]:4.0f} {tool[1]:4.0f} {tool[2]:4.0f}")
            shown["q"], shown["g"] = q.copy(), round(g, 2)

        if status.cget("text") != s.message:
            status.config(text=s.message)

    refresh()
    return root, refresh
