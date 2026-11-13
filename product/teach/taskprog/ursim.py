"""Run a generated URScript program in URSim (UR's free simulator) and check what happened.

Start URSim first (Docker, see README):
  docker run --rm -d --name ursim -e ROBOT_MODEL=UR5E -p 29999:29999 -p 30001-30004:30001-30004 -p 6080:6080 universalrobots/ursim_e-series

This script powers the simulated robot on, releases the brakes, sends the program on port
30002, then listens on the primary interface (port 30001) for the program's own messages
("CT step N done <pose>"), runtime errors and safety stops. It then compares the pose after each
step with the pose the plan expects. The robot only exists in the simulator.
"""

from __future__ import annotations

import re
import socket
import struct
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimResult:
    finished: bool
    errors: list[str] = field(default_factory=list)
    step_poses: dict[int, list[float]] = field(default_factory=dict)  # step id -> [x,y,z,rx,ry,rz]
    messages: list[str] = field(default_factory=list)
    seconds: float = 0.0


def dashboard(host: str, cmd: str, timeout: float = 5.0) -> str:
    with socket.create_connection((host, 29999), timeout=timeout) as s:
        s.recv(4096)  # greeting
        s.sendall((cmd + "\n").encode())
        return s.recv(4096).decode(errors="replace").strip()


def wait_mode(host: str, mode: str, timeout: float = 60.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            if mode in dashboard(host, "robotmode"):
                return True
        except OSError:
            pass
        time.sleep(1.0)
    return False


def prepare(host: str) -> None:
    """Power on and release brakes (simulator only)."""
    for _ in range(60):  # URSim needs about 30-60 s after start before the dashboard answers
        try:
            dashboard(host, "robotmode")
            break
        except OSError:
            time.sleep(2.0)
    if "RUNNING" in dashboard(host, "robotmode"):
        return
    dashboard(host, "close safety popup")
    dashboard(host, "unlock protective stop")
    dashboard(host, "power on")
    if not wait_mode(host, "IDLE", 60):
        raise RuntimeError("URSim did not power on")
    dashboard(host, "brake release")
    if not wait_mode(host, "RUNNING", 60):
        raise RuntimeError("URSim did not release the brakes")


def _robot_messages(buf: bytes):
    """Yield (kind, text) from primary interface packets in buf; return the unused tail."""
    out = []
    while len(buf) >= 5:
        size, ptype = struct.unpack(">iB", buf[:5])
        if size < 5 or size > 1_000_000:
            return out, b""  # out of sync; drop and resync on the next packet
        if len(buf) < size:
            break
        body, buf = buf[5:size], buf[size:]
        if ptype != 20 or len(body) < 10:
            continue
        mtype = body[9]
        rest = body[10:]
        if mtype == 0:
            out.append(("text", rest.decode(errors="replace")))
        elif mtype == 10 and len(rest) >= 8:
            line, col = struct.unpack(">ii", rest[:8])
            out.append(("runtime", f"line {line} col {col}: {rest[8:].decode(errors='replace')}"))
        elif mtype == 6 and len(rest) >= 17:
            code, arg, level = struct.unpack(">iii", rest[:12])
            out.append(("error" if level >= 2 else "info", f"C{code}A{arg}: {rest[17:].decode(errors='replace')}"))
        elif mtype == 7 and len(rest) >= 9:
            n = rest[8]
            out.append(("key", (rest[9:9 + n] + b" " + rest[9 + n:]).decode(errors="replace")))
        elif mtype == 5:
            out.append(("safety", "safety mode message"))
    return out, buf


POSE_RE = re.compile(r"CT step (\d+) done p\[([^\]]+)\]")


def run(host: str, program: str, timeout: float = 300.0) -> SimResult:
    prepare(host)
    res = SimResult(finished=False)
    prim = socket.create_connection((host, 30001), timeout=2.0)
    t0 = time.time()
    try:
        # The primary interface streams robot state 10 times a second. Keep reading it so the
        # buffer stays in sync; robot messages sent before our program are ignored below.
        with socket.create_connection((host, 30002), timeout=5.0) as s:
            s.sendall(program.encode())
        buf = b""
        started = False
        while time.time() - t0 < timeout:
            try:
                chunk = prim.recv(65536)
            except socket.timeout:
                chunk = b""
            buf += chunk
            msgs, buf = _robot_messages(buf)
            for kind, text in msgs:
                if kind == "text":
                    res.messages.append(text)
                    started = True
                    m = POSE_RE.search(text)
                    if m:
                        res.step_poses[int(m.group(1))] = [float(v) for v in m.group(2).split(",")]
                    if "CT program finished" in text:
                        res.finished = True
                    if "CT ERROR" in text:
                        res.errors.append(text)
                elif kind in ("runtime", "error", "safety"):
                    res.errors.append(f"{kind}: {text}")
                elif kind == "key" and ("compile" in text.lower() or "error" in text.lower()):
                    res.errors.append(f"key: {text}")
            if res.finished or res.errors:
                break
            if not started and time.time() - t0 > 15:
                state = dashboard(host, "running")
                if "false" in state:
                    res.errors.append("the program did not start (compile error? see URSim's log)")
                    break
        safety = dashboard(host, "safetystatus")
        if "NORMAL" not in safety:
            res.errors.append(safety)
    finally:
        prim.close()
        res.seconds = time.time() - t0
    return res


def compare(res: SimResult, expected: dict[int, np.ndarray], tol_mm: float = 1.0) -> list[str]:
    """Pose after each step in URSim vs the plan. Returns problems (empty = all within tol_mm)."""
    problems = []
    for step, pose in expected.items():
        got = res.step_poses.get(step)
        if got is None:
            problems.append(f"step {step}: no pose reported")
            continue
        d = np.linalg.norm(np.array(got[:3]) - pose[:3, 3]) * 1000
        if d > tol_mm:
            problems.append(f"step {step}: ends {d:.1f} mm from the planned pose")
    return problems
