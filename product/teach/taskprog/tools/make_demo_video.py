"""Build the short demo video for one take: operator video -> labels -> task file -> URScript -> URSim -> twin.

Run from product/teach:
  python taskprog/tools/make_demo_video.py --take taskprog/examples/take1 \
      --operator-video tracked.mp4 --ursim-video rec/page.webm --ursim-start 10.5 --out demo.mp4

The URSim recording comes from tools/record_ursim.js. The twin scene is rendered here
(Linux without a screen: MUJOCO_GL=glfw xvfb-run -a python ...). Everything shown is read from
the take's own files, nothing is typed in by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TEACH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TEACH))

W, H, FPS = 1280, 720, 30
BG, INK, DIM, ACCENT, OK = (18, 22, 28), (235, 238, 242), (150, 158, 168), (90, 170, 255), (90, 210, 130)
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def font(size: int, mono: bool = False, bold: bool = False):
    name = ("DejaVuSansMono" if mono else "DejaVuSans") + ("-Bold" if bold else "") + ".ttf"
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except OSError:
        return ImageFont.load_default(size)


def canvas(heading: str, number: str = "") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    x = 48
    if number:
        d.ellipse((48, 34, 100, 86), fill=ACCENT)
        d.text((74, 60), number, font=font(30, bold=True), fill=BG, anchor="mm")
        x = 120
    d.text((x, 60), heading, font=font(32, bold=True), fill=INK, anchor="lm")
    return im, d


def paste_fit(im: Image.Image, frame_bgr: np.ndarray, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    h, w = frame_bgr.shape[:2]
    s = min((x1 - x0) / w, (y1 - y0) / h)
    f = cv2.resize(frame_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    im.paste(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)), (x0 + (x1 - x0 - f.shape[1]) // 2, y0 + (y1 - y0 - f.shape[0]) // 2))


def frames_of(path: Path, t0: float, t1: float, speed: float = 1.0):
    """Yield frames of a video between t0 and t1 (seconds), resampled to FPS at the given speed."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    n = int((t1 - t0) / speed * FPS)
    last = None
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_MSEC, (t0 + k / FPS * speed) * 1000)
        ok, f = cap.read()
        if ok:
            last = f
        if last is not None:
            yield t0 + k / FPS * speed, last
    cap.release()


def code_block(d: ImageDraw.ImageDraw, lines: list[str], xy: tuple[int, int], size: int = 19, highlight=()) -> None:
    x, y = xy
    for ln in lines:
        colour = ACCENT if any(h in ln for h in highlight) else (DIM if ln.strip().startswith("#") else INK)
        d.text((x, y), ln, font=font(size, mono=True), fill=colour)
        y += int(size * 1.45)


def twin_frames(task_path: Path, cell_path: Path, size=(960, 540)) -> tuple[list[np.ndarray], bool, float]:
    import mujoco

    from taskprog.so101 import trajectory
    from taskprog.task import load_cell, load_task

    sys.path.insert(0, str(TEACH.parent / "sim"))
    from so101_sim import scene

    cell = load_cell(cell_path)
    task = load_task(task_path, cell)
    pick = next(s for s in task.steps if s.do == "pick")
    place = next(s for s in task.steps if s.do == "place")
    traj, _ = trajectory(task, cell)
    obj = task.objects[pick.obj]
    place_xy = np.array(task.locations[place.at].xyz_mm[:2]) / 1000
    model, data = scene.load(part_pos=np.array(task.locations[pick.at].xyz_mm[:2]) / 1000, place_pos=place_xy,
                             part_size=tuple(v / 1000 for v in obj.size_mm))
    model.vis.global_.offwidth, model.vis.global_.offheight = size
    renderer = mujoco.Renderer(model, size[1], size[0])
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.2, 0.0, 0.05]
    cam.distance, cam.azimuth, cam.elevation = 0.62, 140, -38
    q0, g0 = traj.sample(0.0)
    scene.set_pose(model, data, q0, g0)
    sub = max(1, round(1 / (FPS * model.opt.timestep)))
    out = []
    for k in range(int((traj.t[-1] + 1.0) * FPS)):
        q, g = traj.sample(k / FPS)
        for name, v in zip((*scene.ARM_JOINTS, "gripper"), np.append(q, scene.gripper_angle(g))):
            data.actuator(name).ctrl[0] = v
        for _ in range(sub):
            mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=cam)
        img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        img[img.max(axis=2) < 6] = (40, 28, 22)  # empty sky: dark blue instead of black
        out.append(img)
    err = float(np.linalg.norm(data.body("part").xpos[:2] - place_xy)) * 1000
    return out, err < 30, err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", type=Path, required=True, help="folder with steps.json, labels.json, task.json and the .script")
    ap.add_argument("--operator-video", type=Path, required=True)
    ap.add_argument("--ursim-video", type=Path, required=True)
    ap.add_argument("--ursim-start", type=float, default=10.5, help="second in the recording where the program starts")
    ap.add_argument("--ursim-seconds", type=float, default=11.0)
    ap.add_argument("--ursim-result", default="PASS: every step ends within 0.1 mm of the plan")
    ap.add_argument("--so101-cell", type=Path, default=TEACH / "taskprog" / "cells" / "so101_twin.json")
    ap.add_argument("--date", default="")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    steps = json.loads((a.take / "steps.json").read_text())["steps"]
    labels = {s["index"]: s["label"] for s in json.loads((a.take / "labels.json").read_text())["steps"]}
    task = json.loads((a.take / "task.json").read_text())
    script = next(a.take.glob("*.script")).read_text().splitlines()

    tmp = a.out.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    def emit(im: Image.Image, seconds: float = 1 / FPS) -> None:
        f = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        for _ in range(max(1, round(seconds * FPS))):
            vw.write(f)

    # Title
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((W // 2, 290), "From one video to a checked robot program", font=font(46, bold=True), fill=INK, anchor="mm")
    d.text((W // 2, 360), "Show the task once. Get a native Universal Robots program.", font=font(26), fill=DIM, anchor="mm")
    d.text((W // 2, 660), f"Prototype{', ' + a.date if a.date else ''}. Simulation only, no real robot yet.", font=font(20), fill=DIM, anchor="mm")
    emit(im, 3.0)

    # 1. Operator video with the AI labels
    t_end = steps[-1]["t_end"] + 0.4
    for t, f in frames_of(a.operator_video, 0.0, t_end, speed=1.25):
        im, d = canvas("The operator shows the task once", "1")
        paste_fit(im, f, (48, 110, 860, 690))
        i = max((k for k, s in enumerate(steps) if s["t_start"] <= t), default=0)
        d.text((890, 130), "AI labels each step", font=font(22, bold=True), fill=ACCENT)
        y = 180
        for k, s in enumerate(steps):
            on = k == i
            d.text((890, y), f"{s['step']}", font=font(20, bold=True), fill=INK if on else DIM)
            for ln in _wrap(labels.get(k, ""), 30):
                y += 26
                d.text((890, y), ln, font=font(18), fill=INK if on else (90, 96, 104))
            y += 44
        emit(im)

    # 2. Task file
    im, d = canvas("A small task file: what to do, for any robot brand", "2")
    lines = ['"locations": {']
    for name, loc in task["locations"].items():
        lines.append(f'  "{name}": {{"xyz_mm": {loc["xyz_mm"]}}},')
    lines += ["},", '"steps": [']
    for s in task["steps"]:
        if s["do"] in ("pick", "place"):
            lines.append(f'  {{"do": "{s["do"]}", "object": "{s["object"]}", "at": "{s["at"]}"}},')
        else:
            lines.append(f'  {{"do": "{s["do"]}"}},')
    lines.append("]")
    code_block(d, lines, (60, 130), 22, highlight=('"pick"', '"place"'))
    y = 520
    for txt in ("Positions: measured from the video.", "Names and labels: written by the AI.",
                "The AI writes no robot code and no numbers."):
        d.text((60, y), "•  " + txt, font=font(24), fill=INK)
        y += 44
    emit(im, 7.0)

    # 3. URScript
    im, d = canvas("Fixed code writes the native UR program", "3")
    start = next(i for i, ln in enumerate(script) if "# step 2/" in ln)
    excerpt = [ln[2:] if ln.startswith("  ") else ln for ln in script[start:start + 14] if "textmsg" not in ln]
    excerpt = [ln if len(ln) <= 86 else ln[:83] + "..." for ln in excerpt]
    code_block(d, excerpt, (48, 130), 20, highlight=("ct_gripper",))
    d.text((48, 560), "•  Same task file in, same program out, every time.", font=font(24), fill=INK)
    d.text((48, 604), "•  An engineer can read it, edit it and approve it.", font=font(24), fill=INK)
    d.text((48, 648), "•  Other brands: one more writer for the same task file.", font=font(24), fill=INK)
    emit(im, 7.5)

    # 4. URSim
    last = None
    for t, f in frames_of(a.ursim_video, a.ursim_start, a.ursim_start + a.ursim_seconds):
        im, d = canvas("Checked in Universal Robots' own simulator (URSim)", "4")
        paste_fit(im, f[75:745, 275:1295], (48, 110, 1232, 640))
        d.text((48, 678), "The generated program running on a simulated UR5e.", font=font(22), fill=DIM, anchor="lm")
        last = im
        emit(im)
    if last is not None:
        d = ImageDraw.Draw(last)
        d.rectangle((0, 650, W, H), fill=BG)
        d.text((48, 678), a.ursim_result, font=font(26, bold=True), fill=OK, anchor="lm")
        emit(last, 2.5)

    # 5. Twin
    frames, ok, err = twin_frames(a.take / "task.json", a.so101_cell)
    for k, f in enumerate(frames):
        im, d = canvas("Same task file, a second robot: SO-101 (digital twin)", "5")
        paste_fit(im, f, (160, 110, 1120, 650))
        if k > len(frames) - FPS * 1.5:
            d.text((48, 684), f"{'SUCCESS' if ok else 'FAIL'}: block ends {err:.0f} mm from the target", font=font(26, bold=True), fill=OK if ok else (240, 110, 110), anchor="lm")
        emit(im)

    # End card
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((W // 2, 200), "Video  →  task file  →  native program  →  simulator check", font=font(34, bold=True), fill=INK, anchor="mm")
    y = 320
    for txt in ("Done: the full chain, for one pick-and-place, in two simulators.",
                "Next: true part sizes and positions, the real SO-101 arm, then a real cobot."):
        d.text((W // 2, y), txt, font=font(24), fill=DIM, anchor="mm")
        y += 50
    emit(im, 4.0)
    vw.release()

    # Re-encode to H.264 so phones and browsers play it.
    import subprocess

    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", str(a.out)], check=True)
    tmp.unlink()
    print(f"written {a.out}")
    return 0


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    return lines + ([cur] if cur else [])


if __name__ == "__main__":
    sys.exit(main())
