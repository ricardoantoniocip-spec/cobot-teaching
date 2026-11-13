"""Build the 40-second ad-style video for one take, with voice-over and music.

Run from product/teach (Linux without a screen: MUJOCO_GL=glfw xvfb-run -a python ...):
  python taskprog/tools/make_ad_video.py --take taskprog/examples/take1 \
      --operator-video tracked.mp4 --ursim-video rec/page.webm --ursim-start 10.5 \
      --voice-dir tts --out ad.mp4

Voice: one wav per line in --voice-dir (hook, show, ai, code, sim, twin, end), made with any
text-to-speech tool from the lines in SCRIPT below. Without --voice-dir the video has music only.
Music is synthesised here (no licence needed). All pictures and text come from the take's files.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

TEACH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TEACH))
sys.path.insert(0, str(Path(__file__).resolve().parent))

W, H, FPS = 1280, 720, 30
INK, DIM, ACCENT, OK, PANEL = (240, 243, 248), (150, 160, 178), (64, 156, 255), (70, 220, 140), (22, 28, 40)

SCRIPT = {
    "hook": "What if teaching a cobot a new task took one video?",
    "show": "Show the task once. Just do it, the way you always do.",
    "ai": "AI watches, and names every step.",
    "code": "Then fixed, tested code writes the robot's own program. Native Universal Robots script, that an engineer can read, and approve.",
    "sim": "Every program is checked in the robot maker's own simulator, before it touches a real machine.",
    "twin": "And the same task runs on other robots too.",
    "end": "Video in. Robot program out. Which task should your cobot learn first?",
}
# scene name -> (start, end) seconds; the voice line starts 0.4 s into its scene
SCENES = {"hook": (0.0, 4.2), "show": (4.2, 10.2), "ai": (10.2, 14.4), "code": (14.4, 23.9),
          "sim": (23.9, 30.6), "twin": (30.6, 34.6), "end": (34.6, 42.0)}
VOICE_DELAY = 0.4

FONTS = {
    "display": ["/usr/share/fonts/opentype/inter/InterDisplay-ExtraBold.otf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "bold": ["/usr/share/fonts/opentype/inter/Inter-SemiBold.otf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "text": ["/usr/share/fonts/opentype/inter/Inter-Medium.otf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "mono": ["/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Medium.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"],
}
_font_cache: dict = {}


def font(kind: str, size: int):
    key = (kind, size)
    if key not in _font_cache:
        for path in FONTS[kind]:
            try:
                _font_cache[key] = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
        else:
            _font_cache[key] = ImageFont.load_default(size)
    return _font_cache[key]


def ease(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def ramp(t: float, t0: float, dur: float = 0.45) -> float:
    return ease((t - t0) / dur)


def background(t: float) -> Image.Image:
    """Dark blue gradient with two slow glows."""
    yy, xx = np.mgrid[0:H:4, 0:W:4].astype(np.float32)
    base = np.zeros((H // 4, W // 4, 3), np.float32)
    base[:] = (10, 13, 22)
    for cx, cy, r, col, ph in ((0.2, 0.25, 420, (20, 60, 140), 0.0), (0.85, 0.8, 480, (10, 90, 110), 2.0)):
        x0 = (cx + 0.04 * np.sin(0.25 * t + ph)) * W
        y0 = (cy + 0.05 * np.cos(0.2 * t + ph)) * H
        g = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * r * r))
        base += g[..., None] * np.array(col, np.float32) * 0.6
    im = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    return im.convert("RGBA")


class Frame:
    def __init__(self, t: float):
        self.im = background(t)

    def text(self, xy, s: str, kind="text", size=28, fill=INK, alpha=1.0, anchor="la", dy=0.0):
        if alpha <= 0.003:
            return
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(layer).text((xy[0], xy[1] + dy), s, font=font(kind, size), fill=(*fill, int(255 * min(1, alpha))), anchor=anchor)
        self.im = Image.alpha_composite(self.im, layer)

    def rise(self, xy, s, t, t0, **kw):
        """Text that fades in and slides up from t0."""
        a = ramp(t, t0)
        self.text(xy, s, alpha=a, dy=(1 - a) * 18, **kw)

    def panel(self, box, alpha=1.0, fill=PANEL, radius=18, outline=None):
        if alpha <= 0.003:
            return
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.rounded_rectangle(box, radius, fill=(*fill, int(235 * alpha)), outline=(*(outline or (60, 72, 96)), int(255 * alpha)), width=2)
        self.im = Image.alpha_composite(self.im, layer)

    def picture(self, bgr: np.ndarray, box, alpha=1.0, radius=18, zoom=1.0, glow=ACCENT):
        """A video frame cropped to fill the box, rounded corners, soft glow behind."""
        if alpha <= 0.003:
            return
        x0, y0, x1, y1 = (int(v) for v in box)
        bw, bh = x1 - x0, y1 - y0
        h, w = bgr.shape[:2]
        s = max(bw / w, bh / h) * zoom
        f = cv2.resize(bgr, (int(w * s) + 1, int(h * s) + 1), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        ox, oy = (f.shape[1] - bw) // 2, (f.shape[0] - bh) // 2
        f = cv2.cvtColor(f[oy:oy + bh, ox:ox + bw], cv2.COLOR_BGR2RGB)
        mask = Image.new("L", (bw, bh), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, bw - 1, bh - 1), radius, fill=int(255 * alpha))
        glow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(glow_layer).rounded_rectangle((x0 - 6, y0 - 6, x1 + 6, y1 + 6), radius + 6, fill=(*glow, int(90 * alpha)))
        self.im = Image.alpha_composite(self.im, glow_layer.filter(ImageFilter.GaussianBlur(18)))
        self.im.paste(Image.fromarray(f), (x0, y0), mask)

    def chip(self, xy, s, alpha=1.0, fill=ACCENT, ink=(8, 12, 20), size=22, dy=0.0):
        if alpha <= 0.003:
            return 0
        fnt = font("bold", size)
        w = int(fnt.getlength(s)) + 28
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        x, y = xy[0], xy[1] + dy
        d.rounded_rectangle((x, y, x + w, y + size + 18), (size + 18) // 2, fill=(*fill, int(255 * alpha)))
        d.text((x + 14, y + 9 + size / 2), s, font=fnt, fill=(*ink, int(255 * alpha)), anchor="lm")
        self.im = Image.alpha_composite(self.im, layer)
        return w

    def headline(self, number: str, s: str, t: float, t0: float):
        a = ramp(t, t0)
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.ellipse((56, 44, 112, 100), fill=(*ACCENT, int(255 * a)))
        d.text((84, 72), number, font=font("display", 32), fill=(8, 12, 20, int(255 * a)), anchor="mm")
        d.text((132, 72 + (1 - a) * 14), s, font=font("display", 44), fill=(*INK, int(255 * a)), anchor="lm")
        self.im = Image.alpha_composite(self.im, layer)

    def fade(self, amount: float):
        if amount > 0.003:
            black = Image.new("RGBA", (W, H), (6, 8, 14, int(255 * min(1, amount))))
            self.im = Image.alpha_composite(self.im, black)

    def bgr(self) -> np.ndarray:
        return cv2.cvtColor(np.array(self.im.convert("RGB")), cv2.COLOR_RGB2BGR)


class Clip:
    """Random access to a video's frames by time, cached in memory at a small size."""

    def __init__(self, path: Path, t0: float, t1: float, crop=None, max_w: int = 960):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(path)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.t0 = t0
        cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
        self.frames = []
        while True:
            ok, f = cap.read()
            if not ok or cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 > t1 + 0.2:
                break
            if crop:
                x0, y0, x1, y1 = crop
                f = f[y0:y1, x0:x1]
            if f.shape[1] > max_w:
                f = cv2.resize(f, (max_w, int(f.shape[0] * max_w / f.shape[1])), interpolation=cv2.INTER_AREA)
            self.frames.append(f)
        cap.release()

    def at(self, t: float) -> np.ndarray:
        i = int(round((t - self.t0) * self.fps))
        return self.frames[min(max(i, 0), len(self.frames) - 1)]


# ------------------------------------------------------------------ audio

SR = 44100


def _note(freq: float, dur: float, kind: str) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    if kind == "pad":
        x = sum(np.sin(2 * np.pi * freq * m * t + p) / m ** 1.6 for m, p in ((1, 0), (2, 0.5), (3, 1.1))) + 0.5 * np.sin(2 * np.pi * freq * 1.005 * t)
        env = np.minimum(1, t / 0.6) * np.minimum(1, (dur - t) / 0.8)
        return x * env
    if kind == "pluck":
        return (np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t)) * np.exp(-t * 7)
    if kind == "kick":
        f = 45 + 70 * np.exp(-t * 28)
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9)
    raise ValueError(kind)


def music(total: float) -> np.ndarray:
    """Calm, forward-moving bed: four chords, a plucked pattern, a soft pulse."""
    out = np.zeros(int(total * SR) + SR)
    bar = 2.4  # seconds per chord (100 bpm, 4 beats)
    a = 110.0
    semis = lambda n: a * 2 ** (n / 12)  # noqa: E731
    chords = [(0, 7, 12, 15), (-4, 3, 8, 12), (3, 10, 15, 19), (-2, 5, 10, 14)]  # Am F C G
    n_bars = int(total / bar) + 1
    for b in range(n_bars):
        ch = chords[b % 4]
        t0 = b * bar
        for n in ch:
            x = _note(semis(n), bar + 0.6, "pad") * 0.10
            i = int(t0 * SR)
            out[i:i + len(x)] += x[: len(out) - i]
        if 4.2 <= t0 < 35.0:  # movement only in the middle of the video
            for k in range(8):
                n = ch[(k * 2 + (k // 3)) % 4] + 12
                x = _note(semis(n), 0.5, "pluck") * 0.10
                i = int((t0 + k * bar / 8) * SR)
                out[i:i + len(x)] += x[: len(out) - i]
            for k in range(4):
                x = _note(0, 0.35, "kick") * (0.32 if k % 2 == 0 else 0.18)
                i = int((t0 + k * bar / 4) * SR)
                out[i:i + len(x)] += x[: len(out) - i]
    out = out[: int(total * SR)]
    t = np.arange(len(out)) / SR
    return out * np.minimum(1, t / 1.5) * np.minimum(1, (total - t) / 2.5)


def read_wav(path: Path) -> np.ndarray:
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        n = int(len(x) * SR / sr)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    return x / (np.abs(x).max() + 1e-6)


def soundtrack(total: float, voice_dir: Path | None) -> np.ndarray:
    voice = np.zeros(int(total * SR))
    if voice_dir:
        for name, (t0, _) in SCENES.items():
            p = voice_dir / f"{name}.wav"
            if p.exists():
                x = read_wav(p) * 0.85
                i = int((t0 + VOICE_DELAY) * SR)
                voice[i:i + len(x)] += x[: len(voice) - i]
    # Duck the music under the voice.
    env = np.abs(voice)
    win = int(0.25 * SR)
    env = np.convolve(env, np.ones(win) / win, mode="same")
    gain = 0.95 - 0.72 * np.clip(env / 0.05, 0, 1)
    gain = np.convolve(gain, np.ones(win) / win, mode="same")
    mix = voice + music(total) * gain * (0.55 if voice_dir else 0.9)
    return np.clip(mix / max(1.0, np.abs(mix).max() / 0.97), -1, 1)


def write_wav(path: Path, x: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).astype(np.int16).tobytes())


# ------------------------------------------------------------------ scenes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", type=Path, required=True)
    ap.add_argument("--operator-video", type=Path, required=True)
    ap.add_argument("--ursim-video", type=Path, required=True)
    ap.add_argument("--ursim-start", type=float, default=10.5)
    ap.add_argument("--ursim-seconds", type=float, default=11.0)
    ap.add_argument("--voice-dir", type=Path)
    ap.add_argument("--so101-cell", type=Path, default=TEACH / "taskprog" / "cells" / "so101_twin.json")
    ap.add_argument("--footer", default="Cobot teaching by demonstration  ·  prototype, shown in simulation")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from make_demo_video import twin_frames

    steps = json.loads((a.take / "steps.json").read_text())["steps"]
    labels_doc = json.loads((a.take / "labels.json").read_text())
    labels = {s["index"]: s for s in labels_doc["steps"]}
    task = json.loads((a.take / "task.json").read_text())
    script = next(a.take.glob("*.script")).read_text().splitlines()
    t_demo_end = steps[-1]["t_end"] + 0.3

    operator = Clip(a.operator_video, 0.0, t_demo_end)
    ursim = Clip(a.ursim_video, a.ursim_start, a.ursim_start + a.ursim_seconds, crop=(278, 180, 905, 640))
    twin, twin_ok, twin_err = twin_frames(a.take / "task.json", a.so101_cell)

    i0 = next(i for i, ln in enumerate(script) if "# step 2/" in ln)
    code = [ln[2:].split("  #")[0] for ln in script[i0:i0 + 13] if "textmsg" not in ln and ln.strip()]
    code = [ln if len(ln) <= 62 else ln[:59] + "..." for ln in code]
    task_lines = [f'{s["do"]} @ {s["at"]}' if "at" in s else s["do"] for s in task["steps"]]
    key_steps = [(i, s) for i, s in enumerate(steps) if s["step"] in ("reach", "grasp", "move", "release")]

    total = SCENES["end"][1]
    tmp = a.out.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    for k in range(int(total * FPS)):
        t = k / FPS
        fr = Frame(t)
        name = next(n for n, (s0, s1) in SCENES.items() if s0 <= t < s1)
        s0, s1 = SCENES[name]
        u = t - s0

        if name == "hook":
            fr.rise((W // 2, 290), "What if teaching a cobot", t, 0.3, kind="display", size=68, anchor="mm")
            a2 = ramp(t, 1.5)
            w1, w2 = font("display", 68).getlength("took "), font("display", 68).getlength("one video?")
            x = (W - w1 - w2) / 2
            fr.text((x, 380), "took ", kind="display", size=68, alpha=a2, anchor="lm", dy=(1 - a2) * 18)
            fr.text((x + w1, 380), "one video?", kind="display", size=68, fill=ACCENT, alpha=a2, anchor="lm", dy=(1 - a2) * 18)
        elif name == "show":
            fr.headline("1", "Show it once.", t, s0 + 0.1)
            tv = min(t_demo_end, u / (s1 - s0 - 0.3) * t_demo_end)
            g = ramp(t, s0 + 0.1, 0.6)
            fr.picture(operator.at(tv), (150, 130, 1130, 680), alpha=g, zoom=1.0 + 0.04 * u / (s1 - s0))
            fr.chip((174, 150), "one take, an ordinary webcam", alpha=ramp(t, s0 + 1.2), fill=(12, 16, 26), ink=INK, size=20)
        elif name == "ai":
            fr.headline("2", "AI names every step.", t, s0 + 0.1)
            n_on = min(len(key_steps), int(u / 0.75) + 1)
            i_now, s_now = key_steps[n_on - 1]
            fr.picture(operator.at((s_now["t_start"] + s_now["t_end"]) / 2 if s_now["t_end"] > s_now["t_start"] else s_now["t_start"]), (64, 150, 700, 620))
            y = 160
            for j, (i, s) in enumerate(key_steps):
                al = ramp(t, s0 + 0.15 + j * 0.75, 0.35)
                lab = labels.get(i, {})
                fr.panel((740, y, 1216, y + 100), alpha=al, outline=ACCENT if j == n_on - 1 else None)
                fr.text((764, y + 18), s["step"].upper(), kind="bold", size=20, fill=ACCENT, alpha=al)
                where = f"  ·  {lab['location']}" if lab.get("location") else ""
                fr.text((764 + font("bold", 20).getlength(s["step"].upper()), y + 18), where, kind="bold", size=20, fill=OK, alpha=al)
                words, rows, cur = lab.get("label", "").split(), [], ""
                for wd in words:
                    if len(cur) + len(wd) + 1 > 46:
                        rows.append(cur)
                        cur = wd
                    else:
                        cur = (cur + " " + wd).strip()
                for r_i, row in enumerate((rows + [cur])[:2]):
                    fr.text((764, y + 48 + r_i * 24), row, size=18, fill=INK, alpha=al)
                y += 116
        elif name == "code":
            fr.headline("3", "Code writes the robot's own program.", t, s0 + 0.1)
            al = ramp(t, s0 + 0.2)
            fr.panel((56, 150, 400, 470), alpha=al)
            fr.text((80, 172), "task file", kind="bold", size=20, fill=DIM, alpha=al)
            for j, ln in enumerate(task_lines):
                fr.text((80, 216 + j * 44), ln if len(ln) < 26 else ln[:24] + "…", kind="mono", size=19, fill=ACCENT if ln.startswith(("pick", "place")) else INK, alpha=ramp(t, s0 + 0.4 + j * 0.2))
            fr.text((428, 300), "→", kind="display", size=54, fill=ACCENT, alpha=ramp(t, s0 + 1.3), anchor="mm")
            ar = ramp(t, s0 + 1.5)
            fr.panel((456, 150, 1224, 560), alpha=ar, fill=(12, 15, 24))
            fr.text((480, 172), "URScript  ·  Universal Robots", kind="bold", size=20, fill=DIM, alpha=ar)
            typed = int(max(0.0, u - 1.9) * 95)  # characters per second
            y = 214
            for ln in code:
                if typed <= 0:
                    break
                shown = ln[:typed]
                typed -= len(ln)
                col = DIM if ln.startswith("#") else (OK if "ct_gripper" in ln else INK)
                fr.text((480, y), shown, kind="mono", size=17, fill=col)
                y += 30
            for j, (txt, at) in enumerate((("readable", 5.6), ("same result every time", 6.3), ("engineer approves", 7.0))):
                fr.chip((56 + j * 0 + sum(int(font("bold", 22).getlength(x)) + 44 for x in ("readable", "same result every time")[:j]), 600), txt, alpha=ramp(t, s0 + at), dy=(1 - ramp(t, s0 + at)) * 12)
        elif name == "sim":
            fr.headline("4", "Checked before it touches a robot.", t, s0 + 0.1)
            tv = a.ursim_start + min(1.0, u / (s1 - s0 - 1.6)) * a.ursim_seconds
            fr.picture(ursim.at(tv), (64, 140, 700, 606), glow=(120, 140, 170))
            fr.chip((84, 158), "URSim  ·  simulated UR5e", fill=(12, 16, 26), ink=INK, size=20)
            names = [s["do"] + (f" {s['at']}" if "at" in s else "") for s in task["steps"]]
            y = 160
            for j, nm in enumerate(names):
                done = u > (j + 1) / len(names) * (s1 - s0 - 1.8)
                fr.panel((740, y, 1216, y + 70), outline=OK if done else None)
                fr.text((766, y + 35), nm, kind="mono", size=22, fill=INK, anchor="lm")
                fr.text((1190, y + 35), "✓" if done else "…", kind="bold", size=28, fill=OK if done else DIM, anchor="rm")
                y += 84
            ap_ = ramp(t, s1 - 1.7)
            fr.chip((740, y + 14), "PASS  ·  within 0.1 mm of the plan", alpha=ap_, fill=OK, size=24, dy=(1 - ap_) * 12)
        elif name == "twin":
            fr.headline("5", "Same task. Other robots.", t, s0 + 0.1)
            fr.picture(twin[min(len(twin) - 1, int(u / (s1 - s0) * len(twin)))], (190, 140, 1090, 646))
            fr.chip((214, 160), "SO-101  ·  digital twin", fill=(12, 16, 26), ink=INK, size=20)
            if twin_ok:
                ap_ = ramp(t, s1 - 1.2)
                fr.chip((214, 580), "SUCCESS", alpha=ap_, fill=OK, size=24)
        elif name == "end":
            fr.rise((W // 2, 230), "Video in.", t, s0 + 0.3, kind="display", size=76, anchor="mm")
            fr.rise((W // 2, 325), "Robot program out.", t, s0 + 1.3, kind="display", size=76, fill=ACCENT, anchor="mm")
            fr.rise((W // 2, 460), "Which task should your cobot learn first?", t, s0 + 2.6, kind="bold", size=36, anchor="mm")
            fr.text((W // 2, 660), a.footer, size=20, fill=DIM, alpha=ramp(t, s0 + 3.4), anchor="mm")

        # soft cut between scenes, fade in and out at the ends
        edge = min(u, s1 - t)
        fr.fade(max(0.0, 1 - edge / 0.18) * 0.85 if name != "hook" or u > 1 else 0.0)
        fr.fade(max(0.0, 1 - t / 0.5))
        fr.fade(max(0.0, 1 - (total - t) / 0.8))
        vw.write(fr.bgr())
    vw.release()

    wav = a.out.with_suffix(".wav")
    write_wav(wav, soundtrack(total, a.voice_dir))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-i", str(wav), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                    "-af", "loudnorm=I=-16:TP=-1.5:LRA=7", "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-shortest", "-movflags", "+faststart", str(a.out)], check=True)
    tmp.unlink()
    wav.unlink()
    print(f"written {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
