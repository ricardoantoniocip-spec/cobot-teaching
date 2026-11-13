"""Find grasp and release times from where the OBJECT moves, not from the hand's shape.

Why: a whole-hand grasp looks like a relaxed hand to the hand tracker (take 1), so pinch
distance cannot tell when the object is held. But the object itself is unambiguous: it is at
place A in the first frames, at place B in the last frames, and gone from A in between.

How (fixed camera, one object moved once, no AI model):
  1. start image = median of the first half second, end image = median of the last half second.
  2. Where they differ there are blobs. The object gives two: A (there at the start, gone at the
     end) and B (the other way round). The right pair is the one where the start image at A
     looks like the end image at B (same colours).
  3. Grasp = the last moment most of A still looks like the start image (the hand may cover a
     part of the object before it lifts). Release = the first moment most of B looks like the
     end image and stays that way.

Command (from product/teach):  python -m retarget.object_motion VIDEO [--debug OUT.jpg]
It prints the times in the form the video app takes:  --grips 3.0-8.5
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

DEBUG = False
WORK_WIDTH = 320  # frames are shrunk to this width; plenty for blobs, fast on a laptop


@dataclass
class ObjectMotion:
    t_grasp: float
    t_release: float
    pick_px: tuple[float, float]  # object centre at the start, in original image pixels
    place_px: tuple[float, float]
    size_px: tuple[float, float]  # object width, height in original image pixels
    match: float  # 0..1, how alike the object looks at A (start) and B (end)
    notes: list[str]
    like_start: np.ndarray = field(repr=False, default=None)  # per frame: share of A that still shows the object
    like_end: np.ndarray = field(repr=False, default=None)  # per frame: share of B that already shows the object

    def grips_arg(self) -> str:
        return f"{self.t_grasp:.2f}-{self.t_release:.2f}"


class NoObjectMotion(ValueError):
    pass


def read_small(video: Path, t_start: float = 0.0, t_end: float | None = None) -> tuple[np.ndarray, float, float]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    scale = 1.0
    i = -1
    while True:
        ok, f = cap.read()
        i += 1
        if not ok or (t_end is not None and i / fps > t_end):
            break
        if i / fps < t_start:
            continue
        scale = WORK_WIDTH / f.shape[1]
        frames.append(cv2.GaussianBlur(cv2.resize(f, (WORK_WIDTH, int(f.shape[0] * scale))), (5, 5), 0))
    cap.release()
    if len(frames) < 10:
        raise NoObjectMotion("the video is too short")
    return np.array(frames), fps, scale


def _blobs(mask: np.ndarray, min_area: int) -> list[tuple[int, int, int, int, np.ndarray]]:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            out.append((x, y, w, h, lab == i))
    return out


def _hist(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], mask.astype(np.uint8), [18, 8], [0, 180, 0, 256])
    return cv2.normalize(h, None).flatten()


def _like(grey: np.ndarray, ref_grey: np.ndarray, box) -> np.ndarray:
    """Per frame: normalised correlation of the box with the same box in a reference picture.
    1 = looks the same, about 0 = something else. Lighting changes do not matter."""
    x, y, w, h = box
    ref = ref_grey[y:y + h, x:x + w].astype(np.float32).ravel()
    ref = ref - ref.mean()
    v = grey[:, y:y + h, x:x + w].reshape(len(grey), -1)
    v = v - v.mean(axis=1, keepdims=True)
    return (v @ ref) / (np.linalg.norm(v, axis=1) * np.linalg.norm(ref) + 1e-6)


def find_object_motion(frames: np.ndarray, fps: float, scale: float = 1.0, diff_threshold: int = 28, ignore_top: float = 0.0) -> ObjectMotion:
    n = len(frames)
    k = max(3, int(0.5 * fps))
    grey = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames], dtype=np.float32)
    win = max(1, int(0.15 * fps))
    smooth = lambda x: np.convolve(np.pad(x, win, mode="edge"), np.ones(2 * win + 1) / (2 * win + 1), mode="valid")  # noqa: E731

    # 1. Where was something at the start that is gone later? Compare the start picture with
    #    the typical picture of the second half (a median ignores a hand passing through).
    start = np.median(frames[:k], axis=0).astype(np.uint8)
    late = np.median(frames[n // 2::2], axis=0).astype(np.uint8)
    start_grey = cv2.cvtColor(start, cv2.COLOR_BGR2GRAY)
    mask = cv2.absdiff(start, late).max(axis=2) > diff_threshold
    if ignore_top:
        mask[: int(mask.shape[0] * ignore_top)] = False
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    blobs = _blobs(mask, int(0.0015 * mask.size))
    if not blobs:
        raise NoObjectMotion("nothing that stood still at the start is gone later (or the camera moved)")

    # 2. For each compact blob: take its picture from the start, wait until it has left, then
    #    look for that picture anywhere else in every later frame, at a few sizes (the object may
    #    end nearer to the camera, or partly outside the picture). The best, clean view of it
    #    gives the place B and a reference moment with the hand out of the way. Plain patches
    #    (sofa, wall) match anywhere, so patches with detail count more.
    best = None
    views: dict = {}
    for x, y, w, h, bm in blobs:
        if bm.sum() > 0.05 * mask.size or bm.sum() / (w * h) < 0.5 or w < 6 or h < 6:
            continue  # too big (the person) or too ragged to be one object
        ls = smooth(_like(grey, start_grey, (x, y, w, h)))
        gone = np.nonzero(ls < 0.35)[0]
        if not len(gone) or ls[:k].min() < 0.9:
            continue  # never leaves, or was not standing still at the start
        i_lift = int(gone[0])
        patch = start[y:y + h, x:x + w]
        detail = min(1.0, float(patch.std()) / 40.0)
        for part in (1.0, 0.7):
            tpl0 = patch[: max(4, int(h * part))]
            for sc in (0.7, 0.85, 1.0, 1.2, 1.4, 1.7):
                tpl = cv2.resize(tpl0, None, fx=sc, fy=sc)
                th, tw = tpl.shape[:2]
                if th >= start.shape[0] or tw >= start.shape[1]:
                    continue
                for i in range(i_lift, n, 2):
                    res = cv2.matchTemplate(frames[i], tpl, cv2.TM_CCOEFF_NORMED)
                    res[max(0, y - th):y + h, max(0, x - tw):x + w] = -1.0  # not on top of A
                    _, score, _, (bx, by) = cv2.minMaxLoc(res)
                    if DEBUG and score > 0.75:
                        print(f"  blob {x},{y} {w}x{h} detail {detail:.2f} part {part} scale {sc} frame {i} score {score:.2f} at {bx},{by}")
                    bh = min(int(round(th / part)), start.shape[0] - by)
                    views.setdefault((x, y), []).append((score, i, (bx, by, tw, bh)))
                    if best is None or score * detail > best[0]:
                        best = (score * detail, score, (x, y, w, h), (bx, by, tw, bh), i, i_lift, ls)
    if best is None or best[1] < 0.6 or best[0] < 0.3:
        raise NoObjectMotion("could not find an object that stands still at the start and stands somewhere else later (fixed camera? hands away from the object in the first half second?)")
    _, same, a, b, i_ref, i_lift, ls = best
    # If the object is moved more than once, use the first place it stands: the earliest view
    # that is nearly as good as the best one.
    first = min((v for v in views[a[:2]] if v[0] >= 0.92 * same), key=lambda v: v[1])
    if first[1] < i_ref:
        same, i_ref, b = first
    notes = []
    if same < 0.7:
        notes.append(f"the object looks rather different at pick and place (match {same:.2f}); check the debug picture")

    # 3. Set-down: B looks like "object standing there" (the reference moment) from when on?
    ref = np.median(grey[max(0, i_ref - win):i_ref + win + 1], axis=0)
    le = smooth(_like(grey, ref, b))
    i_down = i_ref
    while i_down > i_lift and le[i_down - 1] >= 0.5:
        i_down -= 1
    i_gone_again = i_ref
    while i_gone_again < n - 1 and le[i_gone_again + 1] >= 0.5:
        i_gone_again += 1
    if i_gone_again < n - 1 - win:
        notes.append(f"the object is moved again at about {i_gone_again / fps:.1f} s; only the first move is used")

    # Grasp just before the lift and release just after the set-down: at both moments the hand
    # is on the object and the object stands still, which is what the robot must copy.
    margin = int(round(0.2 * fps))
    i_grasp = max(0, i_lift - margin)
    i_release = min(n - 1, i_down + margin)
    if i_down <= i_lift:
        notes.append("set-down found before lift; times are unreliable")
    if i_lift < k:
        notes.append("the object moves in the first half second; start the video with your hands away from it")

    def centre(box) -> tuple[float, float]:
        return (box[0] + box[2] / 2) / scale, (box[1] + box[3] / 2) / scale

    return ObjectMotion(
        t_grasp=i_grasp / fps,
        t_release=i_release / fps,
        pick_px=centre(a),
        place_px=centre(b),
        size_px=(a[2] / scale, a[3] / scale),
        match=float(same),
        notes=notes,
        like_start=ls,
        like_end=le,
    )


def debug_image(video: Path, m: ObjectMotion, out: Path) -> None:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tiles = []
    for label, t in (("grasp", m.t_grasp), ("release", m.t_release)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, f = cap.read()
        if not ok:
            continue
        for (x, y), colour, name in ((m.pick_px, (0, 200, 255), "pick"), (m.place_px, (0, 220, 0), "place")):
            w, h = m.size_px
            cv2.rectangle(f, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), colour, 2)
            cv2.putText(f, name, (int(x - w / 2), int(y - h / 2) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        cv2.putText(f, f"{label} t = {t:.2f} s", (12, f.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        tiles.append(f)
    cap.release()
    if tiles:
        cv2.imwrite(str(out), np.hstack(tiles))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m retarget.object_motion", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--debug", type=Path, help="write a picture of the grasp and release frames with the pick and place boxes")
    p.add_argument("--ignore-top", type=float, default=0.0, help="ignore this fraction at the top of the image (e.g. 0.12 for a burnt-in clock)")
    p.add_argument("--start", type=float, default=0.0, help="use the video from this second (hand not on the object yet)")
    p.add_argument("--end", type=float, help="use the video up to this second (object placed, hand away)")
    a = p.parse_args(argv)
    try:
        frames, fps, scale = read_small(a.video, a.start, a.end)
        m = find_object_motion(frames, fps, scale, ignore_top=a.ignore_top)
        m.t_grasp += a.start
        m.t_release += a.start
    except NoObjectMotion as e:
        print(f"No object motion found: {e}")
        return 1
    print(f"object picked at pixel ({m.pick_px[0]:.0f}, {m.pick_px[1]:.0f}), placed at ({m.place_px[0]:.0f}, {m.place_px[1]:.0f}), "
          f"about {m.size_px[0]:.0f} x {m.size_px[1]:.0f} px, match {m.match:.2f}")
    print(f"grasp at {m.t_grasp:.2f} s, release at {m.t_release:.2f} s")
    for note in m.notes:
        print(f"note: {note}")
    if a.debug:
        debug_image(a.video, m, a.debug)
        print(f"debug picture: {a.debug}")
    print(f"--grips {m.grips_arg()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
