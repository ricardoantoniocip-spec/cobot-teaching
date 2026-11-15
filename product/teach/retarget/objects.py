"""Find the object in the video by NAME, follow it, and read pick and place from where it goes.

Why: the hand alone is a weak signal. A whole-hand grip on a wide part looks like an open hand
(take 1), and a plain picture comparison (object_motion.py) only works when the object stands
out from the background. An open-vocabulary detector finds "juice carton" (or any part you can
name) in every picture, also while your hand is on it.

How:
  1. Every half second, a detector looks for the object. With --object "juice carton" it looks
     for exactly that; without a name it looks for common things (box, bottle, part, tool ...)
     and takes the one that goes from one resting place to another.
  2. The boxes are linked into one track. Where the object stands still it "rests"; between two
     resting places it is being moved. Each move is one pick and place (so a part that is moved
     twice gives two).
  3. The exact moments come from every frame, not only the half-second ones: the grasp is when
     the resting place stops looking like the object, the release is when the new place starts
     looking like it and stays so (the same test object_motion.py uses).
  4. If the hand track is given, each grasp is checked: was the tracked hand on the object?

Model: LLMDet tiny (Apache-2.0, 173 M parameters) through Hugging Face transformers. Any
Grounding-DINO-style model id works (--model). Needs `pip install -r requirements-vision.txt`.
About 5 s per picture on a 4-core CPU, so about 3 minutes for a 17 s video; much faster on a GPU.

Command (from product/teach):
  python -m retarget.objects VIDEO --object "juice carton" [--debug OUT.jpg]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .object_motion import NoObjectMotion, _like

DEFAULT_MODEL = "iSEE-Laboratory/llmdet_tiny"
# Looked for when no name is given. Plain words work best with these detectors.
GENERIC_NAMES = ["box", "carton", "bottle", "can", "cup", "block", "part", "tool", "bag", "object"]
NOT_OBJECTS = {"hand", "person", "arm", "man", "woman"}
EVERY_S = 0.5
MIN_SCORE = 0.3


@dataclass
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in image pixels

    @property
    def centre(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])

    @property
    def size(self) -> np.ndarray:
        return np.array([self.box[2] - self.box[0], self.box[3] - self.box[1]])


@dataclass
class Move:
    """One pick and place of the object."""

    t_grasp: float
    t_release: float
    pick_box: tuple[float, float, float, float]
    place_box: tuple[float, float, float, float]
    hand_on_object: bool | None = None  # None = no hand track was given

    @property
    def pick_px(self) -> tuple[float, float]:
        """Where the object stands: bottom centre of its box (the point on the surface)."""
        return (self.pick_box[0] + self.pick_box[2]) / 2, self.pick_box[3]

    @property
    def place_px(self) -> tuple[float, float]:
        return (self.place_box[0] + self.place_box[2]) / 2, self.place_box[3]


@dataclass
class ObjectTrack:
    name: str  # what the detector called it most often (or the name you gave)
    score: float  # median detector score, 0..1
    times: np.ndarray  # the looked-at moments, seconds
    boxes: list[tuple[float, float, float, float] | None]  # one per moment, None = not seen
    moves: list[Move] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    seen_fraction: float = 0.0

    def grip_events(self) -> list[tuple[float, float]]:
        return [(m.t_grasp, m.t_release) for m in self.moves]

    def to_json(self) -> dict:
        r = lambda v: [round(float(x), 1) for x in v]  # noqa: E731
        return {
            "name": self.name, "score": round(self.score, 2), "seen_fraction": round(self.seen_fraction, 2),
            "moves": [{"t_grasp": round(m.t_grasp, 2), "t_release": round(m.t_release, 2), "pick_box_px": r(m.pick_box),
                       "place_box_px": r(m.place_box), "pick_px": r(m.pick_px), "place_px": r(m.place_px),
                       "hand_on_object": m.hand_on_object} for m in self.moves],
            "notes": self.notes,
        }  # fmt: skip


# ---------------------------------------------------------------- the detector (needs torch + transformers)


class Detector:
    """Open-vocabulary detector. Loaded once; `find(image, names)` returns Detections."""

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None):
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as e:
            raise ImportError("object detection needs torch and transformers: pip install -r requirements-vision.txt") from e
        self.torch = torch
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("cuda asked for, but this torch build cannot use the graphics card. Run: python -m so101_sim devices")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")  # the graphics card when torch can use it
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()
        self.model_id = model_id

    def find(self, image_bgr: np.ndarray, names: list[str], min_score: float = MIN_SCORE) -> list[Detection]:
        from PIL import Image

        img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        text = " ".join(n.strip().lower().rstrip(".") + "." for n in names)
        inputs = self.processor(images=img, text=text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=min_score, text_threshold=0.25, target_sizes=[img.size[::-1]])[0]  # fmt: skip
        labels = res.get("text_labels", res.get("labels"))
        return [Detection(str(lab), float(s), tuple(float(v) for v in b)) for s, lab, b in zip(res["scores"], labels, res["boxes"])]


# ---------------------------------------------------------------- geometry (no model needed, unit tested)


def iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    return w * h / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h)


def merge_same_box(dets: list[Detection]) -> list[tuple[Detection, list[str]]]:
    """The detector may call one thing 'carton', 'box' and 'object'. Keep the best box, remember the names."""
    out: list[tuple[Detection, list[str]]] = []
    for d in sorted(dets, key=lambda d: -d.score):
        for kept, names in out:
            if iou(kept.box, d.box) > 0.6:
                names.append(d.label)
                break
        else:
            out.append((d, [d.label]))
    return out


def clean(dets: list[Detection], image_shape, max_area: float = 0.3) -> list[tuple[Detection, list[str]]]:
    """Drop hands, people and boxes that fill the picture (sofa, table)."""
    area = image_shape[0] * image_shape[1]
    keep = [d for d in dets if d.label not in NOT_OBJECTS and d.label and d.size.prod() < max_area * area]
    return merge_same_box(keep)


def link_tracks(per_frame: list[list[tuple[Detection, list[str]]]]) -> list[dict]:
    """Link boxes over time into tracks. A box joins the track whose last box is the nearest
    of about the same size; a thing in the hand may jump a few box widths in half a second."""
    tracks: list[dict] = []
    for i, dets in enumerate(per_frame):
        free = list(tracks)
        for d, names in sorted(dets, key=lambda x: -x[0].score):
            best, best_cost = None, 1e9
            for tr in free:
                last = tr["dets"][tr["last"]]
                gap = i - tr["last"]
                diag = float(np.linalg.norm(last.size))
                dist = float(np.linalg.norm(d.centre - last.centre)) / (diag + 1e-6)
                ratio = float(np.sqrt(d.size.prod() / (last.size.prod() + 1e-6)))
                if dist > 2.5 * gap + 0.5 or not 0.45 < ratio < 2.2 or gap > 6:
                    continue
                cost = dist + abs(np.log(ratio))
                if cost < best_cost:
                    best, best_cost = tr, cost
            if best is None:
                tracks.append({"dets": {i: d}, "names": list(names), "last": i})
            else:
                best["dets"][i] = d
                best["names"] += names
                best["last"] = i
                free.remove(best)
    return tracks


def rests_and_moves(centres: list[np.ndarray | None], sizes: list[np.ndarray | None], still: float = 0.08, min_len: int = 3) -> list[tuple[int, int]]:
    """Runs of looked-at moments where the object stands still: [(first index, last index), ...].
    Still = the box centre stays within `still` box diagonals of where the run began, for at
    least `min_len` moments (a carton carried slowly is not standing)."""
    runs: list[tuple[int, int]] = []
    seen = [i for i, c in enumerate(centres) if c is not None]
    k = 0
    while k < len(seen):
        start = seen[k]
        members = [start]
        while k + 1 < len(seen):
            nxt = seen[k + 1]
            anchor = np.median([centres[i] for i in members], axis=0)
            if np.linalg.norm(centres[nxt] - anchor) > still * np.linalg.norm(sizes[start]):
                break
            members.append(nxt)
            k += 1
        if len(members) >= min_len:
            runs.append((members[0], members[-1]))
        k += 1
    return runs


def choose_object(tracks: list[dict], n: int, named: bool) -> tuple[dict, list[tuple[int, int]]]:
    """The object of the demo: the track that rests, is moved, and rests somewhere else.
    Among several, the one that travels furthest (in its own box sizes) and is seen most."""
    best, best_score, best_runs = None, 0.0, []
    for tr in tracks:
        idx = sorted(tr["dets"])
        if len(idx) < 3:
            continue
        centres = [tr["dets"][i].centre if i in tr["dets"] else None for i in range(n)]
        sizes = [tr["dets"][i].size if i in tr["dets"] else None for i in range(n)]
        runs = rests_and_moves(centres, sizes)
        if len(runs) < 2:
            continue
        travel = sum(
            np.linalg.norm(centres[b[0]] - centres[a[1]]) / np.linalg.norm(sizes[a[0]]) for a, b in zip(runs[:-1], runs[1:])
        )
        if travel < 0.5:
            continue
        score = travel * len(idx) / n * float(np.median([d.score for d in tr["dets"].values()]))
        if score > best_score:
            best, best_score, best_runs = tr, score, runs
    if best is None:
        what = "the object you named" if named else "an object"
        raise NoObjectMotion(f"did not see {what} stand still, get moved and stand still somewhere else")
    return best, best_runs


def refine_move(grey: np.ndarray, fps: float, box_a, box_b, i_rest_a: int, i_rest_b: int, i_ref_b: int) -> tuple[int, int, list[str]]:
    """Frame-exact lift and set-down between two resting places.

    grey: every frame, small and grey. box_a / box_b: resting boxes (x, y, w, h in grey pixels).
    i_rest_a: a frame where the object still rests at A. i_rest_b: a frame where it rests at B.
    i_ref_b: a frame at B with the hand most likely out of the way (the reference picture).
    """
    win = max(1, int(0.15 * fps))
    smooth = lambda x: np.convolve(np.pad(x, win, mode="edge"), np.ones(2 * win + 1) / (2 * win + 1), mode="valid")  # noqa: E731
    notes = []
    ref_a = np.median(grey[max(0, i_rest_a - win):i_rest_a + 1], axis=0)
    la = smooth(_like(grey, ref_a, box_a))
    gone = np.nonzero(la[i_rest_a:i_rest_b + 1] < 0.6)[0]  # the place starts to look different: it is being lifted
    if len(gone):
        i_lift = i_rest_a + int(gone[0])
    else:
        i_lift = (i_rest_a + i_rest_b) // 2
        notes.append("could not see the exact lift; used the middle between the two resting places")
    ref_b = np.median(grey[max(0, i_ref_b - win):i_ref_b + win + 1], axis=0)
    lb = smooth(_like(grey, ref_b, box_b))
    i_down = i_rest_b
    while i_down > i_lift and lb[i_down - 1] >= 0.5:
        i_down -= 1
    if lb[i_rest_b] < 0.5:
        notes.append("the object looks different once set down (hand still on it?); the release time is rough")
    return i_lift, i_down, notes


# ---------------------------------------------------------------- video in, track out


def read_frames(video: Path, t_start: float = 0.0, t_end: float | None = None) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, i = [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        t = i / fps
        if t >= t_start and (t_end is None or t <= t_end):
            frames.append(img)
        i += 1
    cap.release()
    if len(frames) < fps:
        raise NoObjectMotion("the video is too short")
    return frames, fps


def track_object(frames: list[np.ndarray], fps: float, name: str | None = None, detector=None, every_s: float = EVERY_S,
                 hand_px: dict[int, np.ndarray] | None = None, progress: bool = False) -> ObjectTrack:  # fmt: skip
    """frames: every frame of the (trimmed) video. Times in the result count from the first frame.
    hand_px: frame index -> a point on the hand (pixels), from the arm tracker, to check each grasp."""
    detector = detector or Detector()
    names = [name] if name else GENERIC_NAMES
    step = max(1, int(round(every_s * fps)))
    key = list(range(0, len(frames), step))
    per_frame = []
    for k, i in enumerate(key):
        per_frame.append(clean(detector.find(frames[i], names), frames[i].shape))
        if progress and (k + 1) % 10 == 0:
            print(f"  looked at {k + 1}/{len(key)} pictures")
    if name:
        # A named object: the surest box in each picture is it. (Linking is for the unnamed case,
        # where several things are found and the moved one must be picked out.)
        best = {k: max(d, key=lambda x: x[0].score)[0] for k, d in enumerate(per_frame) if d}
        tracks = [{"dets": best, "names": [name], "last": max(best, default=0)}]
    else:
        tracks = link_tracks(per_frame)
    track, runs = choose_object(tracks, len(key), named=bool(name))
    boxes = [track["dets"][k].box if k in track["dets"] else None for k in range(len(key))]
    specific = [n for n in track["names"] if n != "object"] or track["names"]
    out = ObjectTrack(name=name or Counter(specific).most_common(1)[0][0],
                      score=float(np.median([d.score for d in track["dets"].values()])),
                      times=np.array(key) / fps, boxes=boxes, seen_fraction=len(track["dets"]) / len(key))  # fmt: skip
    if not name:
        out.notes.append(f"no object name given: the detector calls it '{out.name}'. Give the real name with --object for a surer result.")

    # Frame-exact times on small grey pictures.
    scale = 320 / frames[0].shape[1]
    grey = np.array([cv2.cvtColor(cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY) for f in frames], dtype=np.float32)
    margin = int(round(0.2 * fps))

    def rest_box(run) -> tuple[float, float, float, float]:
        return tuple(np.median([boxes[k] for k in range(run[0], run[1] + 1) if boxes[k] is not None], axis=0))

    def small(box) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = (int(round(v * scale)) for v in box)
        x0, y0 = max(0, x0), max(0, y0)
        return x0, y0, max(4, min(grey.shape[2], x1) - x0), max(4, min(grey.shape[1], y1) - y0)

    for a, b in zip(runs[:-1], runs[1:]):
        box_a, box_b = rest_box(a), rest_box(b)
        i_lift, i_down, notes = refine_move(grey, fps, small(box_a), small(box_b), key[a[1]], key[b[0]], key[b[1]])
        i_grasp, i_release = max(0, i_lift - margin), min(len(frames) - 1, i_down + margin)
        move = Move(i_grasp / fps, i_release / fps, box_a, box_b)
        if hand_px is not None:
            near = [hand_px[i] for i in range(i_grasp, min(i_lift + margin, len(frames) - 1) + 1) if i in hand_px]
            grow = 0.5 * max(box_a[2] - box_a[0], box_a[3] - box_a[1])
            move.hand_on_object = any(box_a[0] - grow <= p[0] <= box_a[2] + grow and box_a[1] - grow <= p[1] <= box_a[3] + grow for p in near)
            if not move.hand_on_object:
                notes.append(f"at {move.t_grasp:.1f} s the object moves but the tracked hand is not on it (other hand? wrong --arm?)")
        out.moves.append(move)
        out.notes += notes
    if out.seen_fraction < 0.6:
        out.notes.append(f"the object was found in only {out.seen_fraction:.0%} of the pictures looked at")
    return out


def debug_image(frames: list[np.ndarray], fps: float, tr: ObjectTrack, out: Path) -> None:
    """One picture per grasp and release, with the object's box, to check by eye."""
    tiles = []
    for m in tr.moves:
        for t, box, text in ((m.t_grasp, m.pick_box, "grasp"), (m.t_release, m.place_box, "release")):
            img = frames[min(len(frames) - 1, int(round(t * fps)))].copy()
            x0, y0, x1, y1 = (int(v) for v in box)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 3)
            cv2.putText(img, f"{text} {t:.2f} s: {tr.name}", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            tiles.append(img)
    if tiles:
        rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
        cv2.imwrite(str(out), cv2.resize(np.vstack(rows), None, fx=0.5, fy=0.5))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m retarget.objects", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--object", dest="name", help="what the object is, in plain words, e.g. \"juice carton\"")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--every", type=float, default=EVERY_S, help="look for the object every this many seconds")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float)
    p.add_argument("--debug", type=Path, help="write a picture of every grasp and release with the object's box")
    a = p.parse_args(argv)
    try:
        frames, fps = read_frames(a.video, a.start, a.end)
        tr = track_object(frames, fps, a.name, Detector(a.model), a.every, progress=True)
    except NoObjectMotion as e:
        print(f"No object motion found: {e}")
        return 1
    print(f"object: {tr.name} (score {tr.score:.2f}, seen in {tr.seen_fraction:.0%} of the pictures)")
    for m in tr.moves:
        print(f"  grasp at {m.t_grasp + a.start:.2f} s, release at {m.t_release + a.start:.2f} s, "
              f"from pixel ({m.pick_px[0]:.0f}, {m.pick_px[1]:.0f}) to ({m.place_px[0]:.0f}, {m.place_px[1]:.0f})")  # fmt: skip
    for note in tr.notes:
        print(f"note: {note}")
    if a.debug:
        debug_image(frames, fps, tr, a.debug)
        print(f"debug picture: {a.debug}")
    print("--grips " + ",".join(f"{m.t_grasp + a.start:.2f}-{m.t_release + a.start:.2f}" for m in tr.moves))
    return 0


if __name__ == "__main__":
    sys.exit(main())
