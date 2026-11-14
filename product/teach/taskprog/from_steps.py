"""Demo steps (steps.json from the video app) + labels (labels.json from the vision LLM) -> task.json.

Who decides what:
  - positions and timing come only from the measured demo (steps.json)
  - names, descriptions and step labels come from the vision LLM (labels.json)
  - object size and grasp height come from the engineer (command line), until we measure them
If the LLM output is missing or a name is invalid, a plain default is used and a review note
is added for the engineer. The LLM can never change a number.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .task import NAME_RE, RESERVED, TASK_FORMAT, clean_text, parse_task


def _ok_name(v) -> bool:
    return isinstance(v, str) and bool(NAME_RE.match(v)) and v not in RESERVED


def _same_place(seen: dict | None, pick_no: int) -> bool:
    """Does pick number `pick_no` start where the place before it ended, going by the object's boxes?"""
    try:
        a, b = seen["moves"][pick_no - 2]["place_box_px"], seen["moves"][pick_no - 1]["pick_box_px"]
    except (TypeError, KeyError, IndexError):
        return False
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return False
    return w * h / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h) > 0.5


def _mm(xyz) -> list[float]:
    return [round(float(v) * 1000.0, 1) for v in xyz]


def steps_to_task(
    steps_doc: dict,
    labels: dict | None = None,
    object_size_mm=(20.0, 20.0, 40.0),
    grasp_height_mm: float | None = None,
    flat_table: bool = False,
    size_measured: bool = False,
) -> dict:
    labels = labels or {}
    review: list[str] = []
    steps = steps_doc.get("steps", [])
    by_index = {}
    for lab in labels.get("steps", []):
        if isinstance(lab, dict) and isinstance(lab.get("index"), int):
            by_index[lab["index"]] = lab
            if lab.get("matches_step_type") is False:
                review.append(f"vision LLM doubts demo step {lab['index']} ({steps[lab['index']]['step'] if 0 <= lab['index'] < len(steps) else '?'}): {clean_text(lab.get('comment', ''))}")

    objects_l = [o for o in labels.get("objects", []) if isinstance(o, dict)]
    obj_name = objects_l[0].get("name") if objects_l else None
    obj_desc = clean_text(objects_l[0].get("description", "")) if objects_l else ""
    seen = steps_doc.get("object") if isinstance(steps_doc.get("object"), dict) else None
    if seen:
        # The video app followed the object with a detector. Its name is used when no label gives one.
        seen_name = re.sub(r"[^a-z0-9]+", "_", clean_text(str(seen.get("name", ""))).lower()).strip("_")[:40]
        if not obj_name and _ok_name(seen_name):
            obj_name = seen_name
            who = "you" if seen.get("named_by") == "you" else "the object detector"
            obj_desc = obj_desc or f"named by {who}; detector score {seen.get('score', '?')}"
            if seen.get("named_by") != "you":
                review.append(f"the object name '{seen_name}' is the detector's guess; check it")
        for k, mv in enumerate(seen.get("moves", []), start=1):
            if mv.get("hand_on_object") is False:
                review.append(f"pick {k}: the object moved but the tracked hand was not on it; check the video at {mv.get('t_grasp')} s")
        for note in seen.get("notes", []):
            review.append("object detector: " + clean_text(str(note), 300))
    if not _ok_name(obj_name):
        if obj_name:
            review.append(f"object name {clean_text(str(obj_name))!r} from the LLM is not a valid name; used 'part'")
        obj_name = "part"
    if len(objects_l) > 1:
        review.append("the LLM saw more than one object; this version handles one object per task")

    gh = grasp_height_mm if grasp_height_mm is not None else min(object_size_mm[2] / 2, 30.0)
    if not size_measured:
        review.append(f"object size {object_size_mm[0]:g} x {object_size_mm[1]:g} x {object_size_mm[2]:g} mm and grasp height {gh:g} mm are assumed, not measured")

    locations: dict[str, dict] = {}
    task_steps: list[dict] = [{"do": "home", "label": "start at home"}]

    def loc_name(i: int, default: str) -> str:
        n = by_index.get(i, {}).get("location")
        if _ok_name(n) and n not in locations:
            return n
        if n:
            review.append(f"location name {clean_text(str(n))!r} for demo step {i} is invalid or used twice; used '{default}'")
        return default

    seg_start = 0.0
    picks = places = 0
    last_place: str | None = None
    for i, s in enumerate(steps):
        kind = s.get("step")
        if kind in ("reach", "move"):
            seg_start = float(s["t_start"])
        elif kind in ("grasp", "release"):
            xyz = _mm(s["to_xyz"])
            surface = 0.0 if flat_table else round(xyz[2] - gh, 1)
            if kind == "grasp":
                picks += 1
                if last_place and _same_place(seen, picks):
                    # The object track shows this pick starts where the last place ended. Use that
                    # one location, not a second one a few millimetres off from hand tracking.
                    task_steps.append({"do": "pick", "object": obj_name, "at": last_place,
                                       "label": clean_text(by_index.get(i, {}).get("label", "")) or "pick (unlabelled)",
                                       "demo": {"t_start": round(seg_start, 3), "t_end": round(float(s["t_end"]), 3)}})
                    continue
                name = loc_name(i, f"pick_{picks}")
            else:
                places += 1
                name = loc_name(i, f"place_{places}")
            locations[name] = {"xyz_mm": [xyz[0], xyz[1], surface], "yaw_deg": 0.0,
                               "note": clean_text(by_index.get(i, {}).get("location_description", "")) or f"from demo step {i}"}
            label = clean_text(by_index.get(i, {}).get("label", "")) or f"{'pick' if kind == 'grasp' else 'place'} (unlabelled)"
            task_steps.append({"do": "pick" if kind == "grasp" else "place", "object": obj_name, "at": name,
                               "label": label, "demo": {"t_start": round(seg_start, 3), "t_end": round(float(s["t_end"]), 3)}})
            if kind == "release":
                last_place = name
    task_steps.append({"do": "home", "label": "back to home"})
    for k, st in enumerate(task_steps):
        st["id"] = k + 1

    if flat_table:
        review.append("heights flattened to one table (--flat-table): the demo's measured heights were not used")
    review.append("yaw of every location is 0: the video app does not measure object orientation yet")
    if steps_doc.get("mode") == "body":
        review.append("positions come from body mode (arm scaled to the SO-101, not true millimetres)")

    name = labels.get("task_name")
    if not _ok_name(name):
        if name:
            review.append(f"task name {clean_text(str(name))!r} is not valid; used 'demo_task'")
        name = "demo_task"
    for r in labels.get("review", []):
        review.append("vision LLM: " + clean_text(r, 300))

    task = {
        "format": TASK_FORMAT,
        "name": name,
        "description": clean_text(labels.get("task_description", ""), 300),
        "units": {"length": "mm", "angle": "deg", "time": "s"},
        "source": {"video": steps_doc.get("video"), "steps_mode": steps_doc.get("mode"), "arm": steps_doc.get("arm"),
                   "labelled_by": clean_text(labels.get("_labelled_by", "nobody")) or "nobody"},
        "objects": {obj_name: {"size_mm": list(object_size_mm), "grasp_height_mm": gh, "description": obj_desc}},
        "locations": locations,
        "defaults": {"approach_mm": 50, "travel_height_mm": 100, "grip_time_s": 0.5, "speed": "normal"},
        "steps": task_steps,
        "review": review,
    }
    parse_task(task)  # raises TaskError with every problem if the result is not valid
    return task


def dumps(task: dict) -> str:
    """JSON with short number lists on one line, easier to read on a phone."""
    text = json.dumps(task, indent=2)
    return re.sub(r"\[\s*([-0-9.,\s]+?)\s*\]", lambda m: "[" + ", ".join(v.strip() for v in m.group(1).split(",")) + "]", text) + "\n"


def convert(steps_path: Path, out_path: Path, labels_path: Path | None = None, **kw) -> dict:
    steps_doc = json.loads(Path(steps_path).read_text())
    labels = json.loads(Path(labels_path).read_text()) if labels_path else None
    task = steps_to_task(steps_doc, labels, **kw)
    Path(out_path).write_text(dumps(task))
    return task
