"""Vision LLM labelling: frames from the demo video + steps.json -> labels.json.

The model sees a few frames per demo step and answers in a fixed JSON shape (structured
output). It names the task, the object and the locations, writes one plain label per step and
says when a step looks wrong. It never sees or writes coordinates or robot code.

Needs `pip install anthropic` and an API key (ANTHROPIC_API_KEY, or `ant auth login`).
Without a key, `--frames-only` writes the frames and the prompt so a person, or Claude in a
chat, can fill in labels.json by hand in the same shape.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

MODEL = "claude-opus-5-5"

LABELS_SCHEMA = {
    "type": "object",
    "properties": {
        "task_name": {"type": "string", "description": "snake_case, max 40 chars, starts with a letter, e.g. move_carton_to_seat"},
        "task_description": {"type": "string", "description": "one plain sentence"},
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "snake_case, e.g. juice_carton"},
                    "description": {"type": "string"},
                },
                "required": ["name", "description"],
                "additionalProperties": False,
            },
        },
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "label": {"type": "string", "description": "what the operator does, max 12 words"},
                    "object": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "location": {"anyOf": [{"type": "string"}, {"type": "null"}],
                                 "description": "snake_case name of the place, only for grasp and release steps"},
                    "location_description": {"type": "string"},
                    "matches_step_type": {"type": "boolean", "description": "false if the frames do not show this kind of step"},
                    "comment": {"type": "string"},
                },
                "required": ["index", "label", "object", "location", "location_description", "matches_step_type", "comment"],
                "additionalProperties": False,
            },
        },
        "review": {"type": "array", "items": {"type": "string"}, "description": "anything an engineer should check"},
    },
    "required": ["task_name", "task_description", "objects", "steps", "review"],
    "additionalProperties": False,
}

PROMPT = """You are labelling a video of a person demonstrating a pick-and-place task that a robot will repeat.
A tracker has already cut the video into steps (reach, grasp, move, release, retreat). For each step you get
frames from its start, middle and end, with the time in seconds.

For every step, give a short plain label of what the person does, the object involved, and for grasp and
release steps a snake_case name for the place (for example bin_a, conveyor_end, armrest). Name the task and
the object in snake_case. If the frames do not show the kind of step the tracker says (for example no object
in the hand during "move"), set matches_step_type to false and explain in comment. Put anything an engineer
should double-check in review. Do not guess sizes or positions; they are measured elsewhere.

Steps from the tracker:
{steps}
"""


def frame_times(steps: list[dict], video_t0: float = 0.0) -> list[tuple[int, float]]:
    """Video times to look at for each step. Step times count from the first tracked frame;
    video_t0 (in steps.json) is where that frame sits in the video."""
    out = []
    for i, s in enumerate(steps):
        a, b = float(s["t_start"]) + video_t0, float(s["t_end"]) + video_t0
        ts = [a] if b - a < 0.2 else [a, (a + b) / 2, b]
        out += [(i, round(t, 2)) for t in ts]
    return out


def grab_frames(video: Path, times: list[tuple[int, float]], width: int = 640) -> list[tuple[int, float, bytes]]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {video}")
    out = []
    for i, t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        if w > width:
            frame = cv2.resize(frame, (width, int(h * width / w)))
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        out.append((i, t, jpg.tobytes()))
    cap.release()
    return out


def _steps_text(steps: list[dict]) -> str:
    return "\n".join(f"  {i}: {s['step']} from {s['t_start']:.2f} s to {s['t_end']:.2f} s" for i, s in enumerate(steps))


def write_frames(video: Path, steps_doc: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = steps_doc["steps"]
    for i, t, jpg in grab_frames(video, frame_times(steps, float(steps_doc.get("video_t0", 0.0)))):
        (out_dir / f"step{i}_t{t:05.2f}.jpg").write_bytes(jpg)
    (out_dir / "prompt.txt").write_text(PROMPT.format(steps=_steps_text(steps)) + "\nAnswer as JSON in this shape:\n" + json.dumps(LABELS_SCHEMA, indent=2))
    return out_dir


def label(video: Path, steps_doc: dict, model: str = MODEL) -> dict:
    import anthropic

    steps = steps_doc["steps"]
    content: list[dict] = []
    for i, t, jpg in grab_frames(video, frame_times(steps, float(steps_doc.get("video_t0", 0.0)))):
        content.append({"type": "text", "text": f"step {i} ({steps[i]['step']}), t = {t:.2f} s"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.standard_b64encode(jpg).decode()}})
    content.append({"type": "text", "text": PROMPT.format(steps=_steps_text(steps))})

    client = anthropic.Anthropic()
    kwargs = dict(
        model=model,
        max_tokens=16000,
        messages=[{"role": "user", "content": content}],
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": LABELS_SCHEMA}},
    )
    if model == MODEL:
        # If the model declines, the API retries on a fallback model in the same call.
        response = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
    else:
        response = client.messages.create(**kwargs)
    if response.stop_reason == "refusal":
        raise RuntimeError("the model declined to label this video")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("the answer was cut off (max_tokens)")
    text = next(b.text for b in response.content if b.type == "text")
    labels = json.loads(text)
    labels["_labelled_by"] = response.model
    return labels
