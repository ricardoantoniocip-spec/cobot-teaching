"""Exercise 3: what the vision LLM gets, what it may answer, and what it can never change.

Run (from the repo root, venv active). No API key, no video:
    python learning\\2026-11-14-steps-from-demo\\ex3_labels.py

Uses product/teach/taskprog/label.py (frame times, schema) and from_steps.py (labels + steps
-> task file) on take 1's steps.json and labels.json.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "product" / "teach"))

from taskprog.from_steps import steps_to_task  # noqa: E402
from taskprog.label import LABELS_SCHEMA, frame_times  # noqa: E402

TAKE1 = REPO / "product/teach/taskprog/examples/take1"
steps_doc = json.loads((TAKE1 / "steps.json").read_text())
labels = json.loads((TAKE1 / "labels.json").read_text())

# 1. Which video frames the LLM is shown: start, middle, end of each step; one for instant steps.
t0 = float(steps_doc.get("video_t0", 0.0))
print(f"1. Frames sent to the LLM (video_t0 = {t0} s{'' if 'video_t0' in steps_doc else ', missing in this older file so 0 is used'})")
for i, t in frame_times(steps_doc["steps"], t0):
    print(f"   step {i} ({steps_doc['steps'][i]['step']:7}) at {t:5.2f} s")

# 2. The answer shape the LLM must follow (structured output).
print("\n2. The LLM must answer with these keys:")
print("   top level:", ", ".join(LABELS_SCHEMA["required"]))
print("   per step: ", ", ".join(LABELS_SCHEMA["properties"]["steps"]["items"]["required"]))


def keys(node) -> set[str]:
    out = set()
    if isinstance(node, dict):
        out |= set(node.get("properties", {}))
        for v in node.values():
            out |= keys(v)
    elif isinstance(node, list):
        for v in node:
            out |= keys(v)
    return out


nums = [k for k in keys(LABELS_SCHEMA) if k.endswith("_mm") or "xyz" in k or k in ("x", "y", "z")]
print(f"   keys for positions or sizes: {nums or 'none'}  (extra keys are not allowed: additionalProperties false)")

# 3. Good labels -> task file. Names from the LLM, positions from steps.json.
task = steps_to_task(steps_doc, labels, flat_table=True)
print(f"\n3. Task '{task['name']}', object '{list(task['objects'])[0]}'")
for name, loc in task["locations"].items():
    print(f"   location {name:10} xyz_mm {loc['xyz_mm']}  ({loc['note']})")
for s in task["steps"]:
    print(f"   step {s['id']}: {s['do']:5} {s.get('at', ''):10} \"{s['label']}\"")

# 4. A careless LLM: bad names, a doubt, and an attempt to move the pick point.
bad = copy.deepcopy(labels)
bad["task_name"] = "Move Carton!"
bad["steps"][1]["location"] = "sofa armrest"
bad["steps"][1]["xyz_mm"] = [999, 999, 999]  # not in the schema; from_steps never reads it
bad["steps"][2]["matches_step_type"] = False
bad["steps"][2]["comment"] = "the hand looks empty in the middle frame"
task2 = steps_to_task(steps_doc, bad, flat_table=True)
print(f"\n4. Careless labels -> task '{task2['name']}', locations {list(task2['locations'])}")
print(f"   pick location xyz_mm {list(task2['locations'].values())[0]['xyz_mm']} (unchanged)")
for r in task2["review"][:3]:
    print(f"   review: {r}")

# 5. No labels at all still gives a valid task, with plain names.
task3 = steps_to_task(steps_doc, None, flat_table=True)
print(f"\n5. No labels -> task '{task3['name']}', locations {list(task3['locations'])}, object '{list(task3['objects'])[0]}'")
