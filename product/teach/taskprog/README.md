# taskprog: from a demo to a native robot program

```
video app          vision LLM          this package (fixed code, no AI)
steps.json  --->  labels.json  --->  task.json + cell.json  --->  check  --->  URScript  ---> URSim
(measured)        (names, labels)    (what)     (which robot)                  SO-101 traj ---> MuJoCo twin / real arm
```

The LLM never writes robot code and never writes a number. It names the task, the object and
the places, and labels each step. Positions come from the demo, sizes from the engineer.
A fixed generator turns the task into the robot's own language. Format: [TASK_FORMAT.md](TASK_FORMAT.md).

## Commands (run from product/teach)

| Command | What it does |
|---|---|
| `python -m taskprog frames VIDEO steps.json OUT_DIR` | Frames per step + the prompt, to label by hand (no API key) |
| `python -m taskprog label VIDEO steps.json labels.json` | Claude labels the steps from the frames (needs `pip install anthropic` and an API key) |
| `python -m taskprog from-steps steps.json task.json --labels labels.json` | Demo + labels to a task file. Add `--object-size W D H`, `--grasp-height`, `--flat-table` |
| `python -m taskprog check task.json --cell CELL` | Offline checks: reach, straight lines, singularities, heights, work area, review notes |
| `python -m taskprog urscript task.json --cell cells/ur5e_table.json OUT.script` | Writes URScript. Refuses if `check` finds an error |
| `python -m taskprog ursim task.json --cell cells/ur5e_table.json` | Runs the program in URSim and compares the pose after every step with the plan |
| `python -m taskprog so101 task.json --cell cells/so101_twin.json OUT.csv` | SO-101 joint trajectory; play it with `python -m retarget replay OUT.csv --port COM5` |
| `python -m taskprog twin task.json --cell cells/so101_twin.json` | Plays it in the MuJoCo twin (product/sim), prints SUCCESS/FAIL |

## Take 1 (your sofa video), 13 Nov 2026

Files in [examples/take1](examples/take1): steps.json (from the video app), labels.json,
task.json, move_carton_to_seat.script (URScript), traj_so101.csv.

| Check | Result |
|---|---|
| Task file valid, offline check on UR5e and SO-101 | PASS, 0 errors, 6 review notes |
| URSim 5.26, UR5e, Docker | PASS: runs to the end in 10 s, every step ends within 0.1 mm of the plan |
| MuJoCo twin, SO-101 | SUCCESS: block ends 12 mm from the place zone centre |

Honest limits of take 1:
- Labels were written by Claude in the project chat from 5 frames, not by the `label` command (no API key in the cloud). Run `label` once on the laptop to test the API path.
- The object size is the twin's 20 x 20 x 40 mm block, not the 1 l carton. Heights were flattened to one table. Positions are body-mode (scaled to the SO-101), not real millimetres.
- So this proves the chain (demo -> task -> program -> two simulators), not that a UR would grip a carton on a sofa.

## URSim on your laptop (optional, about 20 minutes)

Needs Docker Desktop (Windows, WSL 2).

```powershell
docker run --rm -d --name ursim -e ROBOT_MODEL=UR5E -p 29999:29999 -p 30001-30004:30001-30004 -p 6080:6080 universalrobots/ursim_e-series
# wait about a minute, then watch the robot in the browser: http://localhost:6080/vnc.html
cd C:\dev\cobot-teaching\product\teach
python -m taskprog ursim taskprog\examples\take1\task.json --cell taskprog\cells\ur5e_table.json
docker stop ursim
```

## What `check` and URSim catch

- Out of reach, a straight line that leaves the reach or flips a joint, nearby wrist/elbow/shoulder singularities (UR DH parameters from Universal Robots).
- Tool point below the floor, location outside the cell's work area.
- In URSim: compile errors, `get_inverse_kin` failures, protective stops, and any step that ends more than 1 mm from the plan.
- Not caught yet: collisions with fixtures (no fixture model), grip success on a real part.

## Tests

`python -m pytest -q tests` (38 tests, about 30 s; the twin test needs mujoco and skips without it).

## Demo video

`tools/record_ursim.js` records URSim's screen while the program runs; `tools/make_demo_video.py`
builds the 50-second video (operator video with labels, task file, URScript, URSim, twin) from a
take's own files. Take 1's video is in the project files (demo-video/). Found while recording:
the first cell had the table behind the home pose, so the arm reached over backwards. `check`
now warns about that posture and the example cell faces the table.
