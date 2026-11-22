# Lesson 5: From steps to a robot program

Series "what we built", part 3 of 4. About 15 minutes reading, 30 minutes exercises. Laptop, no GPU.

## Results first
- Take 1's steps + labels became a task file with 4 steps: home, pick at `armrest`, place at `sofa_seat`, home.
- Offline check on a UR5e cell and on the SO-101 cell: PASS, 0 errors, 6 review warnings for the engineer.
- URScript: 60 lines. In URSim (UR's own simulator, 13 Nov) it ran to the end and every step ended within 0.1 mm of the plan.
- SO-101: a 7.7 s joint trajectory. In the MuJoCo twin the block ended 12 mm from the zone: SUCCESS.
- Regenerated today from steps.json: task file, URScript and SO-101 trajectory are **identical, byte for byte**, to the reviewed examples. Same input, same program.
- Limits: the carton was replaced by the twin's 20 mm block, heights were flattened, positions are body-mode. This proves the chain, not that a UR would grip a carton on a sofa.

## The chain

```
 task.json (what)      cell.json (which robot)
        \                  /
         make_plan  (fixed rules, plan.py)
                |
         check  (offline, check.py)
           /                 \
   urscript (ur.py)      so101 (so101.py)
        |                     |
      URSim             MuJoCo twin / real arm
```

No AI anywhere in this lesson. The LLM's job ended at labels.json (lesson 4).

## 1. The task file (format 0.2): what to do
Spec: `product/teach/taskprog/TASK_FORMAT.md`. Factory picture: the task file is the **routing / work instruction**; the cell file is the **machine setup sheet**. Same routing, different machine.

- **objects**: `size_mm` (width across the jaws, depth, height) and `grasp_height_mm` (where the jaws close, up from the object's bottom).
- **locations**: `xyz_mm` = x, y of the object centre and **z of the surface it stands on**, in the work frame (the table). Gripper heights are derived: grasp = surface + grasp height, approach = grasp + 50 mm.
- **grid** (0.2): one location, many slots. `"count": [3, 2], "pitch_mm": [60, 60]` is a 3 x 2 tray. `xyz_mm` is slot 1; slots go along x, then y, then up (`task.py:59-79`). A step says `"slot": 3`, `"next"` or `"same"`.
- **signals** (0.2): names and directions only, `{"cycle_done": "input"}`. The task never names a pin.
- **steps**: `home`, `move`, `pick`, `place`, `wait`, `set_output` (optionally a pulse), `wait_input` (with a timeout), `repeat`.
- `repeat` is written out step by step before anything else (`_flatten`, `task.py:426-449`), so step ids count the unrolled steps.
- **Rules** that reject a task (`task.py:372-394` and the spec): place what you do not hold, pick while holding, end while holding, slot outside the grid, place into a filled slot, unknown names, unknown keys.

Take 1's task: `product/teach/taskprog/examples/take1/task.json`. Kitting and machine tending examples (written by hand, not from a demo) are next to it.

## 2. The cell file: which robot
`product/teach/taskprog/cells/ur5e_table.json` as an example. Written once per cell by the engineer:
- `robot` (ur3e, ur5e, ur10e, so101), `work_frame` (where the table is in the robot base frame), `tcp_mm`, `payload_kg`.
- `gripper` (which digital output, or so101), `home_joints_deg`.
- `speeds` by name, capped at 120 deg/s and 500 mm/s (`task.py:173-174`).
- `release_gap_mm`, `min_tool_height_mm`, `work_area_mm`.
- `signals` (0.2): the pin for each task signal, e.g. `{"cycle_done": {"io": "standard_in", "pin": 1}}`.
- `simulate`: a pretend machine, URSim only.

## 3. The plan: one pick = four motions
`make_plan` (`product/teach/taskprog/plan.py:76-113`) expands every step with fixed rules:

```
pick part at L:   ptp  to  L + grasp height + approach   (joint move)
                  line to  L + grasp height              (straight down)
                  grip close, wait grip_time_s
                  line to  L + grasp height + approach   (straight up)
```

`place` is the same with the gripper opening, 2 mm higher (`release_gap_mm`, line 104).

Geometry you know: a location's pose is its surface point with the tool z axis pointing down (`TOOL_DOWN` = 180 degrees about x, line 50; `location_pose`, lines 61-66). "50 mm above" is a shift along minus tool z (`raise_pose`, lines 69-73). Each motion keeps the location **name** and the height, so the program says `pose_trans(loc_armrest, p[0, 0, -0.07, 0, 0, 0])` instead of bare numbers. Re-teach one location on the pendant and everything that uses it follows.

## 4. The generator: plan to URScript
`urscript` in `product/teach/taskprog/ur.py:213-353`. Plain string templates. It only writes numbers from the task and cell files, and comments from the cleaned labels.
- Header: "NOT APPROVED", the task file's sha256 and the cell name (lines 215, 221-231). You can always prove which file made which program.
- `work = p[...]`: one pose for the whole table (line 238). Locations are `pose_trans(work, ...)` (lines 243-247).
- Grid slots are offsets from slot 1 (lines 248-259). Note the y and z signs flip: the location frame has the tool pointing down (line 255).
- `ptp` becomes `movej(get_inverse_kin(target, qnear=home_q))`, a line becomes `movel(target)` (lines 319-324).
- After each step: `textmsg("CT step N done ", get_actual_tcp_pose())` (lines 304, 350). URSim's report is built on this.
- `wait_input`: a loop with a timeout, then `halt` with a message (lines 335-349).

## 5. The offline check
`check` (`product/teach/taskprog/check.py:27-89`) runs before any simulator. Each finding is ERROR (no program is written), WARN (an engineer must look) or INFO.
- Every pose: tool point above `min_tool_height_mm`, inside `work_area_mm` (lines 32-41).
- UR cells: `reach_check` (`ur.py:123-167`) does IK for every target **like the controller**, starting from home (line 137), with UR's published DH parameters (lines 18-24). Flags: out of reach (over 1 mm or 0.5 degree off), a joint swinging over 170 degrees, a strange posture (leaning back, elbow flipped).
- Straight lines are walked in 20 small steps. A joint jumping over 15 degrees in one step = a singularity on the way (lines 146-161).
- Singularities by name (lines 101-112): wrist (joint 5 near 0 or 180), elbow (arm straight), shoulder (wrist centre within 8 cm of the base axis). Same three you met in your robotics minor.
- SO-101 cells: the IK must reach every target within 5 mm (check.py lines 43-49).
- Signals: each one needs a pin in the cell, the right direction, and not the gripper's pin (lines 61-80).
- Every `review` note in the task becomes a WARN (lines 85-86). So the engineer sees "object size assumed, not measured" every time.

## 6. URSim: the controller's own opinion
`python -m taskprog ursim TASK --cell CELL` (`taskprog/ursim.py`). URSim runs in Docker on your laptop (see `product/teach/taskprog/README.md`). The script powers the simulated robot on (dashboard port 29999), sends the program (port 30002), listens to the robot's messages (port 30001), and compares the pose after every step with the plan. Pass = finished, no errors, every step within 1 mm (`compare`, line 154). This catches what our own IK might get wrong: compile errors, IK failures in the controller, protective stops. Not run today (needs Docker); it passed on 13 Nov.

## 7. SO-101 output: the same plan as joint angles
`trajectory` (`product/teach/taskprog/so101.py:63-120`) turns the same plan into joints at 30 Hz:
- The SO-101's tool point is the tip of the fixed finger, not the middle of the jaws. So the target is shifted by half the object width plus half a finger (6 mm), along the line from the base (lines 40-50).
- `ptp` = joint interpolation with a smooth cosine start and stop (lines 58-60, 82-93).
- `line` = straight line in space, with IK at every sample (lines 94-106).
- Gripper ramps open or closed (lines 107-112). The SO-101 has no I/O, so signal steps become short pauses (lines 113-117).
- The `twin` command needs exactly one pick and one place on a flat table and the so101 cell (`twin.py:22-29`).

## Exercises (on your laptop, 30 minutes)

```powershell
cd C:\dev\cobot-teaching\product\teach
C:\dev\venvs\teach\Scripts\Activate.ps1
mkdir $env:TEMP\take1 -Force
```

### Exercise 1: regenerate take 1 and prove it is the same
```powershell
python -m taskprog from-steps taskprog\examples\take1\steps.json $env:TEMP\take1\task.json --labels taskprog\examples\take1\labels.json --flat-table
Compare-Object (Get-Content $env:TEMP\take1\task.json) (Get-Content taskprog\examples\take1\task.json)
python -m taskprog check $env:TEMP\take1\task.json --cell taskprog\cells\ur5e_table.json
python -m taskprog urscript $env:TEMP\take1\task.json --cell taskprog\cells\ur5e_table.json $env:TEMP\take1\take1.script
Compare-Object (Get-Content $env:TEMP\take1\take1.script) (Get-Content taskprog\examples\take1\move_carton_to_seat.script)
python -m taskprog so101 $env:TEMP\take1\task.json --cell taskprog\cells\so101_twin.json $env:TEMP\take1\traj.csv
python -m taskprog twin $env:TEMP\take1\task.json --cell taskprog\cells\so101_twin.json --video $env:TEMP\take1\twin.mp4
```

Expected (checked; `Compare-Object` prints nothing when the files are the same):

```
Task 'move_carton_to_seat' with 4 steps written to ...
  review: object size 20 x 20 x 40 mm and grasp height 20 mm are assumed, not measured
  ... 5 more review lines
  check: PASS (0 errors, 6 warnings for the engineer)
  URScript written to ...take1.script (60 lines)
  232 samples, 7.7 s, written to ...traj.csv
  joint tracking, worst: 0.6 deg
  tool point vs IK, worst: 2.1 mm
  part moved 134 mm, ends 12 mm from the place zone centre
  task: SUCCESS
```

The twin takes about a minute. Watch `twin.mp4`. Then open `take1.script` and find the four motions of the pick.

### Exercise 2: what one pick becomes
```powershell
cd C:\dev\cobot-teaching
python learning\2026-11-14-steps-to-robot-program\ex2_plan.py
```

Expected (checked, first lines):

```
1. Plan for 'move_carton_to_seat' on ur5e_table: 12 motions
   step kind    location   height  tool point in work frame (mm)   UR5e joints (deg)
      2 ptp     armrest       70  [176, 46, 70]   [168, -87, 130, -133, -90, -102]   above armrest
      2 line    armrest       20  [176, 46, 20]   [168, -80, 133, -143, -90, -102]   down to grasp height
      2 grip                                         close gripper
      2 line    armrest       70  [176, 46, 70]   ...                               back up from armrest
2. SO-101: 232 samples at 30 Hz = 7.7 s. Worst IK miss 0.0 mm (down to release height), limit 5 mm.
```

Question: why is the place height 72 mm and not 70?

### Exercise 3: your own kitting task, then break it
Files in this folder: `my_task.json` (4 parts from a feeder into a 2 x 2 tray, then a `tray_full` pulse) and `my_cell.json` (the UR5e table cell plus a pin for `tray_full`).

```powershell
cd C:\dev\cobot-teaching\product\teach
$L = "..\..\learning\2026-11-14-steps-to-robot-program"
python -m taskprog check $L\my_task.json --cell taskprog\cells\ur5e_table.json
python -m taskprog check $L\my_task.json --cell $L\my_cell.json
python -m taskprog urscript $L\my_task.json --cell $L\my_cell.json $env:TEMP\take1\my.script
```

Expected (checked):

```
  ERROR step 10: signal 'tray_full' is not in the cell's signals (which pin is it?)
  check: FAIL (1 errors, 0 warnings for the engineer)
  ...
  check: PASS (0 errors, 0 warnings for the engineer)
  URScript written to ...my.script (123 lines)
```

Why step 10? The repeat is unrolled: home is 1, 4 x (pick, place) is 2-9. In `my.script` find `loc_tray_s2`: it is `pose_trans(loc_tray, p[0.06, 0, 0, ...])`, an offset from slot 1.

Now break it, one edit at a time (undo each before the next), and run the second `check` again:

| Edit in my_task.json | Expected message |
|---|---|
| `"times": 4` -> `5` | `slot 5 is outside 1..4 of 'tray'` |
| delete the `pick` line inside the repeat | `step 2: place part but the gripper holds nothing` (and steps 3-5) |
| feeder `"xyz_mm": [600, -200, 0]` | `ERROR step 2: feeder (600, -200) mm is outside the cell's work area` (repeated for every pick motion) |

## Self-check
1. You move the whole kitting table 30 mm to the left in the real cell. What do you change, and in which file?
2. Why does the generator write `pose_trans(loc_tray, offset)` for slots 2-4 instead of four separate taught poses?
3. `check` passes on the UR5e cell. Name two things that can still go wrong in URSim or on the real robot.

## For this afternoon's take
Think like the task file: one object, one clear pick place and one clear place place, both on surfaces you could measure with a ruler. Put the part on the **same table** for pick and place (one flat surface) so you do not need `--flat-table`, and measure the part's width and height so you can pass `--object-size W D H --measured` instead of the 20 mm block.

---

## Answers
1. Only the cell file: `work_frame.xyz_mm` (y + 30 mm, since y is left). In the URScript that is the single `work = p[...]` pose. The task file does not change.
2. So re-teaching slot 1 on the pendant moves the whole tray, and the pitch is stated once. It is also how an integrator would write it by hand.
3. Any two of: collisions with fixtures (no fixture model yet), the controller's IK picking a different solution, a protective stop, the part not really in the gripper (no grip check), a wrong object size or grasp height, signals wired differently from the cell file.

Exercise questions:
- Exercise 2: a place goes down to grasp height **plus** the cell's `release_gap_mm` (2 mm on the UR5e cell, `plan.py:103-104`), so the part is let go just above the surface instead of being pushed into it.
