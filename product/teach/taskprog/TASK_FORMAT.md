# Task file format (cobot-task/0.2)

One task = one JSON file. It says **what** to do, not which robot does it.
The robot, its tool and its speeds are in a separate **cell file**.
Examples: [take1](examples/take1/task.json) (from a demo), [kitting tray](examples/kitting_tray/task.json),
[machine tending](examples/machine_tending/task.json). Files written as 0.1 still load.

## Why two files and no free robot code

```
demo video -> steps.json -> vision LLM labels -> task.json  +  cell.json  -> fixed generator -> URScript / SO-101 / later FANUC, Doosan
              (measured)     (names only)        (what)        (which robot)   (no AI)
```

- The LLM only names and labels. Every number comes from the measured demo or the engineer.
- The generator is fixed code: same task + same cell = the same program, byte for byte.
- A new brand = a new generator over the same plan. The task file does not change.
- An integrator reads the task file in a minute and can fix a location without touching code.

## Units and frames

- mm, degrees, seconds.
- Work frame: the table the task happens on. x and y on the table, z up.
  The cell file says where the work frame sits on the robot.
- The tool always points straight down (top-down picks). Enough for kitting and most
  pick-and-place; side picks need a later version.

## Top level

| Key | Required | Meaning |
|---|---|---|
| `format` | yes | `"cobot-task/0.2"` |
| `name` | yes | snake_case, max 40 chars. Becomes the program name. |
| `description` | no | One sentence. |
| `units` | no | Must be `{"length": "mm", "angle": "deg", "time": "s"}` if given. |
| `source` | no | Where it came from (video, labeller). Not used by the generator. |
| `objects` | yes | Things the gripper holds. |
| `locations` | yes | Named places. A place can be a grid (tray, pallet). |
| `signals` | no | Machine signals the task uses: `{"cycle_start": "output", "cycle_done": "input"}`. |
| `defaults` | no | `approach_mm` (50), `travel_height_mm` (100), `grip_time_s` (0.5), `speed` ("normal"). |
| `steps` | yes | The skills, in order. |
| `review` | no | Open questions for the engineer. `check` shows each as a warning. |

## objects

```json
"juice_carton": {"size_mm": [70, 70, 195], "grasp_height_mm": 150, "description": "1 l carton"}
```

`size_mm` = width across the jaws, depth, height. `grasp_height_mm` = where the gripper's tool
point closes, measured up from the object's bottom (default: half the height, max 30).

## locations

```json
"bin_a": {"xyz_mm": [400, -120, 0], "yaw_deg": 0, "note": "left bin"}
```

`xyz_mm` = x, y of the object's centre and **z of the surface it stands on**, in the work frame.
So "the carton stands on the armrest" is stored once, and the gripper heights are derived:
grasp = surface + grasp height, approach = grasp + approach. `yaw_deg` turns the gripper about
the vertical (the SO-101 cannot, it ignores it).

### Grid: a tray or pallet (0.2)

```json
"tray": {"xyz_mm": [150, 100, 0], "grid": {"count": [3, 2], "pitch_mm": [60, 60]}}
```

One location, many slots. `xyz_mm` is **slot 1**. `count` = columns (along x), rows (along y).
`pitch_mm` = centre to centre; negative runs the other way. Slots are numbered along x, then y,
then up: a 3 x 2 tray is 1 2 3 / 4 5 6. `yaw_deg` turns the whole grid. For a pallet with layers
add `"layers": 2, "layer_height_mm": 40`. In the robot program slot 1 is one taught pose and the
other slots are offsets from it, so re-teaching slot 1 moves the whole tray.

## steps (skills)

| `do` | Keys | Robot does |
|---|---|---|
| `home` | | Joint move to the cell's home joints |
| `move` | `to`, `height_mm` | Joint move to `height_mm` above a location (default `travel_height_mm`) |
| `pick` | `object`, `at`, `approach_mm` | Joint move above, straight down, close, wait, straight up |
| `place` | `object`, `at`, `approach_mm` | Joint move above, straight down to grasp + release gap, open, wait, straight up |
| `wait` | `seconds` | Wait (0..60 s) |
| `set_output` | `signal`, `value` (true), `pulse_s` | Set a signal. With `pulse_s` it goes back after that time |
| `wait_input` | `signal`, `value` (true), `timeout_s` | Wait until the signal has the value. After `timeout_s` the program stops with a message |
| `repeat` | `times`, `steps` | Do the inner steps `times` times. Not inside another repeat |

At a grid, `pick`, `place` and `move` need `slot`: a number, `"next"` (1, 2, 3 ... each time
the task uses that grid) or `"same"` (the slot used last). A repeat is written out step by
step in the robot program, so every round is checked and reported on its own.

Every step may also have `id` (1, 2, 3 in order), `label` (one line of text, ends up as a
comment), `speed` (a speed name from the cell) and `demo` (`t_start`, `t_end` in the video).

Rules that make a task invalid: unknown keys, unknown skills, unknown names, placing what you
don't hold, picking while holding, ending while holding, numbers out of range, a slot outside
the grid, placing into a slot the task already filled, picking from a slot it already emptied,
a signal that is not declared or has the wrong direction, more than 500 steps.

## Cell file (cobot-cell/0.2)

Example: [cells/ur5e_table.json](cells/ur5e_table.json), [cells/so101_twin.json](cells/so101_twin.json).
Written by the engineer once per robot cell: `robot` (ur3e, ur5e, ur10e, so101), `work_frame`
(`xyz_mm`, `yaw_deg` in the robot base frame), `tcp_mm`, `payload_kg`, `gripper` (digital output
pin, or so101), `home_joints_deg`, `speeds` (named, capped at 120 deg/s and 500 mm/s),
`release_gap_mm`, `min_tool_height_mm`, `work_area_mm` (x min, x max, y min, y max).

`signals` (0.2) gives every task signal a pin: `{"cycle_done": {"io": "standard_in", "pin": 1}}`.
`io` is `standard_in`, `configurable_in`, `tool_in`, `standard_out`, `configurable_out` or
`tool_out`. The task never names a pin, so the same task runs on a cell wired differently.
`check` fails if a signal has no pin, the wrong direction, or the gripper's pin.

`simulate` is for URSim only: `[{"when": "cycle_start", "then": "cycle_done", "after_s": 2}]`
adds a pretend machine to the program. URSim's inputs cannot be set from outside, so in that
cell the input reads back an output. See [cells/ur5e_tending_sim.json](cells/ur5e_tending_sim.json).
The program header and `check` both say it is not for a real robot.

## Not built yet

Side grasps (full tool orientation), collision checks against fixtures, checks that a part is
really there (gripper feedback, sensors), analog and Modbus signals, a FANUC TP generator.
