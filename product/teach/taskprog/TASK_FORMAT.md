# Task file format (cobot-task/0.1)

One task = one JSON file. It says **what** to do, not which robot does it.
The robot, its tool and its speeds are in a separate **cell file**.
Example: [examples/take1/task.json](examples/take1/task.json).

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
| `format` | yes | `"cobot-task/0.1"` |
| `name` | yes | snake_case, max 40 chars. Becomes the program name. |
| `description` | no | One sentence. |
| `units` | no | Must be `{"length": "mm", "angle": "deg", "time": "s"}` if given. |
| `source` | no | Where it came from (video, labeller). Not used by the generator. |
| `objects` | yes | Things the gripper holds. |
| `locations` | yes | Named places. |
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

## steps (skills)

| `do` | Keys | Robot does |
|---|---|---|
| `home` | | Joint move to the cell's home joints |
| `move` | `to`, `height_mm` | Joint move to `height_mm` above a location (default `travel_height_mm`) |
| `pick` | `object`, `at`, `approach_mm` | Joint move above, straight down, close, wait, straight up |
| `place` | `object`, `at`, `approach_mm` | Joint move above, straight down to grasp + release gap, open, wait, straight up |
| `wait` | `seconds` | Wait (0..60 s) |

Every step may also have `id` (1, 2, 3 in order), `label` (one line of text, ends up as a
comment), `speed` (a speed name from the cell) and `demo` (`t_start`, `t_end` in the video).

Rules that make a task invalid: unknown keys, unknown skills, unknown names, placing what you
don't hold, picking while holding, ending while holding, numbers out of range.

## Cell file (cobot-cell/0.1)

Example: [cells/ur5e_table.json](cells/ur5e_table.json), [cells/so101_twin.json](cells/so101_twin.json).
Written by the engineer once per robot cell: `robot` (ur3e, ur5e, ur10e, so101), `work_frame`
(`xyz_mm`, `yaw_deg` in the robot base frame), `tcp_mm`, `payload_kg`, `gripper` (digital output
pin, or so101), `home_joints_deg`, `speeds` (named, capped at 120 deg/s and 500 mm/s),
`release_gap_mm`, `min_tool_height_mm`, `work_area_mm` (x min, x max, y min, y max).

## Planned for 0.2 (not built)

Digital inputs and outputs for machine tending (`wait_input`, `set_output`), side grasps (full
tool orientation), pallet patterns (one location, many slots), a FANUC TP generator.
