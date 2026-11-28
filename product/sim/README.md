# SO-101 digital twin (product/sim)

A MuJoCo copy of the SO-101 follower on a table, with a block to pick and a green zone to place it.
Use it while the parts ship: the whole pipeline runs here first, then the same files go to the real arm.

```
scripted "operator" demo  ->  retarget (IK)  ->  twin replay + check  ->  episodes  ->  LeRobotDataset  ->  train ACT  ->  evaluate in twin
   (later: video, sleeve)      product/teach        so101_sim replay       demos        to-lerobot         lerobot-train     evaluate
```

The arm model is the official one (TheRobotStudio/SO-ARM100, same commit as
`hardware/so101/so101_new_calib.urdf`). It is downloaded on first use (16 MB, not in git).

## Commands (run from `product/sim`)

| Command | What it does |
|---|---|
| `python -m so101_sim check` | Twin vs retarget kinematics. Must say OK (under 0.1 mm). |
| `python -m so101_sim view` | 3D viewer + **Teach panel**. Drag the sliders in the viewer's "Control" panel to jog, record waypoints, o / c gripper, save, replay. `--port COM4`: the real arm follows. See "Teach mode" below. |
| `python -m so101_sim path paths/NAME.path.json` | Show a saved teach path and write its CSV and task file again |
| `python -m so101_sim scripted-demo demo.csv` | A pick-and-place hand demo, same format as a video demo |
| `python -m so101_sim replay traj.csv --video replay.mp4` | Plays a retarget trajectory in the twin, prints tracking, tool error and SUCCESS/FAIL |
| `python -m so101_sim demos 50 --out sim_data` | 50 random part/place positions, keeps the successful runs as training episodes |
| `python -m so101_sim from-video my_task.mp4` | Your own video to a twin replay and a training episode (see "From your own video") |
| `python -m so101_sim live` | LIVE: your arm leads, the robot follows, and you see all the tracking (see "Live") |
| `python -m so101_sim devices` | Says what this computer uses: graphics card, 3D view, camera |
| `python -m so101_sim to-lerobot sim_data USER/so101_sim_pick_place --root sim_data/lerobot` | Episodes to a LeRobotDataset (add `--push` to upload it, private, to your Hugging Face account) |
| `python -m so101_sim evaluate POLICY_DIR --trials 20` | Runs a trained policy in the twin and counts successes |

## Laptop setup (Windows, PowerShell), about 20 minutes

Use the same venv as product/teach (Python 3.12, because LeRobot 0.6.1 needs it).

```powershell
cd C:\dev\cobot-teaching
git pull
C:\dev\venvs\teach\Scripts\Activate.ps1          # or create it: py -3.12 -m venv C:\dev\venvs\teach
pip install -r product\teach\requirements.txt -r product\sim\requirements.txt

cd product\sim
python -m so101_sim check                         # downloads the model the first time
python -m pytest -q tests                         # 3 tests, about 5 s
python -m so101_sim view                          # play with the arm
```

### Step 1: one demo through the real pipeline (10 min)

```powershell
python -m so101_sim scripted-demo demo.csv
cd ..\teach
python -m retarget convert ..\sim\demo.csv ..\sim\traj.csv
cd ..\sim
python -m so101_sim replay traj.csv --video replay.mp4
```

Expect `task: SUCCESS` and a video of the arm moving the blue block onto the green zone.

### Step 2: make training data (about 30-60 min, leave it running)

```powershell
python -m so101_sim demos 50 --out sim_data
pip install "lerobot[dataset,training]==0.6.1"
hf auth login                                     # token from huggingface.co/settings/tokens (write)
python -m so101_sim to-lerobot sim_data ricardocipriano/so101_sim_pick_place --root sim_data\lerobot --push
```

### Step 3: train ACT in Colab (free T4 GPU, same as Tess lesson 2)

In lesson 2's notebook, change the dataset line to your own dataset:

```
!lerobot-train --dataset.repo_id=ricardocipriano/so101_sim_pick_place --policy.type=act \
  --policy.device=cuda --steps=20000 --batch_size=8 --output_dir=outputs/train/act_sim \
  --policy.push_to_hub=true --policy.repo_id=ricardocipriano/act_so101_sim --wandb.enable=false
```

### Step 4: score it in the twin (the capability study)

```powershell
hf download ricardocipriano/act_so101_sim --local-dir outputs\act_so101_sim
python -m so101_sim evaluate outputs\act_so101_sim --trials 20 --video-dir outputs\eval
```

Write the score in learning/progress.md. Pass mark, same as week 2 of the learning plan: 12/20.

## Live: your arm leads, the robot follows (15 Nov 2026)

One window. Left: the webcam with your body, both hands and the objects drawn on it. Right: the
robot in the twin doing what your arm does, and a panel with the numbers.

```powershell
python -m so101_sim devices                       # what will run on the graphics card
python -m so101_sim live                          # body tracking + twin
python -m so101_sim live --objects                # also find common things on the table
python -m so101_sim live --objects "juice carton" # or one thing by name
python -m so101_sim live --port COM5              # also drive the real SO-101 (when it is built)
python -m so101_sim live --video take.mp4 --no-window --save out.mp4 --log out.csv   # a recorded video
```

How to start: sit as you will work, **rest your working hand on the table and hold still for
2 seconds** (yellow bar). That sets your body directions, your arm length and the table height.
Then move. Open hand = gripper open, closed hand = gripper closed, anything in between too.
Your wrist steers the gripper: tip your fingers down or up and wrist_flex turns by the same angle;
turn your palm like a key and wrist_roll turns the same way. Zero is your hand as it rested during
calibration, and that resting place becomes the middle of the robot's working area (20 cm in front
of its base), so the robot has room to move in every direction. `--no-wrist` switches this off; `--roll-gain -1` or `--flex-gain -1` if one
turns the wrong way on your setup.

Clutch (default): the robot moves by as much as your hand moves, starting from where it stands, so
it never jumps (after calibrating, after a pause, or when your hand comes back into view). Press `g`
to let go: the robot holds while you move your hand somewhere comfortable, `g` again to carry on,
like lifting a mouse. `--absolute` switches the clutch off.

IK is exact (closed form for the SO-101's 5 joints: no search, no flipping between arm shapes);
`--ik numeric` brings back the general solver. `--pose-model lite` makes body tracking faster,
`heavy` more accurate. The twin is drawn every third picture to keep the camera loop fast.

Framing a laptop camera: sit 1 to 1.5 m back, screen tilted so both shoulders and your hand on the
table are in the picture, turned about 30 degrees so the working shoulder is not behind your arm,
light from the front.

Keys: `q` quit, `c` calibrate again, `SPACE` pause or continue the robot, `o` objects on or off,
`r` put the block back, `s` save a picture.

What the panel tells you:

| Line | Meaning | If it looks wrong |
|---|---|---|
| WAITING / CALIBRATING / FOLLOWING / HOLD | what the follower is doing | WAITING: your shoulder and arm must be in view for the 2 s calibration; after that only wrist and hand are needed. HOLD: the hand is not seen |
| sure of shoulder, elbow, wrist | how sure the tracker is of each point (1 = sure; `!` and a hollow red dot = under 0.5, a guess) | shoulder low: sit further back or turn a little |
| wrist tilt, roll | your hand's tilt and palm turn since calibration, degrees | jumps around: the hand is edge-on to the camera, turn it a little towards the camera |
| arm seen, hand seen, tracking ms | the body tracker on this picture | more light, plain sleeve, hand not edge-on to the camera |
| hand openness, gripper % | 0.9 flat hand, 0.45 fist | a wide part keeps the number high: that is the known weak spot |
| tool point x y z | where the robot's gripper is asked to go (mm, robot base) | AT EDGE OF REACH: your hand is outside the robot's scaled working area |
| IK miss, behind | how far the joints end from that point; how far the speed limit holds them back | "behind" stays large: you move faster than `--max-speed` |
| objects on cuda / cpu, one look takes | the object detector | on cpu a look takes 5 to 10 s, so the boxes are old (age is printed on each box) |

### Graphics card

- **Object detector**: runs on an NVIDIA card when torch was installed with CUDA. `devices` tells
  you. If it says CPU:
  `pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124`
- **Twin 3D view**: MuJoCo draws with the graphics card by itself.
- **Body tracking** (MediaPipe): CPU. Its Python package has no graphics card mode on Windows, and it
  does not need one (tens of milliseconds per picture).

### The real arm (`--port`), UNTESTED until the servos arrive

It starts PAUSED: the twin follows you, the arm does not. Press `SPACE` to let the arm follow,
`SPACE` again to stop it. Joints are capped at 45 deg/s (`--max-speed`). On quit it goes back to
where it started, then switches off (it goes limp: keep a hand near it). First do
`hardware/so101/first-power-up.md`.

### Limits

- Positions are scaled from your arm, not true millimetres. One camera is weak in depth, so
  reaching towards or away from the camera is the least accurate direction.
- The SO-101 has 5 joints. Its gripper points down near the table and tilts forward as your hand
  goes up; your wrist tilt adds to that and your palm turn rolls it. It cannot turn sideways
  (yaw) the way your hand can, and near the edge of the reach position wins over tilt.
- Wrist tilt and roll come from the hand tracker's 3D estimate, which is weakest when the hand is
  edge-on to the camera.
- Tested on take 1 as a recorded video (arm seen in 100% of pictures, hand in 82%, IK miss under
  4 mm) and on made-up landmarks. Not yet tested with a live webcam, on Windows, or on a graphics card.

## From your own video (12 Nov 2026)

Film yourself doing a pick and place, and get a checked training episode. No markers on your body.

```
your video -> arm tracking (MediaPipe) -> hand open or closed -> SO-101 joint angles (IK) -> twin replay -> episode -> LeRobotDataset
```

One-time setup, in the same venv:

```powershell
pip install -r product\teach\requirements-pose.txt     # MediaPipe, runs on the CPU (no GPU needed)
```

Then, from `product\sim`:

```powershell
python -m so101_sim from-video C:\path\to\my_task.mp4 --arm right
```

MediaPipe prints many `INFO:` and `W0000` lines. Ignore them. It prints three stages and writes a folder `video_data\my_task\`:

| File | What to look for |
|---|---|
| `tracked.mp4` | Your video with the arm drawn on it. The dot on your hand turns red when your hand is closed, with the openness number next to it. |
| `arm_track.csv` | Your elbow, wrist and grasp point per frame (metres from your shoulder), elbow angle, hand openness, grip. |
| `steps.json` | The demo cut into steps: reach, grasp, move, release, retreat. `label` is empty: a vision model fills it later. |
| `replay_copy.mp4` | The robot following your hand path sample by sample. |
| `replay_steps.mp4` | A clean robot motion rebuilt from where you gripped and released. This one becomes the episode. |
| `result.json` | Block and zone positions, success of both replays. |

Each video that succeeds adds one file to `video_data\episodes\`. Film 30-50 takes with the block in
different places, then:

```powershell
python -m so101_sim to-lerobot video_data\episodes ricardocipriano/so101_video_pick_place --root video_data\lerobot
```

and train ACT on it exactly as in steps 3 and 4 below.

### How to film

- Phone or webcam on a tripod or a stack of books. It must not move.
- Your whole working arm in view the whole time: shoulder, elbow, wrist and hand. Both shoulders visible.
- Sit facing the camera or at an angle to it. Not with your back to it.
- Open hand = not holding. Closed hand = holding. Use any grip you like (whole hand is fine), but start with a flat open hand and open it flat again after you let go.
- One pick and one place per video, 5-15 seconds. Start and end with an open hand.
- Good light, sleeves that contrast with the table.

### How the grasp is found

The app measures how open your hand is in every frame: 0.9 is a flat hand, 0.45 is a fist. Under
0.78 for at least 0.4 s is a closed hand, and it only counts as a grasp when the hand is clearly
closed (typically under 0.70). The moment it closes is the grasp, the moment it opens is the release.

### Best: tell the app what you move

```
pip install -r ../teach/requirements-vision.txt     (once; torch and transformers)
python -m so101_sim from-video my_task.mp4 --arm right --object "juice carton"
```

An object detector (LLMDet tiny, an open model from Hugging Face, about 700 MB on first use)
finds the object by name every half second, also while your hand is on it. Where the object
stands still it rests; between two resting places it is being moved. That gives the grasp, the
release and the pick and place spots, for every time you move it, and the app checks that your
tracked hand is on the object at each grasp. Check `object_track.jpg`. The name, the times and
the boxes go into `steps.json`, and from there into the task file and the robot program
(`python -m taskprog from-steps`).

About 5 s per picture on a laptop CPU, so about 3.5 minutes for a 17 s video; a GPU is about ten
times faster. Limits: one object per video, fixed camera, plain words work best ("juice carton",
"metal bracket"). The twin replays the first pick and place only. Tested on take 1 (both moves
found, twin SUCCESS) and on made-up videos.

### Without a name

A wide part (a carton, a bottle) keeps your fingers almost straight, so the hand never clearly
closes. The app then says so and looks at the object instead, with no extra option needed: first
with the detector (it looks for common things and takes the one that is moved; on take 1 it finds
the carton and calls it "bottle"), and if torch is not installed, with a plain picture comparison.
You can also ask for the picture comparison directly:

```
python -m so101_sim from-video my_task.mp4 --arm right --grips object
```

It looks for the object where it stood at the start, finds where it stands later, and takes the
moments it leaves and arrives as grasp and release. Check `object_motion.jpg` in the output
folder. Limits: one object, one move, fixed camera; if the hand hides almost nothing of the
object, the release can be found too early. Tested on take 1's overlay video and on made-up videos.

If both fail, set the times by hand: open `tracked.mp4`, read the time printed top left when you
take hold and when you let go, and pass them. `--end` cuts off the part where you reach for the camera.

```powershell
python -m so101_sim from-video my_task.mp4 --arm right --grips 3.0-8.5 --end 9.5
```

### What the numbers mean (body mode, the default)

Your shoulder is placed on the robot's shoulder and your arm is scaled to the robot's reach (about
half size). The lowest point of your hand is taken as the table. So the robot does the same motion,
smaller, and the twin puts the block where your hand closed. Positions are not true millimetres.

For the real arm and real parts you need true positions: tape a printed ArUco marker on the table at a
measured place and add `--markers markers.json --camera camera.json --hand-width-mm 85`
(see product/teach/README.md for the two files). This mode is unit tested but not yet tried on a real video.
Together with `--object "name"` the marker also gives the object's own place on the table in mm
(pick and place spot, rough size), and the task file uses those instead of the hand's position.

### Limits of this first version

- Tested on made-up landmarks through to a successful twin episode, and on one real video (12 Nov 2026,
  laptop webcam, 15 frames a second, a juice carton moved from the sofa armrest to the seat). Arm and
  hand were found in 129 of 141 frames. The grip was not found automatically (whole-hand grasp), so the
  times were set with `--grips`. `steps` then succeeded in the twin, `copy` did not (80% of the path
  reachable, and the pick was 9 cm above the "table").
- One camera cannot see depth well. Expect a centimetre or two of error, which is why `copy` often
  misses the block and `steps` is used for training.
- One grasp and one release per video. The gripper always points down.
- The robot follows your hand, not your elbow. A human arm has 7 joints and the SO-101 has 5, so the
  joint angles cannot be copied one to one.

## What is tested (8 Nov 2026, in the cloud, Linux, Python 3.13)

- `check`: twin and retarget agree on the tool point to 0.004 mm over 200 random poses.
- Scripted demo -> retarget -> twin: 20 of 20 random positions succeed (part 18-28 cm in front, 5-15 cm to the side).
  An earlier version without a fixed home pose got 17/20: failures at the far edge of the reach, where IK missed by
  2-3 cm or the block tipped over. Useful to know for the real cell.
- **Correction, 15 Nov 2026:** those 20 "successes" ended with the block lying on its side. The score only checked
  where the block ended. The gripper dragged the block over when it went up after the release. Fixed: the finger
  now moves 8 mm off the part first, and a run fails when the part falls over. Re-run with the fix, same 20 positions: 20 of 20 upright in the zone, 10 mm typical from the centre (it was about 22 mm).
  **Training data made before this date teaches the tipping release: make it again (`demos 50`) and retrain.**
- Servo tracking in the twin: worst 0.3 degrees behind the command.
- `to-lerobot`: made a LeRobotDataset v3 with lerobot 0.6.1, loaded it back, and ran 20 ACT training
  steps on CPU (loss falling). `evaluate` ran a trial with that 20-step policy (it fails, as expected).
- Not tested: Windows, a full ACT training run, `--push`, the Colab step.

## Known limits

- The scripted demo is perfect and always grips the same way. Real demos vary more. Useful for learning the
  tools, not proof that the method works.
- Sim units match LeRobot's SO-101 follower (degrees, gripper 0-100). This assumes the twin's zero (middle of each
  joint range) matches your LeRobot calibration. Check with `retarget check` when the arm is built.
- Grip physics in MuJoCo is approximate (convex hulls of the jaw meshes). The grasp offsets in
  `so101_sim/scripted.py` were tuned in the twin and will need a real-arm check.
- The top camera is an ideal camera. For sim-to-real, mount your real camera at the same place
  (75 cm above the table, 22 cm in front of the base) or move `top` in `scene.py` to match yours.


## Teach mode (like a teach pendant)

```
python -m so101_sim view                  # twin only
python -m so101_sim view --port COM4      # real arm follows the twin (SPACE = follow / pause)
```

One window: the pendant with the 3D twin inside (drag = turn, right-drag = move, wheel = zoom, double-click = reset;
green dots = taught waypoints, orange = gripper step there). `--viewer` also opens MuJoCo's own viewer.

1. Press **TEACH** (turns green).
2. Jog with the sliders or the − / + buttons in the **Jog** card (hold + to keep moving). **Record waypoint** (or `w` / Enter) stores the pose as a MoveJ.
3. `o` opens, `c` closes the gripper; in Teach mode that is a step in the path.
4. Type a name, **Save**. Files in `paths\`: `NAME.path.json` (the path), `NAME.csv` (joint trajectory for
   `python -m retarget replay`), `NAME.task.json` (task file for the URScript / SO-101 generator, when the path picks and places).
5. **Load...** a path, **Play in twin** (real arm holds still), **Run on robot** (asks first; real arm follows the twin,
   capped at `--max-speed`, default 60 deg/s). **STOP** / Esc halts both.

**Freedrive** (with `--port`): press **Freedrive** (or `f`) in the Teach panel, like the freedrive button on a UR.
Hold the arm first: the arm motors switch off and it has no gravity hold. Move it by hand, the twin copies it,
**Record waypoint** stores the pose, `o` / `c` still drive the gripper. Press Freedrive again: the arm holds where you left it.

Closing the window: the real arm first moves slowly (20 deg/s) back to the pose it had when you started (put it in the
folded rest pose before starting), then the motors switch off. A running program also waits for the real gripper to
finish closing before it moves on.

**Set box at gripper** (`b`): hold the gripper tip just above the real box (don't press on it) and press the button:
the twin's box jumps to that x, y so the twin matches your bench. Saved with the program and restored on load.

**Soft grip** (real arm): closing goes slowly and stops when the jaws meet the part (the jaw lags the command, or the
motor load passes `--grip-load`, default 150 of 1000, measured on the real gripper 2026-11-28: empty close 84, sponge 180, box 200). It then squeezes a little past the contact point and holds.
Find the right number first with the gripper-only test (from product\teach): `python -m retarget grip-test --port COM4`.
The Jog card shows "part held at N %" and the load. Without a part it closes all the way. Same during Run on robot.

Replay is MoveJ: all joints start and stop together, peak joint speed = the Speed box (default 40 deg/s). The gripper
moves only while the arm stands still. Undo removes the last step. `--plain` gives the old viewer without the panel.
