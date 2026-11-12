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
| `python -m so101_sim view` | 3D viewer. Open the "Control" panel on the right and drag the sliders to move each joint. |
| `python -m so101_sim scripted-demo demo.csv` | A pick-and-place hand demo, same format as a video demo |
| `python -m so101_sim replay traj.csv --video replay.mp4` | Plays a retarget trajectory in the twin, prints tracking, tool error and SUCCESS/FAIL |
| `python -m so101_sim demos 50 --out sim_data` | 50 random part/place positions, keeps the successful runs as training episodes |
| `python -m so101_sim from-video my_task.mp4` | Your own video to a twin replay and a training episode (see "From your own video") |
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

## From your own video (12 Nov 2026)

Film yourself doing a pick and place, and get a checked training episode. No markers on your body.

```
your video -> arm tracking (MediaPipe) -> pinch point + grip -> SO-101 joint angles (IK) -> twin replay -> episode -> LeRobotDataset
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
| `tracked.mp4` | Your video with the arm drawn on it. The dot between thumb and index turns red when it sees a grip. |
| `arm_track.csv` | Your elbow, wrist and pinch point per frame (metres from your shoulder), elbow angle, grip. |
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
- Grip with thumb and index finger, like the robot's two-finger gripper. Open your hand wide between grips.
- One pick and one place per video, 5-15 seconds. Start and end with an open hand.
- Good light, sleeves that contrast with the table.

### If the grip is not found

The automatic grip only sees a thumb-and-index pinch. A whole-hand grasp of a wide part (a carton, a
bottle) looks the same as a relaxed hand. Then set the grip by hand: open `tracked.mp4`, read the time
printed top left when you take hold and when you let go, and pass them. `--end` cuts off the part
where you reach for the camera.

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
