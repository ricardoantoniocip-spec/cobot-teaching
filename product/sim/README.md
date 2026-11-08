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
