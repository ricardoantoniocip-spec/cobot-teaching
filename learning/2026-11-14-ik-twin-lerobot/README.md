# Lesson 6: IK, the twin, and a LeRobot episode

Series "what we built", part 4 of 4. About 15 minutes reading, 30 minutes exercises. Laptop, no GPU.

## Results first
- IK puts the SO-101's tool point on take 1's grasp point with 0.00 mm error, fingers straight down.
- Take 1's "steps" motion replayed in the MuJoCo twin: SUCCESS, block 22 mm from the zone centre, joints at most 0.5 degrees behind the command.
- That replay becomes one training episode: 355 frames (11.8 s at 30 fps), 6 joint numbers for state and for action, and a 320 x 240 top-camera image per frame.
- Today it was turned into a LeRobotDataset with lerobot 0.6.1 and loaded back (one episode, 355 frames). This is the format ACT trains on (lesson 2).
- Take 1's "copy" motion (the robot following your hand sample by sample) failed: only 80% of the path was reachable and the pick was 9 cm above the "table".

## The chain

```
 hand path (demo.csv)  or  clean "steps" path
          |
 retarget: IK per sample, smooth, cap speed
          |   (kinematics.py, retarget.py)
 traj.csv: 5 joints + gripper, 30 Hz
          |
 MuJoCo twin replay (replay.py)
          |   success = block within 30 mm, resting
 episode_000.npz (state, action, images)
          |   only successes are saved
 to-lerobot -> LeRobotDataset
          |
 ACT training (Colab) -> evaluate in the twin
```

## 1. SO-101 kinematics, from the URDF
`product/teach/retarget/kinematics.py`. No link length is typed in the code: every joint comes from the official URDF file `hardware/so101/so101_new_calib.urdf` (lines 1-4, 79-97).
- A URDF joint has an **origin** (a fixed 4x4 from the parent link) and an **axis**. It is the same job as DH parameters, written differently: DH squeezes each link into 4 numbers, URDF gives the full offset and the axis.
- **FK** = multiply origin, rotate about the axis by q, next joint (lines 106-117).
- **Jacobian** column i = (z_i x (p - o_i), z_i) (lines 119-127). The geometric Jacobian from your robotics minor.
- 5 joints: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll (line 16). Limits are read from the URDF too.
- The twin uses the same model: `python -m so101_sim check` compares them at 200 random poses. Today: worst 0.0037 mm.

## 2. IK: damped least squares
`solve_ik` (lines 153-185), one step:

```
dq = Jᵀ (J Jᵀ + λ² I)⁻¹ e        λ = 0.01
```

- `e` = position error (weight 1) and orientation error (weight 0.05) (line 168). Position matters most.
- Each step is capped at 0.2 rad and clipped to the joint limits (lines 180-182).
- Why "damped": near a singularity the plain inverse asks for huge joint moves. The λ² term trades a little accuracy for a calm answer. Like limiting a controller's gain near instability.
- `solve_ik_with_restarts` (lines 188-207): if the miss is over 2 mm, try up to 8 random starts and keep the best.

**What 5 joints can and cannot do** (exercise 1 shows both):
- With the fingers pointing straight down, `wrist_roll` turns the jaws about the vertical. So a 45 degree jaw angle is reachable.
- Tilting the fingers sideways, out of the vertical plane through the base, is not. IK then compromises position and orientation.
- The code always closes the jaws along the line from the base (`tool_down`, lines 210-220; `taskprog/so101.py:40-50`). That matches the grasp geometry tuned in the twin. It is a choice, not a hard limit for top-down picks: a jaw angle (the task file's `yaw_deg`) could be supported later.
- A solver has failure modes too. With a joint at its limit, this IK can stall far from the best answer (exercise 1, last question).

## 3. retarget: a hand path becomes a joint trajectory
`retarget` (`product/teach/retarget/retarget.py:44-65`):
1. For each sample: tool pose = hand pose x `hand_to_tool` (identity until calibrated against the leader arm, line 21), then IK, warm-started from the previous sample (lines 48-53).
2. 5-sample moving average on the joints (line 54, lines 68-77).
3. Stretch time so no joint goes faster than the cap (line 55, lines 80-87): 60 deg/s in the video app (`product/sim/so101_sim/from_video.py:72`), 30 deg/s by default for the first real-arm replays.
4. Gripper open when grip < 0.5 (line 56).
5. Report: share of samples within 5 mm, worst error, how much slower than the demo.

## 4. Two robot motions per video: "copy" and "steps"
`demo_to_episode` (`product/sim/so101_sim/from_video.py:56-97`):
- **copy**: retarget your own hand path (lines 75-77). Shows how well your motion carries over.
- **steps**: a clean scripted path through only two points from your video, the grasp and the release (lines 79-81). The path is `scripted.pick_place_demo` (`product/sim/so101_sim/scripted.py:53-86`): home, 50 mm above, down to 8 mm above the table, close, up, across, down, open, up, home.
- The block is placed where your hand closed, pushed out 16 mm (half the 20 mm block plus half a finger), because the SO-101's tool point is the tip of the fixed finger (`scripted.py:39-42`).
- **Only `steps` is saved by default** (line 86).

Be honest about what that means: in body mode, a "video episode" keeps **where** you picked and placed, not **how** you moved. ACT trained on such episodes learns the scripted motion style. Your own style only enters via `--use copy`, which fails more often until tracking improves (table-marker mode, the sleeve).

## 5. The twin replay: a go/no-go gauge for every demo
`replay` (`product/sim/so101_sim/replay.py:49-105`):
- MuJoCo physics in 2 ms steps (`scene.py:30`), commands at 30 Hz like the real arm (line 69).
- Each frame it records **state** = the simulated joint angles, **action** = the commanded angles (lines 76-77), and a top-camera image (camera `top`, 75 cm above the table, `scene.py:47`).
- Success = block ends within 30 mm of the zone centre **and** rests on the table, not stuck in the gripper (lines 95, 103).

Factory picture: virtual commissioning before the real cell, used as an end-of-line gauge. Only good parts (successful replays) go into the dataset.

## 6. The episode file and LeRobot
`product/sim/so101_sim/episodes.py`:
- Units match LeRobot's SO-101 follower: joints in degrees, gripper 0 (closed) to 100 (open), names like `shoulder_pan.pos` (lines 18-28). So a policy trained in the twin sends the same kind of numbers as one trained on real demos.
- Assumption to check on the real arm: the twin's zero (middle of each joint range) equals your LeRobot calibration zero (docstring, lines 1-7).
- `npz_to_lerobot` (lines 44-67): one LeRobotDataset with `observation.state` (6), `action` (6), `observation.images.top` (video), a task sentence ("Pick the blue block and place it on the green zone.", line 20), fps 30, `robot_type` so101_follower.

## 7. How it feeds ACT
Recall [lesson 2](../2026-10-31-act-action-chunks/lesson.md): ACT takes one camera image plus the joint state and predicts the next 100 actions (a chunk). At 30 fps that is 3.3 s of motion. Our dataset has exactly those keys. The full recipe is in `product/sim/README.md`, steps 2-4:

```powershell
python -m so101_sim demos 50 --out sim_data          # or 30-50 video takes
python -m so101_sim to-lerobot sim_data ricardocipriano/so101_sim_pick_place --root sim_data\lerobot --push
# Colab, T4 GPU, lesson 2's notebook:
#   lerobot-train --dataset.repo_id=... --policy.type=act --steps=20000 ...
python -m so101_sim evaluate outputs\act_so101_sim --trials 20
```

`evaluate` is the capability study: 20 trials at new random positions, count the successes. Pass mark 12/20.

## Exercises (on your laptop, 30 minutes)

```powershell
cd C:\dev\cobot-teaching
C:\dev\venvs\teach\Scripts\Activate.ps1
python -m pip install -r product\sim\requirements.txt   # mujoco, if not yet installed
cd product\sim; python -m so101_sim check; cd ..\..      # downloads the arm model once
mkdir C:\dev\tmp\lesson6 -Force; cd C:\dev\tmp\lesson6
```

### Exercise 1: FK and IK on take 1's grasp point
```powershell
python C:\dev\cobot-teaching\learning\2026-11-14-ik-twin-lerobot\ex1_ik.py
```

Expected (checked):

```
2. FK all zero: joints [0.0, 0.0, 0.0, 0.0, 0.0] -> tool point [391, 0, 226] mm, finger axis [1.0, 0.0, 0.0]
2. FK home    : joints [0.0, -21.8, 29.2, 82.6, 0.0] -> tool point [180, 0, 60] mm, finger axis [0.0, 0.0, -1.0]
3. IK to take 1's grasp point [176, 46, 81] mm, yaw 14.7 deg (radial)
   joints [-18.3, -20.0, 18.7, 91.3, -3.7]
   misses by 0.00 mm and 0.0 deg
4a. Same point, jaws turned 45 deg about the vertical: misses by 0.0 mm and 0.0 deg. wrist_roll -3.7 -> 39.1 deg
4b. Same point, fingers tilted 30 deg sideways: misses by 16.5 mm and 29.5 deg: a compromise, orientation counts only 0.05.
5. Point 450 mm in front: misses by 100 mm (retarget counts a sample as reached within 5 mm)
```

Now edit part 5: change `0.45` to `0.33`, run, then `0.30`. The nearer point should be easier. Is it? (Checked: 0.33 misses by 5.0 mm, 0.30 by 32.8 mm.) Look at the wrist_flex limit in part 1 and guess why. The answer is at the bottom.

### Exercise 2: take 1's training episode, made in the twin
About 1-2 minutes.

```powershell
python C:\dev\cobot-teaching\learning\2026-11-14-ik-twin-lerobot\ex2_episode.py
```

Expected (checked):

```
block at [0.1918, 0.0501] m, place zone at [0.2379, -0.0779] m (from take 1 result.json)
1. scripted path: 325 samples, 10.8 s, grip closed for 4.8 s
2. retarget: 325 samples, 100% reached within tolerance, worst error 0.0 mm, replay 1.00x demo duration
3. twin replay:
   joint tracking, worst: 0.5 deg
   tool point vs IK, worst: 1.5 mm
   part moved 147 mm, ends 22 mm from the place zone centre
   task: SUCCESS
4. saved ...\l4_out\episode_000.npz and replay_steps.mp4
```

Watch `l4_out\replay_steps.mp4`. Then move the place zone outside the tested area: in the script, change `np.array(res_json["place_xy"])` to `np.array([0.30, -0.20])` and run again. Expected (checked):

```
2. retarget: 325 samples, 58% reached within tolerance, worst error 41.1 mm, replay 5.75x demo duration
   part moved 175 mm, ends 99 mm from the place zone centre
   task: FAIL
4. not saved: the replay failed
```

Why is the replay 5.75x slower? (Hint: section 3, step 3.) Change it back before exercise 3.

### Exercise 3: inside the episode
```powershell
python C:\dev\cobot-teaching\learning\2026-11-14-ik-twin-lerobot\ex3_inspect_episode.py l4_out\episode_000.npz
```

Expected (checked):

```
1. episode_000.npz: 355 frames at 30 fps = 11.8 s
   state   (355, 6) float32
   action  (355, 6) float32
   images  (355, 240, 320, 3) uint8
2. Ranges over the episode
   shoulder_pan.pos   state   -18.3 ..   21.5   action   -18.3 ..   21.5
   gripper.pos        state     0.4 ..   74.5   action     1.3 ..   71.6
3. shoulder_lift: state matches the action best 1 frame(s) later (33 ms). ...
4. gripper command crosses 50 at frames [90, 234] = [3.0, 7.8] s
5. LeRobotDataset features: observation.state (6), action (6), observation.images.top (240, 320, 3) video, ...
```

If lerobot 0.6.1 is installed in the venv, part 5 also builds `l4_out\lerobot` and prints `built ...: 1 episode, 355 frames`. Without it, the script says how to install it. Compare part 3 with your lesson 1 answer: why does the action lead the state?

## Self-check
1. Why does the IK weight orientation at only 0.05?
2. A video episode was saved from the "steps" motion. Which two numbers from your video does it really contain?
3. The twin replay succeeds but the block ends 28 mm from the zone. Would you keep that episode for training? What would you check?

## For this afternoon's take
For training data you need **variety**: if the take works, film 5 more with the part in different places on the table (within reach of a robot at the camera's left or right), not all in one spot. Keep each 5-15 s, one pick and one place, and run `from-video` on each so only successful replays land in `video_data\episodes`.

---

## Answers
1. The SO-101 has 5 joints, so it cannot match every position **and** orientation. A low weight lets position win: the tool reaches the point and gets as close as it can to the asked orientation.
2. The grasp point and the release point (each x, y; the height is set by the script). The motion between them is scripted.
3. It passes (under 30 mm), so the code keeps it. But 28 mm is near the limit: check `replay_steps.mp4` for a block that slid or tipped, and check whether the pick or place point was outside the tested area (x 0.18-0.28 m, |y| 0.05-0.15 m). Like a part just inside tolerance: accept, but look for the cause before it drifts out.

Exercise questions:
- Exercise 1, 0.30 vs 0.33: the IK got stuck with wrist_flex pinned at its 95 degree limit. Clipping a joint at its limit can stop damped least squares from finding the better answer: a 0.1 mm solution with the fingers 4 degrees off vertical exists. The solver is a tool with failure modes; `check` and the twin are there to catch them.
- Exercise 2, 5.75x slower: near the edge of reach the IK solution jumps between samples, and `limit_speed` stretches the whole trajectory until no joint goes over 60 deg/s.
