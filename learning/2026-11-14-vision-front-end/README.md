# Lesson 3: The vision front end (video to hand path)

Series "what we built", part 1 of 4. About 15 minutes reading, 30 minutes exercises. Laptop, no GPU.

## Results first
- Take 1 (12 Nov, sofa, juice carton): your right arm and hand were found in 129 of 141 frames.
- The grip was **not** found. You held the carton with the whole hand, so the thumb-index gap stayed wide (the overlay shows "pinch 0.97" while you hold it).
- So the grip times were typed by hand (`--grips`). Then the clean "steps" replay succeeded in the twin (block 22 mm from the zone). The "copy" replay, which follows your hand sample by sample, failed.
- This lesson explains every one of those words, from the code.

## The whole pipeline (all 4 lessons)

```
 your video (phone/webcam)
      |
 L3  MediaPipe: 33 body + 21 hand points
      |   pose.py track_video
 L3  pinch point, grip ratio, scaling
      |   pose.py frames_to_demo
      |   -> demo.csv (hand path, robot frame)
 L4  cut at grip changes
      |   pose.py segment_steps
      |   (or --grips / object motion)
      v
   steps.json
      |                 \
 program path         training path
      |                    |
 L4  vision LLM       L6  IK (retarget)
     labels.json           |
      |               L6  MuJoCo twin replay
 L5  task.json             |  only successes kept
     + cell.json           |
      |               L6  episode_000.npz
 L5  check                 |
      |               L6  LeRobotDataset
 L5  URScript / SO-101     |
     URSim / twin          v
                      ACT (lesson 2)
```

L3 = this lesson, L4 = steps, L5 = robot programs, L6 = IK, twin and LeRobot. File: line references below are in `product/teach/retarget/pose.py` unless another file is named.

## 1. What a landmark is
A landmark is a named point on the body that a neural network **estimates** from the picture. No sticker, no marker. MediaPipe has two models, both run on the CPU:
- **Pose**: 33 body points. We use 3 per arm: shoulder, elbow, wrist (numbers 12, 14, 16 for your right arm, lines 42-45).
- **Hand**: 21 points per hand. We use wrist (0), thumb tip (4), index knuckle (5), index tip (8), little-finger knuckle (17) (line 49).

Factory picture: like an inspection camera finding fiducials, except the "fiducials" are guessed from what a shoulder or a fingertip usually looks like. So every point is a measurement with noise, not a fact.

The two models are created in `track_video` (lines 128-133), in VIDEO mode: each frame is tracked using the previous one, which is steadier than treating each frame alone. Model files are downloaded once (lines 36-39, 100-111).

## 2. Image coordinates vs world coordinates
Each point comes twice (dataclass `ArmFrame`, lines 54-64):

| | Units | Origin | Used for |
|---|---|---|---|
| image (`pose_px`, `hand_px`) | 0-1 of the image width/height, times pixels (line 157) | top-left of the image | picking the right hand, drawing, PnP |
| world (`pose_world`, `hand_world`) | metres | body: midpoint of the hips. Hand: centre of the hand | all geometry: angles, lengths, pinch point |

Sources: [Pose landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker), [Hand landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker).

Catch: with one camera, depth is a **learned guess**. The model has seen many people and estimates "the elbow is probably 20 cm closer to the camera". Lengths in world coordinates are good to a few centimetres, not millimetres.

Your kinematics habit applies: camera axes are x right, y down, z away from the camera (line 56). The code converts to your body's axes (section 5) before anything goes to the robot.

## 3. Confidence, and what fails
- Every body point has a **visibility** 0-1. Shoulder, elbow and wrist must all be at least 0.5 or the frame is dropped (line 47, line 156).
- The hand model does not know whose arm a hand belongs to. The code takes the hand whose wrist is nearest the arm's wrist, and only if it is within 0.6 x the forearm length in pixels (`_pick_hand`, lines 190-200). It ignores the model's "Left"/"Right" label, because a mirrored video flips it (the MediaPipe docs say the same).
- Frames with no usable hand are filled by straight lines from the neighbours; gaps over 0.5 s give a warning (lines 365-372).

What fails in practice (you will see some in exercise 1):
- **Side view, forearm pointing at the camera.** In take 1 at 4.0 s your forearm is only about 100 px long in the image, so the "near the wrist" gate is only 60 px.
- **Far arm hidden.** Your left elbow had visibility 0.1-0.4 in take 1. Fine, because the right arm did the task.
- **Hand behind the object**, motion blur (take 1 was 15 frames per second), poor light.
- **Drawn lines on the video.** `tracked.mp4` has the skeleton painted on your hand; the hand model then misses that hand. Always feed the original video.
- **Whole-hand grasp** (next section).

## 4. Pinch point and grip ratio
The robot follows one point: the **pinch point**, halfway between thumb tip and index tip (line 330). Why not copy your arm angles? A human arm has 7 joints, the SO-101 has 5. They do not map one to one. The tool point does.

The grip comes from the **grip ratio** (lines 238-241):

```
ratio = |thumb tip - index tip| / |wrist - index knuckle|
```

Dividing by palm length makes it scale-free: a big hand, a small hand, near or far from the camera, the same number. About 1 = open, under 0.5 = pinching.

Two thresholds, not one (lines 244-253, settings lines 70-71):
- closes when the ratio drops **below 0.5**,
- opens when it rises **above 0.7**,
- in between, it keeps the last state.

That is a Schmitt trigger, or a dead band on a level switch. One threshold at 0.6 would chatter every time tracker noise crosses it. Before that, a 5-sample running median removes single bad frames (line 380), like a moving-median chart ignoring one outlier.

**Why take 1 failed:** you wrapped your hand round a 70 mm carton. Thumb and index tips stayed far apart, so the ratio stayed around 0.9-1.0 the whole time. No grip, so no grasp, so the app stopped with "no grasp followed by a release" (`product/sim/so101_sim/from_video.py:60-62`). The fix was to type the times: `--grips 3.0-8.5` (lesson 4 explains this and the automatic "object" option).

## 5. Body mode (the default): your arm, shrunk to the robot
Plain video, no markers. Steps in `frames_to_demo` (lines 300-408):
1. **Your own axes.** Forward, left and up are built from your shoulder line and hips-to-shoulders, as medians over the whole video (`body_axes`, lines 256-273). So turning the camera does not move the robot path (exercise 3 proves it).
2. **Arm relative to your shoulder**, in those axes (lines 322-331).
3. **Scale.** Your arm length (upper arm + forearm + wrist-to-pinch, medians) is mapped to 0.33 m robot reach (lines 352-355, setting line 80). Your shoulder sits on the robot's shoulder-lift joint (line 356).
4. **Shrink to fit.** If the path goes over 0.27 m from the robot base, the scale drops until it fits (lines 357-362).
5. **Table height.** Your lowest hand point (2nd percentile) becomes 8 mm above the table (line 363).

So positions are **not true millimetres**. It is "the same motion, about half size". In take 1 the grasp came out 81 mm above the "table": the armrest is higher than the lowest point your hand reached (the seat).

## 6. Table-marker mode (for the real cell)
For true positions, tape a printed ArUco marker on the table at a measured place and pass `--markers markers.json --camera camera.json --hand-width-mm 85`.
- **ArUco** = a square black-and-white code. Its 4 corners are found in the image. **PnP** (Perspective-n-Point) solves "where must the camera be for these 4 known corners to land on these pixels?" That gives camera-to-table as a 4x4 transform (`video.py:73-82`, `cv2.solvePnP`).
- The same PnP trick runs on the hand: 21 hand points in metres vs their pixels give hand-in-camera (lines 276-285). The hand's real size sets the distance, so tell it your knuckle width (lines 337-341). Without it, expect 10-15% scale error (line 341).
- Then a chain of transforms you know from your robotics minor (line 348):

```
T_base_pinch = T_base_table · inv(T_cam_table) · T_cam_pinch
```

- Without `camera.json`, the lens is guessed from a 60 degree field of view (`video.py:37-41`). Another error source: calibrate the webcam once.
- Status: unit tested, not yet tried on a real video.

## 7. Smoothing
- Hand position: gaps filled, then a 0.2 s centred moving average (lines 372-374, setting line 72).
- Resampled to 30 Hz, the robot's rate (lines 387-389).
- Tool always pointing down, jaws facing along the line from the robot base (line 390).
- Grip: last seen value fills gaps, 5-sample median, then the two thresholds (lines 377-384).
- Later, IK joint angles get one more 5-sample moving average (`retarget.py:68-77`).

## Exercises (on your laptop, 30 minutes)
Setup once (same venv as the twin):

```powershell
cd C:\dev\cobot-teaching
C:\dev\venvs\teach\Scripts\Activate.ps1
pip install -r product\teach\requirements-pose.txt
```

### Exercise 1: one frame through MediaPipe
Use your **original** take 1 video. (Or `tracked.mp4` from the project files, video-app/take1, to see the failure.) First run downloads 17 MB of models.

```powershell
python learning\2026-11-14-vision-front-end\ex1_landmarks.py C:\path\to\take1.mp4 --t 4.0
```

Ignore the `INFO`/`W0000`/`WARNING` lines. Expected on `tracked.mp4` at 4.0 s (checked):

```
shoulder #12  image (0.314, 0.338) = ( 402,  244) px  world (+0.234, -0.358, -0.227) m  visibility 1.00
elbow    #14  ...                                      visibility 0.95
wrist    #16  ...                                      visibility 0.87
lowest visibility 0.87 -> USED (pose.py needs 0.5)
elbow angle 140 deg (180 = straight arm)
hand 0: model calls it 'Left', wrist at (461, 482) px, 202 px from your right wrist (limit 60 px = 0.6 x forearm)
-> no hand accepted for the right arm in this frame
```

That "hand 0" is your resting left hand. The right hand is missed because of the painted skeleton. On the original video you should get "hand accepted" and a grip ratio near 0.9-1.0 at 4.0 s. Try `--t 2.0` and `--t 6.0` as well. Look at `ex1_frame.jpg`.

Write down: the grip ratio at 4.0 s on your original video, and the forearm length in pixels.

### Exercise 2: grip ratio, two thresholds, whole hand
No video needed.

```powershell
python learning\2026-11-14-vision-front-end\ex2_grip_ratio.py
```

Expected (checked):

```
B. Slow close and open, with tracker noise (sd 0.06)
   one threshold (0.6):       grip changes 8 times
   two thresholds (0.5/0.7):  grip changes 2 times  (one pick = 2)
C. Whole-hand grasp (thumb-index gap fixed at 80 mm)
   grip ratio min 0.84, max 0.84: never below 0.5
   steps found: reach
D. Same video with --grips 3.0-6.5
   steps found: reach > grasp > move > release > retreat
```

Then edit part B: set `NOISE_SD = 0.12`. Does the two-threshold version still give 2?

### Exercise 3: body mode by hand
```powershell
python learning\2026-11-14-vision-front-end\ex3_body_mode.py
```

Expected (checked):

```
   scale = robot reach 0.33 / 0.717 = 0.460
2. frames_to_demo says: mode body, scale 0.460  (same: the path fits)
3. Pick at  [253  83   8] mm ...
4. Camera turned 40 deg: path changes by at most 0.0000 mm
5. Operator reaches 60 cm forward: scale 0.313 (was 0.460)
```

The made-up operator has odd arm lengths (39 cm upper arm, 19 cm forearm) because its elbow is drawn crudely. Yours will be closer to 30/27 cm. Question: in part 5, why did the scale drop?

## Self-check (answer before you look)
1. Why does the code divide the thumb-index gap by the palm length?
2. In body mode, the twin says the block is at x = 192 mm. Can you put a real block at 192 mm in front of the real SO-101 and expect a grasp?
3. Your take has the hand lost for 1.2 s in the middle. What does the code do, and what will you see?

## For this afternoon's take
Film from the side or at an angle with **both shoulders and the whole working arm in view**, in good light, camera fixed. **Pinch** the part with thumb and index like a two-finger gripper (not the whole hand), open the hand wide between grips, keep the part fully visible at the start and end with your hand away from it, and keep it 5-15 s.

---

## Answers
1. To make it scale-free. Hand size and distance from the camera change the raw gap; the ratio stays about 1 open and under 0.5 pinched for anyone.
2. No. Body mode scales your arm to the robot's reach; it is the same motion at about half size, not true millimetres. For real positions use table-marker mode (ArUco + PnP), and still expect a few centimetres of error with one camera.
3. It fills the gap with a straight line between the last and next seen positions and prints "hand lost for 1.2 s at t=... : the path is a straight line across it". The grip keeps its last seen value.

Exercise questions:
- Exercise 2, noise 0.12 (checked): one threshold flips 18 times, two thresholds 4 times. The 0.2 wide dead band is now crossed by chance. That is why `frames_to_demo` also runs a 5-sample median before the thresholds (line 380). Lower noise (better light, closer camera) beats cleverer filters.
- Exercise 3, part 5: the far reach pushed the path past 0.27 m from the robot base, so the code shrank the scale until it fitted (lines 357-362).
