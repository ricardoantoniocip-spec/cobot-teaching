# Teaching pipeline (product/teach)

## retarget: from operator video to SO-101 motion

This is the bridge from "operator does the task" to "arm repeats it".

```
video (or sleeve)  ->  hand pose + grip  ->  tool pose  ->  IK  ->  joint trajectory  ->  replay on the arm
     video                demo.csv                convert              traj.csv              replay
```

| Command | What it does | Needs the arm? |
|---|---|---|
| `example OUT.csv` | Makes a synthetic pick-and-place hand demo | No |
| `video VIDEO markers.json OUT.csv` | Tracks an ArUco marker on the back of the hand against a table marker | No |
| `convert DEMO.csv OUT.csv` | Hand path to SO-101 joint angles (IK from the official URDF), smooths, caps speed, reports reachability | No |
| `check --port COM5` | Reads the arm's joints and prints where the code thinks the gripper tip is | Yes |
| `replay TRAJ.csv --port COM5` | Plays the trajectory, slowly, after you type `yes` | Yes |

### Run it on the laptop (Windows, PowerShell)

```powershell
cd C:\dev\cobot-teaching\product\teach
py -3.12 -m venv C:\dev\venvs\teach
C:\dev\venvs\teach\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q tests

# 1. Without the arm
python -m retarget example demo.csv
python -m retarget convert demo.csv traj.csv

# 2. From your own video: print markers 0 (table, 6 cm) and 1 (hand, 4 cm) from DICT_4X4_50,
#    tape marker 1 to the back of your hand, film with a fixed webcam.
#    grips.csv has columns t_close,t_open (seconds) for each grasp, until EMG/FSR is ready.
python -m retarget video my_task.mp4 retarget\markers.example.json demo.csv --grip-events grips.csv
python -m retarget convert demo.csv traj.csv

# 3. On the arm (install lerobot==0.6.1 in the same venv, arm calibrated with LeRobot first)
python -m retarget check --port COM5
python -m retarget replay traj.csv --port COM5 --max-speed-deg 20
```

### Before the first replay
1. Run `check` in 3-5 poses and measure the real gripper tip with a ruler. If a joint is off, the
   URDF zero and the LeRobot calibration zero differ: set its sign or offset in `JointMapping` (robot.py).
2. Measure where the table marker sits in the robot base frame and put it in `markers.json`.
3. Calibrate the webcam (OpenCV chessboard) and pass `--camera camera.json`. Without it, positions
   can be off by centimetres.
4. Keep `--max-speed-deg` at 20-30. LeRobot also clips every step to 5 degrees.
5. Ctrl+C stops the replay. The arm goes limp when the script exits, so support it.

### Status (31 Oct 2026)
- Tested here: 10 unit tests (FK/IK against the URDF, retarget on a synthetic demo, speed cap,
  a synthetic two-marker video giving about 5 mm error with one camera).
- Not tested: the real arm, a real video. `robot.py` is written against LeRobot 0.6.1 but has not
  touched hardware.
- Not built yet: `hand_to_tool` calibration with the leader arm, EMG/FSR grip, the relabel step
  (record the robot's own camera and joints during replay as LeRobot episodes, then train ACT).

Robot model: `hardware/so101/so101_new_calib.urdf` (official, see its README).
