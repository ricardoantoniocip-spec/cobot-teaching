# SO-101 follower: first power-up (for the day the servos arrive)

About 1.5 hours. Written 15 Nov 2026, before the hardware existed: the LeRobot commands are
checked against LeRobot 0.6.1's own code, our `bringup` script is tested against a stand-in robot
only. **Nothing here has run on a real arm yet.** Keep a hand on the power switch.

## Before you plug anything in

- Power supply: 7.4 V servos take the 5 V supply, 12 V servos take the 12 V supply
  (`build-parts-list.md`). Read the voltage on the servo label before you connect: a 12 V supply
  on 7.4 V servos destroys them. The USB cable is for data only.
- Arm clamped to the table. Nothing within 40 cm.
- Venv on: `C:\dev\venvs\teach\Scripts\Activate.ps1`, with `pip install "lerobot[feetech]==0.6.1"`.

## 1. Find the port (2 min)

```powershell
lerobot-find-port
```
Unplug the USB cable when it asks. It prints the port, for example `COM5`. Use it below.

## 2. Give each motor its number (20 min, before the motors go into the arm if you can)

```powershell
lerobot-setup-motors --robot.type=so101_follower --robot.port=COM5
```
It asks for one motor at a time, starting with the gripper (6) and ending with the shoulder pan (1).
Connect **only that motor** to the board, press Enter, write its number on it with a marker.

## 3. Build the arm, then calibrate (15 min)

```powershell
lerobot-calibrate --robot.type=so101_follower --robot.port=COM5 --robot.id=follower
```
Put every joint in the middle of its range, press Enter, then move each joint through its full
range by hand. The result is saved under the id `follower`. Our scripts use the same id.

## 4. Check every joint against our model (15 min)

```powershell
cd C:\dev\cobot-teaching\product\teach
python -m retarget bringup --port COM5
```
- Step 1 reads the joints. Nothing moves.
- Step 2 moves one joint at a time by 10 degrees at 15 degrees a second, and tells you which way
  the gripper tip should go ("30 mm forward and 12 mm up"). Answer `y`, `n` (other way), `x`
  (did not move) or `q` (stop). An `n` flips that joint's sign; run it once more to confirm.
- Step 3 opens and closes the gripper.
- Step 4 is the first real move: the gripper goes to 18 cm in front of the base, 6 cm up. Measure
  where the tip really is with a ruler. Within 10 mm is good.

The answers are saved in `product/teach/so101_mapping.json`. Replay and live mode read that file.

## 5. Replay take 1 (10 min)

```powershell
cd ..\sim
python -m so101_sim from-video C:\path\to\take1.mp4 --arm right --object "juice carton"
cd ..\teach
python -m retarget replay ..\sim\video_data\take1\traj_steps.csv --port COM5
```
30 degrees a second. Film it: this is the first clip for interviews.

## 6. Live: your arm leads (10 min)

```powershell
cd ..\sim
python -m so101_sim live --port COM5
```
The arm starts paused. Calibrate (hand on the table, 2 s), check that the twin does what you
expect, then press SPACE.

## If something is wrong

| What you see | Likely cause |
|---|---|
| A motor is not found | wrong id, loose cable, or two motors with the same id on the bus |
| "asked +10 deg but the motor reports +0.0" | joint blocked, cable, or the supply is off |
| Arm shakes or hums | supply too weak or a joint is at its end stop |
| Tip is more than 10 mm off in step 4 | calibration middle pose was off: calibrate again |
| Arm drops when a script ends | normal: LeRobot switches the motors off on disconnect. Hold it. |

Write down anything odd (which joint, what number) and tell Claude.
