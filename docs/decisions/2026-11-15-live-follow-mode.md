# Live mode: the operator's arm leads, the SO-101 follows (2026-11-15)

## What
`python -m so101_sim live`: webcam, body tracking on every picture, the robot (twin now, real arm
with `--port`) follows the hand at once. One window shows the body, hands, object boxes and the
numbers behind the robot's motion.

- A 2 s calibration (hand resting on the table) replaces "measure over the whole video": body
  directions, arm length, table height.
- The gripper points down near the table and tilts forward as the hand goes up. With the gripper
  kept straight down, the 5-joint SO-101 only reaches about 8 cm above the table.
- Joint speed is capped (90 deg/s twin, 45 deg/s real arm). The real arm starts paused.
- Object detection runs beside the loop, on the graphics card when torch can use it. Body tracking stays on the CPU.

## Why
Ricardo asked for it (14 Nov) for two reasons: to test the idea directly once the arm is built
(USB to the laptop), and to see what the vision finds so he can judge and troubleshoot it.

## What would change it
- The live view shows the hand track is too noisy in depth: add the table marker or a second camera before more robot work.
- The real arm shakes or lags at these speeds: lower the cap, add a stronger filter.
- A customer wants teleoperation as a product: it is not one today; this is a test tool.
