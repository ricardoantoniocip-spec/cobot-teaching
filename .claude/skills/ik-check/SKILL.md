---
name: ik-check
description: Set up and verify forward and inverse kinematics, frames and workspace for the SO-101 (5-DoF), PAROL6 and cobots, and the retargeting from the sleeve's hand pose to robot joints. Use for any kinematics, IK, calibration, frame or reachability question.
---

# IK check

Use the robotics-kinematics skill for theory (DH, Jacobian, singularities). This skill is the project routine.

1. Get the robot model: the official URDF (SO-101 from the LeRobot / SO-ARM100 repo, PAROL6 from its repo). Link the commit you used. Never type link lengths from memory.
2. Forward kinematics: load the URDF in Python (e.g. `ikpy`, `pinocchio` or `roboticstoolbox-python`), compute the end-effector pose for 5 known joint sets, and compare with the real arm (measure with a ruler or the camera + ArUco). Note the error in mm.
3. Inverse kinematics: damped least squares with weights. The SO-101 has 5 DoF, so it can't reach every orientation: weight position > approach direction > roll, and project out yaw outside its vertical plane.
4. Workspace: sample the task area on a grid; report the fraction reachable within 5 mm (target 95 % or more) and where singularities or joint limits bite.
5. Frames for the sleeve: write down the chain world → camera → hand pod → virtual gripper TCP → robot base, and how each transform is calibrated (leader arm as ground truth).
6. Retargeting test: replay a recorded hand path through IK, check joint limits, speed limits and smoothness, then replay on the follower at low speed with a confirmation step.
7. Save code to `product/teach/kinematics/` with a test per function (see the robotics-testing skill) and results to `product/teach/kinematics/README.md`.
8. Reply with the numbers: FK error, reachable fraction, worst spots.

Anything that moves the real arm: speed limit and explicit confirmation first.
