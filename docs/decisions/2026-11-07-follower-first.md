# 2026-11-07 Build the SO-101 follower first, leader later

What: print and build only the SO-101 follower arm now (~€124 in parts, hardware/so101/build-parts-list.md). No leader arm yet.

Why: the robot is trained from operator video and a digital twin, not from leader-arm teleop. The follower is still needed to prove the sim-to-real step works and to demo to integrators. The leader only adds a teleop baseline and sleeve calibration, which can wait.

What would change it: video/twin training fails on the real arm and teleop demos are needed to compare or co-train; or sleeve calibration (gate A) needs a leader arm as ground truth.
