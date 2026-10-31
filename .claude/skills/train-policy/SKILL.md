---
name: train-policy
description: Record demonstrations, train a robot policy (ACT first, then SmolVLA or another VLA) with LeRobot, and evaluate it on the SO-101 with a fixed trial protocol. Use when it's time to record, train, fine-tune or evaluate a model.
---

# Train and evaluate a policy

1. Check the current LeRobot docs and installed version (`pip show lerobot`) and use the commands for that version. Write them into the run file so the run can be repeated.
2. Define the task in `product/teach/runs/YYYY-MM-DD-<task>/README.md`: what success looks like, the object positions, the number of demos, the capture method (leader arm, sleeve, or mix).
3. Record: same camera positions every time, vary the object position on purpose, discard failed demos. Note the time spent (hours-to-deploy starts here). Push the dataset to Hugging Face as private.
4. Visualise a few episodes before training (rerun-lerobot skill or the LeRobot dataset viewer) to catch bad cameras or dropped frames.
5. Train: ACT as the baseline. Use free Colab or a rented GPU if the laptop has no NVIDIA GPU. Log loss curves and the config. Fine-tune a VLA (vla-model-scan picks which) only after the ACT baseline works.
6. Evaluate: 20 trials, same protocol for every model, record success, failure type and cycle time per trial in a table. Pass mark for the first task: 12 of 20.
7. Compare with earlier runs (Gate D: leader-only vs sleeve vs mix).
8. Log total hours-to-deploy: recording + training + evaluation + fixes.
9. Reply: success rate, hours-to-deploy, main failure mode, what to try next.

Real arm: speed limit and an explicit confirmation step before every run. Never publish datasets or models without Ricardo's OK (trade secret).
