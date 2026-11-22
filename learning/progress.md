# Learning progress

Tess (ml-tutor) logs each lesson here. Plan: week 1 LeRobot in simulation, week 2 SO-101 demos and ACT, week 3 IMU fusion, week 4 EMG.

| Date | Topic | Exercise | Done | Open questions |
|---|---|---|---|---|
| 2026-10-31 | Lesson 1: a demo is process data (LeRobotDataset, episodes, observation vs action), learning/2026-10-31-what-is-a-demo/ | Episode length stats + state vs action plot on lerobot/pusht | Quiz yes (answers matched the key word for word); exercise results not shared | Ask for his episode length numbers |
| 2026-10-31 | Lesson 2: ACT and action chunks, learning/2026-10-31-act-action-chunks/ | Train ACT 2,000 steps on aloha_sim_transfer_cube_human, loss table, predicted vs recorded chunk | No | - |
| 2026-11-14 | Lesson 3: the vision front end (MediaPipe landmarks, image vs world coords, grip ratio and its two thresholds, why take 1's whole-hand grasp gave no grip, body mode vs table-marker mode, smoothing), learning/2026-11-14-vision-front-end/ | ex1 one frame through MediaPipe on take 1, ex2 grip ratio and hysteresis, ex3 body-mode scale by hand | No | Grip ratio at 4.0 s on his original take 1; forearm length in px |
| 2026-11-14 | Lesson 4: from hand path to named steps (segment_steps rules, steps.json and video_t0, --grips, object motion by template matching, vision-LLM labels and schema; honest: rule-based, not a trained classifier), learning/2026-11-14-steps-from-demo/ | ex1 cutter on take 1 + one bad frame, ex2 object motion curves on take 1, ex3 labels -> task, careless LLM | No | Compare his by-eye grasp/release times with --grips object on the new take |
| 2026-11-14 | Lesson 5: from steps to a robot program (task format 0.2, cell file, plan, URScript generator, check, URSim, SO-101 trajectory), learning/2026-11-14-steps-to-robot-program/ | ex1 regenerate take 1 and compare byte for byte + twin, ex2 plan listing, ex3 own kitting task and three breaks | No | - |
| 2026-11-14 | Lesson 6: SO-101 IK (URDF, damped least squares), retarget, copy vs steps, twin replay, episode file, LeRobot export, link to ACT, learning/2026-11-14-ik-twin-lerobot/ | ex1 FK/IK on take 1's grasp point, ex2 take 1 episode in the twin, ex3 inside the episode (+ LeRobotDataset if lerobot installed) | No | Ask him to film 5 more takes with the part in different places |

Notes 2026-11-14 (lessons 3-6, the "what we built" series):
- All exercise scripts were run in the cloud on Python 3.12 (Linux) with take 1's tracked.mp4. Lesson 3 exercise 1 on tracked.mp4 misses the working hand (the painted skeleton confuses the hand model); on his original video it should find it. Not yet run on Windows.
- Found while writing: retarget's IK (solve_ik) can stall with a joint at its limit, e.g. 32.8 mm miss at (300, 0, 50) mm fingers down while a 0.1 mm solution exists. Used as a lesson 6 question; reported to the project thread.
