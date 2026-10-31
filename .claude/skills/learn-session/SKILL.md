---
name: learn-session
description: Run a short tutoring session that teaches Ricardo robot learning (LeRobot, datasets, ACT, VLAs, training, evaluation, IMU fusion, EMG) step by step, with an exercise and a progress log. Use when he says "teach me", "explain", "next lesson" or asks how something in model training works.
---

# Learning session

1. Read `learning/progress.md` (create it if missing) to see what he has done. The 4-week plan: week 1 LeRobot in simulation (ACT, Colab, LeRobotDataset); week 2 SO-101, record 50 demos, train ACT, 20 trials; week 3 IMU fusion (complementary filter, Madgwick, BNO085 on ESP32-S3, ArUco yaw correction); week 4 EMG (envelope, onset, MyoWare + FSR).
2. Pick the next topic, or the one he asked about. Say in one line why it matters for the cobot product.
3. Teach it in this order, short:
   - The idea in plain words and one picture-in-words or analogy (he is a production engineer: use factory examples).
   - The minimum maths, only if needed.
   - A runnable example: a Colab notebook or a Python file in `learning/YYYY-MM-DD-<topic>/`, working on free Colab unless he has a GPU.
   - One exercise he does himself (15-45 minutes), with how to tell it worked.
   - Three quiz questions; answers at the bottom of the file.
4. Point to one good official resource (LeRobot docs, Hugging Face course, paper), with the link.
5. Log the session in `learning/progress.md`: date, topic, exercise, done or not, questions he still has.
6. Reply on his phone in under 12 lines: the key idea, the exercise, the link to the file.

Check the current LeRobot docs before giving commands; its command names change between versions. Don't skip ahead of what he understands: ask one check question before moving to the next topic.
