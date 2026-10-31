# Lesson 1: A demonstration is process data

Week 1, session 1. About 1 hour. No hardware needed.

## Why it matters for the cobot
Everything we sell starts with a recorded demo. If the demo data is bad, no model can fix it. So first you learn what a demo looks like inside.

## The idea
Think of a station log on the line. Every tick you record:
- what the sensors see (the **observation**: camera images, joint angles), and
- what the operator or controller commanded (the **action**: target joint angles for the next step).

One run of the task, start to finish, is an **episode**. A **dataset** is many episodes. LeRobot stores this as a `LeRobotDataset`: a table of frames, plus videos for the cameras.

A **policy** is a function: observation in, action out. Training means: fit that function so it copies what the operator did in the same situation. This is called behaviour cloning. It is like fitting a controller to a skilled operator's moves, instead of tuning gains by hand.

Your IK background helps here: on the SO-101 the action is just a vector of target joint positions (6 numbers). No IK needed; the policy learns joint targets directly.

## Minimum maths
- One frame: observation `o_t`, action `a_t`, at time `t`.
- `fps` = frames per second. At 10 fps, 1 second of demo = 10 rows.
- Policy: `a_t = π(o_t)`. Training makes `π(o_t)` close to the recorded `a_t`.

## Run it (free Colab)
Open `lesson1.ipynb` from this folder in Colab (File > Open notebook > GitHub tab, tick "include private repos", search `cobot-teaching`). Or paste the cells from `lesson1.py`. CPU runtime is fine.

It loads `lerobot/pusht`, a small simulated task: push a T-shaped block into place. You will print the dataset, look at one frame and plot one episode.

## Exercise (30 minutes)
Do this yourself in the notebook, in the empty cell at the end:
1. Find the shortest and the longest episode (in frames and seconds).
2. Compute the mean and standard deviation of episode length. Treat it like cycle time on a station.
3. Plot `observation.state` and `action` for the longest episode on the same chart (x position only).

How to tell it worked: you get two lengths that differ a lot, and in the plot the action line runs slightly **ahead** of the state line. Write down in one sentence why.

## Quiz
1. In the SO-101 arm, what would the action vector be, and how many numbers does it have?
2. A dataset has 50 episodes recorded at 30 fps, each about 20 s. Roughly how many frames?
3. Why does the action lead the state in your plot?

## Read next
LeRobot dataset docs: https://huggingface.co/docs/lerobot/lerobot-dataset-v3

Reply to Tess with your three answers and your one sentence from the exercise. Next lesson: what ACT does with this data.

---

## Answers (look after you try)
1. Six target joint positions (5 joints + gripper), the leader arm's joint readings at that tick.
2. 50 x 20 s x 30 fps = 30,000 frames.
3. The action is the target for the next step; the arm (state) follows it with a small delay, like a setpoint and the measured value in a control loop.
