# Lesson 2: What ACT does with your demos

Week 1, session 2. About 1.5 hours. Colab with a free GPU.

## Why it matters for the cobot
ACT is the model we will train on your SO-101 demos in week 2, and on the sleeve data at Gate D. You need to know what it predicts and what its training log means.

## The idea
In lesson 1 a policy gave **one** action per observation. That works badly with human demos: small errors pile up, and the operator's hesitations make the robot jitter.

ACT (Action Chunking with Transformers) predicts a **chunk**: the next 100 actions in one go, from one camera image plus the joint angles. The robot plays the chunk out, then asks again.

Factory picture: a single-step policy is like jogging an axis one click at a time while watching the part. ACT is like a CNC look-ahead: it plans a short motion segment, then executes it smoothly. At 50 fps, 100 actions = 2 seconds of motion.

The three parts, plainly:
1. **Vision backbone (ResNet18)**: turns the camera image into features, like a feature extractor on an inspection camera.
2. **Transformer**: mixes image features and joint angles and outputs the 100-step chunk.
3. **VAE "style" variable** (training only): humans do the same task in slightly different ways. This small extra input soaks up that style so the model does not average two ways into one bad middle path. At run time it is set to zero (the "average operator").

## Minimum maths
Training loss = **L1 loss** + 10 x **KL loss**.
- L1 = mean absolute error between predicted chunk and recorded chunk (in normalised units). This is the "how far off" number, like mean absolute deviation from setpoint.
- KL = keeps the style variable well behaved. Small and stable is good.
Training = adjust the weights a tiny bit each step so the loss goes down. Like tuning a controller by repeated trials, except the tuner is automatic (gradient descent).

## Run it (Colab, GPU)
Open `lesson2.ipynb` the same way as lesson 1. Before running: **Runtime > Change runtime type > T4 GPU**.

It trains ACT for 2,000 steps on `lerobot/aloha_sim_transfer_cube_human` (50 simulated demos: one arm picks a cube and hands it to the other). Expect 10-20 minutes. A real run uses 50,000-100,000 steps; we only want to see it learn.

## Exercise (30 minutes, while it trains and after)
1. While it trains, copy the `loss` value every 200 steps into the table cell. Plot loss vs step.
2. After training, run the chunk cell. It plots ACT's 100-step prediction against what the human actually did, for one joint.
3. Write one sentence: after 2,000 steps, is the prediction close, roughly right, or poor? Where does it go wrong first: start or end of the chunk?

How to tell it worked: loss falls fast in the first few hundred steps and then flattens. The predicted line follows the recorded one at the start of the chunk and drifts later.

## Quiz (answer in your own words first)
1. The SO-101 records at 30 fps. With chunk size 100, how many seconds does one chunk cover?
2. Why does ACT predict a chunk instead of one action?
3. Two operators pick a part: one goes left of an obstacle, one goes right. What would a model without the style variable likely do, and why is that bad?

## Read next
- LeRobot ACT docs: https://huggingface.co/docs/lerobot/act
- The ACT paper (sections 1 and 3 only): https://arxiv.org/abs/2304.13705

Reply to Tess with your loss table, your sentence and your quiz answers. Next lesson: evaluating a policy, the robot version of a capability study.

---

## Answers (look after you try)
1. 100 / 30 = about 3.3 seconds.
2. Fewer decisions means fewer chances for small errors to pile up, and the motion is smooth instead of jittery. It also copes better with pauses in human demos.
3. It would average the two and drive straight into the obstacle. Averaging two good strategies can give a bad one. The style variable lets the model keep them apart.
