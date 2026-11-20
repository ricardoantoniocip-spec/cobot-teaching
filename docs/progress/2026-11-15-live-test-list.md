# Morning test list: live mode on your laptop (about 10 minutes)

Everything is on the branch of pull request #18. PowerShell:

```powershell
cd C:\dev\cobot-teaching
git fetch origin
git checkout claude/project-thread-3i1q0t
git pull
C:\dev\venvs\teach\Scripts\Activate.ps1
pip install -r product\teach\requirements-pose.txt -r product\teach\requirements-vision.txt
cd product\sim
```

## 1. What will run on the graphics card (1 min)

```powershell
python -m so101_sim devices
```

Wanted: `object detector: GRAPHICS CARD (... GTX 1650 ...)`. If it says CPU:

```powershell
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124
python -m so101_sim devices
```

## 2. Your arm leads, the twin follows (4 min)

```powershell
python -m so101_sim live
```

1. Sit at a table, webcam in front of you, both shoulders and your right arm in view.
2. Rest your right hand on the table and hold still until the yellow bar is full.
3. Move your hand left, right, up, forward. Open and close it.

Please note for me:
- Does the robot go the same way as your hand for left and right, up and down, forward and back?
- Does the gripper close when you close your hand? What "hand openness" number do you see for a flat hand and for a fist?
- How many "pictures/s" does the panel show?
- Where does it go wrong: HOLD (hand lost), AT EDGE OF REACH, or jumpy motion?

## 3. With objects (3 min)

Put a cup or a small box on the table.

```powershell
python -m so101_sim live --objects "cup" --save live_test.mp4 --log live_test.csv
```

Please note: the line "objects on cuda: one look takes ... s", and whether the box sits on the cup.
Press `q` to stop. Then send me `live_test.mp4` and `live_test.csv` from `product\sim`, or just
tell me the numbers.

## 4. Only if you have time: sim data again (leave it running)

The old sim data shows a release that tips the block over (found last night, now fixed).

```powershell
python -m so101_sim demos 50 --out sim_data_v2
```

Then train ACT again on `sim_data_v2` as in the README, steps 2 to 4.
