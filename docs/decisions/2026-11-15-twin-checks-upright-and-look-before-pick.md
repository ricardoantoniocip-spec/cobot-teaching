# The twin checks that the part stands; the robot looks before a re-pick (2026-11-15)

## What
1. A twin run only counts as a success when the part ends in the place zone **and upright**.
2. After every release the SO-101's finger moves 8 mm off the part before it goes up
   (task programs and the scripted demo).
3. `taskprog twin --look`: before every pick after the first, the pick position is where the part
   really stands. In the twin that comes from the simulator.
4. With a table marker, the video app writes the object's pick and place position in true mm
   into `steps.json`, and the task file uses them.

## Why
The second pick of take 1 failed in the twin. The cause was not only a 12 mm offset: the block
fell over at every release, because it leaned on the finger and was dragged up. The old score
did not look at that, so 20 of 20 scripted demos passed while leaving the block on its side.

## Consequences
- Sim training data made before 15 Nov shows a release that tips the part. Make it again and retrain ACT.
- Numbers quoted before this date for twin success are position-only.

## What would change it
- The real arm does not drag the part (different friction): keep the back-off anyway, it costs 0.4 s.
- Parts that are meant to be laid down: the upright check must become "ends in the pose the task asks for".
