# SO-101 follower: print files

## Files
- `BambuLabA1mini_Follower_SO101.stl`: all follower parts in one file, already oriented (12 pieces, largest 142 mm).
- `Gauge_0.STL`: small fit-check gauge for an STS3215 servo (51 x 35 x 10 mm).

## Source and licence
Copied unchanged from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100),
commit `a758567c3978dfeefe282ede0500085a48fe8f78`:
- `STL/SO101/Follower/BambuLabA1mini_Follower_SO101.stl` (SHA-256 `3b207908...9705b6`, same as upstream)
- `STL/Gauges/Gauge_0.STL`

Licence: Apache 2.0, copy in `LICENSE-SO-ARM100`. Redistribution is allowed if the licence goes with it.

## How to print (Bambu A1, 256 mm bed)
1. **Fit check first.** Print `Gauge_0.STL` and push an STS3215 servo into it. It should slide in snugly, no force, no wobble. Too tight or too loose: fix the printer (flow / XY compensation) before the big print.
2. Open `BambuLabA1mini_Follower_SO101.stl` in Bambu Studio. Printer: **Bambu Lab A1**, 0.4 mm nozzle.
3. The file is laid out for two A1 mini plates (about 370 x 370 mm in total), so it is too big for one plate as is. Right-click the object, **Split > To objects**, then **Arrange** (A). Bambu Studio spreads the parts over plates; expect 2 plates on the A1.
4. Do not rotate parts. They are already oriented to need few supports.
5. Settings (upstream advice): PLA or PLA+, 0.2 mm layer height, **15 % infill**.
6. Supports: on, but only where needed (threshold about 45 degrees). No supports inside horizontal screw holes; paint them off with the support blocker if needed.
7. Clean the plate, slice, check the preview, print.

Upstream print guide: https://github.com/TheRobotStudio/SO-ARM100#printing-the-parts
