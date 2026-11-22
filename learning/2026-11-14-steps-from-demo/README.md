# Lesson 4: From a hand path to named steps

Series "what we built", part 2 of 4. About 15 minutes reading, 30 minutes exercises. Laptop, no GPU.

## Results first
- Take 1 was cut into 5 steps: reach (0-3.07 s), grasp (3.07 s), move (3.07-8.53 s), release (8.53 s), retreat (8.53-9.47 s).
- The grasp and release times were typed by hand, because the grip ratio never closed (lesson 3).
- The automatic "object" method finds **4.67 s and 7.78 s** on the same video: the moments the carton lifts and lands. It also notices you moved the carton a second time at about 11.8 s.
- The step names ("grip the carton near its top with the whole hand") were written by Claude in the project chat from 5 frames, in the same JSON shape the API call uses. The API path (`taskprog label`) has not run on your laptop yet.
- Honest summary: **the cutting is rule-based, not a trained classifier.** One signal (grip closed or open) and fixed naming rules. The LLM only adds names.

## Where this sits

```
 hand path + grip (demo.csv)      <- lesson 3
      |
 grip from: pinch ratio | --grips 3.0-8.5 | --grips object
      |
 segment_steps: cut at every grip change
      |
 steps.json  (times, xyz, holding, label = null)
      |
 taskprog label: frames + prompt -> vision LLM
      |
 labels.json (names only, fixed JSON shape)
      |
 from-steps -> task.json            -> lesson 5
```

## 1. The cutter: one signal, fixed rules
`segment_steps` in `product/teach/retarget/pose.py:414-435`:
- Find every sample where the grip changes (line 421). Those are the cuts.
- Between two cuts, name the piece by one rule (line 427):
  - holding = **move**
  - not holding and it is the last piece after a grasp = **retreat**
  - otherwise not holding = **reach**
- At each cut, add an instant step: **grasp** if the grip closed, **release** if it opened (lines 431-434).
- Each step gets `t_start`, `t_end`, `from_xyz`, `to_xyz` (robot frame, metres), `holding`, and `label: null`.

Factory picture: a time study of one work cycle, where a single clamp sensor marks the element boundaries. Clamp closes = "grasp", clamp opens = "release", everything in between is named by its position.

Then `pick_and_place_points` (lines 438-446) takes the **first** grasp and the release after it. That pair is what the twin and the program use.

### steps.json
Written by `video_to_demo` (lines 466-467): `video`, `mode` (body or table-marker), `arm`, `video_t0`, `steps`.

**Two clocks.** Step times count from the first tracked frame ("demo time"). `--grips` and `--start` use video seconds, as printed on `tracked.mp4` (lines 76-78, 382). `video_t0` says where demo time 0 sits in the video (lines 94, 408), so the labeller grabs the right frames (`taskprog/label.py:76-84`). Take 1's steps.json is older than that field, so 0 is assumed. That is right for take 1 (tracking started at frame 0) but check it on new takes.

## 2. Three ways to get the grip signal
1. **Pinch ratio** (automatic, lesson 3). Works for a thumb-index pinch only.
2. **`--grips 3.0-8.5`** (manual). `parse_grips` (`product/sim/so101_sim/from_video.py:100-113`) reads "close-open" pairs in video seconds. They replace the automatic grip completely (`pose.py:381-382`).
3. **`--grips object`** (automatic, from the object). `product/teach/retarget/object_motion.py`. Explained next.

### Object motion: watch the part, not the hand
Idea (docstring, lines 1-18): the object stands at A at the start and at B at the end. When it leaves A, it was grasped. When it appears at B and stays, it was released. No AI model, classic image processing:

1. **Shrink** frames to 320 px wide and blur a little (lines 31, 54-74). Faster, and less noise.
2. **What changed?** Start picture = median of the first 0.5 s. Late picture = median of the second half (the median hides a hand passing through). Pixels that differ by more than 28 grey levels form blobs (lines 113-121).
3. **Is it an object?** Skip blobs that are too big (that is you) or ragged (line 133). Then measure, per frame, how much A still looks like the start picture (`_like`, lines 93-101). That is **normalised correlation**: 1 = same picture, about 0 = something else, and a lighting change does not matter. Below 0.35 = the object has left (line 136).
4. **Where did it go?** **Template matching** (lines 142-158): slide the start picture of the object over every later frame, at 6 sizes, and score the likeness at every position. Best score = place B. Plain patches (sofa fabric) match anywhere, so a patch with more detail scores higher (line 141, 157). Too weak a match = stop with "no object motion" (line 159).
5. **Times** (lines 171-187): walk back from the best view of B until it stops looking like the object = set-down. Grasp = lift minus 0.2 s, release = set-down plus 0.2 s. At both moments the object is held and still, which is what the robot must copy.

Take 1, exercise 2 shows the curves: A drops to about 0.55 at 2.4 s (your hand covers the carton), gone at 4.9 s (lift). B jumps to 0.54 at 7.6 s (set down). The hand-typed times (3.07 and 8.53 s) include the time your hand was on the carton. Neither is wrong: one marks "hand on part", the other "part moves". For a robot program the second is the cleaner definition.

Limits (`product/sim/README.md`): fixed camera, one object, one move, hands away from it in the first 0.5 s.

## 3. Vision LLM labelling: names only
`product/teach/taskprog/label.py`:
- **Which frames:** start, middle and end of each step; one frame for instant steps (lines 76-84). Resized to 640 px wide JPEGs (lines 87-105).
- **What it is asked** (prompt, lines 61-73): label each step, name the object and the places in snake_case, say if a step does not look like its type, put doubts in `review`. "Do not guess sizes or positions."
- **Fixed answer shape:** `LABELS_SCHEMA` (lines 20-59) is sent as structured output (line 136): the API forces exactly these keys, no extras (`additionalProperties: false`). There is no key for a position or a size.
- **No API key?** `python -m taskprog frames VIDEO steps.json OUT_DIR` writes the frames and the prompt so a person (or Claude in a chat) can fill in labels.json by hand (lines 112-118). That is how take 1 was labelled.

How labels become a task (`taskprog/from_steps.py`, lesson 5 goes further):
- Positions and times come **only** from steps.json. Names come from labels.json. Sizes from you (docstring, lines 1-9).
- A bad name falls back to a plain default (`pick_1`, `part`, `demo_task`) plus a review note (lines 48-51, 63-69, 101-105). Names must match `^[a-z][a-z0-9_]{0,39}$` and not be a URScript keyword (`taskprog/task.py:28-30`).
- Text is cleaned to one line of plain ASCII (`task.py:226-232`), because labels end up as comments in robot code. A label cannot sneak a line of code into the program.
- `matches_step_type: false` becomes a review note for the engineer (lines 43-44).

## 4. Honest: rules now, a learned classifier later
What we have is not machine learning in the cutting. It is a threshold on one signal plus naming by position.

Good:
- Transparent. You can explain every cut.
- Needs no training data.
- Same input, same output.

Weak (exercise 1 part 4 shows it):
- One bad frame in the grip signal turns 5 steps into 9.
- Only reach / grasp / move / release / retreat. It cannot see "insert", "push", "screw", "inspect", "hand over".
- No confidence per step. A step is either there or not.
- The video app assumes one pick and one place per video.

What a learned step classifier would add later: a **temporal action segmentation** model. Input: per-frame features (landmarks, hand-to-object distance, image features). Output: a label per frame with a probability, for many step types. A well-known example is MS-TCN ([paper](https://arxiv.org/abs/1903.01945)). It needs dozens to hundreds of labelled demos, and you would judge it like a measurement system: agreement with a human's labels per frame, and how many milliseconds the boundaries are off. The sleeve's EMG/FSR grip sensing (Gate B) attacks the same problem from the hardware side: measure the grip instead of guessing it from pixels.

## Exercises (on your laptop, 30 minutes)

```powershell
cd C:\dev\cobot-teaching
C:\dev\venvs\teach\Scripts\Activate.ps1
```

### Exercise 1: the cutter on take 1's numbers
```powershell
python learning\2026-11-14-steps-from-demo\ex1_segment.py
```

Expected (checked):

```
2. segment_steps on the rebuilt path
   reach > grasp > move > release > retreat
   same step names and times as steps.json: True
3. block      at [0.1918, 0.0501] m
3. place zone at [0.2379, -0.0779] m
   result.json says part_xy [0.1918, 0.0501], place_xy [0.2379, -0.0779]
4. Same demo, grip lost for ONE frame at 6.0 s
   reach > grasp > move > release > reach > grasp > move > release > retreat
   9 steps instead of 5. ...
   place point now [0.201, -0.018, 0.059] m instead of [0.223, -0.073, 0.04] m
```

Part 3 shows where the twin's block comes from: your grasp point pushed 16 mm outward (half the 20 mm block plus half a finger). Question: in part 4, which lesson 3 filter would have removed that one bad frame?

### Exercise 2: object motion on take 1
Your original take 1 video, or `tracked.mp4` from the project files (video-app/take1). About 20 s.

```powershell
python learning\2026-11-14-steps-from-demo\ex2_object_motion.py C:\path\to\take1.mp4
```

Expected on `tracked.mp4` (checked; the original should be within a frame or two):

```
object at pick (660, 344) px, at place (954, 620) px, about 72 x 152 px, look-alike score 0.80
 t (s)  A still shows the object   B already shows the object
   1.9  +0.98 ####################  -0.39
   2.4  +0.57 ###########           -0.38
   4.7  +0.44 #########             -0.36   <- grasp
   5.7  -0.06                       -0.32
   7.6  +0.15 ###                   +0.54 ###########   <- release
   9.0  -0.15                       +1.00 ####################
grasp at 4.67 s, release at 7.78 s  ->  --grips 4.67-7.78
note: the object is moved again at about 11.8 s; only the first move is used
```

Open `ex2_object_motion.jpg`: are the pick and place boxes on the carton? Then run with `--end 11` and see if the note goes away.

### Exercise 3: what the LLM sees and what it may change
No key, no video.

```powershell
python learning\2026-11-14-steps-from-demo\ex3_labels.py
```

Expected (checked):

```
1. Frames sent to the LLM (video_t0 = 0.0 s, missing in this older file so 0 is used)
   step 0 (reach  ) at  0.00 s ... 11 frames in total
2. ...keys for positions or sizes: none
3. Task 'move_carton_to_seat', object 'juice_carton'
   location armrest    xyz_mm [176.3, 46.1, 0.0]  (top of the left sofa armrest)
4. Careless labels -> task 'demo_task', locations ['pick_1', 'sofa_seat']
   pick location xyz_mm [176.3, 46.1, 0.0] (unchanged)
   review: vision LLM doubts demo step 2 (move): the hand looks empty in the middle frame
5. No labels -> task 'demo_task', locations ['pick_1', 'place_1'], object 'part'
```

Note part 4: the careless LLM tried `xyz_mm: [999, 999, 999]` and nothing changed.

## Self-check
1. The demo has grip closed 2-4 s, open 4-4.1 s, closed 4.1-7 s. How many steps does `segment_steps` make, and what are they?
2. Why does object motion put the grasp 0.2 s **before** the lift, not at the moment your hand touches the carton?
3. Name one thing a learned step classifier could do that the current cutter cannot.

## For this afternoon's take
Start and end with your hand **away** from the part for at least a second, part fully visible. Move it **once**, then stop the recording (take 1's second move at 11.8 s confused nothing, but it is noise). Note the times you grasp and release by eye from `tracked.mp4`, so you can compare them with `--grips object`.

---

## Answers
1. 9: reach, grasp, move, release, reach, grasp, move, release, retreat. The 0.1 s opening is read as a real release and grasp. (In the real pipeline the 5-sample median and the two thresholds usually remove such a blip first.)
2. Because the robot copy wants a moment when the object is held and still. The hand touching the carton is hard to see in pixels; the lift is clear. 0.2 s back from the lift, the hand is already on it and it has not moved yet.
3. Any of: more step types (insert, push, screw), a confidence per step, robustness to one bad frame, several picks per video.

Exercise questions:
- Exercise 1: the 5-sample running median on the grip ratio (`pose.py:380`) removes a one-frame dip before the thresholds. With `--grips` typed by hand there is no such filter, and none is needed.
- Exercise 2, `--end 11` (checked): same times, 4.67 s and 7.78 s, and the "moved again" note is gone, because the second move is cut off.
