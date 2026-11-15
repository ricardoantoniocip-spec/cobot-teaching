# Whole-project review: shortfalls and the path to a SaaS product (2026-11-14)

Asked by Ricardo: evaluate the whole project, name the shortfalls, say how to make it complete.
The end goal is software as a service. Everything below is from the repo as it stands today.
No market numbers are used, so there are no source links.

## In one minute

- The pipeline exists end to end **on a screen**: video, body and object tracking, steps, task file,
  URScript, checked in URSim and in the SO-101 twin.
- It has been proven on **one video** (take 1) and **zero real robots**.
- The three biggest gaps are not vision models. They are: **true positions** (millimetres in the
  robot's frame), **a robot that looks before it picks**, and **a product a customer can touch**
  (today it is a command line on your laptop).
- The business side is further behind than the tech: 0 interviews, `customers/pipeline.csv` is empty.

## What works today (with the evidence)

| Part | State | Evidence |
|---|---|---|
| Body and hand tracking from a plain video | Works | Take 1: arm in 100%, hand in 82% of pictures (live run tonight) |
| Object found by name, followed, grasp and release read from it | Works on take 1 | 37 of 37 looks, both moves found (PR #18) |
| Demo cut into steps, task file, URScript | Works | Take 1, kitting tray and machine tending pass URSim |
| SO-101 twin, training data export, ACT training loop | Works in sim, with a correction | scripted demos re-scored 15 Nov after a tipping fault (see 3b); ACT must be retrained on new data |
| Live mode: your arm leads, the twin follows, all tracking shown | New tonight | `python -m so101_sim live`, tested on take 1 and made-up data |

## Shortfalls, most important first

### 1. Positions are not real
Body mode scales your arm onto the robot. Nothing is in true millimetres, so no program made today
would put a real gripper on a real part. One camera is also weak in depth (a centimetre or two).
**Fix:** a printed marker on the table gives the camera's place; the object's box then gives the pick
and place spot on the table in mm. Built on 15 Nov (`from-video --object ... --markers ...`), tested on
geometry only (typical 7 mm, worst 31 mm). Needs one real take with a printed marker. Later a second
camera or a depth camera for height.

### 2. The robot is blind at run time
The generated program goes to fixed taught positions. That is exactly what normal programming does,
and it is the cheap case. Our offer is "hard tasks with varying parts". For that the robot must
**look before each pick**. The twin shows it: with a look before the second pick, take 1's two
moves succeed; without it they fail, because the part stands 19 mm from where the program expects it.
**Fix:** built in the twin on 15 Nov (`taskprog twin --look`, 7 of 8 random two-move tasks against
0 of 8). Still to do: the same on the SO-101 with a real camera, then as a camera call in URScript.

### 3. One video is not a test
Every number we have is from take 1, filmed on a sofa with a juice carton. We do not know how often
the pipeline works.
**Fix:** a fixed test set of about 20 takes on a table (small part, shiny part, two parts, bad
light, left arm), each with the true grasp time and place written down, and one command that scores
all of them. Every change is then judged by that score, not by eye.

### 3b. Our own score was wrong once already
On 15 Nov the twin turned out to count a run as a success while the block lay on its side: the
gripper tipped it over at every release, in all 20 "successful" scripted demos and in the ACT
training data made from them. It is fixed, and the twin now checks that the part stands. The
lesson is the point of item 3: a score needs to check what a person would check.

### 4. Nothing has touched hardware
`retarget/robot.py` has never driven a servo. The twin's zero pose is assumed to match LeRobot's
calibration. Gripper offsets were tuned in sim only.
**Fix (from end of November, when the servos arrive):** power-up kit, joint-by-joint check against
the twin, replay take 1's steps, then live follow. A UR check on a real arm needs a partner
(integrator or a lab); URSim only proves the program runs, not that it is safe or reaches.

### 5. The object model is thin
One object per video. A box in the picture, no outline, no size in mm, no rotation, no grasp
direction. The gripper always points down. Parts that must be turned, stacked or inserted are out.
**Fix, in order:** outline with SAM 2.1 (size and rotation), several objects per video, side grasps.

### 6. There is no product, only tools
A SaaS needs things that do not exist yet:

| Needed | Today |
|---|---|
| A place to upload or record a demo | command line; a phone capture page exists but is not linked to processing |
| Processing on a server with a graphics card, as a queue of jobs | runs by hand, about 3.5 minutes on a CPU for 17 s |
| A review screen: timeline, steps, object boxes, edit and approve | `steps.json` and pictures in a folder |
| Cell setup: robot, gripper, camera calibration, safety limits | a hand-written cell file |
| Getting the program onto the robot | copy a `.script` file by hand |
| Accounts, projects, versions, an audit trail of who approved what | none |
| Automatic tests on every change (CI) | none: pull requests run no checks |
| Privacy: videos show people and factory floors | not addressed (faces, where video is stored, on-site option) |

### 7. Robot programs: one brand, simulated only
URScript only. No FANUC or Doosan. No real gripper drivers (Robotiq, OnRobot). No collision check
against the cell. Safety stays the integrator's job, and we should say so in writing.

### 8. The language model step is untested
The step labels were written in chat. `taskprog label` has only run against a stand-in. Nobody has
measured how often a vision model names the steps right.

### 9. Speed
The object detector takes about 5 s per picture on a CPU. Not yet timed on your GTX 1650. Live
object boxes need a graphics card to be useful.

### 10. Scope and time
Sleeve, GR00T, ACT training, website, live mode, URScript: a lot for 10 to 15 hours a week. The
sleeve and GR00T stay parked. ACT is only needed where fixed programs fail.

### 11. Customers and company
0 interviews against a gate of 15 by 31 January. Company gate G0 (employment contract check, 30 Nov)
has no decision record. Without interviews we do not know which of the gaps above a customer would
pay to close first.

## What "complete" looks like (SaaS, first version)

```
1 Record or upload a demo      phone or webcam, browser
2 Process                      server with a graphics card: body, objects, steps, labels
3 Review                       web page: video with boxes, step list, pick and place on a table map. Edit, approve.
4 Set up the cell              pick robot and gripper, show the camera a table marker, set limits
5 Check                        offline checks + simulator; a report the engineer signs
6 Deliver                      download the program, or push it to the robot through a small local agent
7 Run                          the robot looks before each pick (camera on the cell)
```

Steps 2 and 5 exist as command-line tools. Steps 1, 3, 4, 6 and 7 are the missing product.

## Order of work (part-time, about six weeks)

| # | What | Why first | Who |
|---|---|---|---|
| 1 | Table marker: true mm for pick and place | everything real depends on it | built 15 Nov; you film one take with a printed marker |
| 2 | Look before each pick | closes the gap between offer and build | built in the twin 15 Nov; real camera next |
| 3 | Test set of 20 takes + one scoring command | stops guessing | you film (1 hour), Claude builds the scorer |
| 4 | CI: tests run on every pull request | cheap, protects all of the above | Claude |
| 5 | Real SO-101: power-up, twin check, replay, live follow | first hardware proof, a 60 s video for interviews | you, with the kit |
| 6 | Review page (local web page first) | the first piece a customer can touch | Claude |
| 7 | 5 interviews before Christmas, show the video | decides what comes after | you |

Parked until a customer asks: sleeve, GR00T, PAROL6, more model research, FANUC and Doosan output.

## What would change this order
- An integrator says "I only care about UR, give me the program": move a real-UR test up.
- Interviews say programming time is not the pain: stop building and rethink (gate G1).
- The 20-take score is under half: vision work goes back to the top.
