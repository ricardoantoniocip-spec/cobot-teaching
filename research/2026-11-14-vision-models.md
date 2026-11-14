# Machine vision for demos: what is out there and what we use (2026-11-14)

Question from Ricardo: the vision in the video app is not good enough at finding objects and
actions. What do open models on Hugging Face and GitHub offer, and what should we build in?

All timings below are my own runs on take 1 (1280 x 720, 15 fps, 17 s) on a 4-core cloud CPU,
no GPU. Licences are from each model card or repository (links in the tables).

## Where the old system was weak

| Part | Old | Problem seen on take 1 |
|---|---|---|
| Object | none; only "what changed between first and last picture" | no name, no size, breaks when the object does not stand out, only the first move is found |
| Grasp and release | hand shape (pinch, then whole-hand openness) | a wide carton keeps the fingers straight: no grasp found, falls back to picture comparison |
| Actions | cut at hand open/close | no check that the hand is on the object; a second pick is missed |
| Body | MediaPipe pose + hand | arm and hand found in about 91% of frames; depth is the weak axis with one camera |

## Object detection: find a thing by name ("open vocabulary")

These models take a picture and words ("juice carton") and return boxes. No training.
Test: 5 frames of take 1, one with the carton standing free, four with a hand on or near it.

| Model | Licence | Size | Found carton (5 frames), score in the hand | CPU time per picture |
|---|---|---|---|---|
| [LLMDet tiny](https://hf.co/iSEE-Laboratory/llmdet_tiny) | Apache-2.0 | 173 M | 5 of 5, 0.63 to 0.81 | 5.4 s |
| [Grounding DINO tiny](https://hf.co/IDEA-Research/grounding-dino-tiny) | Apache-2.0 | 172 M | 5 of 5, 0.43 to 0.87, two frames with a second wrong box | 5.4 s |
| [OWLv2 base](https://hf.co/google/owlv2-base-patch16-ensemble) | Apache-2.0 | 155 M | 5 of 5, 0.37 to 0.61 | 3.2 s |
| [OmDet-Turbo tiny](https://hf.co/omlab/omdet-turbo-swin-tiny-hf) | Apache-2.0 | 115 M | not run: its extra dependency did not load here | - |
| [YOLO-World](https://github.com/AILab-CVC/YOLO-World), YOLOE via [Ultralytics](https://github.com/ultralytics/ultralytics) | GPL-3.0 / AGPL-3.0 | small | not run | fast |

Choice: **LLMDet tiny**. Highest and steadiest scores with the hand on the object, and Apache-2.0.
YOLO-World and Ultralytics are the fast options, but GPL/AGPL would force us to open our own
code or buy a licence, so they are out for a product. OWLv2 is the fallback if speed on a CPU
matters more than score.

Naming the object without being told: [Florence-2 base](https://hf.co/microsoft/Florence-2-base)
(MIT, 232 M) called the carton "a bag of red and yellow candies", "bottle" and "pink box". Not
good enough to name parts. So the name comes from the operator (`--object "juice carton"`), or,
without one, the detector looks for a list of plain words and we take the thing that moves.
On take 1 that finds the right object and calls it "bottle"; the engineer is asked to check.

## Following the object through the video

| Option | Licence | Verdict |
|---|---|---|
| Detector every 0.5 s + picture comparison for the exact frame | ours | built; 37 pictures for take 1, about 3.5 minutes on CPU |
| [SAM 2.1 tiny](https://hf.co/facebook/sam2.1-hiera-tiny) (follows a mask through a video) | Apache-2.0, 39 M | next step when we need the object's outline (size, orientation), best with a GPU |
| [CoTracker3](https://hf.co/facebook/cotracker3) (follows points) | CC-BY-NC-4.0 | non-commercial, not usable |

## Hand and body

| Model | Licence | What it gives | Verdict |
|---|---|---|---|
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) pose + hand (what we use) | Apache-2.0 | 33 body + 21 hand points, rough 3D | keep: light, runs on any laptop, gives 3D |
| [RTMW](https://arxiv.org/abs/2407.08634) via [rtmlib](https://github.com/Tau-J/rtmlib) | Apache-2.0 | 133 points, body + both hands, 2D | tested: right arm in 61 of 64 frames (95%), hand in 59 of 64 (92%), 0.28 s per frame on CPU. A little better coverage than MediaPipe, but 2D only, so not a drop-in |
| [HaMeR](https://github.com/geopavlakos/hamer), [WiLoR](https://github.com/rolpotamias/WiLoR) (full 3D hand mesh) | need the MANO hand model (non-commercial licence); WiLoR weights CC-BY-NC-ND | best 3D hands | not usable in a product without a licence deal |
| [100DOH hand-object detector](https://github.com/ddshan/hand_object_detector) | MIT | hand boxes + "is it touching something" | old build chain (custom compiled layers); we get the same answer from hand track + object box |

Conclusion for body tracking: the big gain is not a better skeleton, it is a second camera or a
table marker for true scale. Models with better 3D hands exist but their licences block
commercial use. RTMW is the candidate if MediaPipe loses the arm in a real cell.

## Actions

Research models that name actions from video (video transformers, VLMs) answer "what is
happening" in words, but not "at which frame, at which place" well enough to write a robot
program. For pick and place the reliable signal is physical: the object rests, leaves, rests
somewhere else, and the hand is on it. So actions are read from the object track plus the hand
track, with fixed rules. A vision LLM still labels the steps in words (taskprog `label`).

## What was built from this

`product/teach/retarget/objects.py`: detector every 0.5 s, link boxes, rests and moves,
frame-exact grasp and release, hand-on-object check. Wired into `so101_sim from-video --object`
(SO-101 twin) and `taskprog from-steps` (task file, URScript). Results in
`product/teach/taskprog/README.md` and the decision record
`docs/decisions/2026-11-14-object-detector-for-demos.md`.
