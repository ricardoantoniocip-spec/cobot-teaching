# The object, found by name, is the main signal for pick and place (2026-11-14)

**What:** The video app now finds the moved object with an open-vocabulary detector (LLMDet
tiny, Apache-2.0, through Hugging Face transformers), follows it, and reads grasp, release and
the pick and place spots from where the object rests and moves. The operator names the object
(`--object "juice carton"`); without a name the detector looks for common things and takes the
one that is moved. The hand track stays for the robot's path and to check that the hand is on
the object at each grasp. The object's name and every move go into steps.json, the task file,
the URScript and the SO-101 twin. Built in product/teach/retarget/objects.py. Research and
model comparison: research/2026-11-14-vision-models.md.

**Why:** Ricardo: the vision was not at the standard we want for objects and actions. On take 1
the hand never reads as closed (wide carton), and the picture comparison found only the first
move. The detector finds the carton in every picture looked at, also in the hand, and finds
both moves. An object that rests, leaves and rests elsewhere is a physical fact, which is a
surer basis for a robot program than hand shape. Only Apache/MIT models: GPL and AGPL
detectors (YOLO-World, Ultralytics) and non-commercial hand models (HaMeR, WiLoR, CoTracker)
were left out on purpose.

**What would change it:** parts the detector cannot find by name (small, shiny or look-alike
parts on a busy bench: then fine-tune a detector on a few photos per part, or use a marker),
several different objects in one demo (then one track per name), speed on site (about 5 s per
picture on a laptop CPU: then OWLv2 or a GPU), or a need for the object's outline and turn
(then add SAM 2 on top of the box).
