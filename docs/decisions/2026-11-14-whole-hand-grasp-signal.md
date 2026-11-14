# Grasp signal: whole hand open or closed

**Date:** 2026-11-14

**What:** Ricardo decided the video intake must let the operator grip with the whole hand, not only thumb and index finger. Open hand = not holding. Hand closes = the part is grabbed. Hand opens = released. This replaces the thumb-and-index pinch rule in the tracker (`retarget/pose.py`) and in the phone scan (`product/capture`).

How it is measured (proposed by Claude): openness = distance from wrist to the four fingertips, divided by the length of the straight fingers. Measured on photos: flat hand 0.87-0.93, fist 0.45. Closed is under 0.78 for at least 0.4 s, and a grasp only counts when the hand is clearly closed (typically under 0.70). The robot follows a point between the thumb tip and the index and middle fingertips.

**Why:** The pinch rule missed whole-hand grasps and the phone scan flagged good videos as a problem. Operators should not have to change how they grip.

**Known limit:** A wide part keeps the fingers almost straight. On take 1 (juice carton) the hand reads 0.63-0.87 while holding, which overlaps with an open hand, and a first version found two false grasps in mid-air. So a half-closed hand is not counted, and the app then finds grasp and release from the object's motion (`--grips object`, now automatic). Parts the SO-101 gripper can hold are small enough for the fingers to curl.

**What would change it:** Real videos of small parts where the hand signal is wrong more than about 1 in 10 takes. Then make the object's motion the main signal, or let the vision LLM mark grasp and release.
