---
name: vla-model-scan
description: Compare vision-language-action (VLA) and imitation-learning policies (SmolVLA, pi0/pi0.5, OpenVLA-OFT, GR00T, ACT, Diffusion Policy) for the SO-101 and later cobots, and recommend what to try next. Use when choosing or re-checking a model, or when a new VLA is released.
---

# VLA model scan

1. Read `docs/context.md` (hardware, gates) and the newest `research/vla/*.md` if it exists.
2. For each candidate, open the official source (paper, GitHub repo, Hugging Face model card, LeRobot docs) and record:
   - What it is: VLA (vision + language + action) or plain imitation policy (ACT, Diffusion Policy).
   - Parameters and the GPU it needs to fine-tune and to run (VRAM), and whether a free Colab or a rented GPU is enough.
   - Supported in LeRobot? Works with the SO-101 out of the box?
   - Demos needed for a new task (as reported, with the source).
   - Inference speed (Hz) on modest hardware.
   - Licence: can we use it commercially for customer deployments?
3. Score each for our case: small kitting task, 50-100 demos, 5-DoF arm, laptop + cloud GPU, commercial use.
4. Recommend one model to try next and one baseline (ACT stays the baseline until something beats it on the same 20 trials).
5. Save `research/vla/YYYY-MM-DD.md`: table, recommendation, what changed since last time, source URL for every number.
6. Reply in under 8 lines.

Never take a number from a search snippet. Say "not reported" when a source doesn't give it.
