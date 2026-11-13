# GR00T N1.7 one-day test plan (not started)

Status 2026-11-12: plan only. No GPU rented, nothing run. Waiting for Ricardo's go.

## The question
Does a fine-tuned GR00T N1.7 beat our ACT baseline on the SO-101 MuJoCo twin, on the same 50 sim episodes? And is NVIDIA's video_to_data (V2D) workflow usable for us?

## Short answers first
- **V2D is not a one-day test for us.** Its end-to-end example is built for the Sharpa robot hand on the Vega humanoid, in Isaac Lab, with an RL "source expert" trained for 20,000 iterations before GR00T is even touched. It needs one 48 GB GPU and Docker. Porting it to the SO-101 is weeks, not a day. Source: https://github.com/nvidia-isaac/video_to_data/blob/main/docs/e2e_workflow.md
- **GR00T on our SO-101 data is a one-day test.** Isaac-GR00T ships an SO-100/SO-101 fine-tuning example. Source: https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/SO100/README.md
- **So the day = GR00T fine-tune + twin comparison with ACT.** V2D gets 1 to 2 hours at most, only to read the code and maybe run the reconstruction stage on one iPhone video, for ideas for our own app (PR #10).

## Licence (commercial use)
- GR00T N1.7 weights: NVIDIA Open Model License. It says "Models are commercially usable", we may make and sell derivative models, NVIDIA claims no ownership of outputs. Code is Apache 2.0. Sources: https://huggingface.co/nvidia/GR00T-N1.7-3B/blob/main/LICENSE and https://github.com/NVIDIA/Isaac-GR00T#license
- Conditions to respect:
  - Write "Built on NVIDIA Cosmos" in product docs or website (backbone is Cosmos-Reason2-2B), section 3.2.
  - Ship the licence + notice "Licensed by NVIDIA Corporation under the NVIDIA Open Model License" if we hand the model to a customer, section 3.1.
  - Don't remove safety guardrails; follow NVIDIA Trustworthy AI terms, section 2.3.
  - Licence ends if we sue anyone claiming the model infringes IP, section 2.1. NVIDIA may update the terms.
  - US law (Delaware), export rules apply.
- V2D's hand reconstruction uses MANO hand models. MANO's free licence is for non-commercial research only (a commercial licence is sold separately), so V2D as-is is a research tool, not product code. Check before any customer use. Source: https://mano.is.tue.mpg.de/license.html
- Verdict: GR00T is fine for a paid deployment service. Not legal advice; re-check at first customer contract.

## GPU and cost (RunPod, on-demand, page dated 9 Nov 2026)
Source: https://www.runpod.io/pricing

| GPU | VRAM | $/hour | Notes |
|---|---|---|---|
| **L40S (pick)** | 48 GB | 1.09 | GR00T fine-tune needs 40 GB+; NVIDIA lists L40 as recommended. Also runs V2D. |
| RTX A6000 | 48 GB | 0.59 | Cheapest that fits; GR00T says it works but slower. Backup if L40S is sold out. |
| H100 PCIe | 80 GB | 2.89 | ~1.6x faster steps than L40 (115 vs 190 ms/step, GR00T README). Not worth it for 50 episodes. |

GR00T hardware needs: https://github.com/NVIDIA/Isaac-GR00T#hardware-requirements

**Estimate: about 8 GPU hours x $1.09 = ~$9, plus ~150 GB volume for a week (~$15/month pro rata, ~$4). Set a hard cap of $30 (~€28).** Stop the pod when idle; idle volume costs double.

## Before the day (free, 30 min, Ricardo)
1. Accept the gated model on Hugging Face: https://huggingface.co/nvidia/Cosmos-Reason2-2B (GR00T loads it on first use).
2. Make a HF read token for the pod (dataset ricardocipriano/so101_sim_pick_place is private).
3. Make a RunPod account, add $30 credit, no auto top-up.
4. Have the ACT baseline score: `python -m so101_sim evaluate outputs/act_so101_sim --trials 20` on the laptop. If ACT is not trained yet, train it first, or the day has nothing to compare against.

## The day (about 8 hours, mostly waiting)
| # | Step | Time | Done when |
|---|---|---|---|
| 1 | Start L40S pod (PyTorch CUDA 12.8 image), clone Isaac-GR00T and our repo, install with uv | 1 h | `uv run python -c "import gr00t"` works, GPU visible |
| 2 | Convert our dataset: LeRobot v3 -> v2 (GR00T script), AV1 -> H.264 video (GR00T can't decode AV1 reliably) | 30 min | Dataset folder loads in GR00T's data loader |
| 3 | Write `modality.json` + config for our robot: 6 joints, one `top` camera, task text "pick the cube and place it" | 30 min | Zero-shot inference on one frame returns 6 numbers |
| 4 | Fine-tune `nvidia/GR00T-N1.7-3B`, tag `NEW_EMBODIMENT`, 1 GPU, start with the example's 10,000 steps | 2 to 3 h (estimate; will measure first 100 steps) | Loss falls and flattens; checkpoint saved |
| 5 | Open-loop eval on 3 held-out episodes (GR00T's `open_loop_eval.py`) | 15 min | Predicted joints track the recorded ones in the plots |
| 6 | Closed-loop eval in the twin: run GR00T as a server on the pod, a small client in `so101_sim` sends top image + joints, gets actions. Same 20 trials and seeds as ACT | 1.5 h (client is new code, ~100 lines) | 20 trials done, videos saved |
| 7 | Optional V2D look: read code, run reconstruction on one iPhone clip | 1 to 2 h | Notes on what to borrow for PR #10 |
| 8 | Copy checkpoint + results to HF (private), stop pod, delete it | 15 min | Pod gone, spend under $30 |

Our twin renders headless on Linux with Xvfb + `MUJOCO_GL=glfw` (as in product/sim), so steps 4 to 6 all run on the pod.

## How we know it worked
- **Plumbing pass:** GR00T fine-tunes on our data and runs 20 twin trials end to end. This alone is worth the day.
- **Result pass:** GR00T scores at least 12/20 in the twin (our pass mark, product/sim/README.md).
- **Worth switching:** GR00T beats ACT by 3 or more successes out of 20 on the same seeds. If it doesn't, ACT (and next SmolVLA) stays the default, because ACT trains on Colab or the laptop for free and GR00T needs a rented GPU for every task.
- Also record: training time, $ spent, inference speed (GR00T lists ~8 Hz on L40 in plain PyTorch, which may be too slow for our 30 fps control; source: model card https://huggingface.co/nvidia/GR00T-N1.7-3B).

## Risks
- GR00T N1.7 uses relative end-effector actions by default; our data is absolute joints. Use the SO-100 example's setting (relative joints, absolute gripper) and check step 5 before burning hours.
- One low-res camera (320x240) and 50 sim episodes is small for a 3B model; it may not beat ACT. That is a valid result.
- Can't run on the laptop (GTX 1650, 4 GB) for training or inference (needs 16 GB+). A real-arm demo later needs a GPU box or the pod as a policy server.
