# Project context (read this first)

Shared facts for every agent and skill. Last updated 2026-10-31.

## Ricardo
- Production engineer in Eindhoven, mechanical engineering background, robotics minor (kinematics, IK). Python, Lean/LSS, OEE.
- Portuguese and South African. English native, Afrikaans, basic Portuguese.
- Part time, about 10-15 h/week next to a full-time job (unconfirmed). Solo now; an ML / computer vision co-founder is the gap.
- Reads on his phone. Short sentences, results first.

## The offer
- Fixed-price "new task in a week" deployment for system integrators and high-mix SMEs, built on open models (LeRobot).
- First task type: kitting and pick-and-place, then machine tending.
- Working prices (Ricardo sets the final number): EUR 7,500 per task in NL, EUR 5,000 in PT, first 3 PT pilots EUR 3,000. Support EUR 300 (NL) / EUR 200 (PT) per cell per month. Ideas from the competitor scan: a retrain price and a success clause.
- Sell hard-to-program tasks only (varying parts, kitting, flexible parts). Simple fixed tasks are cheaper to program the normal way.
- Stay robot-agnostic: UR, Doosan, FANUC CRX already installed at the customer.
- Requirement (2026-10-31): an LLM writes the native robot program (URScript first, then FANUC, Doosan) from the labelled demo steps, checked in sim and approved by an engineer. Learned policies only where variation needs them. See docs/decisions/2026-10-31-llm-writes-robot-programs.md.

## The product
- Capture: operator sensor sleeve (IMU + ArUco marker pods on torso, upper arm, forearm, back of hand; EMG for grip timing; FSR fingertip pads; 2 webcams). The back-of-hand pod matters most: the robot follows the hand in task space.
- SO-101 leader arm = baseline and ground truth for calibrating the sleeve. Then PAROL6, then a real cobot via ROS 2.
- Biggest technical risk: the gap between a human hand and a robot gripper. Answer: co-train with leader-arm demos.
- Parts for the sleeve about EUR 450-650; SO-101 kit about EUR 500 (Eckstein, Germany).
- Code: product/teach/retarget turns a video (ArUco marker on the hand) or a hand-pose CSV into an SO-101 joint trajectory and replays it (README there). Not yet tested on the real arm.
- Patents: the broad sleeve idea is prior art. Keep methods as trade secrets, publish no method details. Possibly protectable later: calibrating the sleeve against the leader arm.

## Plan and gates
- Sleeve build: 12 weeks, 2 Nov 2026 to 24 Jan 2027.
  - Gate A (week 7): IMU + camera fusion within 10 mm.
  - Gate B (week 9): EMG grip detection.
  - Gate C (week 11): retarget and replay on the arm.
  - Gate D (week 12): ACT trained on leader-only vs sleeve vs mix, 20 trials each.
- Company gates:
  - G0, 30 Nov 2026: employment contract checked (IP, side activities, non-compete).
  - G1, 31 Jan 2027: pain proven. At least 5 of 15 interviews say a new task takes 40+ hours. Fewer than 3 of 15 means rethink.
  - G2, 30 Jun 2027: demo (task set up in under 40 h) and 1 pilot agreed.
  - G3, 31 Dec 2027: co-founder signed, 2-3 paid pilots, real-cobot demo.

## Customers and leads
- Portugal first for users and pilots, Netherlands for funding and co-founder, South Africa later, US after proof.
- Lead lists: customers/pipeline-2026-10-27.csv (30 PT + NL integrators), customers/pipeline-pt-2026-10-29.csv (15 PT leads), customers/pipeline-za-2026-10-31.csv (4 SA integrators). Interviews go into customers/pipeline.csv.
- Contacts come from public sources only (company sites, press). Don't scrape LinkedIn and don't use Ricardo's own connections (his choice, 2026-10-31).
- Door openers in PT: Universal Robots Portugal, Growskills (UR distributor), INESC TEC CRIIS, PRODUTECH DIH.
- Sales argument in PT: customers can get part of their robot investment subsidised through PT2030 programmes (check the current call).
- Key hypothesis to test: programming time is a top-three barrier to more automation. Also ask integrators their day rate and hours per new task.
- Outreach: week-1 drafts with named people in customers/outreach/2026-10-31.md (4 PT to send first, plus NL and SA). Target 4 per week. Nothing sent yet as of 2026-10-31.

## Competitors (details and sources in the competitor scan, research/ once saved)
- Closest in Europe: Micropsi (Berlin), Mimic Robotics (Zurich, operator wearables), RoboTwin (Prague). Others: Wandelbots, RobCo, Standard Bots, T-Robotics, Physical Intelligence.
- No one found that sells a fixed price per task in PT or NL.

## Funding
- See funding/grants-2026-10-27.md for amounts, deadlines and source links.
- NL: WBSO 2027 deadline 20 Dec 2026 (needs an eenmanszaak + eHerkenning; the payout probably needs more hours than he has). MIT Zuid feasibility next round expected around April 2027. HighTechXL for co-founder matching.
- PT: nothing fits until there is a PT Lda. Tech Foundry Portugal expected around June 2027. INESC TEC as R&D partner later.
- EU: EIC Accelerator aimed at 2028, not the 3 Feb 2027 cut-off.

## Hard rules (from CLAUDE.md)
- Nothing from his employer in this repo: no code, data, documents or internal system names.
- Never send, post or schedule messages. Draft only.
- Source URL for every number about markets, funding or competitors.
- Never invent traction, interview results or quotes.
