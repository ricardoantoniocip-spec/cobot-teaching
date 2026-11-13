# Project context (read this first)

Shared facts for every agent and skill. Last updated 2026-11-13.

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
- How (2026-11-13): the vision LLM only names and labels demo steps; a brand-neutral task file (pick/place/move, mm) plus a cell file go through a fixed generator to URScript or an SO-101 trajectory, checked offline, in URSim and in the twin. product/teach/taskprog; take 1 passes all three. See docs/decisions/2026-11-13-task-file-then-code.md.
- Task format 0.2 (2026-11-13): grid locations (tray, pallet), set_output, wait_input with timeout, repeat. Kitting and machine tending examples pass URSim (machine simulated). See docs/decisions/2026-11-13-grids-and-signals-in-task-file.md.
- Requirement (2026-10-31): an LLM writes the native robot program (URScript first, then FANUC, Doosan) from the labelled demo steps, checked in sim and approved by an engineer. Learned policies only where variation needs them. See docs/decisions/2026-10-31-llm-writes-robot-programs.md.

## The product
- Capture: operator sensor sleeve (IMU + ArUco marker pods on torso, upper arm, forearm, back of hand; EMG for grip timing; FSR fingertip pads; 2 webcams). The back-of-hand pod matters most: the robot follows the hand in task space.
- SO-101 follower = real-world test of video and digital-twin training (built first). Leader arm optional later as baseline and sleeve calibration. Then PAROL6, then a real cobot via ROS 2.
- Biggest technical risk: the gap between a human hand and a robot gripper. Answer: co-train with leader-arm demos.
- Parts for the sleeve about EUR 450-650. SO-101 pair: self-print on the Bambu, buy servos and electronics: about EUR 226 China route or EUR 376 Eckstein (hardware/so101/build-parts-list.md). Decision 2026-11-07: build the follower first (about EUR 124); leader optional later.
- Code: product/teach/retarget turns a video (ArUco marker on the hand) or a hand-pose CSV into an SO-101 joint trajectory and replays it (README there). Not yet tested on the real arm.
- Digital twin (2026-11-08): product/sim is a MuJoCo SO-101 with a pick-and-place table. Scripted demo -> retarget -> twin replay succeeds 20/20; episodes export to a LeRobotDataset for ACT training and `evaluate` scores a trained policy in the twin (README there). Not yet run on the laptop.
- Video app (2026-11-12): `python -m so101_sim from-video my_task.mp4` tracks the operator's arm in a plain video (MediaPipe, CPU), maps the pinch point to SO-101 joints, replays in the twin and saves a LeRobot episode if it succeeds. Default "body mode" scales his arm to the robot (not true mm); table-marker mode gives true positions. Tested on made-up landmarks only, not yet on a real pick-and-place video (product/sim/README.md, decision 2026-11-12).
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
- Outreach: week-1 drafts in customers/outreach/2026-10-31.md (4 PT first, plus NL and SA); drafts and LinkedIn links for the 15 "connect next" people in customers/outreach/2026-11-07.md. Target 4 per week. Nothing sent yet as of 2026-11-07.

## Competitors (details and sources in the competitor scan, research/ once saved)
- Closest in Europe: Micropsi (Berlin), Mimic Robotics (Zurich, operator wearables), RoboTwin (Prague). Others: Wandelbots, RobCo, Standard Bots, T-Robotics, Physical Intelligence.
- New in Oct 2026: Machenta (London, $40M seed, one-shot task learning trained in sim) and Bleu Robotics (France, EUR 5M pre-seed, few-example learning for humanoids). See research/briefs/2026-11-02-market-brief.md.
- New in Nov 2026: Tutor Intelligence (US, $75M Series B) sells robot work by the hour or pick, trained from human demos, own robots in US warehouses. See research/briefs/2026-11-09-market-brief.md.
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
