# SO-101 self-build: parts to order (2026-11-07)

You print the parts on the Bambu. You buy servos, boards, power, clamps.
Sources: official BOM https://github.com/TheRobotStudio/SO-ARM100 (README, "Parts For Two Arms" / "One Follower Arm"),
Eckstein configurator https://eckstein-shop.de/lerobot-soarm101-diy-parts-assembly-series.

## Servos (verified against the official BOM)
- Follower: 6x Feetech STS3215, all 1/345 gear (C001). Official spec is 7.4 V. Option: 12 V 30 kg.cm version (needs 12 V 5 A+ supply).
- Leader: 6x STS3215 7.4 V, mixed: 1x 1/345 (C001), 2x 1/191 (C044), 3x 1/147 (C046). The leader is always 7.4 V.
- Joint mapping of the leader ratios: follow the SO-101 assembly guide (the README BOM does not map them).

## Option A: follower only
| Part | China route (official BOM, EU column) | Eckstein (DE, incl. 19% VAT) |
|---|---|---|
| 6x STS3215 1/345 | 6x €12.2 | €146.10 (7.4 V) or €158.70 (12 V Pro) |
| Driver board, USB-C, power supply, 2 clamps | €11.4 + €7 + €15.7 + €8 | €33.53 (5 V 4 A) or €37.53 (12 V 5 A) |
| Screwdriver set | €9 | own tools |
| **Total** | **€124** | **€180 (7.4 V) / €196 (12 V)** |

## Option B: leader + follower
| Part | China route | Eckstein |
|---|---|---|
| 12 servos (7x 1/345, 2x 1/191, 3x 1/147) | 12x €12.2 | leader set €146.10 + follower set €146.10 (or €158.70 Pro) |
| 2 boards, 2 USB-C, 2 supplies, 4 clamps | ~€72 | 2 accessory packs €67-71 |
| Screwdriver set | €9 | own tools |
| **Total** | **€226** | **€359 (7.4 V) / €376 (12 V follower)** |

Compare: earlier Eckstein cart with printed parts €441 (€499 with 2 cameras); assembled China pair €388.

Notes
- China route: BOM links are Alibaba (servos) and Amazon.fr (rest). Check that the price includes NL VAT (21%); if not, add ~€25-50 plus a customs fee. Shipping from China 2-4 weeks.
- Eckstein: in stock, 1-3 days dispatch. NL shipping cost only shows in the cart (not checked).
- Screws: the servos come with screws in the box; the BOM lists no extra screws (only 2x M3 for the optional compliant gripper).
- Cameras not included above (wrist cam ~€20, top cam ~€37 at Eckstein).

## Printing on the Bambu (256 x 256 mm bed)
- Files: STL/SO101 in the SO-ARM100 repo. Use the Ender plates (220 x 220): Ender_Follower_SO101.stl, Ender_Leader_SO101.stl.
- Official settings: PLA+, 0.4 mm nozzle, 0.2 mm layers, 15% infill, supports on but ignore slopes over 45 deg, no supports in horizontal screw holes.
- Print the gauge first (Gauge_0.STL / Gauge_tight_1.STL) to check servo fit before printing everything.
- Material: leader in PLA+ (fine, light load). Follower in PETG, or PETG-CF for stiffness. CF needs a hardened steel nozzle (order with the printer).
- Filament: 1 kg of each is plenty for one arm each (estimate, not verified).

## Recommendation (updated 2026-11-07)
Build the follower first (Option A, ~€124 China route, or €196 Eckstein with 12 V servos). Training happens from video and the digital twin (MuJoCo), so the follower is the real-world proof that video -> twin -> robot works, and the demo for integrators.
Add the leader later (~€100 more) only if you want a teleop baseline to compare against, or to calibrate the sleeve. See docs/decisions/2026-11-07-follower-first.md.
