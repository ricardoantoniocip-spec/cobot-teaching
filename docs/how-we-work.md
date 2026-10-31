# How we work with Claude

## Ask any time
Write a task in the Claude project (phone or laptop). One task per message: the outcome, the input, where the result goes.

## Where to see them
- Phone or browser: GitHub, repo cobot-teaching, folders `.claude/agents` and `.claude/skills`.
- Laptop: `C:\dev\cobot-teaching\.claude\agents` and `...\.claude\skills`.
- Shared facts all of them read first: `docs/context.md`. Keep it current; it is how they know the plan.

## Agents (.claude/agents)
Virtual staff with a name and a job description. Ask for one by name, e.g. "Iris, here are my notes from the Growskills call" or "ask Kai to check the IK for the SO-101".

Business
| Name | Agent | Use them for |
|---|---|---|
| Marco | market-scout | Market, competitor and funding research, new leads |
| Iris | interview-analyst | Call notes to interview file, pipeline and G1 tally |
| Olivia | outreach-writer | Messages in PT, NL or EN (never sends) |
| Greta | grant-writer | Go/no-go on a grant, application drafts |
| Pedro | pilot-proposal | Fixed-price "new task in a week" offers |
| Petra | pitch-reviewer | Sceptical investor or assessor review |

Technical
| Name | Agent | Use them for |
|---|---|---|
| Vera | vla-researcher | Vision-language-action models: which one, what it needs, fine-tuning plans |
| Elena | electrical-engineer | Wiring, power, sensors and ESP32 firmware for the sleeve and arms |
| Kai | kinematics-expert | Kinematics, IK, frames, calibration, retargeting hand motion to the robot |
| Tess | ml-tutor | Teaching you to record data, train and evaluate models, lesson by lesson |
| Rui | robot-engineer | Writing the pipeline code, via pull requests |

Chain them: "Vera picks a model, then Tess walks me through training it" or "Marco finds 10 integrators in Norte, then Olivia drafts messages".

To add an agent: one new file in .claude/agents with a name, a description (when to use it) and its rules.

## Skills (.claude/skills)
Step-by-step routines. Claude uses them by itself when the job matches, or say "use the wiring-check skill".

Written for us
| Skill | Used by | What it does |
|---|---|---|
| analyse-interview | Iris | Notes to interview file, pipeline row, G1 tally |
| draft-outreach | Olivia | Messages in PT, NL or EN, never sent |
| grant-check | Greta | Open calls, deadlines, go/no-go |
| weekly-progress | anyone | What moved this week, gates, next 3 tasks |
| log-decision | anyone | Records a decision and updates context |
| vla-model-scan | Vera | Compares VLA models for our setup |
| wiring-check | Elena | Wiring table, power budget, safety checklist |
| ik-check | Kai | FK, IK, workspace and retargeting checks on the real arm |
| learn-session | Tess | One lesson: idea, example, exercise, quiz, progress log |
| train-policy | Tess, Rui, Vera | Record, train, evaluate 20 trials, log hours-to-deploy |

Imported from GitHub (open-source, licence and source in each folder)
| Skill | Used by | Source |
|---|---|---|
| robotics-kinematics | Kai | MP-AI-20/mechanical-engineering-skills (MIT) |
| robotics-testing | Kai, Rui | arpitg1304/robotics-agent-skills (Apache-2.0) |
| fine-tuning-serving-openpi | Vera | Orchestra-Research/AI-Research-SKILLs (MIT) |
| rerun-lerobot | Vera, Tess | rerun-io/rerun (MIT) |
| embedded-engineer | Elena | captainluzik/oh-my-embedded (MIT, trimmed) |

Worth installing on the laptop (official Hugging Face skills, kept up to date by them). In Claude Code on the laptop type:
```
/plugin marketplace add huggingface/skills
/plugin install hf-cli@huggingface/skills
```
That lets Claude upload datasets and models to Hugging Face for you (private by default; check before anything goes public).

To add a skill: a folder in .claude/skills with a SKILL.md (name, description, steps).

## Scheduled tasks (cloud, laptop can be off)
| Task | When (Amsterdam time) | Writes |
|---|---|---|
| Market brief | Monday 07:47 | research/briefs/ |
| Grant deadlines | 1st of the month, 07:38 | funding/grants-DATE.md |
| Follow-up check | Friday 15:43 | customers/followups/ |
| Progress note | Sunday 18:41 | docs/progress/ |

Each run commits its report to main and posts a short summary in the project thread.

## Laptop work
Anything touching the arm or camera runs on the laptop (C:\dev\cobot-teaching). It has to be on with the Claude app open.
