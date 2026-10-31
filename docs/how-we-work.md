# How we work with Claude

## Ask any time
Write a task in the Claude project (phone or laptop). One task per message: the outcome, the input, where the result goes.

## Where to see them
- Phone or browser: GitHub, repo cobot-teaching, folders `.claude/agents` and `.claude/skills`.
- Laptop: `C:\dev\cobot-teaching\.claude\agents` and `...\.claude\skills`.
- Shared facts all of them read first: `docs/context.md`. Keep it current; it is how they know the plan.

## Agents (.claude/agents)
Virtual staff with a job description. Call one by name, e.g. "use interview-analyst on these notes from my call with Growskills".

| Agent | Use it for |
|---|---|
| market-scout | Market, competitor and funding research |
| interview-analyst | Turning call notes into an interview file, updating the pipeline and the hypothesis tally |
| outreach-writer | Drafting messages in PT, NL or EN (never sends) |
| grant-writer | Go/no-go on a grant and drafting application sections |
| robot-engineer | Python for the teaching pipeline and SO-101, via pull requests |
| pilot-proposal | Turning an interested prospect into a fixed-price "new task in a week" offer |
| pitch-reviewer | Sceptical investor or assessor review before anything goes out |

Chain them for bigger jobs: "market-scout finds 10 integrators in Norte, then outreach-writer drafts messages for them".

To add an agent: one new file in .claude/agents with a name, a description (when to use it) and its rules.

## Skills (.claude/skills)
Step-by-step routines for jobs that repeat. Claude uses them by itself when the job matches, or say "use the weekly-progress skill".

| Skill | What it does | Example ask |
|---|---|---|
| analyse-interview | Notes to interview file, pipeline row, G1 tally | "Analyse my call with WRK: <notes>" |
| draft-outreach | Messages in PT, NL or EN, never sent | "Draft outreach to the next 5 PT leads" |
| grant-check | Open calls, deadlines, go/no-go | "Run the grant check" |
| weekly-progress | What moved this week, gates, next 3 tasks | "Where are we?" |
| log-decision | Records a decision and updates context | "Log the decision: PAROL6 waits until March" |

Agents are who does the work; skills are how. Several agents follow a skill (interview-analyst uses analyse-interview, outreach-writer uses draft-outreach).

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
