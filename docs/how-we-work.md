# How we work with Claude

## Ask any time
Write a task in the Claude project (phone or laptop). One task per message: the outcome, the input, where the result goes.

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
