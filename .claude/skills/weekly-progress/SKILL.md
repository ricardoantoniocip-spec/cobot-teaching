---
name: weekly-progress
description: Write Ricardo's weekly progress note from the repo (commits, interviews, pipeline, build plan gates, deadlines) and plan next week. Use for the Sunday progress note or when he asks "where are we?".
---

# Weekly progress note

1. Collect facts from the last 7 days:
   - `git log --since="7 days ago" --oneline` on main.
   - New files in `customers/interviews/`, changes to `customers/pipeline*.csv`, new drafts in `customers/outreach/`.
   - New files in `research/`, `funding/`, `docs/decisions/`, `hardware/`, `product/`.
2. Compare with the plan in `docs/context.md`:
   - Which week of the sleeve build is it (week 1 starts Mon 2 Nov 2026)? Is the next gate (A-D) on track?
   - Company gates G0-G3: days left, what is still missing.
   - Interview count against G1 (5 of 15 with 40+ h/task by 31 Jan 2027).
   - Deadlines in the next 30 days (from the newest `funding/grants-*.md`).
3. Write `docs/progress/YYYY-MM-DD.md`:
   - Done this week (facts only, with file links).
   - Numbers: interviews done, leads contacted, demos recorded, hours logged if known.
   - Off track or blocked, and what Ricardo needs to decide.
   - Next week: at most 3 tasks, sized for 10-15 hours.
4. Reply in under 10 lines with the same, numbers first.

Only report what the repo shows. If nothing happened, say so plainly; don't pad.
