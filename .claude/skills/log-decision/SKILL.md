---
name: log-decision
description: Record a new decision as docs/decisions/YYYY-MM-DD-short-name.md (what, why, what would change it) and update docs/context.md. Use whenever Ricardo decides something about the idea, customers, market, pricing, hardware, funding or the company.
---

# Log a decision

1. Confirm the decision in one sentence from what Ricardo said. If it's still an open question, don't log it.
2. Create `docs/decisions/YYYY-MM-DD-short-name.md`:

   ```
   # <decision in a few words>
   Date: YYYY-MM-DD

   ## What
   ## Why
   ## What would change it
   ## Replaces
   (an earlier decision file, or "none")
   ```
3. If it replaces an earlier decision, add a line "Replaced by <file>" to that file.
4. Update the matching section of `docs/context.md` and, if it changes the summary, the "Decisions so far" list in `CLAUDE.md`.
5. Commit on a branch and open a pull request, or commit to main if Ricardo asked.
6. Reply in two lines: what was recorded and the link.
