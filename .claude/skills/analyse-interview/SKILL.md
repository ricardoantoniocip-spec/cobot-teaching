---
name: analyse-interview
description: Turn customer call notes, a voice memo transcript or a Wispr Flow meeting into the standard interview file, update the pipeline and the hypothesis tally. Use after every customer or integrator conversation.
---

# Analyse an interview

1. Get the raw input: pasted notes, a transcript, or a meeting from Wispr Flow if Ricardo names one. Ask for the company name and date only if they are missing.
2. Copy `customers/interviews/TEMPLATE.md` to `customers/interviews/YYYY-MM-DD-<company>.md` (company in lowercase, hyphens).
3. Fill every field from the input only.
   - Use their numbers exactly as said (hours per task, batch sizes, day rate).
   - Anything not discussed: write "not asked".
   - Put the person's own words under Quotes, in the original language.
4. Update `customers/pipeline.csv`: add or update the row (status `interviewed`, last_contact, next_step, interview_file). If the company is in one of the lead lists, copy its source and why_them.
5. Recount the tally across all files in `customers/interviews/`:
   - How many name programming time as a top-three barrier.
   - How many say a new task takes 40+ hours (gate G1 needs 5 of 15 by 31 Jan 2027).
   - Integrator day rates and hours per task heard so far.
6. Save the tally in `customers/tally.md` (one line per interview, totals at the top).
7. Reply to Ricardo in under 8 lines: what was new, support or weaken the hypothesis, tally against G1, the one follow-up to send. If a pilot looks possible, suggest the pilot-proposal agent.
8. Commit on a branch and open a pull request, unless Ricardo asked to commit to main.

Never invent quotes or numbers. Never put anything from Ricardo's employer in the file.
