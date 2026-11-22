# Source

Imported 2026-11-14 from https://github.com/trailofbits/skills/tree/main/plugins/modern-python (commit 149d7b2).
Licence: CC BY-SA 4.0 (see LICENSE). Copyright Trail of Bits (author William Tan). If we ever publish a changed version, it must keep this licence and credit.

Changes:
- Copied the skill only. Left out the plugin's SessionStart hook, which puts shims on PATH that block bare `python` and `pip`. That would break `python -m so101_sim` and LeRobot commands on the Windows laptop.
- Added a short "In this repo" note at the top of SKILL.md.
