# The LLM fills a task file; fixed code writes the robot program (2026-11-13)

**What:** Refines docs/decisions/2026-10-31-llm-writes-robot-programs.md. The LLM does not write
URScript. A vision LLM names and labels the demo steps; the measured demo gives every position;
the result is a brand-neutral task file (pick, place, move, home, wait; JSON, mm). A fixed,
deterministic generator turns the task file plus a cell file (robot, tool, speeds) into URScript
or an SO-101 trajectory, checks it offline, then in URSim and the MuJoCo twin. Built in
product/teach/taskprog. Take 1 passes all three checks.

**Why:** A program an integrator will put on a real cell must be repeatable and checkable. Free
LLM code can differ on every run and can hide a wrong number. With a task file, the LLM can only
get names and labels wrong, which the engineer sees at a glance, and the same task file serves
UR now and FANUC or Doosan later (one new generator each). The pitch stays true: "AI turns your
demo into a native UR program", with the AI doing the understanding and fixed code doing the
writing.

**What would change it:** tasks that the five skills cannot express (then add skills, not free
code), customers who want to edit the program in the robot's own editor rather than the task file
(then generate Polyscope programs), or a model that writes verified robot code reliably enough to
skip the task file.
