# Kitting and machine tending are task-file skills, written out step by step (2026-11-13)

**What:** Task format 0.2 adds what the first two task types need. A location can be a grid
(tray or pallet: slot 1 taught, the rest from count and pitch). New skills `set_output` and
`wait_input` use signal names; the cell file maps each name to a pin. `repeat` exists in the
task file, but the generator writes every round out in the robot program. `wait_input` always
has a timeout that stops the program. Built in product/teach/taskprog; both examples pass URSim.

**Why:** Kitting and machine tending are the first task types we sell (CLAUDE.md). Keeping pins
out of the task keeps it brand-neutral and reusable on a cell wired differently. Writing loops
out makes every step checkable offline and in URSim (pose after each step), which matters more
now than a short program. A wait without a timeout hangs a cell silently.

**What would change it:** integrators who want a real loop with a counter in the program (to
resume a half-full pallet), large pallets where the written-out program gets too long for the
controller, or signals that need more than on/off (analog, Modbus, PROFINET).
