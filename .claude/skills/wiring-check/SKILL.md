---
name: wiring-check
description: Plan or check wiring, power and connectors for the sensor sleeve (ESP32-S3, BNO085 IMUs, MyoWare 2.0 EMG, FSR pads) and the SO-101 arms (servo bus, power supplies). Use before soldering, powering up, ordering parts, or when something electrical misbehaves.
---

# Wiring check

1. Read `docs/context.md` and `hardware/` for the current parts list and any existing diagrams.
2. For every part, look up the datasheet or official guide and note: supply voltage, logic level, current draw, interface (I2C, SPI, UART, analog), pins. Link each source.
3. Make or check the wiring table in `hardware/wiring-<subsystem>.md`:
   | From (part, pin) | To (part, pin) | Signal | Voltage | Wire / connector | Notes |
4. Power budget: sum the current per supply rail, add 30 % margin, pick the regulator or supply. Keep the 12 V follower-arm supply and the 7.4 V or 5 V leader-arm supply separate; never swap them.
5. Check the usual mistakes:
   - 5 V into a 3.3 V pin (ESP32-S3 GPIOs are 3.3 V only).
   - Missing common ground between boards.
   - I2C address clashes (two BNO085 on one bus need different addresses or a second bus) and pull-up resistors.
   - ADC pins that clash with Wi-Fi on the ESP32 (use ADC1 pins for EMG and FSR).
   - FSR needs a voltage divider; pick the resistor for the force range.
   - Cable strain relief and connectors on a moving arm (sleeve cables must survive bending).
6. Safety rules, always:
   - Anything attached to a person's skin (EMG electrodes, sleeve) runs on battery or from a laptop running on battery, or through a USB isolator. Never from a mains supply or a laptop on its charger while worn.
   - Fuse every supply that can deliver more than 2 A. Servo supplies get a power switch within reach.
   - LiPo batteries: protected cells, a proper charger, never charged unattended.
7. Write what to buy (part, quantity, shop link, price) if anything is missing.
8. Reply in under 10 lines: OK or what to fix, in order of risk.

If unsure about a pin or a voltage, say so and point to the datasheet page. Never guess on power.
