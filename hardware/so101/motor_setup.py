"""One-motor-at-a-time helper for the SO-101 follower (LeRobot 0.6.1, Feetech STS3215).

Same result as `lerobot-setup-motors --robot.type=so101_follower`, but one motor per call,
so it can be driven step by step from chat.

    python motor_setup.py COM5 scan              # what is on the bus: ids, baud rates, voltage
    python motor_setup.py COM5 watch             # ping ids 1-6 every second (wiggle-test cables)
    python motor_setup.py COM5 set gripper      # give the ONLY connected motor id 6 @ 1 Mbps

Motor names and ids: shoulder_pan 1, shoulder_lift 2, elbow_flex 3, wrist_flex 4,
wrist_roll 5, gripper 6.
"""

import sys

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

MOTORS = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}


def make_bus(port: str) -> FeetechMotorsBus:
    return FeetechMotorsBus(
        port=port,
        motors={n: Motor(i, "sts3215", MotorNormMode.RANGE_M100_100) for n, i in MOTORS.items()},
    )


def scan(port: str) -> None:
    found = FeetechMotorsBus.scan_port(port)
    if not found:
        print("No motor answered at any baud rate. Check power, cable and port.")
        return
    bus = make_bus(port)
    bus._connect(handshake=False)
    for baud, ids in found.items():
        print(f"baud {baud}: ids {ids}")
        bus.set_baudrate(baud)
        for i in ids:
            try:
                v, _, _ = bus._read(62, 1, i)  # Present_Voltage, units of 0.1 V
                print(f"  id {i}: supply {v / 10:.1f} V")
            except Exception as e:  # noqa: BLE001
                print(f"  id {i}: voltage read failed ({e})")
    bus.port_handler.closePort()


def watch(port: str) -> None:
    """Ping ids 1-6 at 1 Mbps over and over; one line per second. Ctrl+C to stop."""
    import time

    bus = make_bus(port)
    bus._connect(handshake=False)
    bus.set_baudrate(bus.default_baudrate)
    drops = 0
    try:
        while True:
            seen = []
            for name, i in MOTORS.items():
                _, comm, _ = bus.packet_handler.ping(bus.port_handler, i)
                seen.append(f"{i}:{'ok' if comm == 0 else '--'}")
            line = "  ".join(seen)
            if "--" in line:
                drops += 1
            print(f"{time.strftime('%H:%M:%S')}  {line}   drops so far: {drops}", flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        bus.port_handler.closePort()


def set_one(port: str, motor: str) -> None:
    if motor not in MOTORS:
        sys.exit(f"Unknown motor '{motor}'. Use one of: {', '.join(MOTORS)}")
    bus = make_bus(port)
    bus.setup_motor(motor)
    bus.port_handler.closePort()
    print(f"OK: '{motor}' now has id {MOTORS[motor]} at {bus.default_baudrate} baud.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    port, cmd = sys.argv[1], sys.argv[2]
    if cmd == "scan":
        scan(port)
    elif cmd == "watch":
        watch(port)
    elif cmd == "set" and len(sys.argv) == 4:
        set_one(port, sys.argv[3])
    else:
        sys.exit(__doc__)
