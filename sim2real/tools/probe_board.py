"""Probe the Hiwonder Raspberry Pi expansion board before writing the real-hardware control loop.

Answers three questions the policy-deployment code needs and that cannot be guessed correctly:

1. IMU units/sign/noise: is accel in g or m/s^2, is gyro in deg/s or rad/s, what does "at rest" read?
2. Bus-servo read latency: is `bus_servo_read_position` fast enough to poll all 18 joints every
   policy tick (33 ms at the trained 30 Hz control rate), or is it too slow and joint velocity will
   need a different source (e.g. estimated from commanded targets instead of measured feedback)?
3. Tick <-> physical-direction convention for one servo: which way does the leg move as the raw
   position value (0-1000 range) increases?

Run this on the Pi that is physically wired to the expansion board. It only reads sensors unless
--move-test is passed, in which case it moves exactly one servo a small amount and back.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sim2real.vendor import ros_robot_controller_sdk as rrc

IMU_CHANNEL_NAMES = ("ax", "ay", "az", "gx", "gy", "gz")

# Hiwonder LX-series bus servos: keep test moves well inside the mechanical range so a
# miscalibrated joint cannot slam into a hard stop during this probe.
SAFE_TICK_RANGE = (100, 900)


def probe_imu(board: "rrc.Board", seconds: float) -> None:
    print(f"\n=== IMU: sampling for {seconds:.1f}s (keep the robot still and roughly level) ===")
    samples: list[tuple[float, ...]] = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        reading = board.get_imu()
        if reading is not None:
            samples.append(reading)
        time.sleep(0.005)

    if not samples:
        print("No IMU packets received. Check wiring/device path and that the board is powered.")
        return

    print(f"Collected {len(samples)} samples.\n")
    print(f"{'channel':>8} {'mean':>10} {'stdev':>10}")
    for i, name in enumerate(IMU_CHANNEL_NAMES):
        column = [s[i] for s in samples]
        mean = statistics.fmean(column)
        stdev = statistics.pstdev(column) if len(column) > 1 else 0.0
        print(f"{name:>8} {mean:>10.4f} {stdev:>10.4f}")

    print(
        "\nInterpretation: at rest, exactly one of ax/ay/az should be near +/-9.81 (m/s^2) or "
        "+/-1.0 (g) and the other two near 0 -- whichever axis is roughly vertical given how the "
        "board is mounted. That tells you the accel unit and which physical axis is 'up'. The gz/gy/gx "
        "mean should be near 0 with small stdev (noise floor); to find the gyro unit, rotate the "
        "board by about a quarter turn over ~1 second and re-run -- a peak near 1.57 means rad/s, a "
        "peak near 90 means deg/s."
    )


def probe_servo_read_latency(board: "rrc.Board", servo_id: int, iterations: int) -> int | None:
    print(f"\n=== Bus servo {servo_id}: read-latency benchmark ({iterations} reads) ===")
    latencies_ms: list[float] = []
    last_position: int | None = None
    for _ in range(iterations):
        start = time.perf_counter()
        result = board.bus_servo_read_position(servo_id)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        if result is not None:
            last_position = result[0]

    if last_position is None:
        print(f"No valid reply from servo {servo_id}. Check its ID and that it's powered/wired.")
        return None

    print(f"Current position (raw ticks): {last_position}")
    print(
        f"Round trip: min={min(latencies_ms):.2f}ms mean={statistics.fmean(latencies_ms):.2f}ms "
        f"max={max(latencies_ms):.2f}ms"
    )
    budget_ms = 1000.0 / 30.0
    per_joint_budget_ms = budget_ms / 18
    print(
        f"\nInterpretation: the trained policy runs at 30Hz ({budget_ms:.1f}ms per tick). Reading all "
        f"18 joints sequentially like this would cost about {statistics.fmean(latencies_ms) * 18:.1f}ms "
        f"per tick (budget is {budget_ms:.1f}ms total, i.e. {per_joint_budget_ms:.2f}ms/joint if split "
        "evenly with everything else the loop has to do). If that number is a large fraction of the "
        "budget or larger, we cannot read live joint velocity from the servos fast enough and will "
        "need another approach (e.g. finite-differencing at a slower rate, or estimating velocity from "
        "commanded targets)."
    )
    return last_position


def move_test(board: "rrc.Board", servo_id: int, delta: int, duration: float, skip_confirm: bool) -> None:
    print(f"\n=== Bus servo {servo_id}: guided move test ===")
    current = board.bus_servo_read_position(servo_id)
    if current is None:
        print(f"Could not read servo {servo_id}'s current position; aborting move test.")
        return
    current_pos = current[0]
    target_pos = max(SAFE_TICK_RANGE[0], min(SAFE_TICK_RANGE[1], current_pos + delta))
    actual_delta = target_pos - current_pos

    print(f"Current position: {current_pos} ticks. Planned move: {actual_delta:+d} ticks over {duration}s.")
    print("This will physically move ONE leg joint. Make sure it can move freely (not pinned, not near")
    print("a person's fingers, robot supported so it can't tip) before continuing.")
    if not skip_confirm:
        reply = input("Type 'yes' to proceed: ").strip().lower()
        if reply != "yes":
            print("Aborted.")
            return

    board.bus_servo_enable_torque(servo_id, True)
    board.bus_servo_set_position(duration, [[servo_id, target_pos]])
    time.sleep(duration + 0.2)

    moved = board.bus_servo_read_position(servo_id)
    moved_pos = moved[0] if moved is not None else None
    print(f"Position after move: {moved_pos} (commanded {target_pos}, delta requested {actual_delta:+d})")

    board.bus_servo_set_position(duration, [[servo_id, current_pos]])
    time.sleep(duration + 0.2)
    restored = board.bus_servo_read_position(servo_id)
    restored_pos = restored[0] if restored is not None else None
    print(f"Position after returning: {restored_pos} (should be close to original {current_pos})")

    print(
        "\nInterpretation: note which physical direction the leg moved for a positive tick delta -- "
        "that's this joint's tick-to-radian sign convention. Repeat per joint later; don't assume all "
        "18 share the same sign (left/right legs are often mirrored)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="/dev/ttyAMA0", help="Serial device for the expansion board.")
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument("--imu-seconds", type=float, default=5.0, help="How long to sample the IMU at rest.")
    parser.add_argument(
        "--servo-id", type=int, default=None, help="If given, benchmark read latency for this servo ID."
    )
    parser.add_argument(
        "--move-test", action="store_true", help="Also perform a small guided move on --servo-id and back."
    )
    parser.add_argument("--move-delta", type=int, default=30, help="Tick delta for --move-test.")
    parser.add_argument("--move-duration", type=float, default=0.3, help="Seconds for the test move.")
    parser.add_argument(
        "--yes", action="store_true", help="Skip the interactive confirmation before --move-test."
    )
    parser.add_argument(
        "--read-iterations", type=int, default=50, help="Number of reads for the latency benchmark."
    )
    args = parser.parse_args()

    board = rrc.Board(device=args.device, baudrate=args.baudrate)
    board.enable_reception()
    time.sleep(0.2)  # let the reader thread sync onto the packet stream

    probe_imu(board, args.imu_seconds)

    if args.servo_id is not None:
        probe_servo_read_latency(board, args.servo_id, args.read_iterations)
        if args.move_test:
            move_test(board, args.servo_id, args.move_delta, args.move_duration, args.yes)
    else:
        print("\n(no --servo-id given, skipping bus-servo latency/move tests)")


if __name__ == "__main__":
    main()
