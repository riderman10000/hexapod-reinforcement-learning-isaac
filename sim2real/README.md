# Real-hardware deployment (Hiwonder LX-224HV + Raspberry Pi)

**Status: gathering hardware data before writing the control loop.** No policy runs on real hardware yet.
This mirrors the discipline in [`sim2sim/`](../sim2sim/): verify the actual interface empirically before
building on top of it, rather than guessing.

## Hardware

- 18x Hiwonder LX-224HV serial bus servos (same Hiwonder bus protocol family as the LX-16A/LX-224).
- Onboard Raspberry Pi, connected to a Hiwonder "Raspberry Pi expansion board" carrying an STM32
  sub-controller. That sub-controller talks to the bus servos *and* an onboard MPU6050 IMU, and relays
  both to the Pi over one serial link (`/dev/ttyAMA0` at 1,000,000 baud by default).
- [`vendor/ros_robot_controller_sdk.py`](vendor/ros_robot_controller_sdk.py) is Hiwonder's own SDK for that
  board (from their TonyPi/SpiderPi example code), vendored here unmodified since it isn't a pip package.
  It exposes `Board.get_imu()` and `Board.bus_servo_set_position/read_position/enable_torque(...)`, which is
  everything the deployment runner needs — there is no separate raw-protocol driver to write.

## Why this isn't the full runner yet

The policy's observation contract ([`sim2sim/configs/policy_interface.yaml`](../sim2sim/configs/policy_interface.yaml))
needs several things that don't map onto this SDK for free, and guessing wrong here means either the runner
doesn't work or, worse, sends a bad command to a real motor:

- **No body linear velocity sensor.** MuJoCo/Isaac give this for free; real hardware needs a state estimator
  (IMU + leg/kinematic odometry) that doesn't exist yet.
- **IMU is raw accel+gyro only** (`get_imu()` returns 6 floats: `ax, ay, az, gx, gy, gz`), with unknown units
  until measured — could be g's or m/s^2, deg/s or rad/s. No fused orientation is provided, so
  `projected_gravity_body` needs a filter (e.g. complementary filter) built on top of raw accel+gyro.
- **No joint velocity feedback.** Only `bus_servo_read_position` exists; velocity would have to be
  finite-differenced, and `bus_servo_read_position` looks like a synchronous, one-servo-at-a-time round
  trip — unknown yet whether reading all 18 joints fits inside a 30 Hz (33 ms) control tick.
- **Servo positions are raw ticks (0-1000-ish range), not radians.** Need a per-joint ticks<->radians
  calibration, a sign convention per joint (legs are likely mirrored left/right), and to reconcile the
  servo's own hardware trim offset (`bus_servo_set_offset`) with the policy's software `default_joint_pos`.
- **`bus_servo_set_position(duration, positions)` is a timed move command**, not an instantaneous
  torque/PD target — needs a short `duration` per tick to approximate continuous tracking, and that
  hasn't been validated against how the servo firmware actually behaves under rapidly-updated setpoints.

## Step 1: probe the board

Run on the Pi that's wired to the expansion board:

```bash
# IMU-only pass first (robot still, roughly level)
python -m sim2real.tools.probe_board --imu-seconds 5

# add a read-latency benchmark against one known servo ID
python -m sim2real.tools.probe_board --imu-seconds 5 --servo-id 1

# once you're confident the robot is supported and that joint can move freely,
# also run the guided move test to determine that joint's tick<->direction sign
python -m sim2real.tools.probe_board --servo-id 1 --move-test
```

[`tools/probe_board.py`](tools/probe_board.py) reports IMU mean/stdev per channel (to work out units and
sign), the round-trip latency of a single `bus_servo_read_position` call (to judge whether polling all 18
joints per tick is feasible), and — only with `--move-test`, and only after an interactive confirmation —
moves exactly one servo a small amount and back so you can observe which physical direction corresponds to
an increasing raw position value for that joint.

Report back what it prints (especially the latency numbers and the IMU means at rest) before the state
estimator, calibration table, and control loop get built — those depend directly on these numbers.
