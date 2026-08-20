"""Run the exported hexapod policy in standalone MuJoCo."""

from __future__ import annotations

import argparse
import json
import random
import threading
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

from .config import load_policy_interface
from .metrics import RolloutMetrics, plot_rollout
from .simulation import HexapodSimulation, RolloutOptions

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "assets" / "mujoco" / "hexapod.xml"
DEFAULT_INTERFACE = PROJECT_ROOT / "sim2sim" / "configs" / "policy_interface.yaml"
QUIT_KEYCODES = frozenset((27, 256, ord("q"), ord("Q")))


def _is_quit_key(keycode: int) -> bool:
    """Accept ASCII and GLFW-style Q/Escape key codes."""
    return int(keycode) in QUIT_KEYCODES


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True, help="Exported RSL-RL policy.onnx path.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--interface", type=Path, default=DEFAULT_INTERFACE)
    parser.add_argument("--duration", type=float, default=60.0, help="Maximum rollout duration; default: 60 seconds.")
    parser.add_argument(
        "--until-closed",
        action="store_true",
        help="Run until Q/Esc, viewer close, or a fall instead of using --duration.",
    )
    parser.add_argument("--settle-seconds", type=float, default=0.0, help="Optional neutral-control settling time.")
    parser.add_argument("--ramp-seconds", type=float, default=0.0, help="Optional action ramp; zero matches training.")
    parser.add_argument("--phase-offset", type=float, default=0.0, help="Gait phase offset in radians.")
    parser.add_argument("--random-phase", action="store_true", help="Sample gait phase uniformly using --seed.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--velocity-limit-mode",
        choices=("hard", "soft", "none"),
        default="hard",
        help="Joint speed treatment: legacy hard state clip, smooth braking experiment, or no speed limiter.",
    )
    parser.add_argument("--headless", action="store_true", help="Run without the interactive MuJoCo viewer.")
    parser.add_argument("--fast", action="store_true", help="Do not synchronize viewer playback to real time.")
    parser.add_argument("--validate-only", action="store_true", help="Validate model/policy/config and exit.")
    parser.add_argument("--plot", action="store_true", help="Write rollout.png after execution.")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.duration <= 0.0 or args.settle_seconds < 0.0 or args.ramp_seconds < 0.0:
        raise ValueError("Duration must be positive; settle and ramp times cannot be negative")
    if args.until_closed and args.headless:
        raise ValueError("--until-closed requires the interactive viewer; do not combine it with --headless")

    interface = load_policy_interface(args.interface)
    simulation = HexapodSimulation(args.model, args.policy, interface)
    print(json.dumps(simulation.contract_summary(), indent=2))
    if args.validate_only:
        return

    random_generator = random.Random(args.seed)
    phase_offset = random_generator.uniform(0.0, 2.0 * 3.141592653589793) if args.random_phase else args.phase_offset
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = PROJECT_ROOT / "sim2sim" / "results" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    metrics = RolloutMetrics(output_dir, interface.policy_joint_names)
    options = RolloutOptions(
        duration_seconds=None if args.until_closed else args.duration,
        settle_seconds=args.settle_seconds,
        ramp_seconds=args.ramp_seconds,
        real_time=not args.fast and not args.headless,
        phase_offset=phase_offset,
        velocity_limit_mode=args.velocity_limit_mode,
    )

    stop_event = threading.Event()

    def key_callback(keycode: int) -> None:
        if _is_quit_key(keycode):
            stop_event.set()

    if args.headless:
        viewer_context = nullcontext(None)
    else:
        import mujoco.viewer

        simulation.reset()
        viewer_context = mujoco.viewer.launch_passive(
            simulation.model,
            simulation.data,
            key_callback=key_callback,
        )
        print("Viewer controls: press Q or Esc to stop and save rollout metrics.")

    with viewer_context as viewer:
        if viewer is not None:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = simulation.base_body_id
            viewer.cam.distance = 0.8
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -25.0
        simulation.run(
            options=options,
            metrics=metrics,
            viewer=viewer,
            stop_requested=stop_event.is_set,
        )

    if args.plot:
        plot_rollout(metrics.csv_path, output_dir / "rollout.png")
    print(f"Metrics: {metrics.csv_path}")
    print(f"Summary: {metrics.summary_path}")


if __name__ == "__main__":
    main()
