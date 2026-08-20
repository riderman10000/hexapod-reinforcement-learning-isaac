"""Run controlled MuJoCo A/B tests for joint velocity-limit treatments."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt

from sim2sim.hexapod_mujoco.config import load_policy_interface
from sim2sim.hexapod_mujoco.metrics import RolloutMetrics
from sim2sim.hexapod_mujoco.simulation import HexapodSimulation, RolloutOptions

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "assets" / "mujoco" / "hexapod.xml"
DEFAULT_INTERFACE = PROJECT_ROOT / "sim2sim" / "configs" / "policy_interface.yaml"
MODES = ("hard", "soft", "none")


def _plot(summaries: dict[str, dict], output_path: Path) -> None:
    metrics = [
        ("mean_world_forward_velocity", "forward velocity [m/s]"),
        ("mean_abs_world_lateral_velocity", "absolute lateral velocity [m/s]"),
        ("mean_abs_yaw_rate", "absolute yaw rate [rad/s]"),
        ("maximum_joint_velocity_limit_ratio", "maximum speed / configured limit"),
        ("action_sign_change_fraction", "action sign-change fraction"),
        ("actuator_force_component_saturation_fraction", "force saturation fraction"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    for axis, (metric, label) in zip(axes.flat, metrics, strict=True):
        values = [summaries[mode][metric] for mode in MODES]
        axis.bar(MODES, values, color=("#4c78a8", "#59a14f", "#e15759"))
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("MuJoCo velocity-limit A/B comparison")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--interface", type=Path, default=DEFAULT_INTERFACE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--random-phase", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.duration <= 0.0:
        raise ValueError("Duration must be positive")

    interface = load_policy_interface(args.interface)
    random_generator = random.Random(args.seed)
    phase_offset = random_generator.uniform(0.0, 2.0 * 3.141592653589793) if args.random_phase else 0.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for mode in MODES:
        mode_dir = args.output_dir / mode
        simulation = HexapodSimulation(args.model, args.policy, interface)
        metrics = RolloutMetrics(mode_dir, interface.policy_joint_names)
        summaries[mode] = simulation.run(
            options=RolloutOptions(
                duration_seconds=args.duration,
                phase_offset=phase_offset,
                velocity_limit_mode=mode,
            ),
            metrics=metrics,
        )
    payload = {
        "schema_version": 1,
        "duration_seconds": args.duration,
        "phase_offset": phase_offset,
        "modes": summaries,
    }
    with (args.output_dir / "velocity_mode_comparison.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    _plot(summaries, args.output_dir / "velocity_mode_comparison.png")
    print(json.dumps(payload, indent=2))
    print(f"Velocity-mode comparison: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
