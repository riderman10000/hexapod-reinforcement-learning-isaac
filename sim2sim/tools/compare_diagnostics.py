"""Compare Isaac and MuJoCo diagnostic captures and generate plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _load_sweep(path: Path) -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            joint = grouped.setdefault(
                row["joint_name"],
                {name: [] for name in ("time", "position_delta", "foot_dx", "foot_dy", "foot_dz")},
            )
            for name in joint:
                joint[name].append(float(row[name]))
    return {
        joint_name: {name: np.asarray(values) for name, values in signals.items()}
        for joint_name, signals in grouped.items()
    }


def _cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator < 1.0e-10:
        return float("nan")
    return float(np.dot(first, second) / denominator)


def _joint_grid_names() -> list[list[str]]:
    return [
        [f"body_leg_{leg}", f"leg_{leg}_1_2", f"leg_{leg}_2_3"]
        for leg in range(6)
    ]


def _compare_observations(isaac: dict, mujoco: dict, output_dir: Path) -> dict:
    isaac_observation = np.asarray(isaac["observation"])
    mujoco_observation = np.asarray(mujoco["observation"])
    isaac_action = np.asarray(isaac["raw_action"])
    mujoco_action = np.asarray(mujoco["raw_action"])
    term_metrics = {}
    for term_name, isaac_values in isaac["observation_terms"].items():
        first = np.asarray(isaac_values)
        second = np.asarray(mujoco["observation_terms"][term_name])
        difference = second - first
        term_metrics[term_name] = {
            "rmse": float(np.sqrt(np.mean(np.square(difference)))),
            "maximum_absolute_error": float(np.max(np.abs(difference))),
        }

    figure, axes = plt.subplots(2, 1, figsize=(12, 8))
    term_names = list(term_metrics)
    axes[0].bar(range(len(term_names)), [term_metrics[name]["maximum_absolute_error"] for name in term_names])
    axes[0].set_xticks(range(len(term_names)), term_names, rotation=35, ha="right")
    axes[0].set_ylabel("maximum absolute error")
    axes[0].set_title("Neutral observation parity: MuJoCo minus Isaac")
    action_indices = np.arange(len(isaac_action))
    width = 0.42
    axes[1].bar(action_indices - width / 2, isaac_action, width, label="Isaac observation")
    axes[1].bar(action_indices + width / 2, mujoco_action, width, label="MuJoCo observation")
    axes[1].set_xlabel("policy action index")
    axes[1].set_ylabel("raw ONNX action")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "observation_action_comparison.png", dpi=160)
    plt.close(figure)

    observation_difference = mujoco_observation - isaac_observation
    action_difference = mujoco_action - isaac_action
    return {
        "observation_rmse": float(np.sqrt(np.mean(np.square(observation_difference)))),
        "observation_maximum_absolute_error": float(np.max(np.abs(observation_difference))),
        "raw_action_rmse": float(np.sqrt(np.mean(np.square(action_difference)))),
        "raw_action_maximum_absolute_error": float(np.max(np.abs(action_difference))),
        "terms": term_metrics,
    }


def _compare_sweeps(isaac: dict, mujoco: dict, output_dir: Path, joint_delta: float) -> dict:
    missing = sorted(set(isaac) ^ set(mujoco))
    if missing:
        raise ValueError(f"Joint sweep files contain different joint sets: {missing}")
    joint_results = {}
    figure, axes = plt.subplots(6, 3, figsize=(15, 18), sharex=True, sharey=True)
    for leg_index, row_names in enumerate(_joint_grid_names()):
        for column_index, joint_name in enumerate(row_names):
            isaac_signals = isaac[joint_name]
            mujoco_signals = mujoco[joint_name]
            isaac_final_foot = np.asarray(
                [isaac_signals[axis][-1] for axis in ("foot_dx", "foot_dy", "foot_dz")]
            )
            mujoco_final_foot = np.asarray(
                [mujoco_signals[axis][-1] for axis in ("foot_dx", "foot_dy", "foot_dz")]
            )
            direction_cosine = _cosine_similarity(isaac_final_foot, mujoco_final_foot)
            isaac_final_position = float(isaac_signals["position_delta"][-1])
            mujoco_final_position = float(mujoco_signals["position_delta"][-1])
            response_threshold = min(0.005, 0.25 * joint_delta)
            coordinate_sign_match = (
                isaac_final_position > response_threshold and mujoco_final_position > response_threshold
            )
            foot_direction_match = bool(np.isfinite(direction_cosine) and direction_cosine >= 0.8)
            joint_results[joint_name] = {
                "isaac_final_position_delta": isaac_final_position,
                "mujoco_final_position_delta": mujoco_final_position,
                "isaac_final_foot_delta": isaac_final_foot.tolist(),
                "mujoco_final_foot_delta": mujoco_final_foot.tolist(),
                "foot_direction_cosine": direction_cosine,
                "coordinate_sign_match": coordinate_sign_match,
                "foot_direction_match": foot_direction_match,
                "axis_pass": coordinate_sign_match and foot_direction_match,
            }
            axis = axes[leg_index, column_index]
            axis.plot(isaac_signals["time"], isaac_signals["position_delta"], label="Isaac")
            axis.plot(mujoco_signals["time"], mujoco_signals["position_delta"], label="MuJoCo")
            axis.axhline(joint_delta, color="black", linestyle="--", linewidth=0.7)
            cosine_text = "n/a" if not np.isfinite(direction_cosine) else f"{direction_cosine:.2f}"
            axis.set_title(f"{joint_name}\nfoot cosine={cosine_text}")
            if leg_index == 5:
                axis.set_xlabel("time [s]")
            if column_index == 0:
                axis.set_ylabel("joint delta [rad]")
    axes[0, 0].legend()
    figure.suptitle("+0.05 rad joint-step response and downstream-foot direction", y=1.0)
    figure.tight_layout()
    figure.savefig(output_dir / "joint_sweep_comparison.png", dpi=160)
    plt.close(figure)
    return {
        "all_axes_pass": all(result["axis_pass"] for result in joint_results.values()),
        "passing_joint_count": sum(result["axis_pass"] for result in joint_results.values()),
        "joint_count": len(joint_results),
        "joints": joint_results,
    }


def _write_report(path: Path, result: dict) -> None:
    observation = result["observation_action"]
    sweep = result["joint_sweep"]
    lines = [
        "# Isaac–MuJoCo diagnostic comparison",
        "",
        f"- Observation RMSE: `{observation['observation_rmse']:.6g}`",
        f"- Maximum observation error: `{observation['observation_maximum_absolute_error']:.6g}`",
        f"- Raw-action RMSE: `{observation['raw_action_rmse']:.6g}`",
        f"- Maximum raw-action error: `{observation['raw_action_maximum_absolute_error']:.6g}`",
        f"- Axis checks passed: `{sweep['passing_joint_count']}/{sweep['joint_count']}`",
        "",
        "| Joint | Isaac final Δq | MuJoCo final Δq | Foot direction cosine | Pass |",
        "|---|---:|---:|---:|:---:|",
    ]
    for joint_name, joint in sweep["joints"].items():
        cosine = joint["foot_direction_cosine"]
        cosine_text = "n/a" if not np.isfinite(cosine) else f"{cosine:.3f}"
        lines.append(
            f"| `{joint_name}` | {joint['isaac_final_position_delta']:.4f} | "
            f"{joint['mujoco_final_position_delta']:.4f} | {cosine_text} | "
            f"{'yes' if joint['axis_pass'] else 'no'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-dir", type=Path, required=True)
    parser.add_argument("--mujoco-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    isaac_snapshot = _load_json(args.isaac_dir / "neutral_snapshot.json")
    mujoco_snapshot = _load_json(args.mujoco_dir / "neutral_snapshot.json")
    isaac_summary = _load_json(args.isaac_dir / "joint_sweep_summary.json")
    mujoco_summary = _load_json(args.mujoco_dir / "joint_sweep_summary.json")
    if isaac_summary["joint_delta"] != mujoco_summary["joint_delta"]:
        raise ValueError("Joint sweeps used different command deltas")
    result = {
        "schema_version": 1,
        "observation_action": _compare_observations(isaac_snapshot, mujoco_snapshot, args.output_dir),
        "joint_sweep": _compare_sweeps(
            _load_sweep(args.isaac_dir / "joint_sweep.csv"),
            _load_sweep(args.mujoco_dir / "joint_sweep.csv"),
            args.output_dir,
            float(isaac_summary["joint_delta"]),
        ),
    }
    with (args.output_dir / "comparison.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    _write_report(args.output_dir / "report.md", result)
    print(json.dumps(result, indent=2))
    print(f"Comparison report: {(args.output_dir / 'report.md').resolve()}")


if __name__ == "__main__":
    main()
