"""Contract tests for the standalone MuJoCo policy runner."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from sim2sim.hexapod_mujoco.config import load_policy_interface
from sim2sim.hexapod_mujoco.joint_mapping import JointMapping
from sim2sim.hexapod_mujoco.metrics import RolloutMetrics
from sim2sim.hexapod_mujoco.observations import ObservationBuilder
from sim2sim.hexapod_mujoco.policy import OnnxPolicy
from sim2sim.hexapod_mujoco.simulation import HexapodSimulation, RolloutOptions

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "assets" / "mujoco" / "hexapod.xml"
INTERFACE_PATH = PROJECT_ROOT / "sim2sim" / "configs" / "policy_interface.yaml"


@pytest.fixture(scope="module")
def interface():
    return load_policy_interface(INTERFACE_PATH)


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(MODEL_PATH))


def test_model_dimensions_and_timing(model, interface) -> None:
    assert (model.nq, model.nv, model.nu) == (25, 24, 18)
    assert model.nkey == 1
    assert np.isclose(model.opt.timestep, interface.physics_dt)
    assert np.isclose(model.opt.timestep * interface.decimation, interface.policy_dt)


def test_name_based_joint_mapping_is_complete(model, interface) -> None:
    mapping = JointMapping.from_model(model, interface.policy_joint_names)
    assert set(mapping.policy_names) == set(mapping.mujoco_names)
    assert len(set(mapping.qpos_addresses)) == 18
    assert len(set(mapping.dof_addresses)) == 18
    assert len(set(mapping.actuator_ids)) == 18
    assert np.allclose(mapping.lower_limits, -0.3491)
    assert np.allclose(mapping.upper_limits, 0.3491)
    assert np.allclose(mapping.neutral_positions, 0.0)
    assert np.allclose(model.dof_armature[mapping.dof_addresses], 0.0001)


def test_neutral_observation_matches_isaac_contract(model, interface) -> None:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    mapping = JointMapping.from_model(model, interface.policy_joint_names)
    observation = ObservationBuilder(model, interface, mapping).build(data, control_step=0)

    assert observation.shape == (70,)
    assert observation.dtype == np.float32
    assert np.allclose(observation[0:6], 0.0, atol=1.0e-7)
    assert np.allclose(observation[6:9], [0.0, 0.0, -1.0], atol=1.0e-7)
    assert np.allclose(observation[9:12], [0.5, 0.0, 0.0], atol=1.0e-7)
    assert np.allclose(observation[12:14], [1.0, 0.0], atol=1.0e-7)
    assert np.allclose(observation[14:16], [0.0, 1.0], atol=1.0e-7)
    assert np.allclose(observation[16:], 0.0, atol=1.0e-7)


def test_position_drive_parameters(model, interface) -> None:
    mapping = JointMapping.from_model(model, interface.policy_joint_names)
    expected_kp = []
    expected_kv = []
    for name in interface.policy_joint_names:
        if name.startswith("body_leg_"):
            expected_kp.append(40.0)
            expected_kv.append(2.0)
        elif name.endswith("_1_2"):
            expected_kp.append(50.0)
            expected_kv.append(2.5)
        else:
            expected_kp.append(60.0)
            expected_kv.append(3.0)
    assert np.allclose(model.actuator_gainprm[mapping.actuator_ids, 0], expected_kp)
    assert np.allclose(-model.actuator_biasprm[mapping.actuator_ids, 2], expected_kv)
    assert np.allclose(model.actuator_forcerange[mapping.actuator_ids], [-1.961, 1.961])


def test_exported_policy_if_available(model, interface) -> None:
    policies = sorted((PROJECT_ROOT / "logs" / "rsl_rl" / "hexapod_direct").glob("*/exported/policy.onnx"))
    if not policies:
        pytest.skip("No locally exported policy.onnx is available")
    policy = OnnxPolicy(policies[-1], interface)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    mapping = JointMapping.from_model(model, interface.policy_joint_names)
    observation = ObservationBuilder(model, interface, mapping).build(data, control_step=0)
    actions = policy(observation)
    assert actions.shape == (18,)
    assert np.isfinite(actions).all()


def test_short_headless_control_loop_if_policy_available(interface, tmp_path) -> None:
    policies = sorted((PROJECT_ROOT / "logs" / "rsl_rl" / "hexapod_direct").glob("*/exported/policy.onnx"))
    if not policies:
        pytest.skip("No locally exported policy.onnx is available")
    simulation = HexapodSimulation(MODEL_PATH, policies[-1], interface)
    metrics = RolloutMetrics(tmp_path, interface.policy_joint_names)
    summary = simulation.run(options=RolloutOptions(duration_seconds=0.1), metrics=metrics)
    assert summary["samples"] == 3
    assert summary["maximum_abs_joint_velocity"] <= interface.joint_velocity_limit + 1.0e-9
    assert metrics.csv_path.is_file()
    assert metrics.summary_path.is_file()
