"""Validated ONNX policy inference wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from .config import PolicyInterface


class OnnxPolicy:
    """Load the exported RSL-RL actor, including its embedded normalizer."""

    def __init__(self, path: str | Path, interface: PolicyInterface) -> None:
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"ONNX policy not found: {self.path}")
        self.interface = interface
        self.session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if (
            len(inputs) != 1
            or inputs[0].name != interface.input_name
            or inputs[0].shape != [1, interface.observation_size]
        ):
            raise ValueError(
                f"Policy input must be {interface.input_name}[1,{interface.observation_size}], "
                f"found {[(item.name, item.shape) for item in inputs]}"
            )
        if (
            len(outputs) != 1
            or outputs[0].name != interface.output_name
            or outputs[0].shape != [1, interface.action_size]
        ):
            raise ValueError(
                f"Policy output must be {interface.output_name}[1,{interface.action_size}], "
                f"found {[(item.name, item.shape) for item in outputs]}"
            )

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        if observation.shape != (self.interface.observation_size,):
            raise ValueError(
                f"Expected observation shape {(self.interface.observation_size,)}, got {observation.shape}"
            )
        actions = self.session.run(
            [self.interface.output_name],
            {self.interface.input_name: observation[np.newaxis, :].astype(np.float32, copy=False)},
        )[0][0]
        if actions.shape != (self.interface.action_size,) or not np.isfinite(actions).all():
            raise FloatingPointError(
                f"Invalid policy action: shape={actions.shape}, finite={np.isfinite(actions).all()}"
            )
        return actions.astype(np.float32, copy=False)
