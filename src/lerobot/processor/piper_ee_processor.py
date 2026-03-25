#!/usr/bin/env python

# Copyright 2025 THU VLN Lab. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Processor step for Piper end-effector control.

Converts delta actions from gamepad to absolute end-effector positions
that can be sent directly to the Piper robot.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature

from .core import EnvTransition, TransitionKey
from .pipeline import ProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register("piper_delta_to_absolute_ee")
@dataclass
class PiperDeltaToAbsoluteEEStep(ProcessorStep):
    """
    Converts delta end-effector actions to absolute positions for Piper robot.

    This step maintains the current end-effector position and applies delta
    movements from the teleoperator to compute new absolute positions.

    The processor reads the current EE position from the robot's observation
    at each step to ensure accurate tracking.

    Attributes:
        ee_step_size: Step size in meters for each delta unit (default 0.02m = 2cm)
        max_gripper_pos: Maximum gripper position in mm
        ee_bounds: Dictionary with min/max bounds for x, y, z in meters
    """

    ee_step_size: float = 0.02  # 2cm per unit delta
    max_gripper_pos: float = 70.0  # mm

    # Default workspace bounds (meters)
    ee_bounds: dict = field(default_factory=lambda: {
        "x_min": 0.136, "x_max": 0.38,
        "y_min": -0.159, "y_max": 0.220,
        "z_min": 0.135, "z_max": 0.3,
    })

    # Internal state
    current_gripper: float = 35.0  # mm
    _target_x: float = 0.0
    _target_y: float = 0.0
    _target_z: float = 0.0
    _target_initialised: bool = False
    use_gripper: bool = True

    def _init_target_from_obs(self, obs: dict) -> None:
        """Initialise internal target once from the first observation."""
        if not isinstance(obs, dict) or "ee.x" not in obs:
            return
        def _v(v):
            return float(v.item() if hasattr(v, 'item') else v)
        self._target_x = _v(obs["ee.x"])
        self._target_y = _v(obs["ee.y"])
        self._target_z = _v(obs["ee.z"])
        if "gripper.pos" in obs:
            self.current_gripper = _v(obs["gripper.pos"])
        self._target_initialised = True
        logging.info(
            f"PiperEE target initialised: "
            f"({self._target_x:.4f}, {self._target_y:.4f}, {self._target_z:.4f})"
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition

        new_transition = transition.copy()

        # Lazy-init: read position only once at start (like test_spacemouse2realman.py)
        if not self._target_initialised:
            obs = transition.get(TransitionKey.OBSERVATION, {})
            self._init_target_from_obs(obs)

        # Extract delta values from action
        if isinstance(action, torch.Tensor):
            action_squeezed = action.squeeze()
            if action_squeezed.dim() == 0:
                action_squeezed = action_squeezed.unsqueeze(0)
            delta_x = action_squeezed[0].item() if len(action_squeezed) > 0 else 0.0
            delta_y = action_squeezed[1].item() if len(action_squeezed) > 1 else 0.0
            delta_z = action_squeezed[2].item() if len(action_squeezed) > 2 else 0.0
            gripper_action = action_squeezed[3].item() if len(action_squeezed) > 3 else 1.0
        elif isinstance(action, dict):
            delta_x = action.get("delta_x", 0.0)
            delta_y = action.get("delta_y", 0.0)
            delta_z = action.get("delta_z", 0.0)
            gripper_action = action.get("gripper", 1.0)
        else:
            return transition

        # Accumulate delta onto internal target (NOT read-back, avoids oscillation)
        self._target_x += delta_x * self.ee_step_size
        self._target_y += delta_y * self.ee_step_size
        self._target_z += delta_z * self.ee_step_size

        # Clip to workspace bounds
        self._target_x = np.clip(self._target_x, self.ee_bounds["x_min"], self.ee_bounds["x_max"])
        self._target_y = np.clip(self._target_y, self.ee_bounds["y_min"], self.ee_bounds["y_max"])
        self._target_z = np.clip(self._target_z, self.ee_bounds["z_min"], self.ee_bounds["z_max"])

        # Handle gripper (0=close, 1=stay, 2=open)
        gripper_step = 5.0  # mm per step
        if gripper_action < 0.5:
            self.current_gripper = max(0.0, self.current_gripper - gripper_step)
        elif gripper_action > 1.5:
            self.current_gripper = min(self.max_gripper_pos, self.current_gripper + gripper_step)

        if np.random.random() < 0.05:
            logging.debug(
                f"PiperEE: delta=({delta_x:.4f}, {delta_y:.4f}, {delta_z:.4f}) "
                f"target=({self._target_x:.4f}, {self._target_y:.4f}, {self._target_z:.4f}) "
                f"gripper={self.current_gripper:.1f}"
            )

        # Build output — dimension matches use_gripper setting
        if self.use_gripper:
            absolute_action = torch.tensor([
                self._target_x, self._target_y, self._target_z,
                self.current_gripper,
            ], dtype=torch.float32)
        else:
            absolute_action = torch.tensor([
                self._target_x, self._target_y, self._target_z,
            ], dtype=torch.float32)

        new_transition[TransitionKey.ACTION] = absolute_action
        return new_transition

    def reset(self) -> None:
        """Reset internal state for new episode."""
        self._target_initialised = False
        self.current_gripper = 35.0

    def get_config(self) -> dict[str, Any]:
        return {
            "ee_step_size": self.ee_step_size,
            "max_gripper_pos": self.max_gripper_pos,
            "use_gripper": self.use_gripper,
            "ee_bounds": self.ee_bounds,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
