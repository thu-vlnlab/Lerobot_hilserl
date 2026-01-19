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

    # Internal state for gripper
    current_gripper: float = 35.0  # mm

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Process transition: convert delta action to absolute EE position.

        Uses the current robot EE position from observation and applies
        delta movements to compute target position.

        Args:
            transition: Environment transition containing delta action

        Returns:
            Modified transition with absolute EE position action
        """
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition

        new_transition = transition.copy()

        # Get current EE position from observation (raw joint positions stored in OBSERVATION)
        obs = transition.get(TransitionKey.OBSERVATION, {})

        # Default position if observation not available
        current_x = 0.3
        current_y = 0.0
        current_z = 0.2

        # Try to get current EE position from observation
        if isinstance(obs, dict):
            # Check for EE position keys (from PiperFollowerEndEffector)
            if "ee.x" in obs:
                val = obs["ee.x"]
                current_x = float(val.item() if hasattr(val, 'item') else val)
            if "ee.y" in obs:
                val = obs["ee.y"]
                current_y = float(val.item() if hasattr(val, 'item') else val)
            if "ee.z" in obs:
                val = obs["ee.z"]
                current_z = float(val.item() if hasattr(val, 'item') else val)
            # Get current gripper position
            if "gripper.pos" in obs:
                val = obs["gripper.pos"]
                self.current_gripper = float(val.item() if hasattr(val, 'item') else val)

        # Extract delta values from action
        if isinstance(action, torch.Tensor):
            action_squeezed = action.squeeze()
            if action_squeezed.dim() == 0:
                action_squeezed = action_squeezed.unsqueeze(0)

            # Action format: [delta_x, delta_y, delta_z, (gripper)]
            delta_x = action_squeezed[0].item() if len(action_squeezed) > 0 else 0.0
            delta_y = action_squeezed[1].item() if len(action_squeezed) > 1 else 0.0
            delta_z = action_squeezed[2].item() if len(action_squeezed) > 2 else 0.0
            gripper_action = action_squeezed[3].item() if len(action_squeezed) > 3 else 1.0  # 1 = stay
        elif isinstance(action, dict):
            delta_x = action.get("delta_x", 0.0)
            delta_y = action.get("delta_y", 0.0)
            delta_z = action.get("delta_z", 0.0)
            gripper_action = action.get("gripper", 1.0)
        else:
            return transition

        # Compute target position by applying delta to current position
        target_x = current_x + delta_x * self.ee_step_size
        target_y = current_y + delta_y * self.ee_step_size
        target_z = current_z + delta_z * self.ee_step_size

        # Clip to workspace bounds
        target_x = np.clip(target_x, self.ee_bounds["x_min"], self.ee_bounds["x_max"])
        target_y = np.clip(target_y, self.ee_bounds["y_min"], self.ee_bounds["y_max"])
        target_z = np.clip(target_z, self.ee_bounds["z_min"], self.ee_bounds["z_max"])

        # Handle gripper (0=close, 1=stay, 2=open)
        gripper_step = 5.0  # mm per step
        if gripper_action < 0.5:  # Close
            self.current_gripper = max(0.0, self.current_gripper - gripper_step)
        elif gripper_action > 1.5:  # Open
            self.current_gripper = min(self.max_gripper_pos, self.current_gripper + gripper_step)
        # else: stay

        # Log for debugging (only occasionally)
        if np.random.random() < 0.05:  # 5% of the time
            logging.debug(
                f"PiperEE: delta=({delta_x:.3f}, {delta_y:.3f}, {delta_z:.3f}) "
                f"current=({current_x:.3f}, {current_y:.3f}, {current_z:.3f}) "
                f"target=({target_x:.3f}, {target_y:.3f}, {target_z:.3f}) "
                f"gripper={self.current_gripper:.1f}mm"
            )

        # Create absolute position action for robot
        # Format: [ee.x, ee.y, ee.z, gripper.pos] in meters and mm
        absolute_action = torch.tensor([
            target_x,  # x in meters
            target_y,  # y in meters
            target_z,  # z in meters
            self.current_gripper,  # gripper in mm
        ], dtype=torch.float32)

        new_transition[TransitionKey.ACTION] = absolute_action

        return new_transition

    def reset(self) -> None:
        """Reset internal state for new episode."""
        self.current_gripper = 35.0

    def get_config(self) -> dict[str, Any]:
        """Return configuration for serialization."""
        return {
            "ee_step_size": self.ee_step_size,
            "max_gripper_pos": self.max_gripper_pos,
            "ee_bounds": self.ee_bounds,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Transform features - no changes needed."""
        return features
