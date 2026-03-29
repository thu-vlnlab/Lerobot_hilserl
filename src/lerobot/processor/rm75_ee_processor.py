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
End-effector processor for RM75-B with suction head.

Scaling chain (single source of control speed):
    SpaceMouse raw axis  (±350 full deflection)
        × translation_scale          ← set in SpaceMouseTeleopConfig
        = delta  (meters/frame)      ← already in physical units
        × ee_step_size = 1.0         ← no-op; kept for interface compatibility
        → accumulated to target_pose

To change speed: only adjust translation_scale.
    translation_scale = desired_max_speed_m_s / 350 / control_fps
    Example: 0.05 m/s max @ 100 Hz → 0.05/350/100 ≈ 0.0000143

Action recorded to dataset:
    action = target_pose_xyz - obs_ee_xyz_at_record_time
    This represents the actual commanded displacement in robot base frame (meters),
    which is the natural action space for imitation learning policies.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature

from .core import EnvTransition, TransitionKey
from .pipeline import ProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register("rm75_delta_to_absolute_ee")
@dataclass
class RM75DeltaToAbsoluteEEStep(ProcessorStep):
    """
    Converts SpaceMouse delta actions to absolute EE commands for RM75-B.

    Control path:
        - Initialises internal target from first observation (no read-back afterwards)
        - Accumulates delta * ee_step_size onto target each step
        - Clips target to workspace bounds
        - Stores target in transition for the env to execute

    Recording path (computed externally in control loop):
        action_recorded = target_xyz - obs_ee_xyz  (meters, robot base frame)

    Suction (optional):
        Binary state (0.0=off, 1.0=on) passed through directly from teleop.

    Attributes:
        ee_step_size: Multiplier on delta before accumulation.
            Set to 1.0 when teleop already outputs metric deltas (SpaceMouse
            with translation_scale).  Only change if using a different teleop.
        use_suction: Include suction.state in output action tensor.
        ee_bounds: Workspace limits in meters.
    """

    ee_step_size: float = 1.0   # SpaceMouse translation_scale already in meters/frame
    use_suction: bool = False

    ee_bounds: dict = field(default_factory=lambda: {
        "x_min": -0.5, "x_max": 0.5,
        "y_min": -0.5, "y_max": 0.5,
        "z_min": 0.0,  "z_max": 0.7,
    })

    # Internal state — initialised on first observation
    _target_x: float = 0.0
    _target_y: float = 0.0
    _target_z: float = 0.0
    _target_initialised: bool = False
    _suction_state: float = 0.0

    def _init_from_obs(self, obs: dict) -> None:
        """Read current EE pose once at start; accumulate deltas from here on."""
        if not isinstance(obs, dict) or "ee.x" not in obs:
            return

        def _v(v):
            return float(v.item() if hasattr(v, "item") else v)

        self._target_x = _v(obs["ee.x"])
        self._target_y = _v(obs["ee.y"])
        self._target_z = _v(obs["ee.z"])
        if "suction.state" in obs:
            self._suction_state = _v(obs["suction.state"])
        self._target_initialised = True
        logging.info(
            f"RM75EE target initialised: "
            f"x={self._target_x:.4f} y={self._target_y:.4f} z={self._target_z:.4f}"
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition

        new_transition = transition.copy()

        # One-time init from first observation
        if not self._target_initialised:
            obs = transition.get(TransitionKey.OBSERVATION, {})
            self._init_from_obs(obs)

        # Extract delta values from action tensor or dict
        if isinstance(action, torch.Tensor):
            a = action.squeeze()
            if a.dim() == 0:
                a = a.unsqueeze(0)
            delta_x = a[0].item() if len(a) > 0 else 0.0
            delta_y = a[1].item() if len(a) > 1 else 0.0
            delta_z = a[2].item() if len(a) > 2 else 0.0
            suction  = a[3].item() if (self.use_suction and len(a) > 3) else self._suction_state
        elif isinstance(action, dict):
            delta_x = action.get("delta_x", 0.0)
            delta_y = action.get("delta_y", 0.0)
            delta_z = action.get("delta_z", 0.0)
            suction  = action.get("suction.state", self._suction_state)
        else:
            return transition

        # Accumulate delta onto internal target (no read-back — avoids oscillation)
        self._target_x += delta_x * self.ee_step_size
        self._target_y += delta_y * self.ee_step_size
        self._target_z += delta_z * self.ee_step_size
        self._suction_state = suction

        # Clip to workspace bounds
        self._target_x = np.clip(self._target_x, self.ee_bounds["x_min"], self.ee_bounds["x_max"])
        self._target_y = np.clip(self._target_y, self.ee_bounds["y_min"], self.ee_bounds["y_max"])
        self._target_z = np.clip(self._target_z, self.ee_bounds["z_min"], self.ee_bounds["z_max"])

        # Build absolute target action for robot
        if self.use_suction:
            absolute_action = torch.tensor(
                [self._target_x, self._target_y, self._target_z, self._suction_state],
                dtype=torch.float32,
            )
        else:
            absolute_action = torch.tensor(
                [self._target_x, self._target_y, self._target_z],
                dtype=torch.float32,
            )

        new_transition[TransitionKey.ACTION] = absolute_action

        # Store current target for external record-time delta computation
        new_transition["rm75_target_xyz"] = (self._target_x, self._target_y, self._target_z)

        return new_transition

    def get_target_xyz(self) -> tuple[float, float, float]:
        """Return current internal target for computing recorded action delta."""
        return (self._target_x, self._target_y, self._target_z)

    def reset(self) -> None:
        """Reset internal state for new episode (re-init from obs on next call)."""
        self._target_initialised = False
        self._suction_state = 0.0

    def get_config(self) -> dict[str, Any]:
        return {
            "ee_step_size": self.ee_step_size,
            "use_suction": self.use_suction,
            "ee_bounds": self.ee_bounds,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
