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

import logging
import math
import time
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_rm75_follower import RM75FollowerConfig, RM75FollowerEndEffectorConfig
from .rm75_interface import RM75BInterface

logger = logging.getLogger(__name__)

RAD_TO_DEG = 180.0 / math.pi
DEG_TO_RAD = math.pi / 180.0


class RM75Follower(Robot):
    """
    RealMan RM75-B 7-DOF robot arm with ZhiXing 90D gripper.

    Joint-space control mode: actions are 7 joint positions (degrees) + gripper (0~1000).
    """

    config_class = RM75FollowerConfig
    name = "rm75_follower"

    JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]

    def __init__(self, config: RM75FollowerConfig):
        super().__init__(config)
        self.config = config
        self._arm: RM75BInterface | None = None
        self._connected = False
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        features = {f"{joint}.pos": float for joint in self.JOINT_NAMES}
        features["gripper.pos"] = float
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info(f"Connecting to RM75-B at {self.config.robot_ip}:{self.config.robot_port}...")

        self._arm = RM75BInterface(
            ip=self.config.robot_ip,
            port=self.config.robot_port,
            enable_gripper=self.config.enable_gripper,
        )

        self._connected = True

        for cam in self.cameras.values():
            cam.connect()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.info("RM75-B calibration is handled at hardware level. Skipping.")

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = {}

        # Read 7 joint positions in radians, convert to degrees
        joints_rad = self._arm.get_joint_positions()
        joints_deg = joints_rad * RAD_TO_DEG
        for i, joint_name in enumerate(self.JOINT_NAMES):
            obs_dict[f"{joint_name}.pos"] = float(joints_deg[i])

        # Read gripper position (0~1000)
        obs_dict["gripper.pos"] = self._arm.get_gripper_position()

        # Capture images
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Extract joint goals in degrees, convert to radians for SDK
        goal_joints_rad = []
        for joint_name in self.JOINT_NAMES:
            key = f"{joint_name}.pos"
            deg = action.get(key, 0.0)
            goal_joints_rad.append(deg * DEG_TO_RAD)

        self._arm.set_joint_positions(goal_joints_rad)

        # Gripper
        gripper_goal = action.get("gripper.pos", None)
        if gripper_goal is not None:
            self._arm.set_gripper_position(float(gripper_goal))

        return action

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self._arm is not None:
            self._arm.close()
            self._arm = None

        self._connected = False

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")


class RM75FollowerEndEffector(RM75Follower):
    """
    RM75-B with end-effector space control.

    Actions: ee.x, ee.y, ee.z (meters) + gripper.pos (0~1000).
    Orientation is locked on first call to send_action().
    Observations include both joint positions and end-effector pose.
    """

    config_class = RM75FollowerEndEffectorConfig
    name = "rm75_follower_ee"

    def __init__(self, config: RM75FollowerEndEffectorConfig):
        super().__init__(config)
        self.config = config
        self._fixed_orientation: list[float] | None = None

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        base_features = super().observation_features
        ee_features = {
            "ee.x": float,
            "ee.y": float,
            "ee.z": float,
            "ee.rx": float,
            "ee.ry": float,
            "ee.rz": float,
        }
        return {**base_features, **ee_features}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            "ee.x": float,
            "ee.y": float,
            "ee.z": float,
            "gripper.pos": float,
        }

    def get_observation(self) -> dict[str, Any]:
        obs_dict = super().get_observation()

        if self._arm is not None:
            pose = self._arm.get_current_pose()  # [x, y, z, rx, ry, rz] in meters/radians
            obs_dict["ee.x"] = pose[0]
            obs_dict["ee.y"] = pose[1]
            obs_dict["ee.z"] = pose[2]
            obs_dict["ee.rx"] = pose[3]
            obs_dict["ee.ry"] = pose[4]
            obs_dict["ee.rz"] = pose[5]

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Lock orientation on first call
        if self._fixed_orientation is None:
            pose = self._arm.get_current_pose()
            self._fixed_orientation = [pose[3], pose[4], pose[5]]
            logger.info(
                f"Fixed orientation locked: rx={pose[3]:.4f}, ry={pose[4]:.4f}, rz={pose[5]:.4f}"
            )

        # Get current pose for defaults
        current_pose = self._arm.get_current_pose()

        # Target position from action (meters)
        target_x = action.get("ee.x", current_pose[0])
        target_y = action.get("ee.y", current_pose[1])
        target_z = action.get("ee.z", current_pose[2])

        # Clip to workspace bounds
        wb = self.config.workspace_bounds
        target_x = max(wb["min"][0], min(wb["max"][0], target_x))
        target_y = max(wb["min"][1], min(wb["max"][1], target_y))
        target_z = max(wb["min"][2], min(wb["max"][2], target_z))

        # Use locked orientation
        rx, ry, rz = self._fixed_orientation

        # Build target pose and send
        target_pose = [target_x, target_y, target_z, rx, ry, rz]
        self._arm.movep_canfd(target_pose, follow=True)

        # Gripper
        gripper_goal = action.get("gripper.pos", None)
        if gripper_goal is not None:
            self._arm.set_gripper_position(float(gripper_goal))

        return action

    def disconnect(self) -> None:
        self._fixed_orientation = None
        super().disconnect()
