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
import time
import sys
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_piper_follower import PiperFollowerConfig, PiperFollowerEndEffectorConfig

# Add piper_sdk to path - adjust this path based on your installation
# You may need to: pip install -e /path/to/piper_sdk
try:
    from piper_sdk import C_PiperInterface_V2
except ImportError:
    # Try adding the local path
    sys.path.insert(0, "/home/qzl/data/piper/piper_sdk")
    from piper_sdk import C_PiperInterface_V2

logger = logging.getLogger(__name__)


class PiperFollower(Robot):
    """
    Piper Follower Arm - A 6-DOF robot arm with gripper controlled via CAN bus.

    This class provides a LeRobot-compatible interface for the Piper robot arm,
    supporting both joint-space control and observation.
    """

    config_class = PiperFollowerConfig
    name = "piper_follower"

    # Joint names for the 6-DOF arm
    JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

    def __init__(self, config: PiperFollowerConfig):
        super().__init__(config)
        self.config = config

        # Initialize Piper interface (will connect later)
        self._piper: C_PiperInterface_V2 | None = None
        self._connected = False

        # Initialize cameras
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor features: 6 joints + gripper."""
        features = {f"{joint}.pos": float for joint in self.JOINT_NAMES}
        features["gripper.pos"] = float
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera features."""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Observation space: joint positions + gripper + camera images."""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Action space: joint positions + gripper."""
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """Check if robot and cameras are connected."""
        return self._connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to the Piper robot arm via CAN bus.

        Args:
            calibrate: If True, run calibration if needed (not typically needed for Piper).
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info(f"Connecting to Piper on {self.config.can_name}...")

        # Initialize Piper interface
        self._piper = C_PiperInterface_V2(
            can_name=self.config.can_name,
            judge_flag=self.config.judge_flag,
            can_auto_init=self.config.can_auto_init,
        )

        # Connect to the robot
        self._piper.ConnectPort()

        # Wait for connection to stabilize
        time.sleep(0.5)

        # Enable the robot arm
        self._enable_arm()

        self._connected = True

        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()

        logger.info(f"{self} connected.")

    def _enable_arm(self) -> None:
        """Enable the robot arm motors."""
        if self._piper is None:
            return

        # Enable arm (motion control mode 1 = joint control)
        self._piper.EnableArm(7)  # Enable all motors
        self._piper.GripperCtrl(gripper_angle=0, gripper_effort=self.config.gripper_effort, gripper_code=0x01)

        # Wait for arm to be ready
        time.sleep(0.1)

        # Set to position control mode
        self._piper.MotionCtrl_2(0x01, 0x01, 50)  # Joint control mode, position mode

        time.sleep(0.1)
        logger.info("Piper arm enabled and ready")

    def _disable_arm(self) -> None:
        """Disable the robot arm motors."""
        if self._piper is None:
            return

        self._piper.DisableArm(7)  # Disable all motors
        self._piper.GripperCtrl(gripper_angle=0, gripper_effort=0, gripper_code=0x00)
        logger.info("Piper arm disabled")

    @property
    def is_calibrated(self) -> bool:
        """Piper doesn't require software calibration."""
        return True

    def calibrate(self) -> None:
        """
        Calibration for Piper is typically done at the hardware level.
        This is a no-op for LeRobot compatibility.
        """
        logger.info("Piper calibration is handled at hardware level. Skipping.")

    def configure(self) -> None:
        """Configure the robot after connection."""
        pass

    def get_observation(self) -> dict[str, Any]:
        """
        Get current observation from the robot.

        Returns:
            Dictionary containing joint positions, gripper position, and camera images.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = {}

        # Read joint states
        start = time.perf_counter()
        joint_msgs = self._piper.GetArmJointMsgs()
        joint_state = joint_msgs.joint_state

        # Convert from 0.001 degrees to degrees
        joints_deg = [
            joint_state.joint_1 / 1000.0,
            joint_state.joint_2 / 1000.0,
            joint_state.joint_3 / 1000.0,
            joint_state.joint_4 / 1000.0,
            joint_state.joint_5 / 1000.0,
            joint_state.joint_6 / 1000.0,
        ]

        if self.config.use_degrees:
            for i, joint_name in enumerate(self.JOINT_NAMES):
                obs_dict[f"{joint_name}.pos"] = joints_deg[i]
        else:
            # Normalize to -100 to 100 range based on joint limits
            for i, joint_name in enumerate(self.JOINT_NAMES):
                limits = self.config.joint_limits[joint_name]
                normalized = 200.0 * (joints_deg[i] - limits[0]) / (limits[1] - limits[0]) - 100.0
                obs_dict[f"{joint_name}.pos"] = normalized

        # Read gripper state
        gripper_msgs = self._piper.GetArmGripperMsgs()
        gripper_mm = gripper_msgs.gripper_state.grippers_angle / 1000.0  # Convert from 0.001mm to mm

        if self.config.use_degrees:
            obs_dict["gripper.pos"] = gripper_mm
        else:
            # Normalize to 0-100 range
            obs_dict["gripper.pos"] = 100.0 * gripper_mm / self.config.gripper_max_mm

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Send action commands to the robot.

        Args:
            action: Dictionary containing target joint positions and gripper position.

        Returns:
            The action actually sent (may be clipped for safety).
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Extract joint goals
        goal_joints = {}
        for joint_name in self.JOINT_NAMES:
            key = f"{joint_name}.pos"
            if key in action:
                goal_joints[joint_name] = action[key]

        # Extract gripper goal
        gripper_goal = action.get("gripper.pos", None)

        # Apply safety limits if configured
        if self.config.max_relative_target is not None:
            current_obs = self.get_observation()
            goal_present = {
                key: (goal_joints.get(key.replace(".pos", ""), current_obs[key]), current_obs[key])
                for key in current_obs if key.endswith(".pos") and "gripper" not in key
            }
            safe_goals = ensure_safe_goal_position(goal_present, self.config.max_relative_target)
            for key, val in safe_goals.items():
                joint_name = key.replace(".pos", "")
                if joint_name in goal_joints:
                    goal_joints[joint_name] = val

        # Convert to Piper format (0.001 degrees)
        if self.config.use_degrees:
            joints_milli = [int(goal_joints.get(name, 0) * 1000) for name in self.JOINT_NAMES]
        else:
            # Convert from normalized (-100 to 100) to degrees
            joints_milli = []
            for i, joint_name in enumerate(self.JOINT_NAMES):
                limits = self.config.joint_limits[joint_name]
                normalized = goal_joints.get(joint_name, 0)
                deg = limits[0] + (normalized + 100.0) / 200.0 * (limits[1] - limits[0])
                joints_milli.append(int(deg * 1000))

        # Send joint command
        self._piper.JointCtrl(
            joint_1=joints_milli[0],
            joint_2=joints_milli[1],
            joint_3=joints_milli[2],
            joint_4=joints_milli[3],
            joint_5=joints_milli[4],
            joint_6=joints_milli[5],
        )

        # Send gripper command if provided
        if gripper_goal is not None:
            if self.config.use_degrees:
                gripper_milli = int(gripper_goal * 1000)  # mm to 0.001mm
            else:
                # Convert from 0-100 to mm
                gripper_mm = gripper_goal / 100.0 * self.config.gripper_max_mm
                gripper_milli = int(gripper_mm * 1000)

            self._piper.GripperCtrl(
                gripper_angle=gripper_milli,
                gripper_effort=self.config.gripper_effort,
                gripper_code=0x01,
            )

        # Return the action that was sent
        sent_action = {f"{name}.pos": goal_joints.get(name, 0) for name in self.JOINT_NAMES}
        if gripper_goal is not None:
            sent_action["gripper.pos"] = gripper_goal

        return sent_action

    def disconnect(self) -> None:
        """Disconnect from the robot and cameras."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Disable arm if configured
        if self.config.disable_torque_on_disconnect:
            self._disable_arm()

        # Disconnect from Piper
        if self._piper is not None:
            self._piper.DisconnectPort()
            self._piper = None

        self._connected = False

        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")


class PiperFollowerEndEffector(PiperFollower):
    """
    Piper Follower Arm with end-effector space control.

    Used for HIL-SERL training where actions are in end-effector space.
    Inherits from PiperFollower and adds end-effector specific functionality.
    """

    config_class = PiperFollowerEndEffectorConfig
    name = "piper_follower_ee"

    def __init__(self, config: PiperFollowerEndEffectorConfig):
        super().__init__(config)
        self.config = config

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Observation includes end-effector pose in addition to joint positions."""
        base_features = super().observation_features
        # Add end-effector pose (x, y, z, rx, ry, rz)
        ee_features = {
            "ee.x": float,
            "ee.y": float,
            "ee.z": float,
            "ee.rx": float,
            "ee.ry": float,
            "ee.rz": float,
        }
        return {**base_features, **ee_features}

    def get_observation(self) -> dict[str, Any]:
        """Get observation including end-effector pose."""
        obs_dict = super().get_observation()

        # Get forward kinematics (end-effector pose)
        if self._piper is not None:
            fk = self._piper.GetFK(mode="feedback")
            # FK returns [X, Y, Z, RX, RY, RZ] in mm and degrees
            if fk and len(fk) >= 6:
                obs_dict["ee.x"] = fk[0][0] / 1000.0  # mm to m
                obs_dict["ee.y"] = fk[0][1] / 1000.0
                obs_dict["ee.z"] = fk[0][2] / 1000.0
                obs_dict["ee.rx"] = fk[0][3]  # degrees
                obs_dict["ee.ry"] = fk[0][4]
                obs_dict["ee.rz"] = fk[0][5]

        return obs_dict

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Action space for end-effector control: x, y, z + gripper."""
        return {
            "ee.x": float,
            "ee.y": float,
            "ee.z": float,
            "gripper.pos": float,
        }

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Send end-effector pose control command.

        Args:
            action: Dictionary containing target ee position (x, y, z) and gripper.
                   x, y, z in meters, gripper in mm.

        Returns:
            The action actually sent.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Get current end-effector pose to fill in missing values
        current_obs = self.get_observation()

        # Extract target position (convert meters to mm for SDK)
        target_x = action.get("ee.x", current_obs.get("ee.x", 0.3)) * 1000  # m to mm
        target_y = action.get("ee.y", current_obs.get("ee.y", 0.0)) * 1000
        target_z = action.get("ee.z", current_obs.get("ee.z", 0.2)) * 1000

        # Keep current orientation (or use provided)
        target_rx = action.get("ee.rx", current_obs.get("ee.rx", 0.0))
        target_ry = action.get("ee.ry", current_obs.get("ee.ry", 0.0))
        target_rz = action.get("ee.rz", current_obs.get("ee.rz", 0.0))

        # Convert to SDK units (0.001 mm and 0.001 degrees)
        X = int(target_x * 1000)
        Y = int(target_y * 1000)
        Z = int(target_z * 1000)
        RX = int(target_rx * 1000)
        RY = int(target_ry * 1000)
        RZ = int(target_rz * 1000)

        # Set motion mode to end-effector control
        self._piper.MotionCtrl_2(0x01, 0x00, 50, 0x00)

        # Send end-effector pose command
        self._piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)

        # Handle gripper if provided
        gripper_goal = action.get("gripper.pos", None)
        if gripper_goal is not None:
            gripper_mm = gripper_goal
            if not self.config.use_degrees:
                gripper_mm = gripper_goal / 100.0 * self.config.gripper_max_mm
            gripper_milli = int(gripper_mm * 1000)
            self._piper.GripperCtrl(
                gripper_angle=gripper_milli,
                gripper_effort=self.config.gripper_effort,
                gripper_code=0x01,
            )

        return action
