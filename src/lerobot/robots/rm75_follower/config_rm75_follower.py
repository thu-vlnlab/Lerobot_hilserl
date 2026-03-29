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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("rm75_follower")
@dataclass
class RM75FollowerConfig(RobotConfig):
    """Configuration for the RealMan RM75-B 7-DOF robot arm.

    RM75-B is a 7-DOF robot arm controlled via TCP/IP,
    with a ZhiXing 90D gripper controlled via Modbus RTU.
    """

    # Robot TCP connection
    robot_ip: str = "192.168.1.18"
    robot_port: int = 8080

    # Gripper
    enable_gripper: bool = True
    gripper_open_pos: int = 1000    # ZhiXing 90D range 0~1000
    gripper_close_pos: int = 0

    # Camera configurations
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Joint limits in degrees (7 joints)
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "joint_1": (-178.0, 178.0),
        "joint_2": (-130.0, 130.0),
        "joint_3": (-135.0, 135.0),
        "joint_4": (-178.0, 178.0),
        "joint_5": (-128.0, 128.0),
        "joint_6": (-178.0, 178.0),
        "joint_7": (-360.0, 360.0),
    })

    # Home / reset joint positions in degrees
    home_joint_positions: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    # Reset speed (percentage, 0~100)
    reset_speed: int = 20

    # Safety: slow-stop on disconnect
    slow_stop_on_disconnect: bool = True


@RobotConfig.register_subclass("rm75_follower_ee")
@dataclass
class RM75FollowerEndEffectorConfig(RM75FollowerConfig):
    """Configuration for RM75-B in end-effector space control.

    Used for HIL-SERL training where actions are in end-effector space (x, y, z + gripper).
    Orientation is locked to the initial pose.
    """

    # Workspace bounds in meters [x, y, z]
    workspace_bounds: dict[str, list[float]] = field(
        default_factory=lambda: {
            "min": [-0.5, -0.5, 0.0],
            "max": [0.5, 0.5, 0.7],
        }
    )

    # Maximum translation per control cycle (meters)
    max_translation_per_cycle: float = 0.001

    # Lock orientation to initial pose (first-call lock)
    lock_orientation: bool = True

    # Suction head (USB relay controller)
    enable_suction: bool = False
    suction_port: str = "/dev/ttyUSB0"
    suction_baud_rate: int = 9600
    suction_channel: int = 1
