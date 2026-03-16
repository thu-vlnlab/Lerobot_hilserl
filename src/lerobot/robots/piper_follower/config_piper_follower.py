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


@RobotConfig.register_subclass("piper_follower")
@dataclass
class PiperFollowerConfig(RobotConfig):
    """Configuration for the Piper Follower robot arm.

    Piper is a 6-DOF robot arm with a gripper, controlled via CAN bus.
    """

    # CAN port name (e.g., "can0", "can1")
    can_name: str = "can0"

    # Whether to check CAN port status
    judge_flag: bool = True

    # Whether to auto-initialize CAN on startup
    can_auto_init: bool = True

    # Disable torque when disconnecting (False = keep holding position, safer)
    disable_torque_on_disconnect: bool = False

    # Use degrees for joint positions (if False, uses normalized range -100 to 100)
    use_degrees: bool = True

    # Maximum relative target for safety (in degrees or normalized units)
    max_relative_target: float | dict[str, float] | None = None

    # Camera configurations
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Joint limits in degrees
    # |joint_name|     limit(rad)       |    limit(angle)    |
    # |joint1    |   [-2.6179, 2.6179]  |    [-150.0, 150.0] |
    # |joint2    |   [0, 3.14]          |    [0, 180.0]      |
    # |joint3    |   [-2.967, 0]        |    [-170, 0]       |
    # |joint4    |   [-1.745, 1.745]    |    [-100.0, 100.0] |
    # |joint5    |   [-1.22, 1.22]      |    [-70.0, 70.0]   |
    # |joint6    |   [-2.09439, 2.09439]|    [-120.0, 120.0] |
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "joint_1": (-150.0, 150.0),
        "joint_2": (0.0, 180.0),
        "joint_3": (-170.0, 0.0),
        "joint_4": (-100.0, 100.0),
        "joint_5": (-70.0, 70.0),
        "joint_6": (-120.0, 120.0),
    })

    # Gripper range in mm (0 = closed, ~70mm = fully open for standard gripper)
    gripper_max_mm: float = 70.0

    # Default gripper effort (0-5000 corresponding to 0-5 N/m)
    gripper_effort: int = 1000


@RobotConfig.register_subclass("piper_follower_ee")
@dataclass
class PiperFollowerEndEffectorConfig(PiperFollowerConfig):
    """Configuration for the Piper Follower robot arm in end-effector space.

    Used for HIL-SERL training where actions are in end-effector space.
    """

    # Default bounds for the end-effector position (in meters)
    end_effector_bounds: dict[str, list[float]] = field(
        default_factory=lambda: {
            "min": [-0.5, -0.5, 0.0],   # min x, y, z
            "max": [0.5, 0.5, 0.5],     # max x, y, z
        }
    )

    # Maximum gripper position (0-100%)
    max_gripper_pos: float = 100.0

    # Maximum step size for the end-effector in x, y, z direction (meters)
    end_effector_step_sizes: dict[str, float] = field(
        default_factory=lambda: {
            "x": 0.02,
            "y": 0.02,
            "z": 0.02,
        }
    )
