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

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("spacemouse")
@dataclass
class SpaceMouseTeleopConfig(TeleoperatorConfig):
    """Configuration for SpaceMouse 6DOF teleoperator.

    Uses libspnav to read from a 3Dconnexion SpaceMouse device.
    Outputs delta_x, delta_y, delta_z translations + gripper from buttons.
    """

    # Path to libspnav shared library (None = auto-detect)
    libspnav_path: str | None = None

    # Raw axis dead-zone (full range ~+-350)
    deadzone: int = 40

    # Enable gripper control via buttons
    use_gripper: bool = True

    # Axis remapping: SpaceMouse raw [0..5] -> robot [x, y, z, rx, ry, rz]
    axis_map: list[int] = field(default_factory=lambda: [2, 0, 1, 5, 3, 4])

    # Axis sign flips after remapping
    axis_signs: list[int] = field(default_factory=lambda: [1, -1, 1, -1, 1, 1])

    # Translation scale: raw axis value * scale -> normalized delta
    translation_scale: float = 0.0032

    # Button assignments for gripper
    gripper_close_button: int = 0
    gripper_open_button: int = 1

    # Suction head toggle (single button, press = flip on↔off)
    use_suction: bool = False
    suction_toggle_button: int = 0
