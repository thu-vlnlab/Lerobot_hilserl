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
import select
import sys
import termios
import tty
from enum import IntEnum
from typing import Any

import numpy as np

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_spacemouse import SpaceMouseTeleopConfig
from .spacemouse_reader import SpaceMouseReader

logger = logging.getLogger(__name__)


class GripperAction(IntEnum):
    CLOSE = 0
    STAY = 1
    OPEN = 2


class SpaceMouseTeleop(Teleoperator):
    """
    Teleoperator using a 3Dconnexion SpaceMouse for 6DOF input.

    Outputs delta_x, delta_y, delta_z translations and gripper commands,
    matching the same format as GamepadTeleop for compatibility with HIL-SERL.
    """

    config_class = SpaceMouseTeleopConfig
    name = "spacemouse"

    def __init__(self, config: SpaceMouseTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type
        self.reader: SpaceMouseReader | None = None
        self._old_term_settings = None
        self._suction_state: bool = False
        self._suction_btn_prev: bool = False  # for edge detection

    @property
    def action_features(self) -> dict:
        if self.config.use_suction:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "suction.state": 3},
            }
        elif self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (3,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
            }

    @property
    def feedback_features(self) -> dict:
        return {}

    def connect(self) -> None:
        self.reader = SpaceMouseReader(
            lib_path=self.config.libspnav_path,
            deadzone=self.config.deadzone,
        )
        self.reader.open()
        self.reader.start()

        # Set terminal to raw mode for non-blocking keyboard read
        try:
            self._old_term_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except Exception:
            self._old_term_settings = None

        logger.info("SpaceMouseTeleop connected.")
        logger.info("Keyboard: Enter=成功结束  Backspace=重录  q=退出")

    def get_action(self) -> dict[str, Any]:
        raw_axes = self.reader.get_axes()  # [x, y, z, rx, ry, rz] raw with deadzone

        # Remap axes: pick translation axes (first 3 after remapping)
        axis_map = self.config.axis_map
        axis_signs = self.config.axis_signs

        # Apply axis_map to get reordered axes, then apply signs
        remapped = [raw_axes[axis_map[i]] * axis_signs[i] for i in range(6)]

        # Extract translation (first 3 remapped axes) and scale
        scale = self.config.translation_scale
        delta_x = remapped[0] * scale
        delta_y = remapped[1] * scale
        delta_z = remapped[2] * scale

        action_dict = {
            "delta_x": np.float32(delta_x),
            "delta_y": np.float32(delta_y),
            "delta_z": np.float32(delta_z),
        }

        if self.config.use_suction:
            # Edge-triggered toggle: button press (off→on edge) flips suction state
            btn_cur = bool(self.reader.get_button(self.config.suction_toggle_button))
            if btn_cur and not self._suction_btn_prev:
                self._suction_state = not self._suction_state
                logger.info(f"Suction toggled: {'ON' if self._suction_state else 'OFF'}")
            self._suction_btn_prev = btn_cur
            action_dict["suction.state"] = np.float32(1.0 if self._suction_state else 0.0)
        elif self.config.use_gripper:
            # Button 0 = close, Button 1 = open, neither = stay
            btn_close = self.reader.get_button(self.config.gripper_close_button)
            btn_open = self.reader.get_button(self.config.gripper_open_button)

            if btn_close:
                gripper_action = GripperAction.CLOSE.value
            elif btn_open:
                gripper_action = GripperAction.OPEN.value
            else:
                gripper_action = GripperAction.STAY.value

            action_dict["gripper"] = gripper_action

        return action_dict

    def get_teleop_events(self) -> dict[str, Any]:
        if self.reader is None:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        # SpaceMouse self-centers: any non-zero axis = intervention active
        axes = self.reader.get_axes()
        is_intervention = any(a != 0 for a in axes[:3])  # Only check translation axes

        # Non-blocking keyboard check for episode events
        terminate = False
        success = False
        rerecord = False
        key = self._read_key()
        if key == "\n" or key == "\r":  # Enter = success
            success = True
            terminate = True
            logger.info("Keyboard: Enter pressed → episode SUCCESS")
        elif key == "\x7f" or key == "\x08":  # Backspace = rerecord
            rerecord = True
            terminate = True
            logger.info("Keyboard: Backspace pressed → RERECORD episode")
        elif key == "q":  # q = terminate (no success)
            terminate = True
            logger.info("Keyboard: q pressed → TERMINATE episode")

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord,
        }

    def _read_key(self) -> str | None:
        """Non-blocking single key read from stdin."""
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1)
        except Exception:
            pass
        return None

    def disconnect(self) -> None:
        # Restore terminal settings
        if self._old_term_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_term_settings)
            except Exception:
                pass
            self._old_term_settings = None

        if self.reader is not None:
            self.reader.stop()
            self.reader = None
        logger.info("SpaceMouseTeleop disconnected.")

    def is_connected(self) -> bool:
        return self.reader is not None

    def calibrate(self) -> None:
        pass

    def is_calibrated(self) -> bool:
        return True

    def configure(self) -> None:
        pass

    def send_feedback(self, feedback: dict) -> None:
        pass
