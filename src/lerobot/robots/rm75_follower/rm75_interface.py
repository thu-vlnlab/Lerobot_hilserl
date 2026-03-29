"""Hardware interface for the RealMan RM75-B 7-DOF arm via TCP/IP,
with ZhiXing 90D gripper controlled via Modbus RTU over the arm's end-effector RS485.

Adapted from spacemouse2rm75b/rm75b.py.
"""

import logging
import math
import time
from typing import List, Union

import serial

import numpy as np

from Robotic_Arm.rm_robot_interface import (
    RoboticArm,
    rm_thread_mode_e,
    rm_peripheral_read_write_params_t,
)

logger = logging.getLogger(__name__)


class SuctionController:
    """USB relay controller for suction head (solenoid valve).

    Protocol: hex command  A0 CH OP CHECK
        CH    = channel number (1-indexed)
        OP    = 0x01 (relay ON / suction active) or 0x00 (relay OFF / release)
        CHECK = 0xA0 + CH + OP  (simple checksum)

    Compatible with common USB-UART relay boards (e.g., SainSmart).
    """

    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 9600, channel: int = 1):
        self.port = port
        self.baud = baud
        self.channel = channel
        self._serial: serial.Serial | None = None
        self._state: bool = False  # current suction state (True = ON)

    def connect(self) -> None:
        self._serial = serial.Serial(self.port, self.baud, timeout=0.5)
        time.sleep(0.1)  # allow serial hardware to stabilise
        logger.info(f"SuctionController connected on {self.port} ch={self.channel}")

    def set_state(self, on: bool) -> None:
        """Activate (on=True) or deactivate (on=False) the suction head."""
        if on == self._state:
            return  # no change — skip serial write to avoid relay clicking
        if self._serial is None or not self._serial.is_open:
            logger.warning("SuctionController: serial not open, skipping set_state")
            return
        op = 0x01 if on else 0x00
        ch = self.channel
        check = (0xA0 + ch + op) & 0xFF
        self._serial.write(bytes([0xA0, ch, op, check]))
        self._state = on
        print(f"[SUCTION] relay {'ON' if on else 'OFF'}")

    def get_state(self) -> bool:
        return self._state

    def disconnect(self) -> None:
        if self._serial and self._serial.is_open:
            try:
                self.set_state(False)   # safety: release on disconnect
            except Exception:
                pass
            self._serial.close()
        self._serial = None
        logger.info("SuctionController disconnected.")


DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi

# Modbus constants for ZhiXing 90D gripper
_MODBUS_PORT = 1            # End-effector RS485
_MODBUS_BAUDRATE = 115200
_MODBUS_TIMEOUT = 2         # unit: 100ms
_GRIPPER_DEVICE_ADDR = 1    # Modbus slave address
_GRIPPER_ENABLE_REG = 256   # Write 1 to enable
_GRIPPER_POS_HIGH_REG = 258 # 32-bit position high word
_GRIPPER_POS_LOW_REG = 259  # 32-bit position low word   (range 0~1000)
_GRIPPER_SPEED_REG = 260    # Speed 0~100%
_GRIPPER_TRIGGER_REG = 264  # Write 1 to trigger motion
_GRIPPER_FEEDBACK_REG = 258 # Read-back position register

_GRIPPER_CHANGE_THRESHOLD = 10   # position dead-band (out of 0~1000)
_GRIPPER_MIN_INTERVAL = 0.05     # seconds between Modbus writes


class RM75BInterface:
    """Hardware interface for the RealMan RM75-B 7-DOF arm via TCP/IP,
    with ZhiXing 90D gripper controlled via Modbus RTU over the arm's end-effector RS485.
    """

    def __init__(self, ip: str = "192.168.1.18", port: int = 8080, enable_gripper: bool = False):
        self.ip = ip
        self.port = port
        self.enable_gripper = enable_gripper
        self._last_gripper_pos = -9999  # force first write
        self._last_gripper_time = 0.0

        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle = self.arm.rm_create_robot_arm(ip, port)
        if handle.id == -1:
            raise RuntimeError(f"Failed to connect to RM75-B at {ip}:{port}")
        logger.info(f"Connected to RM75-B at {ip}:{port}  (handle id={handle.id})")

        if enable_gripper:
            self._init_gripper()

    def _init_gripper(self):
        """Configure Modbus RTU on end-effector RS485 and enable the ZhiXing 90D."""
        ret = self.arm.rm_set_modbus_mode(_MODBUS_PORT, _MODBUS_BAUDRATE, _MODBUS_TIMEOUT)
        if ret != 0:
            logger.warning(f"rm_set_modbus_mode returned {ret}")
        time.sleep(0.5)

        # Enable gripper: write 1 to register 256
        params = rm_peripheral_read_write_params_t(_GRIPPER_DEVICE_ADDR, _GRIPPER_ENABLE_REG, 1)
        ret = self.arm.rm_write_single_register(params, 1)
        if ret != 0:
            logger.warning(f"Gripper enable returned {ret}")
        time.sleep(1.0)
        logger.info("ZhiXing 90D gripper enabled via Modbus RTU.")

    def get_joint_positions(self) -> np.ndarray:
        """Returns current joint positions in radians. Shape: (7,)"""
        ret, degrees = self.arm.rm_get_joint_degree()
        if ret != 0:
            logger.warning(f"rm_get_joint_degree returned {ret}")
            return np.zeros(7)
        return np.array(degrees[:7]) * DEG_TO_RAD

    def get_current_pose(self) -> list[float]:
        """Return current end-effector pose as [x, y, z, rx, ry, rz].
        Position in meters, orientation in radians."""
        ret, pose = self.arm.rm_get_current_arm_state()
        if ret != 0:
            logger.warning(f"rm_get_current_arm_state returned {ret}")
            return [0.0] * 6
        # pose dict contains "pose": [x, y, z, rx, ry, rz] (meters + radians)
        pos = pose["pose"]
        return [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]]

    def get_full_state(self) -> tuple[np.ndarray, list[float]]:
        """Single SDK call returning (joints_rad, pose).
        Avoids separate rm_get_joint_degree + rm_get_current_arm_state calls.
        Returns: (joints shape (7,) in radians, [x,y,z,rx,ry,rz])
        """
        ret, state = self.arm.rm_get_current_arm_state()
        if ret != 0:
            logger.warning(f"rm_get_current_arm_state returned {ret}")
            return np.zeros(7), [0.0] * 6
        pose = state["pose"]
        # rm_get_current_arm_state also returns joint angles in degrees under "joint"
        joints_deg = state.get("joint", [0.0] * 7)
        joints_rad = np.array(joints_deg[:7]) * DEG_TO_RAD
        return joints_rad, [pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]]

    def movep_canfd(self, pose: list[float], follow: bool = True):
        """Send Cartesian pose command via rm_movep_canfd.
        Args:
            pose: [x, y, z, rx, ry, rz] — meters + radians
            follow: True for follow mode (smoother tracking)
        """
        ret = self.arm.rm_movep_canfd(pose, follow)
        if ret != 0:
            logger.warning(f"rm_movep_canfd returned {ret}")

    def set_joint_positions(self, positions: Union[List[float], np.ndarray]):
        """Set joint positions in radians. Shape: (7,)
        Uses rm_movej_canfd with follow=False (low-following mode) for streaming."""
        degrees = [float(p * RAD_TO_DEG) for p in positions]
        ret = self.arm.rm_movej_canfd(degrees, follow=False)
        if ret != 0:
            logger.warning(f"rm_movej_canfd returned {ret}")

    def _write_gripper_reg(self, addr, value):
        """Write a single 16-bit Modbus register on the gripper."""
        params = rm_peripheral_read_write_params_t(_GRIPPER_DEVICE_ADDR, addr, 1)
        self.arm.rm_write_single_register(params, value)

    def get_gripper_position(self) -> float:
        """Read current gripper position (0~1000) from Modbus feedback register."""
        if not self.enable_gripper:
            return 0.0
        params = rm_peripheral_read_write_params_t(_GRIPPER_DEVICE_ADDR, _GRIPPER_FEEDBACK_REG, 1)
        ret, value = self.arm.rm_read_holding_registers(params)
        if ret != 0:
            logger.warning(f"rm_read_holding_registers returned {ret}")
            return 0.0
        return float(value)

    def set_gripper_position(self, position: float):
        """Set gripper position (0~1000, 0=closed, 1000=open).
        Writes 32-bit position as two separate 16-bit registers (high + low word),
        then triggers motion. Rate-limited to avoid flooding Modbus."""
        if not self.enable_gripper:
            return

        pos_int = int(max(0, min(1000, round(position))))

        # Skip if position hasn't changed enough
        if abs(pos_int - self._last_gripper_pos) < _GRIPPER_CHANGE_THRESHOLD:
            return

        # Skip if not enough time since last write
        now = time.time()
        if now - self._last_gripper_time < _GRIPPER_MIN_INTERVAL:
            return

        # Write 32-bit position as two 16-bit registers
        self._write_gripper_reg(_GRIPPER_POS_HIGH_REG, (pos_int >> 16) & 0xFFFF)
        self._write_gripper_reg(_GRIPPER_POS_LOW_REG, pos_int & 0xFFFF)
        # Trigger motion
        self._write_gripper_reg(_GRIPPER_TRIGGER_REG, 1)

        self._last_gripper_pos = pos_int
        self._last_gripper_time = now

    def go_home(self, joints_deg=None, speed=20):
        """Move to home position at given speed, blocking.
        Args:
            joints_deg: target joint angles in degrees. None = all zeros.
            speed: movement speed percentage (0~100).
        """
        if joints_deg is None:
            joints_deg = [0.0] * 7
        ret = self.arm.rm_movej(list(joints_deg), v=speed, r=0, connect=0, block=1)
        if ret != 0:
            logger.warning(f"rm_movej go_home returned {ret}")

    def close(self):
        """Close Modbus port and disconnect from arm."""
        if self.enable_gripper:
            try:
                self.arm.rm_close_modbus_mode(_MODBUS_PORT)
            except Exception as e:
                logger.warning(f"rm_close_modbus_mode: {e}")
        try:
            self.arm.rm_delete_robot_arm()
        except Exception as e:
            logger.warning(f"rm_delete_robot_arm: {e}")
        logger.info(f"RM75-B at {self.ip}:{self.port} disconnected.")
