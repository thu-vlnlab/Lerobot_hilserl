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
Piper Follower 机器人测试

包含两类测试：
1. Mock 测试 - 不需要真实硬件，用于 CI/CD
2. 硬件测试 - 需要连接真实 Piper 机器人

运行方式：
    # 运行所有 mock 测试
    pytest tests/robots/test_piper_follower.py -v

    # 运行硬件测试（需要连接机器人）
    pytest tests/robots/test_piper_follower.py -v -m hardware

    # 跳过硬件测试
    pytest tests/robots/test_piper_follower.py -v -m "not hardware"
"""

import time
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

# 尝试导入 piper_sdk，如果不可用则跳过相关测试
try:
    from piper_sdk import C_PiperInterface_V2

    PIPER_SDK_AVAILABLE = True
except ImportError:
    PIPER_SDK_AVAILABLE = False
    C_PiperInterface_V2 = None

from lerobot.robots.piper_follower import PiperFollower, PiperFollowerConfig


# ============================================================================
# Mock 测试辅助函数
# ============================================================================


def _make_piper_mock() -> MagicMock:
    """创建 Piper SDK 的 mock 对象"""
    piper = MagicMock(name="PiperInterfaceMock")

    # 模拟关节状态
    joint_state = MagicMock()
    joint_state.joint_1 = 10000  # 10.0 degrees (单位: 0.001 度)
    joint_state.joint_2 = 20000  # 20.0 degrees
    joint_state.joint_3 = -30000  # -30.0 degrees
    joint_state.joint_4 = 5000  # 5.0 degrees
    joint_state.joint_5 = -10000  # -10.0 degrees
    joint_state.joint_6 = 15000  # 15.0 degrees

    joint_msgs = MagicMock()
    joint_msgs.joint_state = joint_state
    piper.GetArmJointMsgs.return_value = joint_msgs

    # 模拟夹爪状态
    gripper_state = MagicMock()
    gripper_state.grippers_angle = 35000  # 35.0 mm (单位: 0.001 mm)

    gripper_msgs = MagicMock()
    gripper_msgs.gripper_state = gripper_state
    piper.GetArmGripperMsgs.return_value = gripper_msgs

    # 模拟末端位姿
    end_pose = MagicMock()
    end_pose.X_axis = 300000  # 300 mm
    end_pose.Y_axis = 0
    end_pose.Z_axis = 200000  # 200 mm
    end_pose.RX_axis = 0
    end_pose.RY_axis = 0
    end_pose.RZ_axis = 0

    end_pose_msgs = MagicMock()
    end_pose_msgs.end_pose = end_pose
    piper.GetArmEndPoseMsgs.return_value = end_pose_msgs

    # 模拟方法
    piper.ConnectPort.return_value = None
    piper.DisconnectPort.return_value = None
    piper.EnableArm.return_value = None
    piper.DisableArm.return_value = None
    piper.MotionCtrl_2.return_value = None
    piper.JointCtrl.return_value = None
    piper.GripperCtrl.return_value = None

    return piper


@pytest.fixture
def mock_piper_follower():
    """创建带 mock 的 PiperFollower fixture"""
    piper_mock = _make_piper_mock()

    with patch(
        "lerobot.robots.piper_follower.piper_follower.C_PiperInterface_V2",
        return_value=piper_mock,
    ):
        cfg = PiperFollowerConfig(
            can_name="can0",
            use_degrees=True,
        )
        robot = PiperFollower(cfg)
        robot._piper = piper_mock  # 直接注入 mock
        yield robot, piper_mock

        if robot.is_connected:
            robot._connected = False  # 强制断开，避免调用真实的断开方法


# ============================================================================
# Mock 测试
# ============================================================================


class TestPiperFollowerMock:
    """使用 mock 的单元测试（不需要真实硬件）"""

    def test_config_defaults(self):
        """测试默认配置"""
        cfg = PiperFollowerConfig()

        assert cfg.can_name == "can0"
        assert cfg.use_degrees is True
        assert cfg.gripper_max_mm == 70.0
        assert cfg.gripper_effort == 1000
        assert len(cfg.joint_limits) == 6

    def test_joint_limits(self):
        """测试关节限位配置"""
        cfg = PiperFollowerConfig()

        assert cfg.joint_limits["joint_1"] == (-150.0, 150.0)
        assert cfg.joint_limits["joint_2"] == (0.0, 180.0)
        assert cfg.joint_limits["joint_3"] == (-170.0, 0.0)
        assert cfg.joint_limits["joint_4"] == (-100.0, 100.0)
        assert cfg.joint_limits["joint_5"] == (-70.0, 70.0)
        assert cfg.joint_limits["joint_6"] == (-120.0, 120.0)

    def test_observation_features(self, mock_piper_follower):
        """测试观测特征定义"""
        robot, _ = mock_piper_follower

        obs_features = robot.observation_features

        # 应该包含 6 个关节 + 1 个夹爪
        expected_keys = [f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"]
        for key in expected_keys:
            assert key in obs_features
            assert obs_features[key] == float

    def test_action_features(self, mock_piper_follower):
        """测试动作特征定义"""
        robot, _ = mock_piper_follower

        action_features = robot.action_features

        # 应该包含 6 个关节 + 1 个夹爪
        expected_keys = [f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"]
        for key in expected_keys:
            assert key in action_features
            assert action_features[key] == float

    def test_connect_calls_sdk(self, mock_piper_follower):
        """测试连接时调用 SDK 方法"""
        robot, piper_mock = mock_piper_follower

        # 模拟连接
        robot._connected = False
        with patch.object(robot, "_enable_arm"):
            robot._piper = piper_mock
            robot._connected = True

        assert robot.is_connected

    def test_get_observation_returns_correct_values(self, mock_piper_follower):
        """测试获取观测返回正确的值"""
        robot, piper_mock = mock_piper_follower
        robot._connected = True

        obs = robot.get_observation()

        # 检查关节角度（mock 返回的值 / 1000）
        assert abs(obs["joint_1.pos"] - 10.0) < 0.01
        assert abs(obs["joint_2.pos"] - 20.0) < 0.01
        assert abs(obs["joint_3.pos"] - (-30.0)) < 0.01
        assert abs(obs["joint_4.pos"] - 5.0) < 0.01
        assert abs(obs["joint_5.pos"] - (-10.0)) < 0.01
        assert abs(obs["joint_6.pos"] - 15.0) < 0.01

        # 检查夹爪位置
        assert abs(obs["gripper.pos"] - 35.0) < 0.01

    def test_send_action_calls_joint_ctrl(self, mock_piper_follower):
        """测试发送动作时调用 JointCtrl"""
        robot, piper_mock = mock_piper_follower
        robot._connected = True

        action = {
            "joint_1.pos": 10.0,
            "joint_2.pos": 20.0,
            "joint_3.pos": -30.0,
            "joint_4.pos": 5.0,
            "joint_5.pos": -10.0,
            "joint_6.pos": 15.0,
            "gripper.pos": 40.0,
        }

        robot.send_action(action)

        # 验证 JointCtrl 被调用，参数应该是 0.001 度单位
        piper_mock.JointCtrl.assert_called_once_with(
            joint_1=10000,
            joint_2=20000,
            joint_3=-30000,
            joint_4=5000,
            joint_5=-10000,
            joint_6=15000,
        )

        # 验证 GripperCtrl 被调用
        piper_mock.GripperCtrl.assert_called()


# ============================================================================
# 硬件测试（需要连接真实机器人）
# ============================================================================


@pytest.mark.hardware
@pytest.mark.skipif(not PIPER_SDK_AVAILABLE, reason="piper_sdk not installed")
class TestPiperFollowerHardware:
    """
    真实硬件测试

    运行前请确保：
    1. Piper 机械臂已上电
    2. CAN 接口已配置: sudo ip link set can0 type can bitrate 1000000 && sudo ip link set up can0
    3. 急停按钮已释放

    运行命令: pytest tests/robots/test_piper_follower.py -v -m hardware
    """

    @pytest.fixture
    def real_robot(self):
        """创建连接真实硬件的 PiperFollower"""
        cfg = PiperFollowerConfig(
            can_name="can0",
            use_degrees=True,
            disable_torque_on_disconnect=True,
        )
        robot = PiperFollower(cfg)
        yield robot

        # 清理
        if robot.is_connected:
            robot.disconnect()

    def test_connect_and_disconnect(self, real_robot):
        """测试连接和断开"""
        assert not real_robot.is_connected

        real_robot.connect()
        assert real_robot.is_connected

        real_robot.disconnect()
        assert not real_robot.is_connected

    def test_read_joint_states(self, real_robot):
        """测试读取关节状态"""
        real_robot.connect()

        # 连续读取 5 次
        for i in range(5):
            obs = real_robot.get_observation()

            print(f"\n[{i}] 关节状态:")
            for j in range(1, 7):
                key = f"joint_{j}.pos"
                assert key in obs
                print(f"  Joint {j}: {obs[key]:.2f}°")

            assert "gripper.pos" in obs
            print(f"  Gripper: {obs['gripper.pos']:.2f}mm")

            time.sleep(0.2)

    def test_joint_values_in_range(self, real_robot):
        """测试关节值在合理范围内"""
        real_robot.connect()

        obs = real_robot.get_observation()

        # 检查关节值是否在限位范围内
        limits = real_robot.config.joint_limits
        for j in range(1, 7):
            key = f"joint_{j}.pos"
            joint_name = f"joint_{j}"
            value = obs[key]
            min_val, max_val = limits[joint_name]

            assert min_val <= value <= max_val, f"{key} = {value} 超出范围 [{min_val}, {max_val}]"

    def test_read_consistency(self, real_robot):
        """测试连续读取的一致性（机械臂静止时）"""
        real_robot.connect()

        readings = []
        for _ in range(10):
            obs = real_robot.get_observation()
            readings.append([obs[f"joint_{j}.pos"] for j in range(1, 7)])
            time.sleep(0.05)

        # 计算每个关节的标准差
        import numpy as np

        readings = np.array(readings)
        stds = np.std(readings, axis=0)

        print("\n关节读取标准差:")
        for j, std in enumerate(stds, 1):
            print(f"  Joint {j}: {std:.4f}°")
            # 静止时标准差应该很小（< 1 度）
            assert std < 1.0, f"Joint {j} 读取不稳定，标准差 = {std:.4f}"


# ============================================================================
# 交互式测试（手动运行）
# ============================================================================


def interactive_test_connection():
    """
    交互式测试：基础连接

    运行方式: python -c "from tests.robots.test_piper_follower import interactive_test_connection; interactive_test_connection()"
    """
    print("=" * 60)
    print("Piper Follower 连接测试")
    print("=" * 60)

    cfg = PiperFollowerConfig(can_name="can0", use_degrees=True)
    robot = PiperFollower(cfg)

    print("[1/3] 正在连接...")
    robot.connect()
    print("连接成功!")

    print("\n[2/3] 读取关节状态 (5次):")
    for i in range(5):
        obs = robot.get_observation()
        joints = [f"{obs[f'joint_{j}.pos']:.1f}" for j in range(1, 7)]
        gripper = obs["gripper.pos"]
        print(f"  [{i}] Joints: {joints}° | Gripper: {gripper:.1f}mm")
        time.sleep(0.3)

    print("\n[3/3] 断开连接...")
    robot.disconnect()
    print("测试完成!")


def interactive_test_with_camera():
    """
    交互式测试：带摄像头

    运行方式: python -c "from tests.robots.test_piper_follower import interactive_test_with_camera; interactive_test_with_camera()"
    """
    from lerobot.cameras.opencv import OpenCVCameraConfig

    print("=" * 60)
    print("Piper Follower + 摄像头测试")
    print("=" * 60)

    cfg = PiperFollowerConfig(
        can_name="can0",
        use_degrees=True,
        cameras={
            "observation.images.front": OpenCVCameraConfig(
                index_or_path=0,
                width=640,
                height=480,
                fps=30,
            )
        },
    )
    robot = PiperFollower(cfg)

    print("[1/3] 正在连接机器人和摄像头...")
    robot.connect()
    print("连接成功!")

    print("\n[2/3] 读取观测:")
    obs = robot.get_observation()

    print("  关节状态:")
    for j in range(1, 7):
        print(f"    Joint {j}: {obs[f'joint_{j}.pos']:.2f}°")
    print(f"    Gripper: {obs['gripper.pos']:.2f}mm")

    print(f"\n  图像形状: {obs['observation.images.front'].shape}")

    print("\n[3/3] 断开连接...")
    robot.disconnect()
    print("测试完成!")


if __name__ == "__main__":
    # 直接运行此文件时，执行交互式测试
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--camera":
        interactive_test_with_camera()
    else:
        interactive_test_connection()
