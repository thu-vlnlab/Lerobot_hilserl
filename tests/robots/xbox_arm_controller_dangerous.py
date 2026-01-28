#!/usr/bin/env python3
"""
[DANGEROUS] Xbox控制器控制Piper机械臂 - 危险模式
初始位置不是零位，而是预设的工作位置

位置控制的奇异点太多，不建议使用位置控制模式
Xbox控制器控制Piper机械臂
使用速度控制模式，LB键作为使能开关


==============================================================================
                            操作说明
==============================================================================

启动方式:
    python3 xbox_arm_controller_dangerous.py              # 默认使用 can0
    python3 xbox_arm_controller_dangerous.py --can can1   # 指定 CAN 端口
    python3 xbox_arm_controller_dangerous.py -c can_piper # 简写形式

控制方式:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  【重要】按住 LB 键才能使能控制，松开 LB 立即停止                    │
    └─────────────────────────────────────────────────────────────────────┘

    摇杆控制:
        左摇杆 Y 轴  → 末端 X 方向移动 (前后)
        左摇杆 X 轴  → 末端 Y 方向移动 (左右)
        右摇杆 X 轴  → 末端 RZ 旋转 (偏航/Yaw)
        右摇杆 Y 轴  → 末端 Z 方向移动 (上下)

    扳机控制:
        LT 扳机      → 末端 RY 正向旋转 (俯仰/Pitch)
        RT 扳机      → 末端 RY 负向旋转 (俯仰/Pitch)

    按钮控制:
        A 按钮       → 末端 RX 正向旋转 (翻滚/Roll)
        B 按钮       → 末端 RX 负向旋转 (翻滚/Roll)
        LB 按钮      → 【使能开关】按住才能控制

    十字键控制 (增量模式):
        十字键 ↑     → 按住持续张开夹爪
        十字键 ↓     → 按住持续闭合夹爪
        松开         → 停止

    退出:
        Ctrl + C     → 安全退出程序

安全机制:
    - 松开 LB 立即停止所有运动
    - 位置软限位保护
    - CAN 通信超时检测
    - 连续错误自动紧急停止

==============================================================================
"""

import pygame
import time
import sys
import argparse
from typing import Optional, Tuple, List
from collections import deque
from piper_sdk import C_PiperInterface_V2


# ============================================================================
# [DANGEROUS] 预设初始关节位置 (单位: 0.001度)
# ============================================================================
INIT_JOINT_POSITIONS = {
    'J1': 2362,      # 2.362°
    'J2': 77140,     # 77.140°
    'J3': -56995,    # -56.995°
    'J4': -5172,     # -5.172°
    'J5': 57728,     # 57.728°
    'J6': -3150      # -3.150°
}


class ControlConfig:
    """控制参数配置"""

    # 速度缩放因子
    LINEAR_VELOCITY_SCALE = 50.0   # mm/s (摇杆满偏时的速度)
    ANGULAR_VELOCITY_SCALE = 30.0  # deg/s

    # 速度限制
    MAX_LINEAR_VELOCITY = 100.0
    MAX_ANGULAR_VELOCITY = 60.0

    # 死区设置
    JOYSTICK_DEADZONE = 0.1  # 10%
    TRIGGER_DEADZONE = 0.05  # 5%

    # 位置限位
    POSITION_LIMITS = {
        'X': (-100, 500),      # mm
        'Y': (-400, 400),
        'Z': (-50, 600),
        'RX': (-180, 180),     # degrees
        'RY': (-180, 180),
        'RZ': (-180, 180)
    }

    # 软限位缓冲区
    SOFT_LIMIT_BUFFER = 50  # mm或degrees

    # 夹爪参数
    GRIPPER_MAX_POS = 80000     # 最大开度 80mm
    GRIPPER_MIN_POS = 0         # 最小开度 0mm
    GRIPPER_SPEED = 1000
    GRIPPER_STEP = 1500         # 每次循环增量 (按住时持续变化)

    # 控制频率
    CONTROL_FREQUENCY = 140 # Hz


class ControllerState:
    """控制器状态"""

    def __init__(self):
        self.enabled = False
        self.current_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # [X, Y, Z, RX, RY, RZ]
        self.target_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # 目标位姿（累积）
        self.last_sent_target_pose = None  # 上一帧发送的目标位姿
        self.last_tracking_error = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.last_target_delta = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.target_velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.actual_velocity = [0.0, 0.0, 0.0]  # 基于最近1秒位姿变化的实际速度(mm/s)
        self.pose_history = deque(maxlen=500)   # (t, x, y, z)
        self.feedback_hz = 0.0
        self.last_feedback_stamp = None
        self.last_pose = None
        self.last_command_time = None  # 用于“消息延迟”
        self.last_command_time_motion = None  # 用于“运动延迟”
        self.last_command_has_move = False
        self.last_motion_lag_ms = None
        self.last_msg_lag_ms = None
        self.last_update_time = time.time()
        self.gripper_state = "idle"  # idle, opening, closing
        self.gripper_pos = 0  # 当前夹爪位置 (单位: 0.001mm)
        self.pose_initialized = False  # 目标位姿是否已初始化


class SafetyMonitor:
    """安全监控"""

    def __init__(self, config: ControlConfig):
        self.config = config
        self.last_can_communication = time.time()
        self.emergency_stop = False
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        self.watchdog_timeout = 1.0  # 秒

    def check_safety(self, current_pose: List[float], target_pose: List[float]) -> Tuple[bool, str]:
        """
        综合安全检查

        Returns:
            (is_safe, error_message)
        """
        # 检查紧急停止
        if self.emergency_stop:
            return False, "紧急停止已触发"

        # 检查通信超时
        if time.time() - self.last_can_communication > self.watchdog_timeout:
            return False, "CAN通信超时"

        # 检查位置限位
        axis_names = ['X', 'Y', 'Z', 'RX', 'RY', 'RZ']
        for i, axis in enumerate(axis_names):
            limits = self.config.POSITION_LIMITS[axis]
            if not (limits[0] <= target_pose[i] <= limits[1]):
                return False, f"{axis}轴超出限位: {target_pose[i]:.1f}"

        # 检查软限位（接近边界时警告）
        for i, axis in enumerate(axis_names):
            limits = self.config.POSITION_LIMITS[axis]
            unit = "mm" if i < 3 else "°"
            if target_pose[i] < limits[0] + self.config.SOFT_LIMIT_BUFFER:
                distance = target_pose[i] - limits[0]
                # print(f"[WARNING] {axis}轴接近下限: 当前={target_pose[i]:.1f}{unit}, 限制={limits[0]}{unit}, 距离={distance:.1f}{unit}")
            elif target_pose[i] > limits[1] - self.config.SOFT_LIMIT_BUFFER:
                distance = limits[1] - target_pose[i]
                # print(f"[WARNING] {axis}轴接近上限: 当前={target_pose[i]:.1f}{unit}, 限制={limits[1]}{unit}, 距离={distance:.1f}{unit}")

        return True, ""

    def trigger_emergency_stop(self):
        """触发紧急停止"""
        self.emergency_stop = True
        print("\n[EMERGENCY] 紧急停止触发！")


class XboxArmController:
    """Xbox控制器到机械臂的主控制器"""

    def __init__(self, can_port: str = "can_piper"):
        self.can_port = can_port
        self.config = ControlConfig()
        self.state = ControllerState()
        self.safety = SafetyMonitor(self.config)

        self.piper: Optional[C_PiperInterface_V2] = None
        self.joystick: Optional[pygame.joystick.Joystick] = None

        self.should_exit = False
        self.frame_count = 0

    def apply_deadzone(self, value: float, deadzone: float) -> float:
        """应用死区，避免摇杆漂移"""
        if abs(value) < deadzone:
            return 0.0

        # 重新映射，使死区外的值仍然覆盖完整范围
        sign = 1 if value > 0 else -1
        normalized = (abs(value) - deadzone) / (1.0 - deadzone)
        return sign * normalized

    def initialize_controller(self) -> bool:
        """初始化Xbox控制器"""
        try:
            pygame.init()
            pygame.joystick.init()

            joystick_count = pygame.joystick.get_count()
            if joystick_count == 0:
                print("[ERROR] 未检测到游戏控制器")
                return False

            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()

            print(f"[SUCCESS] 检测到控制器: {self.joystick.get_name()}")
            return True

        except Exception as e:
            print(f"[ERROR] 控制器初始化失败: {e}")
            return False

    def initialize_arm(self) -> bool:
        """初始化机械臂"""
        try:
            # 连接CAN端口
            self.piper = C_PiperInterface_V2(self.can_port)
            self.piper.ConnectPort()
            time.sleep(0.1)

            # 设置控制模式
            self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
            time.sleep(0.1)

            # 使能机械臂
            if not self._enable_with_timeout(timeout=5.0):
                raise RuntimeError("机械臂使能超时")

            # 初始化夹爪
            self.piper.GripperCtrl(0, self.config.GRIPPER_SPEED, 0x01, 0)
            time.sleep(0.1)

            # [DANGEROUS] 移动到预设位置（使用关节控制模式）
            print("[DANGEROUS] 正在移动到预设工作位置...")
            print(f"           目标: J1={INIT_JOINT_POSITIONS['J1']*0.001:.2f}° "
                  f"J2={INIT_JOINT_POSITIONS['J2']*0.001:.2f}° "
                  f"J3={INIT_JOINT_POSITIONS['J3']*0.001:.2f}° "
                  f"J4={INIT_JOINT_POSITIONS['J4']*0.001:.2f}° "
                  f"J5={INIT_JOINT_POSITIONS['J5']*0.001:.2f}° "
                  f"J6={INIT_JOINT_POSITIONS['J6']*0.001:.2f}°")

            self.piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)  # 关节控制模式，速度30%
            self.piper.JointCtrl(
                INIT_JOINT_POSITIONS['J1'],
                INIT_JOINT_POSITIONS['J2'],
                INIT_JOINT_POSITIONS['J3'],
                INIT_JOINT_POSITIONS['J4'],
                INIT_JOINT_POSITIONS['J5'],
                INIT_JOINT_POSITIONS['J6']
            )

            # 目标关节角度（度）
            target_joints = [
                INIT_JOINT_POSITIONS['J1'] * 0.001,
                INIT_JOINT_POSITIONS['J2'] * 0.001,
                INIT_JOINT_POSITIONS['J3'] * 0.001,
                INIT_JOINT_POSITIONS['J4'] * 0.001,
                INIT_JOINT_POSITIONS['J5'] * 0.001,
                INIT_JOINT_POSITIONS['J6'] * 0.001
            ]

            # 等待到达目标位置
            for i in range(100):  # 最多等待10秒
                time.sleep(0.1)
                joint = self.piper.GetArmJointMsgs()
                current_joints = [
                    joint.joint_state.joint_1 * 0.001,
                    joint.joint_state.joint_2 * 0.001,
                    joint.joint_state.joint_3 * 0.001,
                    joint.joint_state.joint_4 * 0.001,
                    joint.joint_state.joint_5 * 0.001,
                    joint.joint_state.joint_6 * 0.001
                ]
                # 检查是否接近目标位置 (误差小于2度)
                all_near_target = all(
                    abs(current_joints[j] - target_joints[j]) < 2.0
                    for j in range(6)
                )
                if all_near_target:
                    print("[INFO] 已到达预设位置")
                    break
                if i % 10 == 0:
                    print(f"[INFO] 移动中... J1={current_joints[0]:.1f}° "
                          f"J2={current_joints[1]:.1f}° J3={current_joints[2]:.1f}°")
            else:
                print("[WARNING] 移动超时，继续启动")

            time.sleep(0.2)

            # 切换回末端位姿控制模式
            self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
            time.sleep(0.1)

            # 读取初始位姿
            end_pose = self.piper.GetArmEndPoseMsgs()
            self.state.current_pose = [
                end_pose.end_pose.X_axis / 1000.0,
                end_pose.end_pose.Y_axis / 1000.0,
                end_pose.end_pose.Z_axis / 1000.0,
                end_pose.end_pose.RX_axis / 1000.0,
                end_pose.end_pose.RY_axis / 1000.0,
                end_pose.end_pose.RZ_axis / 1000.0
            ]

            # 用当前位姿初始化目标位姿，防止连接时自动移动
            self.state.target_pose = self.state.current_pose.copy()
            self.state.pose_initialized = True

            # 发送当前位姿作为目标，锁定当前位置
            self.piper.MotionCtrl_2(0x01, 0x00, 10, 0x00)
            self.piper.EndPoseCtrl(
                end_pose.end_pose.X_axis,
                end_pose.end_pose.Y_axis,
                end_pose.end_pose.Z_axis,
                end_pose.end_pose.RX_axis,
                end_pose.end_pose.RY_axis,
                end_pose.end_pose.RZ_axis
            )

            print(f"[SUCCESS] 机械臂初始化完成")
            print(f"          当前位姿: X={self.state.current_pose[0]:.1f} "
                  f"Y={self.state.current_pose[1]:.1f} "
                  f"Z={self.state.current_pose[2]:.1f} "
                  f"RX={self.state.current_pose[3]:.1f} "
                  f"RY={self.state.current_pose[4]:.1f} "
                  f"RZ={self.state.current_pose[5]:.1f}")

            # 更新看门狗时间戳，防止初始化后立即超时
            self.safety.last_can_communication = time.time()
            return True

        except Exception as e:
            print(f"[ERROR] 机械臂初始化失败: {e}")
            return False

    def _enable_with_timeout(self, timeout: float = 5.0) -> bool:
        """使能机械臂并检测状态"""
        start_time = time.time()

        while True:
            elapsed_time = time.time() - start_time

            # 检查所有电机使能状态
            low_spd_info = self.piper.GetArmLowSpdInfoMsgs()
            enable_flag = (
                low_spd_info.motor_1.foc_status.driver_enable_status and
                low_spd_info.motor_2.foc_status.driver_enable_status and
                low_spd_info.motor_3.foc_status.driver_enable_status and
                low_spd_info.motor_4.foc_status.driver_enable_status and
                low_spd_info.motor_5.foc_status.driver_enable_status and
                low_spd_info.motor_6.foc_status.driver_enable_status
            )

            if enable_flag:
                return True

            # 发送使能命令
            self.piper.EnableArm(7)

            # 检查超时
            if elapsed_time > timeout:
                return False

            time.sleep(0.5)

    def initialize(self) -> bool:
        """初始化所有组件"""
        print("="*60)
        print("   [DANGEROUS] Piper机械臂 Xbox控制器 v1.0")
        print("="*60)

        # 初始化控制器
        print("[1/4] 正在检测Xbox控制器...")
        if not self.initialize_controller():
            return False

        # 连接CAN总线
        print(f"[2/4] 正在连接CAN总线 ({self.can_port})...")

        # 初始化机械臂
        print("[3/4] 正在使能机械臂...")
        if not self.initialize_arm():
            return False

        print("[4/4] 系统初始化完成")
        print("="*60)
        print("[DANGEROUS] 系统就绪！使用预设工作位置")
        print("使用说明：")
        print("  - 按住 LB 键使能控制")
        print("  - 左摇杆控制 XY 平面")
        print("  - 右摇杆控制 Z 轴和 RZ 旋转")
        print("  - LT/RT 扳机控制 RY 俯仰")
        print("  - A/B 按钮控制 RX 翻滚")
        print("  - 十字键上/下控制夹爪")
        print("  - 按 Ctrl+C 安全退出")
        print("="*60)

        return True

    def read_controller_inputs(self) -> dict:
        """读取控制器输入并应用死区"""
        pygame.event.pump()

        # 读取摇杆
        left_x = self.apply_deadzone(
            self.joystick.get_axis(0),
            self.config.JOYSTICK_DEADZONE
        )
        left_y = self.apply_deadzone(
            self.joystick.get_axis(1),
            self.config.JOYSTICK_DEADZONE
        )
        right_x = self.apply_deadzone(
            self.joystick.get_axis(3),
            self.config.JOYSTICK_DEADZONE
        )
        right_y = self.apply_deadzone(
            self.joystick.get_axis(4),
            self.config.JOYSTICK_DEADZONE
        )

        # 读取扳机（-1到1，转换为0到1）
        lt_raw = (self.joystick.get_axis(2) + 1) / 2
        rt_raw = (self.joystick.get_axis(5) + 1) / 2

        lt = self.apply_deadzone(lt_raw, self.config.TRIGGER_DEADZONE)
        rt = self.apply_deadzone(rt_raw, self.config.TRIGGER_DEADZONE)

        # 读取按钮
        button_a = self.joystick.get_button(0)
        button_b = self.joystick.get_button(1)
        button_lb = self.joystick.get_button(4)

        # 读取方向键
        hat = self.joystick.get_hat(0) if self.joystick.get_numhats() > 0 else (0, 0)

        return {
            'left_x': left_x,
            'left_y': left_y,
            'right_x': right_x,
            'right_y': right_y,
            'lt': lt,
            'rt': rt,
            'button_a': button_a,
            'button_b': button_b,
            'button_lb': button_lb,
            'dpad_up': hat[1] == 1,
            'dpad_down': hat[1] == -1
        }

    def calculate_target_pose(self, inputs: dict, dt: float) -> List[float]:
        """根据控制器输入计算目标位姿"""
        current = self.state.current_pose.copy()

        # 计算速度 (mm/s 或 deg/s)
        vel_x = -inputs['left_y'] * self.config.LINEAR_VELOCITY_SCALE  # 左摇杆Y控制X方向
        vel_y = inputs['left_x'] * self.config.LINEAR_VELOCITY_SCALE   # 左摇杆X控制Y方向
        vel_z = -inputs['right_y'] * self.config.LINEAR_VELOCITY_SCALE  # Z轴反转

        vel_rz = inputs['right_x'] * self.config.ANGULAR_VELOCITY_SCALE  # Yaw
        vel_ry = (inputs['lt'] - inputs['rt']) * self.config.ANGULAR_VELOCITY_SCALE  # Pitch
        vel_rx = (inputs['button_a'] - inputs['button_b']) * self.config.ANGULAR_VELOCITY_SCALE  # Roll

        # 保存速度（用于显示）
        self.state.target_velocity = [vel_x, vel_y, vel_z, vel_rx, vel_ry, vel_rz]

        # 计算位置增量
        delta = [
            vel_x * dt,
            vel_y * dt,
            vel_z * dt,
            vel_rx * dt,
            vel_ry * dt,
            vel_rz * dt
        ]
        self.state.last_target_delta = delta
        # 如果本帧指令有明显平移增量，记录指令时间用于估算延迟
        if (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]) ** 0.5 > 0.02:
            # 记录发送“有移动意图”的指令时间，用于估算新反馈到达的延迟
            now = time.time()
            self.state.last_command_time = now
            self.state.last_command_time_motion = now
            self.state.last_command_has_move = True

        target = [
            current[0] + vel_x * dt,
            current[1] + vel_y * dt,
            current[2] + vel_z * dt,
            current[3] + vel_rx * dt,
            current[4] + vel_ry * dt,
            current[5] + vel_rz * dt
        ]

        return target

    def send_control_command(self, target_pose: List[float]):
        """发送控制命令到机械臂"""
        try:
            # 记录本帧发送的目标位姿，用于下一帧计算跟踪误差
            self.state.last_sent_target_pose = target_pose.copy()

            # 转换单位：mm → 0.001mm, deg → 0.001deg
            X = int(target_pose[0] * 1000)
            Y = int(target_pose[1] * 1000)
            Z = int(target_pose[2] * 1000)
            RX = int(target_pose[3] * 1000)
            RY = int(target_pose[4] * 1000)
            RZ = int(target_pose[5] * 1000)

            # 计算速度百分比（根据当前速度动态调整）
            max_vel = max(abs(v) for v in self.state.target_velocity)
            if max_vel < 1.0:
                speed_percent = 10
            elif max_vel < 10.0:
                speed_percent = 30
            elif max_vel < 30.0:
                speed_percent = 60
            else:
                speed_percent = 100

            # 设置运动模式
            self.piper.MotionCtrl_2(0x01, 0x00, speed_percent, 0x00)

            # 发送末端位姿命令
            self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)

            self.safety.last_can_communication = time.time()
            self.safety.consecutive_errors = 0

        except Exception as e:
            print(f"[ERROR] 发送控制命令失败: {e}")
            self.safety.consecutive_errors += 1
            if self.safety.consecutive_errors > self.safety.max_consecutive_errors:
                self.safety.trigger_emergency_stop()

    def send_stop_command(self):
        """发送停止命令（保持当前位置）"""
        try:
            current = self.state.current_pose
            X = int(current[0] * 1000)
            Y = int(current[1] * 1000)
            Z = int(current[2] * 1000)
            RX = int(current[3] * 1000)
            RY = int(current[4] * 1000)
            RZ = int(current[5] * 1000)

            self.piper.MotionCtrl_2(0x01, 0x00, 10, 0x00)
            self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)

            # 清零速度
            self.state.target_velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        except Exception as e:
            print(f"[ERROR] 发送停止命令失败: {e}")

    def handle_gripper(self, inputs: dict):
        """处理夹爪控制 - 增量模式"""
        try:
            if inputs['dpad_up']:
                # 按住 ↑ 持续张开
                self.state.gripper_state = "opening"
                self.state.gripper_pos += self.config.GRIPPER_STEP
                # 限制最大值
                if self.state.gripper_pos > self.config.GRIPPER_MAX_POS:
                    self.state.gripper_pos = self.config.GRIPPER_MAX_POS
                self.piper.GripperCtrl(
                    int(self.state.gripper_pos),
                    self.config.GRIPPER_SPEED,
                    0x01,
                    0
                )
            elif inputs['dpad_down']:
                # 按住 ↓ 持续闭合
                self.state.gripper_state = "closing"
                self.state.gripper_pos -= self.config.GRIPPER_STEP
                # 限制最小值
                if self.state.gripper_pos < self.config.GRIPPER_MIN_POS:
                    self.state.gripper_pos = self.config.GRIPPER_MIN_POS
                self.piper.GripperCtrl(
                    int(self.state.gripper_pos),
                    self.config.GRIPPER_SPEED,
                    0x01,
                    0
                )
            else:
                # 松开按键，停止
                if self.state.gripper_state != "idle":
                    self.state.gripper_state = "idle"

        except Exception as e:
            print(f"[ERROR] 夹爪控制失败: {e}")

    def update_current_pose(self):
        """读取当前机械臂位姿"""
        try:
            now = time.time()
            end_pose = self.piper.GetArmEndPoseMsgs()
            self.state.current_pose = [
                end_pose.end_pose.X_axis / 1000.0,
                end_pose.end_pose.Y_axis / 1000.0,
                end_pose.end_pose.Z_axis / 1000.0,
                end_pose.end_pose.RX_axis / 1000.0,
                end_pose.end_pose.RY_axis / 1000.0,
                end_pose.end_pose.RZ_axis / 1000.0
            ]
            # 成功读取位姿，更新看门狗时间戳
            self.safety.last_can_communication = now

            # 反馈频率使用 SDK 统计（来自 CAN 反馈包）
            self.state.feedback_hz = end_pose.Hz

            # 记录反馈包时间戳变化，用于估算“命令->新反馈”延迟
            if end_pose.time_stamp != 0:
                if self.state.last_feedback_stamp is None:
                    self.state.last_feedback_stamp = end_pose.time_stamp
                elif end_pose.time_stamp != self.state.last_feedback_stamp:
                    if self.state.last_command_time is not None:
                        self.state.last_msg_lag_ms = (now - self.state.last_command_time) * 1000.0
                        self.state.last_command_time = None
                    self.state.last_feedback_stamp = end_pose.time_stamp

            # 更新位姿历史，用于计算实际速度
            self.state.pose_history.append((now, self.state.current_pose[0],
                                            self.state.current_pose[1],
                                            self.state.current_pose[2]))
            self._update_actual_velocity(now)

            # 估计指令到运动的延迟（第一次检测到运动时记录）
            if self.state.last_pose is not None and self.state.last_command_has_move:
                dx = self.state.current_pose[0] - self.state.last_pose[0]
                dy = self.state.current_pose[1] - self.state.last_pose[1]
                dz = self.state.current_pose[2] - self.state.last_pose[2]
                moved = (dx * dx + dy * dy + dz * dz) ** 0.5
                if moved > 0.02 and self.state.last_command_time_motion is not None:
                    self.state.last_motion_lag_ms = (now - self.state.last_command_time_motion) * 1000.0
                    self.state.last_command_has_move = False

            self.state.last_pose = self.state.current_pose.copy()
        except Exception as e:
            print(f"[ERROR] 读取位姿失败: {e}")
            self.safety.consecutive_errors += 1

    def _update_actual_velocity(self, now: float):
        """基于最近1秒位姿变化估计实际速度(mm/s)"""
        if len(self.state.pose_history) < 2:
            return
        # 找到1秒前最接近的样本
        target_time = now - 1.0
        oldest = None
        for t, x, y, z in self.state.pose_history:
            if t <= target_time:
                oldest = (t, x, y, z)
            else:
                break
        # 若不足1秒，使用最早样本
        if oldest is None:
            oldest = self.state.pose_history[0]

        t0, x0, y0, z0 = oldest
        dt = now - t0
        if dt <= 1e-6:
            return
        dx = self.state.current_pose[0] - x0
        dy = self.state.current_pose[1] - y0
        dz = self.state.current_pose[2] - z0
        self.state.actual_velocity = [dx / dt, dy / dt, dz / dt]

    def update_display(self):
        """更新状态显示"""
        pose = self.state.current_pose
        vel = self.state.target_velocity
        err = self.state.last_tracking_error
        delta = self.state.last_target_delta
        aval = self.state.actual_velocity
        fb_hz = self.state.feedback_hz
        lag_ms = self.state.last_motion_lag_ms
        msg_lag_ms = self.state.last_msg_lag_ms
        enabled_str = "已使能" if self.state.enabled else "未使能"
        gripper_mm = self.state.gripper_pos / 1000.0  # 转换为 mm

        freq = self.config.CONTROL_FREQUENCY

        # 仅显示平移相关量，统一用毫米，避免角度单位混淆

        # 误差百分比（以增量为分母，使用绝对值避免符号误导）
        def _pct(e: float, d: float) -> float:
            return 0.0 if abs(d) < 1e-6 else (abs(e) / abs(d)) * 100.0

        pct_x = _pct(err[0], delta[0])
        pct_y = _pct(err[1], delta[1])
        pct_z = _pct(err[2], delta[2])

        lag_str = "--" if lag_ms is None else f"{lag_ms:.0f}"
        msg_lag_str = "--" if msg_lag_ms is None else f"{msg_lag_ms:.0f}"

        status = (f"[{enabled_str}] 频率:{freq}Hz | "
                 f"位姿(mm):X={pose[0]:.0f}/Y={pose[1]:.0f}/Z={pose[2]:.0f} | "
                 f"增量(mm):dX={delta[0]:.1f}/dY={delta[1]:.1f}/dZ={delta[2]:.1f} | "
                 f"误差(mm):eX={err[0]:.1f}/eY={err[1]:.1f}/eZ={err[2]:.1f} | "
                 f"误差(%):pX={pct_x:.0f}/pY={pct_y:.0f}/pZ={pct_z:.0f} | "
                 f"目标速度(mm/s):vX={vel[0]:.0f}/vY={vel[1]:.0f}/vZ={vel[2]:.0f} | "
                 f"实际速度(mm/s):aX={aval[0]:.0f}/aY={aval[1]:.0f}/aZ={aval[2]:.0f} | "
                 f"反馈:{fb_hz:.0f}Hz/延迟:{lag_str}ms/消息:{msg_lag_str}ms | "
                 f"[夹爪:{gripper_mm:.1f}mm]")

        # 使用\r返回行首，实现同行更新
        print(f"\r{status}", end='', flush=True)

    def run(self):
        """主控制循环"""
        clock = pygame.time.Clock()

        try:
            while not self.should_exit:
                loop_start = time.time()

                try:
                    # 读取控制器输入
                    inputs = self.read_controller_inputs()

                    # 检查使能键
                    lb_pressed = inputs['button_lb']

                    # 夹爪控制不需要按 LB，随时可用
                    self.handle_gripper(inputs)

                    if lb_pressed:
                        if not self.state.enabled:
                            print("\n[INFO] 控制器已使能")
                            self.state.enabled = True
                            # 重新使能机械臂
                            self.piper.EnableArm(7)
                            time.sleep(0.05)
                            # 重新读取当前位姿作为起点
                            self.update_current_pose()
                            self.state.target_pose = self.state.current_pose.copy()
                            self.state.pose_initialized = True
                            # 发送当前位姿锁定位置
                            end_pose = self.piper.GetArmEndPoseMsgs()
                            self.piper.MotionCtrl_2(0x01, 0x00, 50, 0x00)
                            self.piper.EndPoseCtrl(
                                end_pose.end_pose.X_axis,
                                end_pose.end_pose.Y_axis,
                                end_pose.end_pose.Z_axis,
                                end_pose.end_pose.RX_axis,
                                end_pose.end_pose.RY_axis,
                                end_pose.end_pose.RZ_axis
                            )

                        # 更新当前位姿
                        self.update_current_pose()

                        # 计算上一帧目标 vs 当前位姿的跟踪误差
                        if self.state.last_sent_target_pose is not None:
                            self.state.last_tracking_error = [
                                self.state.last_sent_target_pose[i] - self.state.current_pose[i]
                                for i in range(6)
                            ]

                        # 计算目标位姿
                        dt = time.time() - self.state.last_update_time
                        target_pose = self.calculate_target_pose(inputs, dt)

                        # # 调试输出（每100帧输出一次）
                        # if self.frame_count % 100 == 0:
                        #     print(f"\n[DEBUG] 当前位姿: {self.state.current_pose}")
                        #     print(f"[DEBUG] 目标位姿: {target_pose}")
                        #     print(f"[DEBUG] 输入: LX={inputs['left_x']:.2f} LY={inputs['left_y']:.2f} RX={inputs['right_x']:.2f} RY={inputs['right_y']:.2f}")

                        # 安全检查
                        is_safe, error_msg = self.safety.check_safety(
                            self.state.current_pose,
                            target_pose
                        )

                        if is_safe:
                            # 发送控制命令
                            self.send_control_command(target_pose)
                        else:
                            print(f"\n[SAFETY] {error_msg} - 停止运动")
                            self.send_stop_command()

                    else:
                        if self.state.enabled:
                            print("\n[INFO] 控制器已失能 - 停止运动")
                            self.state.enabled = False

                        # LB未按下，发送停止命令
                        self.send_stop_command()

                    # 更新时间戳
                    self.state.last_update_time = time.time()
                    self.frame_count += 1

                    # 每10帧更新一次显示
                    if self.frame_count % 10 == 0:
                        self.update_display()

                except Exception as e:
                    print(f"\n[ERROR] 控制循环异常: {e}")
                    self.safety.consecutive_errors += 1
                    if self.safety.consecutive_errors > self.safety.max_consecutive_errors:
                        print("[CRITICAL] 连续错误过多，触发紧急停止")
                        self.safety.trigger_emergency_stop()
                        break

                # 控制循环频率
                clock.tick(self.config.CONTROL_FREQUENCY)

        except KeyboardInterrupt:
            print("\n\n[INFO] 用户请求退出...")
        except Exception as e:
            print(f"\n[CRITICAL] 严重错误: {e}")
        finally:
            self.shutdown()

    def shutdown(self):
        """安全关闭系统"""
        print("\n" + "="*60)
        print("正在安全关闭系统...")
        print("="*60)

        try:
            if self.piper is not None:
                # 停止运动
                print("[1/4] 停止机械臂运动...")
                self.send_stop_command()
                time.sleep(0.1)

                # 闭合夹爪
                print("[2/4] 闭合夹爪...")
                self.piper.GripperCtrl(0, 1000, 0x01, 0)
                time.sleep(0.2)

                # 失能机械臂
                print("[3/4] 失能机械臂...")
                self.piper.DisableArm(7)
                time.sleep(0.1)

                # 断开连接
                print("[4/4] 断开CAN连接...")

        except Exception as e:
            print(f"[WARNING] 关闭过程出现异常: {e}")

        finally:
            if self.joystick is not None:
                pygame.quit()

            print("="*60)
            print("系统已安全关闭")
            print("="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="[DANGEROUS] Xbox控制器控制Piper机械臂 - 预设工作位置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python3 xbox_arm_controller_dangerous.py              # 默认使用 can0
    python3 xbox_arm_controller_dangerous.py --can can1   # 指定 CAN 端口
    python3 xbox_arm_controller_dangerous.py -c can_piper # 简写形式
        """
    )
    parser.add_argument(
        "-c", "--can",
        type=str,
        default="can0",
        help="CAN 端口名称 (默认: can0)"
    )
    args = parser.parse_args()

    print(f"[INFO] 使用 CAN 端口: {args.can}")
    controller = XboxArmController(args.can)

    if controller.initialize():
        controller.run()
    else:
        print("[FAILED] 系统初始化失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
