#!/usr/bin/env python3
"""
直接测试 Piper 机器人 + 手柄控制
绕过 LeRobot 的复杂 processor pipeline，直接验证功能
"""

import time
import sys

# 添加项目路径
sys.path.insert(0, "/home/ubuntu/Lerobot_hilserl/src")

import pygame
from piper_sdk import C_PiperInterface_V2


class SimplePiperGamepadTest:
    """简单的 Piper + 手柄测试"""

    def __init__(self, can_name: str = "can0"):
        self.can_name = can_name
        self.piper = None
        self.joystick = None

        # 控制参数
        self.LINEAR_VELOCITY_SCALE = 50.0  # mm/s
        self.DEADZONE = 0.1
        self.GRIPPER_STEP = 1500  # 每次增量

        # 状态
        self.current_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # X, Y, Z, RX, RY, RZ (mm, deg)
        self.gripper_pos = 35000  # 0.001mm 单位

        # 初始关节位置 (0.001度)
        self.INIT_JOINTS = [0, 77140, -56995, 0, 77728, 0]

    def connect_gamepad(self) -> bool:
        """连接手柄"""
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            print("[ERROR] 没有检测到手柄!")
            return False

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"[OK] 手柄已连接: {self.joystick.get_name()}")
        return True

    def connect_robot(self) -> bool:
        """连接机器人"""
        try:
            self.piper = C_PiperInterface_V2(self.can_name)
            self.piper.ConnectPort()
            time.sleep(0.5)

            # 使能
            for i in range(10):
                self.piper.EnableArm(7)
                time.sleep(0.5)
                low_spd = self.piper.GetArmLowSpdInfoMsgs()
                status = [
                    low_spd.motor_1.foc_status.driver_enable_status,
                    low_spd.motor_2.foc_status.driver_enable_status,
                    low_spd.motor_3.foc_status.driver_enable_status,
                    low_spd.motor_4.foc_status.driver_enable_status,
                    low_spd.motor_5.foc_status.driver_enable_status,
                    low_spd.motor_6.foc_status.driver_enable_status,
                ]
                print(f"[INFO] 使能尝试 {i+1}/10: 电机状态={status}")
                if all(status):
                    print("[OK] 机器人已使能")
                    break
            else:
                print("[ERROR] 机器人使能超时，请尝试:")
                print("  1. sudo ip link set can0 down && sudo ip link set can0 up type can bitrate 1000000")
                print("  2. 或者给机器人断电重启")
                return False

            # 初始化夹爪
            self.piper.GripperCtrl(self.gripper_pos, 1000, 0x01, 0)
            time.sleep(0.1)

            return True
        except Exception as e:
            print(f"[ERROR] 机器人连接失败: {e}")
            return False

    def move_to_initial_position(self):
        """移动到初始位置 - 使用关节控制"""
        print("[INFO] 移动到初始位置...")

        # 切换到关节控制模式
        self.piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)
        time.sleep(0.1)

        # 发送关节目标
        self.piper.JointCtrl(
            self.INIT_JOINTS[0],
            self.INIT_JOINTS[1],
            self.INIT_JOINTS[2],
            self.INIT_JOINTS[3],
            self.INIT_JOINTS[4],
            self.INIT_JOINTS[5]
        )

        # 等待到达
        target_deg = [j / 1000.0 for j in self.INIT_JOINTS]
        for i in range(100):
            time.sleep(0.1)
            joint = self.piper.GetArmJointMsgs()
            current = [
                joint.joint_state.joint_1 / 1000.0,
                joint.joint_state.joint_2 / 1000.0,
                joint.joint_state.joint_3 / 1000.0,
                joint.joint_state.joint_4 / 1000.0,
                joint.joint_state.joint_5 / 1000.0,
                joint.joint_state.joint_6 / 1000.0,
            ]
            if all(abs(current[j] - target_deg[j]) < 2.0 for j in range(6)):
                print(f"[OK] 已到达初始位置")
                break
            if i % 10 == 0:
                print(f"[INFO] 移动中... J1={current[0]:.1f}° J2={current[1]:.1f}°")

        # 切换到末端控制模式
        self.piper.MotionCtrl_2(0x01, 0x00, 50, 0x00)
        time.sleep(0.2)

        # 读取当前末端位姿
        end_pose = self.piper.GetArmEndPoseMsgs()
        self.current_pose = [
            end_pose.end_pose.X_axis / 1000.0,  # 0.001mm -> mm
            end_pose.end_pose.Y_axis / 1000.0,
            end_pose.end_pose.Z_axis / 1000.0,
            end_pose.end_pose.RX_axis / 1000.0,  # 0.001deg -> deg
            end_pose.end_pose.RY_axis / 1000.0,
            end_pose.end_pose.RZ_axis / 1000.0,
        ]
        print(f"[OK] 当前位姿: X={self.current_pose[0]:.1f}mm Y={self.current_pose[1]:.1f}mm Z={self.current_pose[2]:.1f}mm")

    def apply_deadzone(self, value: float) -> float:
        """应用死区"""
        if abs(value) < self.DEADZONE:
            return 0.0
        sign = 1 if value > 0 else -1
        return sign * (abs(value) - self.DEADZONE) / (1.0 - self.DEADZONE)

    def read_gamepad(self) -> dict:
        """读取手柄输入"""
        pygame.event.pump()

        left_x = self.apply_deadzone(self.joystick.get_axis(0))
        left_y = self.apply_deadzone(self.joystick.get_axis(1))
        right_y = self.apply_deadzone(self.joystick.get_axis(4))

        # LB 按钮作为使能
        lb_pressed = self.joystick.get_button(4)

        # 方向键控制夹爪
        hat = self.joystick.get_hat(0) if self.joystick.get_numhats() > 0 else (0, 0)

        return {
            'left_x': left_x,
            'left_y': left_y,
            'right_y': right_y,
            'lb': lb_pressed,
            'dpad_up': hat[1] == 1,
            'dpad_down': hat[1] == -1,
        }

    def control_loop(self):
        """主控制循环"""
        print("\n" + "="*60)
        print("控制说明:")
        print("  按住 LB 键使能控制")
        print("  左摇杆 Y: X方向 (前后)")
        print("  左摇杆 X: Y方向 (左右)")
        print("  右摇杆 Y: Z方向 (上下)")
        print("  十字键 上/下: 夹爪开/合")
        print("  Ctrl+C 退出")
        print("="*60 + "\n")

        dt = 1.0 / 50  # 50Hz

        try:
            while True:
                start_time = time.time()

                inputs = self.read_gamepad()

                if inputs['lb']:
                    # 计算速度 (mm/s)
                    vel_x = -inputs['left_y'] * self.LINEAR_VELOCITY_SCALE
                    vel_y = inputs['left_x'] * self.LINEAR_VELOCITY_SCALE
                    vel_z = -inputs['right_y'] * self.LINEAR_VELOCITY_SCALE

                    # 更新位置
                    self.current_pose[0] += vel_x * dt
                    self.current_pose[1] += vel_y * dt
                    self.current_pose[2] += vel_z * dt

                    # 发送命令
                    X = int(self.current_pose[0] * 1000)
                    Y = int(self.current_pose[1] * 1000)
                    Z = int(self.current_pose[2] * 1000)
                    RX = int(self.current_pose[3] * 1000)
                    RY = int(self.current_pose[4] * 1000)
                    RZ = int(self.current_pose[5] * 1000)

                    self.piper.MotionCtrl_2(0x01, 0x00, 50, 0x00)
                    self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)

                    print(f"\r[控制中] X={self.current_pose[0]:.1f} Y={self.current_pose[1]:.1f} Z={self.current_pose[2]:.1f} vel=({vel_x:.1f}, {vel_y:.1f}, {vel_z:.1f})   ", end="")
                else:
                    # 读取当前位置保持同步
                    end_pose = self.piper.GetArmEndPoseMsgs()
                    self.current_pose = [
                        end_pose.end_pose.X_axis / 1000.0,
                        end_pose.end_pose.Y_axis / 1000.0,
                        end_pose.end_pose.Z_axis / 1000.0,
                        end_pose.end_pose.RX_axis / 1000.0,
                        end_pose.end_pose.RY_axis / 1000.0,
                        end_pose.end_pose.RZ_axis / 1000.0,
                    ]
                    print(f"\r[等待LB] X={self.current_pose[0]:.1f} Y={self.current_pose[1]:.1f} Z={self.current_pose[2]:.1f}   ", end="")

                # 夹爪控制
                if inputs['dpad_up']:
                    self.gripper_pos = min(80000, self.gripper_pos + self.GRIPPER_STEP)
                    self.piper.GripperCtrl(self.gripper_pos, 1000, 0x01, 0)
                elif inputs['dpad_down']:
                    self.gripper_pos = max(0, self.gripper_pos - self.GRIPPER_STEP)
                    self.piper.GripperCtrl(self.gripper_pos, 1000, 0x01, 0)

                # 保持频率
                elapsed = time.time() - start_time
                if elapsed < dt:
                    time.sleep(dt - elapsed)

        except KeyboardInterrupt:
            print("\n\n[INFO] 退出控制，保持当前位置...")
            # 发送当前位置保持，不要失能
            if self.piper:
                X = int(self.current_pose[0] * 1000)
                Y = int(self.current_pose[1] * 1000)
                Z = int(self.current_pose[2] * 1000)
                RX = int(self.current_pose[3] * 1000)
                RY = int(self.current_pose[4] * 1000)
                RZ = int(self.current_pose[5] * 1000)
                self.piper.MotionCtrl_2(0x01, 0x00, 10, 0x00)
                self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)

    def run(self):
        """运行测试"""
        if not self.connect_gamepad():
            return
        if not self.connect_robot():
            return

        self.move_to_initial_position()
        self.control_loop()

        # 清理 - 不要 DisableArm，保持使能状态
        if self.piper:
            # self.piper.DisableArm(7)  # 注释掉，防止突然掉落
            self.piper.DisconnectPort()
        pygame.quit()
        print("[INFO] 已断开连接，机器人保持使能状态")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--can", default="can0", help="CAN 接口名称")
    args = parser.parse_args()

    test = SimplePiperGamepadTest(can_name=args.can)
    test.run()
