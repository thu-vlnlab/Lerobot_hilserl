import sys
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from Robotic_Arm.rm_robot_interface import *
from teleop_config import *
import numpy as np
from convert_axis import *

from spacemouse_input import SpaceMouseReader
from rm75b import RM75BInterface

class RealmanSpacemouseTeleop(Node):
    def __init__(self):
        super().__init__('realman_spacemouse_teleop')
        
        # ==========================================
        # 1. 声明参数与常量
        # ==========================================
        self.arm_ip = ARMIP
        self.arm_port = ARMPORT
        self.linear_speed = 1e-6  # m/cycle
        self.angular_speed = 0.005  # rad/cycle
        self.deadband = 30
        self.control_rate = CONTROLRATE   # Hz
        
        # 维护当前目标位姿 [x, y, z, rx, ry, rz]
        self.target_pose = [0.0] * 6

        # target pose(action)
        self.target_pub = self.create_publisher(Float32MultiArray, '/action', 10)
        # current pose(state)
        self.actual_pub = self.create_publisher(Float32MultiArray, '/state', 10)

        # ==========================================
        # Initialization hardware
        # ==========================================
        self.init_spacemouse()
        self.init_robot_arm()
        
        # set arm current pose to start pose
        self.align_initial_pose()

        # ==========================================
        # Control loop
        # ==========================================
        timer_period = 1.0 / self.control_rate
        self.timer = self.create_timer(timer_period, self.control_loop)
        
        self.get_logger().info("✅ 节点初始化完成，开始遥操作与位姿广播...")

    def init_spacemouse(self):
        """初始化 SpaceMouse"""
        self.mouse = SpaceMouseReader()
        self.mouse.open()
        self.mouse.start()

    def init_robot_arm(self):
        """初始化睿尔曼机械臂"""
        self.arm = RM75BInterface(self.arm_ip, self.arm_port, enable_gripper=False)
        res, state = self.arm.arm.rm_get_current_arm_state()
        if res != 0:
            raise RuntimeError(f"Failed to read arm state (ret={res})")
        self.get_logger().info("✅ 睿尔曼机械臂 连接成功")

    def align_initial_pose(self):
        """获取当前机械臂末端位姿，并赋予目标位姿作基准"""
        res, state = self.arm.arm.rm_get_current_arm_state()
        if res != 0:
            self.get_logger().error("❌ 无法获取机械臂当前位姿，初始化失败！")
            self.cleanup()
            sys.exit(1)
        
        # 将元组/特殊结构转换为标准的 List [x, y, z, rx, ry, rz]
        self.target_pose = np.array(state["pose"], dtype=float)
        self.get_logger().info(f"📍 获取初始基准位姿完成: {self.target_pose}")

    def control_loop(self):
        """主控制循环定时器回调 (执行频率: control_rate)"""
        # 1. 读取 SpaceMouse 数据 (非阻塞)
        raw = np.array(self.mouse.get_axes())
        raw = np.array(convert_coordinate_system(raw,apply_extra_rotation=True))[:3]
        
        #TODO:deadboard
        self.target_pose[:3] += raw * self.linear_speed
        self.arm.arm.rm_movep_canfd(self.target_pose.tolist(), follow=False)

        # 3. 获取机械臂当前实际位姿
        res, actual_state = self.arm.arm.rm_get_current_arm_state()
        actual_state = np.array(actual_state['pose'],dtype=float)
        # if res == 0:
            
        # 4. 通过 ROS 2 广播消息
        
        target_msg = Float32MultiArray()
        target_msg.data = self.target_pose[:3].tolist()
        self.target_pub.publish(target_msg)
        
        action_msg = Float32MultiArray()
        action_msg.data = actual_state[:3].tolist()
        self.actual_pub.publish(action_msg)

    
    def cleanup(self):
        """安全清理资源"""
        self.get_logger().info("🛑 正在清理资源并断开连接...")
        self.get_logger().info("🔌 设备已安全断开")


def main(args=None):
    rclpy.init(args=args)
    teleop_node = RealmanSpacemouseTeleop()
    
    try:
        # 启动 ROS 2 事件循环
        rclpy.spin(teleop_node)
    except KeyboardInterrupt:
        teleop_node.get_logger().info("检测到 Ctrl+C，正在退出...")
    finally:
        teleop_node.cleanup()
        teleop_node.destroy_node()
        # 确保 rclpy 正确关闭
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()