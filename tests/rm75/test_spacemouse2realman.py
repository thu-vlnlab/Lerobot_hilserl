from lerobot.robots.rm75_follower import RM75FollowerEndEffector, RM75FollowerEndEffectorConfig
from lerobot.teleoperators.spacemouse import SpaceMouseTeleop, SpaceMouseTeleopConfig

robot_config = RM75FollowerEndEffectorConfig(robot_ip="192.168.1.18")
robot = RM75FollowerEndEffector(robot_config)
robot.connect()

teleop_config = SpaceMouseTeleopConfig()
teleop = SpaceMouseTeleop(teleop_config)
teleop.connect()

import time

# 用当前位置初始化目标，之后只用 delta 累加，不再每次读回
obs = robot.get_observation()
target_x = obs["ee.x"]
target_y = obs["ee.y"]
target_z = obs["ee.z"]
gripper_pos = obs["gripper.pos"]

print("移动 SpaceMouse 控制机械臂... Ctrl+C 退出")
try:
    while True:
        action = teleop.get_action()

        # 累加 delta 到内部目标（不从当前位置读回）
        target_x += action["delta_x"] * 0.001
        target_y += action["delta_y"] * 0.001
        target_z += action["delta_z"] * 0.001

        # 夹爪控制
        g = action.get("gripper", 1)
        if g == 0:    # close
            gripper_pos = max(0, gripper_pos - 50)
        elif g == 2:  # open
            gripper_pos = min(1000, gripper_pos + 50)

        target = {
            "ee.x": target_x,
            "ee.y": target_y,
            "ee.z": target_z,
            "gripper.pos": gripper_pos,
        }

        robot.send_action(target)
        time.sleep(0.02)
except KeyboardInterrupt:
    pass

teleop.disconnect()
robot.disconnect()
