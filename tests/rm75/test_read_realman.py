from lerobot.robots.rm75_follower import RM75FollowerEndEffector, RM75FollowerEndEffectorConfig

config = RM75FollowerEndEffectorConfig(
    robot_ip="192.168.1.18",
    robot_port=8080,
)
robot = RM75FollowerEndEffector(config)
robot.connect()

# 读取观测
obs = robot.get_observation()
print(obs)
# print(f"关节角度: {[obs[f'joint_{i+1}.pos'] for i in range(7)]}")
# print(f"末端位置: x={obs['ee.x']:.4f}, y={obs['ee.y']:.4f}, z={obs['ee.z']:.4f}")
# print(f"末端姿态: rx={obs['ee.rx']:.4f}, ry={obs['ee.ry']:.4f}, rz={obs['ee.rz']:.4f}")
# print(f"夹爪位置: {obs['gripper.pos']}")

robot.disconnect()