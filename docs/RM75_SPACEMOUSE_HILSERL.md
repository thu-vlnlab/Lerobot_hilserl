# RM75-B + SpaceMouse HIL-SERL 使用指南

使用睿尔曼 RM75-B（7-DOF）机械臂和 3Dconnexion SpaceMouse 进行 HIL-SERL 训练。

---

## 目录

1. [系统概览](#系统概览)
2. [硬件要求](#硬件要求)
3. [环境准备](#环境准备)
4. [连接测试](#连接测试)
5. [数据采集](#数据采集)
6. [训练 Reward Classifier](#训练-reward-classifier)
7. [HIL-SERL 训练](#hil-serl-训练)
8. [SpaceMouse 控制说明](#spacemouse-控制说明)
9. [配置参考](#配置参考)
10. [常见问题](#常见问题)

---

## 系统概览

### 当前能力（v1）

| 功能 | 状态 | 说明 |
|------|------|------|
| 末端位置控制 (x/y/z) | 已实现 | 通过 `rm_movep_canfd` 笛卡尔控制 |
| 夹爪控制 | 已实现 | ZhiXing 90D，Modbus RTU，范围 0~1000 |
| 姿态控制 (rx/ry/rz) | 锁定 | 首次调用时锁定当前姿态，不开放控制 |
| 7-DOF 关节观测 | 已实现 | 实时读取 7 个关节角度 |
| 末端位姿观测 | 已实现 | 实时读取 [x, y, z, rx, ry, rz] |
| 单相机视觉 | 已实现 | OpenCV 摄像头 |
| SpaceMouse 遥操作 | 已实现 | 3 轴平移 + 2 按钮夹爪控制 |
| 人工干预检测 | 已实现 | SpaceMouse 自归中，松手即停止干预 |
| 关节空间控制 | 已实现 | `rm75_follower` 模式，7 关节位置控制 |
| HIL-SERL 数据采集 | 已实现 | record 模式，兼容 LeRobotDataset |
| HIL-SERL 训练 | 已实现 | Actor-Learner 架构 |

### 架构图

```
SpaceMouse ──→ SpaceMouseTeleop ──→ delta_x/y/z + gripper
                                         │
                                         ▼
                              PiperDeltaToAbsoluteEEStep
                              (delta → 绝对位置，workspace 裁剪)
                                         │
                                         ▼
                            RM75FollowerEndEffector.send_action()
                            (锁定姿态 + movep_canfd)
                                         │
                                         ▼
                               RM75BInterface (TCP/IP)
                                    │         │
                                    ▼         ▼
                              关节/EE控制   Modbus 夹爪
```

---

## 硬件要求

| 设备 | 型号 | 接口 |
|------|------|------|
| 机械臂 | 睿尔曼 RM75-B (7-DOF) | 以太网 TCP/IP |
| 夹爪 | 知行 ZhiXing 90D | Modbus RTU（末端 RS485） |
| 遥操作设备 | 3Dconnexion SpaceMouse | USB（通过 spacenavd） |
| 摄像头 | USB 摄像头 | USB（OpenCV） |
| 计算机 | Linux x86_64 / aarch64 | - |

---

## 环境准备

### 1. 安装基础环境

```bash
# 克隆仓库
git clone git@github.com:thu-vlnlab/Lerobot_hilserl.git
cd Lerobot_hilserl

# 创建虚拟环境
conda create -n lerobot python=3.10 -c conda-forge
conda activate lerobot

# 安装 LeRobot 及 HIL-SERL 依赖
pip install -e ".[hilserl]"
```

### 2. 安装 RM75-B SDK

```bash
# 安装睿尔曼 Python SDK
pip install Robotic-Arm
```

### 3. 安装 SpaceMouse 驱动

```bash
# Ubuntu / Debian
sudo apt install libspnav-dev spacenavd

# 启动 spacenavd 守护进程
sudo systemctl enable spacenavd
sudo systemctl start spacenavd

# 验证 SpaceMouse 连接
spnavtest
```

### 4. 网络配置

RM75-B 默认 IP 为 `192.168.5.18`，需要将计算机网口配置在同一子网：

```bash
# 设置静态 IP（示例）
sudo ip addr add 192.168.5.100/24 dev eth0
sudo ip link set eth0 up

# 验证连接
ping 192.168.5.18
```

---

## 连接测试

### 测试机械臂连接

```python
from lerobot.robots.rm75_follower import RM75FollowerEndEffector, RM75FollowerEndEffectorConfig

config = RM75FollowerEndEffectorConfig(
    robot_ip="192.168.1.18",
    robot_port=8080,
)
robot = RM75FollowerEndEffector(config)
robot.connect()

# 读取观测
obs = robot.get_observation()
print(f"关节角度: {[obs[f'joint_{i+1}.pos'] for i in range(7)]}")
print(f"末端位置: x={obs['ee.x']:.4f}, y={obs['ee.y']:.4f}, z={obs['ee.z']:.4f}")
print(f"末端姿态: rx={obs['ee.rx']:.4f}, ry={obs['ee.ry']:.4f}, rz={obs['ee.rz']:.4f}")
print(f"夹爪位置: {obs['gripper.pos']}")

robot.disconnect()
```

### 测试 SpaceMouse

```python
from lerobot.teleoperators.spacemouse import SpaceMouseTeleop, SpaceMouseTeleopConfig
from lerobot.teleoperators.utils import TeleopEvents

config = SpaceMouseTeleopConfig()
teleop = SpaceMouseTeleop(config)
teleop.connect()

import time
for _ in range(100):
    action = teleop.get_action()
    events = teleop.get_teleop_events()
    print(f"delta: ({action['delta_x']:.4f}, {action['delta_y']:.4f}, {action['delta_z']:.4f})"
          f"  gripper: {action.get('gripper', '-')}"
          f"  intervening: {events[TeleopEvents.IS_INTERVENTION]}")
    time.sleep(0.02)

teleop.disconnect()
```

### 测试遥操作（SpaceMouse → RM75）

```python
from lerobot.robots.rm75_follower import RM75FollowerEndEffector, RM75FollowerEndEffectorConfig
from lerobot.teleoperators.spacemouse import SpaceMouseTeleop, SpaceMouseTeleopConfig

robot_config = RM75FollowerEndEffectorConfig(robot_ip="192.168.1.18")
robot = RM75FollowerEndEffector(robot_config)
robot.connect()

teleop_config = SpaceMouseTeleopConfig()
teleop = SpaceMouseTeleop(teleop_config)
teleop.connect()

import time
print("移动 SpaceMouse 控制机械臂... Ctrl+C 退出")
try:
    while True:
        action = teleop.get_action()
        obs = robot.get_observation()

        # 将 delta 转为绝对位置
        target = {
            "ee.x": obs["ee.x"] + action["delta_x"] * 0.001,
            "ee.y": obs["ee.y"] + action["delta_y"] * 0.001,
            "ee.z": obs["ee.z"] + action["delta_z"] * 0.001,
            "gripper.pos": obs["gripper.pos"],
        }

        # 夹爪控制
        g = action.get("gripper", 1)
        if g == 0:    # close
            target["gripper.pos"] = max(0, obs["gripper.pos"] - 50)
        elif g == 2:  # open
            target["gripper.pos"] = min(1000, obs["gripper.pos"] + 50)

        robot.send_action(target)
        time.sleep(0.02)
except KeyboardInterrupt:
    pass

teleop.disconnect()
robot.disconnect()
```

---

## 数据采集

### 1. 创建配置文件

创建 `configs_hilserl/env_config_rm75_real.json`：

```json
{
    "env": {
        "type": "real",
        "robot": {
            "type": "rm75_follower_ee",
            "robot_ip": "192.168.1.18",
            "robot_port": 8080,
            "enable_gripper": true,
            "workspace_bounds": {
                "min": [-0.5, -0.5, 0.0],
                "max": [0.5, 0.5, 0.7]
            },
            "cameras": {
                "front": {
                    "type": "opencv",
                    "index_or_path": 0,
                    "width": 640,
                    "height": 480,
                    "fps": 30
                }
            }
        },
        "teleoperator": {
            "type": "spacemouse",
            "deadzone": 40,
            "translation_scale": 0.0032,
            "use_gripper": true
        }
    },
    "dataset": {
        "repo_id": "thu-vlnlab/rm75_pick_lift",
        "root": "data/rm75_demos",
        "num_episodes_to_record": 10
    },
    "mode": "record"
}
```

### 2. 运行数据采集

```bash
python -m lerobot.rl.gym_manipulator \
    --config_path configs_hilserl/env_config_rm75_real.json
```

### 3. 采集流程

1. 程序启动后，机械臂回到初始位置
2. 移动 SpaceMouse 控制机械臂完成任务
3. 按键盘 `Enter` 标记成功并结束当前 episode
4. 机械臂自动 reset，开始下一个 episode
5. 重复直到采集完指定数量

> **注意**：SpaceMouse 只有 2 个按钮，episode 事件（成功/失败/重录）需要通过键盘补充处理。

---

## 训练 Reward Classifier

```bash
python -m lerobot.rl.train_reward_classifier \
    --config_path configs_hilserl/reward_classifier_config_rm75.json
```

配置文件中需要指向采集的数据路径。

---

## HIL-SERL 训练

### 终端 1 — Learner

```bash
conda activate lerobot
python -m lerobot.rl.learner \
    --config_path configs_hilserl/train_config_hilserl_rm75_real.json
```

### 终端 2 — Actor

```bash
conda activate lerobot
python -m lerobot.rl.actor \
    --config_path configs_hilserl/train_config_hilserl_rm75_real.json
```

### 干预方式

- **推动 SpaceMouse** = 进入干预模式（自动检测，任一平移轴非零即触发）
- **松开 SpaceMouse** = 自动退出干预（SpaceMouse 自归中，所有轴归零）
- 无需按按钮切换干预状态，比 Gamepad 更自然

---

## SpaceMouse 控制说明

### 轴映射

SpaceMouse 的 6 个轴经过重映射后对应：

| SpaceMouse 动作 | 机器人方向 | 说明 |
|-----------------|-----------|------|
| 前推/后拉 | X 轴 (前/后) | 经 axis_map 重映射 |
| 左推/右推 | Y 轴 (左/右) | 经 axis_map 重映射 |
| 上提/下压 | Z 轴 (上/下) | 经 axis_map 重映射 |
| 旋转 | 锁定（不使用） | v1 不开放姿态控制 |

### 按钮

| 按钮 | 功能 |
|------|------|
| 左键 (button 0) | 闭合夹爪 |
| 右键 (button 1) | 打开夹爪 |
| 不按 | 夹爪保持当前位置 |

### 缩放调节

如果移动速度太快或太慢，调整 `translation_scale`：

```json
{
    "teleoperator": {
        "translation_scale": 0.0032   // 增大 = 更快，减小 = 更精细
    }
}
```

### 轴方向反转

如果某个方向反了，修改 `axis_signs`：

```json
{
    "teleoperator": {
        "axis_signs": [1, -1, 1, -1, 1, 1]  // 改对应位置的正负号
    }
}
```

---

## 配置参考

### RM75FollowerEndEffectorConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `robot_ip` | `"192.168.5.18"` | 机械臂 IP 地址 |
| `robot_port` | `8080` | 机械臂端口 |
| `enable_gripper` | `true` | 是否启用夹爪 |
| `gripper_open_pos` | `1000` | 夹爪全开位置 (0~1000) |
| `gripper_close_pos` | `0` | 夹爪全闭位置 |
| `workspace_bounds.min` | `[-0.5, -0.5, 0.0]` | 工作空间下界 [x, y, z] 米 |
| `workspace_bounds.max` | `[0.5, 0.5, 0.7]` | 工作空间上界 [x, y, z] 米 |
| `max_translation_per_cycle` | `0.001` | 每周期最大平移量（米） |
| `lock_orientation` | `true` | 锁定末端姿态 |
| `home_joint_positions` | `[0,...,0]` | 7 个关节 home 位置（度） |
| `reset_speed` | `20` | reset 运动速度 (0~100%) |
| `cameras` | `{}` | 相机配置字典 |

### SpaceMouseTeleopConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `libspnav_path` | `null` | libspnav 路径（null=自动查找） |
| `deadzone` | `40` | 轴死区（原始范围 ~+-350） |
| `use_gripper` | `true` | 按钮控制夹爪 |
| `axis_map` | `[2, 0, 1, 5, 3, 4]` | 轴重映射 |
| `axis_signs` | `[1, -1, 1, -1, 1, 1]` | 轴正负号 |
| `translation_scale` | `0.0032` | 平移缩放系数 |
| `gripper_close_button` | `0` | 闭合夹爪按钮 |
| `gripper_open_button` | `1` | 打开夹爪按钮 |

---

## 常见问题

### SpaceMouse 连接失败

```bash
# 检查 spacenavd 是否运行
sudo systemctl status spacenavd

# 重启 spacenavd
sudo systemctl restart spacenavd

# 检查 USB 设备
lsusb | grep -i 3dconnexion

# 查看日志
journalctl -u spacenavd -f
```

### 找不到 libspnav

```bash
# 查找库路径
find / -name "libspnav*" 2>/dev/null

# 手动指定路径
{
    "teleoperator": {
        "libspnav_path": "/usr/lib/x86_64-linux-gnu/libspnav.so.0"
    }
}
```

### RM75-B 连接超时

```bash
# 检查网络连通性
ping 192.168.5.18

# 检查端口
nc -zv 192.168.5.18 8080

# 确认网卡在同一子网
ip addr show | grep 192.168.5
```

### 夹爪不动

1. 检查夹爪供电（末端 RS485 连接）
2. 检查 Modbus 地址是否为 1（ZhiXing 90D 默认地址）
3. 尝试重启机械臂电源

### 机械臂运动方向与 SpaceMouse 不一致

修改 `axis_map` 和 `axis_signs`。建议先运行 SpaceMouse 测试脚本观察原始轴值：

```python
from lerobot.teleoperators.spacemouse.spacemouse_reader import SpaceMouseReader

reader = SpaceMouseReader()
reader.open()
reader.start()

import time
for _ in range(500):
    axes = reader.get_axes()
    print(f"\rT({axes[0]:5d},{axes[1]:5d},{axes[2]:5d})  R({axes[3]:5d},{axes[4]:5d},{axes[5]:5d})", end="")
    time.sleep(0.02)

reader.stop()
```

根据输出调整映射关系。

---

## 与 Piper 方案的对比

| 项目 | Piper | RM75-B |
|------|-------|--------|
| 自由度 | 6-DOF | 7-DOF |
| 通信 | CAN 总线 | TCP/IP 以太网 |
| EE 控制 | SDK `EndPoseCtrl` | SDK `rm_movep_canfd` |
| 夹爪 | SDK 原生 | Modbus RTU (ZhiXing 90D) |
| 遥操作 | Gamepad 手柄 | SpaceMouse 6DOF |
| 干预切换 | 按 LB 键 | 自动（松手即退出） |
| 关节观测 | 6 维 | 7 维 |
| Episode 事件 | Gamepad A/B/X 键 | 键盘补充 |

---

## 文件清单

```
src/lerobot/
├── robots/rm75_follower/
│   ├── __init__.py                 # 导出 + draccus 注册
│   ├── config_rm75_follower.py     # RM75FollowerConfig / RM75FollowerEndEffectorConfig
│   ├── rm75_follower.py            # RM75Follower / RM75FollowerEndEffector
│   └── rm75_interface.py           # RM75BInterface 硬件抽象
├── teleoperators/spacemouse/
│   ├── __init__.py                 # 导出
│   ├── configuration_spacemouse.py # SpaceMouseTeleopConfig
│   ├── spacemouse_reader.py        # SpaceMouseReader (libspnav)
│   └── teleop_spacemouse.py        # SpaceMouseTeleop
```

修改的现有文件：
- `robots/utils.py` — 工厂注册
- `robots/__init__.py` — 模块导入
- `teleoperators/utils.py` — 工厂注册
- `rl/gym_manipulator.py` — reset 逻辑 + EE processor 适配
