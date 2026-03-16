# HIL-SERL for Piper Robot Arm

本指南介绍如何使用 LeRobot 的 HIL-SERL（Human-in-the-Loop Sample Efficient Reinforcement Learning）训练 Piper 机械臂完成抓取任务。

## 目录

1. [环境准备](#环境准备)
2. [硬件设置](#硬件设置)
3. [数据收集](#数据收集)
4. [训练奖励分类器](#训练奖励分类器)
5. [HIL-SERL 训练](#hil-serl-训练)
6. [Gamepad 控制说明](#gamepad-控制说明)

## 环境准备

### 1. 安装依赖

```bash
# 克隆仓库
git clone git@github.com:thu-vlnlab/Lerobot_hilserl.git
cd Lerobot_hilserl

# 创建虚拟环境
conda create -n lerobot python=3.10 -c conda-forge
conda activate lerobot

# 安装 LeRobot 及 HIL-SERL 依赖
pip install -e ".[hilserl]"

# 安装 Piper SDK
pip install -e /path/to/piper_sdk
```

### 2. 设置 CAN 接口

Piper 机械臂通过 CAN 总线通信，需要先设置 CAN 接口：

```bash
# 设置 CAN 接口（波特率 1Mbps）
sudo ip link set can0 type can bitrate 1000000
sudo ip link set up can0

# 验证 CAN 接口
ip link show can0
```

## 硬件设置

### 所需设备

- Piper 6-DOF 机械臂
- USB 摄像头（用于视觉观测）
- Xbox/PS4 手柄（用于遥操作和干预）
- CAN 转 USB 适配器

### 摄像头配置

在 `configs_hilserl/env_config_piper.json` 中配置摄像头：

```json
"cameras": {
  "observation.images.front": {
    "type": "opencv",
    "index_or_path": 0,  // 摄像头索引或路径
    "width": 640,
    "height": 480,
    "fps": 30
  }
}
```

## 数据收集

### 1. 确定工作空间边界

首先需要确定机械臂的工作空间边界，以限制探索范围：

```bash
# 使用手柄或遥操作移动机械臂，记录末端执行器的位置范围
./scripts/run_hilserl.sh test-robot
```

将边界值更新到配置文件的 `end_effector_bounds` 中。

### 2. 收集演示数据

使用 Gamepad 手柄收集演示数据：

```bash
./scripts/run_hilserl.sh record
```

控制方式：
- **左摇杆**: X/Y 方向移动
- **右摇杆上下**: Z 方向移动
- **RT 扳机**: 闭合夹爪
- **LT 扳机**: 打开夹爪
- **A 按钮**: 标记成功（结束当前 episode）
- **B 按钮**: 标记失败
- **X 按钮**: 重新录制当前 episode

### 3. 裁剪图像 ROI

收集完演示后，使用交互式工具裁剪图像感兴趣区域：

```bash
./scripts/run_hilserl.sh crop thu-vlnlab/piper_pick_lift
```

将输出的裁剪参数更新到 `crop_params_dict` 配置中。

## 训练奖励分类器

奖励分类器用于自动检测任务是否成功：

### 1. 收集奖励分类器训练数据

需要收集包含成功和失败状态的数据。修改配置中的 `terminate_on_success` 为 `false`，这样可以在任务成功后继续收集数据。

### 2. 训练分类器

```bash
./scripts/run_hilserl.sh train-reward
```

训练完成后，模型保存在 `outputs/reward_classifier/` 目录。

## HIL-SERL 训练

HIL-SERL 使用 Actor-Learner 架构，需要启动两个进程：

### 终端 1：启动 Learner

```bash
./scripts/run_hilserl.sh train-learner
```

### 终端 2：启动 Actor

```bash
./scripts/run_hilserl.sh train-actor
```

### 人工干预策略

训练过程中可以随时进行人工干预：

1. **初期（0-1000 步）**: 频繁干预，帮助策略学习基本动作
2. **中期（1000-5000 步）**: 在策略偏离时干预
3. **后期（5000+ 步）**: 仅在危险情况下干预

按 **RB 按钮** 进入干预模式，再次按 **RB** 退出干预。

## Gamepad 控制说明

| 按钮/摇杆 | 功能 |
|-----------|------|
| 左摇杆 X | 末端执行器 X 方向 |
| 左摇杆 Y | 末端执行器 Y 方向 |
| 右摇杆 Y | 末端执行器 Z 方向 |
| RT | 闭合夹爪 |
| LT | 打开夹爪 |
| A | 成功/结束 episode |
| B | 失败 |
| X | 重新录制 |
| RB | 进入/退出干预模式 |
| Start | 重置机械臂 |

## 监控训练

如果启用了 WandB，可以在 https://wandb.ai 上实时监控训练进度：

- `episodic_reward`: 每个 episode 的奖励
- `intervention_rate`: 人工干预率（应该逐渐下降）
- `success_rate`: 任务成功率

## 常见问题

### CAN 连接失败

```bash
# 检查 CAN 接口状态
ip link show can0

# 重新设置 CAN 接口
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

### 机械臂不响应

1. 检查机械臂电源
2. 检查 CAN 线连接
3. 确认 CAN 端口名称正确（默认 `can0`）

### Gamepad 未检测到

```bash
# 检查连接的游戏控制器
ls /dev/input/js*

# 使用 jstest 测试
jstest /dev/input/js0
```

## 参考资料

- [LeRobot HIL-SERL 官方文档](https://huggingface.co/docs/lerobot/en/hilserl)
- [Piper SDK 文档](https://github.com/agilexrobotics/piper_sdk)
- [HIL-SERL 论文](https://hil-serl.github.io/)
