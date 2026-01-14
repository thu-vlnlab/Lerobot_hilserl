# Piper机械臂 HIL-SERL 训练完整流程

本文档详细记录了使用Piper 6-DOF机械臂进行HIL-SERL（Human-in-the-Loop Sample Efficient Reinforcement Learning）训练的完整操作流程。

## 目录

1. [环境准备](#1-环境准备)
2. [机械臂设置](#2-机械臂设置)
3. [数据采集](#3-数据采集)
4. [数据处理](#4-数据处理)
5. [训练Reward Classifier](#5-训练reward-classifier)
6. [HIL-SERL训练](#6-hil-serl训练)
7. [常见问题解决](#7-常见问题解决)

---

## 1. 环境准备

### 1.1 安装LeRobot

```bash
# 克隆仓库
cd /home/ubuntu
git clone <repo_url> Lerobot_hilserl
cd Lerobot_hilserl

# 创建conda环境
conda create -n lerobot python=3.10 -y
conda activate lerobot

# 安装LeRobot（包含HIL-SERL支持）
pip install -e ".[hilserl]"
```

### 1.2 安装Piper SDK

```bash
# Piper SDK路径
export PIPER_SDK_PATH=/home/ubuntu/agilex_ws/piper_ros/src/piper_sdk/piper_sdk

# 确保SDK在Python路径中
export PYTHONPATH=$PIPER_SDK_PATH:$PYTHONPATH
```

### 1.3 安装OpenCV（带GUI支持）

用于ROI裁剪的可视化界面：

```bash
conda activate lerobot

# 安装带GUI支持的OpenCV
conda install -c conda-forge py-opencv -y

# 如果存在冲突，删除pip安装的headless版本
rm -rf /home/ubuntu/miniconda3/envs/lerobot/lib/python3.10/site-packages/cv2

# 验证GUI支持
python -c "import cv2; cv2.namedWindow('test'); cv2.destroyAllWindows(); print('GUI OK!')"
```

### 1.4 安装FFmpeg（torchcodec需要）

```bash
# 安装FFmpeg 6（torchcodec不支持FFmpeg 8）
conda install -c conda-forge ffmpeg=6 -y

# 设置库路径
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

---

## 2. 机械臂设置

### 2.1 CAN接口配置

Piper机械臂通过CAN总线通信，每次使用前需要配置：

```bash
# 配置CAN接口（波特率1000000）
sudo ip link set can0 down
sudo ip link set can0 up type can bitrate 1000000

# 验证CAN接口状态
ip link show can0
```

### 2.2 测试机械臂连接

```bash
# 运行测试脚本
python tests/robots/test_piper_gamepad_control.py
```

或使用脚本：

```bash
./scripts/run_hilserl.sh test-robot
```

### 2.3 机械臂使能超时问题

如果遇到"机器人使能超时"错误，通常是因为上次程序未正常退出。解决方法：

```bash
# 重置CAN接口
sudo ip link set can0 down
sudo ip link set can0 up type can bitrate 1000000
```

### 2.4 重要配置说明

**防止机械臂突然失能**：确保配置文件中 `disable_torque_on_disconnect: false`

配置文件位置：`src/lerobot/robots/piper_follower/config_piper_follower.py`

```python
disable_torque_on_disconnect: bool = False  # 重要！防止Ctrl+C时机械臂突然掉电
```

---

## 3. 数据采集

### 3.1 手柄控制说明

使用Xbox Series X手柄控制机械臂：

| 按键/摇杆 | 功能 |
|-----------|------|
| LB (Button 4) | 人工干预标志 - 按住时记录为人工控制 |
| 左摇杆 上/下 | X轴移动（前进/后退） |
| 左摇杆 左/右 | Y轴移动（左/右） |
| 右摇杆 上/下 | Z轴移动（上/下） |
| A按钮 | 标记成功 |
| B按钮 | 标记失败 |
| Y按钮 | 重新录制当前episode |

### 3.2 配置文件

数据采集配置文件：`configs_hilserl/env_config_piper_real.json`

关键参数：
```json
{
  "env": {
    "type": "real",
    "fps": 10,
    "processor": {
      "reset": {
        "terminate_on_success": false  // 成功后继续录制，不立即终止
      }
    }
  }
}
```

### 3.3 运行数据采集

```bash
conda activate lerobot
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 方式1：使用脚本

python -m lerobot.rl.gym_manipulator --config configs_hilserl/env_config_piper_real.json
# 方式2：直接运行
python -m lerobot.rl.gym_manipulator \
    --config configs_hilserl/env_config_piper_real.json \
    --repo-id thu-vlnlab/piper_pick_lift \
    --root data/demos2 \
    --task "pick_and_lift"
```

### 3.4 数据采集流程

1. 确保CAN接口已配置
2. 连接Xbox手柄
3. 运行数据采集命令
4. 机械臂会自动移动到初始位置
5. 按住LB键，使用摇杆控制机械臂完成任务
6. 任务成功后按A键标记成功
7. 如需重新录制按Y键
8. Ctrl+C安全退出（机械臂会保持当前位置，不会失能）

---

## 4. 数据处理

### 4.1 ROI裁剪

将原始图像裁剪并resize到128x128，用于训练：

```bash
conda activate lerobot
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python -m lerobot.rl.crop_dataset_roi \
    --repo-id thu-vlnlab/piper_pick_lift \
    --root data/demos \
    --task "pick_and_lift"
```

这会打开GUI窗口，用鼠标拖拽选择ROI区域。处理后的数据保存在 `data/demos_cropped_resized/`。

### 4.2 验证数据集

检查数据集元数据：

```bash
cat data/demos_cropped_resized/meta/info.json
```

确保 `total_episodes` 和 `splits` 与实际数据一致。如果不一致，需要手动修复：

```python
import json

with open('data/demos_cropped_resized/meta/info.json', 'r') as f:
    info = json.load(f)

# 修改为实际的episode数量
info['total_episodes'] = 10  # 实际数量
info['splits'] = {'train': '0:10'}

with open('data/demos_cropped_resized/meta/info.json', 'w') as f:
    json.dump(info, f, indent=4)
```

---

## 5. 训练Reward Classifier

### 5.1 配置文件

配置文件位置：`configs_hilserl/reward_classifier_config_piper.json`

```json
{
  "dataset": {
    "repo_id": "thu-vlnlab/piper_pick_lift",
    "root": "data/demos_cropped_resized"
  },
  "policy": {
    "type": "reward_classifier",
    "model_name": "helper2424/resnet10",
    "model_type": "cnn",
    "num_cameras": 1,
    "num_classes": 2,
    "hidden_dim": 256,
    "dropout_rate": 0.1,
    "learning_rate": 1e-4,
    "device": "cuda",
    "use_amp": true,
    "push_to_hub": false,
    "input_features": {
      "observation.images.observation.images.front": {
        "type": "VISUAL",
        "shape": [3, 128, 128]
      }
    }
  },
  "output_dir": "outputs/reward_classifier_piper",
  "batch_size": 16,
  "steps": 3000,
  "eval_freq": 300,
  "save_freq": 500,
  "log_freq": 50,
  "wandb": {
    "enable": false,
    "project": "reward_classifier_piper"
  }
}
```

**注意**：
- `root` 使用 `data/demos_cropped_resized`（裁剪后的数据集）
- `shape` 必须是 `[3, 128, 128]`（ResNet10需要这个尺寸产生4x4的feature map）
- `input_features` 的key必须与数据集中的key完全匹配

### 5.2 运行训练

```bash
conda activate lerobot

python -m lerobot.scripts.lerobot_train \
    --config_path configs_hilserl/reward_classifier_config_piper.json
```

或使用脚本：

```bash
./scripts/run_hilserl.sh train-reward configs_hilserl/reward_classifier_config_piper.json
```

---

## 6. HIL-SERL训练

HIL-SERL使用Actor-Learner架构，需要在两个终端中同时运行。

### 6.1 配置文件

训练配置：`configs_hilserl/train_config_hilserl.json`

### 6.2 启动Learner（终端1）

```bash
conda activate lerobot

./scripts/run_hilserl.sh train-learner
# 或
python -m lerobot.rl.learner --config_path configs_hilserl/train_config_hilserl.json
```

### 6.3 启动Actor（终端2）

```bash
conda activate lerobot

./scripts/run_hilserl.sh train-actor
# 或
python -m lerobot.rl.actor --config_path configs_hilserl/train_config_hilserl.json
```

**重要**：必须先启动Learner，等待其初始化完成后再启动Actor。

---

## 7. 常见问题解决

### 7.1 机械臂使能超时

**错误**：`机器人使能超时`

**解决**：
```bash
sudo ip link set can0 down
sudo ip link set can0 up type can bitrate 1000000
```

### 7.2 机械臂突然失能

**问题**：Ctrl+C退出时机械臂突然掉电

**解决**：确保 `disable_torque_on_disconnect: false`

### 7.3 手柄无法控制机械臂

**检查项**：
1. 确认手柄已连接：`ls /dev/input/js*`
2. 确认使用LB按钮（button 4）作为干预键
3. 确认摇杆映射正确（参考 `src/lerobot/teleoperators/gamepad/gamepad_utils.py`）

### 7.4 OpenCV GUI不工作

**错误**：`The function is not implemented. Rebuild the library with GTK+ 2.x`

**解决**：
```bash
# 删除headless版本，使用conda的GUI版本
rm -rf $CONDA_PREFIX/lib/python3.10/site-packages/cv2
conda install -c conda-forge py-opencv -y
```

### 7.5 Episode索引越界

**错误**：`IndexError: Episode index X out of range`

**原因**：`info.json` 中的 `total_episodes` 与实际不符

**解决**：手动修复 `info.json` 中的 `total_episodes` 和 `splits`

### 7.6 Feature名称不匹配

**错误**：`Feature mismatch between dataset/environment and policy config`

**解决**：
1. 查看数据集的 `meta/info.json` 中的实际feature名称
2. 更新配置文件中的 `input_features` 使其完全匹配

### 7.7 Tensor维度不匹配

**错误**：`RuntimeError: The size of tensor a (X) must match the size of tensor b (4)`

**原因**：输入图像尺寸导致CNN输出的feature map不是4x4

**解决**：使用128x128的输入图像（通过crop_dataset_roi处理）

---

## 附录

### A. 关键文件路径

| 文件 | 说明 |
|------|------|
| `configs_hilserl/env_config_piper_real.json` | 数据采集环境配置 |
| `configs_hilserl/reward_classifier_config_piper.json` | Reward Classifier训练配置 |
| `configs_hilserl/train_config_hilserl.json` | HIL-SERL训练配置 |
| `src/lerobot/robots/piper_follower/` | Piper机械臂驱动 |
| `src/lerobot/teleoperators/gamepad/gamepad_utils.py` | 手柄控制映射 |
| `src/lerobot/rl/gym_manipulator.py` | 数据采集主程序 |

### B. 数据集结构

```
data/demos/
├── meta/
│   ├── info.json          # 数据集元信息
│   ├── stats.json         # 统计信息
│   ├── tasks.parquet      # 任务描述
│   └── episodes/          # Episode元数据
├── data/
│   └── chunk-000/
│       └── file-000.parquet  # 状态和动作数据
└── videos/
    └── observation.images.observation.images.front/
        └── chunk-000/
            └── file-000.mp4  # 视频数据
```

### C. 手柄按键映射（Xbox Series X）

```python
# Button映射
Button 0: A
Button 1: B
Button 2: X
Button 3: Y
Button 4: LB  # 干预键
Button 5: RB
Button 6: Back
Button 7: Start

# Axis映射
Axis 0: 左摇杆 X（左右）
Axis 1: 左摇杆 Y（上下）
Axis 2: LT（扳机）
Axis 3: 右摇杆 X（左右）
Axis 4: 右摇杆 Y（上下）
Axis 5: RT（扳机）
```
