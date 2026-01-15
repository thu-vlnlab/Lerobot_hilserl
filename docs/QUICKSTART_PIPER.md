# Piper HIL-SERL 快速入门

从数据采集到训练的完整流程（约30分钟）。

---

## 前置条件

```bash
# 激活环境
conda activate lerobot
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 设置CAN接口
./scripts/run_hilserl.sh setup-can
```

---

## 第一步：采集数据

### 1.1 修改配置（可选）

编辑 `configs_hilserl/env_config_piper_real.json`：

```json
{
  "dataset": {
    "root": "data/demos10",           // 数据保存路径
    "num_episodes_to_record": 10      // 采集episode数量
  }
}
```

### 1.2 运行数据采集

```bash
./scripts/run_hilserl.sh record configs_hilserl/env_config_piper_real.json
```

### 1.3 手柄控制

| 按键 | 功能 |
|------|------|
| 左摇杆 | X/Y方向移动 |
| 右摇杆上下 | Z方向移动 |
| RT | 闭合夹爪 |
| LT | 打开夹爪 |
| **A** | 标记成功，结束episode |
| B | 标记失败 |
| Y | 重新录制当前episode |

**流程**：控制机械臂完成任务 → 按A标记成功 → 自动进入下一episode → 重复直到采集完成

---

## 第二步：训练Reward Classifier

### 2.1 修改配置

编辑 `configs_hilserl/reward_classifier_config_piper.json`：

```json
{
  "dataset": {
    "root": "data/demos10"            // 与采集数据路径一致
  },
  "output_dir": "outputs/reward_classifier_piper10",
  "steps": 8000                       // 训练步数
}
```

### 2.2 运行训练

```bash
./scripts/run_hilserl.sh train-reward configs_hilserl/reward_classifier_config_piper.json
```

训练完成后，模型保存在 `outputs/reward_classifier_piper10/checkpoints/last/pretrained_model`

---

## 第三步：HIL-SERL训练

### 3.1 修改配置

编辑 `configs_hilserl/train_config_hilserl_piper_real.json`：

```json
{
  "env": {
    "processor": {
      "reward_classifier": {
        "pretrained_path": "outputs/reward_classifier_piper10/checkpoints/last/pretrained_model"
      }
    }
  }
}
```

### 3.2 启动训练（需要两个终端）

**终端1 - Learner**：
```bash
conda activate lerobot
./scripts/run_hilserl.sh train-learner configs_hilserl/train_config_hilserl_piper_real.json
```

**终端2 - Actor**（等Learner初始化完成后）：
```bash
conda activate lerobot
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
./scripts/run_hilserl.sh train-actor configs_hilserl/train_config_hilserl_piper_real.json
```

### 3.3 训练中干预

| 按键 | 功能 |
|------|------|
| **LB** | 按住进入干预模式，松开退出 |
| 摇杆 | 干预时控制机械臂 |
| Start | 重置机械臂到初始位置 |

**干预策略**：
- 初期：频繁干预，帮助学习
- 中期：策略偏离时干预
- 后期：仅危险情况干预

---

## 监控训练

### Learner输出

```
[LEARNER] Number of optimization step: 100
[LEARNER] Buffer sizes - Online: 1500, Offline (intervention): 50
```

- **Online**: 所有采集的数据
- **Offline (intervention)**: 人工干预数据

### WandB（可选）

在配置中启用：
```json
"wandb": {
  "enable": true,
  "project": "piper-hilserl-real"
}
```

访问 https://wandb.ai 查看训练曲线。

---

## 数据保存位置

| 数据 | 路径 |
|------|------|
| 采集的演示数据 | `data/demos10/` |
| Reward Classifier | `outputs/reward_classifier_piper10/` |
| HIL-SERL checkpoint | `outputs/train_hilserl_piper/` |
| Online buffer | `outputs/.../dataset/` |
| Offline buffer（干预数据） | `outputs/.../dataset_offline/` |

---

## 常见问题

### 机械臂使能超时

```bash
sudo ip link set can0 down
sudo ip link set can0 up type can bitrate 1000000
```

### 手柄未检测到

```bash
ls /dev/input/js*
jstest /dev/input/js0
```

### KeyError: 'observation.images.front'

检查配置文件中的camera key是否为 `front`（不是 `observation.images.front`）。

---

## （可选）第四步：采集专家数据

专家数据可以预填充offline buffer，加速HIL-SERL训练。

```bash
./scripts/run_hilserl.sh record-expert configs_hilserl/env_config_piper_expert.json
```

**与普通数据采集的区别**：
- `terminate_on_success=true`：任务成功后立即结束episode
- 数据保存在 `data/expert_demos/`
- 需要先训练好reward classifier才能自动检测成功

使用专家数据训练时，修改 `train_config_hilserl_piper_real.json`：
```json
{
  "dataset": {
    "repo_id": "thu-vlnlab/piper_pick_lift",
    "root": "data/expert_demos"
  }
}
```

---

## 完整命令速查

```bash
# 1. 环境准备
conda activate lerobot
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
./scripts/run_hilserl.sh setup-can

# 2. 数据采集（用于reward classifier训练）
./scripts/run_hilserl.sh record configs_hilserl/env_config_piper_real.json

# 3. 训练Reward Classifier
./scripts/run_hilserl.sh train-reward configs_hilserl/reward_classifier_config_piper.json

# 4. （可选）采集专家数据
./scripts/run_hilserl.sh record-expert configs_hilserl/env_config_piper_expert.json

# 5. HIL-SERL训练
# 终端1
./scripts/run_hilserl.sh train-learner configs_hilserl/train_config_hilserl_piper_real.json
# 终端2
./scripts/run_hilserl.sh train-actor configs_hilserl/train_config_hilserl_piper_real.json
```
