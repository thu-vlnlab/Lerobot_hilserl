# RM75-6F 接入 HIL-SERL 说明与改造方案

本文档用于规划将睿尔曼 RM75-6F 机械臂接入 `Lerobot_hilserl`，支持数据采集、Reward Classifier 训练与 HIL-SERL 在线训练。

## 1. 目标

目标不是只做一套“能跑的控制脚本”，而是把 RM75-6F 接成 LeRobot/HIL-SERL 原生支持的机器人，使其具备以下能力：

- 使用 LeRobot 标准 `Robot` 抽象进行控制和观测
- 使用 SpaceMouse 作为主要遥操作设备
- 录制符合 LeRobotDataset 格式的数据集
- 训练视觉 Reward Classifier
- 运行 Actor/Learner 双进程 HIL-SERL 在线训练

## 2. 当前仓库现状

当前工作区里已经有 3 条相关链路：

### 2.1 `Lerobot_hilserl`

这是目标训练主仓库，已经具备：

- `gym_manipulator` 真机环境
- `Robot` / `Teleoperator` 抽象
- Reward Classifier 训练流程
- Actor/Learner 异步训练流程
- 现成的 Piper 接入模板

关键参考文件：

- `src/lerobot/rl/gym_manipulator.py`
- `src/lerobot/robots/piper_follower/config_piper_follower.py`
- `src/lerobot/robots/piper_follower/piper_follower.py`
- `configs_hilserl/env_config_piper_real.json`
- `configs_hilserl/reward_classifier_config_piper.json`
- `configs_hilserl/train_config_hilserl_piper_real.json`

### 2.2 `spacemouse2rm75b`

这是现有的 SpaceMouse 控制睿尔曼链路，已经验证了：

- SpaceMouse 输入读取
- RealMan Python SDK 连接
- 位置/速度透传控制
- 运动录制与 CSV 输出

它适合作为“控制映射和 SDK 使用参考”，但不适合作为最终 HIL-SERL 主控制框架。

### 2.3 `ros2_rm_robot`

这是睿尔曼 ROS2 官方仓库，适合作为：

- bringup 和状态观察工具
- MoveIt 调试工具
- ROS2 话题/接口参考
- UDP 状态发布能力参考

不建议第一版把 HIL-SERL 主控制闭环完全建立在 ROS2 topic 上，否则控制链会更长，同步和延迟更难控。

## 3. 关于 “RM75-6F” 的说明

这里需要先澄清一个命名问题。

在睿尔曼官方 ROS2 仓库里，`rm_75_bringup.launch.py`、`rm_75_6f_bringup.launch.py`、`rm_75_6fb_bringup.launch.py` 是并列存在的。按官方命名习惯：

- `75` 通常表示 RM75 系列机械臂
- `6f` 更像“六维力版本”
- `6fb` 更像“一体化六维力版本”

这不一定等于“6 自由度机械臂”。

因此在正式接入前，需要确认以下两件事：

1. 你的机械臂本体自由度到底是 6 还是 7
2. 你是否接入了六维力传感器，以及它是否要进入观测空间

这两个点会直接影响：

- `observation.state` 的维度
- `reset_joint_positions` 的长度
- IK/EE 控制模型
- 后续是否把 force/torque 纳入 observation

本文档默认第一版采用保守策略：

- 先只接入机械臂本体状态和视觉
- 先不把六维力传感器纳入训练主观测
- 后续再扩展触觉/力觉分支

## 4. 推荐总体方案

推荐采用下面这条主链：

```text
SpaceMouse -> LeRobot Teleoperator -> gym_manipulator -> RM75 Robot -> RealMan Python SDK
                                                    |
                                                    +-> Cameras -> Dataset / Reward Classifier
```

而不是：

```text
SpaceMouse -> 自定义 ROS2 节点 -> /action /state -> 单独录包脚本 -> 再转换到 HIL-SERL
```

原因：

- LeRobot 主框架已经围绕 `Robot + Teleoperator + Processor + Dataset` 设计
- 直接复用主框架，后续训练、回放、评估都更统一
- 现有 `data_collect/realman_contrl.py` 与 `data_record_with_tactile.py` 只提供了过渡方案，状态/动作维度也偏简化

## 5. 四阶段任务清单

下面的阶段按依赖顺序排列，建议不要跳阶段。

### 阶段 1：机器人抽象接入

目标：把 RM75-6F 接成 LeRobot 可识别的 `Robot`。

本阶段任务：

- [ ] 确认 RM75 本体 DOF、夹爪接口、六维力版本信息
- [ ] 确认 RealMan Python SDK 能稳定读取：
  - 当前关节角
  - 当前末端位姿
  - 夹爪状态
- [ ] 在 `src/lerobot/robots/` 下新增 `rm75_follower/`
- [ ] 新增 `config_rm75_follower.py`
- [ ] 新增 `rm75_follower.py`
- [ ] 实现 `connect()`
- [ ] 实现 `get_observation()`
- [ ] 实现 `send_action()`
- [ ] 实现 `disconnect()`
- [ ] 做 `rm75_follower_ee` 版本，第一版只支持：
  - `ee.x`
  - `ee.y`
  - `ee.z`
  - `gripper.pos`
- [ ] 第一版锁定姿态，不开放 `ee.rx/ry/rz` 控制
- [ ] 加入安全边界：
  - 工作空间裁剪
  - 单步最大位移
  - reset pose
  - gripper 范围约束

本阶段交付物：

- 能通过配置创建 RM75 真机环境
- 能从环境里读观测
- 能发动作控制 RM75
- 能稳定 reset 到固定姿态

本阶段验收标准：

- 连续运行 10 分钟不掉连接
- 单步动作方向与实际机械臂方向一致
- reset 不会冲撞
- 夹爪动作不会超限

### 阶段 2：遥操作与数据采集

目标：让 SpaceMouse 成为 HIL-SERL 里的标准 `Teleoperator`，并录制标准数据集。

本阶段任务：

- [ ] 在 `src/lerobot/teleoperators/` 下新增 `spacemouse/`
- [ ] 新增 `configuration_spacemouse.py`
- [ ] 新增 `teleop_spacemouse.py`
- [ ] 复用现有 `spacemouse_input.py` 逻辑
- [ ] 输出标准 action：
  - `delta_x`
  - `delta_y`
  - `delta_z`
  - `gripper`
- [ ] 输出标准 teleop events：
  - `is_intervention`
  - `success`
  - `failure`
  - `rerecord_episode`
  - `terminate_episode`
- [ ] 明确 SpaceMouse 按键映射
- [ ] 如果按键不够，补一套键盘辅助事件输入方案
- [ ] 新增 `configs_hilserl/env_config_rm75_real.json`
- [ ] 接入前视相机
- [ ] 跑通 `record` 模式录制数据
- [ ] 检查数据集中的图像、动作、状态是否同步

本阶段交付物：

- SpaceMouse 原生 teleop
- RM75 原生录制配置
- 一批可用于 reward classifier 的 demo 数据

本阶段验收标准：

- 能连续录制至少 20 个 episode
- 数据集可被 `LeRobotDataset` 正常读取
- 图像、动作、状态时间对齐无明显漂移

### 阶段 3：Reward Classifier

目标：让系统自动判断任务是否成功。

本阶段任务：

- [ ] 确定 RM75 任务定义
- [ ] 采集成功/失败混合数据
- [ ] 做 ROI 裁剪
- [ ] 新增 `configs_hilserl/reward_classifier_config_rm75.json`
- [ ] 训练 reward classifier
- [ ] 在线验证 classifier 预测
- [ ] 调整 `success_threshold`
- [ ] 确认 classifier 对光照/视角变化不过度敏感

本阶段交付物：

- 可用的 reward classifier checkpoint
- RM75 专用 reward 配置

本阶段验收标准：

- 成功帧与失败帧有可区分预测
- 在线运行时不会频繁误报成功
- 真实任务成功后能稳定给出 reward

### 阶段 4：HIL-SERL 在线训练

目标：让 RM75 跑通 Actor/Learner 在线训练闭环。

本阶段任务：

- [ ] 新增 `configs_hilserl/train_config_hilserl_rm75_real.json`
- [ ] 配置 reward classifier 路径
- [ ] 配置 actor/learner 通信参数
- [ ] 配置数据集路径
- [ ] 跑通 learner
- [ ] 跑通 actor
- [ ] 训练初期高频人工干预
- [ ] 记录 `episodic_reward`
- [ ] 记录 `intervention_rate`
- [ ] 记录 `success_rate`
- [ ] 根据训练情况调整：
  - fps
  - EE step size
  - gripper penalty
  - success reward
  - replay buffer 容量

本阶段交付物：

- RM75 的完整 HIL-SERL 训练配置
- 可执行的在线训练流程

本阶段验收标准：

- actor/learner 连续运行稳定
- 干预数据能进入 offline buffer
- 策略成功率随训练轮次提升

## 6. 建议修改方案

### 6.1 第一批新增文件

建议新增以下文件：

```text
src/lerobot/robots/rm75_follower/__init__.py
src/lerobot/robots/rm75_follower/config_rm75_follower.py
src/lerobot/robots/rm75_follower/rm75_follower.py

src/lerobot/teleoperators/spacemouse/__init__.py
src/lerobot/teleoperators/spacemouse/configuration_spacemouse.py
src/lerobot/teleoperators/spacemouse/teleop_spacemouse.py
src/lerobot/teleoperators/spacemouse/spacemouse_input.py

configs_hilserl/env_config_rm75_real.json
configs_hilserl/reward_classifier_config_rm75.json
configs_hilserl/train_config_hilserl_rm75_real.json

docs/RM75_6F_HILSERL_PLAN.md
```

### 6.2 需要修改的现有文件

建议修改以下现有文件：

```text
src/lerobot/teleoperators/utils.py
```

用途：

- 注册 `spacemouse` teleoperator 的构造逻辑

如果 `make_robot_from_config()` 依赖显式导入，也可能需要补充：

```text
src/lerobot/robots/__init__.py
```

用途：

- 导出 `rm75_follower`
- 确保配置注册生效

### 6.3 机器人层设计建议

建议 `rm75_follower` 参考 `piper_follower_ee` 的设计，第一版只做笛卡尔位置控制：

- 观测：
  - 关节角
  - 夹爪位置
  - 末端位置 `ee.x/ee.y/ee.z`
  - 相机图像
- 动作：
  - `ee.x`
  - `ee.y`
  - `ee.z`
  - `gripper.pos`

第一版不要做的内容：

- 不先做 6D 姿态动作
- 不先把 ROS2 topic 放进主控制闭环
- 不先把六维力和触觉并入训练输入
- 不先上复杂 IK 或 MoveIt

### 6.4 Teleop 层设计建议

`spacemouse` teleoperator 建议模仿 `gamepad` 接口风格。

建议按下面思路实现：

- 平移：直接映射到 `delta_x/delta_y/delta_z`
- 旋转：第一版忽略，或仅用于切换模式
- 左右按键：作为成功/失败/重录/干预切换

如果 SpaceMouse 按键不够，建议第一版采用混合方案：

- SpaceMouse：负责连续位移控制
- 键盘：负责 `success/failure/rerecord/intervention`

这样实现复杂度最低，也最容易快速验证训练链路。

### 6.5 配置层设计建议

第一版 RM75 配置建议尽量贴近 Piper 配置，减少变量。

建议的初版配置原则：

- `fps`: 10
- 单相机起步
- `observation.state`: 先只放关节状态
- `reward_classifier.pretrained_path`: 先允许为空
- `terminate_on_success`: 采集阶段为 `false`
- `terminate_on_success`: 专家数据或在线训练阶段可切为 `true`

## 7. 不推荐的方案

以下方案不建议作为第一版主线：

### 7.1 继续沿用 `data_collect/realman_contrl.py + data_record_with_tactile.py`

原因：

- 当前只输出 3 维状态和 3 维动作
- 与 HIL-SERL 主流程的 `Robot/Teleoperator` 体系不一致
- 后续 reward、actor、learner、intervention 都要重复做桥接

### 7.2 先用 ROS2 topic 重写全套控制和采集闭环

原因：

- 控制链更长
- 调试面更广
- 不利于快速验证训练主干

ROS2 更适合放在以下位置：

- bringup
- 状态观测
- MoveIt 调试
- 日志与监控

## 8. 实施顺序建议

建议按下面顺序推进：

1. 先做 `rm75_follower_ee`
2. 用键盘或 Gamepad 临时跑通 `gym_manipulator`
3. 再补 `spacemouse` teleoperator
4. 再采集 RM75 数据
5. 再训练 reward classifier
6. 最后再开 HIL-SERL 在线训练

这样做的好处是：

- 能最快发现“机器人接口”问题
- 不会把“机器人问题”和“遥操作问题”混在一起
- 更容易逐层定位故障

## 9. 第一版落地范围

如果希望尽快落地，建议把第一版范围压缩为：

- 单任务
- 单相机
- 位置控制
- 固定姿态
- 无力觉输入
- 无触觉输入

第一版先证明“能稳定训练”，第二版再逐步加入：

- 六维力输入
- 触觉输入
- 多相机
- 姿态控制
- ROS2 辅助状态桥

## 10. 总结

RM75-6F 接入 HIL-SERL 的正确方向，不是继续堆叠独立脚本，而是把现有 SpaceMouse 控制链和 RealMan SDK 能力，迁移进 LeRobot 的标准机器人/遥操作框架。

一句话概括这次改造：

- 机器人层：做成 `rm75_follower_ee`
- 遥操作层：做成 `spacemouse` teleoperator
- 配置层：补齐 RM75 的 `env / reward / train` 三套配置
- 训练层：复用现成 HIL-SERL 主流程

只要第一阶段和第二阶段做扎实，后面的 reward classifier 和在线训练基本就是配置与数据质量问题，不再是系统结构问题。
