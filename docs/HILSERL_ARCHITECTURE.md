# HIL-SERL 代码架构分析

本文档详细分析 HIL-SERL (Human-in-the-Loop Sample Efficient Reinforcement Learning) 的代码架构，帮助开发者理解系统设计和实现细节。

## 目录

1. [整体架构](#1-整体架构)
2. [核心文件结构](#2-核心文件结构)
3. [Learner 详解](#3-learner-详解)
4. [Actor 详解](#4-actor-详解)
5. [SAC 策略](#5-sac-策略)
6. [ReplayBuffer](#6-replaybuffer)
7. [环境处理管道](#7-环境处理管道)
8. [人类干预机制](#8-人类干预机制)
9. [配置文件结构](#9-配置文件结构)
10. [运行流程](#10-运行流程)
11. [gRPC 通信协议](#11-grpc-通信协议)

---

## 1. 整体架构

HIL-SERL 采用 **Actor-Learner 分布式架构**，将策略执行和策略学习分离到两个独立进程：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           HIL-SERL 架构                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────┐                    ┌──────────────────┐          │
│  │   Actor (真机)   │◄────gRPC────────► │   Learner (GPU)  │          │
│  │                  │     通信           │                  │          │
│  │  - 执行策略      │                    │  - 策略训练      │          │
│  │  - 收集经验      │  ─→ transitions    │  - 更新网络      │          │
│  │  - 人类干预      │  ─→ interactions   │  - 经验回放      │          │
│  │  - 发送transition│  ←─ parameters     │  - 推送参数      │          │
│  └──────────────────┘                    └──────────────────┘          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**设计优势**:
- **解耦**: 真机控制和GPU训练分离，避免相互阻塞
- **异步**: Actor 持续收集数据，Learner 持续训练
- **可扩展**: 理论上可支持多 Actor 并行收集

---

## 2. 核心文件结构

所有 RL 相关代码位于 `src/lerobot/rl/` 目录：

| 文件 | 行数 | 功能描述 |
|------|------|----------|
| `learner.py` | ~1225 | **学习器主循环** - SAC策略训练、replay buffer管理、checkpoint保存 |
| `actor.py` | ~743 | **执行器主循环** - 环境交互、动作执行、人类干预处理 |
| `buffer.py` | ~835 | **经验回放缓冲区** - 存储transition、DrQ数据增强、异步采样 |
| `gym_manipulator.py` | ~1143 | **机器人环境** - Gym接口、传感器处理、动作处理管道 |
| `learner_service.py` | ~118 | **gRPC服务** - Actor-Learner间的通信协议实现 |
| `process.py` | ~100 | **进程管理** - 信号处理、优雅关闭 |
| `queue.py` | ~50 | **队列工具** - 获取队列最新元素 |
| `wandb_utils.py` | ~200 | **日志工具** - WandB集成 |

**策略实现** 位于 `src/lerobot/policies/sac/`:

| 文件 | 功能 |
|------|------|
| `modeling_sac.py` | SAC策略网络实现 (Actor, Critic, Encoder) |
| `configuration_sac.py` | SAC配置类定义 |
| `processor_sac.py` | SAC数据处理器 |

---

## 3. Learner 详解

### 3.1 入口函数

```python
# learner.py:106-119
@parser.wrap()
def train_cli(cfg: TrainRLServerPipelineConfig):
    train(cfg, job_name=cfg.job_name)
```

### 3.2 初始化流程

```python
# learner.py:122-177
def train(cfg, job_name):
    # 1. 配置验证和日志初始化
    cfg.validate()
    init_logging(log_file=log_file)

    # 2. WandB 初始化 (可选)
    wandb_logger = WandBLogger(cfg) if cfg.wandb.enable else None

    # 3. 处理恢复训练逻辑
    cfg = handle_resume_logic(cfg)

    # 4. 设置随机种子和CUDA优化
    set_seed(seed=cfg.seed)
    torch.backends.cudnn.benchmark = True

    # 5. 启动学习器线程
    start_learner_threads(cfg, wandb_logger, shutdown_event)
```

### 3.3 线程/进程架构

```python
# learner.py:180-246
def start_learner_threads(cfg, wandb_logger, shutdown_event):
    # 创建三个通信队列
    transition_queue = Queue()        # 接收 transitions
    interaction_message_queue = Queue()  # 接收交互消息
    parameters_queue = Queue()        # 发送策略参数

    # 启动 gRPC 通信进程/线程
    communication_process = Thread/Process(
        target=start_learner,  # gRPC 服务器
        args=(parameters_queue, transition_queue, ...)
    )

    # 主训练循环
    add_actor_information_and_train(cfg, wandb_logger, ...)
```

### 3.4 主训练循环

这是 Learner 的核心，位于 `learner.py:251-620`:

```python
def add_actor_information_and_train(...):
    # ===== 初始化阶段 =====

    # 1. 创建策略
    policy: SACPolicy = make_policy(cfg.policy, env_cfg=cfg.env)
    policy.train()

    # 2. 推送初始策略参数给 Actor
    push_actor_policy_to_queue(parameters_queue, policy)

    # 3. 创建优化器
    optimizers, lr_scheduler = make_optimizers_and_scheduler(cfg, policy)
    # optimizers = {
    #     "actor": Adam(policy.actor.parameters()),
    #     "critic": Adam(policy.critic_ensemble.parameters()),
    #     "temperature": Adam([policy.log_alpha]),
    #     "discrete_critic": Adam(...) (可选)
    # }

    # 4. 初始化 Replay Buffer
    replay_buffer = initialize_replay_buffer(cfg, device, storage_device)

    # 5. 初始化 Offline Buffer (演示数据 + 人类干预数据)
    if cfg.dataset is not None:
        offline_replay_buffer = initialize_offline_replay_buffer(cfg, ...)
        batch_size = batch_size // 2  # 从两个buffer各采一半
    else:
        offline_replay_buffer = ReplayBuffer(capacity=offline_buffer_capacity)

    # ===== 主循环 =====

    while True:
        # 检查关闭信号
        if shutdown_event.is_set():
            break

        # 步骤1: 处理 Actor 发来的 transitions
        process_transitions(
            transition_queue, replay_buffer, offline_replay_buffer, ...
        )

        # 步骤2: 处理交互消息 (episodic reward 等)
        interaction_message = process_interaction_messages(
            interaction_message_queue, ...
        )

        # 步骤3: 等待 buffer 有足够样本
        if len(replay_buffer) < online_step_before_learning:
            continue

        # 步骤4: UTD (Update-To-Data) 循环
        # 每个环境 step 执行 utd_ratio 次梯度更新
        for _ in range(utd_ratio - 1):
            # 从 online buffer 采样
            batch = next(online_iterator)

            # 如果有 offline buffer，混合采样
            if offline_replay_buffer and len(offline_replay_buffer) > 0:
                batch_offline = next(offline_iterator)
                batch = concatenate_batch_transitions(batch, batch_offline)

            # Critic 前向传播
            observation_features, next_observation_features = get_observation_features(
                policy, observations, next_observations
            )
            critic_output = policy.forward(batch, model="critic")

            # Critic 反向传播
            loss_critic = critic_output["loss_critic"]
            optimizers["critic"].zero_grad()
            loss_critic.backward()
            clip_grad_norm_(policy.critic_ensemble.parameters(), max_norm)
            optimizers["critic"].step()

            # 离散 Critic 更新 (如果有 gripper)
            if policy.config.num_discrete_actions is not None:
                discrete_critic_output = policy.forward(batch, model="discrete_critic")
                # ... 同样的优化流程

            # 软更新 target 网络
            policy.update_target_networks()

        # 步骤5: 最后一次更新 + Actor/Temperature 更新
        batch = next(online_iterator)
        # ... critic 更新 ...

        if optimization_step % policy_update_freq == 0:
            # Actor 更新
            actor_output = policy.forward(batch, model="actor")
            loss_actor = actor_output["loss_actor"]
            optimizers["actor"].zero_grad()
            loss_actor.backward()
            optimizers["actor"].step()

            # Temperature 更新
            temperature_output = policy.forward(batch, model="temperature")
            loss_temperature = temperature_output["loss_temperature"]
            optimizers["temperature"].zero_grad()
            loss_temperature.backward()
            optimizers["temperature"].step()

            policy.update_temperature()

        # 步骤6: 定期推送策略参数
        if time.time() - last_time_policy_pushed > policy_parameters_push_frequency:
            push_actor_policy_to_queue(parameters_queue, policy)

        # 步骤7: 保存 checkpoint
        if optimization_step % save_freq == 0:
            save_training_checkpoint(...)

        optimization_step += 1
```

### 3.5 关键配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `utd_ratio` | 20 | 每个环境step的梯度更新次数 |
| `policy_update_freq` | 1 | Actor更新频率 (相对于Critic) |
| `online_step_before_learning` | 1000 | 开始训练前需要的最小样本数 |
| `policy_parameters_push_frequency` | 2 | 参数推送间隔 (秒) |
| `online_buffer_capacity` | 100000 | 在线buffer容量 |
| `offline_buffer_capacity` | 30000 | 离线buffer容量 |

---

## 4. Actor 详解

### 4.1 入口函数

```python
# actor.py:103-206
@parser.wrap()
def actor_cli(cfg: TrainRLServerPipelineConfig):
    # 1. 验证配置 (跳过目录检查，因为Learner已创建)
    cfg.validate(skip_dir_check=True)

    # 2. 建立与 Learner 的连接
    learner_client, grpc_channel = learner_service_client(host, port)
    establish_learner_connection(learner_client, shutdown_event)

    # 3. 创建通信队列
    parameters_queue = Queue()
    transitions_queue = Queue()
    interactions_queue = Queue()

    # 4. 启动通信线程
    receive_policy_process = Thread(target=receive_policy, ...)
    transitions_process = Thread(target=send_transitions, ...)
    interactions_process = Thread(target=send_interactions, ...)

    # 5. 主执行循环
    act_with_policy(cfg, shutdown_event, ...)
```

### 4.2 主执行循环

```python
# actor.py:211-406
def act_with_policy(cfg, shutdown_event, parameters_queue, transitions_queue, interactions_queue):
    # ===== 初始化 =====

    # 1. 创建环境和处理器
    online_env, teleop_device = make_robot_env(cfg.env)
    env_processor, action_processor = make_processors(
        online_env, teleop_device, cfg.env, cfg.policy.device
    )

    # 2. 创建本地策略副本 (用于推理)
    policy: SACPolicy = make_policy(cfg.policy, env_cfg=cfg.env)
    policy = policy.eval()

    # 3. 重置环境
    obs, info = online_env.reset()
    env_processor.reset()
    action_processor.reset()

    # 4. 处理初始观测
    transition = create_transition(observation=obs, info=info)
    transition = env_processor(transition)

    # ===== 主循环 =====

    for interaction_step in range(cfg.policy.online_steps):
        start_time = time.perf_counter()

        # 步骤1: 策略推理
        observation = {k: v for k, v in transition[OBSERVATION].items()
                       if k in cfg.policy.input_features}
        action = policy.select_action(batch=observation)

        # 步骤2: 环境交互 (包含人类干预处理)
        new_transition = step_env_and_process_transition(
            env=online_env,
            transition=transition,
            action=action,
            env_processor=env_processor,
            action_processor=action_processor,
        )

        # 步骤3: 提取执行的动作 (可能是策略动作或人类干预动作)
        executed_action = new_transition[COMPLEMENTARY_DATA]["teleop_action"]
        reward = new_transition[REWARD]
        done = new_transition.get(DONE, False)
        truncated = new_transition.get(TRUNCATED, False)

        # 步骤4: 检测人类干预
        is_intervening = new_transition[INFO].get(TeleopEvents.IS_INTERVENTION, False)
        if is_intervening:
            episode_intervention = True
            episode_intervention_steps += 1

        # 步骤5: 构建 transition 发送给 Learner
        list_transition_to_send_to_learner.append(Transition(
            state=observation,
            action=executed_action,
            reward=reward,
            next_state=next_observation,
            done=done,
            truncated=truncated,
            complementary_info={
                "discrete_penalty": ...,
                TeleopEvents.IS_INTERVENTION: is_intervening,
            }
        ))

        transition = new_transition

        # 步骤6: Episode 结束处理
        if done or truncated:
            # 更新策略参数
            update_policy_parameters(policy, parameters_queue, device)

            # 发送 transitions 到 Learner
            push_transitions_to_transport_queue(
                list_transition_to_send_to_learner,
                transitions_queue
            )
            list_transition_to_send_to_learner = []

            # 计算干预率
            intervention_rate = episode_intervention_steps / episode_total_steps

            # 发送 episode 统计信息
            interactions_queue.put(python_object_to_bytes({
                "Episodic reward": sum_reward_episode,
                "Interaction step": interaction_step,
                "Episode intervention": int(episode_intervention),
                "Intervention rate": intervention_rate,
            }))

            # 重置环境
            obs, info = online_env.reset()
            env_processor.reset()
            action_processor.reset()
            transition = create_transition(observation=obs, info=info)
            transition = env_processor(transition)

        # 步骤7: 维持 FPS
        if cfg.env.fps is not None:
            precise_sleep(1 / cfg.env.fps - (time.perf_counter() - start_time))
```

### 4.3 通信线程

**接收策略参数**:
```python
# actor.py:462-511
def receive_policy(cfg, parameters_queue, shutdown_event, ...):
    iterator = learner_client.StreamParameters(services_pb2.Empty())
    receive_bytes_in_chunks(iterator, parameters_queue, shutdown_event)
```

**发送 Transitions**:
```python
# actor.py:514-561
def send_transitions(cfg, transitions_queue, shutdown_event, ...):
    learner_client.SendTransitions(
        transitions_stream(shutdown_event, transitions_queue, timeout)
    )
```

**发送交互消息**:
```python
# actor.py:564-614
def send_interactions(cfg, interactions_queue, shutdown_event, ...):
    learner_client.SendInteractions(
        interactions_stream(shutdown_event, interactions_queue, timeout)
    )
```

---

## 5. SAC 策略

### 5.1 组件结构

```
SACPolicy (modeling_sac.py)
│
├── encoder_actor / encoder_critic  (SACObservationEncoder, 可共享)
│   ├── image_encoder
│   │   ├── DefaultImageEncoder (4层CNN)
│   │   └── PretrainedImageEncoder (如 ResNet)
│   ├── spatial_embeddings (SpatialLearnedEmbeddings)
│   ├── post_encoders (Linear + LayerNorm + Tanh)
│   └── state_encoder (Linear + LayerNorm + Tanh)
│
├── actor (Policy)
│   ├── encoder (共享或独立)
│   ├── network (MLP)
│   ├── mean_layer (Linear)
│   └── std_layer (Linear)
│   └── 输出: TanhMultivariateNormalDiag 分布
│
├── critic_ensemble (CriticEnsemble)
│   ├── encoder
│   └── critics (ModuleList of CriticHead)
│       └── 输出: Q(s, a)
│
├── critic_target (CriticEnsemble, EMA 软更新)
│
├── discrete_critic (DiscreteCritic, 可选)
│   └── 用于 gripper 等离散动作
│
├── discrete_critic_target
│
└── log_alpha (Temperature 参数)
```

### 5.2 核心方法

**动作选择** (推理时):
```python
# modeling_sac.py:84-114
@torch.no_grad()
def select_action(self, batch, explore=True):
    # 1. 通过 Actor 网络获取连续动作
    actions, _, _ = self.actor(batch, observations_features)

    # 2. 如果有离散动作 (如 gripper)
    if self.config.num_discrete_actions is not None:
        discrete_action_value = self.discrete_critic(batch)

        # epsilon-greedy 探索
        if explore and random() < epsilon:
            discrete_action = randint(0, num_discrete_actions)
        else:
            discrete_action = argmax(discrete_action_value)

        actions = cat([actions, discrete_action], dim=-1)

    return actions
```

**Forward 方法** (训练时):
```python
# modeling_sac.py:155-234
def forward(self, batch, model="critic"):
    if model == "critic":
        return {"loss_critic": self.compute_loss_critic(...)}

    if model == "discrete_critic":
        return {"loss_discrete_critic": self.compute_loss_discrete_critic(...)}

    if model == "actor":
        return {"loss_actor": self.compute_loss_actor(...)}

    if model == "temperature":
        return {"loss_temperature": self.compute_loss_temperature(...)}
```

**Critic Loss 计算**:
```python
# modeling_sac.py:261-320
def compute_loss_critic(self, observations, actions, rewards, next_observations, done, ...):
    with torch.no_grad():
        # 1. 下一状态的动作和 log_prob
        next_action_preds, next_log_probs, _ = self.actor(next_observations)

        # 2. 计算 target Q 值
        q_targets = self.critic_forward(next_observations, next_action_preds, use_target=True)

        # 3. 子采样 critics (防止过拟合)
        if self.config.num_subsample_critics is not None:
            indices = torch.randperm(num_critics)[:num_subsample_critics]
            q_targets = q_targets[indices]

        # 4. 取最小 Q 值
        min_q = q_targets.min(dim=0)[0]

        # 5. 熵正则化
        if self.config.use_backup_entropy:
            min_q = min_q - (self.temperature * next_log_probs)

        # 6. TD target
        td_target = rewards + (1 - done) * discount * min_q

    # 7. 计算预测 Q 值
    q_preds = self.critic_forward(observations, actions, use_target=False)

    # 8. MSE Loss
    critics_loss = mse_loss(q_preds, td_target).mean(dim=1).sum()

    return critics_loss
```

**Actor Loss 计算**:
```python
# modeling_sac.py:389-405
def compute_loss_actor(self, observations, observation_features=None):
    # 1. 采样动作
    actions_pi, log_probs, _ = self.actor(observations, observation_features)

    # 2. 计算 Q 值
    q_preds = self.critic_forward(observations, actions_pi, use_target=False)
    min_q_preds = q_preds.min(dim=0)[0]

    # 3. 最大化 Q 值 - 熵正则化
    actor_loss = ((self.temperature * log_probs) - min_q_preds).mean()

    return actor_loss
```

**Temperature Loss 计算**:
```python
# modeling_sac.py:381-387
def compute_loss_temperature(self, observations, observation_features=None):
    with torch.no_grad():
        _, log_probs, _ = self.actor(observations, observation_features)

    # 自动调节温度使熵接近 target_entropy
    temperature_loss = (-self.log_alpha.exp() * (log_probs + self.target_entropy)).mean()

    return temperature_loss
```

### 5.3 关键配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_critics` | 10 | Critic 集成数量 |
| `num_subsample_critics` | 2 | 每次更新子采样的 Critic 数量 |
| `critic_target_update_weight` | 0.005 | Target 网络软更新系数 (tau) |
| `discount` | 0.99 | 折扣因子 (gamma) |
| `temperature_init` | 0.01 | 初始温度 |
| `target_entropy` | None | 目标熵 (None 时自动计算为 -dim/2) |
| `shared_encoder` | True | Actor/Critic 是否共享编码器 |
| `freeze_vision_encoder` | False | 是否冻结视觉编码器 |
| `latent_dim` | 256 | 隐藏层维度 |

---

## 6. ReplayBuffer

### 6.1 核心功能

```python
# buffer.py:80-130
class ReplayBuffer:
    def __init__(
        self,
        capacity: int,           # 最大容量
        device: str,             # 采样时的目标设备 (cuda/cpu)
        state_keys: list,        # 状态字典的键列表
        storage_device: str,     # 存储设备 (通常是 cpu 节省 GPU 内存)
        optimize_memory: bool,   # 是否优化内存 (不存储 next_state)
        use_drq: bool,           # 是否使用 DrQ 数据增强
    ):
        self.states = {}           # 状态存储
        self.next_states = {}      # 下一状态存储 (或指向 states)
        self.actions = None        # 动作存储
        self.rewards = None        # 奖励存储
        self.dones = None          # 终止标志存储
        self.complementary_info = {} # 额外信息存储
```

### 6.2 添加 Transition

```python
# buffer.py:190-230
def add(self, state, action, reward, next_state, done, truncated, complementary_info=None):
    # 首次添加时初始化存储
    if not self.initialized:
        self._initialize_storage(state, action, complementary_info)

    # 存储当前 transition
    for key in self.states:
        self.states[key][self.position].copy_(state[key].squeeze(0))
        if not self.optimize_memory:
            self.next_states[key][self.position].copy_(next_state[key].squeeze(0))

    self.actions[self.position].copy_(action.squeeze(0))
    self.rewards[self.position] = reward
    self.dones[self.position] = done

    # 环形缓冲区
    self.position = (self.position + 1) % self.capacity
    self.size = min(self.size + 1, self.capacity)
```

### 6.3 采样

```python
# buffer.py:232-304
def sample(self, batch_size):
    # 随机采样索引
    idx = torch.randint(0, self.size, (batch_size,), device=self.storage_device)

    # 加载状态到目标设备
    batch_state = {key: self.states[key][idx].to(self.device) for key in self.states}

    # 获取下一状态
    if not self.optimize_memory:
        batch_next_state = {key: self.next_states[key][idx].to(self.device) for key in self.states}
    else:
        # 内存优化: next_state[i] = state[i+1]
        next_idx = (idx + 1) % self.capacity
        batch_next_state = {key: self.states[key][next_idx].to(self.device) for key in self.states}

    # DrQ 数据增强 (对图像应用 random shift)
    if self.use_drq and image_keys:
        all_images = cat([batch_state[k], batch_next_state[k] for k in image_keys])
        augmented = self.image_augmentation_function(all_images)
        # 分割回 state 和 next_state...

    return BatchTransition(
        state=batch_state,
        action=self.actions[idx].to(self.device),
        reward=self.rewards[idx].to(self.device),
        next_state=batch_next_state,
        done=self.dones[idx].to(self.device),
        complementary_info=...
    )
```

### 6.4 异步迭代器

```python
# buffer.py:335-387
def _get_async_iterator(self, batch_size, queue_size=2):
    """后台线程预取 batch，避免采样阻塞训练"""
    data_queue = Queue(maxsize=queue_size)

    def producer():
        while not shutdown_event.is_set():
            batch = self.sample(batch_size)
            data_queue.put(batch, timeout=0.5)

    producer_thread = Thread(target=producer, daemon=True)
    producer_thread.start()

    while not shutdown_event.is_set():
        yield data_queue.get(block=True)
```

### 6.5 与 LeRobotDataset 互转

```python
# 从 LeRobotDataset 创建 ReplayBuffer
replay_buffer = ReplayBuffer.from_lerobot_dataset(
    lerobot_dataset=dataset,
    device="cuda",
    state_keys=["observation.images.front", "observation.state"],
    capacity=100000,
)

# 将 ReplayBuffer 保存为 LeRobotDataset
replay_buffer.to_lerobot_dataset(
    repo_id="my_dataset",
    fps=10,
    root="data/output",
)
```

---

## 7. 环境处理管道

### 7.1 处理器架构

```
原始观测 ──→ env_processor ──→ 标准化 transition
              │
              ├── VanillaObservationProcessorStep    (基础观测转换)
              ├── JointVelocityProcessorStep         (添加关节速度, 可选)
              ├── MotorCurrentProcessorStep          (添加电机电流, 可选)
              ├── ForwardKinematicsJointsToEE...     (正运动学, 可选)
              ├── ImageCropResizeProcessorStep       (图像裁剪缩放)
              ├── TimeLimitProcessorStep             (时间限制)
              ├── GripperPenaltyProcessorStep        (夹爪惩罚)
              ├── RewardClassifierProcessorStep      (成功检测)
              ├── AddBatchDimensionProcessorStep     (添加 batch 维度)
              └── DeviceProcessorStep                (设备转移)


动作 ──→ action_processor ──→ 机器人指令
          │
          ├── AddTeleopActionAsComplimentaryDataStep (记录遥操动作)
          ├── AddTeleopEventsAsInfoStep              (添加遥操事件)
          ├── InterventionActionProcessorStep        (人类干预处理)
          ├── PiperDeltaToAbsoluteEEStep             (Piper 末端执行器控制)
          └── InverseKinematicsRLStep                (逆运动学, 可选)
```

### 7.2 关键处理器

**RewardClassifierProcessorStep**:
```python
# 使用预训练的奖励分类器检测任务成功
if self.reward_classifier(observation["image"]) > self.success_threshold:
    reward = self.success_reward
    if self.terminate_on_success:
        done = True
```

**InterventionActionProcessorStep**:
```python
# 人类干预时，使用遥操动作替换策略动作
if teleop_events.get(TeleopEvents.IS_INTERVENTION):
    action = teleop_action
    info[TeleopEvents.IS_INTERVENTION] = True
```

**PiperDeltaToAbsoluteEEStep**:
```python
# 将 delta 动作转换为绝对末端执行器位置
ee_position = current_ee_position + delta * ee_step_size
gripper_pos = clip(gripper_pos + gripper_delta, 0, max_gripper_pos)
```

---

## 8. 人类干预机制

### 8.1 干预检测

干预通过 gamepad 的 LB (Left Bumper) 按钮触发：

```python
# teleoperators/gamepad.py
def get_teleop_events(self):
    return {
        TeleopEvents.IS_INTERVENTION: self.gamepad.left_bumper,  # LB 按钮
        TeleopEvents.SUCCESS: self.gamepad.y_button,             # Y 按钮
        TeleopEvents.TERMINATE_EPISODE: self.gamepad.a_button,   # A 按钮
        TeleopEvents.RERECORD_EPISODE: self.gamepad.x_button,    # X 按钮
    }
```

### 8.2 干预处理流程

```
1. Actor 检测干预 (actor.py:326-331)
   │
   ├─→ is_intervening = info.get(TeleopEvents.IS_INTERVENTION)
   │
   └─→ if is_intervening:
       │   episode_intervention = True
       │   episode_intervention_steps += 1
       │
       └─→ 使用 teleop_action 替换 policy_action

2. Transition 包含干预标记 (actor.py:333-349)
   │
   └─→ complementary_info = {
           TeleopEvents.IS_INTERVENTION: is_intervening
       }

3. Learner 处理干预数据 (learner.py:1144-1191)
   │
   ├─→ replay_buffer.add(**transition)  # 所有数据进入 online buffer
   │
   └─→ if is_intervention:
           offline_replay_buffer.add(**transition)  # 干预数据额外进入 offline buffer
```

### 8.3 干预数据的作用

- **Online Buffer**: 包含所有交互数据 (策略动作 + 人类干预动作)
- **Offline Buffer**: 只包含高质量数据 (演示数据 + 人类干预数据)
- **混合采样**: 训练时从两个 buffer 各采一半，确保高质量数据的比例

---

## 9. 配置文件结构

### 9.1 完整配置示例

```json
{
  "output_dir": "outputs/train_hilserl_piper",
  "resume": false,
  "batch_size": 64,
  "save_checkpoint": true,
  "save_freq": 500,
  "log_freq": 10,
  "seed": 42,

  "policy": {
    "type": "sac",
    "device": "cuda",

    // SAC 超参数
    "temperature_init": 0.01,
    "discount": 0.99,
    "critic_target_update_weight": 0.005,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "temperature_lr": 3e-4,
    "utd_ratio": 20,
    "policy_update_freq": 1,
    "num_critics": 10,
    "num_subsample_critics": 2,

    // Buffer 配置
    "storage_device": "cuda",
    "online_buffer_capacity": 100000,
    "offline_buffer_capacity": 30000,
    "online_step_before_learning": 1000,

    // Actor-Learner 通信
    "actor_learner_config": {
      "policy_parameters_push_frequency": 2,
      "learner_host": "localhost",
      "learner_port": 50051
    },

    // 输入输出特征
    "input_features": {
      "observation.images.front": {
        "type": "VISUAL",
        "shape": [3, 128, 128]
      },
      "observation.state": {
        "type": "STATE",
        "shape": [6]
      }
    },
    "output_features": {
      "action": {
        "type": "ACTION",
        "shape": [4]
      }
    }
  },

  "env": {
    "type": "gym_manipulator",
    "name": "real_robot",
    "fps": 10,
    "task": "pick_and_lift",

    "processor": {
      "control_mode": "gamepad",

      "observation": {
        "add_joint_velocity_to_observation": false,
        "display_cameras": true
      },

      "image_preprocessing": {
        "crop_params_dict": {},
        "resize_size": [128, 128]
      },

      "gripper": {
        "use_gripper": true,
        "gripper_penalty": 0.0
      },

      "reset": {
        "fixed_reset_joint_positions": [0, 77.14, -57.0, 0, 77.73, 0, 0.0],
        "reset_time_s": 5.0,
        "control_time_s": 15.0,
        "terminate_on_success": true
      },

      "reward_classifier": {
        "pretrained_path": "outputs/reward_classifier/checkpoints/last/pretrained_model",
        "success_threshold": 0.85,
        "success_reward": 1.0
      },

      "max_gripper_pos": 70.0
    },

    "robot": {
      "type": "piper_follower_ee",
      "can_name": "can0",
      "id": "piper_arm",
      "use_degrees": true,
      "cameras": {
        "front": {
          "type": "intelrealsense",
          "serial_number_or_name": "338622070324",
          "width": 640,
          "height": 480,
          "fps": 30
        }
      }
    },

    "teleop": {
      "type": "gamepad",
      "use_gripper": true
    }
  },

  "dataset": {
    "repo_id": "piper_pick_lift",
    "root": "data/demos"
  },

  "wandb": {
    "enable": true,
    "project": "piper-hilserl",
    "entity": null
  }
}
```

### 9.2 关键参数说明

| 类别 | 参数 | 说明 |
|------|------|------|
| **训练** | `batch_size` | 每次更新的样本数 |
| | `utd_ratio` | Update-To-Data 比率，每个环境 step 的梯度更新次数 |
| | `online_step_before_learning` | 开始训练前需要收集的最小样本数 |
| **SAC** | `discount` | 折扣因子 gamma |
| | `temperature_init` | 初始温度，控制探索程度 |
| | `num_critics` | Critic 集成数量，减少 Q 值过估计 |
| **Buffer** | `online_buffer_capacity` | 在线 buffer 容量 |
| | `offline_buffer_capacity` | 离线 buffer 容量 (演示 + 干预) |
| **环境** | `fps` | 控制频率 |
| | `control_time_s` | 每个 episode 的最大时长 |
| | `terminate_on_success` | 成功时是否终止 episode |
| **奖励** | `success_threshold` | 成功检测阈值 |
| | `success_reward` | 成功时的奖励值 |

---

## 10. 运行流程

### 10.1 启动命令

```bash
# Terminal 1: 启动 Learner (必须先启动)
./scripts/run_hilserl.sh train-learner configs_hilserl/train_config_hilserl_piper_real_v3.json

# Terminal 2: 启动 Actor
./scripts/run_hilserl.sh train-actor configs_hilserl/train_config_hilserl_piper_real_v3.json
```

### 10.2 Learner 启动流程

```
1. 解析配置文件
2. 初始化日志和 WandB
3. 检查 resume 逻辑
4. 创建 SACPolicy
5. 加载离线演示数据 (如有)
6. 初始化 Replay Buffer
7. 启动 gRPC 服务器
8. 等待 Actor 连接
9. 开始训练循环
```

### 10.3 Actor 启动流程

```
1. 解析配置文件
2. 连接 Learner gRPC 服务
3. 初始化机器人环境
4. 创建本地 SACPolicy 副本
5. 启动通信线程:
   - receive_policy: 接收策略参数
   - send_transitions: 发送 transitions
   - send_interactions: 发送 episode 统计
6. 重置环境
7. 开始交互循环
```

### 10.4 训练循环时序

```
Actor                              Learner
  │                                   │
  │──── Ready() ─────────────────────→│ 建立连接
  │                                   │
  │←─── Parameters (初始策略) ────────│
  │                                   │
  ├─────────────────────────────────────┤ 训练循环开始
  │                                   │
  │  step 1: select_action()          │
  │  step 2: env.step()               │
  │  step 3: 检测干预                  │
  │  step 4: 构建 transition          │
  │                                   │
  │ (episode 结束)                     │
  │                                   │
  │──── Transitions ─────────────────→│ 发送数据
  │                                   │  │
  │──── Interactions ────────────────→│  │ process_transitions()
  │                                   │  │ 添加到 buffer
  │                                   │  │
  │                                   │  │ 训练循环:
  │                                   │  │   sample batch
  │                                   │  │   update critic
  │                                   │  │   update actor
  │                                   │  │   update temperature
  │                                   │  │
  │←─── Parameters (更新后) ──────────│ 推送新参数
  │                                   │
  │  update_policy_parameters()       │
  │                                   │
  └─────────────────────────────────────┘ 继续下一个 episode
```

---

## 11. gRPC 通信协议

### 11.1 服务定义

```protobuf
// transport/services.proto

service LearnerService {
    // Actor 发送 transitions 到 Learner
    rpc SendTransitions (stream Transition) returns (Empty);

    // Actor 发送交互消息到 Learner
    rpc SendInteractions (stream InteractionMessage) returns (Empty);

    // Learner 推送策略参数到 Actor
    rpc StreamParameters (Empty) returns (stream Parameters);

    // 连接检查
    rpc Ready (Empty) returns (Empty);
}
```

### 11.2 消息格式

**Transition**:
```python
Transition = {
    "state": {
        "observation.images.front": Tensor[3, 128, 128],
        "observation.state": Tensor[6],
    },
    "action": Tensor[4],
    "reward": float,
    "next_state": {...},
    "done": bool,
    "truncated": bool,
    "complementary_info": {
        "discrete_penalty": float,
        "is_intervention": bool,
    }
}
```

**InteractionMessage**:
```python
InteractionMessage = {
    "Episodic reward": float,
    "Interaction step": int,
    "Episode intervention": int,
    "Intervention rate": float,
    "Policy frequency [Hz]": float,
}
```

**Parameters**:
```python
Parameters = {
    "policy": state_dict,  # Actor 网络参数
    "discrete_critic": state_dict,  # 离散 Critic 参数 (可选)
}
```

### 11.3 通信实现

数据通过 pickle 序列化后分块传输：

```python
# transport/utils.py

def send_bytes_in_chunks(buffer, message_class, chunk_size=64*1024):
    """将大数据分块发送"""
    for i in range(0, len(buffer), chunk_size):
        chunk = buffer[i:i+chunk_size]
        yield message_class(
            data=chunk,
            is_last=(i + chunk_size >= len(buffer))
        )

def receive_bytes_in_chunks(iterator, queue, shutdown_event):
    """接收分块数据并重组"""
    buffer = b""
    for msg in iterator:
        buffer += msg.data
        if msg.is_last:
            queue.put(buffer)
            buffer = b""
```

---

## 附录: 常用调试技巧

### A.1 查看 Buffer 状态

```python
# 在 Learner 日志中
logging.info(f"Buffer sizes - Online: {len(replay_buffer)}, Offline: {len(offline_replay_buffer)}")
```

### A.2 监控训练指标

#### 基础指标

| 指标 | 说明 | 期望趋势 |
|------|------|----------|
| `Episodic reward` | Episode 累计奖励 | ↑ 上升 |
| `Intervention rate` | 人类干预率 | ↓ 下降 |
| `loss_critic` | Critic 损失 | ↓ 下降后稳定 |
| `loss_actor` | Actor 损失 | 动态变化 |
| `temperature` | 温度参数 | 自动调节 |
| `Optimization frequency loop [Hz]` | 优化循环频率 | 稳定 |

#### Q值监控指标 (Critic)

| 指标 | 说明 | 异常判断 |
|------|------|----------|
| `q_mean` | Q值均值 | 监控Q值尺度 |
| `q_min` | Q值最小值 | 检测Q值下界 |
| `q_max` | Q值最大值 | >100 可能发散 |
| `q_std` | Q值标准差 | Q值分布宽度 |
| `q_critics_std` | Critics间标准差 | 集成分歧度，越大说明critics不一致 |
| `td_error_mean` | TD误差均值 | 不下降=未收敛 |
| `q_target_mean` | Target Q均值 | Bootstrap目标 |
| `reward_batch_mean` | Batch奖励均值 | 采样质量 |
| `reward_batch_min` | Batch奖励最小值 | 采样分布 |
| `reward_batch_max` | Batch奖励最大值 | 采样分布 |

#### 策略监控指标 (Actor)

| 指标 | 说明 | 异常判断 |
|------|------|----------|
| `action_entropy` | 动作熵 H(a\|s) = -E[log π(a\|s)] | 骤降=过早收敛，需增大temperature |
| `policy_mean_norm` | 策略均值范数 (Tanh前) | >5 可能Tanh饱和，梯度消失 |
| `actor_q_mean` | Actor动作的Q值均值 | 策略质量，应逐渐上升 |

#### 训练健康检查

```
问题诊断:
├── q_max 爆炸 (>100)      → Q值发散，降低lr或检查reward设计
├── action_entropy 骤降    → 策略过早收敛，增大temperature_init
├── q_critics_std 持续很大 → Critics不一致，可能过拟合，检查utd_ratio
├── td_error_mean 不降     → 训练未收敛，检查网络结构或学习率
├── Intervention rate 不降 → 策略未学会，检查reward信号
└── actor_q_mean 不升      → 策略质量未提升，检查actor_lr
```

#### 日志频率配置

```json
{
  "log_freq": 200  // 每200个optimization_step记录一次 (默认值)
                   // 换算: 200/utd_ratio=10个env_step ≈ 1秒 (10Hz时)
}
```

### A.3 检查点恢复

```bash
# 从上次检查点恢复训练
# 修改配置文件中的 "resume": true
./scripts/run_hilserl.sh train-learner configs/train_config.json
```

### A.4 常见问题

1. **Actor 连接失败**: 确保 Learner 先启动
2. **Buffer 不增长**: 检查 transitions 是否正常发送
3. **训练不开始**: 检查 `online_step_before_learning` 参数
4. **干预数据未记录**: 检查 `TeleopEvents.IS_INTERVENTION` 标记
