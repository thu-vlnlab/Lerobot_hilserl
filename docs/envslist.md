gym_hil 可用的环境：

  | 环境 ID                              | 任务       | 控制方式              |
  |--------------------------------------|------------|-----------------------|
  | PandaPickCube 系列                   | 抓取立方体 |                       |
  | gym_hil/PandaPickCubeBase-v0         | 基础环境   | 无输入控制            |
  | gym_hil/PandaPickCube-v0             | 带 wrapper | Agent 控制            |
  | gym_hil/PandaPickCubeGamepad-v0      | 带手柄     | 手柄控制 ← 你现在用的 |
  | gym_hil/PandaPickCubeKeyboard-v0     | 带键盘     | 键盘控制              |
  | gym_hil/PandaPickCubeViewer-v0       | 带查看器   | 仅查看                |
  | PandaArrangeBoxes 系列               | 整理箱子   |                       |
  | gym_hil/PandaArrangeBoxesBase-v0     | 基础环境   | 无输入控制            |
  | gym_hil/PandaArrangeBoxes-v0         | 带 wrapper | Agent 控制            |
  | gym_hil/PandaArrangeBoxesGamepad-v0  | 带手柄     | 手柄控制              |
  | gym_hil/PandaArrangeBoxesKeyboard-v0 | 带键盘     | 键盘控制              |

  文件位置：
  /home/qzl/Environment/miniconda3/envs/lerobot/lib/python3.10/site-packages/gym_hil/
  ├── __init__.py                    # 环境注册
  ├── envs/
  │   ├── panda_pick_gym_env.py      # 抓取立方体任务
  │   └── panda_arrange_boxes_gym_env.py  # 整理箱子任务
  └── assets/                        # MuJoCo 模型文件

  两个任务的区别：

  | 任务              | 目标           | 难度 |
  |-------------------|----------------|------|
  | PandaPickCube     | 抓起一个立方体 | 简单 |
  | PandaArrangeBoxes | 整理多个箱子   | 复杂 |