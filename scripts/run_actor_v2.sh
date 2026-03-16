#!/bin/bash
# V2 Actor 训练脚本 - 移动方块任务

# 使用 GPU 渲染加速 (EGL)
export MUJOCO_GL=egl

./scripts/run_hilserl.sh train-actor configs_hilserl/v2_train.json
