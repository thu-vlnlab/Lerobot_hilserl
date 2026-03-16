#!/bin/bash
# V2 分类器训练脚本 - 移动方块任务

# 使用 HuggingFace 镜像解决网络问题
export HF_ENDPOINT=https://hf-mirror.com

python -m lerobot.scripts.lerobot_train --config_path configs_hilserl/v2_reward_classifier.json
