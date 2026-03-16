#!/usr/bin/env python
"""
合并数据集脚本
"""
import os
os.environ["HF_HOME"] = "/home/ubuntu/Lerobot_hilserl/.hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/home/ubuntu/Lerobot_hilserl/.hf_cache/datasets"

from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets

# 加载数据集 - root 是完整路径
ds1 = LeRobotDataset('demosv3', root='/home/ubuntu/Lerobot_hilserl/data/demosv3')
ds2 = LeRobotDataset('demosv3_2', root='/home/ubuntu/Lerobot_hilserl/data/demosv3_2')

# 合并
out_dir = Path('/home/ubuntu/Lerobot_hilserl/data/demov3merge')
merge_datasets([ds1, ds2], output_repo_id='demov3merge', output_dir=out_dir)
print('Merged into', out_dir)
