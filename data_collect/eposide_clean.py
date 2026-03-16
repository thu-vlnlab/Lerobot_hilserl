from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path
import shutil

# 1. 配置路径
root = Path("/home/ubuntu/dataset/cutting")
repo_id = "cutting"

# 2. 加载数据集
dataset = LeRobotDataset(repo_id, root=root)
last_episode_index = dataset.num_episodes - 1

print(f"[INFO] 当前数据集共有 {dataset.num_episodes} 个 episodes (0 到 {last_episode_index})")

confirm = input(f"[DANGER] 是否确认删除最后一个 episode (ID: {last_episode_index})? (y/n): ")

if confirm.lower() == 'y':
    # 3. 从元数据中移除最后一个 episode
    # 在 LeRobot 中，这通常需要重新过滤并覆盖 meta
    if last_episode_index >= 0:
        # 获取除了最后一个以外的所有 episode 索引
        episodes_to_keep = list(range(last_episode_index))
        
        # 这是一个 hack 方法：通过切片保留前面的数据并重新 consolidate
        # 注意：这会修改 meta/episodes.parquet
        dataset.meta.episodes = dataset.meta.episodes.iloc[:last_episode_index]
        dataset.meta.episodes.to_parquet(dataset.meta.root / "meta/episodes.parquet")
        
        # 4. 删除对应的视频文件夹（防止占用空间）
        # 视频路径通常在 videos/observation.cam_name/episode_N.mp4
        for cam_key in dataset.meta.camera_keys:
            video_path = root / repo_id / "videos" / cam_key / f"episode_{last_episode_index}.mp4"
            if video_path.exists():
                video_path.unlink()
                print(f"[INFO] 已删除视频: {video_path}")

        print(f"[SUCCESS] Episode {last_episode_index} 已从索引中移除。")
        print("[TIP] 建议随后运行 dataset.consolidate() 刷新状态。")
    else:
        print("[ERROR] 数据集已空，无需删除。")