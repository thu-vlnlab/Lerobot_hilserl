import pyrealsense2 as rs
import numpy as np
import time
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import signal
import atexit
from std_msgs.msg import Float32MultiArray
import cv2
from teleop_config import * 
from rclpy.node import Node

class LerobotDataCollect(Node):
    def __init__(self, repo_id, dataset_root, fps, cam_serials):
        super().__init__("lerobot_data_collector")
        self.repo_id = repo_id
        self.dataset_root = Path(dataset_root)
        self.fps = fps
        self.cam_serials = cam_serials
        self.pipelines = []
        self.recorder = None
        self.running = False
        self.num_episodes = 0
        self.skip_frames = 0

        self.state_sub = self.create_subscription(Float32MultiArray, '/state', self.state_update, 10)
        self.action_sub = self.create_subscription(Float32MultiArray, '/action', self.action_update, 10)

        self._init_dataset()
        self._init_cameras()

        self.released = False
        atexit.register(self._release_resources)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)


    def _init_dataset(self):
        meta_path = self.dataset_root / "meta" / "info.json"
        if meta_path.exists():
            self.recorder = LeRobotDataset(
                repo_id=self.repo_id,
                root=self.dataset_root,
            )
            self.recorder.start_image_writer(
                num_processes=0,
                num_threads=12,
            )
            self.num_episodes = self.recorder.num_episodes
            print(f"[INFO] 已加载数据集：{self.repo_id}")
        else:
            self.recorder = LeRobotDataset.create(
                repo_id=self.repo_id,
                fps=self.fps,
                root=self.dataset_root,
                features={
                    "observation.cam_wrist": {
                        "dtype": "video",
                        "shape": [480, 640, 3],
                        "names": [
                            "height",
                            "width",
                            "channels"
                        ]
                    },
                    "observation.cam_thrid_part": {
                        "dtype": "video",
                        "shape": [480, 640, 3],
                        "names": [
                            "height",
                            "width",
                            "channels"
                        ]
                    },
                    "observation.state": {
                        "dtype": "float",
                        "shape": [3],
                        "names": [
                            "x",
                            "y",
                            "z"
                        ]
                    },
                    "action": {
                        "dtype": "float",
                        "shape": [3],
                        "names": [
                            "x",
                            "y",
                            "z"
                        ]
                    },
                },
                use_videos=True,
                image_writer_processes=0,
                image_writer_threads=12,
            )
            print(f"[INFO] 新建数据集：{self.repo_id}")

    def _init_cameras(self):
        print("[INFO] 开始初始化 RealSense 相机...")
        success_count = 0
        for serial in self.cam_serials:
            try:
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, self.fps)

                pipeline.start(config)
                self.pipelines.append(pipeline)
                print(f"[INFO] 相机 {serial} 初始化成功")
                success_count += 1

            except Exception as e:
                print(f"[ERROR] 相机 {serial} 初始化失败: {e}")

        print(f"[INFO] 共初始化成功 {success_count}/{len(self.cam_serials)} 个 RealSense 相机。")

    def action_update(self, msg):
        self.action = np.array(msg.data, dtype=np.float64)
    
    def state_update(self, msg):
        self.state = np.array(msg.data, dtype=np.float64)

    def _signal_handler(self, sig, frame):
        print("\n[INFO] 检测到 Ctrl+C，准备结束采集...")
        self.running = False

    def collect_episode(self):
        frame_count = 0
        input("[INFO] 按 Enter 开始采集，Ctrl+C 结束后确认是否保存本次 episode。")
        time.sleep(0.5)
        self.running = True
        print("[INFO] 开始采集数据...")
        try:
            while self.running:
                
                try:
                    time_base = time.time()
                    frames = [p.wait_for_frames().get_color_frame() for p in self.pipelines]
                    if any(f is None for f in frames):
                        continue
                    images = [cv2.cvtColor(np.asanyarray(f.get_data()), cv2.COLOR_BGR2RGB) for f in frames]
                except Exception as e:
                    print(f"[ERROR] 相机帧获取失败: {e}")
                    continue


                frame_data = {
                    "observation.cam_wrist": images[0],
                    "observation.cam_thrid_part": images[1],
                    "observation.state": self.state,
                    "action": self.action,
                    "task": "Cutting tofa",
                }
                try:
                    if self.skip_frames == 0:
                        self.skip_frames = 1
                        continue
                    self.recorder.add_frame(frame_data)

                    #强制频率不超过fps
                    time_cost = time.time() - time_base
                    if time_cost < 1/self.fps:
                        time.sleep(1/self.fps - time_cost)
                    frame_count += 1
                except Exception as e:
                    print(f"[ERROR] 写入帧失败: {e}")

            print(f"[INFO] 已采集帧数：{frame_count}")
            
            choice = input("[INPUT] 是否保存本次采集？(y/n): ").strip().lower()
            if choice == 'y':
                self.recorder.save_episode()
                print("[INFO] Episode 已保存")
                self.recorder.stop_image_writer()
                self.recorder.clear_episode_buffer()

                
            else:
                self.recorder.clear_episode_buffer()
                print("[INFO] 本次采集未保存")

        except Exception as e:
            print(f"[FATAL] 采集中发生异常: {e}")
        finally:
            self._release_resources()

    def _release_resources(self):
        if self.released:
            return
        try:
            self.released = True
            for pipeline in self.pipelines:
                pipeline.stop()
        except Exception as e:
            print(f"[WARN] 相机释放失败: {e}")

        try:
            self.recorder.stop_image_writer()
        except Exception as e:
            print(f"[WARN] 停止图像线程失败: {e}")

        print("[INFO] 已释放资源。")


    

if __name__ == "__main__":
    
    collector = LerobotDataCollect(
        repo_id=REPO,
        dataset_root=REPOROOT,
        fps=20,
        cam_serials=CAMSERIALS
    )
    collector.collect_episode()
