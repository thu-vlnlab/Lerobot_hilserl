import numpy as np
import time
import rclpy
import threading
import signal
import atexit
import cv2
from pathlib import Path

# --- 替换 RealSense 为 Orbbec SDK ---
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode, Context, OBError
from pyorbbecsdk import *
# ----------------------------------

from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Float32MultiArray
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from orbbec_cam import *
# 配置文件加载

from teleop_config import *



EXIT_FLAG = False

def global_signal_handler(sig, frame):
    global EXIT_FLAG
    EXIT_FLAG = True
    print("\n[INFO] 接收到中断信号，正在安全退出...")

signal.signal(signal.SIGINT, global_signal_handler)
signal.signal(signal.SIGTERM, global_signal_handler)

class LerobotDataCollect(Node):
    def __init__(self, repo_id, dataset_root, fps, cam_serials):
        super().__init__("lerobot_data_collector")
        
        self.repo_id = repo_id
        self.dataset_root = Path(dataset_root)
        self.fps = fps
        self.cam_serials = cam_serials
        self.pipelines = []
        self.recorder = None
        self.released = False

        # 数据缓冲
        self.state = np.zeros(3, dtype=np.float32)
        self.action = np.zeros(3, dtype=np.float32)
        self.tactile = np.zeros(1, dtype=np.float32)

        # ROS2 订阅
        self.state_sub = self.create_subscription(Float32MultiArray, '/state', self.state_update, 10)
        self.action_sub = self.create_subscription(Float32MultiArray, '/action', self.action_update, 10)
        self.tactile_sub = self.create_subscription(Float32MultiArray, '/tactile', self.tactile_update, 10)

        self._init_dataset()
        self._init_cameras() # 初始化 Gemini 305

        atexit.register(self._release_resources)

    def _init_dataset(self):
        """初始化 LeRobot 数据集"""
        meta_path = self.dataset_root  / "meta" / "info.json"
        print(meta_path)
        if meta_path.exists():
            print(f"[INFO] 加载现有数据集: {self.repo_id}")
            self.recorder = LeRobotDataset(repo_id=self.repo_id, root=self.dataset_root)
            self.recorder.start_image_writer(num_processes=0, num_threads=12)
        else:
            print(f"[INFO] 创建新数据集: {self.repo_id}")
            self.recorder = LeRobotDataset.create(
                repo_id=self.repo_id,
                fps=self.fps,
                root=self.dataset_root,
                features={
                    "observation.cam_wrist": {"dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channels"]},
                    "observation.cam_thrid_Nodepart": {"dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channels"]},
                    "observation.state": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
                    "tactile": {"dtype": "float32", "shape": (1,), "names": ["mf"]},
                    "action": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
                },
                use_videos=True,
                image_writer_processes=0,
                image_writer_threads=12,
            )

    def _init_cameras(self):
        """初始化 Orbbec Gemini 305 相机"""
        print(f"[INFO] 正在初始化 {len(self.cam_serials)} 个 Orbbec 相机...")
        ctx = Context()
        device_list = ctx.query_devices()
        
        for serial in self.cam_serials:
            found = False
            try:
                # 在连接列表中寻找对应序列号的设备
                for i in range(device_list.get_count()):
                    device = device_list.get_device_by_index(i)
                    if device.get_device_info().get_serial_number() == serial:
                        pipe = Pipeline(device)
                        config = Config()
                        
                        # 配置彩色流 (640x480, 30fps, RGB格式)
                        
                        profiles = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
                        video_profile = profiles.get_video_stream_profile(640, 480, OBFormat.RGB, 30)
                        config.enable_stream(video_profile)

                        pipe.start(config)
                        self.pipelines.append(pipe)
                        print(f"[INFO] Gemini 305 {serial} 已就绪")
                        found = True
                        break
                
                if not found:
                    print(f"[ERROR] 未找到序列号为 {serial} 的设备")
            except Exception as e:
                print(f"[ERROR] 相机 {serial} 启动失败: {e}")

        if len(self.pipelines) < 2:
            raise RuntimeError("相机数量不足，请检查硬件连接和序列号配置")

    def state_update(self, msg): self.state = np.array(msg.data, dtype=np.float32)
    def action_update(self, msg): self.action = np.array(msg.data, dtype=np.float32)
    def tactile_update(self, msg): self.tactile = np.array(msg.data, dtype=np.float32)

    def collect_episode(self):
        """运行单次录制"""
        global EXIT_FLAG
        executor = SingleThreadedExecutor()
        executor.add_node(self)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        input("\n[READY] 按下 Enter 开始采集本轮 (Gemini 305)...")
        print("[RUNNING] 录制中，Ctrl+C 停止...")
        
        EXIT_FLAG = False
        frame_count = 0
        
        try:
            while not EXIT_FLAG:
                start_t = time.time()
                images = []
                
                # 1. 读取所有相机帧
                try:
                    for pipe in self.pipelines:
                        frames = pipe.wait_for_frames(1000)
                        if frames is None:
                            raise ValueError("获取不到帧数据")
                        
                        color_frame = frames.get_color_frame()
                        if color_frame is None:
                            raise ValueError("彩色帧为空")
                        color_image = frame_to_rgb_image(color_frame)
                        
                        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # LeRobot 需要 RGB
                        
                        images.append(color_image)
                        
                except Exception as e:
                    print(f"[WARN] 帧获取失败: {e}")
                    continue

                # 2. 构造数据包
                frame_data = {
                    "observation.cam_wrist": images[0],
                    "observation.cam_thrid_part": images[1],
                    "observation.state": self.state,
                    "tactile": self.tactile,
                    "action": self.action,
                    "task": "cutting tofu"
                }

                # 3. 写入缓存
                self.recorder.add_frame(frame_data)
                frame_count += 1

                # 4. 频率控制
                elapsed = time.time() - start_t
                delay = (1.0 / self.fps) - elapsed
                if delay > 0:
                    time.sleep(delay)
                
                if frame_count % 20 == 0:
                    print(f"  > 已录制 {frame_count} 帧", end="\r")

            print(f"\n[FINISH] 录制结束，共 {frame_count} 帧")
            save_choice = input("[INPUT] 是否保存？(y/n): ").strip().lower()
            if save_choice == 'y':
                self.recorder.save_episode()
                print("[INFO] 已保存")
            else:
                self.recorder.clear_episode_buffer()
                print("[INFO] 已丢弃")

        finally:
            executor.shutdown()

    def _release_resources(self):
        if self.released: return
        self.released = True
        print("\n[CLEANUP] 关闭中...")
        try:
            if self.recorder:
                self.recorder.stop_image_writer()
                self.recorder.finalize()
            for p in self.pipelines:
                p.stop()
            print("[INFO] Gemini 305 相机已安全关闭")
        except Exception as e:
            print(f"[ERROR] 释放资源错误: {e}")

if __name__ == "__main__":
    rclpy.init()
    collector = None
    try:
        collector = LerobotDataCollect(
            repo_id=REPO,
            dataset_root=REPOROOT,
            fps=20,
            cam_serials=CAMSERIALS
        )
        collector.collect_episode()
    except KeyboardInterrupt:
        pass
    finally:
        if collector:
            collector._release_resources()
        print("[EXIT] 程序退出")