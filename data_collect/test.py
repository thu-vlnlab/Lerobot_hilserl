import pyspacemouse
import sys
import os
import time
import numpy as np
from convert_axis import *
sys.path.append("/home/rimbot/teleop/spacemouse2rm75b")
from spacemouse_input import SpaceMouseReader





mouse = SpaceMouseReader()
mouse.open()
mouse.start()
raw = mouse.get_axes()
linear_speed = 0.01
start_pos = np.array([0.]*3)
print(raw)
while True:
    try:
        raw = np.array(mouse.get_axes())
        raw = np.array(convert_coordinate_system(raw,apply_extra_rotation=True))
        raw[:3] = raw[:3] *linear_speed
        start_pos += raw[:3] *linear_speed
        print(start_pos)
        time.sleep(0.03)
    except KeyboardInterrupt:
        break


Traceback (most recent call last):
  File "/home/ubuntu/miniconda3/envs/lerobot/lib/python3.10/threading.py", line 1016, in _bootstrap_inner
    self.run()
  File "/home/ubuntu/miniconda3/envs/lerobot/lib/python3.10/threading.py", line 953, in run
    self._target(*self._args, **self._kwargs)
  File "/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/executors.py", line 298, in spin
    self.spin_once()
  File "/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/executors.py", line 773, in spin_once
    self._spin_once_impl(timeout_sec)
  File "/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/executors.py", line 762, in _spin_once_impl
    handler, entity, node = self.wait_for_ready_callbacks(timeout_sec=timeout_sec)
  File "/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/executors.py", line 745, in wait_for_ready_callbacks
    return next(self._cb_iter)
  File "/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/executors.py", line 646, in _wait_for_ready_callbacks
    raise ExternalShutdownException()
rclpy.executors.ExternalShutdownException

Traceback (most recent call last):
  File "/home/ubuntu/Lerobot_hilserl/data_collect/data_record_with_tactile.py", line 229, in <module>
    rclpy.shutdown()
  File "/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/__init__.py", line 130, in shutdown
    _shutdown(context=context)
  File "/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/utilities.py", line 58, in shutdown
    return context.shutdown()
  File "/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/context.py", line 102, in shutdown
    self.__context.shutdown()
rclpy._rclpy_pybind11.RCLError: failed to shutdown: rcl_shutdown already called on the given context, at ./src/rcl/init.c:241