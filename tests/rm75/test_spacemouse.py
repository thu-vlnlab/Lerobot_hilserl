from lerobot.teleoperators.spacemouse import SpaceMouseTeleop, SpaceMouseTeleopConfig
from lerobot.teleoperators.utils import TeleopEvents

config = SpaceMouseTeleopConfig()
teleop = SpaceMouseTeleop(config)
teleop.connect()

import time
for _ in range(100):
    action = teleop.get_action()
    events = teleop.get_teleop_events()
    print(f"delta: ({action['delta_x']:.4f}, {action['delta_y']:.4f}, {action['delta_z']:.4f})"
          f"  gripper: {action.get('gripper', '-')}"
          f"  intervening: {events[TeleopEvents.IS_INTERVENTION]}")
    time.sleep(0.02)

teleop.disconnect()