# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from lerobot.cameras import opencv, realsense  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.configs import HILSerlRobotEnvConfig
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    AddTeleopActionAsComplimentaryDataStep,
    AddTeleopEventsAsInfoStep,
    DataProcessorPipeline,
    DeviceProcessorStep,
    EnvTransition,
    GripperPenaltyProcessorStep,
    ImageCropResizeProcessorStep,
    InterventionActionProcessorStep,
    JointVelocityProcessorStep,
    MapDeltaActionToRobotActionStep,
    MapTensorToDeltaActionDictStep,
    MotorCurrentProcessorStep,
    Numpy2TorchActionProcessorStep,
    PiperDeltaToAbsoluteEEStep,
    RM75DeltaToAbsoluteEEStep,
    RewardClassifierProcessorStep,
    RobotActionToPolicyActionProcessorStep,
    TimeLimitProcessorStep,
    Torch2NumpyActionProcessorStep,
    TransitionKey,
    VanillaObservationProcessorStep,
    create_transition,
)
from lerobot.processor.converters import identity_transition
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so100_follower,
    piper_follower,
)
from lerobot.robots.robot import Robot
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    ForwardKinematicsJointsToEEObservation,
    GripperVelocityToJoint,
    InverseKinematicsRLStep,
)
from lerobot.teleoperators import (
    gamepad,  # noqa: F401
    keyboard,  # noqa: F401
    make_teleoperator_from_config,
    so101_leader,  # noqa: F401
)
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import ACTION, DONE, OBS_IMAGES, OBS_STATE, REWARD
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

logging.basicConfig(level=logging.INFO)


@dataclass
class DatasetConfig:
    """Configuration for dataset creation and management."""

    repo_id: str
    task: str
    root: str | None = None
    num_episodes_to_record: int = 5
    replay_episode: int | None = None
    push_to_hub: bool = False


@dataclass
class GymManipulatorConfig:
    """Main configuration for gym manipulator environment."""

    env: HILSerlRobotEnvConfig
    dataset: DatasetConfig
    mode: str | None = None  # Either "record", "replay", None
    device: str = "cpu"


def reset_follower_position(robot_arm: Robot, target_position: np.ndarray) -> None:
    """Reset robot arm to target position using smooth trajectory."""
    # For bus-based robots (SO100, Koch, etc.)
    if hasattr(robot_arm, 'bus') and hasattr(robot_arm.bus, 'sync_read'):
        current_position_dict = robot_arm.bus.sync_read("Present_Position")
        current_position = np.array(
            [current_position_dict[name] for name in current_position_dict], dtype=np.float32
        )
        trajectory = torch.from_numpy(
            np.linspace(current_position, target_position, 50)
        )
        for pose in trajectory:
            action_dict = dict(zip(current_position_dict, pose, strict=False))
            robot_arm.bus.sync_write("Goal_Position", action_dict)
            precise_sleep(0.015)
    # For Piper robots - use direct joint control for reset
    elif hasattr(robot_arm, 'JOINT_NAMES') and hasattr(robot_arm, '_piper'):
        obs = robot_arm.get_observation()
        joint_names = list(robot_arm.JOINT_NAMES)
        current_position = np.array(
            [obs.get(f"{name}.pos", 0) for name in joint_names], dtype=np.float32
        )
        # target_position has 7 values: 6 joints + gripper
        target_joints = target_position[:6]
        target_gripper = target_position[6] if len(target_position) > 6 else 35.0

        trajectory = np.linspace(current_position, target_joints, 50)

        # Set to joint control mode
        robot_arm._piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)
        precise_sleep(0.1)

        for pose in trajectory:
            # Convert degrees to 0.001 degrees for SDK
            joints_milli = [int(p * 1000) for p in pose]
            robot_arm._piper.JointCtrl(
                joint_1=joints_milli[0],
                joint_2=joints_milli[1],
                joint_3=joints_milli[2],
                joint_4=joints_milli[3],
                joint_5=joints_milli[4],
                joint_6=joints_milli[5],
            )
            precise_sleep(0.02)

        # Set gripper
        gripper_milli = int(target_gripper * 1000)
        robot_arm._piper.GripperCtrl(gripper_milli, 1000, 0x01, 0)

        # Switch back to end-effector control mode if needed
        robot_arm._piper.MotionCtrl_2(0x01, 0x00, 50, 0x00)
    # For RM75 robots - blocking joint-space reset
    elif hasattr(robot_arm, '_arm') and hasattr(robot_arm._arm, 'go_home'):
        target_joints_deg = target_position[:7].tolist()
        robot_arm._arm.go_home(joints_deg=target_joints_deg)
        if len(target_position) > 7:
            robot_arm._arm.set_gripper_position(float(target_position[7]))
        if hasattr(robot_arm, '_fixed_orientation'):
            robot_arm._fixed_orientation = None
    else:
        logging.warning("reset_follower_position: Unknown robot type, skipping reset")


class RobotEnv(gym.Env):
    """Gym environment for robotic control with human intervention support."""

    def __init__(
        self,
        robot,
        use_gripper: bool = False,
        display_cameras: bool = False,
        reset_pose: list[float] | None = None,
        reset_time_s: float = 5.0,
    ) -> None:
        """Initialize robot environment with configuration options.

        Args:
            robot: Robot interface for hardware communication.
            use_gripper: Whether to include gripper in action space.
            display_cameras: Whether to show camera feeds during execution.
            reset_pose: Joint positions for environment reset.
            reset_time_s: Time to wait during reset.
        """
        super().__init__()

        self.robot = robot
        self.display_cameras = display_cameras

        # Connect to the robot if not already connected.
        if not self.robot.is_connected:
            self.robot.connect()

        # Episode tracking.
        self.current_step = 0
        self.episode_data = None

        # Get joint names - support both bus-based robots (SO100) and direct robots (Piper)
        if hasattr(self.robot, 'bus') and hasattr(self.robot.bus, 'motors'):
            self._joint_names = list(self.robot.bus.motors.keys())
        elif hasattr(self.robot, 'JOINT_NAMES'):
            # Piper and similar robots
            self._joint_names = list(self.robot.JOINT_NAMES)
        else:
            # Fallback: extract from action_features
            self._joint_names = [k.replace('.pos', '') for k in self.robot.action_features.keys()
                                if k.endswith('.pos') and 'gripper' not in k]

        self._image_keys = self.robot.cameras.keys()

        self.reset_pose = reset_pose
        self.reset_time_s = reset_time_s

        self.use_gripper = use_gripper
        self._raw_joint_positions = None

        self._setup_spaces()

    def _get_observation(self) -> dict[str, Any]:
        """Get current robot observation including joint positions and camera images."""
        obs_dict = self.robot.get_observation()
        raw_joint_joint_position = {f"{name}.pos": obs_dict[f"{name}.pos"] for name in self._joint_names}
        joint_positions = np.array([raw_joint_joint_position[f"{name}.pos"] for name in self._joint_names])

        images = {key: obs_dict[key] for key in self._image_keys}

        result = {"agent_pos": joint_positions, "pixels": images, **raw_joint_joint_position}

        # Include EE position if available (for Piper EE control)
        for key in ["ee.x", "ee.y", "ee.z", "ee.rx", "ee.ry", "ee.rz", "gripper.pos"]:
            if key in obs_dict:
                result[key] = obs_dict[key]

        return result

    def _setup_spaces(self) -> None:
        """Configure observation and action spaces based on robot capabilities."""
        current_observation = self._get_observation()

        observation_spaces = {}

        # Define observation spaces for images and other states.
        if current_observation is not None and "pixels" in current_observation:
            prefix = OBS_IMAGES
            observation_spaces = {
                f"{prefix}.{key}": gym.spaces.Box(
                    low=0, high=255, shape=current_observation["pixels"][key].shape, dtype=np.uint8
                )
                for key in current_observation["pixels"]
            }

        if current_observation is not None:
            agent_pos = current_observation["agent_pos"]
            observation_spaces[OBS_STATE] = gym.spaces.Box(
                low=0,
                high=10,
                shape=agent_pos.shape,
                dtype=np.float32,
            )

        self.observation_space = gym.spaces.Dict(observation_spaces)

        # Define the action space for joint positions along with setting an intervention flag.
        action_dim = 3
        bounds = {}
        bounds["min"] = -np.ones(action_dim)
        bounds["max"] = np.ones(action_dim)

        if self.use_gripper:
            action_dim += 1
            bounds["min"] = np.concatenate([bounds["min"], [0]])
            bounds["max"] = np.concatenate([bounds["max"], [2]])

        self.action_space = gym.spaces.Box(
            low=bounds["min"],
            high=bounds["max"],
            shape=(action_dim,),
            dtype=np.float32,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options.

        Returns:
            Tuple of (observation, info) dictionaries.
        """
        # Reset the robot
        # self.robot.reset()
        start_time = time.perf_counter()
        if self.reset_pose is not None:
            log_say("Reset the environment.", play_sounds=True)
            reset_follower_position(self.robot, np.array(self.reset_pose))
            log_say("Reset the environment done.", play_sounds=True)

        precise_sleep(self.reset_time_s - (time.perf_counter() - start_time))

        super().reset(seed=seed, options=options)

        # Reset episode tracking variables.
        self.current_step = 0
        self.episode_data = None
        obs = self._get_observation()
        self._raw_joint_positions = {f"{key}.pos": obs[f"{key}.pos"] for key in self._joint_names}
        return obs, {TeleopEvents.IS_INTERVENTION: False}

    def step(self, action) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Execute one environment step with given action."""
        # Convert tensor to numpy if needed
        if hasattr(action, 'numpy'):
            action = action.numpy()
        elif hasattr(action, 'cpu'):
            action = action.cpu().numpy()

        # Check if robot uses end-effector control (e.g., PiperFollowerEndEffector)
        if "ee.x" in self.robot.action_features:
            # End-effector control: action is [x, y, z, gripper] or similar
            action_keys = list(self.robot.action_features.keys())
            action_dict = {key: float(action[i]) for i, key in enumerate(action_keys) if i < len(action)}
        else:
            # Joint control: action is joint positions
            action_dict = {f"{key}.pos": action[i] for i, key in enumerate(self._joint_names)}

        self.robot.send_action(action_dict)

        obs = self._get_observation()

        self._raw_joint_positions = {f"{key}.pos": obs[f"{key}.pos"] for key in self._joint_names}

        if self.display_cameras:
            self.render()

        self.current_step += 1

        reward = 0.0
        terminated = False
        truncated = False

        return (
            obs,
            reward,
            terminated,
            truncated,
            {TeleopEvents.IS_INTERVENTION: False},
        )

    def render(self) -> None:
        """Display robot camera feeds."""
        import cv2

        current_observation = self._get_observation()
        if current_observation is not None:
            image_keys = [key for key in current_observation if "image" in key]

            for key in image_keys:
                cv2.imshow(key, cv2.cvtColor(current_observation[key].numpy(), cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

    def close(self) -> None:
        """Close environment and disconnect robot."""
        if self.robot.is_connected:
            self.robot.disconnect()

    def get_raw_joint_positions(self) -> dict[str, float]:
        """Get raw joint positions."""
        return self._raw_joint_positions


def make_robot_env(cfg: HILSerlRobotEnvConfig) -> tuple[gym.Env, Any]:
    """Create robot environment from configuration.

    Args:
        cfg: Environment configuration.

    Returns:
        Tuple of (gym environment, teleoperator device).
    """
    # Check if this is a GymHIL simulation environment
    if cfg.name == "gym_hil":
        assert cfg.robot is None and cfg.teleop is None, "GymHIL environment does not support robot or teleop"
        import gym_hil  # noqa: F401

        # Extract gripper settings with defaults
        use_gripper = cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else True
        gripper_penalty = cfg.processor.gripper.gripper_penalty if cfg.processor.gripper is not None else 0.0

        env = gym.make(
            f"gym_hil/{cfg.task}",
            image_obs=True,
            render_mode="human",
            use_gripper=use_gripper,
            gripper_penalty=gripper_penalty,
        )

        return env, None

    # Real robot environment
    assert cfg.robot is not None, "Robot config must be provided for real robot environment"
    assert cfg.teleop is not None, "Teleop config must be provided for real robot environment"

    robot = make_robot_from_config(cfg.robot)
    teleop_device = make_teleoperator_from_config(cfg.teleop)
    teleop_device.connect()

    # Create base environment with safe defaults
    use_gripper = cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else True
    display_cameras = (
        cfg.processor.observation.display_cameras if cfg.processor.observation is not None else False
    )
    reset_pose = cfg.processor.reset.fixed_reset_joint_positions if cfg.processor.reset is not None else None

    env = RobotEnv(
        robot=robot,
        use_gripper=use_gripper,
        display_cameras=display_cameras,
        reset_pose=reset_pose,
    )

    return env, teleop_device


def make_processors(
    env: gym.Env, teleop_device: Teleoperator | None, cfg: HILSerlRobotEnvConfig, device: str = "cpu"
) -> tuple[
    DataProcessorPipeline[EnvTransition, EnvTransition], DataProcessorPipeline[EnvTransition, EnvTransition]
]:
    """Create environment and action processors.

    Args:
        env: Robot environment instance.
        teleop_device: Teleoperator device for intervention.
        cfg: Processor configuration.
        device: Target device for computations.

    Returns:
        Tuple of (environment processor, action processor).
    """
    terminate_on_success = (
        cfg.processor.reset.terminate_on_success if cfg.processor.reset is not None else True
    )

    if cfg.name == "gym_hil":
        action_pipeline_steps = [
            InterventionActionProcessorStep(terminate_on_success=terminate_on_success),
            Torch2NumpyActionProcessorStep(),
        ]

        env_pipeline_steps = [
            Numpy2TorchActionProcessorStep(),
            VanillaObservationProcessorStep(),
            AddBatchDimensionProcessorStep(),
            DeviceProcessorStep(device=device),
        ]

        return DataProcessorPipeline(
            steps=env_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
        ), DataProcessorPipeline(
            steps=action_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
        )

    # Full processor pipeline for real robot environment
    # Get robot and motor information for kinematics
    if hasattr(env.robot, 'bus') and hasattr(env.robot.bus, 'motors'):
        motor_names = list(env.robot.bus.motors.keys())
    elif hasattr(env.robot, 'JOINT_NAMES'):
        motor_names = list(env.robot.JOINT_NAMES)
    else:
        motor_names = [k.replace('.pos', '') for k in env.robot.action_features.keys()
                      if k.endswith('.pos') and 'gripper' not in k]

    # Set up kinematics solver if inverse kinematics is configured
    kinematics_solver = None
    if cfg.processor.inverse_kinematics is not None:
        kinematics_solver = RobotKinematics(
            urdf_path=cfg.processor.inverse_kinematics.urdf_path,
            target_frame_name=cfg.processor.inverse_kinematics.target_frame_name,
            joint_names=motor_names,
        )

    env_pipeline_steps = [VanillaObservationProcessorStep()]

    if cfg.processor.observation is not None:
        if cfg.processor.observation.add_joint_velocity_to_observation:
            env_pipeline_steps.append(JointVelocityProcessorStep(dt=1.0 / cfg.fps))
        if cfg.processor.observation.add_current_to_observation:
            env_pipeline_steps.append(MotorCurrentProcessorStep(robot=env.robot))

    if kinematics_solver is not None:
        env_pipeline_steps.append(
            ForwardKinematicsJointsToEEObservation(
                kinematics=kinematics_solver,
                motor_names=motor_names,
            )
        )

    if cfg.processor.image_preprocessing is not None:
        env_pipeline_steps.append(
            ImageCropResizeProcessorStep(
                crop_params_dict=cfg.processor.image_preprocessing.crop_params_dict,
                resize_size=cfg.processor.image_preprocessing.resize_size,
            )
        )

    # Add time limit processor if reset config exists
    if cfg.processor.reset is not None:
        env_pipeline_steps.append(
            TimeLimitProcessorStep(max_episode_steps=int(cfg.processor.reset.control_time_s * cfg.fps))
        )

    # Add gripper penalty processor if gripper config exists and enabled
    if cfg.processor.gripper is not None and cfg.processor.gripper.use_gripper:
        env_pipeline_steps.append(
            GripperPenaltyProcessorStep(
                penalty=cfg.processor.gripper.gripper_penalty,
                max_gripper_pos=cfg.processor.max_gripper_pos,
            )
        )

    if (
        cfg.processor.reward_classifier is not None
        and cfg.processor.reward_classifier.pretrained_path is not None
    ):
        env_pipeline_steps.append(
            RewardClassifierProcessorStep(
                pretrained_path=cfg.processor.reward_classifier.pretrained_path,
                device=device,
                success_threshold=cfg.processor.reward_classifier.success_threshold,
                success_reward=cfg.processor.reward_classifier.success_reward,
                terminate_on_success=terminate_on_success,
                position_constraint=getattr(cfg.processor.reward_classifier, 'position_constraint', None),
            )
        )

    env_pipeline_steps.append(AddBatchDimensionProcessorStep())
    env_pipeline_steps.append(DeviceProcessorStep(device=device))

    action_pipeline_steps = [
        AddTeleopActionAsComplimentaryDataStep(teleop_device=teleop_device),
        AddTeleopEventsAsInfoStep(teleop_device=teleop_device),
        InterventionActionProcessorStep(
            use_gripper=cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else False,
            terminate_on_success=terminate_on_success,
        ),
    ]

    # Check if robot uses end-effector control (e.g., PiperFollowerEndEffector, RM75FollowerEndEffector)
    uses_ee_control = hasattr(env.robot, 'action_features') and "ee.x" in env.robot.action_features

    # For robots with end-effector control and no IK, add delta to absolute converter
    if uses_ee_control and cfg.processor.inverse_kinematics is None:
        is_rm75_ee = getattr(env.robot, 'name', '') == 'rm75_follower_ee'
        if is_rm75_ee:
            # RM75-B: use dedicated processor (ee_step_size=1.0, translation_scale is the only speed knob)
            ee_bounds = {"x_min": -0.5, "x_max": 0.5, "y_min": -0.5, "y_max": 0.5, "z_min": 0.0, "z_max": 0.7}
            if hasattr(env.robot, 'config') and hasattr(env.robot.config, 'workspace_bounds'):
                wb = env.robot.config.workspace_bounds
                ee_bounds = {
                    "x_min": wb["min"][0], "x_max": wb["max"][0],
                    "y_min": wb["min"][1], "y_max": wb["max"][1],
                    "z_min": wb["min"][2], "z_max": wb["max"][2],
                }
            # use_suction controls whether suction enters the ACTION TENSOR (policy action space).
            # Use teleop's use_suction (not robot's enable_suction) — robot may have hardware
            # connected but still exclude suction from action space for BC/RL policy.
            # Physical relay is handled via side-channel regardless of this flag.
            _teleop_cfg = getattr(teleop_device, 'config', None)
            ee_use_suction = getattr(_teleop_cfg, 'use_suction', False)
            action_pipeline_steps.append(
                RM75DeltaToAbsoluteEEStep(
                    ee_step_size=1.0,
                    use_suction=ee_use_suction,
                    ee_bounds=ee_bounds,
                )
            )
        else:
            # Piper / other EE robots
            ee_use_gripper = cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else True
            max_gripper = cfg.processor.max_gripper_pos if cfg.processor.max_gripper_pos else 70.0
            ee_bounds = {"x_min": 0.136, "x_max": 0.38, "y_min": -0.159, "y_max": 0.220, "z_min": 0.135, "z_max": 0.3}
            if hasattr(env.robot, 'config') and hasattr(env.robot.config, 'workspace_bounds'):
                wb = env.robot.config.workspace_bounds
                ee_bounds = {
                    "x_min": wb["min"][0], "x_max": wb["max"][0],
                    "y_min": wb["min"][1], "y_max": wb["max"][1],
                    "z_min": wb["min"][2], "z_max": wb["max"][2],
                }
                max_gripper = getattr(env.robot.config, 'gripper_open_pos', max_gripper)
            action_pipeline_steps.append(
                PiperDeltaToAbsoluteEEStep(
                    ee_step_size=0.02,
                    max_gripper_pos=max_gripper,
                    use_gripper=ee_use_gripper,
                    ee_bounds=ee_bounds,
                )
            )

    # Replace InverseKinematicsProcessor with new kinematic processors
    if cfg.processor.inverse_kinematics is not None and kinematics_solver is not None:
        # Add EE bounds and safety processor
        inverse_kinematics_steps = [
            MapTensorToDeltaActionDictStep(
                use_gripper=cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else False
            ),
            MapDeltaActionToRobotActionStep(),
            EEReferenceAndDelta(
                kinematics=kinematics_solver,
                end_effector_step_sizes=cfg.processor.inverse_kinematics.end_effector_step_sizes,
                motor_names=motor_names,
                use_latched_reference=False,
                use_ik_solution=True,
            ),
            EEBoundsAndSafety(
                end_effector_bounds=cfg.processor.inverse_kinematics.end_effector_bounds,
            ),
            GripperVelocityToJoint(
                clip_max=cfg.processor.max_gripper_pos,
                speed_factor=1.0,
                discrete_gripper=True,
            ),
            InverseKinematicsRLStep(
                kinematics=kinematics_solver, motor_names=motor_names, initial_guess_current_joints=False
            ),
        ]
        action_pipeline_steps.extend(inverse_kinematics_steps)
        action_pipeline_steps.append(RobotActionToPolicyActionProcessorStep(motor_names=motor_names))

    return DataProcessorPipeline(
        steps=env_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
    ), DataProcessorPipeline(
        steps=action_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
    )


def step_env_and_process_transition(
    env: gym.Env,
    transition: EnvTransition,
    action: torch.Tensor,
    env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
) -> EnvTransition:
    """
    Execute one step with processor pipeline.

    Args:
        env: The robot environment
        transition: Current transition state
        action: Action to execute
        env_processor: Environment processor
        action_processor: Action processor

    Returns:
        Processed transition with updated state.
    """

    # Create action transition
    transition[TransitionKey.ACTION] = action

    # Supply ee.x/y/z to action processor (RM75 processor needs it for one-time init).
    # Use _last_pose cached from previous get_observation() — zero extra TCP calls.
    robot = getattr(env, 'robot', None)
    last_pose = getattr(robot, '_last_pose', None)
    if last_pose is not None and len(last_pose) >= 3:
        transition[TransitionKey.OBSERVATION] = {
            "ee.x": last_pose[0], "ee.y": last_pose[1], "ee.z": last_pose[2],
        }
    elif hasattr(env, "get_raw_joint_positions"):
        transition[TransitionKey.OBSERVATION] = env.get_raw_joint_positions()

    processed_action_transition = action_processor(transition)
    processed_action = processed_action_transition[TransitionKey.ACTION]

    obs, reward, terminated, truncated, info = env.step(processed_action)

    # Side-channel suction: read _suction_state directly from the teleop object
    # (InterventionActionProcessorStep converts the teleop dict to a tensor, losing suction.state)
    _suction_ctrl = getattr(getattr(env, 'robot', None), '_suction', None)
    if _suction_ctrl is not None:
        _teleop = None
        for _step in action_processor.steps:
            if hasattr(_step, 'teleop_device'):
                _teleop = _step.teleop_device
                break
        if _teleop is not None and hasattr(_teleop, '_suction_state'):
            _suction_ctrl.set_state(bool(_teleop._suction_state))

    reward = reward + processed_action_transition[TransitionKey.REWARD]
    terminated = terminated or processed_action_transition[TransitionKey.DONE]
    truncated = truncated or processed_action_transition[TransitionKey.TRUNCATED]
    complementary_data = processed_action_transition[TransitionKey.COMPLEMENTARY_DATA].copy()
    # Keep teleop events from the processor; env info should not clobber them.
    new_info = info.copy()
    new_info.update(processed_action_transition[TransitionKey.INFO])

    new_transition = create_transition(
        observation=obs,
        action=processed_action,
        reward=reward,
        done=terminated,
        truncated=truncated,
        info=new_info,
        complementary_data=complementary_data,
    )
    new_transition = env_processor(new_transition)

    # Carry RM75 target through for record-time action delta computation
    if "rm75_target_xyz" in processed_action_transition:
        new_transition["rm75_target_xyz"] = processed_action_transition["rm75_target_xyz"]

    return new_transition


def piper_control_loop(
    env: gym.Env,
    env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    teleop_device: Teleoperator,
    cfg: GymManipulatorConfig,
) -> None:
    """
    Direct control loop for Piper robot - bypasses complex processor pipeline.
    Similar to test_piper_gamepad_control.py for reliable control.
    """
    dt = 1.0 / cfg.env.fps
    LINEAR_VELOCITY_SCALE = 50.0  # mm/s per unit delta
    GRIPPER_STEP = 5.0  # mm per step

    print(f"Starting Piper control loop at {cfg.env.fps} FPS")
    print("Controls:")
    print("  [IMPORTANT] Hold LB button to enable control!")
    print("  Left stick Y: X direction (forward/backward)")
    print("  Left stick X: Y direction (left/right)")
    print("  Right stick Y: Z direction (up/down)")
    print("  Y button: End episode with SUCCESS")
    print("  A button: End episode with FAILURE")
    print("  X button: Rerecord episode")
    print("  Press Ctrl+C to exit")

    # Get direct access to Piper robot
    robot = env.robot
    piper = robot._piper

    # Reset environment
    obs, info = env.reset()
    env_processor.reset()

    # Read current EE position from robot
    end_pose = piper.GetArmEndPoseMsgs()
    current_pose = [
        end_pose.end_pose.X_axis / 1000.0,  # 0.001mm -> mm
        end_pose.end_pose.Y_axis / 1000.0,
        end_pose.end_pose.Z_axis / 1000.0,
        end_pose.end_pose.RX_axis / 1000.0,  # 0.001deg -> deg
        end_pose.end_pose.RY_axis / 1000.0,
        end_pose.end_pose.RZ_axis / 1000.0,
    ]
    gripper_pos = 0.0  # mm

    print(f"Initial EE position: X={current_pose[0]:.1f}mm Y={current_pose[1]:.1f}mm Z={current_pose[2]:.1f}mm")

    # Process initial observation
    transition = create_transition(observation=obs, info=info)
    transition = env_processor(data=transition)

    # Determine if gripper is used
    use_gripper = cfg.env.processor.gripper.use_gripper if cfg.env.processor.gripper is not None else True

    # Setup dataset if recording
    dataset = None
    if cfg.mode == "record":
        action_features = teleop_device.action_features if teleop_device else {
            "dtype": "float32", "shape": (4,), "names": None
        }
        features = {
            ACTION: action_features,
            REWARD: {"dtype": "float32", "shape": (1,), "names": None},
            DONE: {"dtype": "bool", "shape": (1,), "names": None},
        }
        if use_gripper:
            features["complementary_info.discrete_penalty"] = {
                "dtype": "float32", "shape": (1,), "names": ["discrete_penalty"],
            }
        for key, value in transition[TransitionKey.OBSERVATION].items():
            if key == OBS_STATE:
                features[key] = {"dtype": "float32", "shape": value.squeeze(0).shape, "names": None}
            if "image" in key:
                features[key] = {"dtype": "video", "shape": value.squeeze(0).shape, "names": ["channels", "height", "width"]}

        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id, cfg.env.fps, root=cfg.dataset.root,
            use_videos=True, image_writer_threads=4, image_writer_processes=0, features=features,
        )

    episode_idx = 0
    episode_step = 0
    episode_start_time = time.perf_counter()
    success_locked = False  # Lock success state once Y is pressed

    # Get success_reward from config (default to 1.0 if not specified)
    success_reward = 1.0
    if cfg.env.processor.reward_classifier is not None:
        success_reward = getattr(cfg.env.processor.reward_classifier, 'success_reward', 1.0)
    logging.info(f"Using success_reward: {success_reward}")

    try:
        while episode_idx < cfg.dataset.num_episodes_to_record:
            step_start_time = time.perf_counter()

            # Get teleop input directly
            teleop_action = teleop_device.get_action()
            teleop_events = teleop_device.get_teleop_events()

            is_intervention = teleop_events.get(TeleopEvents.IS_INTERVENTION, False)
            success = teleop_events.get(TeleopEvents.SUCCESS, False)
            failure = teleop_events.get(TeleopEvents.TERMINATE_EPISODE, False)
            rerecord = teleop_events.get(TeleopEvents.RERECORD_EPISODE, False)

            # Lock success state once triggered
            if success:
                success_locked = True

            # Control robot directly when intervention is active
            if is_intervention:
                # Get delta from teleop
                delta_x = teleop_action.get("delta_x", 0.0)
                delta_y = teleop_action.get("delta_y", 0.0)
                delta_z = teleop_action.get("delta_z", 0.0)

                # Calculate velocity and update position
                vel_x = delta_x * LINEAR_VELOCITY_SCALE
                vel_y = delta_y * LINEAR_VELOCITY_SCALE
                vel_z = delta_z * LINEAR_VELOCITY_SCALE

                current_pose[0] += vel_x * dt
                current_pose[1] += vel_y * dt
                current_pose[2] += vel_z * dt

                # Send command to robot
                X = int(current_pose[0] * 1000)
                Y = int(current_pose[1] * 1000)
                Z = int(current_pose[2] * 1000)
                RX = int(current_pose[3] * 1000)
                RY = int(current_pose[4] * 1000)
                RZ = int(current_pose[5] * 1000)

                piper.MotionCtrl_2(0x01, 0x00, 50, 0x00)
                piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)

                if episode_step % 10 == 0:
                    print(f"\r[控制中] Step {episode_step} X={current_pose[0]:.1f} Y={current_pose[1]:.1f} Z={current_pose[2]:.1f}   ", end="")
            else:
                # Read current position to stay in sync
                end_pose = piper.GetArmEndPoseMsgs()
                current_pose = [
                    end_pose.end_pose.X_axis / 1000.0,
                    end_pose.end_pose.Y_axis / 1000.0,
                    end_pose.end_pose.Z_axis / 1000.0,
                    end_pose.end_pose.RX_axis / 1000.0,
                    end_pose.end_pose.RY_axis / 1000.0,
                    end_pose.end_pose.RZ_axis / 1000.0,
                ]
                if episode_step % 10 == 0:
                    print(f"\r[等待LB] Step {episode_step} X={current_pose[0]:.1f} Y={current_pose[1]:.1f} Z={current_pose[2]:.1f}   ", end="")

            # Handle gripper
            gripper_action = teleop_action.get("gripper", 1)
            if gripper_action == 0:  # Close
                gripper_pos = max(0.0, gripper_pos - GRIPPER_STEP)
                piper.GripperCtrl(int(gripper_pos * 1000), 1000, 0x01, 0)
            elif gripper_action == 2:  # Open
                gripper_pos = min(70.0, gripper_pos + GRIPPER_STEP)
                piper.GripperCtrl(int(gripper_pos * 1000), 1000, 0x01, 0)

            # Get observation for recording
            obs = env._get_observation()

            # Record frame
            if cfg.mode == "record" and dataset is not None:
                # Process observation through env_processor for correct format
                temp_transition = create_transition(observation=obs, info={})
                temp_transition = env_processor(temp_transition)

                observations = {
                    k: v.squeeze(0).cpu()
                    for k, v in temp_transition[TransitionKey.OBSERVATION].items()
                    if isinstance(v, torch.Tensor)
                }

                if use_gripper:
                    action_to_record = torch.tensor([
                        teleop_action.get("delta_x", 0.0),
                        teleop_action.get("delta_y", 0.0),
                        teleop_action.get("delta_z", 0.0),
                        float(teleop_action.get("gripper", 1)),
                    ], dtype=torch.float32)
                else:
                    action_to_record = torch.tensor([
                        teleop_action.get("delta_x", 0.0),
                        teleop_action.get("delta_y", 0.0),
                        teleop_action.get("delta_z", 0.0),
                    ], dtype=torch.float32)

                frame = {
                    **observations,
                    ACTION: action_to_record,
                    REWARD: np.array([success_reward if success_locked else 0.0], dtype=np.float32),
                    DONE: np.array([success_locked or failure or rerecord], dtype=bool),
                }
                if use_gripper:
                    frame["complementary_info.discrete_penalty"] = np.array([0.0], dtype=np.float32)
                frame["task"] = cfg.dataset.task
                dataset.add_frame(frame)

            episode_step += 1

            # Handle episode termination
            # terminate_on_success controls whether success ends the episode
            terminate_on_success = cfg.env.processor.reset.terminate_on_success if cfg.env.processor.reset else True
            terminated = failure or rerecord or (success_locked and terminate_on_success)

            if terminated:
                print(f"\n[INFO] Episode ended: success={success}, failure={failure}, rerecord={rerecord}")
                episode_time = time.perf_counter() - episode_start_time
                logging.info(f"Episode ended after {episode_step} steps in {episode_time:.1f}s")

                if dataset is not None:
                    if rerecord:
                        logging.info(f"Re-recording episode {episode_idx}")
                        dataset.clear_episode_buffer()
                    else:
                        episode_idx += 1
                        logging.info(f"Saving episode {episode_idx}")
                        dataset.save_episode()

                episode_step = 0
                episode_start_time = time.perf_counter()
                success_locked = False  # Reset success lock for new episode

                # Reset environment
                obs, info = env.reset()
                env_processor.reset()

                # Re-read current position
                end_pose = piper.GetArmEndPoseMsgs()
                current_pose = [
                    end_pose.end_pose.X_axis / 1000.0,
                    end_pose.end_pose.Y_axis / 1000.0,
                    end_pose.end_pose.Z_axis / 1000.0,
                    end_pose.end_pose.RX_axis / 1000.0,
                    end_pose.end_pose.RY_axis / 1000.0,
                    end_pose.end_pose.RZ_axis / 1000.0,
                ]

            # Maintain fps timing
            precise_sleep(dt - (time.perf_counter() - step_start_time))

    except KeyboardInterrupt:
        print("\n\n[INFO] 退出控制，保持当前位置...")
        # Keep current position
        X = int(current_pose[0] * 1000)
        Y = int(current_pose[1] * 1000)
        Z = int(current_pose[2] * 1000)
        RX = int(current_pose[3] * 1000)
        RY = int(current_pose[4] * 1000)
        RZ = int(current_pose[5] * 1000)
        piper.MotionCtrl_2(0x01, 0x00, 10, 0x00)
        piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)

    if dataset is not None and cfg.dataset.push_to_hub:
        logging.info("Pushing dataset to hub")
        dataset.push_to_hub()


def control_loop(
    env: gym.Env,
    env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    teleop_device: Teleoperator,
    cfg: GymManipulatorConfig,
) -> None:
    """Main control loop for robot environment interaction.
    if cfg.mode == "record": then a dataset will be created and recorded

    Args:
     env: The robot environment
     env_processor: Environment processor
     action_processor: Action processor
     teleop_device: Teleoperator device
     cfg: gym_manipulator configuration
    """
    dt = 1.0 / cfg.env.fps

    # Frequency decoupling: control at cfg.env.fps, record at cfg.env.record_fps
    _record_fps = getattr(cfg.env, 'record_fps', None) or cfg.env.fps
    record_every = max(1, round(cfg.env.fps / _record_fps))
    if record_every > 1:
        print(f"Control @ {cfg.env.fps} Hz, recording @ {_record_fps} Hz (every {record_every} steps)")

    print(f"Starting control loop at {cfg.env.fps} FPS")
    print("Controls:")
    print("- Use gamepad/teleop device for intervention")
    print("- When not intervening, robot will stay still")
    print("- Press Ctrl+C to exit")

    # Reset environment and processors
    obs, info = env.reset()
    complementary_data = (
        {"raw_joint_positions": info.pop("raw_joint_positions")} if "raw_joint_positions" in info else {}
    )
    env_processor.reset()
    action_processor.reset()

    # Process initial observation
    transition = create_transition(observation=obs, info=info, complementary_data=complementary_data)
    transition = env_processor(data=transition)

    # Determine if gripper is used
    use_gripper = cfg.env.processor.gripper.use_gripper if cfg.env.processor.gripper is not None else True

    dataset = None
    if cfg.mode == "record":
        # For gym_hil environments, teleop_device is None, so we get action features from env
        if teleop_device is not None:
            action_features = teleop_device.action_features
        else:
            # gym_hil environment: action space is (4,) for [x, y, z, gripper]
            action_shape = env.action_space.shape
            action_features = {"dtype": "float32", "shape": action_shape, "names": None}
        features = {
            ACTION: action_features,
            REWARD: {"dtype": "float32", "shape": (1,), "names": None},
            DONE: {"dtype": "bool", "shape": (1,), "names": None},
        }
        if use_gripper:
            features["complementary_info.discrete_penalty"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": ["discrete_penalty"],
            }

        for key, value in transition[TransitionKey.OBSERVATION].items():
            if key == OBS_STATE:
                features[key] = {
                    "dtype": "float32",
                    "shape": value.squeeze(0).shape,
                    "names": None,
                }
            if "image" in key:
                features[key] = {
                    "dtype": "video",
                    "shape": value.squeeze(0).shape,
                    "names": ["channels", "height", "width"],
                }

        # Create dataset — use record_fps so video/parquet timestamps are correct
        _dataset_fps = getattr(cfg.env, 'record_fps', None) or cfg.env.fps
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            _dataset_fps,
            root=cfg.dataset.root,
            use_videos=True,
            image_writer_threads=4,
            image_writer_processes=0,
            features=features,
        )

    episode_idx = 0
    episode_step = 0
    episode_start_time = time.perf_counter()

    # Frequency monitor
    _freq_last_time = time.perf_counter()
    _freq_step_count = 0

    while episode_idx < cfg.dataset.num_episodes_to_record:
        step_start_time = time.perf_counter()

        # Create a neutral action (no movement)
        neutral_action = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        if use_gripper:
            neutral_action = torch.cat([neutral_action, torch.tensor([1.0])])  # Gripper stay

        # Use the new step function
        transition = step_env_and_process_transition(
            env=env,
            transition=transition,
            action=neutral_action,
            env_processor=env_processor,
            action_processor=action_processor,
        )
        terminated = transition.get(TransitionKey.DONE, False)
        truncated = transition.get(TransitionKey.TRUNCATED, False)

        if cfg.mode == "record" and (episode_step % record_every == 0):
            observations = {
                k: v.squeeze(0).cpu()
                for k, v in transition[TransitionKey.OBSERVATION].items()
                if isinstance(v, torch.Tensor)
            }

            # For RM75 EE: action = target_xyz - obs_ee_xyz (real displacement in meters)
            is_rm75_ee = getattr(getattr(env, 'robot', None), 'name', '') == 'rm75_follower_ee'
            if is_rm75_ee:
                # Get the internal target from the RM75 processor step
                rm75_target = transition.get("rm75_target_xyz", None)
                if rm75_target is not None:
                    obs_dict = transition[TransitionKey.OBSERVATION]
                    def _v(v): return float(v.item() if hasattr(v, 'item') else v)
                    obs_x = _v(obs_dict.get("ee.x", rm75_target[0]))
                    obs_y = _v(obs_dict.get("ee.y", rm75_target[1]))
                    obs_z = _v(obs_dict.get("ee.z", rm75_target[2]))
                    action_vals = [rm75_target[0] - obs_x, rm75_target[1] - obs_y, rm75_target[2] - obs_z]
                    # Append suction state if enabled
                    action_from_teleop = transition[TransitionKey.COMPLEMENTARY_DATA].get("teleop_action")
                    if action_from_teleop is not None and not isinstance(action_from_teleop, torch.Tensor):
                        action_from_teleop = torch.tensor(action_from_teleop, dtype=torch.float32)
                    if action_from_teleop is not None and len(action_from_teleop) > 3:
                        action_vals.append(float(action_from_teleop[3]))
                    action_to_record = torch.tensor(action_vals, dtype=torch.float32)
                else:
                    action_to_record = transition[TransitionKey.ACTION]
                    if not isinstance(action_to_record, torch.Tensor):
                        action_to_record = torch.tensor(action_to_record, dtype=torch.float32)
            else:
                # Use teleop_action if available, otherwise use the action from the transition
                # For gym_hil, teleop_action is in INFO (priority); for real robot, it's in COMPLEMENTARY_DATA
                action_to_record = transition[TransitionKey.INFO].get("teleop_action")
                if action_to_record is None:
                    action_to_record = transition[TransitionKey.COMPLEMENTARY_DATA].get("teleop_action")
                if action_to_record is None:
                    action_to_record = transition[TransitionKey.ACTION]
                if not isinstance(action_to_record, torch.Tensor):
                    action_to_record = torch.tensor(action_to_record, dtype=torch.float32)

            # Ensure no batch dimension on action
            action_to_record = action_to_record.squeeze(0) if action_to_record.dim() > 1 else action_to_record

            frame = {
                **observations,
                ACTION: action_to_record.cpu(),
                REWARD: np.array([transition[TransitionKey.REWARD]], dtype=np.float32),
                DONE: np.array([terminated or truncated], dtype=bool),
            }
            if use_gripper:
                discrete_penalty = transition[TransitionKey.COMPLEMENTARY_DATA].get("discrete_penalty", 0.0)
                frame["complementary_info.discrete_penalty"] = np.array([discrete_penalty], dtype=np.float32)

            if dataset is not None:
                frame["task"] = cfg.dataset.task
                dataset.add_frame(frame)

        episode_step += 1

        # Handle episode termination
        if terminated or truncated:
            episode_time = time.perf_counter() - episode_start_time
            logging.info(
                f"Episode ended after {episode_step} steps in {episode_time:.1f}s with reward {transition[TransitionKey.REWARD]}"
            )
            episode_step = 0
            episode_idx += 1

            if dataset is not None:
                if transition[TransitionKey.INFO].get(TeleopEvents.RERECORD_EPISODE, False):
                    logging.info(f"Re-recording episode {episode_idx}")
                    dataset.clear_episode_buffer()
                    episode_idx -= 1
                else:
                    logging.info(f"Saving episode {episode_idx}")
                    dataset.save_episode()

            # Reset for new episode
            obs, info = env.reset()
            env_processor.reset()
            action_processor.reset()

            transition = create_transition(observation=obs, info=info)
            transition = env_processor(transition)

        # Maintain fps timing
        precise_sleep(dt - (time.perf_counter() - step_start_time))

        # Print actual frequency every 100 steps
        _freq_step_count += 1
        if _freq_step_count >= 100:
            _now = time.perf_counter()
            actual_hz = _freq_step_count / (_now - _freq_last_time)
            print(f"\r[Hz] actual={actual_hz:.1f}  target={cfg.env.fps}  step={episode_step}", end="", flush=True)
            _freq_last_time = _now
            _freq_step_count = 0

    if dataset is not None and cfg.dataset.push_to_hub:
        logging.info("Pushing dataset to hub")
        dataset.push_to_hub()


def replay_trajectory(
    env: gym.Env, action_processor: DataProcessorPipeline, cfg: GymManipulatorConfig
) -> None:
    """Replay recorded trajectory on robot environment."""
    assert cfg.dataset.replay_episode is not None, "Replay episode must be provided for replay"

    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=[cfg.dataset.replay_episode],
        download_videos=False,
    )
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == cfg.dataset.replay_episode)
    actions = episode_frames.select_columns(ACTION)

    _, info = env.reset()

    for action_data in actions:
        start_time = time.perf_counter()
        transition = create_transition(
            observation=env.get_raw_joint_positions() if hasattr(env, "get_raw_joint_positions") else {},
            action=action_data[ACTION],
        )
        transition = action_processor(transition)
        env.step(transition[TransitionKey.ACTION])
        precise_sleep(1 / cfg.env.fps - (time.perf_counter() - start_time))


@parser.wrap()
def main(cfg: GymManipulatorConfig) -> None:
    """Main entry point for gym manipulator script."""
    env, teleop_device = make_robot_env(cfg.env)
    env_processor, action_processor = make_processors(env, teleop_device, cfg.env, cfg.device)

    print("Environment observation space:", env.observation_space)
    print("Environment action space:", env.action_space)
    print("Environment processor:", env_processor)
    print("Action processor:", action_processor)

    if cfg.mode == "replay":
        replay_trajectory(env, action_processor, cfg)
        exit()

    # Use Piper-specific control loop for Piper robots
    is_piper = hasattr(env, 'robot') and hasattr(env.robot, '_piper')
    if is_piper:
        print("\n[INFO] Detected Piper robot, using direct control loop")
        piper_control_loop(env, env_processor, teleop_device, cfg)
    else:
        control_loop(env, env_processor, action_processor, teleop_device, cfg)


if __name__ == "__main__":
    main()
