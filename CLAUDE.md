# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LeRobot is Hugging Face's robotics library providing models, datasets, and tools for real-world robotics in PyTorch. This fork (`Lerobot_hilserl`) adds **Piper robot arm support** for HIL-SERL (Human-in-the-Loop Sample Efficient Reinforcement Learning) training.

## Common Commands

### Installation
```bash
# Standard install
pip install -e .

# With HIL-SERL support (required for Piper robot)
pip install -e ".[hilserl]"

# With development dependencies
pip install -e ".[dev,test]"
```

### Testing
```bash
# Run all tests
pytest -sv ./tests

# Run single test file
pytest -sv tests/test_specific_feature.py

# End-to-end training tests (require GPU)
make test-end-to-end DEVICE=cuda
```

### Code Quality
```bash
# Install pre-commit hooks
pre-commit install

# Run all checks manually
pre-commit run --all-files
```

### Training and Evaluation
```bash
# Train a policy
lerobot-train --policy=act --dataset.repo_id=lerobot/aloha_mobile_cabinet

# Evaluate a policy
lerobot-eval --policy.path=<model_path> --env.type=aloha --eval.n_episodes=10
```

### HIL-SERL (Piper Robot)
```bash
# Record demonstrations with gamepad
./scripts/run_hilserl.sh record

# Train reward classifier
./scripts/run_hilserl.sh train-reward

# HIL-SERL training (requires two terminals)
./scripts/run_hilserl.sh train-learner  # Terminal 1
./scripts/run_hilserl.sh train-actor    # Terminal 2

# Test robot connection
./scripts/run_hilserl.sh test-robot
```

## Architecture

### Source Layout (`src/lerobot/`)
- **policies/**: Policy implementations (ACT, Diffusion, SAC, TDMPC, SmolVLA, Groot, Pi0, VQ-BeT, etc.)
- **robots/**: Hardware abstraction for robot control (SO100, Piper, Reachy2, etc.)
- **teleoperators/**: Teleoperation devices (gamepad, keyboard, phone)
- **cameras/**: Camera interfaces (OpenCV, RealSense)
- **motors/**: Motor control (Dynamixel, Feetech)
- **datasets/**: LeRobotDataset format handling and utilities
- **envs/**: Simulation environment wrappers
- **rl/**: Reinforcement learning components (actor-learner architecture for HIL-SERL)
- **scripts/**: CLI entry points (`lerobot-train`, `lerobot-eval`, etc.)
- **configs/**: Configuration dataclasses using draccus
- **processor/**: Data preprocessing pipelines for policies

### Key Design Patterns
- **Config-driven**: Uses `draccus` for typed configuration with JSON/YAML overrides
- **Factory pattern**: `policies/factory.py` handles policy instantiation
- **Unified Robot interface**: All robots implement a common `Robot` base class
- **LeRobotDataset**: Standardized format (Parquet + MP4) for robotics data

### Policy Architecture
Each policy in `policies/` contains:
- `configuration_<name>.py`: Config dataclass
- `modeling_<name>.py`: PyTorch model implementation

### Optional Dependencies
Different features require different extras (see `pyproject.toml`):
- `hilserl`: HIL-SERL reinforcement learning
- `smolvla`, `groot`, `pi`: VLA models
- `aloha`, `pusht`, `libero`: Simulation environments
- `feetech`, `dynamixel`: Motor controllers
