#!/bin/bash
# HIL-SERL Training Scripts for Piper Robot

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(dirname $(dirname $(realpath $0)))/src"
export PIPER_SDK_PATH="/home/qzl/data/piper/piper_sdk"
export PYTHONPATH="${PYTHONPATH}:${PIPER_SDK_PATH}"

CONFIG_DIR="$(dirname $(dirname $(realpath $0)))/configs_hilserl"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_usage() {
    echo -e "${GREEN}HIL-SERL Training Scripts for Piper Robot${NC}"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  record          - Record demonstration episodes"
    echo "  crop            - Interactive ROI cropping tool"
    echo "  train-reward    - Train reward classifier"
    echo "  train-learner   - Start HIL-SERL learner process"
    echo "  train-actor     - Start HIL-SERL actor process"
    echo "  test-robot      - Test robot connection"
    echo ""
    echo "Examples:"
    echo "  $0 record                    # Record 15 demo episodes"
    echo "  $0 train-learner             # Start learner in terminal 1"
    echo "  $0 train-actor               # Start actor in terminal 2"
}

setup_can() {
    echo -e "${YELLOW}Setting up CAN interface...${NC}"
    sudo ip link set can0 type can bitrate 1000000
    sudo ip link set up can0
    echo -e "${GREEN}CAN interface ready${NC}"
}

case "$1" in
    record)
        echo -e "${GREEN}Starting demonstration recording...${NC}"
        python -m lerobot.rl.gym_manipulator --config_path "${CONFIG_DIR}/env_config_piper.json"
        ;;

    crop)
        echo -e "${GREEN}Starting ROI cropping tool...${NC}"
        REPO_ID=${2:-"thu-vlnlab/piper_pick_lift"}
        python -m lerobot.rl.crop_dataset_roi --repo-id "${REPO_ID}"
        ;;

    train-reward)
        echo -e "${GREEN}Training reward classifier...${NC}"
        python -m lerobot.scripts.lerobot_train --config_path "${CONFIG_DIR}/reward_classifier_config.json"
        ;;

    train-learner)
        echo -e "${GREEN}Starting HIL-SERL Learner...${NC}"
        echo -e "${YELLOW}Make sure to start the actor in another terminal!${NC}"
        python -m lerobot.rl.learner --config_path "${CONFIG_DIR}/train_config_hilserl.json"
        ;;

    train-actor)
        echo -e "${GREEN}Starting HIL-SERL Actor...${NC}"
        echo -e "${YELLOW}Make sure the learner is already running!${NC}"
        python -m lerobot.rl.actor --config_path "${CONFIG_DIR}/train_config_hilserl.json"
        ;;

    test-robot)
        echo -e "${GREEN}Testing Piper robot connection...${NC}"
        python -c "
import sys
sys.path.insert(0, '${PIPER_SDK_PATH}')
from piper_sdk import C_PiperInterface_V2

piper = C_PiperInterface_V2(can_name='can0')
piper.ConnectPort()
print('Robot connected successfully!')

# Read joint states
joint_msgs = piper.GetArmJointMsgs()
print(f'Joint states: {joint_msgs.joint_state}')

piper.DisconnectPort()
print('Robot disconnected.')
"
        ;;

    setup-can)
        setup_can
        ;;

    *)
        print_usage
        exit 1
        ;;
esac
