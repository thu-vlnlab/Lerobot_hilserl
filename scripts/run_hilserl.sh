#!/bin/bash
# HIL-SERL Training Scripts for Piper Robot

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(dirname $(dirname $(realpath $0)))/src"
export PIPER_SDK_PATH="/home/qzl/data/piper/piper_sdk"
export PYTHONPATH="${PYTHONPATH}:${PIPER_SDK_PATH}"

CONFIG_DIR="$(dirname $(dirname $(realpath $0)))/configs_hilserl"

# Helper to resolve optional --config_path or positional overrides
resolve_config_path() {
    local default_path="$1"
    shift
    local config_path=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --config_path)
                shift
                if [ $# -eq 0 ]; then
                    echo "Error: --config_path requires a value" >&2
                    exit 1
                fi
                config_path="$1"
                shift
                ;;
            --*)
                shift
                ;;
            *)
                if [ -z "$config_path" ]; then
                    config_path="$1"
                fi
                shift
                ;;
        esac
    done
    if [ -z "$config_path" ]; then
        config_path="$default_path"
    fi
    echo "$config_path"
}

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
    echo "  setup-can        - Setup CAN interface for Piper robot"
    echo "  record [config]  - Record demos for reward classifier (terminate_on_success=false)"
    echo "  record-expert    - Record expert demos (terminate_on_success=true, for offline buffer)"
    echo "  crop             - Interactive ROI cropping tool"
    echo "  train-reward [config]  - Train reward classifier"
    echo "  train-learner [config] - Start learner (terminal 1)"
    echo "  train-actor [config]   - Start actor (terminal 2)"
    echo "  test-robot       - Test robot connection"
    echo ""
    echo "Examples:"
    echo "  $0 setup-can                 # Setup CAN interface"
    echo "  $0 record                    # Record for reward classifier training"
    echo "  $0 record-expert             # Record expert demos (success=terminate)"
    echo "  $0 train-reward              # Train reward classifier"
    echo "  $0 train-learner             # Start learner (terminal 1)"
    echo "  $0 train-actor               # Start actor (terminal 2)"
}

setup_can() {
    echo -e "${YELLOW}Setting up CAN interface...${NC}"
    sudo ip link set can0 type can bitrate 1000000
    sudo ip link set up can0
    echo -e "${GREEN}CAN interface ready${NC}"
}

case "$1" in
    record)
        echo -e "${GREEN}Starting demonstration recording (for reward classifier)...${NC}"
        echo -e "${YELLOW}terminate_on_success=false: continue recording after success${NC}"
        CONFIG_PATH=$(resolve_config_path "${CONFIG_DIR}/env_config_piper_real.json" "${@:2}")
        python -m lerobot.rl.gym_manipulator --config "${CONFIG_PATH}"
        ;;

    record-expert)
        echo -e "${GREEN}Starting expert demonstration recording...${NC}"
        echo -e "${YELLOW}terminate_on_success=true: episode ends on success${NC}"
        CONFIG_PATH=$(resolve_config_path "${CONFIG_DIR}/env_config_piper_expert.json" "${@:2}")
        python -m lerobot.rl.gym_manipulator --config "${CONFIG_PATH}"
        ;;

    crop)
        echo -e "${GREEN}Starting ROI cropping tool...${NC}"
        REPO_ID=${2:-"thu-vlnlab/piper_pick_lift"}
        python -m lerobot.rl.crop_dataset_roi --repo-id "${REPO_ID}"
        ;;

    train-reward)
        echo -e "${GREEN}Training reward classifier...${NC}"
        CONFIG_PATH=$(resolve_config_path "${CONFIG_DIR}/reward_classifier_config_piper.json" "${@:2}")
        python -m lerobot.scripts.lerobot_train --config "${CONFIG_PATH}"
        ;;

    train-learner)
        echo -e "${GREEN}Starting HIL-SERL Learner...${NC}"
        echo -e "${YELLOW}Make sure to start the actor in another terminal!${NC}"
        CONFIG_PATH=$(resolve_config_path "${CONFIG_DIR}/train_config_hilserl_piper_real.json" "${@:2}")
        python -m lerobot.rl.learner --config "${CONFIG_PATH}"
        ;;

    train-actor)
        echo -e "${GREEN}Starting HIL-SERL Actor...${NC}"
        echo -e "${YELLOW}Make sure the learner is already running!${NC}"
        CONFIG_PATH=$(resolve_config_path "${CONFIG_DIR}/train_config_hilserl_piper_real.json" "${@:2}")
        python -m lerobot.rl.actor --config "${CONFIG_PATH}"
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
