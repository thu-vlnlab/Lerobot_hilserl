#!/usr/bin/env python
"""
Real-time reward classifier visualization.
Press 'q' to quit.
"""
"""
python scripts/realtime_reward_classifier.py \
      --model-path outputs/reward_classifier_piper2 \
      --realsense \
      --serial 338622070324
"""

import argparse
import cv2
import torch
import numpy as np
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="outputs/reward_classifier_piper2",
                        help="Path to trained reward classifier")
    parser.add_argument("--camera", type=int, default=0, help="Camera index or use 'realsense'")
    parser.add_argument("--realsense", action="store_true", help="Use Intel RealSense camera")
    parser.add_argument("--serial", type=str, default="338622070324", help="RealSense serial number")
    parser.add_argument("--threshold", type=float, default=0.5, help="Success threshold")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model_path}...")
    from lerobot.policies.sac.reward_model.modeling_classifier import Classifier
    from lerobot.policies.sac.reward_model.configuration_classifier import RewardClassifierConfig

    model_path = Path(args.model_path)

    # Check if it's a training output directory with checkpoints
    pretrained_path = model_path / "checkpoints" / "last" / "pretrained_model"
    if pretrained_path.exists():
        model_path = pretrained_path
        print(f"Found checkpoint at {model_path}")

    # Load config manually to handle extra fields
    import json
    config_path = model_path / "config.json"
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Remove fields that are not part of RewardClassifierConfig
    config_dict.pop("type", None)
    config_dict.pop("n_obs_steps", None)
    config_dict.pop("output_features", None)
    config_dict.pop("repo_id", None)
    config_dict.pop("private", None)
    config_dict.pop("tags", None)
    config_dict.pop("license", None)
    config_dict.pop("pretrained_path", None)

    config = RewardClassifierConfig(**config_dict)
    model = Classifier(config)

    # Load weights (直接加载，不需要重映射 - encoder 和 encoders[xxx][0] 是同一对象)
    weights_path = model_path / "model.safetensors"
    if weights_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(weights_path)
        result = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights: {len(state_dict)} keys, missing: {len(result.missing_keys)}")
    else:
        # Try pytorch format
        weights_path = model_path / "pytorch_model.bin"
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()
    print("Model loaded!")

    # Load normalization stats from checkpoint
    from safetensors.torch import load_file as load_safetensors
    norm_path = model_path / "classifier_preprocessor_step_0_normalizer_processor.safetensors"
    img_mean = None
    img_std = None
    if norm_path.exists():
        norm_stats = load_safetensors(norm_path)
        # Find image normalization keys
        image_key = list(config.input_features.keys())[0]
        mean_key = f"{image_key}.mean"
        std_key = f"{image_key}.std"
        if mean_key in norm_stats and std_key in norm_stats:
            img_mean = norm_stats[mean_key].numpy()  # shape [3, 1, 1]
            img_std = norm_stats[std_key].numpy()
            print(f"Loaded normalization: mean={img_mean.flatten()}, std={img_std.flatten()}")
        else:
            print(f"Warning: Normalization keys not found")
    else:
        print(f"Warning: Normalization file not found: {norm_path}")

    # Setup camera
    if args.realsense:
        import pyrealsense2 as rs
        pipeline = rs.pipeline()
        config_rs = rs.config()
        config_rs.enable_device(args.serial)
        config_rs.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config_rs)
        print(f"RealSense camera started (serial: {args.serial})")
    else:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"OpenCV camera {args.camera} started")

    # Image preprocessing (match training: MEAN_STD normalization)
    def preprocess_image(frame):
        # Resize to 128x128 (match training)
        img = cv2.resize(frame, (128, 128))
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))  # shape [3, 128, 128]
        # Apply MEAN_STD normalization if available
        if img_mean is not None and img_std is not None:
            img = (img - img_mean) / img_std
        # Add batch dimension
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        return img

    print("\nPress 'q' to quit")
    print("-" * 40)

    try:
        while True:
            # Get frame
            if args.realsense:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                frame = np.asanyarray(color_frame.get_data())
            else:
                ret, frame = cap.read()
                if not ret:
                    continue

            # Preprocess
            img_tensor = preprocess_image(frame)

            # Predict
            with torch.no_grad():
                # Call predict directly with image list
                output = model.predict([img_tensor])

                # Get probability (binary classification)
                if output.probabilities is not None:
                    success_prob = output.probabilities[0].item()
                else:
                    # Fallback to sigmoid of logits
                    success_prob = torch.sigmoid(output.logits[0]).item()

            # Determine result
            is_success = success_prob >= args.threshold

            # Draw on frame
            color = (0, 255, 0) if is_success else (0, 0, 255)
            status = "SUCCESS" if is_success else "FAILURE"

            # Draw status
            cv2.putText(frame, f"{status}: {success_prob:.2%}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            # Draw progress bar
            bar_width = int(400 * success_prob)
            cv2.rectangle(frame, (10, 60), (410, 90), (100, 100, 100), -1)
            cv2.rectangle(frame, (10, 60), (10 + bar_width, 90), color, -1)
            cv2.rectangle(frame, (10, 60), (410, 90), (255, 255, 255), 2)

            # Draw threshold line
            threshold_x = 10 + int(400 * args.threshold)
            cv2.line(frame, (threshold_x, 55), (threshold_x, 95), (255, 255, 0), 2)

            # Show frame
            cv2.imshow("Reward Classifier", frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        if args.realsense:
            pipeline.stop()
        else:
            cap.release()
        cv2.destroyAllWindows()
        print("Done")

if __name__ == "__main__":
    main()
