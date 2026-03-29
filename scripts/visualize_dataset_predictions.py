#!/usr/bin/env python
"""
Visualize reward classifier predictions on dataset.
Usage:
    python scripts/visualize_dataset_predictions.py \
        --model-path outputs/reward_classifier_piper2 \
        --data-path data/demos5
"""

import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="outputs/reward_classifier_piper2")
    parser.add_argument("--data-path", type=str, default="data/demos5")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for output video")
    parser.add_argument("--fast", action="store_true", help="Fast mode: only print statistics, no visualization")
    parser.add_argument("--output", type=str, default="predictions_output.mp4", help="Output annotated video path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model_path}...")
    from lerobot.policies.sac.reward_model.modeling_classifier import Classifier
    from lerobot.policies.sac.reward_model.configuration_classifier import RewardClassifierConfig

    model_path = Path(args.model_path)
    pretrained_path = model_path / "checkpoints" / "last" / "pretrained_model"
    if pretrained_path.exists():
        model_path = pretrained_path

    # Load config
    config_path = model_path / "config.json"
    with open(config_path, "r") as f:
        config_dict = json.load(f)

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

    # Load weights
    from safetensors.torch import load_file
    weights_path = model_path / "model.safetensors"
    state_dict = load_file(weights_path)

    model_keys = list(model.state_dict().keys())
    print(f"Model state_dict keys: {len(model_keys)}")
    print("Model keys (first 15):")
    for k in model_keys[:15]:
        print(f"  {k}")
    print(f"Checkpoint keys: {len(state_dict)}")

    # Get the image key for mapping
    image_key = list(config.input_features.keys())[0].replace(".", "_")

    # Expand checkpoint: copy encoder.xxx to encoders.{image_key}.0.xxx
    new_state_dict = dict(state_dict)
    for k, v in state_dict.items():
        if k.startswith("encoder."):
            # Also add to encoders
            new_key = k.replace("encoder.", f"encoders.{image_key}.0.")
            new_state_dict[new_key] = v

    # Check if encoder and encoders[0] are the same object
    image_key_for_check = list(model.encoders.keys())[0]
    is_same = model.encoder is model.encoders[image_key_for_check][0]
    print(f"encoder is encoders[xxx][0]: {is_same}")

    # Direct load
    result = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {len(result.missing_keys)}")
    print(f"Unexpected keys: {len(result.unexpected_keys)}")

    if result.missing_keys and len(result.missing_keys) < 40:
        print("Missing keys (not loaded):")
        for k in result.missing_keys[:10]:
            print(f"  - {k}")

    model = model.to(device)
    model.eval()
    print("Model loaded!")

    # Load normalization stats from checkpoint
    norm_path = model_path / "classifier_preprocessor_step_0_normalizer_processor.safetensors"
    img_mean = None
    img_std = None
    if norm_path.exists():
        norm_stats = load_file(norm_path)
        # Find image normalization keys
        image_key = list(config.input_features.keys())[0]
        mean_key = f"{image_key}.mean"
        std_key = f"{image_key}.std"
        if mean_key in norm_stats and std_key in norm_stats:
            img_mean = norm_stats[mean_key].numpy()  # shape [3, 1, 1]
            img_std = norm_stats[std_key].numpy()
            print(f"Loaded normalization: mean={img_mean.flatten()}, std={img_std.flatten()}")
        else:
            print(f"Warning: Normalization keys not found in {norm_path}")
            print(f"Available keys: {list(norm_stats.keys())}")
    else:
        print(f"Warning: Normalization file not found: {norm_path}")

    # Load dataset
    data_path = Path(args.data_path)
    print(f"Loading dataset from {data_path}...")

    # Load parquet data
    import pandas as pd
    parquet_files = list((data_path / "data").glob("**/*.parquet"))
    if not parquet_files:
        print("No parquet files found!")
        return

    df = pd.concat([pd.read_parquet(f) for f in parquet_files])
    print(f"Loaded {len(df)} frames")

    # Load video
    video_dir = data_path / "videos"
    video_key = None
    for key in ["observation.images.observation.images.front", "observation.images.front"]:
        video_path = video_dir / key / "chunk-000" / "file-000.mp4"
        if video_path.exists():
            video_key = key
            break

    if video_key is None:
        # Find any video
        video_files = list(video_dir.glob("**/*.mp4"))
        if video_files:
            video_path = video_files[0]
        else:
            print("No video files found!")
            return

    print(f"Using video: {video_path}")

    # Open video with PyAV (software decoding, supports AV1)
    import av
    av_container = av.open(str(video_path))
    av_stream = av_container.streams.video[0]
    av_stream.codec_context.thread_type = av.codec.context.ThreadType.AUTO
    total_frames = av_stream.frames if av_stream.frames else 0
    fps = float(av_stream.average_rate) if av_stream.average_rate else 30.0
    print(f"Video: {total_frames} frames at {fps:.1f} fps")

    # Get ground truth rewards
    rewards = df["next.reward"].values if "next.reward" in df.columns else None

    # Preprocess function (match training: MEAN_STD normalization)
    def preprocess_image(frame):
        img = cv2.resize(frame, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW, shape [3, 128, 128]
        # Apply MEAN_STD normalization if available
        if img_mean is not None and img_std is not None:
            img = (img - img_mean) / img_std
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        return img

    # Process each frame
    predictions = []
    frame_idx = 0

    if args.fast:
        # Fast mode: process all frames without visualization
        print("\nFast mode: processing all frames...")
        for packet in av_container.demux(av_stream):
            for av_frame in packet.decode():
                frame = av_frame.to_ndarray(format="bgr24")

                img_tensor = preprocess_image(frame)
                with torch.no_grad():
                    output = model.predict([img_tensor])
                    pred_prob = output.probabilities[0].item() if output.probabilities is not None else torch.sigmoid(output.logits[0]).item()

                predictions.append(pred_prob)
                frame_idx += 1
                if frame_idx % 500 == 0:
                    print(f"  {frame_idx}/{total_frames} frames...")

        print(f"Processed {frame_idx} frames")
    else:
        # Visualization mode — pre-decode all frames for looping support
        print("\nDecoding video frames...")
        all_frames = []
        for packet in av_container.demux(av_stream):
            for av_frame in packet.decode():
                all_frames.append(av_frame.to_ndarray(format="bgr24"))
        print(f"Decoded {len(all_frames)} frames. Press 'q' to quit, Space to pause/resume")
        total_frames = len(all_frames)
        paused = False
        display_frame = None

        while True:
            if not paused:
                if frame_idx >= len(all_frames):
                    # Loop back
                    frame_idx = 0

                frame = all_frames[frame_idx]

                # Predict
                img_tensor = preprocess_image(frame)
                with torch.no_grad():
                    output = model.predict([img_tensor])
                    pred_prob = output.probabilities[0].item() if output.probabilities is not None else torch.sigmoid(output.logits[0]).item()

                predictions.append(pred_prob)

                # Get ground truth
                gt_reward = rewards[frame_idx] if rewards is not None and frame_idx < len(rewards) else None

                # Scale up first
                if args.scale != 1.0:
                    display_frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_LINEAR)
                else:
                    display_frame = frame.copy()

                # Draw results on scaled frame
                is_success = pred_prob >= args.threshold
                color = (0, 255, 0) if is_success else (0, 0, 255)

                # Draw prediction (top left)
                cv2.putText(display_frame, f"Pred: {pred_prob:.1%}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Draw ground truth
                if gt_reward is not None:
                    gt_color = (0, 255, 0) if gt_reward > 0.5 else (0, 0, 255)
                    cv2.putText(display_frame, f"GT: {int(gt_reward)}", (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, gt_color, 2)

                # Draw frame info (top right)
                cv2.putText(display_frame, f"{frame_idx}/{total_frames}", (display_frame.shape[1] - 100, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Draw prediction bar (bottom)
                bar_height = 20
                bar_y = display_frame.shape[0] - bar_height - 10
                bar_max_width = 200
                bar_x = 10
                # Background
                cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_max_width, bar_y + bar_height), (50, 50, 50), -1)
                # Fill based on probability
                fill_width = int(bar_max_width * pred_prob)
                cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
                # Border
                cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_max_width, bar_y + bar_height), (255, 255, 255), 1)
                # Threshold line
                threshold_x = bar_x + int(bar_max_width * args.threshold)
                cv2.line(display_frame, (threshold_x, bar_y - 3), (threshold_x, bar_y + bar_height + 3), (0, 255, 255), 2)

                frame_idx += 1

            if display_frame is not None:
                cv2.imshow("Dataset Predictions", display_frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                print("Paused" if paused else "Resumed")

    av_container.close()
    cv2.destroyAllWindows()

    # Print summary
    if predictions:
        predictions = np.array(predictions)
        print(f"\n{'='*50}")
        print(f"=== 统计结果 ===")
        print(f"{'='*50}")
        print(f"总帧数: {len(predictions)}")
        print(f"预测概率均值: {predictions.mean():.2%}")
        print(f"预测为成功的帧数: {(predictions >= args.threshold).sum()} ({(predictions >= args.threshold).mean():.1%})")

        if rewards is not None and len(rewards) >= len(predictions):
            gt = rewards[:len(predictions)]
            pred = (predictions >= args.threshold).astype(float)
            gt_binary = (gt > 0.5).astype(float)

            # 计算混淆矩阵
            TP = ((pred == 1) & (gt_binary == 1)).sum()  # 真阳性：预测成功，实际成功
            TN = ((pred == 0) & (gt_binary == 0)).sum()  # 真阴性：预测失败，实际失败
            FP = ((pred == 1) & (gt_binary == 0)).sum()  # 假阳性：预测成功，实际失败
            FN = ((pred == 0) & (gt_binary == 1)).sum()  # 假阴性：预测失败，实际成功

            accuracy = (TP + TN) / len(predictions)
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0

            print(f"\n--- Ground Truth ---")
            print(f"实际成功帧: {int(gt_binary.sum())} ({gt_binary.mean():.1%})")
            print(f"实际失败帧: {int((1-gt_binary).sum())} ({(1-gt_binary).mean():.1%})")

            print(f"\n--- 混淆矩阵 ---")
            print(f"TP (预测成功, 实际成功): {TP}")
            print(f"TN (预测失败, 实际失败): {TN}")
            print(f"FP (预测成功, 实际失败): {FP} <- 误报")
            print(f"FN (预测失败, 实际成功): {FN} <- 漏报")

            print(f"\n--- 指标 ---")
            print(f"准确率 Accuracy: {accuracy:.1%}")
            print(f"精确率 Precision: {precision:.1%}")
            print(f"召回率 Recall: {recall:.1%}")
            print(f"{'='*50}")

if __name__ == "__main__":
    main()
