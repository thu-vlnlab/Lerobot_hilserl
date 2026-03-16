#!/usr/bin/env python
"""Test the trained reward classifier on the recorded dataset."""

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.sac.reward_model.modeling_classifier import Classifier
from lerobot.policies.factory import make_pre_post_processors


def main():
    # === Configuration ===
    model_path = "/home/qzl/data/Lerobot_hilserl/outputs/train/2026-01-07/17-55-28_reward_classifier/checkpoints/last/pretrained_model"
    dataset_root = "/home/qzl/data/sim_demos"
    repo_id = "test/sim_demo"
    device = "cuda"

    # === Load model ===
    print(f"Loading classifier from {model_path}...")
    classifier = Classifier.from_pretrained(model_path)
    classifier.to(device)
    classifier.eval()
    print("Classifier loaded!")

    # === Load dataset ===
    print(f"\nLoading dataset from {dataset_root}...")
    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)
    print(f"Dataset: {len(dataset)} frames")

    # === Create preprocessor ===
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=classifier.config,
        dataset_stats=dataset.meta.stats
    )

    # === Test on dataset ===
    print("\n=== Testing on dataset ===")

    correct = 0
    total = 0

    # Test on a subset
    test_indices = list(range(0, len(dataset), max(1, len(dataset) // 50)))  # Sample ~50 frames

    results = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    for idx in test_indices:
        sample = dataset[idx]

        # Get ground truth
        gt_reward = sample["next.reward"].item()
        gt_label = "SUCCESS" if gt_reward > 0.5 else "FAILURE"

        # Prepare batch
        batch = {k: v.unsqueeze(0) for k, v in sample.items() if isinstance(v, torch.Tensor)}
        batch = preprocessor(batch)

        # Predict
        with torch.no_grad():
            pred_reward = classifier.predict_reward(batch, threshold=0.5)
            print(f"Predicted reward: {pred_reward.item():.4f}")
            pred_label = "SUCCESS" if pred_reward.item() > 0.5 else "FAILURE"

        # Compare
        is_correct = (gt_label == pred_label)
        correct += int(is_correct)
        total += 1

        # Confusion matrix
        if gt_label == "SUCCESS" and pred_label == "SUCCESS":
            results["TP"] += 1
        elif gt_label == "FAILURE" and pred_label == "FAILURE":
            results["TN"] += 1
        elif gt_label == "FAILURE" and pred_label == "SUCCESS":
            results["FP"] += 1
        elif gt_label == "SUCCESS" and pred_label == "FAILURE":
            results["FN"] += 1

        # Print some examples
        if idx < 10 or not is_correct:
            status = "✓" if is_correct else "✗"
            print(f"  Frame {idx}: GT={gt_label}, Pred={pred_label} {status}")

    # === Results ===
    accuracy = 100 * correct / total
    print(f"\n=== Results ===")
    print(f"Accuracy: {accuracy:.1f}% ({correct}/{total})")
    print(f"\nConfusion Matrix:")
    print(f"  True Positive (SUCCESS→SUCCESS):  {results['TP']}")
    print(f"  True Negative (FAILURE→FAILURE):  {results['TN']}")
    print(f"  False Positive (FAILURE→SUCCESS): {results['FP']}")
    print(f"  False Negative (SUCCESS→FAILURE): {results['FN']}")

    if results['TP'] + results['FP'] > 0:
        precision = results['TP'] / (results['TP'] + results['FP'])
        print(f"\nPrecision: {precision:.1%}")
    if results['TP'] + results['FN'] > 0:
        recall = results['TP'] / (results['TP'] + results['FN'])
        print(f"Recall: {recall:.1%}")


if __name__ == "__main__":
    main()
