import os
import argparse
import json
import torch
import numpy as np

from torch.utils.data import DataLoader

from config import Config
from dataset import EMGSequenceDataset, sequence_collate_fn
from models import (
    BaseInceptionCNN,
    EnhancedCNN,
    BaseInceptionCNNLSTM,
    EnhancedHGRModel,
)
from postprocess import HGRPostProcessor
from metrics import (
    compute_classification_metrics,
    format_metrics_table,
    save_metrics_to_csv,
    save_confusion_matrix_to_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved HGR model checkpoint"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="val",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=Config.BATCH_SIZE,
    )

    parser.add_argument(
        "--max_users",
        type=int,
        default=None,
        help="Override the checkpoint's max_users setting",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional output JSON path",
    )

    return parser.parse_args()


def build_model(model_type):
    """Create the architecture associated with the checkpoint."""

    model_type = model_type.lower()

    if model_type in ("cnn_base", "base_cnn"):
        return BaseInceptionCNN(num_classes=Config.NUM_CLASSES)

    if model_type in ("cnn_enhanced", "enhanced_cnn"):
        return EnhancedCNN(num_classes=Config.NUM_CLASSES)

    if model_type in ("cnn_lstm_base", "base", "base_lstm"):
        return BaseInceptionCNNLSTM(
            num_classes=Config.NUM_CLASSES,
            hidden_dim=Config.LSTM_HIDDEN_DIM,
        )

    if model_type in ("cnn_lstm_enhanced", "enhanced"):
        return EnhancedHGRModel(
            num_classes=Config.NUM_CLASSES,
            hidden_dim=Config.LSTM_HIDDEN_DIM,
        )

    raise ValueError(
        f"Unknown model_type '{model_type}' in checkpoint."
    )


def safe_json_value(value):
    """Convert common NumPy and PyTorch values into JSON-safe values."""

    if isinstance(value, dict):
        return {
            str(k): safe_json_value(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [safe_json_value(v) for v in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if torch.is_tensor(value):
        return value.detach().cpu().tolist()

    return value


def evaluate_model(model, loader, device, postprocessor):
    """Calculate sequence-level raw and post-processed metrics."""

    model.eval()

    y_true = []
    y_raw = []
    y_post = []

    with torch.inference_mode():
        for seqs, targets, lengths, true_gestures, _ in loader:
            seqs = seqs.to(device)

            logits = model(seqs)
            predictions = torch.argmax(logits, dim=-1)

            batch_size = seqs.size(0)

            for i in range(batch_size):
                valid_length = int(lengths[i].item())

                pred_indices = (
                    predictions[i, :valid_length]
                    .detach()
                    .cpu()
                    .tolist()
                )

                pred_names = [
                    Config.IDX_TO_CLASS[int(index)]
                    for index in pred_indices
                ]

                actual_gesture = true_gestures[i]

                raw_prediction = postprocessor.process(
                    pred_names,
                    apply_filter=False,
                )

                post_prediction = postprocessor.process(
                    pred_names,
                    apply_filter=True,
                )

                y_true.append(actual_gesture)
                y_raw.append(raw_prediction)
                y_post.append(post_prediction)

    raw_metrics = compute_classification_metrics(y_true, y_raw)
    post_metrics = compute_classification_metrics(y_true, y_post)

    return raw_metrics, post_metrics


def main():
    args = parse_args()

    device = Config.DEVICE

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Requested split: {args.split}")

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    model_type = checkpoint.get("model_type")

    if not model_type:
        raise ValueError(
            "Checkpoint does not contain model_type. "
            "Identify its architecture before evaluating it."
        )

    # Use the checkpoint's cohort size by default.
    # Passing --max_users explicitly overrides this setting.
    max_users = (
        args.max_users
        if args.max_users is not None
        else checkpoint.get("max_users")
    )

    print(f"Model type: {model_type}")
    print(f"Evaluation users setting: {max_users}")

    model = build_model(model_type)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    dataset = EMGSequenceDataset(
        data_split=args.split,
        max_users=max_users,
        use_cache=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=sequence_collate_fn,
        num_workers=Config.NUM_WORKERS,
    )

    postprocessor = HGRPostProcessor(
        mode=Config.POSTPROCESS_MODE
    )

    raw_metrics, post_metrics = evaluate_model(
        model,
        loader,
        device,
        postprocessor,
    )

    gain = (
        post_metrics["accuracy"]
        - raw_metrics["accuracy"]
    )

    print("\n" + "=" * 65)
    print("EVALUATION RESULTS")
    print("=" * 65)

    print(f"Samples evaluated: {raw_metrics['total_samples']}")

    print(
        f"Raw accuracy: "
        f"{raw_metrics['accuracy']:.2f}%"
    )

    print(
        f"Post-processed accuracy: "
        f"{post_metrics['accuracy']:.2f}%"
    )

    print(
        f"Post-processing accuracy change: "
        f"{gain:+.2f} percentage points"
    )

    print(
        f"Raw macro F1: "
        f"{raw_metrics['macro_f1']:.2f}%"
    )

    print(
        f"Post-processed macro F1: "
        f"{post_metrics['macro_f1']:.2f}%"
    )

    print("\nPost-processed classification report:")
    print(format_metrics_table(post_metrics))

    results_dir = Config.RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    if args.output_file:
        base_path = args.output_file

        if base_path.lower().endswith(".json"):
            base_path = base_path[:-5]

        output_dir = os.path.dirname(base_path)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    else:
        base_path = os.path.join(
            results_dir,
            f"eval_{args.split}_{model_type}",
        )

    json_path = f"{base_path}.json"
    raw_csv_path = f"{base_path}_raw_metrics.csv"
    post_csv_path = f"{base_path}_metrics.csv"
    raw_cm_path = f"{base_path}_raw_confusion_matrix.csv"
    post_cm_path = f"{base_path}_confusion_matrix.csv"

    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split,
        "model_type": model_type,
        "max_users": max_users,
        "batch_size": args.batch_size,
        "postprocess_mode": Config.POSTPROCESS_MODE,
        "raw_metrics": raw_metrics,
        "postprocessed_metrics": post_metrics,
        "postprocess_gain": round(gain, 4),
    }

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            safe_json_value(summary),
            file,
            indent=4,
        )

    save_metrics_to_csv(raw_metrics, raw_csv_path)
    save_metrics_to_csv(post_metrics, post_csv_path)

    save_confusion_matrix_to_csv(
        raw_metrics["confusion_matrix"],
        raw_metrics["class_names"],
        raw_cm_path,
    )

    save_confusion_matrix_to_csv(
        post_metrics["confusion_matrix"],
        post_metrics["class_names"],
        post_cm_path,
    )

    print("\nSaved evaluation files:")
    print(json_path)
    print(raw_csv_path)
    print(post_csv_path)
    print(raw_cm_path)
    print(post_cm_path)


if __name__ == "__main__":
    main()