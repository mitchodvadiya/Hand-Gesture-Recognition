import os
import csv
import json
import time
import gc
import argparse

import numpy as np
import torch

from config import Config
from dataset import get_dataloaders
from models import (
    BaseInceptionCNN,
    EnhancedCNN,
    BaseInceptionCNNLSTM,
    EnhancedHGRModel,
)
from postprocess import HGRPostProcessor
from train import FocalLoss, train_one_epoch
from metrics import (
    compute_classification_metrics,
    save_metrics_to_csv,
    save_confusion_matrix_to_csv,
)


EXPERIMENTS = [
    {
        "name": "cnn_base",
        "title": "CNN baseline",
        "loss": "crossentropy",
        "augment": False,
    },
    {
        "name": "cnn_enhanced",
        "title": "Enhanced CNN",
        "loss": "focal",
        "augment": True,
    },
    {
        "name": "cnn_lstm_base",
        "title": "CNN-LSTM baseline",
        "loss": "crossentropy",
        "augment": False,
    },
    {
        "name": "cnn_lstm_enhanced",
        "title": "Enhanced CNN-LSTM",
        "loss": "focal",
        "augment": True,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and compare HGR models"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=Config.NUM_EPOCHS,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=Config.BATCH_SIZE,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=Config.LEARNING_RATE,
    )

    parser.add_argument(
        "--max_users",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--smoke_test",
        action="store_true",
    )

    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names to run (e.g. 'cnn_lstm_enhanced' or 'all')",
    )

    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip training if a valid checkpoint already exists for this model",
    )

    return parser.parse_args()


def build_model(model_name):
    if model_name == "cnn_base":
        return BaseInceptionCNN(
            num_classes=Config.NUM_CLASSES
        )

    if model_name == "cnn_enhanced":
        return EnhancedCNN(
            num_classes=Config.NUM_CLASSES
        )

    if model_name == "cnn_lstm_base":
        return BaseInceptionCNNLSTM(
            num_classes=Config.NUM_CLASSES,
            hidden_dim=Config.LSTM_HIDDEN_DIM,
        )

    if model_name == "cnn_lstm_enhanced":
        return EnhancedHGRModel(
            num_classes=Config.NUM_CLASSES,
            hidden_dim=Config.LSTM_HIDDEN_DIM,
        )

    raise ValueError(f"Unknown model: {model_name}")


def evaluate_model(model, loader, device, postprocessor):
    """Compute sequence metrics for raw and filtered predictions."""

    model.eval()

    y_true = []
    y_raw = []
    y_post = []

    total_inference_seconds = 0.0

    with torch.inference_mode():
        for seqs, targets, lengths, true_gestures, _ in loader:
            seqs = seqs.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            start = time.perf_counter()
            logits = model(seqs)

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            total_inference_seconds += time.perf_counter() - start

            predictions = torch.argmax(logits, dim=-1)

            for i in range(seqs.size(0)):
                valid_length = int(lengths[i].item())

                indices = (
                    predictions[i, :valid_length]
                    .detach()
                    .cpu()
                    .tolist()
                )

                predicted_names = [
                    Config.IDX_TO_CLASS[int(index)]
                    for index in indices
                ]

                actual_gesture = true_gestures[i]

                y_true.append(actual_gesture)

                y_raw.append(
                    postprocessor.process(
                        predicted_names,
                        apply_filter=False,
                    )
                )

                y_post.append(
                    postprocessor.process(
                        predicted_names,
                        apply_filter=True,
                    )
                )

    raw_metrics = compute_classification_metrics(
        y_true,
        y_raw,
    )

    post_metrics = compute_classification_metrics(
        y_true,
        y_post,
    )

    count = max(1, len(y_true))

    latency_ms = (
        total_inference_seconds / count
    ) * 1000.0

    return raw_metrics, post_metrics, latency_ms


def get_class_weights(train_loader, device):
    """Calculate class weights from training labels only."""

    targets = []

    for sample in train_loader.dataset.samples:
        targets.extend(
            sample["targets"].tolist()
        )

    if not targets:
        raise ValueError("The training dataset contains no targets.")

    targets = torch.tensor(
        targets,
        dtype=torch.long,
    )

    counts = torch.bincount(
        targets,
        minlength=Config.NUM_CLASSES,
    ).float()

    if torch.any(counts == 0):
        raise ValueError(
            "At least one class is missing from the training dataset. "
            f"Class counts: {counts.tolist()}"
        )

    weights = counts.sum() / (
        Config.NUM_CLASSES * counts
    )

    weights = weights / weights.mean()

    return weights.to(device)


def train_experiment(
    experiment,
    args,
    epochs,
    max_users,
    device,
):
    name = experiment["name"]
    loss_type = experiment["loss"]
    augment = experiment["augment"]

    print("\n" + "=" * 70)
    print(f"Training: {experiment['title']}")
    print(f"Loss: {loss_type}")
    print(f"Augmentation: {augment}")
    print("=" * 70)

    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size,
        max_users=max_users,
        use_cache=True,
        augment=augment,
    )

    class_weights = get_class_weights(
        train_loader,
        device,
    )

    model = build_model(name).to(device)

    if loss_type == "focal":
        criterion = FocalLoss(
            gamma=Config.FOCAL_GAMMA,
            alpha=class_weights,
            label_smoothing=Config.LABEL_SMOOTHING,
        )
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=1e-5,
    )

    postprocessor = HGRPostProcessor(
        mode=Config.POSTPROCESS_MODE
    )

    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    checkpoint_path = os.path.join(
        Config.CHECKPOINT_DIR,
        f"best_model_{name}_{loss_type}.pth",
    )

    checkpoint_candidates = [
        checkpoint_path,
        os.path.join(Config.CHECKPOINT_DIR, f"best_model_{name}.pth"),
    ]
    if name == "cnn_lstm_base":
        checkpoint_candidates.append(
            os.path.join(Config.CHECKPOINT_DIR, "best_model_base_crossentropy.pth")
        )

    existing_checkpoint = None
    for cand in checkpoint_candidates:
        if os.path.isfile(cand):
            existing_checkpoint = cand
            break

    if args.skip_existing and existing_checkpoint:
        print(f"\n[FAST-TRACK] Found existing checkpoint for {name}: {existing_checkpoint}")
        print("Evaluating existing checkpoint directly without re-training...")
        ckpt = torch.load(existing_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        raw_metrics, post_metrics, latency_ms = evaluate_model(
            model,
            val_loader,
            device,
            postprocessor,
        )
        del model
        gc.collect()
        return {
            "name": name,
            "title": experiment["title"],
            "loss_type": loss_type,
            "augmentation": augment,
            "checkpoint": existing_checkpoint,
            "checkpoint_epoch": ckpt.get("epoch", 1),
            "training_seconds": ckpt.get("training_seconds", 0.0),
            "latency_ms_per_sequence": round(latency_ms, 4),
            "raw_metrics": raw_metrics,
            "post_metrics": post_metrics,
        }

    best_macro_f1 = -1.0
    patience_counter = 0

    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        raw_metrics, post_metrics, latency_ms = evaluate_model(
            model,
            val_loader,
            device,
            postprocessor,
        )

        scheduler.step()

        current_f1 = post_metrics["macro_f1"]

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_acc:.2f}% | "
            f"Raw accuracy: {raw_metrics['accuracy']:.2f}% | "
            f"Post accuracy: {post_metrics['accuracy']:.2f}% | "
            f"Post macro F1: {current_f1:.2f}% | "
            f"Latency: {latency_ms:.2f} ms/sequence | "
            f"Time: {time.perf_counter() - epoch_start:.1f}s"
        )

        # Select the checkpoint using validation macro F1.
        if current_f1 > best_macro_f1:
            best_macro_f1 = current_f1
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_type": name,
                    "loss_type": loss_type,
                    "max_users": max_users,
                    "selection_metric": "postprocessed_macro_f1",
                    "val_macro_f1": current_f1,
                    "val_post_acc": post_metrics["accuracy"],
                    "val_metrics": post_metrics,
                    "raw_metrics": raw_metrics,
                    "batch_size": args.batch_size,
                    "learning_rate": args.lr,
                    "epochs_requested": epochs,
                    "random_seed": getattr(
                        Config,
                        "RANDOM_SEED",
                        9,
                    ),
                    "postprocess_mode": Config.POSTPROCESS_MODE,
                },
                checkpoint_path,
            )

            print(
                f"Saved best checkpoint: {checkpoint_path}"
            )

        else:
            patience_counter += 1

            if (
                patience_counter >= args.patience
                and not args.smoke_test
            ):
                print("Early stopping.")
                break

    training_seconds = time.perf_counter() - start_time

    # IMPORTANT: reload the exact checkpoint selected above.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Reevaluate the checkpoint on the same validation loader.
    raw_metrics, post_metrics, latency_ms = evaluate_model(
        model,
        val_loader,
        device,
        postprocessor,
    )

    del model
    del optimizer
    del scheduler
    gc.collect()

    return {
        "name": name,
        "title": experiment["title"],
        "loss_type": loss_type,
        "augmentation": augment,
        "checkpoint": checkpoint_path,
        "checkpoint_epoch": checkpoint["epoch"],
        "training_seconds": round(training_seconds, 2),
        "latency_ms_per_sequence": round(latency_ms, 4),
        "raw_metrics": raw_metrics,
        "post_metrics": post_metrics,
    }


def main():
    args = parse_args()

    device = (
        torch.device(args.device)
        if args.device
        else Config.DEVICE
    )

    epochs = 2 if args.smoke_test else args.epochs

    max_users = (
        args.max_users
        if args.max_users is not None
        else (5 if args.smoke_test else None)
    )

    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max users: {max_users}")
    print(f"Validation selection metric: post-processed macro F1")

    if args.models:
        requested = [m.strip().lower() for m in args.models.split(",")]
        experiments_to_run = [e for e in EXPERIMENTS if e["name"].lower() in requested]
        if not experiments_to_run:
            raise ValueError(f"No matching experiments found for --models '{args.models}'. Available: {[e['name'] for e in EXPERIMENTS]}")
    else:
        experiments_to_run = EXPERIMENTS

    results = []

    for experiment in experiments_to_run:
        result = train_experiment(
            experiment=experiment,
            args=args,
            epochs=epochs,
            max_users=max_users,
            device=device,
        )

        results.append(result)

        name = result["name"]
        raw_metrics = result["raw_metrics"]
        post_metrics = result["post_metrics"]

        # Save separate raw and post-processed results.
        save_metrics_to_csv(
            raw_metrics,
            os.path.join(
                Config.RESULTS_DIR,
                f"{name}_raw_metrics.csv",
            ),
        )

        save_metrics_to_csv(
            post_metrics,
            os.path.join(
                Config.RESULTS_DIR,
                f"{name}_metrics.csv",
            ),
        )

        save_confusion_matrix_to_csv(
            raw_metrics["confusion_matrix"],
            raw_metrics["class_names"],
            os.path.join(
                Config.RESULTS_DIR,
                f"{name}_raw_confusion_matrix.csv",
            ),
        )

        save_confusion_matrix_to_csv(
            post_metrics["confusion_matrix"],
            post_metrics["class_names"],
            os.path.join(
                Config.RESULTS_DIR,
                f"{name}_confusion_matrix.csv",
            ),
        )

        # Save an individual JSON record with checkpoint provenance.
        record = {
            "model": name,
            "checkpoint": result["checkpoint"],
            "checkpoint_epoch": result["checkpoint_epoch"],
            "loss_type": result["loss_type"],
            "augmentation": result["augmentation"],
            "max_users": max_users,
            "raw_metrics": raw_metrics,
            "postprocessed_metrics": post_metrics,
            "training_seconds": result["training_seconds"],
            "inference_latency_ms_per_sequence": (
                result["latency_ms_per_sequence"]
            ),
        }

        with open(
            os.path.join(
                Config.RESULTS_DIR,
                f"{name}_evaluation.json",
            ),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(record, file, indent=2, default=str)

        # Incrementally save completed experiments.
        summary_csv = os.path.join(
            Config.RESULTS_DIR,
            "model_comparison_results.csv",
        )

        with open(
            summary_csv,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

            writer.writerow(
                [
                    "Model",
                    "Training enhancement",
                    "Checkpoint",
                    "Checkpoint epoch",
                    "Raw accuracy (%)",
                    "Post-processed accuracy (%)",
                    "Raw macro F1 (%)",
                    "Post-processed macro F1 (%)",
                    "Training duration (s)",
                    "Inference latency (ms/sequence)",
                    "Samples",
                ]
            )

            for item in results:
                writer.writerow(
                    [
                        item["name"],
                        (
                            "Augmentation + focal loss"
                            if item["augmentation"]
                            else "None"
                        ),
                        item["checkpoint"],
                        item["checkpoint_epoch"],
                        item["raw_metrics"]["accuracy"],
                        item["post_metrics"]["accuracy"],
                        item["raw_metrics"]["macro_f1"],
                        item["post_metrics"]["macro_f1"],
                        item["training_seconds"],
                        item["latency_ms_per_sequence"],
                        item["post_metrics"]["total_samples"],
                    ]
                )

        print(f"Saved results for {name}.")

    summary_json = os.path.join(
        Config.RESULTS_DIR,
        "model_comparison_results.json",
    )

    with open(
        summary_json,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "device": str(device),
                "epochs_requested": epochs,
                "batch_size": args.batch_size,
                "max_users": max_users,
                "selection_metric": "postprocessed_macro_f1",
                "postprocess_mode": Config.POSTPROCESS_MODE,
                "experiments": results,
            },
            file,
            indent=2,
            default=str,
        )

    print("\n" + "=" * 100)
    print("FINAL MODEL COMPARISON")
    print("=" * 100)

    for item in results:
        print(
            f"{item['name']:<20} "
            f"Raw Acc: {item['raw_metrics']['accuracy']:>6.2f}% | "
            f"Post Acc: {item['post_metrics']['accuracy']:>6.2f}% | "
            f"Raw F1: {item['raw_metrics']['macro_f1']:>6.2f}% | "
            f"Post F1: {item['post_metrics']['macro_f1']:>6.2f}%"
        )

    print(f"\nCSV:  {summary_csv}")
    print(f"JSON: {summary_json}")


if __name__ == "__main__":
    main()