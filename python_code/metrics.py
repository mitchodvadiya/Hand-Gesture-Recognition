import numpy as np
from config import Config

def compute_classification_metrics(y_true: list, y_pred: list, class_names: list = Config.ALL_CLASSES) -> dict:
    """
    Computes comprehensive multi-class classification metrics:
    - Overall Accuracy
    - Macro Precision, Macro Recall, Macro F1-score
    - Per-class Precision, Recall, F1, and Support
    - Confusion Matrix (K x K)
    """
    K = len(class_names)
    cls_to_idx = {name: i for i, name in enumerate(class_names)}
    
    # Build confusion matrix: rows = true, cols = predicted
    cm = np.zeros((K, K), dtype=np.int64)
    for yt, yp in zip(y_true, y_pred):
        r = cls_to_idx.get(yt, -1)
        c = cls_to_idx.get(yp, -1)
        if r >= 0 and c >= 0:
            cm[r, c] += 1
            
    total_samples = len(y_true)
    total_correct = int(np.trace(cm))
    accuracy = 100.0 * total_correct / max(1, total_samples)
    
    per_class = {}
    precisions = []
    recalls = []
    f1s = []
    
    for i, cls in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = int(cm[i, :].sum())
        
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        
        per_class[cls] = {
            "precision": round(prec * 100.0, 2),
            "recall": round(rec * 100.0, 2),
            "f1_score": round(f1 * 100.0, 2),
            "support": support,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn)
        }
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        
    macro_precision = float(np.mean(precisions) * 100.0)
    macro_recall = float(np.mean(recalls) * 100.0)
    macro_f1 = float(np.mean(f1s) * 100.0)

    # Weighted metrics based on class support
    supports = [int(cm[i, :].sum()) for i in range(K)]
    total_support = sum(supports)
    if total_support > 0:
        weighted_precision = float(sum(p * s for p, s in zip(precisions, supports)) / total_support * 100.0)
        weighted_recall = float(sum(r * s for r, s in zip(recalls, supports)) / total_support * 100.0)
        weighted_f1 = float(sum(f * s for f, s in zip(f1s, supports)) / total_support * 100.0)
    else:
        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_f1 = 0.0
    
    return {
        "total_samples": total_samples,
        "total_correct": total_correct,
        "accuracy": round(accuracy, 2),
        "macro_precision": round(macro_precision, 2),
        "macro_recall": round(macro_recall, 2),
        "macro_f1": round(macro_f1, 2),
        "weighted_precision": round(weighted_precision, 2),
        "weighted_recall": round(weighted_recall, 2),
        "weighted_f1": round(weighted_f1, 2),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names)
    }


def format_confusion_matrix(cm_data: list, class_names: list = Config.ALL_CLASSES) -> str:
    """Formats confusion matrix into an aligned ASCII table."""
    cm = np.array(cm_data)
    col_width = max(max(len(c) for c in class_names), 9) + 2
    
    lines = []
    lines.append("Confusion Matrix (Rows: Actual, Cols: Predicted):")
    header = " " * 15 + "".join(f"{c:>{col_width}}" for c in class_names)
    lines.append(header)
    lines.append("-" * len(header))
    
    for i, name in enumerate(class_names):
        row_str = f"{name:<15}" + "".join(f"{cm[i, j]:>{col_width}}" for j in range(len(class_names)))
        lines.append(row_str)
        
    return "\n".join(lines)


def format_metrics_table(metrics: dict) -> str:
    """Formats per-class metrics and macro averages into an aligned ASCII table."""
    lines = []
    lines.append(f"Overall Accuracy:   {metrics['accuracy']:.2f}%")
    lines.append(f"Macro Precision:    {metrics['macro_precision']:.2f}%")
    lines.append(f"Macro Recall:       {metrics['macro_recall']:.2f}%")
    lines.append(f"Macro F1-Score:     {metrics['macro_f1']:.2f}%")
    if "weighted_f1" in metrics:
        lines.append(f"Weighted F1-Score:  {metrics['weighted_f1']:.2f}%")
    lines.append("")
    
    header = f"{'Gesture':<15} | {'Precision (%)':<14} | {'Recall (%)':<12} | {'F1-Score (%)':<13} | {'Support':<8}"
    lines.append(header)
    lines.append("-" * len(header))
    
    for cls, val in metrics["per_class"].items():
        row = f"{cls:<15} | {val['precision']:<14.2f} | {val['recall']:<12.2f} | {val['f1_score']:<13.2f} | {val['support']:<8}"
        lines.append(row)
        
    lines.append("-" * len(header))
    return "\n".join(lines)


def save_metrics_to_csv(metrics: dict, csv_filepath: str):
    """Saves per-class and summary metrics to a CSV file."""
    import csv
    with open(csv_filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Summary rows
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Overall Accuracy (%)", metrics["accuracy"]])
        writer.writerow(["Macro Precision (%)", metrics["macro_precision"]])
        writer.writerow(["Macro Recall (%)", metrics["macro_recall"]])
        writer.writerow(["Macro F1-Score (%)", metrics["macro_f1"]])
        if "weighted_f1" in metrics:
            writer.writerow(["Weighted F1-Score (%)", metrics["weighted_f1"]])
        writer.writerow(["Total Evaluated Samples", metrics["total_samples"]])
        writer.writerow(["Total Correct Predictions", metrics["total_correct"]])
        writer.writerow([])
        
        # Per-class metrics
        writer.writerow(["Gesture", "Precision (%)", "Recall (%)", "F1-Score (%)", "Support", "TP", "FP", "FN"])
        for cls, val in metrics["per_class"].items():
            writer.writerow([
                cls,
                val["precision"],
                val["recall"],
                val["f1_score"],
                val["support"],
                val["true_positives"],
                val["false_positives"],
                val["false_negatives"]
            ])


def save_confusion_matrix_to_csv(cm_data: list, class_names: list, csv_filepath: str):
    """Saves confusion matrix to a CSV file with labeled axes."""
    import csv
    with open(csv_filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Actual \\ Predicted"] + list(class_names))
        for i, name in enumerate(class_names):
            writer.writerow([name] + list(cm_data[i]))

