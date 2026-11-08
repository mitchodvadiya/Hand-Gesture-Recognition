import csv
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

files = {
    "Baseline CNN-LSTM": Path("results/cnn_lstm_base_confusion_matrix.csv"),
    "Proposed Enhanced CNN-LSTM": Path("results/cnn_lstm_enhanced_confusion_matrix.csv"),
}

for model, path in files.items():
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    labels = rows[0][1:]
    matrix = [list(map(int, row[1:])) for row in rows[1:] if row]

    y_true, y_pred = [], []
    for i, row in enumerate(matrix):
        for j, count in enumerate(row):
            y_true.extend([labels[i]] * count)
            y_pred.extend([labels[j]] * count)

    accuracy = accuracy_score(y_true, y_pred) * 100
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    print(f"\n{model}")
    print(f"Samples: {len(y_true)}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Macro precision: {p.mean() * 100:.2f}%")
    print(f"Macro recall: {r.mean() * 100:.2f}%")
    print(f"Macro F1: {f1.mean() * 100:.2f}%")
    print("\nClass       Precision   Recall   F1       FP")
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fp = sum(row[i] for row in matrix) - tp
        print(
            f"{label:10s} {p[i]*100:8.2f}% "
            f"{r[i]*100:8.2f}% {f1[i]*100:7.2f}% {fp:6d}"
        )
