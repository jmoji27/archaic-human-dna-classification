from logging import config
import os
import json
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    matthews_corrcoef,
)
from sklearn.preprocessing import label_binarize
from itertools import cycle

from src.models.CNN import CNN1D 
from src.models.CNNwithoutPooling import CNN1D_no_pooling 
from src.models.RNN import RNNModel
from src.models.Transformer import TransformerModel
from src.models.Danq import DanqModel

# Datasets and Collate Functions
from src.dataset import DNADataset, variable_length_collate
from src.danq_dataset import DanqDataset, danq_collate


from src.configs.cnn_config import cnn_config
from src.configs.rnn_config import rnn_config
from src.configs.transformer_config import transformer_config
from src.configs.danq_config import danq_config

SEQ_LEN = {
    "original":   85,
    "longerbp":   120,
    "bottleneck": 85,
    "multiclass": 85,
    "HumanvsNeanderthal": 85,
    "DenisovanvsNeanderthal": 85
}


CLASS_NAMES = {
    "original": ["Human", "Denisovan"],
    "longerbp": ["Human", "Denisovan"],
    "bottleneck": ["Human", "Denisovan"],
    "multiclass": ["Human", "Denisovan", "Neanderthal"],
    "HumanvsNeanderthal": ["Human", "Neanderthal"],
    "DenisovanvsNeanderthal": ["Denisovan", "Neanderthal"],
}


def plot_confusion_matrix(all_preds, all_labels, num_classes, dataset_type, save_dir, model_name):
    labels_names = CLASS_NAMES.get(dataset_type, [str(i) for i in range(num_classes)])

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        cm[true][pred] += 1

    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(6 + num_classes, 5 + num_classes))
    im = ax.imshow(cm_pct, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="Row-normalised %")

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(labels_names, fontsize=12)
    ax.set_yticklabels(labels_names, fontsize=12)
    ax.set_xlabel("Predicted", fontsize=13, fontweight="bold")
    ax.set_ylabel("True", fontsize=13, fontweight="bold")
    ax.set_title(f"{model_name.upper()} — {dataset_type} — Confusion Matrix (test set)",
                 fontsize=13, fontweight="bold", pad=14)

    thresh = 50
    for i in range(num_classes):
        for j in range(num_classes):
            color = "white" if cm_pct[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=11, color=color)

    plt.tight_layout()
    os.makedirs(f"{save_dir}/graphs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{save_dir}/graphs/{model_name}_{dataset_type}_confusion_test_{timestamp}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved → {path}")

    print(f"\n  Confusion matrix (counts) — rows=true, cols=predicted:")
    header = "         " + "  ".join(f"{n:>12}" for n in labels_names)
    print(header)
    for i, row_name in enumerate(labels_names):
        row_str = "  ".join(f"{cm[i, j]:>12}" for j in range(num_classes))
        print(f"  {row_name:>8}  {row_str}")
    print()


def plot_auroc(all_probs, all_labels, num_classes, dataset_type, save_dir, model_name):
    labels_names = CLASS_NAMES.get(dataset_type, [str(i) for i in range(num_classes)])

    plt.figure(figsize=(7, 5))
    colors = cycle(["#2196F3", "#FF9800", "#4CAF50"])

    if num_classes == 2:
        fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2,
                 label=f"{labels_names[1]} vs {labels_names[0]}  (AUC = {roc_auc:.3f})")
    else:
        all_labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))
        for i, (name, color) in enumerate(zip(labels_names, colors)):
            fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, linewidth=2,
                     label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.500)")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"{model_name.upper()} — {dataset_type} — ROC Curve (test set)",
              fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(f"{save_dir}/graphs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{save_dir}/graphs/{model_name}_{dataset_type}_auroc_test_{timestamp}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  AUROC plot saved → {path}")


def plot_precision_recall(all_probs, all_labels, num_classes, dataset_type, save_dir, model_name):
    labels_names = CLASS_NAMES.get(dataset_type, [str(i) for i in range(num_classes)])

    plt.figure(figsize=(7, 5))
    colors = cycle(["#2196F3", "#FF9800", "#4CAF50"])

    if num_classes == 2:
        precision, recall, _ = precision_recall_curve(all_labels, all_probs[:, 1])
        ap = average_precision_score(all_labels, all_probs[:, 1])
        plt.plot(recall, precision, linewidth=2,
                 label=f"{labels_names[1]} (AP = {ap:.3f})")
    else:
        all_labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))
        for i, (name, color) in enumerate(zip(labels_names, colors)):
            precision, recall, _ = precision_recall_curve(all_labels_bin[:, i], all_probs[:, i])
            ap = average_precision_score(all_labels_bin[:, i], all_probs[:, i])
            plt.plot(recall, precision, color=color, linewidth=2,
                     label=f"{name} (AP = {ap:.3f})")

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(f"{model_name.upper()} — {dataset_type} — Precision-Recall Curve (test set)",
              fontsize=13, fontweight="bold")
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(f"{save_dir}/graphs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{save_dir}/graphs/{model_name}_{dataset_type}_pr_curve_test_{timestamp}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Precision-Recall plot saved → {path}")


def save_to_master_results(all_labels, all_preds, labels_names, accuracy, mcc, model_name, dataset_type):
    report_dict = classification_report(
        all_labels, all_preds, target_names=labels_names, digits=4, output_dict=True
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []

    for class_name in labels_names:
        rows.append({
            "timestamp": timestamp,
            "model": model_name,
            "dataset": dataset_type,
            "class": class_name,
            "precision": round(report_dict[class_name]["precision"], 4),
            "recall": round(report_dict[class_name]["recall"], 4),
            "f1": round(report_dict[class_name]["f1-score"], 4),
            "support": int(report_dict[class_name]["support"]),
            "accuracy": round(accuracy, 4),
            "mcc": round(mcc, 4),
            "macro_f1": round(report_dict["macro avg"]["f1-score"], 4),
            "weighted_f1": round(report_dict["weighted avg"]["f1-score"], 4),
        })

    master_path = "results/master_results.csv"
    os.makedirs("results", exist_ok=True)
    df_new = pd.DataFrame(rows)

    if os.path.exists(master_path):
        df_existing = pd.read_csv(master_path)

        already_exists = (
                (df_existing["model"] == model_name) &
                (df_existing["dataset"] == dataset_type)
        ).any()

        if already_exists:
            print(
                f"  WARNING: results for {model_name}/{dataset_type} already exist in master_results.csv — skipping save to avoid duplicates.")
            return

        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(master_path, index=False)
    print(f"  Master results updated → {master_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--dataset_type", type=str, required=True)
    parser.add_argument("--model_path",   type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_map = {
        "cnn":            (CNN1D,            cnn_config),
        "cnn_no_pooling": (CNN1D_no_pooling, cnn_config),
        "rnn":            (RNNModel,         rnn_config),
        "transformer":    (TransformerModel, transformer_config),
        "danq":           (DanqModel,        danq_config),
    }

    if args.model not in model_map:
        raise ValueError(f"Unknown model '{args.model}'. Choose from: {list(model_map)}")

    ModelClass, config = model_map[args.model]

    if args.model == "cnn_no_pooling":
        seq_l = SEQ_LEN.get(args.dataset_type, 85)
        model = ModelClass(config, args.num_classes, seq_length=seq_l).to(device)
    else:
        model = ModelClass(config, args.num_classes).to(device)

    # ── load weights ──────────────────────────────────────────────────
    model.load_state_dict(
        torch.load(args.model_path, map_location=device, weights_only=True),
        strict=True   # was strict=False — now fails loudly on any mismatch
    )
    model.eval()
    print(f"Loaded weights from {args.model_path}")

    # ── Dataset Mapping & Collate Function setup
    batch_size = config.get("batch_size", {}).get(args.dataset_type, 64)

    if args.model == "danq":
        test_dataset = DanqDataset(args.test_path, train=False)
        collate_fn = danq_collate
    else:
        test_dataset = DNADataset(args.test_path, train=False)
        collate_fn = variable_length_collate

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )
    print(f"Test samples: {len(test_dataset):,}")

    # ── Inference Pipeline Execution
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y    = x.to(device), y.to(device)
            logits  = model(x)
            probs   = torch.softmax(logits, dim=1)
            preds   = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # ── Metrics Calculation
    labels_names = CLASS_NAMES.get(args.dataset_type, [str(i) for i in range(args.num_classes)])
    accuracy = (all_preds == all_labels).mean()
    mcc = matthews_corrcoef(all_labels, all_preds)

    print(f"\n{'=' * 60}")
    print(f"  Test Accuracy : {accuracy:.4f}")
    print(f"  MCC           : {mcc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=labels_names, digits=4))
    print(f"{'=' * 60}\n")

    # ── Evaluation Plots and Saving Outputs
    save_dir = f"results/{args.model}/{args.dataset_type}/test"
    os.makedirs(save_dir, exist_ok=True)

    plot_confusion_matrix(all_preds, all_labels, args.num_classes, args.dataset_type, save_dir, args.model)
    plot_auroc(all_probs, all_labels, args.num_classes, args.dataset_type, save_dir, args.model)
    plot_precision_recall(all_probs, all_labels, args.num_classes, args.dataset_type, save_dir, args.model)
    save_to_master_results(all_labels, all_preds, labels_names, accuracy, mcc, args.model, args.dataset_type)

    # Export complete run data object
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "model": args.model,
        "dataset": args.dataset_type,
        "model_path": args.model_path,
        "test_samples": len(test_dataset),
        "accuracy": float(accuracy),
        "mcc": float(mcc),
        "classification_report": classification_report(
            all_labels, all_preds, target_names=labels_names, digits=4, output_dict=True
        ),
    }
    out_file = f"{save_dir}/{args.model}_{args.dataset_type}_test_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"  Results saved → {out_file}")


if __name__ == "__main__":
    main()