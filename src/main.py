import os
import json
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib.pyplot as plt

from src.models.CNN import CNN1D
from src.models.RNN import RNNModel
from src.models.Transformer import TransformerModel
from src.models.Danq import DanqModel
from src.dataset import DNADataset, variable_length_collate  # FIX: import collate fn
from src.train import train_model, train_rnn

from src.configs.cnn_config import cnn_config
from src.configs.rnn_config import rnn_config
from src.configs.transformer_config import transformer_config
from src.configs.danq_config import danq_config


# SEQ_LEN is no longer used for padding — sequences are returned at their
# natural length and padded per-batch by variable_length_collate.
# Kept here only for reference / any downstream use.
SEQ_LEN = {
    "original":   85,
    "longerbp":   120,
    "bottleneck": 85,
    "multiclass": 85,
    "HumanvsNeanderthal": 85,
    "DenisovanvsNeanderthal": 85
}



def plot_training_history(history, save_dir, model_name, dataset_type):
    os.makedirs(f"{save_dir}/graphs", exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"],   "r-", label="Val Loss",   linewidth=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name.upper()} — {dataset_type} — Loss", fontweight="bold")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc", linewidth=2)
    ax2.plot(epochs, history["val_acc"],   "r-", label="Val Acc",   linewidth=2)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"{model_name.upper()} — {dataset_type} — Accuracy", fontweight="bold")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"{save_dir}/graphs/{model_name}_{dataset_type}_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {plot_path}")


def plot_confusion_matrix(model, loader, device, num_classes, save_dir, model_name, dataset_type):
    
    CLASS_NAMES = {
        "original":              ["Human", "Denisovan"],
        "longerbp":              ["Human", "Denisovan"],
        "bottleneck":            ["Human", "Denisovan"],
        "multiclass":            ["Human", "Denisovan", "Neanderthal"],
        "HumanvsNeanderthal":    ["Human", "Neanderthal"],
        "DenisovanvsNeanderthal":["Denisovan", "Neanderthal"],
    }
    labels_names = CLASS_NAMES.get(dataset_type, [str(i) for i in range(num_classes)])

    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = torch.argmax(model(x), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Build matrix: cm[true][pred]
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        cm[true][pred] += 1

    # Row-normalise to get per-class recall percentages
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(6 + num_classes, 5 + num_classes))
    im = ax.imshow(cm_pct, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="Row-normalised %")

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(labels_names, fontsize=12)
    ax.set_yticklabels(labels_names, fontsize=12)
    ax.set_xlabel("Predicted", fontsize=13, fontweight="bold")
    ax.set_ylabel("True",      fontsize=13, fontweight="bold")
    ax.set_title(f"{model_name.upper()} — {dataset_type} — Confusion Matrix (val set)",
                 fontsize=13, fontweight="bold", pad=14)

    # Annotate each cell with count and percentage
    thresh = 50  # switch text colour at 50% so it stays readable
    for i in range(num_classes):
        for j in range(num_classes):
            color = "white" if cm_pct[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=11, color=color)

    plt.tight_layout()
    os.makedirs(f"{save_dir}/graphs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{save_dir}/graphs/{model_name}_{dataset_type}_confusion_{timestamp}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved → {path}")

    # Also print to terminal so you can read it without opening the file
    print(f"\n  Confusion matrix (counts) — rows=true, cols=predicted:")
    header = "         " + "  ".join(f"{n:>12}" for n in labels_names)
    print(header)
    for i, row_name in enumerate(labels_names):
        row_str = "  ".join(f"{cm[i,j]:>12}" for j in range(num_classes))
        print(f"  {row_name:>8}  {row_str}")
    print()


def plot_auroc(model, loader, device, num_classes, save_dir, model_name, dataset_type):
    from sklearn.metrics import roc_curve, auc
    from itertools import cycle

    CLASS_NAMES = {
        "original":               ["Human", "Archaic"],
        "longerbp":               ["Human", "Archaic"],
        "bottleneck":             ["Human", "Archaic"],
        "multiclass":             ["Human", "Denisovan", "Neanderthal"],
        "HumanvsNeanderthal":     ["Human", "Neanderthal"],
        "DenisovanvsNeanderthal": ["Denisovan", "Neanderthal"],
    }
    labels_names = CLASS_NAMES.get(dataset_type, [str(i) for i in range(num_classes)])

    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            probs = torch.softmax(model(x), dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_probs  = np.array(all_probs)   # shape: (N, num_classes)
    all_labels = np.array(all_labels)  # shape: (N,)

    plt.figure(figsize=(7, 5))
    colors = cycle(["#2196F3", "#FF9800", "#4CAF50"])

    if num_classes == 2:
        # Binary: one curve using P(class 1)
        fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
        roc_auc     = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2,
                 label=f"{labels_names[1]} vs {labels_names[0]}  (AUC = {roc_auc:.3f})")
    else:
        # Multiclass: one-vs-rest curve per class
        from sklearn.preprocessing import label_binarize
        all_labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))
        for i, (name, color) in enumerate(zip(labels_names, colors)):
            fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
            roc_auc     = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, linewidth=2,
                     label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.500)")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"{model_name.upper()} — {dataset_type} — ROC Curve (val set)",
              fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(f"{save_dir}/graphs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{save_dir}/graphs/{model_name}_{dataset_type}_auroc_{timestamp}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  AUROC plot saved → {path}")


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        type=str, required=True)
    parser.add_argument("--train_path",   type=str, required=True)
    parser.add_argument("--val_path",     type=str, required=True)
    parser.add_argument("--num_classes",  type=int, required=True)
    parser.add_argument("--dataset_type", type=str, required=True)
    parser.add_argument("--epochs",       type=int, default=20)
    args = parser.parse_args()

    if args.dataset_type not in SEQ_LEN:
        raise ValueError(f"Unknown dataset_type '{args.dataset_type}'. Choose from: {list(SEQ_LEN)}")

   

    # ── model
    model_map = {
        "cnn":         (CNN1D,             cnn_config),
        "rnn":         (RNNModel,          rnn_config),
        "transformer": (TransformerModel,  transformer_config),
        "danq":        (DanqModel,         danq_config),
    }

    if args.model not in model_map:
        raise ValueError(f"Unknown model '{args.model}'. Choose from: {list(model_map)}")
    
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   

    ModelClass, config = model_map[args.model]
    model = ModelClass(config, args.num_classes).to(device)

    
    batch_size = config.get("batch_size", {}).get(args.dataset_type, 64)  # default batch size if not specified per dataset
    print(f"Device: {device} | Batch: {batch_size}")
    print("Sequences returned at natural length — padded per-batch by variable_length_collate")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model} | Params: {n_params:,}")

    # ── data
    # FIX: DNADataset no longer takes L or padding_strategy — sequences are
    # returned at their original length. variable_length_collate pads each
    # batch to the longest sequence in that batch, making padding length
    # random and uncorrelated with class.
    train_dataset = DNADataset(args.train_path, train=True)
    val_dataset   = DNADataset(args.val_path,   train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=variable_length_collate,  # FIX: dynamic per-batch padding
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=variable_length_collate,  # FIX: same collate for val
    )
    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    # ── train
    save_dir  = f"results/{args.model}/{args.dataset_type}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{args.model}_{args.dataset_type}_best_model.pt"

    if args.model == "rnn":
        history = train_rnn(
            model, train_loader, val_loader, device,
            num_epochs=args.epochs,
            save_path=save_path,
            train_labels=train_dataset.labels,
            lr=config.get("lr", {}).get(args.dataset_type, 1e-3),
        )
   
    else:
        history = train_model(
            model, train_loader, val_loader, device,
            num_epochs=args.epochs,
            save_path=save_path,
            train_labels=train_dataset.labels,
            lr=config.get("lr", {}).get(args.dataset_type, 1e-3),
        )

    # ── plot training curves
    plot_training_history(history, save_dir, args.model, args.dataset_type)

    # ── confusion matrix on val set using the best checkpoint
    # Load best weights (saved during training whenever val_acc improved)
    # so the matrix reflects the best model, not the last epoch.
    model.load_state_dict(torch.load(save_path, map_location=device))
    plot_confusion_matrix(
        model, val_loader, device,
        num_classes=args.num_classes,
        save_dir=save_dir,
        model_name=args.model,
        dataset_type=args.dataset_type,
    )

    plot_auroc(
        model, val_loader, device,
        num_classes=args.num_classes,
        save_dir=save_dir,
        model_name=args.model,
        dataset_type=args.dataset_type,
    )

    # ── save run metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_data = {
        "model":           args.model,
        "dataset":         args.dataset_type,
        "model_config":    config,
        "training_config": {"batch_size": batch_size, "epochs": args.epochs},
        "history":         history,
    }
    out_file = f"{save_dir}/{args.model}_{args.dataset_type}_run_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(run_data, f, indent=4)
    print(f"  Run saved → {out_file}")


if __name__ == "__main__":
    main()