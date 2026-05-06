"""
utils.py — LR finder, Optuna hyperparameter search, and class balance checker.

Directory structure assumed:
    data/
    ├── original/
    │   ├── train.csv
    │   ├── val.csv
    │   └── test.csv
    ├── longerbp/
    ├── multiclass/
    ├── bottleneck/
    ├── HumanvsNeanderthal/
    └── DenisovanvsNeanderthal/

Each CSV must have columns: 'sequence', 'label'

CHANGES vs previous version:
  - Optuna objective now uses balanced_accuracy_score instead of raw val_acc
  - LR search range extended from [1e-5, 1e-3] to [1e-5, 1e-2] to include
    the region your LR finder identified as optimal (9.33e-3)
  - Optuna now collects per-class recall and logs it as trial user attributes
    so you can inspect Denisovan recall across trials after the search
  - Added a helper `evaluate_balanced` used both in Optuna and standalone
"""

import os
import math
import torch
import optuna
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, classification_report

# ── local imports ─────────────────────────────────────────────────────────────
from src.dataset import DNADataset, variable_length_collate
from src.models.CNN import CNN1D as CNN
from src.configs.cnn_config import cnn_config
from src.train import train_model

# ── dataset paths ─────────────────────────────────────────────────────────────
DATA_ROOT          = "dataset/multiclass/original"
DATA_ROOT_FOR_BINARY = "dataset/binary"

DATASET_NAMES = [
    "original",
    "longerbp",
    "bottleneck",
    "HumanvsNeanderthal",
    "DenisovanvsNeanderthal",
    "multiclass",
]

NUM_CLASSES = {
    "original":               2,
    "longerbp":               2,
    "bottleneck":             2,
    "HumanvsNeanderthal":     2,
    "DenisovanvsNeanderthal": 2,
    "multiclass":             3,
}

# Class index → human-readable name, used for per-class recall logging
CLASS_NAMES = {
    "original":               {0: "Human", 1: "Archaic"},
    "longerbp":               {0: "Human", 1: "Archaic"},
    "bottleneck":             {0: "Human", 1: "Archaic"},
    "HumanvsNeanderthal":     {0: "Human", 1: "Neanderthal"},
    "DenisovanvsNeanderthal": {0: "Denisovan", 1: "Neanderthal"},
    "multiclass":             {0: "Human", 1: "Denisovan", 2: "Neanderthal"},
}


def get_csv_paths(dataset_name: str) -> dict[str, str]:
    if dataset_name == "multiclass":
        base = "dataset/multiclass/original"
    else:
        base = os.path.join("dataset/binary", dataset_name)
    return {
        "train": os.path.join(base, "train.csv"),
        "val":   os.path.join(base, "val.csv"),
        "test":  os.path.join(base, "test.csv"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — full evaluation pass returning balanced acc + per-class recall
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_balanced(model, loader, device, dataset_name: str) -> dict:
    """
    Run a full inference pass over `loader` and return:
      - balanced_accuracy  : mean of per-class recall (what Optuna maximises)
      - per_class_recall   : dict {class_name: recall_float}
      - all_preds / all_labels : raw arrays for any further analysis

    Why balanced accuracy?
    ----------------------
    balanced_accuracy = mean(recall_per_class)

    On a 2-class problem with 60% Neanderthal recall and 94% Denisovan recall
    the raw accuracy might be ~80% and look fine, but balanced accuracy is
    (0.60 + 0.94) / 2 = 0.77 — it penalises the model for ignoring one class.
    Optuna maximising this forces it to find architectures that do well on
    BOTH classes, not just the easy one.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y  = x.to(device), y.to(device)
            logits = model(x)
            preds  = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    # Per-class recall
    names = CLASS_NAMES.get(dataset_name, {})
    per_class_recall = {}
    for cls_idx, cls_name in names.items():
        mask   = all_labels == cls_idx
        if mask.sum() == 0:
            continue
        recall = (all_preds[mask] == cls_idx).mean()
        per_class_recall[cls_name] = float(recall)

    return {
        "balanced_accuracy": bal_acc,
        "per_class_recall":  per_class_recall,
        "all_preds":         all_preds,
        "all_labels":        all_labels,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. CLASS BALANCE CHECKER
# ─────────────────────────────────────────────────────────────────────────────

def check_class_balance(dataset_names: list[str] = DATASET_NAMES) -> None:
    import pandas as pd

    print("=" * 55)
    print("CLASS BALANCE CHECK")
    print("=" * 55)

    for name in dataset_names:
        paths = get_csv_paths(name)
        if not os.path.exists(paths["train"]):
            print(f"\n[{name}] train.csv not found — skipping")
            continue

        df     = pd.read_csv(paths["train"])
        counts = df["label"].value_counts().sort_index()
        total  = len(df)

        print(f"\n{name}  (n={total})")
        for label, count in counts.items():
            pct  = count / total * 100
            flag = "  ⚠ IMBALANCED" if pct < 40 or pct > 60 else ""
            print(f"  Class {label}: {count:>6}  ({pct:.1f}%){flag}")

    print("\n" + "=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# 2. LR FINDER
# ─────────────────────────────────────────────────────────────────────────────

def lr_finder(
    model,
    train_loader,
    device,
    start_lr:  float = 1e-7,
    end_lr:    float = 1e-1,
    num_iter:  int   = 200,
    save_path: str   = "lr_finder.png",
) -> float:
    """
    Leslie Smith LR Range Test.
    Returns suggested_lr = 10x below the LR at minimum smoothed loss.
    NOTE: treat this as a rough upper bound — for focal loss specifically,
    divide the suggestion by 3-5 before using as your final LR, since
    focal loss amplifies gradients on hard examples.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
    criterion = torch.nn.CrossEntropyLoss()

    lrs, losses = [], []
    best_loss   = float("inf")
    avg_loss    = 0.0
    beta        = 0.98

    mult      = (end_lr / start_lr) ** (1.0 / num_iter)
    lr        = start_lr
    data_iter = iter(train_loader)

    model.train()

    for i in range(num_iter):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed = avg_loss / (1 - beta ** (i + 1))

        lrs.append(math.log10(lr))
        losses.append(smoothed)

        if smoothed < best_loss:
            best_loss = smoothed

        if smoothed > 4 * best_loss:
            print(f"Loss exploded at step {i} — stopping early.")
            break

        lr *= mult
        for pg in optimizer.param_groups:
            pg["lr"] = lr

    best_idx     = losses.index(min(losses))
    suggested_lr = 10 ** (lrs[best_idx] - 1)

    plt.figure(figsize=(9, 4))
    plt.plot(lrs, losses, linewidth=2)
    plt.axvline(x=lrs[best_idx], color="red", linestyle="--",
                label=f"min loss @ lr=10^{lrs[best_idx]:.2f}")
    plt.axvline(x=math.log10(suggested_lr), color="green", linestyle="--",
                label=f"suggested lr = {suggested_lr:.2e}")
    plt.xlabel("log₁₀(Learning Rate)")
    plt.ylabel("Smoothed Loss")
    plt.title("LR Finder — use the green line as your learning rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"\nSuggested LR: {suggested_lr:.2e}  (saved plot → {save_path})")

    return suggested_lr


# ─────────────────────────────────────────────────────────────────────────────
# 3. OPTUNA HYPERPARAMETER SEARCH  ← main changes are here
# ─────────────────────────────────────────────────────────────────────────────

def run_optuna_search(
    dataset_name:  str,
    num_classes:   int,
    device,
    n_trials:      int  = 50,
    search_epochs: int  = 20,
    es_patience:   int  = 5,
    study_name:    str  = None,
) -> optuna.Study:
    """
    Optuna hyperparameter search — now optimising for BALANCED ACCURACY.

    KEY CHANGES vs previous version
    ────────────────────────────────
    1. Objective: balanced_accuracy_score instead of max(val_acc)
       - Forces Optuna to find architectures that recall BOTH classes well
       - Previously a trial that got 94% Neanderthal / 60% Denisovan looked
         great at ~80% accuracy; now it scores (0.94+0.60)/2 = 0.77 and
         loses to a trial that gets 80% / 80% = 0.80 balanced acc

    2. LR search range: [1e-5, 1e-2] instead of [1e-5, 1e-3]
       - Your LR finder identified 9.33e-3 as optimal — this was completely
         outside the old search range, so Optuna was architecturally blind
         to configs that perform well at higher LRs
       - Now the full relevant range is covered

    3. Per-class recall logged as trial user attributes
       - After the search you can call:
             study.trials_dataframe()
         and see Denisovan_recall / Neanderthal_recall for every trial,
         so you can pick a trial that balances the two if needed

    4. Objective evaluated on a FULL val pass at the end of training
       (same as before) but using evaluate_balanced() for consistency

    WHAT TO EXPECT
    ──────────────
    - Trials will be slower than before because evaluate_balanced() runs a
      full val inference pass — negligible cost vs the training epochs
    - Best balanced accuracy will likely be lower than the old best val_acc
      number — this is correct, not a regression. You were measuring the
      wrong thing before.
    - You should see Denisovan recall climb from ~60% toward ~70-75% in the
      best trials, at the cost of maybe 2-3% Neanderthal recall
    - With 50 trials × 20 epochs the search takes roughly the same wall time
      as before
    """
    paths      = get_csv_paths(dataset_name)
    batch_size = cnn_config["batch_size"][dataset_name]

    train_dataset = DNADataset(paths["train"], train=True)
    val_dataset   = DNADataset(paths["val"],   train=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=variable_length_collate,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=variable_length_collate,
    )

    def objective(trial: optuna.Trial) -> float:
        # ── sample architecture ───────────────────────────────────────────
        num_conv  = trial.suggest_int("num_conv_layers", 2, 4)
        num_dense = trial.suggest_int("num_dense_layers", 1, 3)

        config = {
            "num_conv_layers": num_conv,
            "conv_filters": [
                trial.suggest_categorical(f"filters_{i}", [32, 64, 128, 256])
                for i in range(num_conv)
            ],
            "conv_width": [
                trial.suggest_categorical(f"width_{i}", [5, 7, 9, 11, 15, 19])
                for i in range(num_conv)
            ],
            "conv_stride":   1,
            "max_pool_size":   2,
            "max_pool_stride": 2,
            "dropout_rate_conv": [
                trial.suggest_float(f"drop_conv_{i}", 0.1, 0.6)
                for i in range(num_conv)
            ],
            "num_dense_layers": num_dense,
            "dense_filters": [
                trial.suggest_categorical(f"dense_{i}", [64, 128, 256])
                for i in range(num_dense)
            ],
            "dropout_rate_dense": [
                trial.suggest_float(f"drop_dense_{i}", 0.3, 0.6)
                for i in range(num_dense)
            ],
        }

        # ── CHANGE 1: extended LR range ───────────────────────────────────
        # Old range: [1e-5, 1e-3] — missed the 9.33e-3 sweet spot entirely
        # New range: [1e-5, 1e-2] — covers the full relevant region
        lr           = trial.suggest_float("lr",           1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)

        model = CNN(config, num_classes=num_classes).to(device)

        # train_model internally uses FocalLoss + class weights and saves
        # the checkpoint with the best val_acc — that's fine, we override
        # the scoring below with balanced accuracy
        history = train_model(
            model,
            train_loader,
            val_loader,
            device=device,
            num_epochs=search_epochs,
            save_path=f"optuna_trial_{trial.number}.pth",
            train_labels=train_dataset.labels,
            lr=lr,
            weight_decay=weight_decay,
            early_stopping_patience=es_patience,
        )

        # ── CHANGE 2: load the best checkpoint for evaluation ─────────────
        # train_model saves the best val_acc checkpoint — load it so we
        # evaluate the actually-best weights, not the last epoch weights
        # (last epoch may have overfit relative to the saved checkpoint)
        best_ckpt = f"optuna_trial_{trial.number}.pth"
        if os.path.exists(best_ckpt):
            model.load_state_dict(torch.load(best_ckpt, map_location=device))

        # ── CHANGE 3: score on balanced accuracy ──────────────────────────
        result = evaluate_balanced(model, val_loader, device, dataset_name)

        # Log per-class recall so you can inspect them after the search
        # Access via: study.trials_dataframe() or trial.user_attrs
        for cls_name, recall in result["per_class_recall"].items():
            trial.set_user_attr(f"{cls_name}_recall", round(recall, 4))
        trial.set_user_attr("raw_val_acc", round(
            float(np.mean(result["all_preds"] == result["all_labels"])), 4
        ))

        bal_acc = result["balanced_accuracy"]
        print(
            f"  Trial {trial.number:>3} | "
            f"bal_acc={bal_acc:.4f} | "
            + " | ".join(
                f"{k}={v:.3f}" for k, v in result["per_class_recall"].items()
            )
        )

        return bal_acc  # ← Optuna maximises this

    # ── run study ─────────────────────────────────────────────────────────────
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name or f"cnn_{dataset_name}_balanced",
        # Strongly recommended: persist to disk so you can resume if it crashes
        storage=f"sqlite:///optuna_{dataset_name}.db",
        load_if_exists=True,
    )

    print(f"\nStarting Optuna search: {n_trials} trials on '{dataset_name}'")
    print(f"  Objective   : balanced_accuracy_score (mean per-class recall)")
    print(f"  LR range    : [1e-5, 1e-2]  (extended from [1e-5, 1e-3])")
    print(f"  search_epochs={search_epochs}, es_patience={es_patience}\n")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # ── results ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"OPTUNA RESULTS — {dataset_name}")
    print("=" * 60)
    print(f"  Best Balanced Acc : {study.best_value:.4f}")
    print(f"  Best Params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    print(f"\n  Per-class recall at best trial:")
    for k, v in study.best_trial.user_attrs.items():
        print(f"    {k}: {v}")

    # ── full trials dataframe — useful for post-hoc analysis ──────────────────
    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    df.to_csv(f"optuna_results_{dataset_name}.csv", index=False)
    print(f"\n  Full results saved → optuna_results_{dataset_name}.csv")

    # ── plots ─────────────────────────────────────────────────────────────────
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        # Optimization history
        vals = [t.value for t in study.trials if t.value is not None]
        axes[0].plot(vals, marker="o", markersize=3, linewidth=1)
        axes[0].axhline(y=study.best_value, color="red", linestyle="--",
                        label=f"best={study.best_value:.4f}")
        axes[0].set_xlabel("Trial")
        axes[0].set_ylabel("Balanced Accuracy")
        axes[0].set_title(f"Optuna History — {dataset_name}")
        axes[0].legend()

        # Per-class recall scatter across trials
        names = list(CLASS_NAMES.get(dataset_name, {}).values())
        for cls_name in names:
            recalls = [
                t.user_attrs.get(f"{cls_name}_recall", None)
                for t in study.trials
                if t.value is not None
            ]
            trial_nums = [
                t.number for t in study.trials
                if t.value is not None and t.user_attrs.get(f"{cls_name}_recall") is not None
            ]
            axes[1].scatter(trial_nums, recalls, label=cls_name, s=15, alpha=0.7)

        axes[1].set_xlabel("Trial")
        axes[1].set_ylabel("Per-class Recall")
        axes[1].set_title("Per-class Recall Across Trials")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(f"optuna_history_{dataset_name}.png", dpi=150)
        plt.show()
    except Exception as e:
        print(f"  (Plot failed: {e})")

    return study


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LR finder + Optuna search utils")
    parser.add_argument("--task",    type=str, default="DenisovanvsNeanderthal",
                        choices=DATASET_NAMES)
    parser.add_argument("--mode",    type=str, default="balance",
                        choices=["balance", "lr_finder", "optuna"])
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--trials",  type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.mode == "balance":
        check_class_balance()

    elif args.mode == "lr_finder":
        paths         = get_csv_paths(args.task)
        batch_size    = cnn_config["batch_size"][args.task]
        train_dataset = DNADataset(paths["train"], train=True)
        train_loader  = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=variable_length_collate,
        )
        model        = CNN(cnn_config, num_classes=args.classes).to(device)
        suggested_lr = lr_finder(model, train_loader, device,
                                 save_path=f"lr_finder_{args.task}.png")
        print(f"\nUse this LR in your config: {suggested_lr:.2e}")

    elif args.mode == "optuna":
        study = run_optuna_search(
            dataset_name=args.task,
            num_classes=args.classes,
            device=device,
            n_trials=args.trials,
        )