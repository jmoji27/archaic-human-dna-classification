import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()



def train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs,
    save_path,
    train_labels,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    early_stopping_patience: int = 15,  # stop if no improvement for N epochs
):
    # Class-weighted loss — computed from train_labels before any training
    classes, counts = torch.unique(train_labels, sorted=True, return_counts=True)
    total   = counts.sum()
    weights = (total / counts).float().to(device)
    criterion = FocalLoss(gamma=2.0, weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Step on val_acc (mode='max') so scheduler triggers when learning stalls
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=8, factor=0.5, verbose=True
    )

    history = {
        "train_loss": [],
        "val_loss":   [],
        "train_acc":  [],
        "val_acc":    [],
    }

    best_acc = 0.0
    epochs_without_improvement = 0  # early stopping counter

    for epoch in range(num_epochs):
        # ── train ────────────────────────────────────────────────────────
        model.train()
        total_loss    = 0.0
        train_correct = 0
        train_total   = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss    += loss.item() * x.size(0)
            preds          = torch.argmax(logits, dim=1)
            train_correct += (preds == y).sum().item()
            train_total   += y.size(0)

        train_loss = total_loss    / len(train_loader.dataset)
        train_acc  = train_correct / train_total

        # ── validate ─────────────────────────────────────────────────────
        model.eval()
        val_loss_total = 0.0
        val_correct    = 0
        val_total      = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y   = x.to(device), y.to(device)
                logits  = model(x)
                loss    = criterion(logits, y)
                val_loss_total += loss.item() * x.size(0)
                preds           = torch.argmax(logits, dim=1)
                val_correct    += (preds == y).sum().item()
                val_total      += y.size(0)

        val_loss = val_loss_total / len(val_loader.dataset)
        val_acc  = val_correct    / val_total

        # ── scheduler + checkpoint ───────────────────────────────────────
        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_without_improvement = 0  # reset counter on improvement
            torch.save(model.state_dict(), save_path)
        else:
            epochs_without_improvement += 1

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"No improvement: {epochs_without_improvement}/{early_stopping_patience}")

        # ── early stopping ───────────────────────────────────────────────
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs "
                  f"({early_stopping_patience} epochs without improvement).")
            break

    print(f"\nBest Val Acc: {best_acc:.4f}  →  saved to {save_path}")
    return history