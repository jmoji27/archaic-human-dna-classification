import torch
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_danq(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs,
    save_path,
    train_labels,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 20,
):
    # Label smoothing: small for binary, off for multiclass.
    # With only 3 classes, smoothing=0.1 over-penalises confident correct
    # predictions — so we drop it to 0 for multiclass.
    num_classes = int(train_labels.max().item()) + 1
    smoothing   = 0.05 if num_classes == 2 else 0.0

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=smoothing)

    # AdamW: same as Adam but weight decay is applied correctly
    # (decoupled from the gradient update, prevents overfitting more reliably)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    warmup_epochs = 3
    scheduler = CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-5
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):

        # Linear warm-up for the first few epochs
        if epoch < warmup_epochs:
            for g in optimizer.param_groups:
                g["lr"] = lr * (epoch + 1) / warmup_epochs

        # ── Train ────────────────────────────────────────────────────────
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model(x, mask=mask)
            loss   = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += y.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        train_acc  = correct    / total

        # ── Validate ─────────────────────────────────────────────────────
        model.eval()
        val_loss_total, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                logits          = model(x, mask=mask)
                loss            = criterion(logits, y)
                val_loss_total += loss.item() * x.size(0)
                val_correct    += (logits.argmax(1) == y).sum().item()
                val_total      += y.size(0)

        val_loss = val_loss_total / len(val_loader.dataset)
        val_acc  = val_correct    / val_total

        if epoch >= warmup_epochs:
            scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1:>3}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    print(f"\nBest Val Acc: {best_acc:.4f}  →  {save_path}")
    return history
