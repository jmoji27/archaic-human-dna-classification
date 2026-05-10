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
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
):
    classes, counts = torch.unique(train_labels, sorted=True, return_counts=True)
    weights = (counts.sum() / counts).float().to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    warmup_epochs = 3
    scheduler = CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-5
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):

        if epoch < warmup_epochs:
            for g in optimizer.param_groups:
                g["lr"] = lr * (epoch + 1) / warmup_epochs

        # ── train ────────────────────────────────────────────────────────
        model.train()
        total_loss, train_correct, train_total = 0.0, 0, 0

        for x, y, mask in train_loader:          # 3-tuple from danq_collate
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model(x, mask=mask)
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
        val_loss_total, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                logits = model(x, mask=mask)
                loss   = criterion(logits, y)
                val_loss_total += loss.item() * x.size(0)
                preds           = torch.argmax(logits, dim=1)
                val_correct    += (preds == y).sum().item()
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
                print(f"  Early stopping at epoch {epoch+1}")
                break

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    print(f"\nBest Val Acc: {best_acc:.4f}  →  saved to {save_path}")
    return history