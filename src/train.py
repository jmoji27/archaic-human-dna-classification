import torch
from torch.utils.data import DataLoader, Dataset


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs,
    save_path,
    train_labels,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
):
    # Class-weighted loss — computed from train_labels before any training
    classes, counts = torch.unique(train_labels, sorted=True, return_counts=True)
    total   = counts.sum()
    weights = (total / counts).float().to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Step on val_acc (mode='max') so scheduler triggers when learning stalls
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=4, factor=0.5, verbose=True
    )

    # FIX: history now tracks all four metrics that main.py's plot function
    # expects. Previously only train_loss and val_acc were populated, causing
    # a KeyError when plot_training_history accessed train_acc and val_loss.
    history = {
        "train_loss": [],
        "val_loss":   [],   # FIX: was missing
        "train_acc":  [],   # FIX: was missing
        "val_acc":    [],
    }
    best_acc = 0.0

    for epoch in range(num_epochs):

        # ── train ────────────────────────────────────────────────────────
        model.train()
        total_loss   = 0.0
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
        train_acc  = train_correct / train_total   # FIX: computed alongside loss

        # ── validate ─────────────────────────────────────────────────────
        model.eval()
        val_loss_total = 0.0
        val_correct    = 0
        val_total      = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y   = x.to(device), y.to(device)
                logits  = model(x)
                loss    = criterion(logits, y)          # FIX: compute val loss
                val_loss_total += loss.item() * x.size(0)
                preds           = torch.argmax(logits, dim=1)
                val_correct    += (preds == y).sum().item()
                val_total      += y.size(0)

        val_loss = val_loss_total / len(val_loader.dataset)  # FIX: was never computed
        val_acc  = val_correct    / val_total

        # ── scheduler + checkpoint ───────────────────────────────────────
        scheduler.step(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)       # FIX: populate val_loss
        history["train_acc"].append(train_acc)     # FIX: populate train_acc
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    print(f"\nBest Val Acc: {best_acc:.4f}  →  saved to {save_path}")
    return history