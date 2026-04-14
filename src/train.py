import torch


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs,
    save_path,
    lr: float = 1e-3,           # ← now a parameter, not hardcoded
    weight_decay: float = 0.0,
):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Step on val_acc (mode='max') so it actually triggers when learning stalls
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=4, factor=0.5, verbose=True
    )

    history = {"train_loss": [], "val_acc": []}
    best_acc = 0.0

    for epoch in range(num_epochs):
        # ── train ────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        total_loss /= len(train_loader.dataset)

        # ── validate ─────────────────────────────────────────────────────────
        model.eval()
        correct = 0
        total   = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds    = torch.argmax(model(x), dim=1)
                correct += (preds == y).sum().item()
                total   += y.size(0)
        acc = correct / total

        # ── scheduler + checkpoint ───────────────────────────────────────────
        scheduler.step(acc)             # step on val_acc, not loss
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)

        history["train_loss"].append(total_loss)
        history["val_acc"].append(acc)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss:.4f} | Val Acc: {acc:.4f}")

    print(f"\nBest Val Acc: {best_acc:.4f}  →  saved to {save_path}")
    return history