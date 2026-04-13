import torch

def train_model(model, train_loader, val_loader, device, num_epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = {
        "train_loss": [],
        "val_acc": []
    }

    for epoch in range(num_epochs):
        model.train()

        total_loss = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        # averaging
        total_loss /= len(train_loader.dataset)

        # --- validation ---
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                outputs = model(x)
                preds = torch.argmax(outputs, dim=1)

                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / total

        # --- store history ---
        history["train_loss"].append(total_loss)
        history["val_acc"].append(acc)

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss:.4f} | Val Acc: {acc:.4f}")

    return history