import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
from datetime import datetime

from src.models.CNN import CNN1D
from src.models.RNN import RNNModel
from src.models.Transformer import TransformerModel
from src.models.Danq import DanqModel
from src.dataset import DNADataset
from src.train import train_model

# import configs
from src.configs.cnn_config import cnn_config
from src.configs.lstm_config import rnn_config
from src.configs.transformer_config import transformer_config
from src.configs.danq_config import danq_config


def get_hparams(config, dataset_type):
    return {
        "batch_size": config["batch_size"][dataset_type],
        "lr": config["lr"][dataset_type],
    }


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--dataset_type", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── model selection + config ownership
    model_map = {
        "cnn":         (CNN1D, cnn_config),
        "rnn":         (RNNModel, rnn_config),
        "transformer": (TransformerModel, transformer_config),
        "danq":        (DanqModel, danq_config),
    }

    if args.model not in model_map:
        raise ValueError(f"Unknown model: {args.model}")

    ModelClass, config = model_map[args.model]

    # ── get hyperparameters from config
    params = get_hparams(config, args.dataset_type)
    batch_size = params["batch_size"]
    lr         = params["lr"]

    # ── datasets
    train_dataset = DNADataset(args.train_path, train=True)
    val_dataset   = DNADataset(args.val_path, train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}  Batch: {batch_size}  LR: {lr}")

    # ── model init
    model = ModelClass(config, args.num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model} | Parameters: {n_params:,}")

    # ── train
    save_dir  = f"results/{args.model}/{args.dataset_type}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{args.model}_{args.dataset_type}_best_model.pt"

    history = train_model(
        model,
        train_loader,
        val_loader,
        device,
        num_epochs=args.epochs,
        save_path=save_path,
        lr=lr,
    )

    # ── save metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_data = {
        "model": args.model,
        "dataset": args.dataset_type,
        "model_config": config,
        "training_config": {
            "batch_size": batch_size,
            "lr": lr,
            "epochs": args.epochs
        },
        "history": history,
    }

    out_file = f"{save_dir}/{args.model}_{args.dataset_type}_run_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(run_data, f, indent=4)

    print(f"Run saved → {out_file}")


if __name__ == "__main__":
    main()