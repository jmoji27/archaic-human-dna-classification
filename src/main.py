import os
import json
import time
import argparse
import torch
from torch.utils.data import DataLoader

from src.models.CNN import CNN1D
from src.models.RNN import RNNModel
from src.models.Transformer import TransformerModel
from src.models.Danq import DanqModel

from src.dataset import DNADataset
from src.train import train_model

from datetime import datetime


def main():
    # --- reproducibility ---
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--dataset_type", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- dataset-dependent training config ---
    if args.dataset_type == "binary":
        batch_size = 64
        epochs = args.epochs

    elif args.dataset_type == "multiclass":
        batch_size = 32
        epochs = args.epochs

    elif args.dataset_type == "bottleneck":
        batch_size = 32
        epochs = args.epochs

    else:
        raise ValueError("Invalid dataset type")

    # --- model configs (fixed across datasets) ---
    cnn_config = {
        "num_conv_layers": 2,
        "conv_filters": [64, 128],
        "conv_width": [7, 5],
        "conv_stride": 1,
        "dropout_rate_conv": 0.3,
        "max_pool_size": 2,
        "max_pool_stride": 2,
        "num_dense_layers": 1,
        "dense_filters": 128,
        "dropout_rate_dense": 0.5
    }

    rnn_config = {
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.3
    }

    transformer_config = {
        "d_model": 128,
        "num_heads": 4,
        "num_layers": 2,
        "dim_feedforward": 256,
        "dropout": 0.1
    }

    danq_config = {
        "conv_filters": 320,
        "kernel_size": 26,
        "lstm_hidden": 320
    }

    # --- dataset ---
    train_dataset = DNADataset(args.train_path, train=True)
    val_dataset = DNADataset(args.val_path, train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # --- model selection ---
    if args.model == "cnn":
        config = cnn_config
        model = CNN1D(config, args.num_classes)

    elif args.model == "rnn":
        config = rnn_config
        model = RNNModel(config, args.num_classes)

    elif args.model == "transformer":
        config = transformer_config
        model = TransformerModel(config, args.num_classes)

    elif args.model == "danq":
        config = danq_config
        model = DanqModel(config, args.num_classes)

    else:
        raise ValueError("Invalid model type")

    model = model.to(device)

    # --- train ---
    history = train_model(model, train_loader, val_loader, device, epochs)

    # --- prepare run data ---
    run_data = {
        "model": args.model,
        "dataset": args.dataset_type,
        "model_config": config,
        "training_config": {
            "batch_size": batch_size,
            "epochs": epochs
        },
        "history": history
    }

    # --- save results ---
    os.makedirs("results", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_dir = f"results/{args.model}/{args.dataset_type}"
    os.makedirs(save_dir, exist_ok=True)

    filename = f"{save_dir}/{args.model}_{args.dataset_type}_run_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(run_data, f, indent=4)

    


if __name__ == "__main__":
    main()