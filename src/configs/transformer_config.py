transformer_config = {
    # architecture
    "d_model": 128,
    "num_heads": 4,
    "num_layers": 2,
    "dim_feedforward": 256,
    "dropout": 0.1,

    # training
    "batch_size": {
        "binary": 64,
        "multiclass": 32,
        "bottleneck": 32,
    },
    "lr": {
        "binary": 5e-4,
        "multiclass": 3e-4,
        "bottleneck": 3e-4,
    },
}