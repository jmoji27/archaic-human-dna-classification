rnn_config = {
    # architecture
    "hidden_size": 128,
    "num_layers":  2,
    "dropout":     0.3,

    # training
    "batch_size": {
        "binary": 64,
        "multiclass": 32,
        "bottleneck": 32,
    },
    "lr": {
        "binary": 1e-3,
        "multiclass": 5e-4,
        "bottleneck": 5e-4,
    },
}