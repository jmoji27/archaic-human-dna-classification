danq_config = {
    # architecture
    "conv_filters": 320,
    "kernel_size":  26,
    "lstm_hidden":  320,

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