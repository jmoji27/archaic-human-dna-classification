cnn_config = {
    # architecture
    "num_conv_layers":    4,
    "conv_filters":       [128, 64, 64, 32],
    "conv_width":         [19, 19, 15, 5],
    "conv_stride":        1,
    "max_pool_size":      3,
    "max_pool_stride":    2,
    "dropout_rate_conv":  [0.35, 0.23, 0.5, 0.34],

    "num_dense_layers":   2,
    "dense_filters":      [256, 64],
    "dropout_rate_dense": [0.38, 0.43],

    # training per dataset
    "batch_size": {
        "original": 256,
        "longerbp": 64,
        "multiclass": 128,
        "bottleneck": 32,
        "HumanvsNeanderthal": 128,
        "DenisovanvsNeanderthal": 256
    },
    "lr": {
        "original": 5e-3, #previous was 1e-4
        "longerbp": 1e-4,
        "multiclass": 5e-3,
        "bottleneck": 1e-4,
        "HumanvsNeanderthal": 1.45e-4,
        "DenisovanvsNeanderthal": 1.5e-3,
    },
    "weight_decay": {
        "original": 0.0042, #previous was 1e-3
        "longerbp": 0.0042,
        "multiclass": 1.3e-3,
        "bottleneck": 0.0042,
        "HumanvsNeanderthal": 2.5e-3,
        "DenisovanvsNeanderthal": 2.34e-3,
    }
}