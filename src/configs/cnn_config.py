cnn_config = {
    # architecture
    "num_conv_layers":    4,
    "conv_filters":       [64, 128, 256, 256],
    "conv_width":         [15, 11,  7, 7],
    "conv_stride":        1,
    "max_pool_size":      3,
    "max_pool_stride":    2,
    "dropout_rate_conv":  [0.2, 0.3, 0.4, 0.4],

    "num_dense_layers":   2,
    "dense_filters":      [256, 128],
    "dropout_rate_dense": [0.5, 0.5],

    # training per dataset
    "batch_size": {
        "original": 32,
        "longerbp": 32,
        "multiclass": 64,
        "bottleneck": 64,
        "HumanvsNeanderthal": 32,
        "DenisovanvsNeanderthal": 32
    },
    "lr": {
        "original": 5e-5, #previous was 1e-4
        "longerbp": 1e-4,
        "multiclass": 3e-4,
        "bottleneck": 1e-4,
        "HumanvsNeanderthal": 1e-4,
        "DenisovanvsNeanderthal": 1e-4,
    },
}