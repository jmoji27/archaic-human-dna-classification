cnn_config = {
    # architecture
    "num_conv_layers":    4,
    "conv_filters":       [64, 128, 256, 256],
    "conv_width":         [15, 11,  7, 5],
    "conv_stride":        1,
    "max_pool_size":      3,
    "max_pool_stride":    2,
    "dropout_rate_conv":  [0.3, 0.4, 0.5, 0.5],

    "num_dense_layers":   2,
    "dense_filters":      [128, 64],
    "dropout_rate_dense": [0.5, 0.5],

    # training per dataset
    "batch_size": {
        "original": 256,
        "longerbp": 64,
        "multiclass": 32,
        "bottleneck": 32,
        "HumanvsNeanderthal": 64,
        "DenisovanvsNeanderthal": 64
    },
    "lr": {
        "original": 5e-5, #previous was 1e-4
        "longerbp": 1e-4,
        "multiclass": 3e-4,
        "bottleneck": 1e-4,
        "HumanvsNeanderthal": 5e-5,
        "DenisovanvsNeanderthal": 5e-5,
    },
}