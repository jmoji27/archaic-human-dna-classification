cnn_config = {
    # architecture
    "num_conv_layers":    3,
    "conv_filters":       [64, 128, 256],
    "conv_width":         [7, 5, 3],
    "conv_stride":        1,
    "max_pool_size":      2,
    "max_pool_stride":    2,
    "dropout_rate_conv":  0.2,

    "num_dense_layers":   1,
    "dense_filters":      128,
    "dropout_rate_dense": 0.1,

    # training per dataset
    "batch_size": {
        "binary": 64,
        "multiclass": 32,
        "bottleneck": 32,
    },
    "lr": {
        "binary": 1e-3,
        "multiclass": 1e-4,
        "bottleneck": 1e-4,
    },
}