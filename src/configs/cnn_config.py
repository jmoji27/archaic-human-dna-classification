cnn_config = {
    # architecture
    "num_conv_layers":    4,
    "conv_filters":       [128, 64, 64, 32],
    "conv_width":         [19, 19, 15, 5],
    "conv_stride":        1,
    "max_pool_size":      3,
    "max_pool_stride":    2,
    "dropout_rate_conv":  [0.4, 0.2, 0.5, 0.4],

    "num_dense_layers":   2,
    "dense_filters":      [64,256],
    "dropout_rate_dense": [0.38, 0.43],

    # training per dataset
    "batch_size": {
        "original": 1024,
        "longerbp": 64,
        "multiclass": 128,
        "bottleneck": 512,
        "HumanvsNeanderthal": 128,
        "DenisovanvsNeanderthal": 256,
    },
    "lr": {
        "original": 1e-3, #previous was 1e-4
        "longerbp": 5e-4,
        "multiclass": 1e-4,
        "bottleneck": 5e-4,
        "HumanvsNeanderthal": 1.45e-4,
        "DenisovanvsNeanderthal": 1.2e-4,
    },
    "weight_decay": {
        "original": 1e-5, #since its rlly large dataset, we can afford less regularization
        "longerbp": 1e-4,
        "multiclass": 1e-3,
        "bottleneck": 5e-4,
        "HumanvsNeanderthal": 1e-4,
        "DenisovanvsNeanderthal":1e-3,
    }
}