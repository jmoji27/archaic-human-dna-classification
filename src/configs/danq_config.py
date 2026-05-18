danq_config = {
    # Conv — single kernel width; 11bp captures TF-binding-site-sized motifs
    "conv_filters": 128,
    "conv_width":   11,

    # Pooling — gentle for short sequences
    # Original paper used size=13, stride=13 on 1000bp; scaled down for 35-120bp
    "pool_size":   2,
    "pool_stride": 2,

    # BiLSTM
    "lstm_hidden": 128,
    "lstm_layers": 1,

    # Dropout
    "dropout_conv":  0.1,
    "dropout_lstm":  0.2,
    "dropout_dense": 0.3,

    # Learning rates per dataset
    "lr": {
        "original":               3e-4,
        "longerbp":               3e-4,
        "multiclass":             1e-4,
        "bottleneck":             3e-4,
        "HumanvsNeanderthal":     3e-4,
        "DenisovanvsNeanderthal": 3e-4,
    },
    "batch_size": {
        "original":               32,
        "longerbp":               32,
        "multiclass":             32,
        "bottleneck":             64,
        "HumanvsNeanderthal":     32,
        "DenisovanvsNeanderthal": 32,
    },
}
