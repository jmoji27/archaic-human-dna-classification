danq_config = {
    # Conv motif scanner
    # Original paper: 320 filters, width 26 for 1000bp
    # Scaled for 35-85bp: smaller width so kernel fits on shortest reads
    "conv_filters": 128,
    "conv_width":   11,   # 11bp — captures transcription-factor-sized motifs

    # Pooling
    # Original: size 13, stride 13 (aggressive for 1000bp)
    # For 35-85bp: gentle pool so BiLSTM still sees spatial structure
    "pool_size":    3,
    "pool_stride":  2,

    # BiLSTM
    # Original: 320 hidden units
    # Scaled down — sequence is short, don't need massive hidden state
    "lstm_hidden":  128,
    "lstm_layers":  1,    # start with 1, add second only if underfitting

    # Dropout
    "dropout_conv":  0.1,
    "dropout_lstm":  0.2,
    "dropout_dense": 0.3,

    # Dense head
    "dense_units": 128,

    # Training — same LR structure as cnn_config
    "lr": {
        "original":   3e-4,
        "longerbp":   3e-4,
        "multiclass": 3e-4,
        "bottleneck": 3e-4,
    },
    "batch_size": {
        "original":   32,
        "longerbp":   32,
        "multiclass": 32,
        "bottleneck": 64,
    },
}