#  - "large" -> Human vs Neanderthal, Multiclass, original
# ==============================================================================
ARCH_MODE = "large" 

rnn_config = {
    # Architecture (automatically driven by ARCH_MODE)
    "hidden_size": 64 if ARCH_MODE == "small" else 128,
    "num_layers":  1 if ARCH_MODE == "small" else 2,
    "dropout":     0.3,

    # Training parameters
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