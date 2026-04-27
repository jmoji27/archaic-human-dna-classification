mlp_config = {
    "vocab_size": 64,  # 4^3 for k=3
    "hidden_dims": [256, 128],
    "dropout": 0.3,
    
    "batch_size": {
        "binary": 64,
        "multiclass": 64,
        "longerbp": 64,
    },
    "lr": {
        "binary": 0.001,
        "multiclass": 0.001,
        "longerbp": 0.001,
    }
}