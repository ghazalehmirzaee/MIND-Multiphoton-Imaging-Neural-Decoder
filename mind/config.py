"""
Fixed configuration with proper parameter separation for each model type.
The key insight: remove all normalization to preserve natural signal characteristics.
"""
import torch

DEFAULT_CONFIG = {
    # Data parameters
    "data": {
        "window_size": 15,
        "step_size": 1,
        "test_size": 0.15,
        "val_size": 0.15,
        "batch_size": 32,
        "num_workers": 4,
        "binary_classification": True,
        "mat_file": "/home/ghazal/Documents/NS_Projects/NS_P2_050325/MIND-Multiphoton-Imaging-Neural-Decoder/data/raw/SFL13_5_8112021_003_new_modified.mat",
        "xlsx_file": "/home/ghazal/Documents/NS_Projects/NS_P2_050325/MIND-Multiphoton-Imaging-Neural-Decoder/data/raw/SFL13_5_8112021_003_new.xlsx"
    },

    # Model parameters - ALL WITHOUT NORMALIZATION
    "models": {
        "random_forest": {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
            "n_jobs": -1,
            "random_state": 42,
            "criterion": "gini",
            "bootstrap": True
        },
        "svm": {
            # Core SVM parameters (no preprocessing parameters here)
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "class_weight": "balanced",
            "probability": True,
            "random_state": 42,
            # PCA parameters handled separately in model
            "use_pca": False,  # Disabled to preserve raw signal characteristics
            "pca_variance": 0.95  # Only used if use_pca=True
        },
        "mlp": {
            "hidden_layer_sizes": (64, 128, 32),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "batch_size": "auto",
            "learning_rate": "adaptive",
            "learning_rate_init": 0.0001,  # Reduced for better stability with raw signals
            "max_iter": 500,  # Increased for better convergence
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 20,  # Increased patience
            "random_state": 42
        },
        "fcnn": {
            "hidden_dims": [256, 128, 64],
            "output_dim": 2,
            "dropout_rate": 0.4,
            "learning_rate": 0.0001,  # Reduced for stability
            "weight_decay": 1e-5,
            "batch_size": 32,
            "num_epochs": 100,
            "patience": 15,
            "random_state": 42
        },
        "cnn": {
            "n_filters": [32, 64, 128],
            "kernel_size": 5,
            "output_dim": 2,
            "dropout_rate": 0.2,
            "learning_rate": 0.0005,
            "weight_decay": 1e-4,
            "batch_size": 32,
            "num_epochs": 50,
            "patience": 10,
            "random_state": 42
        }
    },

    # Training parameters
    "training": {
        "optimize_hyperparams": False,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "output_dir": "outputs/results"
    },

    # Visualization parameters with scientific colors
    "visualization": {
        "output_dir": "outputs/figures",
        "dpi": 300,
        "format": "png",
        "signal_colors": {
            "calcium_signal": "#356d9e",  # Scientific blue
            "deltaf_signal": "#4c8b64",   # Scientific green
            "deconv_signal": "#a85858"    # Scientific red
        }
    },

    # Binary classification parameters
    "classification": {
        "task": "binary",
        "labels": ["No footstep", "Contralateral"],
        "n_classes": 2
    }
}


def get_config():
    """Get default configuration with device check."""
    if not torch.cuda.is_available() and DEFAULT_CONFIG["training"]["device"] == "cuda":
        DEFAULT_CONFIG["training"]["device"] = "cpu"

    return DEFAULT_CONFIG

