"""Configuration constants for the Fraud Review Workbench."""

# Reproducibility
RANDOM_STATE = 42

# Data split
TEST_SIZE = 0.20

# Cross-validation
N_SPLITS = 5

# Threshold selection grid
THRESHOLD_GRID = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

# Decision bands (probability thresholds for approve/review/block)
# These are demonstration values - adjust based on analysis
APPROVE_THRESHOLD = 0.25
BLOCK_THRESHOLD = 0.70

# Model configurations
LOGISTIC_REGRESSION_CONFIG = {
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

RANDOM_FOREST_CONFIG = {
    "n_estimators": 300,
    "min_samples_leaf": 2,
    "class_weight": "balanced_subsample",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

SMOTE_CONFIG = {
    "random_state": RANDOM_STATE,
    "k_neighbors": 5,
}

# Artifact paths
FIGURES_DIR = "artifacts/figures"
METRICS_DIR = "artifacts/metrics"
MODELS_DIR = "artifacts/models"