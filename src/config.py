import numpy as np
from sklearn.model_selection import KFold

# =============================================================================
# A. Data Configuration (MUST BE UPDATED LOCALLY)
# =============================================================================

# >>> ⚠️ CHANGE THIS TO YOUR ACTUAL ABSOLUTE FILE PATH ⚠️ <<<
DATA_FILE_PATH = r"final data 21-11-2025.xlsx" 

# Target Column (Compressive Strength) - Final CORRECTED name
TARGET_COL = "Compressive Strength, MPa" 

# All Available Feature Columns (13 features based on your header list)
FEATURE_COLS = [
    'Age, days',    # Must be first for consistency 
    'C, kg/m3', 
    'M, kg/m3', 
    'CC, kg/m3', 
    'LS, kg/m3', 
    'g, kg/m3', 
    'CB, kg/m3', 
    'G, kg/m3', 
    'SF, kg/m3', 
    'FA, kg/m3', 
    'CA, kg/m3', 
    'W, kg/m3', 
    'SP, kg/m3',
]

# 5-Fold Cross-Validation Strategy
CV_STRATEGY = KFold(n_splits=5, shuffle=True, random_state=66)

# =============================================================================
# B. Optimal Model Hyperparameters (Simplified Names)
# =============================================================================

# Gradient Boosting (GB) parameters
BEST_GB_PARAMS = {
    'n_estimators': 300, 
    'max_depth': 4, 
    'min_samples_split': 2, 
    'learning_rate': 0.1,
    'random_state': 42
}

# XGBoost (XGBoost) parameters
BEST_XGBOOST_PARAMS = {
    'n_estimators': 350, 
    'max_depth': 6, 
    'learning_rate': 0.1, 
    'subsample': 0.7,
    'colsample_bytree': 0.7, 
    'random_state': 42
}

# Multi-Layer Perceptron (MLP) parameters
BEST_MLP_PARAMS = {
    'hidden_layer_sizes': (100, 50, 25), 
    'activation': 'relu', 
    'solver': 'adam', 
    'max_iter': 500, 
    'random_state': 42
}

# Random Forest (RF) parameters
BEST_RF_PARAMS = {
    'n_estimators': 300, 
    'max_depth': 10, 
    'min_samples_split': 5,
    'random_state': 42
}

# =============================================================================
# C. Simple Ensemble Weights and Deployment Test Data
# =============================================================================

# Simple Ensemble Weights (50/50 Split)
SIMPLE_ALPHA_GB = 0.50
SIMPLE_WEIGHT_XGB = 0.50

# Example mix for deployment prediction (MUST match the 13 features in FEATURE_COLS order)
EXAMPLE_MIX = np.array([
    365,    # Age, days
    250.0,  # C, kg/m3
    10.0,   # M, kg/m3
    5.0,    # CC, kg/m3
    0.0,    # LS, kg/m3
    5.0,    # g, kg/m3
    5.0,    # CB, kg/m3
    5.0,    # G, kg/m3
    0.0,    # SF, kg/m3
    180.0,  # FA, kg/m3
    1100.0, # CA, kg/m3
    150.0,  # W, kg/m3
    10.0    # SP, kg/m3
])