import numpy as np # <-- CRITICAL FIX: Add numpy import

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from src.config import BEST_GB_PARAMS, BEST_XGBOOST_PARAMS, BEST_MLP_PARAMS, BEST_RF_PARAMS

# Check if XGBoost is available
try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except ImportError:
    XGBRegressor = None
    HAVE_XGB = False
    print("WARNING: XGBoost not installed. Skipping XGBoost-based models.")

# =============================================================================
# MODEL DEFINITION
# =============================================================================

def get_high_performance_models():
    """Returns a dictionary of tuned regressors with simplified keys."""
    models = {
        "GB": GradientBoostingRegressor(**BEST_GB_PARAMS),
        "Random Forest": RandomForestRegressor(**BEST_RF_PARAMS),
        "MLP": MLPRegressor(**BEST_MLP_PARAMS),
    }

    if HAVE_XGB:
        xgb_reg = XGBRegressor(**BEST_XGBOOST_PARAMS)
        models["XGBoost"] = xgb_reg

        # Define Stacking Regressor based on top candidates
        estimators = [
            ('gb', models["GB"]),
            ('xgb', xgb_reg),
            ('mlp', models["MLP"])
        ]
        
        # Use Ridge as the final meta-estimator
        stacking_reg = StackingRegressor(
            estimators=estimators, 
            final_estimator=Ridge(alpha=1.0) 
        )
        models["Stacking (GB+XGB+MLP)"] = stacking_reg
        
    return models

# =============================================================================
# METRICS AND EVALUATION
# =============================================================================

def evaluate_regression_model(y_true, y_pred, model_name="Model"):
    """Calculates standard and PWA metrics for regression model evaluation."""
    
    # Standard Metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    r = np.sqrt(np.abs(r2)) # Correlation coefficient R

    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # PWA (Prediction Within Accuracy) Calculation
    def calculate_pwa(y_true, y_pred, tolerance):
        relative_error = np.abs((y_true - y_pred) / y_true)
        pwa = np.mean(relative_error <= tolerance) * 100
        return pwa

    pwa_5 = calculate_pwa(y_true, y_pred, 0.05)
    pwa_10 = calculate_pwa(y_true, y_pred, 0.10)
    pwa_15 = calculate_pwa(y_true, y_pred, 0.15)

    print(f"| {model_name:<25} | RMSE: {rmse:.4f} | R²: {r2:.4f} | PWA_10%: {pwa_10:.2f}%")

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2,
        "R": r,
        "MAPE": mape,
        "PWA_5%": pwa_5,
        "PWA_10%": pwa_10,
        "PWA_15%": pwa_15,
    }