# =============================================================================
# main_thesis_script.py: Thesis Execution and Orchestration (Q1 Journal Standard)
# =============================================================================

import pandas as pd
import numpy as np 
import joblib
from tqdm import tqdm
from sklearn.model_selection import cross_val_predict, train_test_split 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# Import Modules
from src.config import (
    DATA_FILE_PATH, TARGET_COL, FEATURE_COLS, CV_STRATEGY, 
    SIMPLE_ALPHA_GB, SIMPLE_WEIGHT_XGB, EXAMPLE_MIX
)
from src.models import get_high_performance_models, evaluate_regression_model, HAVE_XGB
from src.visuals import (
    plot_model_performance, 
    plot_residuals_vs_features, 
    plot_feature_importance, 
    plot_final_metric_comparison,
    # === CRITICAL FIX: ADD THIS NEW FUNCTION ===
    plot_test_set_comparison 
)

# Define the global split ratio
TEST_SIZE_RATIO = 0.20 # 20% of data reserved for final, unseen test

# =============================================================================
## 1. DATA LOADING AND PREPROCESSING SETUP
# =============================================================================

def load_and_prepare_data():
    """Loads data, cleans it, and defines the preprocessing pipeline."""
    print("1. Loading, cleaning, and splitting data...")
    df = pd.read_excel(DATA_FILE_PATH, sheet_name=0)
    df = df.dropna(how="all").loc[:, ~df.columns.str.contains(r"^Unnamed")]
    
    # Corrected target column reference and feature selection
    df = df.dropna(subset=[TARGET_COL]) 
    X = df[FEATURE_COLS].copy() 
    y = df[TARGET_COL].copy()

    # CRITICAL CHANGE: SPLIT DATA INTO TRAIN/TEST
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE_RATIO, random_state=60 # Fixed random state for reproducibility
    )
    print(f"   -> Total Samples: {len(X)}")
    print(f"   -> Training Samples (for CV/Tuning): {len(X_train)}")
    print(f"   -> Test Samples (for Final Evaluation): {len(X_test)}")


    # Define Preprocessing Pipeline
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)],
        remainder='passthrough'
    )
    
    # Returns the split data and the unfitted preprocessing object
    return X_train, X_test, y_train, y_test, preprocess

# =============================================================================
## 2. MODEL TRAINING AND EVALUATION (CV on TRAIN DATA)
# =============================================================================

def train_and_evaluate_cv(X_train, X_test, y_train, y_test, preprocess):
    """
    Trains all models, collects CV predictions and metrics, fits final pipelines on X_train, 
    and then evaluates those final pipelines on the unseen X_test data.
    """
    print("\n2. Training and evaluating models (5-Fold CV on Training Data)...")
    high_perf_models = get_high_performance_models() 
    metrics_all = {}
    predictions_cv = {}
    pipelines_final = {} 
    
    MODELS_TO_FULL_PLOT = ["GB", "XGBoost", "Stacking (GB+XGB+MLP)"]

    for model_name, reg in high_perf_models.items():
        print("-" * 70)
        print(f"🏃‍♂️ Evaluating Model: **{model_name}**")
        
        pipe = Pipeline(steps=[("preprocess", preprocess), ("model", reg)])
        
        # --- A) CV Prediction and Metrics (on X_train) ---
        y_pred_cv = cross_val_predict(pipe, X_train, y_train, cv=CV_STRATEGY, n_jobs=-1)
        predictions_cv[model_name] = y_pred_cv

        # Store CV metrics
        metrics_all[f"{model_name} (CV)"] = evaluate_regression_model(y_train, y_pred_cv, model_name=f"{model_name} (CV)")
        
        # --- B) Fit final pipeline on entire training set (X_train) ---
        pipe.fit(X_train, y_train)
        pipelines_final[model_name] = pipe
        
        # Save the FINAL pipeline (FIT on X_train)
        safe_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')
        joblib.dump(pipe, f"final_pipeline_{safe_name}.joblib")
        
        # --- C) Test Set Evaluation (on X_test) ---
        y_pred_test = pipe.predict(X_test)
        # Store Test Set metrics
        metrics_all[f"{model_name} (Test Set)"] = evaluate_regression_model(y_test, y_pred_test, model_name=f"{model_name} (Test Set)")
        print(f"   -> RMSE (Test Set): {metrics_all[f'{model_name} (Test Set)']['RMSE']:.4f} MPa")
        
        # Generate CV visualizations (measures model robustness/bias)
        if model_name in MODELS_TO_FULL_PLOT:
            plot_model_performance(y_train, y_pred_cv, model_name=f"{model_name} (CV)")
            
    return metrics_all, predictions_cv, pipelines_final

# =============================================================================
## 3. ENSEMBLE ANALYSIS AND FINAL TEST (Simple Average)
# =============================================================================

def perform_ensemble_analysis_and_test(X_train, X_test, y_train, y_test, metrics_all, predictions_cv, pipelines_final):
    """
    Calculates metrics for the Simple Average Ensemble (GB + XGB) on CV data 
    AND evaluates the final ensemble on the unseen X_test data.
    """
    
    if HAVE_XGB and "XGBoost" in predictions_cv and "GB" in predictions_cv: 
        model_name_ens = "GB + XGB Simple Ensemble"
        
        # --- A) Ensemble Performance on CV Data ---
        y_pred_gb_cv = predictions_cv["GB"]
        y_pred_xgb_cv = predictions_cv["XGBoost"]
        y_pred_simple_cv = (SIMPLE_ALPHA_GB * y_pred_gb_cv) + (SIMPLE_WEIGHT_XGB * y_pred_xgb_cv)
        
        metrics_all[f"{model_name_ens} (CV)"] = evaluate_regression_model(
            y_train, y_pred_simple_cv, model_name=f"{model_name_ens} (CV)"
        )
        
        print("\n" + "=" * 70)
        print(f"**🏅 CHAMPION ENSEMBLE (CV Metrics):**")
        print("=" * 70)
        print(f"   -> RMSE (CV): {metrics_all[f'{model_name_ens} (CV)']['RMSE']:.4f} MPa")
        
        # Final Residual Checks for Champion Model (using CV predictions)
        plot_model_performance(y_train, y_pred_simple_cv, model_name=f"{model_name_ens} (CV)")
        plot_residuals_vs_features(
            X_train, y_train, y_pred_simple_cv, 
            model_name=f"{model_name_ens} (CV)", 
            top_features=['Age, days', 'C, kg/m3', 'W, kg/m3']
        )
        
        # --- B) Ensemble Performance on UNSEEN TEST Data ---
        print("\n" + "=" * 70)
        print(f"**🚀 FINAL GENERALIZATION TEST ON UNSEEN {TEST_SIZE_RATIO*100:.0f}% TEST DATA**")
        print("=" * 70)
        
        # Get predictions from the final pipelines (fit on X_train)
        gb_pipe = pipelines_final["GB"] 
        xgb_pipe = pipelines_final["XGBoost"]

        y_pred_gb_test = gb_pipe.predict(X_test)
        y_pred_xgb_test = xgb_pipe.predict(X_test)
        
        # Final Ensemble Prediction on the test set
        y_pred_simple_test = (SIMPLE_ALPHA_GB * y_pred_gb_test) + (SIMPLE_WEIGHT_XGB * y_pred_xgb_test)

        model_name_test = f"{model_name_ens} (Test Set)"
        metrics_all[model_name_test] = evaluate_regression_model(
            y_test, y_pred_simple_test, model_name=model_name_test
        )
        
        print(f"   -> RMSE (TEST SET): {metrics_all[model_name_test]['RMSE']:.4f} MPa")
        print(f"   -> R² (TEST SET): {metrics_all[model_name_test]['R²']:.4f}")
        
    return metrics_all, model_name_ens

# =============================================================================
## 4. FINAL SUMMARY AND DEPLOYMENT TEST
# =============================================================================

# =============================================================================
## 4. FINAL SUMMARY AND DEPLOYMENT TEST
# =============================================================================

def final_summary_and_test(X_test, metrics_final, pipelines_final, best_model_name_ens):
    """Generates final tables, plots (including dedicated Test Set comparisons), and performs the deployment prediction test."""
    
    final_df = pd.DataFrame(metrics_final).T
    
    # --- Final Thesis Summary Table (Including CV and Test Set results) ---
    print("\n" + "=" * 70)
    print("4. FINAL THESIS SUMMARY: Model Comparison (CV vs. Test Set)")
    print("=" * 70)
    
    # Define the comparison rows based on the required structure
    models_to_compare = [
        "Random Forest", "GB", "XGBoost", "Stacking (GB+XGB+MLP)", best_model_name_ens
    ]
    
    comparison_data = []
    
    for model in models_to_compare:
        # Get CV results
        cv_key = f"{model} (CV)"
        if cv_key in final_df.index:
            row_cv = final_df.loc[cv_key, ["RMSE", "R²", "MAE", "PWA_10%"]].to_dict()
            comparison_data.append({
                'Model': model,
                'Evaluation Type': 'CV (Training)',
                **row_cv
            })
        
        # Get Test Set results
        test_key = f"{model} (Test Set)"
        if test_key in final_df.index:
            row_test = final_df.loc[test_key, ["RMSE", "R²", "MAE", "PWA_10%"]].to_dict()
            comparison_data.append({
                'Model': model,
                'Evaluation Type': 'Test Set (Unseen)',
                **row_test
            })

    comparison_df = pd.DataFrame(comparison_data)
    
    # Rename columns for the final table presentation
    comparison_df.rename(columns={
        'R²': 'R^2',
        'PWA_10%': 'PWA_10%' 
    }, inplace=True)

    # Sort the table to put the champion ensemble test set result first
    champion_test_index = comparison_df[
        (comparison_df['Model'] == best_model_name_ens) & 
        (comparison_df['Evaluation Type'] == 'Test Set (Unseen)')
    ].index
    
    if not champion_test_index.empty:
        # Move the champion test row to the top
        comparison_df = pd.concat([
            comparison_df.loc[champion_test_index],
            comparison_df.drop(champion_test_index)
        ]).reset_index(drop=True)
    
    print("\n--- Consolidated Performance Comparison Table ---")
    print(comparison_df.to_markdown(index=False, floatfmt=".4f"))

    # --- Generate Final Comparison Plots ---
    
    # 1. Comparison of ALL metrics (CV and Test) in the original format:
    plot_final_metric_comparison(final_df) 

    # 2. Dedicated Test Set R^2 Figure 
    print("\nGenerating dedicated Test Set R^2 comparison figure...")
    plot_test_set_comparison(final_df, 'R²', 'model_generalization')

    # 3. Dedicated Test Set RMSE Figure 
    print("Generating dedicated Test Set RMSE comparison figure...")
    plot_test_set_comparison(final_df, 'RMSE', 'model_generalization')

    # --- Feature Importance Analysis ---
    print("\n" + "=" * 70)
    print("=== FEATURE IMPORTANCE ANALYSIS (Trained on 80% Data) ===")
    print("=" * 70)
    
    tree_models_to_check = ["GB", "XGBoost", "Random Forest"] 
    for model_name in tree_models_to_check:
        if model_name in pipelines_final:
            try:
                importances = pipelines_final[model_name].named_steps['model'].feature_importances_
                feature_imp_df = pd.DataFrame({'Feature': FEATURE_COLS, 'Importance': importances})
                feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)
                
                print(f"\n--- {model_name} Top 5 Features ---")
                print(feature_imp_df.head(5).to_markdown(index=False, floatfmt=".4f"))
                plot_feature_importance(feature_imp_df, model_name)
            except AttributeError:
                pass 

    # --- Deployment Prediction Example ---
    print("\n" + "=" * 70)
    print(f"=== 🔮 DEPLOYMENT PREDICTION: {best_model_name_ens} ===")
    print("=" * 70)
    
    example_df = pd.DataFrame([EXAMPLE_MIX], columns=FEATURE_COLS)
    
    gb_pipe = pipelines_final["GB"] 
    xgb_pipe = pipelines_final["XGBoost"]

    gb_pred = gb_pipe.predict(example_df)[0]
    xgb_pred = xgb_pipe.predict(example_df)[0]
    
    final_simple_pred = (SIMPLE_ALPHA_GB * gb_pred) + (SIMPLE_WEIGHT_XGB * xgb_pred)

    print(f"Data Input: {EXAMPLE_MIX}")
    print("\n--- Prediction Components ---")
    print(f"GB Prediction: {gb_pred:.2f} MPa (Weight: {SIMPLE_ALPHA_GB:.2f})")
    print(f"XGBoost Prediction: {xgb_pred:.2f} MPa (Weight: {SIMPLE_WEIGHT_XGB:.2f})")
    print(f"\n**FINAL SIMPLE ENSEMBLE PREDICTION:** **{final_simple_pred:.2f} MPa**")
# =============================================================================
## 5. MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore") 
    
    # 1. Load and Split Data
    X_train, X_test, y_train, y_test, preprocess = load_and_prepare_data()
    
    # 2. Train Models (CV on Train data, Final model fit on Train data, Test set evaluation)
    metrics_all, predictions_cv, pipelines_final = train_and_evaluate_cv(X_train, X_test, y_train, y_test, preprocess)
    
    # 3. Ensemble Analysis and Test Set Evaluation
    metrics_final, best_model_name_ens = perform_ensemble_analysis_and_test(
        X_train, X_test, y_train, y_test, metrics_all, predictions_cv, pipelines_final
    )
    
    # 4. Final Outputs
    final_summary_and_test(X_test, metrics_final, pipelines_final, best_model_name_ens)