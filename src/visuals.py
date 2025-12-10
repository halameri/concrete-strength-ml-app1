# =============================================================================
# visuals.py: Thesis-Quality Visualization Functions (Simplified)
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_performance(y_true, y_pred, model_name=""):
    """Generates a comprehensive 2x2 plot for residual analysis and parity check."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # 1. Parity Plot (Actual vs Predicted)
    axes[0].scatter(y_true, y_pred, alpha=0.6, color='#1f77b4')
    max_val = max(y_true.max(), y_pred.max()); min_val = min(y_true.min(), y_pred.min())
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Fit")
    axes[0].set_xlim(min_val, max_val); axes[0].set_ylim(min_val, max_val) 
    axes[0].set_xlabel("Actual Compressive Strength (MPa)"); axes[0].set_ylabel("Predicted Compressive Strength (MPa)")
    axes[0].set_title("A) Actual vs Predicted (Parity Plot)")
    axes[0].grid(True, alpha=0.5, linestyle=':')

    # 2. Residuals vs Predicted
    axes[1].scatter(y_pred, residuals, alpha=0.6, color='green')
    axes[1].axhline(y=0, color='r', linestyle='-', linewidth=1.5)
    axes[1].set_xlabel("Predicted Compressive Strength (MPa)"); axes[1].set_ylabel("Residuals (MPa)")
    axes[1].set_title("B) Residuals vs Predicted (Random Error Check)")
    axes[1].grid(True, alpha=0.5, linestyle=':')

    # 3. Residual Distribution
    sns.histplot(residuals, bins=25, kde=True, ax=axes[2], color='orange')
    axes[2].axvline(x=0, color='r', linestyle='--', linewidth=1.5)
    axes[2].set_xlabel("Residual (MPa)"); axes[2].set_ylabel("Count")
    axes[2].set_title("C) Residual Distribution (Normality Check)")
    axes[2].grid(True, alpha=0.5, linestyle=':')

    # 4. Residuals vs Actual
    axes[3].scatter(y_true, residuals, alpha=0.6, color='purple')
    axes[3].axhline(y=0, color='r', linestyle='-', linewidth=1.5)
    axes[3].set_xlabel("Actual Compressive Strength (MPa)"); axes[3].set_ylabel("Residual (MPa)")
    axes[3].set_title("D) Residuals vs Actual")
    axes[3].grid(True, alpha=0.5, linestyle=':')

    fig.suptitle(f"Model Performance and Residual Analysis – {model_name} (5-Fold CV)", fontsize=16, fontweight='heavy')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_residuals_vs_features(X_data, y_true, y_pred, model_name="", top_features=None):
    """
    Generates plots of residuals against key input features to check for homoscedasticity.
    """
    residuals = y_true - y_pred
    
    # NOTE: top_features must align with the corrected list in config.py, 
    # e.g., ['Age, days', 'C, kg/m3', 'W, kg/m3']
    if top_features is None:
        top_features = ['Age, days', 'C, kg/m3', 'W, kg/m3']
        
    num_plots = len(top_features)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]
    elif num_plots > 1:
        axes = axes.flatten()

    print(f"\n--- Generating Residuals vs Features plots for {model_name} ---")
    
    for i, feature in enumerate(top_features):
        sns.scatterplot(x=X_data[feature], y=residuals, ax=axes[i], alpha=0.6, color='darkred')
        axes[i].axhline(y=0, color='blue', linestyle='--', linewidth=1)
        axes[i].set_xlabel(f"{feature} (Input)")
        axes[i].set_ylabel("Residual (MPa)")
        axes[i].set_title(f"Residuals vs {feature}")
        axes[i].grid(True, alpha=0.4)

    fig.suptitle(f"Homoscedasticity Check: Residuals vs Key Features – {model_name}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_feature_importance(feature_imp_df, model_name):
    """Plots the top 10 feature importances for a given tree model."""
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_imp_df.head(10), palette='viridis')
    plt.title(f'{model_name} Top 10 Feature Importance')
    plt.tight_layout()
    plt.show()

def plot_final_metric_comparison(final_df):
    """Generates RMSE and PWA comparison plots."""
    
    # --- RMSE Comparison ---
    plt.figure(figsize=(12, 6))
    final_df["RMSE"].sort_values().plot(kind="bar", color=sns.color_palette("magma", n_colors=len(final_df)))
    plt.ylabel("RMSE (MPa)")
    plt.title("Comparison of Final Model Performance (RMSE)")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.show() # 

    # --- PWA Comparison ---
    pwa_df = final_df[["PWA_5%", "PWA_10%", "PWA_15%"]].sort_values("PWA_15%", ascending=False)
    
    plt.figure(figsize=(12, 7))
    pwa_df.T.plot(kind='bar', ax=plt.gca())
    
    plt.ylabel("Prediction Within Accuracy (PWA)")
    plt.title("Comparison of Model Accuracy Across PWA Thresholds")
    
    # Define ticks for the x-axis to align with the labels
    plt.xticks(
        ticks=[0, 1, 2],
        rotation=0, 
        ha='center', 
        labels=['PWA $\\leq 5\%$', 'PWA $\\leq 10\%$', 'PWA $\\leq 15\%$']
    )
    plt.grid(axis='y', linestyle='--')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show() # 

# --- ADD THIS NEW FUNCTION TO src/visuals.py ---

# =============================================================================
# visuals.py: CORRECTED plot_test_set_comparison Function
# =============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np # Ensure numpy is imported

# ... (Previous functions: plot_model_performance, plot_residuals_vs_features, plot_feature_importance, plot_final_metric_comparison) ...
# ... (Keep all your existing functions above this line) ...

def plot_test_set_comparison(metrics_df, metric_name, filename_prefix):
    """
    Generates a bar plot comparing ONLY the Test Set performance for a specific metric (R2 or RMSE).
    This function filters out all Cross-Validation (CV) metrics.
    
    Args:
        metrics_df (pd.DataFrame): DataFrame containing metrics for all models (CV and Test).
        metric_name (str): The metric to plot ('R²' or 'RMSE').
        filename_prefix (str): Prefix for the saved plot file.
    """
    
    # 1. Filter DataFrame to include only 'Test Set' results
    test_df = metrics_df[metrics_df.index.str.contains('Test Set')].copy()
    
    # 2. Extract clean model names by removing the suffix and sort
    test_df['Model'] = test_df.index.str.replace(' (Test Set)', '', regex=False)
    
    # 3. Prepare data for plotting (R² descending, RMSE ascending)
    plot_data = test_df.sort_values(by=metric_name, ascending=(metric_name != 'R²')).reset_index(drop=True)
    
    plt.figure(figsize=(10, 6))
    
    # Define colors and plot
    sns.barplot(
        x='Model', 
        y=metric_name, 
        data=plot_data, 
        palette="viridis" if metric_name == 'R²' else "coolwarm"
    )
    
    # 4. Add metric values on top of bars using direct bar index (i)
    # The plot_data DataFrame has been reset and aligns directly with the bar positions
    for i, row in plot_data.iterrows():
        plt.text(
            i,  # Use the index 'i' for the X-position (the bar's center)
            row[metric_name] + (0.005 if metric_name == 'R²' else 0.2), # Adjust text position (Y-position)
            f'{row[metric_name]:.4f}', 
            color='black', 
            ha="center",
            fontsize=10
        )

    plt.title(f'Test Set Generalization: Comparison of {metric_name} Across Models', fontsize=16)
    plt.ylabel(f'{metric_name} (Performance)', fontsize=14)
    plt.xlabel('Model', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 5. Save the figure
    plt.savefig(f'{filename_prefix}_test_set_{metric_name.replace("²", "2")}_comparison.png')
    plt.close()