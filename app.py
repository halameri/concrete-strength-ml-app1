import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
from tqdm import tqdm # Kept for context, but not strictly required for Streamlit runtime

# =============================================================================
# A. CONFIGURATION (Based on 13 Feature Model)
# =============================================================================

# Define the exact 13 features and their order as used during model training.
# This list is CRITICAL for the ColumnTransformer in the pipeline.
MODEL_FEATURE_COLS = [
    'Age, days',     # Must be first for consistency 
    'C, kg/m3', 
    'M, kg/m3', 
    'CC, kg/m3', 
    'LS, kg/m3', 
    'g, kg/m3', 
    'CB, kg/m3', 
    'G, kg/m3', 
    'SF, kg/m3', 
    'FA, kg/m3', # Fine Aggregate
    'CA, kg/m3', # Coarse Aggregate
    'W, kg/m3',  # Water
    'SP, kg/m3', # Superplasticizer
]

# Define components for the physical constraint (Sum of Cementitious Materials)
CEMENTITIOUS_COMPONENTS = [
    'C, kg/m3', 'M, kg/m3', 'CC, kg/m3', 'LS, kg/m3', 
    'g, kg/m3', 'CB, kg/m3', 'G, kg/m3', 'SF, kg/m3', 
    'FA, kg/m3' # Assumed part of the total powder check
]

try:
    # Attempt to import config (recommended for production)
    from src.config import (
        DATA_FILE_PATH, TARGET_COL, SIMPLE_ALPHA_GB, SIMPLE_WEIGHT_XGB, 
    )
    # Validate imported FEATURE_COLS against the known model list if needed, 
    # but for a quick fix, we hardcode the known correct list above.
    
    # Placeholder for constants not in the imported config
    DEFAULT_MAX_CEMENTITIOUS_SUM = 1000 
except ImportError:
    st.warning("Warning: Could not import configuration from src.config. Using hardcoded constants.")
    DATA_FILE_PATH = "path/to/your/data.xlsx"
    TARGET_COL = "Compressive Strength, MPa"
    SIMPLE_ALPHA_GB = 0.50
    SIMPLE_WEIGHT_XGB = 0.50
    DEFAULT_MAX_CEMENTITIOUS_SUM = 1000

# Feature columns used throughout the app are now MODEL_FEATURE_COLS
FEATURE_COLS = MODEL_FEATURE_COLS

# =============================================================================
# B. MODEL LOADING & CORE FUNCTIONS
# =============================================================================

@st.cache_resource
def load_models():
    """Loads the trained ensemble pipelines."""
    try:
        # NOTE: Ensure these files exist in the same directory or accessible path
        gb_pipe = joblib.load("final_pipeline_gb.joblib")
        xgb_pipe = joblib.load("final_pipeline_xgboost.joblib")
        return gb_pipe, xgb_pipe
    except FileNotFoundError:
        st.error("Model files not found! Please ensure 'final_pipeline_gb.joblib' and 'final_pipeline_xgboost.joblib' are correctly placed.")
        return None, None

def get_ensemble_prediction(data_df: pd.DataFrame, gb_pipe, xgb_pipe):
    """Calculates the simple ensemble prediction."""
    
    # 1. Ensure input is a DataFrame
    if not isinstance(data_df, pd.DataFrame) or data_df.empty:
        raise ValueError("Input data must be a non-empty Pandas DataFrame.")
        
    # --- CRITICAL FIX: ENFORCE FEATURE ORDER and presence ---
    # This step ensures the ColumnTransformer receives the correct 13 columns 
    # in the correct order, resolving the input mismatch error.
    try:
        data_df = data_df[FEATURE_COLS]
    except KeyError as e:
        # This occurs if the input data lacks one of the 13 required columns
        st.exception(f"Fatal Error: Input data is missing required feature(s): {e}. Expected 13 features.")
        raise
    # --- END CRITICAL FIX ---
        
    gb_pred = gb_pipe.predict(data_df)
    xgb_pred = xgb_pipe.predict(data_df)
    
    # Simple weighted average
    final_pred = (SIMPLE_ALPHA_GB * gb_pred) + (SIMPLE_WEIGHT_XGB * xgb_pred)
    return final_pred

@st.cache_data
def load_data_bounds():
    """Loads the data and calculates min/max bounds for the inverse search."""
    try:
        df = pd.read_excel(DATA_FILE_PATH, sheet_name=0)
        
        required_cols = FEATURE_COLS + [TARGET_COL]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
             raise ValueError(f"Data file is missing columns: {missing_cols}")
             
        df = df.dropna(subset=required_cols)
        
        bounds = {}
        for col in FEATURE_COLS:
            min_val = df[col].min()
            max_val = df[col].max()
            bounds[col] = (float(min_val), float(max_val))
        return bounds
    except Exception as e:
        st.error(f"Error loading data for bounds calculation: {e}. Check DATA_FILE_PATH and column names.")
        return {}


# =============================================================================
# C. WEB APP LAYOUT AND LOGIC
# =============================================================================

def app():
    st.set_page_config(
        page_title="Concrete Strength ML Predictor",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🧪 Concrete Strength ML Predictor & Optimizer")
    st.markdown("---")

    gb_pipe, xgb_pipe = load_models()
    if gb_pipe is None or xgb_pipe is None:
        return

    st.sidebar.title("App Mode")
    mode = st.sidebar.radio(
        "Select Operation:",
        ("Predict Strength (Forward)", "Optimize Mix (Inverse)"),
        key="app_mode"
    )
    st.sidebar.markdown("---")
    st.sidebar.info(f"Using Ensemble Model: **GB ({SIMPLE_ALPHA_GB}) + XGBoost ({SIMPLE_WEIGHT_XGB})**")

    if mode == "Predict Strength (Forward)":
        forward_prediction_ui(gb_pipe, xgb_pipe)
    else:
        inverse_optimization_ui(gb_pipe, xgb_pipe)

# --- 1. Forward Prediction UI ---
def forward_prediction_ui(gb_pipe, xgb_pipe, initial_mix_data=None):
    
    is_validation = initial_mix_data is not None
    
    if not is_validation:
        st.header("1️⃣ Predict Compressive Strength from a Mix Design")
        st.markdown("**Note:** This mode allows any input, including values outside the training range (extrapolation risk).")
        st.caption("Prediction Pipeline: Input $\\rightarrow$ Preprocessing/Imputation $\\rightarrow$ Model $\\rightarrow$ Output")
        
    input_data = {}
    max_val_float = 1500.0 
    
    # Use 3 columns for layout
    cols = st.columns(3)
    
    for i, feature in enumerate(FEATURE_COLS):
        col_index = i % 3
        
        initial_val = None
        if initial_mix_data and feature in initial_mix_data:
            initial_val = initial_mix_data[feature]

        with cols[col_index]:
            
            # --- Age Input Handling ---
            if 'Age, days' in feature:
                step_size = 1
                default_v = 28
                
                input_value = default_v
                if initial_val is not None:
                    input_value = int(round(float(initial_val))) 

                val = st.number_input(
                    label=feature, 
                    min_value=1, 
                    max_value=730, 
                    value=input_value, 
                    step=step_size,
                    format="%d",
                    key=f'input_{feature}_{is_validation}'
                )
            # --- Material Input Handling ---
            else:
                step_size = 1.0
                # Set dynamic defaults based on component type
                if 'C,' in feature: default_v = 350.0
                elif 'W,' in feature: default_v = 180.0
                elif 'CA,' in feature: default_v = 1000.0
                elif 'FA,' in feature: default_v = 700.0
                elif 'SP,' in feature: default_v = 5.0
                elif feature in CEMENTITIOUS_COMPONENTS: default_v = 50.0 
                else: default_v = 10.0 # Catch-all for other additives
                
                input_value = default_v
                if initial_val is not None:
                    input_value = float(initial_val)
                
                val = st.number_input(
                    label=feature, 
                    min_value=0.0,
                    max_value=max_val_float, 
                    value=float(input_value),
                    step=step_size,
                    format="%.3f",
                    key=f'input_{feature}_{is_validation}'
                )
            
            input_data[feature] = val
    
    if not is_validation:
        if st.button("Calculate Strength ➡️", type="primary"):
            try:
                # Use a Series and convert to DF to maintain correct feature order
                input_series = pd.Series(input_data, index=FEATURE_COLS)
                input_df = input_series.to_frame().T
                
                prediction = get_ensemble_prediction(input_df, gb_pipe, xgb_pipe)[0]
                
                st.subheader("Results:")
                st.metric(label="Predicted Compressive Strength", value=f"{prediction:.2f} MPa")
                st.success("Prediction calculated successfully.")
                return prediction
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.error("Please verify model files and input consistency.")
                return None
    
    return input_data

# --- 2. Inverse Optimization UI ---
def inverse_optimization_ui(gb_pipe, xgb_pipe):
    st.header("2️⃣ Optimize Mix Design for Target Strength")
    st.markdown("Finds the optimal mix design to meet a **Target Strength**, constrained by your custom requirements.")
    
    bounds = load_data_bounds()
    if not bounds:
        st.warning("Cannot run optimization. Failed to load data bounds.")
        return

    # --- Target and Iteration Setup ---
    col_target, col_iters = st.columns(2)
    with col_target:
        target_strength = st.number_input(
            "Target Compressive Strength (MPa):",
            min_value=1.0, max_value=120.0, value=60.0, step=1.0, key="target_strength"
        )
    with col_iters:
        n_iterations = st.slider(
            "Search Iterations:",
            min_value=10000, max_value=100000, value=50000, step=10000, key="n_iters"
        )
        
    st.subheader("Optimization Constraints (Customizable)")
    
    # --- Constraint Selection ---
    adjustable_features = [f for f in FEATURE_COLS if 'Age, days' not in f]

    constrained_features = st.multiselect(
        "Select parameters to manually constrain (e.g., Min Cement or Max Water):",
        options=adjustable_features,
        default=CEMENTITIOUS_COMPONENTS + ['W, kg/m3']
    )

    # --- Age and Cement Sum Constraints (Fixed Position) ---
    st.markdown("#### Fixed Constraints:")
    col_age, col_cement = st.columns(2)
    
    # 1. Age Constraint
    min_age_data, max_age_data = 1, 365
    if 'Age, days' in bounds:
        min_age_data = int(np.floor(bounds['Age, days'][0]))
        max_age_data = int(np.ceil(bounds['Age, days'][1]))
        
    with col_age:
        st.markdown("**Age Range (days):** (Must be within data bounds)")
        col_min_age, col_max_age = st.columns(2)
        with col_min_age:
            min_age = st.number_input("Minimum Age", min_value=1, max_value=max_age_data, value=min_age_data, step=1, key="min_age")
        with col_max_age:
            max_age = st.number_input("Maximum Age", min_value=min_age, max_value=max_age_data, value=max_age_data, step=1, key="max_age")

    # 2. Cement Sum Constraint
    with col_cement:
        max_cement_sum = st.number_input(
            "Maximum Total Cementitious Sum ($\text{kg/m}^3$):",
            min_value=100.0, max_value=1500.0, value=float(DEFAULT_MAX_CEMENTITIOUS_SUM), step=50.0, key="max_cement_sum", format="%.1f"
        )
        st.caption(f"Limits the sum of cementitious components: {', '.join(CEMENTITIOUS_COMPONENTS)}")

    # --- Dynamic Constraints (Based on Multiselect) ---
    st.markdown("#### Parameter Value Constraints (Min/Max):")
    user_constraints = {}
    
    if constrained_features:
        cols_constraints = st.columns(2)
        
        for i, feature in enumerate(constrained_features):
            data_min, data_max = bounds.get(feature, (0.0, 1500.0))
            data_min, data_max = float(data_min), float(data_max)
            
            with cols_constraints[i % 2]:
                st.markdown(f"**{feature}:** (Data range: {data_min:.1f} - {data_max:.1f})")
                
                col_min, col_max = st.columns(2)
                with col_min:
                    min_val = st.number_input(f"Min {feature.split(',')[0]}", 
                                              min_value=data_min, max_value=data_max, 
                                              value=data_min, step=0.1, format="%.2f", key=f"min_{feature}")
                with col_max:
                    max_val = st.number_input(f"Max {feature.split(',')[0]}", 
                                              min_value=data_min, max_value=data_max, 
                                              value=data_max, step=0.1, format="%.2f", key=f"max_{feature}")
                
                user_constraints[feature] = (min_val, max_val)
    else:
        st.info("No individual parameters selected for value constraint. Optimization will use only global data min/max.")


    if st.button("Start Optimization & Validation 🚀", type="primary"):
        
        # --- Prepare constraints for optimizer ---
        effective_bounds = {k: list(v) for k, v in bounds.items()}
        effective_bounds['Age, days'] = [float(min_age), float(max_age)]
        
        for feature, (user_min, user_max) in user_constraints.items():
            effective_bounds[feature][0] = max(effective_bounds[feature][0], user_min)
            effective_bounds[feature][1] = min(effective_bounds[feature][1], user_max)
        
        effective_bounds = {k: tuple(v) for k, v in effective_bounds.items()}
        
        with st.spinner("Running constrained iterative optimization..."):
            best_mix_array, best_prediction, min_error = run_optimization(
                gb_pipe, xgb_pipe, target_strength, n_iterations, effective_bounds, max_cement_sum
            )

        st.subheader("Optimization Results & Validation:")
        
        if best_mix_array is not None:
            best_mix_dict = dict(zip(FEATURE_COLS, best_mix_array))

            st.success("✅ **Step 1: Optimal Mix Found by Inverse Search**")
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric("Target Strength", f"{target_strength:.2f} MPa")
                st.metric("Predicted Strength from Inverse Search", f"{best_prediction:.2f} MPa")
            with col_res2:
                st.metric("Minimum Error Achieved", f"{min_error:.4f} MPa")
                
            results_series = pd.Series(best_mix_dict, index=FEATURE_COLS)
            results_df = results_series.to_frame(name='Optimal Quantity (kg/m³ or days)').T
            st.dataframe(results_df.round(3), use_container_width=True)
            
            st.markdown("---")
            st.info("🔄 **Step 2: Validation Check**")

            validation_df = pd.DataFrame([best_mix_dict], columns=FEATURE_COLS)
            final_validation_pred = get_ensemble_prediction(validation_df, gb_pipe, xgb_pipe)[0]

            col_val1, col_val2 = st.columns(2)
            with col_val1:
                st.metric("Forward Validation Prediction", f"{final_validation_pred:.2f} MPa")
            with col_val2:
                validation_error = abs(target_strength - final_validation_pred)
                st.metric("Validation Error (vs. Target)", f"{validation_error:.4f} MPa")
                
            st.caption("The Forward Validation Prediction confirms the output of the optimal mix found.")
        else:
            st.warning("Optimization failed to find a valid mix within the data constraints. Try adjusting the constraints or increasing iterations.")


# --- 3. Optimization Logic (Uses effective_bounds and max_cement_sum) ---
def run_optimization(gb_pipe, xgb_pipe, target_strength, n_iterations, effective_bounds, max_cement_sum):
    """Performs the constrained random search using the effective_bounds and max_cement_sum."""
    best_mix = None
    best_prediction = None
    min_error = float('inf')
    
    feature_names = FEATURE_COLS
    progress_bar = st.progress(0, text="Optimizing... 0%")
    
    # --- FIX: Ensure CEMENTITIOUS_COMPONENTS are correctly indexed ---
    # This uses the correct global list CEMENTITIOUS_COMPONENTS
    try:
        component_indices = [feature_names.index(comp) for comp in CEMENTITIOUS_COMPONENTS if comp in feature_names]
        if not component_indices:
            st.error("Error: None of the defined CEMENTITIOUS_COMPONENTS are in FEATURE_COLS.")
            return None, None, float('inf')
    except ValueError as e:
        st.error(f"Error mapping cement components: {e}")
        return None, None, float('inf')


    for i in range(n_iterations):
        mix_values = []
        
        # 1. Generate mix based on EFFECTIVE BOUNDS
        for feature in feature_names:
            min_val, max_val = effective_bounds[feature]
            
            if 'Age, days' in feature:
                val = random.randint(int(min_val), int(max_val))
            else:
                val = random.uniform(min_val, max_val)
            mix_values.append(val)
        
        current_mix = np.array(mix_values)
        
        # 2. Check Physical Constraint: Cementitious Sum 
        current_c_sum = current_mix[component_indices].sum()
        if current_c_sum > max_cement_sum:
            continue 

        # Create DataFrame for prediction
        mix_df = pd.DataFrame([current_mix], columns=feature_names)
        
        # 3. Prediction
        prediction = get_ensemble_prediction(mix_df, gb_pipe, xgb_pipe)[0]
        error = abs(target_strength - prediction)
        
        # 4. Update best mix
        if error < min_error:
            min_error = error
            best_mix = current_mix
            best_prediction = prediction

        if i % (n_iterations // 100) == 0:
            progress_bar.progress(i / n_iterations, text=f"Optimizing... {i}/{n_iterations}")

    progress_bar.progress(1.0, text="Optimization Complete!")
    return best_mix, best_prediction, min_error


if __name__ == "__main__":
    app()