import torch
import matplotlib.pyplot as plt
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from huggingface_hub import hf_hub_download
import numpy as np
from tqdm import tqdm
import os
import seaborn as sns
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set CUDA device to GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

MODEL = "moirai"  # model name: choose from {'moirai', 'moirai-moe'}
SIZE = "large"  # model size: choose from {'small', 'base', 'large'}
PDT = 8  # prediction length: any positive integer
CTX = 64  # context length: any positive integer
PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
BSZ = 128  # batch size
TEST = int(10*PDT)  # test set length: any positive integer
NUM_SAMPLES = 100  # number of samples for uncertainty estimation

# Create results directory structure
results_dir = f"results/{MODEL}-{SIZE}/ctx{CTX}"
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved to: {results_dir}")

# Read data into pandas DataFrame
csv_path = "/home/sa53869/time-series/moirai/time-moe-eval/ETT-small/ETTm2.csv"
df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

# Focus on HUFL column only
df_hufl = df[['HUFL']].copy()
print(f"Focusing on HUFL column only. Data shape: {df_hufl.shape}")

# Convert into GluonTS dataset
ds = PandasDataset(dict(df_hufl))

# Get dataset information dynamically
num_series = len(list(ds))
print(f"Dataset info:")
print(f"Number of time series: {num_series}")
print(f"Available columns in CSV: {df.columns.tolist()}")
print(f"Target column being forecasted: {list(ds)[0]['target'][:5]}")

# Split into train/test set
train, test_template = split(ds, offset=-TEST)

# Generate test data 
test_data_full = test_template.generate_instances(
    prediction_length=PDT,
    windows=TEST // PDT,
    distance=PDT,
)

# Calculate actual number of windows
actual_windows = len(list(test_data_full.input))
print(f"Windows per series: {TEST // PDT}")
print(f"Expected total windows ({num_series} series × {TEST // PDT}): {num_series * (TEST // PDT)}")
print(f"Actual total windows: {actual_windows}")

# Load the base module once
print("Loading base model module...")
if MODEL == "moirai":
    base_module = MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{SIZE}")
elif MODEL == "moirai-moe":
    base_module = MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{SIZE}")

def create_model_with_context_length(context_length):
    """Create a new model with specific context length"""
    if MODEL == "moirai":
        model = MoiraiForecast(
            module=base_module,
            prediction_length=1,
            context_length=context_length,
            patch_size=PSZ,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    elif MODEL == "moirai-moe":
        model = MoiraiMoEForecast(
            module=base_module,
            prediction_length=1,
            context_length=context_length,
            patch_size=16,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    return model.create_predictor(batch_size=BSZ)

def calculate_forecast_uncertainty(predictor, input_data, prediction_length) -> Dict[str, np.ndarray]:
    """
    Calculate forecast uncertainty using Moirai's built-in sampling capability
    """
    print(f"Generating forecast with {NUM_SAMPLES} samples for uncertainty estimation...")
    
    # Create forecast input
    forecast_input = [input_data]
    
    # Get forecast with samples (Moirai automatically generates multiple samples)
    forecast = next(iter(predictor.predict(forecast_input)))
    
    # Extract samples - shape should be [num_samples, prediction_length]
    forecast_samples = forecast.samples
    print(f"Forecast samples shape: {forecast_samples.shape}")
    
    # Ensure we have the right dimensions
    if forecast_samples.ndim == 3:
        forecast_samples = forecast_samples.squeeze(0)  # Remove batch dimension if present
    
    # Calculate statistics across samples (axis=0 since samples are in first dimension)
    mean_forecast = np.mean(forecast_samples, axis=0)
    std_forecast = np.std(forecast_samples, axis=0)
    median_forecast = np.median(forecast_samples, axis=0)
    
    # Calculate different uncertainty measures
    cv_uncertainty = std_forecast / (np.abs(mean_forecast) + 1e-6)
    q25 = np.percentile(forecast_samples, 25, axis=0)
    q75 = np.percentile(forecast_samples, 75, axis=0)
    iqr_uncertainty = (q75 - q25) / (np.abs(median_forecast) + 1e-6)
    entropy_uncertainty = np.log(std_forecast + 1e-6)
    min_forecast = np.min(forecast_samples, axis=0)
    max_forecast = np.max(forecast_samples, axis=0)
    range_uncertainty = (max_forecast - min_forecast) / (np.abs(mean_forecast) + 1e-6)
    
    print(f"Forecast uncertainty statistics:")
    print(f"  Mean CV uncertainty: {np.mean(cv_uncertainty):.4f}")
    print(f"  Mean IQR uncertainty: {np.mean(iqr_uncertainty):.4f}")
    print(f"  Mean entropy uncertainty: {np.mean(entropy_uncertainty):.4f}")
    
    return {
        'samples': forecast_samples,
        'mean': mean_forecast,
        'std': std_forecast,
        'median': median_forecast,
        'cv_uncertainty': cv_uncertainty,
        'iqr_uncertainty': iqr_uncertainty,
        'entropy_uncertainty': entropy_uncertainty,
        'range_uncertainty': range_uncertainty,
        'q25': q25,
        'q75': q75,
        'min': min_forecast,
        'max': max_forecast
    }

def calculate_autoregressive_input_uncertainty(input_data, window_id) -> Dict[str, Any]:
    """
    Calculate uncertainty autoregressively for input context reconstruction using Moirai
    """
    print(f"Starting autoregressive input uncertainty estimation for window {window_id}...")
    
    full_sequence = input_data["target"]
    
    # Ensure we have exactly CTX samples for analysis
    if len(full_sequence) > CTX:
        analysis_sequence = full_sequence[-CTX:]
        print(f"Window {window_id}: Using last {CTX} samples from {len(full_sequence)} available")
    else:
        analysis_sequence = full_sequence
        print(f"Window {window_id}: Using all {len(analysis_sequence)} samples (less than CTX={CTX})")
    
    sequence_length = len(analysis_sequence)
    
    # Storage for results at each autoregressive step
    ar_uncertainties = []
    ar_predictions = []
    ar_samples_all = []
    ar_errors = []
    ar_cv_uncertainties = []
    ar_iqr_uncertainties = []
    ar_entropy_uncertainties = []
    
    print(f"Window {window_id}: Computing autoregressive uncertainty for {sequence_length} samples (starting from position 1)...")
    
    # Skip position 0 since there's no context to predict from - start from position 1
    for pos in tqdm(range(1, sequence_length), desc=f"Window {window_id} - AR uncertainty"):
        true_value = analysis_sequence[pos]
        
        # Use preceding context to predict current position
        context_data = analysis_sequence[:pos]
        current_context_length = len(context_data)
        
        # Create model with appropriate context length
        predictor = create_model_with_context_length(current_context_length)
        
        # Create input for prediction
        step_input_data = input_data.copy()
        step_input_data["target"] = context_data
        
        try:
            # Get forecast with uncertainty (Moirai generates samples automatically)
            step_forecast = next(iter(predictor.predict([step_input_data])))
            step_samples = step_forecast.samples  # Shape: [num_samples, 1]
            
            # Flatten samples if needed
            if step_samples.ndim > 1:
                step_samples = step_samples.flatten()
            
            # Calculate statistics
            pred_mean = np.mean(step_samples)
            pred_std = np.std(step_samples)
            pred_cv = pred_std / (abs(pred_mean) + 1e-6)
            pred_iqr = np.percentile(step_samples, 75) - np.percentile(step_samples, 25)
            pred_iqr_uncertainty = pred_iqr / (abs(pred_mean) + 1e-6)
            pred_entropy = np.log(pred_std + 1e-6)
            prediction_error = abs(pred_mean - true_value)
            
        except Exception as e:
            print(f"  Error at position {pos}: {e}, using fallback")
            # Fallback: use simple prediction with artificial uncertainty
            pred_mean = context_data[-1] if len(context_data) > 0 else 0.0
            pred_samples = np.random.normal(pred_mean, abs(pred_mean) * 0.1, NUM_SAMPLES)
            pred_std = np.std(pred_samples)
            pred_cv = pred_std / (abs(pred_mean) + 1e-6)
            pred_iqr = np.percentile(pred_samples, 75) - np.percentile(pred_samples, 25)
            pred_iqr_uncertainty = pred_iqr / (abs(pred_mean) + 1e-6)
            pred_entropy = np.log(pred_std + 1e-6)
            prediction_error = abs(pred_mean - true_value)
        
        # Store results
        ar_predictions.append(pred_mean)
        ar_uncertainties.append(prediction_error)  # Use prediction error as importance
        ar_cv_uncertainties.append(pred_cv)
        ar_iqr_uncertainties.append(pred_iqr_uncertainty)
        ar_entropy_uncertainties.append(pred_entropy)
        ar_samples_all.append(pred_samples if 'pred_samples' in locals() else np.array([pred_mean]))
        ar_errors.append(prediction_error)
        
        if pos <= 5 or pos % 10 == 0:  # Print progress (pos starts from 1)
            print(f"  Position {pos}: error={prediction_error:.4f}, CV_unc={pred_cv:.4f}")
    
    print(f"Completed autoregressive uncertainty estimation for window {window_id} - analyzed {sequence_length-1} positions (skipped position 0)")
    
    return {
        'ar_predictions': np.array(ar_predictions),
        'ar_uncertainties': np.array(ar_uncertainties),  # This is prediction errors (importance)
        'ar_cv_uncertainties': np.array(ar_cv_uncertainties),
        'ar_iqr_uncertainties': np.array(ar_iqr_uncertainties),
        'ar_entropy_uncertainties': np.array(ar_entropy_uncertainties),
        'ar_samples': ar_samples_all,
        'ar_errors': np.array(ar_errors),
        'true_context': analysis_sequence,
        'context_length': sequence_length,
        'window_id': window_id
    }

def perform_forecasting_comparison_with_uncertainty(input_data, label_data, importance_scores, window_id):
    """
    Perform forecasting with uncertainty analysis for three different context selection strategies
    """
    # Get the target data and ensure it's exactly CTX length
    target_data = input_data['target']
    
    # Ensure we use the same sequence as importance analysis
    if len(target_data) > CTX:
        context_target = target_data[-CTX:]
    else:
        context_target = target_data
    
    actual_context_length = len(context_target)
    reduced_ctx = max(1, int(0.75 * actual_context_length))  # 75% of actual context
    
    print(f"Window {window_id} Forecasting with Uncertainty:")
    print(f"  Full context length: {actual_context_length}")
    print(f"  Reduced context length (75%): {reduced_ctx}")
    
    # 1. Full context forecasting with uncertainty
    print(f"  1. Full context forecasting with uncertainty...")
    forecast_input_data_full = {
        'target': context_target,
        'start': input_data['start'],
        'item_id': input_data.get('item_id', 0)
    }
    
    # Create model for full PDT prediction
    if MODEL == "moirai":
        model_full = MoiraiForecast(
            module=base_module,
            prediction_length=PDT,
            context_length=actual_context_length,
            patch_size=PSZ,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    elif MODEL == "moirai-moe":
        model_full = MoiraiMoEForecast(
            module=base_module,
            prediction_length=PDT,
            context_length=actual_context_length,
            patch_size=16,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    predictor_full = model_full.create_predictor(batch_size=BSZ)
    
    forecast_uncertainty_full = calculate_forecast_uncertainty(predictor_full, forecast_input_data_full, PDT)
    
    # 2. Random 75% context forecasting with uncertainty
    print(f"  2. Random 75% context forecasting with uncertainty...")
    np.random.seed(42 + window_id)  # Reproducible random selection
    random_indices = np.sort(np.random.choice(actual_context_length, reduced_ctx, replace=False))
    context_target_random = context_target[random_indices]
    
    forecast_input_data_random = {
        'target': context_target_random,
        'start': input_data['start'],
        'item_id': input_data.get('item_id', 0)
    }
    
    if MODEL == "moirai":
        model_random = MoiraiForecast(
            module=base_module,
            prediction_length=PDT,
            context_length=reduced_ctx,
            patch_size=PSZ,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    elif MODEL == "moirai-moe":
        model_random = MoiraiMoEForecast(
            module=base_module,
            prediction_length=PDT,
            context_length=reduced_ctx,
            patch_size=16,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    predictor_random = model_random.create_predictor(batch_size=BSZ)
    
    forecast_uncertainty_random = calculate_forecast_uncertainty(predictor_random, forecast_input_data_random, PDT)
    
    # 3. Most important 75% context forecasting with uncertainty
    print(f"  3. Most important 75% context forecasting with uncertainty...")
    
    # Ensure importance scores match the context length
    if len(importance_scores) == actual_context_length:
        context_importance = importance_scores
    elif len(importance_scores) > actual_context_length:
        context_importance = importance_scores[-actual_context_length:]
    else:
        mean_importance = np.mean(importance_scores) if len(importance_scores) > 0 else 1.0
        context_importance = np.concatenate([
            np.full(actual_context_length - len(importance_scores), mean_importance),
            importance_scores
        ])
    
    # Select the most important samples
    most_important_indices = np.argsort(context_importance)[-reduced_ctx:]
    most_important_indices = np.sort(most_important_indices)
    context_target_most_important = context_target[most_important_indices]
    
    forecast_input_data_most_important = {
        'target': context_target_most_important,
        'start': input_data['start'],
        'item_id': input_data.get('item_id', 0)
    }
    
    if MODEL == "moirai":
        model_most_important = MoiraiForecast(
            module=base_module,
            prediction_length=PDT,
            context_length=reduced_ctx,
            patch_size=PSZ,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    elif MODEL == "moirai-moe":
        model_most_important = MoiraiMoEForecast(
            module=base_module,
            prediction_length=PDT,
            context_length=reduced_ctx,
            patch_size=16,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    predictor_most_important = model_most_important.create_predictor(batch_size=BSZ)
    
    forecast_uncertainty_most_important = calculate_forecast_uncertainty(predictor_most_important, forecast_input_data_most_important, PDT)
    
    # Get true values for comparison
    true_values = label_data["target"][:PDT]
    
    # Calculate forecast metrics for all three methods
    def calculate_metrics(forecast_mean, true_vals):
        forecast_errors = np.abs(forecast_mean - true_vals)
        mae = np.mean(forecast_errors)
        mse = np.mean((forecast_mean - true_vals) ** 2)
        rmse = np.sqrt(mse)
        return mae, rmse, forecast_errors
    
    # Full context
    mae_full, rmse_full, forecast_errors_full = calculate_metrics(forecast_uncertainty_full['mean'], true_values)
    
    # Random context
    mae_random, rmse_random, forecast_errors_random = calculate_metrics(forecast_uncertainty_random['mean'], true_values)
    
    # Most important context
    mae_most_important, rmse_most_important, forecast_errors_most_important = calculate_metrics(forecast_uncertainty_most_important['mean'], true_values)
    
    print(f"  Results:")
    print(f"    Full Context (len={actual_context_length})  - MAE: {mae_full:.4f}, RMSE: {rmse_full:.4f}, CV_unc: {np.mean(forecast_uncertainty_full['cv_uncertainty']):.4f}")
    print(f"    Random 75% (len={reduced_ctx})             - MAE: {mae_random:.4f}, RMSE: {rmse_random:.4f}, CV_unc: {np.mean(forecast_uncertainty_random['cv_uncertainty']):.4f}")
    print(f"    Most Imp 75% (len={reduced_ctx})           - MAE: {mae_most_important:.4f}, RMSE: {rmse_most_important:.4f}, CV_unc: {np.mean(forecast_uncertainty_most_important['cv_uncertainty']):.4f}")
    
    return {
        'window_id': window_id,
        'actual_context_length': actual_context_length,
        'reduced_ctx': reduced_ctx,
        # Full context results
        'mae_full': mae_full,
        'rmse_full': rmse_full,
        'forecast_errors_full': forecast_errors_full,
        'forecast_uncertainty_full': forecast_uncertainty_full,
        # Random context results
        'mae_random': mae_random,
        'rmse_random': rmse_random,
        'forecast_errors_random': forecast_errors_random,
        'forecast_uncertainty_random': forecast_uncertainty_random,
        # Most important context results
        'mae_most_important': mae_most_important,
        'rmse_most_important': rmse_most_important,
        'forecast_errors_most_important': forecast_errors_most_important,
        'forecast_uncertainty_most_important': forecast_uncertainty_most_important,
        # Common data
        'true_values': true_values,
        'random_indices': random_indices,
        'most_important_indices': most_important_indices
    }

def create_uncertainty_comparison_plots(all_importance_results, all_forecast_results, save_path=None):
    """
    Create simplified uncertainty analysis plots focusing on position-wise analysis
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Moirai Input Context Uncertainty Analysis (HUFL Column)\n'
                f'Model: {MODEL}-{SIZE} | Context: {CTX} | Prediction: {PDT} | Samples: {NUM_SAMPLES}', 
                fontsize=16, fontweight='bold')
    
    # Collect all data
    all_ar_uncertainties = np.concatenate([result['ar_cv_uncertainties'] for result in all_importance_results])
    all_ar_errors = np.concatenate([result['ar_errors'] for result in all_importance_results])
    all_importance_scores = np.concatenate([result['ar_uncertainties'] for result in all_importance_results])
    
    # 1. Position-wise uncertainty evolution (individual windows)
    ax1 = axes[0, 0]
    max_length = max(len(result['ar_cv_uncertainties']) for result in all_importance_results)
    
    # Plot individual windows with different colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_importance_results)))
    for i, result in enumerate(all_importance_results):
        # Start positions from 2 since we skip position 0 (first position)
        positions = np.arange(2, len(result['ar_cv_uncertainties']) + 2)
        ax1.plot(positions, result['ar_cv_uncertainties'], 
                alpha=0.7, color=colors[i], linewidth=2, 
                label=f'Window {result["window_id"]}')
    
    ax1.set_title('AR Uncertainty by Context Position\n(Individual Windows)', fontweight='bold')
    ax1.set_xlabel('Context Position')
    ax1.set_ylabel('AR CV Uncertainty')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Position-wise uncertainty evolution (aggregate)
    ax2 = axes[0, 1]
    
    # Calculate average uncertainty by position
    position_uncertainties = np.zeros(max_length)
    position_counts = np.zeros(max_length)
    position_std = np.zeros(max_length)
    
    for result in all_importance_results:
        seq_len = len(result['ar_cv_uncertainties'])
        for pos in range(seq_len):
            position_uncertainties[pos] += result['ar_cv_uncertainties'][pos]
            position_counts[pos] += 1
    
    # Calculate mean and std for each position
    mean_position_uncertainty = np.divide(position_uncertainties, position_counts, 
                                        out=np.zeros_like(position_uncertainties), where=position_counts!=0)
    
    # Calculate std for each position
    for pos in range(max_length):
        if position_counts[pos] > 1:
            values_at_pos = []
            for result in all_importance_results:
                if pos < len(result['ar_cv_uncertainties']):
                    values_at_pos.append(result['ar_cv_uncertainties'][pos])
            position_std[pos] = np.std(values_at_pos) if len(values_at_pos) > 1 else 0
    
    valid_positions = position_counts > 0
    pos_indices = np.arange(2, max_length + 2)[valid_positions]  # Start from position 2
    mean_vals = mean_position_uncertainty[valid_positions]
    std_vals = position_std[valid_positions]
    
    ax2.plot(pos_indices, mean_vals, 'purple', linewidth=3, marker='o', 
            markersize=6, label='Mean AR Uncertainty')
    ax2.fill_between(pos_indices, mean_vals - std_vals, mean_vals + std_vals, 
                    alpha=0.3, color='purple', label='±1 Std Dev')
    
    # Add trend line
    if len(pos_indices) > 2:
        z = np.polyfit(pos_indices, mean_vals, 1)
        p = np.poly1d(z)
        ax2.plot(pos_indices, p(pos_indices), "r--", linewidth=2, 
                label=f'Trend (slope: {z[0]:.4f})')
    
    ax2.set_title('Average AR Uncertainty by Position\n(With Confidence Band)', fontweight='bold')
    ax2.set_xlabel('Context Position')
    ax2.set_ylabel('AR CV Uncertainty')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Position-wise importance/error evolution (aggregate)
    ax3 = axes[0, 2]
    
    # Calculate average importance (error) by position
    position_importance = np.zeros(max_length)
    position_imp_std = np.zeros(max_length)
    
    for result in all_importance_results:
        seq_len = len(result['ar_uncertainties'])
        for pos in range(seq_len):
            position_importance[pos] += result['ar_uncertainties'][pos]
    
    mean_position_importance = np.divide(position_importance, position_counts, 
                                       out=np.zeros_like(position_importance), where=position_counts!=0)
    
    # Calculate std for importance at each position
    for pos in range(max_length):
        if position_counts[pos] > 1:
            values_at_pos = []
            for result in all_importance_results:
                if pos < len(result['ar_uncertainties']):
                    values_at_pos.append(result['ar_uncertainties'][pos])
            position_imp_std[pos] = np.std(values_at_pos) if len(values_at_pos) > 1 else 0
    
    mean_imp_vals = mean_position_importance[valid_positions]
    std_imp_vals = position_imp_std[valid_positions]
    
    ax3.plot(pos_indices, mean_imp_vals, 'orange', linewidth=3, marker='s', 
            markersize=6, label='Mean Sample Importance (Error)')
    ax3.fill_between(pos_indices, mean_imp_vals - std_imp_vals, mean_imp_vals + std_imp_vals, 
                    alpha=0.3, color='orange', label='±1 Std Dev')
    
    # Add trend line
    if len(pos_indices) > 2:
        z_imp = np.polyfit(pos_indices, mean_imp_vals, 1)
        p_imp = np.poly1d(z_imp)
        ax3.plot(pos_indices, p_imp(pos_indices), "r--", linewidth=2, 
                label=f'Trend (slope: {z_imp[0]:.4f})')
    
    ax3.set_title('Average Sample Importance by Position\n(Prediction Error)', fontweight='bold')
    ax3.set_xlabel('Context Position')
    ax3.set_ylabel('Sample Importance (Absolute Error)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Detailed sample-level analysis for selected window
    ax4 = axes[1, 0]
    
    # Select window with most interesting uncertainty pattern (highest variance)
    window_variances = [np.var(result['ar_cv_uncertainties']) for result in all_importance_results]
    selected_window_idx = np.argmax(window_variances)
    selected_result = all_importance_results[selected_window_idx]
    
    # Full context positions
    context_positions = np.arange(1, len(selected_result['true_context']) + 1)
    # AR predictions positions (start from position 2 since we skip position 0)
    ar_positions = np.arange(2, len(selected_result['ar_predictions']) + 2)
    
    # Plot true values, predictions, and uncertainty
    ax4_twin = ax4.twinx()
    
    # True values on left axis (full context)
    ax4.plot(context_positions, selected_result['true_context'], 'b-', 
            linewidth=3, marker='o', markersize=4, label='True Values')
    # AR predictions on left axis (offset by 1 position)
    ax4.plot(ar_positions, selected_result['ar_predictions'], 'g--', 
            linewidth=2, marker='^', markersize=4, label='AR Predictions')
    
    # Uncertainty on right axis (same positions as AR predictions)
    ax4_twin.plot(ar_positions, selected_result['ar_cv_uncertainties'], 'purple', 
                 linewidth=2, marker='s', markersize=5, label='AR CV Uncertainty')
    ax4_twin.plot(ar_positions, selected_result['ar_uncertainties'], 'orange', 
                 linewidth=2, marker='d', markersize=5, label='Sample Importance (Error)')
    
    ax4.set_title(f'Sample-Level Analysis\nWindow {selected_result["window_id"]} (Highest Uncertainty Variance)', 
                 fontweight='bold')
    ax4.set_xlabel('Context Position')
    ax4.set_ylabel('Value', color='blue')
    ax4_twin.set_ylabel('Uncertainty / Importance', color='purple')
    
    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    # 5. Uncertainty distribution by position (heatmap style)
    ax5 = axes[1, 1]
    
    # Create matrix for heatmap
    uncertainty_matrix = np.full((len(all_importance_results), max_length), np.nan)
    for i, result in enumerate(all_importance_results):
        uncertainty_matrix[i, :len(result['ar_cv_uncertainties'])] = result['ar_cv_uncertainties']
    
    im = ax5.imshow(uncertainty_matrix, aspect='auto', cmap='viridis', 
                   interpolation='nearest')
    
    ax5.set_title('AR Uncertainty Heatmap\n(Rows: Windows, Cols: Positions)', fontweight='bold')
    ax5.set_xlabel('Context Position')
    ax5.set_ylabel('Window Index')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax5, shrink=0.8)
    cbar.set_label('AR CV Uncertainty')
    
    # 6. Statistics summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Calculate comprehensive summary statistics
    summary_text = f"""POSITION-WISE UNCERTAINTY STATISTICS

Total Windows: {len(all_importance_results)}
Context Length: {max_length} positions
Total AR Steps: {len(all_ar_uncertainties)}

UNCERTAINTY BY POSITION:
Early Positions (1-5):
• Mean Uncertainty: {np.nanmean(mean_position_uncertainty[:5]):.4f}
• Mean Importance: {np.nanmean(mean_position_importance[:5]):.4f}

Middle Positions ({max_length//2-2}-{max_length//2+2}):
• Mean Uncertainty: {np.nanmean(mean_position_uncertainty[max_length//2-2:max_length//2+3]):.4f}
• Mean Importance: {np.nanmean(mean_position_importance[max_length//2-2:max_length//2+3]):.4f}

Late Positions (last 5):
• Mean Uncertainty: {np.nanmean(mean_position_uncertainty[-5:]):.4f}
• Mean Importance: {np.nanmean(mean_position_importance[-5:]):.4f}

OVERALL TRENDS:
• Uncertainty trend: {"Increasing" if z[0] > 0 else "Decreasing"} (slope: {z[0]:.4f})
• Importance trend: {"Increasing" if z_imp[0] > 0 else "Decreasing"} (slope: {z_imp[0]:.4f})

CORRELATIONS:
• Position vs Uncertainty: {np.corrcoef(pos_indices, mean_vals)[0, 1]:.3f}
• Position vs Importance: {np.corrcoef(pos_indices, mean_imp_vals)[0, 1]:.3f}
• Uncertainty vs Importance: {np.corrcoef(all_ar_uncertainties, all_importance_scores)[0, 1]:.3f}

FORECAST PERFORMANCE:
• Mean MAE (Full): {np.mean([r['mae_full'] for r in all_forecast_results]):.4f}
• Mean MAE (Random 75%): {np.mean([r['mae_random'] for r in all_forecast_results]):.4f}
• Mean MAE (Most Imp 75%): {np.mean([r['mae_most_important'] for r in all_forecast_results]):.4f}

IMPROVEMENT:
• Most Imp vs Random: {((np.mean([r['mae_most_important'] for r in all_forecast_results]) / np.mean([r['mae_random'] for r in all_forecast_results])) - 1) * 100:+.2f}%
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Uncertainty analysis plots saved to: {save_path}")
    
    plt.show()

def mean_absolute_scaled_error(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    """Calculate Mean Absolute Scaled Error (MASE)"""
    naive_mae = np.mean(np.abs(np.diff(y_train)))
    if naive_mae == 0:
        naive_mae = 1e-10
    forecast_mae = np.mean(np.abs(y_true - y_pred))
    mase = forecast_mae / naive_mae
    return mase

def create_position_focused_plots(all_importance_results, save_path=None):
    """
    Create additional plots specifically focused on position-wise uncertainty patterns
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Position-Focused Uncertainty Analysis (HUFL Column)\n'
                f'Model: {MODEL}-{SIZE} | Context Length: {CTX}', 
                fontsize=16, fontweight='bold')
    
    max_length = max(len(result['ar_cv_uncertainties']) for result in all_importance_results)
    
    # 1. Box plots of uncertainty by position
    ax1 = axes[0, 0]
    
    # Collect data for each position
    position_data = []
    positions_with_data = []
    
    for pos in range(max_length):
        values_at_pos = []
        for result in all_importance_results:
            if pos < len(result['ar_cv_uncertainties']):
                values_at_pos.append(result['ar_cv_uncertainties'][pos])
        
        if len(values_at_pos) > 1:  # Only include positions with multiple samples
            position_data.append(values_at_pos)
            positions_with_data.append(pos + 2)  # +2 since we skip position 0
    
    if position_data:
        bp = ax1.boxplot(position_data, positions=positions_with_data, widths=0.6, 
                        patch_artist=True, showfliers=True)
        
        # Color boxes with gradient
        colors = plt.cm.viridis(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax1.set_title('Distribution of AR Uncertainty by Position\n(Box Plots)', fontweight='bold')
    ax1.set_xlabel('Context Position')
    ax1.set_ylabel('AR CV Uncertainty')
    ax1.grid(True, alpha=0.3)
    
    # 2. Line plot showing individual window trajectories with emphasis on patterns
    ax2 = axes[0, 1]
    
    # Calculate trajectory characteristics
    trajectory_stats = []
    for result in all_importance_results:
        uncertainties = result['ar_cv_uncertainties']
        if len(uncertainties) > 2:
            # Calculate trend
            positions = np.arange(len(uncertainties))
            slope = np.polyfit(positions, uncertainties, 1)[0]
            variance = np.var(uncertainties)
            trajectory_stats.append({'slope': slope, 'variance': variance, 'result': result})
    
    # Sort by interesting patterns
    trajectory_stats.sort(key=lambda x: abs(x['slope']), reverse=True)
    
    # Plot most interesting trajectories
    n_trajectories = min(10, len(trajectory_stats))
    colors = plt.cm.tab10(np.linspace(0, 1, n_trajectories))
    
    for i in range(n_trajectories):
        result = trajectory_stats[i]['result']
        slope = trajectory_stats[i]['slope']
        positions = np.arange(2, len(result['ar_cv_uncertainties']) + 2)  # Start from position 2
        
        line_style = '-' if slope > 0 else '--'
        ax2.plot(positions, result['ar_cv_uncertainties'], 
                color=colors[i], linewidth=2, linestyle=line_style,
                label=f'W{result["window_id"]} (slope: {slope:.3f})')
    
    ax2.set_title('Individual Window Uncertainty Trajectories\n(Top 10 by Trend Magnitude)', fontweight='bold')
    ax2.set_xlabel('Context Position')
    ax2.set_ylabel('AR CV Uncertainty')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Correlation matrix between different positions
    ax3 = axes[1, 0]
    
    # Create correlation matrix
    n_positions = min(20, max_length)  # Limit to first 20 positions for readability
    correlation_matrix = np.full((n_positions, n_positions), np.nan)
    
    for i in range(n_positions):
        for j in range(n_positions):
            values_i = []
            values_j = []
            
            for result in all_importance_results:
                if i < len(result['ar_cv_uncertainties']) and j < len(result['ar_cv_uncertainties']):
                    values_i.append(result['ar_cv_uncertainties'][i])
                    values_j.append(result['ar_cv_uncertainties'][j])
            
            if len(values_i) > 2:
                correlation_matrix[i, j] = np.corrcoef(values_i, values_j)[0, 1]
    
    im = ax3.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax3.set_title('Position-to-Position Uncertainty Correlations\n(First 20 Positions)', fontweight='bold')
    ax3.set_xlabel('Context Position')
    ax3.set_ylabel('Context Position')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
    cbar.set_label('Correlation Coefficient')
    
    # Set ticks
    tick_positions = np.arange(0, n_positions, 5)
    ax3.set_xticks(tick_positions)
    ax3.set_yticks(tick_positions)
    ax3.set_xticklabels(tick_positions + 2)  # +2 since we skip position 0
    ax3.set_yticklabels(tick_positions + 2)  # +2 since we skip position 0
    
    # 4. Position importance ranking
    ax4 = axes[1, 1]
    
    # Calculate average importance and uncertainty for each position
    position_avg_importance = np.zeros(max_length)
    position_avg_uncertainty = np.zeros(max_length)
    position_counts = np.zeros(max_length);
    
    for result in all_importance_results:
        for pos in range(len(result['ar_uncertainties'])):
            position_avg_importance[pos] += result['ar_uncertainties'][pos]
            position_avg_uncertainty[pos] += result['ar_cv_uncertainties'][pos]
            position_counts[pos] += 1
    
    valid_positions = position_counts > 0
    position_avg_importance = np.divide(position_avg_importance, position_counts, 
                                       out=np.zeros_like(position_avg_importance), where=valid_positions)
    position_avg_uncertainty = np.divide(position_avg_uncertainty, position_counts, 
                                        out=np.zeros_like(position_avg_uncertainty), where=valid_positions)
    
    # Rank positions by importance
    valid_indices = np.where(valid_positions)[0]
    importance_ranking = np.argsort(position_avg_importance[valid_indices])[::-1]  # Descending order
    
    # Show top 15 most important positions
    n_top = min(15, len(importance_ranking))
    top_positions = valid_indices[importance_ranking[:n_top]] + 2  # +2 for position offset (skip 0)
    top_importance = position_avg_importance[valid_indices[importance_ranking[:n_top]]]
    top_uncertainty = position_avg_uncertainty[valid_indices[importance_ranking[:n_top]]]
    
    # Create combined bar plot
    x = np.arange(n_top)
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, top_importance, width, label='Sample Importance (Error)', 
                   color='orange', alpha=0.7)
    
    # Scale uncertainty to make it visible on same plot
    uncertainty_scaled = top_uncertainty * (max(top_importance) / max(top_uncertainty))
    bars2 = ax4.bar(x + width/2, uncertainty_scaled, width, label='AR Uncertainty (scaled)', 
                   color='purple', alpha=0.7)
    
    ax4.set_title(f'Top {n_top} Most Important Context Positions\n(Ranked by Average Prediction Error)', 
                 fontweight='bold')
    ax4.set_xlabel('Rank')
    ax4.set_ylabel('Average Values')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'Pos {pos}' for pos in top_positions], rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add values on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        ax4.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.001, 
                f'{top_importance[i]:.3f}', ha='center', va='bottom', fontsize=8)
        ax4.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.001, 
                f'{top_uncertainty[i]:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Position-focused plots saved to: {save_path}")
    
    plt.show()

# Create results directory structure
results_base_dir = "results"
model_dir = f"{MODEL}-{SIZE}"
context_dir = f"ctx{CTX}"
results_dir = os.path.join(results_base_dir, model_dir, context_dir)

# Create directories if they don't exist
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved to: {results_dir}")

# Main analysis loop with uncertainty
print("\nAnalyzing sample importance and forecasting with uncertainty across all windows...")

all_importance_results = []
all_forecast_results = []
input_it = iter(test_data_full.input)
label_it = iter(test_data_full.label)

for window_idx, (input_data, label_data) in enumerate(zip(input_it, label_it)):
    if window_idx >= actual_windows:
        break
    
    print(f"\n{'='*80}")
    print(f"Processing Window {window_idx + 1}/{actual_windows}")
    print(f"{'='*80}")
    
    # 1. Autoregressive input uncertainty analysis (replaces sample importance)
    print(f"1. Computing autoregressive input uncertainty...")
    importance_result = calculate_autoregressive_input_uncertainty(input_data, window_idx + 1)
    all_importance_results.append(importance_result)
    
    # 2. Forecasting comparison with uncertainty using different context selection strategies
    print(f"2. Performing forecasting comparison with uncertainty...")
    forecast_result = perform_forecasting_comparison_with_uncertainty(
        input_data, label_data, importance_result['ar_uncertainties'], window_idx + 1
    )
    all_forecast_results.append(forecast_result)
    
    # Print window summary
    print(f"\nWindow {window_idx + 1} Summary:")
    print(f"  Autoregressive Input Analysis:")
    print(f"    Samples analyzed: {importance_result['context_length']}")
    print(f"    Mean AR CV uncertainty: {np.mean(importance_result['ar_cv_uncertainties']):.4f}")
    print(f"    Mean AR error: {np.mean(importance_result['ar_errors']):.4f}")
    print(f"  Forecasting Performance with Uncertainty:")
    print(f"    Full vs Random: {((forecast_result['mae_random'] / forecast_result['mae_full']) - 1) * 100:+.2f}% MAE change")
    print(f"    Full vs Most Imp: {((forecast_result['mae_most_important'] / forecast_result['mae_full']) - 1) * 100:+.2f}% MAE change")
    print(f"    Forecast uncertainty (Full): {np.mean(forecast_result['forecast_uncertainty_full']['cv_uncertainty']):.4f}")

# Create comprehensive uncertainty analysis plots
print(f"\n{'='*80}")
print("CREATING UNCERTAINTY ANALYSIS VISUALIZATIONS")
print(f"{'='*80}")

create_uncertainty_comparison_plots(all_importance_results, all_forecast_results, 
                                  os.path.join(results_dir, 'uncertainty_analysis.png'))

# Create additional position-focused plots
print(f"\n{'='*80}")
print("CREATING POSITION-FOCUSED VISUALIZATIONS")
print(f"{'='*80}")

create_position_focused_plots(all_importance_results, os.path.join(results_dir, 'position_analysis.png'))

# Aggregate analysis across all windows
print(f"\n{'='*80}")
print("AGGREGATE UNCERTAINTY ANALYSIS")
print(f"{'='*80}")

# Combine all importance scores and uncertainty data
all_importance_scores = np.concatenate([result['ar_uncertainties'] for result in all_importance_results])
all_ar_cv_uncertainties = np.concatenate([result['ar_cv_uncertainties'] for result in all_importance_results])
all_ar_iqr_uncertainties = np.concatenate([result['ar_iqr_uncertainties'] for result in all_importance_results])
all_ar_entropy_uncertainties = np.concatenate([result['ar_entropy_uncertainties'] for result in all_importance_results])
all_ar_errors = np.concatenate([result['ar_errors'] for result in all_importance_results])

# Combine all forecasting scores with uncertainty
all_forecast_mae_full = np.array([result['mae_full'] for result in all_forecast_results])
all_forecast_rmse_full = np.array([result['rmse_full'] for result in all_forecast_results])
all_forecast_errors_full = np.concatenate([result['forecast_errors_full'] for result in all_forecast_results])

all_forecast_mae_random = np.array([result['mae_random'] for result in all_forecast_results])
all_forecast_rmse_random = np.array([result['rmse_random'] for result in all_forecast_results])
all_forecast_errors_random = np.concatenate([result['forecast_errors_random'] for result in all_forecast_results])

all_forecast_mae_most_important = np.array([result['mae_most_important'] for result in all_forecast_results])
all_forecast_rmse_most_important = np.array([result['rmse_most_important'] for result in all_forecast_results])
all_forecast_errors_most_important = np.concatenate([result['forecast_errors_most_important'] for result in all_forecast_results])

# Extract forecast uncertainty data
all_forecast_cv_full = np.concatenate([result['forecast_uncertainty_full']['cv_uncertainty'] for result in all_forecast_results])
all_forecast_cv_random = np.concatenate([result['forecast_uncertainty_random']['cv_uncertainty'] for result in all_forecast_results])
all_forecast_cv_most_important = np.concatenate([result['forecast_uncertainty_most_important']['cv_uncertainty'] for result in all_forecast_results])

all_forecast_iqr_full = np.concatenate([result['forecast_uncertainty_full']['iqr_uncertainty'] for result in all_forecast_results])
all_forecast_entropy_full = np.concatenate([result['forecast_uncertainty_full']['entropy_uncertainty'] for result in all_forecast_results])

print(f"Total samples analyzed (AR uncertainty): {len(all_ar_cv_uncertainties)}")
print(f"Total windows processed: {len(all_importance_results)}")
print(f"Total forecast points per method: {len(all_forecast_errors_full)}")

# Overall statistics with uncertainty
print(f"\nOverall Forecasting Statistics with Uncertainty:")
print(f"Full Context:")
print(f"  Mean MAE: {np.mean(all_forecast_mae_full):.4f}")
print(f"  Mean RMSE: {np.mean(all_forecast_rmse_full):.4f}")
print(f"  Mean CV Uncertainty: {np.mean(all_forecast_cv_full):.4f}")
print(f"  Mean IQR Uncertainty: {np.mean(all_forecast_iqr_full):.4f}")

print(f"Random 75% Context:")
print(f"  Mean MAE: {np.mean(all_forecast_mae_random):.4f}")
print(f"  Mean RMSE: {np.mean(all_forecast_rmse_random):.4f}")
print(f"  Mean CV Uncertainty: {np.mean(all_forecast_cv_random):.4f}")

print(f"Most Important 75% Context:")
print(f"  Mean MAE: {np.mean(all_forecast_mae_most_important):.4f}")
print(f"  Mean RMSE: {np.mean(all_forecast_rmse_most_important):.4f}")
print(f"  Mean CV Uncertainty: {np.mean(all_forecast_cv_most_important):.4f}")

# Autoregressive uncertainty statistics
print(f"\nAutoregressive Input Uncertainty Statistics:")
print(f"  Mean AR CV Uncertainty: {np.mean(all_ar_cv_uncertainties):.4f}")
print(f"  Std AR CV Uncertainty: {np.std(all_ar_cv_uncertainties):.4f}")
print(f"  Mean AR IQR Uncertainty: {np.mean(all_ar_iqr_uncertainties):.4f}")
print(f"  Mean AR Entropy Uncertainty: {np.mean(all_ar_entropy_uncertainties):.4f}")
print(f"  Mean AR Error: {np.mean(all_ar_errors):.4f}")

# Correlation analysis
print(f"\nCorrelation Analysis:")
ar_unc_error_corr = np.corrcoef(all_ar_cv_uncertainties, all_ar_errors)[0, 1]
importance_ar_unc_corr = np.corrcoef(all_importance_scores, all_ar_cv_uncertainties)[0, 1]
forecast_unc_perf_corr = np.corrcoef(all_forecast_cv_full, all_forecast_errors_full)[0, 1]

print(f"  AR Uncertainty vs AR Error: {ar_unc_error_corr:.3f}")
print(f"  Sample Importance vs AR Uncertainty: {importance_ar_unc_corr:.3f}")
print(f"  Forecast Uncertainty vs Forecast Error: {forecast_unc_perf_corr:.3f}")

# Performance comparison with uncertainty
print(f"\nPerformance Comparison with Uncertainty:")
print(f"Random vs Full: {((np.mean(all_forecast_mae_random) / np.mean(all_forecast_mae_full)) - 1) * 100:+.2f}% MAE change")
print(f"Most Imp vs Full: {((np.mean(all_forecast_mae_most_important) / np.mean(all_forecast_mae_full)) - 1) * 100:+.2f}% MAE change")
print(f"Most Imp vs Random: {((np.mean(all_forecast_mae_most_important) / np.mean(all_forecast_mae_random)) - 1) * 100:+.2f}% MAE change")

print(f"\nUncertainty Comparison (Full vs Random vs Most Important):")
print(f"CV Uncertainty change (Random vs Full): {((np.mean(all_forecast_cv_random) / np.mean(all_forecast_cv_full)) - 1) * 100:+.2f}%")
print(f"CV Uncertainty change (Most Imp vs Full): {((np.mean(all_forecast_cv_most_important) / np.mean(all_forecast_cv_full)) - 1) * 100:+.2f}%")

# Save results with uncertainty
print(f"\nSaving results with uncertainty...")
results_summary = {
    # Original data
    'all_importance_scores': all_importance_scores,
    'all_forecast_mae_full': all_forecast_mae_full,
    'all_forecast_rmse_full': all_forecast_rmse_full,
    'all_forecast_mae_random': all_forecast_mae_random,
    'all_forecast_rmse_random': all_forecast_rmse_random,
    'all_forecast_mae_most_important': all_forecast_mae_most_important,
    'all_forecast_rmse_most_important': all_forecast_rmse_most_important,
    'all_forecast_errors_full': all_forecast_errors_full,
    'all_forecast_errors_random': all_forecast_errors_random,
    'all_forecast_errors_most_important': all_forecast_errors_most_important,
    
    # Uncertainty data
    'all_ar_cv_uncertainties': all_ar_cv_uncertainties,
    'all_ar_iqr_uncertainties': all_ar_iqr_uncertainties,
    'all_ar_entropy_uncertainties': all_ar_entropy_uncertainties,
    'all_ar_errors': all_ar_errors,
    'all_forecast_cv_full': all_forecast_cv_full,
    'all_forecast_cv_random': all_forecast_cv_random,
    'all_forecast_cv_most_important': all_forecast_cv_most_important,
    'all_forecast_iqr_full': all_forecast_iqr_full,
    'all_forecast_entropy_full': all_forecast_entropy_full,
    
    # Metadata
    'per_window_importance_results': all_importance_results,
    'per_window_forecast_results': all_forecast_results,
    'total_windows': len(all_importance_results),
    'total_samples': len(all_importance_scores),
    
    # Correlation metrics
    'ar_uncertainty_error_correlation': ar_unc_error_corr,
    'importance_ar_uncertainty_correlation': importance_ar_unc_corr,
    'forecast_uncertainty_performance_correlation': forecast_unc_perf_corr,
}

np.savez(os.path.join(results_dir, 'uncertainty_analysis_results.npz'), **results_summary)
print(f"Results saved to '{os.path.join(results_dir, 'uncertainty_analysis_results.npz')}')")

# Create position-focused plots
print(f"\n{'='*80}")
print("CREATING POSITION-FOCUSED PLOTS")
print(f"{'='*80}")

create_position_focused_plots(all_importance_results, os.path.join(results_dir, 'position_focused_analysis.png'))

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE!")
print(f"{'='*80}")
print(f"All results saved to: {results_dir}")
print(f"  • uncertainty_analysis.png - Comprehensive uncertainty plots")
print(f"  • position_analysis.png - Position-focused analysis")
print(f"  • position_focused_analysis.png - Detailed position analysis")
print(f"  • uncertainty_analysis_results.npz - Numerical results data")
print(f"\nProcessing Summary:")
print(f"  • Processed {len(all_importance_results)} windows")
print(f"  • Analyzed {len(all_ar_cv_uncertainties)} AR steps for uncertainty")
print(f"  • Mean MAE (Full): {np.mean(all_forecast_mae_full):.4f}")
print(f"  • Mean MAE (Random 75%): {np.mean(all_forecast_mae_random):.4f}")
print(f"  • Mean MAE (Most Important 75%): {np.mean(all_forecast_mae_most_important):.4f}")
print(f"  • Mean AR CV uncertainty: {np.mean(all_ar_cv_uncertainties):.4f}")
print(f"  • Mean forecast CV uncertainty: {np.mean(all_forecast_cv_full):.4f}")