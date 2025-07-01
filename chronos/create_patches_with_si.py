import argparse
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from chronos import BaseChronosPipeline
import random
import os
from tqdm import tqdm


def autoregressive_predict_context_simplified(pipeline, ground_truth_context, dataset_mean):
    """Predict context autoregressively and compute both Gaussian and quantile-based surprisal."""
    predictions = []
    gaussian_surprisals = []
    quantile_surprisals = []
    context_length = len(ground_truth_context)
    
    for i in tqdm(range(context_length), desc="Predicting samples"):
        if i == 0:
            # Use dataset mean for first prediction context
            context = torch.tensor([dataset_mean], dtype=torch.float32)
        else:
            # Use ground truth values from 0 to i-1 as context
            context = torch.tensor(ground_truth_context[:i], dtype=torch.float32)
        
        # Get samples from predict() for Gaussian surprisal computation
        preds = pipeline.predict(
            context=context,
            prediction_length=1,  # Predict one step ahead
            num_samples=1000
        )

        # Get quantile predictions for quantile-based surprisal
        quantiles, mean = pipeline.predict_quantiles(
            context=context,
            prediction_length=1,
            quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        )

        pred = mean[0, 0].item()
        predictions.append(pred)
        
        # Method 1: Gaussian surprisal using prediction samples
        # Extract samples and compute variance
        samples = preds[0, :, 0].cpu().numpy()  # Shape: [num_samples]
        pred_mean = np.mean(samples)
        pred_var = np.var(samples)
        
        # True observed value for this timestep
        true_value = ground_truth_context[i]
        
        # Compute Gaussian surprisal: -log2(p(x | μ, σ²))
        # For Gaussian: s(x) = -log₂(1/(σ√(2π)) * exp(-(x-μ)²/(2σ²)))
        # Simplified: s(x) = log₂(σ√(2π)) + (x-μ)²/(2σ²ln(2))
        if pred_var > 0:
            pred_std = np.sqrt(pred_var)
            gaussian_surprisal = np.log2(pred_std * np.sqrt(2 * np.pi)) + \
                               ((true_value - pred_mean) ** 2) / (2 * pred_var * np.log(2))
        else:
            # Handle case where variance is 0 (very rare)
            gaussian_surprisal = 0.0
        
        gaussian_surprisals.append(gaussian_surprisal)
        
        # Method 2: Quantile-based surprisal (original method)
        # Higher spread = higher uncertainty = higher surprisal
        q_values = quantiles[0, 0, :].cpu().numpy()
        spread = np.std(q_values)
        quantile_surprisal = np.log2(1 + spread)  # Heuristic mapping
        quantile_surprisals.append(quantile_surprisal)
    
    return np.array(predictions), np.array(gaussian_surprisals), np.array(quantile_surprisals)


def detect_patches_from_surprisal(surprisals, threshold_multiplier=1.5, min_patch_size=5, 
                                 monotonic_mode=False):
    """
    Detect patch boundaries based on surprisal spikes.
    
    Args:
        surprisals: Array of surprisal values
        threshold_multiplier: Multiplier for mean surprisal to determine spike threshold
        min_patch_size: Minimum size of a patch
        monotonic_mode: If True, use Approximate Monotonic Constraint (diff-based)
    
    Returns:
        patch_boundaries: List of patch boundary indices
        threshold: The threshold value used for spike detection
    """
    if monotonic_mode:
        # Approximate Monotonic Constraint: use differences between consecutive surprisal values
        surprisal_diffs = np.diff(surprisals)
        # Only consider positive differences (increases in surprisal)
        positive_diffs = surprisal_diffs[surprisal_diffs > 0]
        
        if len(positive_diffs) > 0:
            mean_diff = np.mean(positive_diffs)
            std_diff = np.std(positive_diffs)
            threshold = mean_diff + threshold_multiplier * std_diff
            
            # Find significant increases in surprisal
            spike_indices = []
            for i, diff in enumerate(surprisal_diffs):
                if diff > threshold:
                    spike_indices.append(i + 1)  # +1 because diff[i] is between point i and i+1
        else:
            threshold = 0
            spike_indices = []
        
        spikes = np.array(spike_indices)
    else:
        # Original method: absolute surprisal values
        mean_surprisal = np.mean(surprisals)
        std_surprisal = np.std(surprisals)
        threshold = mean_surprisal + threshold_multiplier * std_surprisal
        
        # Find spikes above threshold
        spikes = np.where(surprisals > threshold)[0]
    
    # Create patch boundaries
    patch_boundaries = [0]  # Start with first position
    
    current_patch_start = 0
    for spike_idx in spikes:
        # Only create boundary if we have minimum patch size
        if spike_idx - current_patch_start >= min_patch_size:
            patch_boundaries.append(spike_idx)
            current_patch_start = spike_idx
    
    # Add final boundary
    if len(surprisals) - 1 not in patch_boundaries:
        patch_boundaries.append(len(surprisals) - 1)
    
    return patch_boundaries, threshold


def cap_outlier_surprisals(surprisals, method='iqr', factor=1.5):
    """
    Cap outlier surprisal values to prevent a few extreme values from skewing the analysis.
    
    Args:
        surprisals: Array of surprisal values
        method: Method for outlier detection ('iqr', 'percentile', 'zscore')
        factor: Factor for outlier detection (1.5 for IQR, 3 for z-score, 95-99 for percentile)
    
    Returns:
        Capped surprisal values
    """
    surprisals = np.array(surprisals)
    
    if method == 'iqr':
        # Interquartile Range method
        Q1 = np.percentile(surprisals, 25)
        Q3 = np.percentile(surprisals, 75)
        IQR = Q3 - Q1
        upper_bound = Q3 + factor * IQR
        capped_surprisals = np.minimum(surprisals, upper_bound)
        
    elif method == 'percentile':
        # Percentile-based capping (factor should be between 90-99)
        upper_bound = np.percentile(surprisals, factor)
        capped_surprisals = np.minimum(surprisals, upper_bound)
        
    elif method == 'zscore':
        # Z-score method
        mean_val = np.mean(surprisals)
        std_val = np.std(surprisals)
        upper_bound = mean_val + factor * std_val
        capped_surprisals = np.minimum(surprisals, upper_bound)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    num_capped = np.sum(surprisals > upper_bound)
    if num_capped > 0:
        print(f"    Capped {num_capped} outlier values above {upper_bound:.3f} using {method} method")
    
    return capped_surprisals


def main():
    parser = argparse.ArgumentParser(description='Autoregressive prediction with Gaussian and quantile-based surprisal patches for 3 random chunks.')
    parser.add_argument('--csv_path', type=str, 
                        default='../datasets/time-moe-eval/ETT-small/ETTm2.csv',
                        help='Path to the CSV file containing the time series data')
    parser.add_argument('--col_num', type=int, default=1, 
                        help='Column number (0-indexed) to process')
    parser.add_argument('--context_length', type=int, default=256, 
                        help='Length of context to predict autoregressively')
    parser.add_argument('--model', type=str, default='amazon/chronos-t5-small',
                        help='Chronos model to use for prediction')
    parser.add_argument('--threshold_multiplier', type=float, default=1.5,
                        help='Multiplier for mean+std to determine surprisal spike threshold')
    parser.add_argument('--min_patch_size', type=int, default=1,
                        help='Minimum size of a patch')
    parser.add_argument('--monotonic_mode', action='store_true',
                        help='Use Approximate Monotonic Constraint (threshold on surprisal differences)')
    parser.add_argument('--outlier_method', type=str, default='iqr', 
                        choices=['iqr', 'percentile', 'zscore'],
                        help='Method for capping outlier surprisal values (default: iqr)')
    parser.add_argument('--outlier_factor', type=float, default=1.5,
                        help='Factor for outlier detection (1.5 for IQR, 3 for z-score, 95-99 for percentile)')
    args = parser.parse_args()

    # Load data
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    if args.col_num >= len(df.columns):
        raise ValueError(f"Column number {args.col_num} is out of range for the CSV file")
    series = df.iloc[:, args.col_num].values

    # Calculate dataset mean
    dataset_mean = np.mean(series)
    print(f"Dataset mean: {dataset_mean:.4f}")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {args.model} on {device}...")
    pipeline = BaseChronosPipeline.from_pretrained(
        args.model,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )

    # Select 3 random chunks of context_length
    max_start = len(series) - args.context_length
    if max_start < 3:
        raise ValueError(f"Time series too short for 3 chunks of length {args.context_length}")
    random_starts = random.sample(range(max_start), 3)

    # Generate autoregressive predictions for each chunk
    all_ground_truths = []
    all_predictions = []
    all_gaussian_surprisals = []
    all_quantile_surprisals = []
    all_gaussian_patch_boundaries = []
    all_quantile_patch_boundaries = []
    all_gaussian_thresholds = []
    all_quantile_thresholds = []

    for i, start in enumerate(random_starts):
        print(f"\nProcessing chunk {i+1}/3 (start index {start})...")
        ground_truth = series[start:start+args.context_length]
        preds, gaussian_surprisals, quantile_surprisals = autoregressive_predict_context_simplified(
            pipeline, ground_truth, dataset_mean)
        
        # # Cap outlier surprisal values to prevent extreme values from skewing analysis
        # print(f"  Original Gaussian surprisal range: [{np.min(gaussian_surprisals):.3f}, {np.max(gaussian_surprisals):.3f}]")
        # gaussian_surprisals = cap_outlier_surprisals(gaussian_surprisals, method=args.outlier_method, factor=args.outlier_factor)
        # print(f"  Capped Gaussian surprisal range: [{np.min(gaussian_surprisals):.3f}, {np.max(gaussian_surprisals):.3f}]")
        
        # print(f"  Original Quantile surprisal range: [{np.min(quantile_surprisals):.3f}, {np.max(quantile_surprisals):.3f}]")
        # quantile_surprisals = cap_outlier_surprisals(quantile_surprisals, method=args.outlier_method, factor=args.outlier_factor)
        # print(f"  Capped Quantile surprisal range: [{np.min(quantile_surprisals):.3f}, {np.max(quantile_surprisals):.3f}]")
        
        # Detect patches based on Gaussian surprisal spikes
        gaussian_patch_boundaries, gaussian_threshold = detect_patches_from_surprisal(
            gaussian_surprisals, args.threshold_multiplier, args.min_patch_size, args.monotonic_mode
        )
        
        # Detect patches based on quantile surprisal spikes
        quantile_patch_boundaries, quantile_threshold = detect_patches_from_surprisal(
            quantile_surprisals, args.threshold_multiplier, args.min_patch_size, args.monotonic_mode
        )
        
        all_ground_truths.append(ground_truth)
        all_predictions.append(preds)
        all_gaussian_surprisals.append(gaussian_surprisals)
        all_quantile_surprisals.append(quantile_surprisals)
        all_gaussian_patch_boundaries.append(gaussian_patch_boundaries)
        all_quantile_patch_boundaries.append(quantile_patch_boundaries)
        all_gaussian_thresholds.append(gaussian_threshold)
        all_quantile_thresholds.append(quantile_threshold)
        
        print(f"  Gaussian method: {len(gaussian_patch_boundaries)-1} patches with threshold {gaussian_threshold:.4f}")
        print(f"  Quantile method: {len(quantile_patch_boundaries)-1} patches with threshold {quantile_threshold:.4f}")
        print(f"  Mode: {'Monotonic (Surprisal diff)' if args.monotonic_mode else 'Absolute Surprisal'}")
        print(f"  Gaussian patch boundaries: {gaussian_patch_boundaries}")
        print(f"  Quantile patch boundaries: {quantile_patch_boundaries}")

    # Create results directory structure
    dataset_name = os.path.splitext(os.path.basename(args.csv_path))[0]
    results_dir = f"results_si/{dataset_name}/col_{args.col_num}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create separate plots for each chunk with two subplots
    for i in range(3):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Subplot 1: Gaussian Surprisal
        ax1_twin = ax1.twinx()
        
        # Plot ground truth and predictions
        ax1.plot(all_ground_truths[i], label="Ground Truth", linestyle='-', 
                linewidth=2, alpha=0.8, color='blue')
        ax1.plot(all_predictions[i], label="Autoregressive Predictions", linestyle='--', 
                linewidth=2, alpha=0.8, color='green')
        
        # Plot Gaussian surprisal
        ax1_twin.plot(all_gaussian_surprisals[i], label="Gaussian Surprisal", 
                     color='red', linewidth=1.5, alpha=0.7)
        
        # Plot threshold
        if args.monotonic_mode:
            ax1_twin.axhline(y=all_gaussian_thresholds[i], color='orange', linestyle=':', 
                           linewidth=2, alpha=0.8, label=f'Gaussian Diff Threshold ({all_gaussian_thresholds[i]:.3f})')
        else:
            ax1_twin.axhline(y=all_gaussian_thresholds[i], color='orange', linestyle=':', 
                           linewidth=2, alpha=0.8, label=f'Gaussian Threshold ({all_gaussian_thresholds[i]:.3f})')
        
        # Plot patch boundaries
        for boundary in all_gaussian_patch_boundaries[i]:
            ax1.axvline(x=boundary, color='purple', linestyle=':', 
                       linewidth=2, alpha=0.6)
        
        # Add patch boundary legend entry
        ax1.axvline(x=-1, color='purple', linestyle=':', linewidth=2, 
                   alpha=0.6, label='Patch Boundaries')
        
        # Set labels and title for subplot 1
        gaussian_patch_count = len(all_gaussian_patch_boundaries[i]) - 1
        mode_str = "Monotonic Gaussian-Diff" if args.monotonic_mode else "Absolute Gaussian"
        ax1.set_title(f"Chunk {i+1} - Gaussian Surprisal Method ({mode_str}) - {gaussian_patch_count} patches")
        ax1.set_ylabel("Value", color='black')
        ax1_twin.set_ylabel("Gaussian Surprisal (bits)", color='red')
        ax1_twin.tick_params(axis='y', labelcolor='red')
        
        # Add legends for subplot 1
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Quantile Surprisal
        ax2_twin = ax2.twinx()
        
        # Plot ground truth and predictions
        ax2.plot(all_ground_truths[i], label="Ground Truth", linestyle='-', 
                linewidth=2, alpha=0.8, color='blue')
        ax2.plot(all_predictions[i], label="Autoregressive Predictions", linestyle='--', 
                linewidth=2, alpha=0.8, color='green')
        
        # Plot quantile surprisal
        ax2_twin.plot(all_quantile_surprisals[i], label="Quantile Surprisal", 
                     color='red', linewidth=1.5, alpha=0.7)
        
        # Plot threshold
        if args.monotonic_mode:
            ax2_twin.axhline(y=all_quantile_thresholds[i], color='orange', linestyle=':', 
                           linewidth=2, alpha=0.8, label=f'Quantile Diff Threshold ({all_quantile_thresholds[i]:.3f})')
        else:
            ax2_twin.axhline(y=all_quantile_thresholds[i], color='orange', linestyle=':', 
                           linewidth=2, alpha=0.8, label=f'Quantile Threshold ({all_quantile_thresholds[i]:.3f})')
        
        # Plot patch boundaries
        for boundary in all_quantile_patch_boundaries[i]:
            ax2.axvline(x=boundary, color='purple', linestyle=':', 
                       linewidth=2, alpha=0.6)
        
        # Add patch boundary legend entry
        ax2.axvline(x=-1, color='purple', linestyle=':', linewidth=2, 
                   alpha=0.6, label='Patch Boundaries')
        
        # Set labels and title for subplot 2
        quantile_patch_count = len(all_quantile_patch_boundaries[i]) - 1
        mode_str = "Monotonic Quantile-Diff" if args.monotonic_mode else "Absolute Quantile"
        ax2.set_title(f"Chunk {i+1} - Quantile Surprisal Method ({mode_str}) - {quantile_patch_count} patches")
        ax2.set_xlabel("Sample Index")
        ax2.set_ylabel("Value", color='black')
        ax2_twin.set_ylabel("Quantile Surprisal (bits)", color='red')
        ax2_twin.tick_params(axis='y', labelcolor='red')
        
        # Add legends for subplot 2
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        mode_suffix = "_monotonic" if args.monotonic_mode else "_absolute"
        plot_path = os.path.join(results_dir, f"chunk_{i+1}_surprisal_comparison{mode_suffix}.pdf")
        plt.savefig(plot_path, format='pdf', bbox_inches='tight')
        print(f"Plot for chunk {i+1} saved to: {plot_path}")
        plt.show()

    # Calculate and print metrics
    print(f"\nPrediction Metrics ({'Monotonic Mode' if args.monotonic_mode else 'Absolute Mode'}):")
    for i in range(3):
        mae = np.mean(np.abs(all_ground_truths[i] - all_predictions[i]))
        mse = np.mean((all_ground_truths[i] - all_predictions[i]) ** 2)
        rmse = np.sqrt(mse)
        
        # Gaussian method metrics
        avg_gaussian_surprisal = np.mean(all_gaussian_surprisals[i])
        gaussian_patch_count = len(all_gaussian_patch_boundaries[i]) - 1
        avg_gaussian_patch_size = args.context_length / gaussian_patch_count if gaussian_patch_count > 0 else args.context_length
        
        # Quantile method metrics
        avg_quantile_surprisal = np.mean(all_quantile_surprisals[i])
        quantile_patch_count = len(all_quantile_patch_boundaries[i]) - 1
        avg_quantile_patch_size = args.context_length / quantile_patch_count if quantile_patch_count > 0 else args.context_length
        
        print(f"Chunk {i+1}: MAE={mae:.4f}, RMSE={rmse:.4f}")
        print(f"         Gaussian: Avg Surprisal={avg_gaussian_surprisal:.4f} bits, Patches={gaussian_patch_count}, Avg Patch Size={avg_gaussian_patch_size:.1f}, Threshold={all_gaussian_thresholds[i]:.4f}")
        print(f"         Quantile: Avg Surprisal={avg_quantile_surprisal:.4f} bits, Patches={quantile_patch_count}, Avg Patch Size={avg_quantile_patch_size:.1f}, Threshold={all_quantile_thresholds[i]:.4f}")


if __name__ == '__main__':
    main()
