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
    """Predict context autoregressively and compute entropy from token probabilities."""
    predictions = []
    entropies = []
    context_length = len(ground_truth_context)
    
    for i in tqdm(range(context_length), desc="Predicting samples"):
        if i == 0:
            # Use dataset mean for first prediction context
            context = torch.tensor([dataset_mean], dtype=torch.float32)
        else:
            # Use ground truth values from 0 to i-1 as context
            context = torch.tensor(ground_truth_context[:i], dtype=torch.float32)
        
        # Get predictions with logits for entropy computation
        preds, logits = pipeline.predict(
            context=context,
            prediction_length=1,  # Predict one step ahead
            num_samples=1,
            return_logits=True
        )

        # Extract logits for the predicted token
        token_logits = logits[0, 0, 0, :]  # Shape: [vocab_size]
        probs = torch.softmax(token_logits, dim=-1)
        
        # Compute entropy: H(X) = -sum(p(x) * log2(p(x)))
        entropy = -torch.sum(probs * torch.log2(probs + 1e-10)).item()
        entropies.append(entropy)
        
        # Get the predicted token for this timestep
        predicted_token = preds[0, 0, 0].item()
        predictions.append(predicted_token)
    
    return np.array(predictions), np.array(entropies)


def detect_patches_from_entropy(entropies, threshold_multiplier=1.5, min_patch_size=5, 
                               monotonic_mode=False, fixed_threshold=None):
    """
    Detect patch boundaries based on entropy spikes.
    
    Args:
        entropies: Array of entropy values
        threshold_multiplier: Multiplier for mean entropy to determine spike threshold
        min_patch_size: Minimum size of a patch
        monotonic_mode: If True, use Approximate Monotonic Constraint (diff-based)
        fixed_threshold: If not None, use this fixed threshold instead of calculating one
    
    Returns:
        patch_boundaries: List of patch boundary indices
        threshold: The threshold value used for spike detection
    """
    if monotonic_mode:
        # Approximate Monotonic Constraint: use differences between consecutive entropy values
        entropy_diffs = np.diff(entropies)
        
        if fixed_threshold is not None:
            # Use the provided fixed threshold for entropy differences
            threshold = fixed_threshold
        else:
            # Calculate threshold from positive differences (increases in entropy)
            positive_diffs = entropy_diffs[entropy_diffs > 0]
            
            if len(positive_diffs) > 0:
                mean_diff = np.mean(positive_diffs)
                std_diff = np.std(positive_diffs)
                threshold = mean_diff + threshold_multiplier * std_diff
            else:
                threshold = 0
        
        # Find significant increases in entropy (new patch boundaries)
        spike_indices = []
        for i, diff in enumerate(entropy_diffs):
            if diff > threshold:
                spike_indices.append(i + 1)  # +1 because diff[i] is between point i and i+1
        
        spikes = np.array(spike_indices)
    else:
        # Original method: absolute entropy values
        if fixed_threshold is not None:
            # Use the provided fixed threshold for absolute entropy values
            threshold = fixed_threshold
        else:
            # Calculate threshold from absolute entropy values
            mean_entropy = np.mean(entropies)
            std_entropy = np.std(entropies)
            threshold = mean_entropy + threshold_multiplier * std_entropy
        
        # Find spikes above threshold
        spikes = np.where(entropies > threshold)[0]
    
    # Create patch boundaries
    patch_boundaries = [0]  # Start with first position
    
    current_patch_start = 0
    for spike_idx in spikes:
        # Only create boundary if we have minimum patch size
        if spike_idx - current_patch_start >= min_patch_size:
            patch_boundaries.append(spike_idx)
            current_patch_start = spike_idx
    
    # Add final boundary
    if len(entropies) - 1 not in patch_boundaries:
        patch_boundaries.append(len(entropies) - 1)
    
    return patch_boundaries, threshold


def find_optimal_threshold(entropies, target_patch_size, min_patch_size=5, monotonic_mode=False, max_iterations=10):
    """
    Find the optimal threshold that produces patches with the target average size.
    
    Args:
        entropies: Array of entropy values for the entire dataset
        target_patch_size: Desired average patch size
        min_patch_size: Minimum size of a patch
        monotonic_mode: If True, use Approximate Monotonic Constraint (diff-based)
        max_iterations: Maximum number of iterations to find optimal threshold
    
    Returns:
        optimal_threshold: The threshold that produces patches closest to target size
        final_avg_patch_size: The actual average patch size achieved
        iterations_used: Number of iterations used
    """
    print(f"\nFinding optimal threshold for target patch size: {target_patch_size}")
    
    # Initial threshold estimate using standard method
    if monotonic_mode:
        entropy_diffs = np.diff(entropies)
        positive_diffs = entropy_diffs[entropy_diffs > 0]
        if len(positive_diffs) > 0:
            initial_threshold = np.mean(positive_diffs) + 1.5 * np.std(positive_diffs)
        else:
            initial_threshold = 0
    else:
        initial_threshold = np.mean(entropies) + 1.5 * np.std(entropies)
    
    # Binary search bounds
    threshold_low = 0.0
    threshold_high = initial_threshold * 3  # Start with a wide range
    current_threshold = initial_threshold
    
    best_threshold = current_threshold
    best_diff = float('inf')
    
    for iteration in range(max_iterations):
        # Test current threshold
        patch_boundaries, _ = detect_patches_from_entropy(
            entropies, threshold_multiplier=None, min_patch_size=min_patch_size, 
            monotonic_mode=monotonic_mode, fixed_threshold=current_threshold
        )
        
        num_patches = len(patch_boundaries) - 1
        if num_patches > 0:
            current_avg_patch_size = len(entropies) / num_patches
        else:
            current_avg_patch_size = len(entropies)
        
        diff_from_target = abs(current_avg_patch_size - target_patch_size)
        
        print(f"  Iteration {iteration + 1}: threshold={current_threshold:.4f}, "
              f"avg_patch_size={current_avg_patch_size:.1f}, diff={diff_from_target:.1f}")
        
        # Check if we found a better threshold
        if diff_from_target < best_diff:
            best_threshold = current_threshold
            best_diff = diff_from_target
        
        # Check if we're within tolerance
        if diff_from_target <= 1.0:
            print(f"  Found optimal threshold: {current_threshold:.4f} "
                  f"(avg patch size: {current_avg_patch_size:.1f})")
            return current_threshold, current_avg_patch_size, iteration + 1
        
        # Adjust threshold using binary search
        if current_avg_patch_size > target_patch_size:
            # Patches too large, need higher threshold (more boundaries)
            threshold_low = current_threshold
            current_threshold = (current_threshold + threshold_high) / 2
        else:
            # Patches too small, need lower threshold (fewer boundaries)
            threshold_high = current_threshold
            current_threshold = (threshold_low + current_threshold) / 2
        
        # Prevent infinite loops with very small ranges
        if abs(threshold_high - threshold_low) < 1e-6:
            break
    
    # Return best threshold found
    patch_boundaries, _ = detect_patches_from_entropy(
        entropies, threshold_multiplier=None, min_patch_size=min_patch_size,
        monotonic_mode=monotonic_mode, fixed_threshold=best_threshold
    )
    num_patches = len(patch_boundaries) - 1
    final_avg_patch_size = len(entropies) / num_patches if num_patches > 0 else len(entropies)
    
    print(f"  Best threshold found: {best_threshold:.4f} "
          f"(avg patch size: {final_avg_patch_size:.1f}, diff: {best_diff:.1f})")
    
    return best_threshold, final_avg_patch_size, max_iterations


def main():
    parser = argparse.ArgumentParser(description='Autoregressive prediction with entropy-based patches for 3 random chunks.')
    parser.add_argument('--csv_path', type=str, 
                        default='../datasets/time-moe-eval/ETT-small/ETTm2.csv',
                        help='Path to the CSV file containing the time series data')
    parser.add_argument('--col_num', type=int, default=1, 
                        help='Column number (0-indexed) to process')
    parser.add_argument('--context_length', type=int, default=256, 
                        help='Length of context to predict autoregressively')
    parser.add_argument('--model', type=str, default='../model_weights/chronos/chronos-t5-small',
                        help='Chronos model to use for prediction')
    parser.add_argument('--threshold_multiplier', type=float, default=1.5,
                        help='Multiplier for mean+std to determine spike threshold (ignored if target_patch_size is set)')
    parser.add_argument('--target_patch_size', type=int, default=8,
                        help='Target average patch size. If set, will automatically find optimal threshold.')
    parser.add_argument('--threshold_search_k', type=int, default=None,
                        help='Number of samples to use for threshold search. If not set, uses entire dataset.')
    parser.add_argument('--min_patch_size', type=int, default=1,
                        help='Minimum size of a patch')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU ID to use. If not specified, uses CPU. Example: --gpu 0')
    args = parser.parse_args()

    # Load data
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    if args.col_num >= len(df.columns):
        raise ValueError(f"Column number {args.col_num} is out of range for the CSV file")
    series = df.iloc[:, args.col_num].values

    # Validate threshold_search_k argument
    if args.threshold_search_k is not None:
        if args.threshold_search_k <= 0:
            raise ValueError("threshold_search_k must be positive")
        if args.threshold_search_k > len(series):
            raise ValueError(f"threshold_search_k ({args.threshold_search_k}) cannot be larger than dataset size ({len(series)})")
        if args.target_patch_size is None:
            print("Warning: threshold_search_k specified but target_patch_size not set. threshold_search_k will be ignored.")

    # Calculate dataset mean
    dataset_mean = np.mean(series)
    print(f"Dataset mean: {dataset_mean:.4f}")

    # Load model
    if args.gpu is not None:
        if torch.cuda.is_available():
            if args.gpu < torch.cuda.device_count():
                device = f"cuda:{args.gpu}"
            else:
                print(f"Warning: GPU {args.gpu} not available. Available GPUs: 0-{torch.cuda.device_count()-1}. Using CPU instead.")
                device = "cpu"
        else:
            print("Warning: CUDA not available. Using CPU instead.")
            device = "cpu"
    else:
        device = "cpu"
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

    # If target patch size is specified, compute entropy for dataset to find optimal thresholds
    optimal_thresholds = {'absolute': None, 'monotonic': None}
    dataset_avg_patch_sizes = {'absolute': None, 'monotonic': None}
    if args.target_patch_size is not None:
        # Determine which part of the dataset to use for threshold search
        if args.threshold_search_k is not None:
            search_series = series[:args.threshold_search_k]
            search_description = f"first {args.threshold_search_k} samples"
        else:
            search_series = series
            search_description = "entire dataset"
            
        print(f"\nComputing entropy for {search_description} to find optimal thresholds...")
        print(f"Target patch size: {args.target_patch_size}")
        
        # Compute entropy for the search series
        _, search_entropies = autoregressive_predict_context_simplified(
            pipeline, search_series, dataset_mean)
        
        # Find optimal thresholds for both modes
        print("\n--- Finding optimal threshold for Absolute mode ---")
        optimal_thresholds['absolute'], dataset_avg_patch_sizes['absolute'], abs_iterations = find_optimal_threshold(
            search_entropies, args.target_patch_size, args.min_patch_size, 
            monotonic_mode=False, max_iterations=1000
        )
        
        print("\n--- Finding optimal threshold for Monotonic mode ---")
        optimal_thresholds['monotonic'], dataset_avg_patch_sizes['monotonic'], mon_iterations = find_optimal_threshold(
            search_entropies, args.target_patch_size, args.min_patch_size, 
            monotonic_mode=True, max_iterations=1000
        )
        
        print(f"\nOptimal thresholds found:")
        print(f"  Absolute mode: {optimal_thresholds['absolute']:.4f} (dataset avg size: {dataset_avg_patch_sizes['absolute']:.1f})")
        print(f"  Monotonic mode: {optimal_thresholds['monotonic']:.4f} (dataset avg size: {dataset_avg_patch_sizes['monotonic']:.1f})")
        print(f"Threshold search performed on: {search_description}")

    # Generate autoregressive predictions for each chunk
    all_ground_truths = []
    all_predictions = []
    all_entropies = []
    all_patch_boundaries = {'absolute': [], 'monotonic': []}
    all_thresholds = {'absolute': [], 'monotonic': []}

    for i, start in enumerate(random_starts):
        print(f"\nProcessing chunk {i+1}/3 (start index {start})...")
        ground_truth = series[start:start+args.context_length]
        preds, entropies = autoregressive_predict_context_simplified(
            pipeline, ground_truth, dataset_mean)
        
        # Process both modes
        modes = {'absolute': False, 'monotonic': True}
        
        for mode_name, is_monotonic in modes.items():
            print(f"\n  --- {mode_name.capitalize()} Mode ---")
            
            # Detect patches based on entropy spikes
            if optimal_thresholds[mode_name] is not None:
                # Use the optimal threshold found from full dataset analysis
                patch_boundaries, threshold = detect_patches_from_entropy(
                    entropies, threshold_multiplier=None, min_patch_size=args.min_patch_size, 
                    monotonic_mode=is_monotonic, fixed_threshold=optimal_thresholds[mode_name]
                )
            else:
                # Use traditional threshold calculation
                patch_boundaries, threshold = detect_patches_from_entropy(
                    entropies, args.threshold_multiplier, args.min_patch_size, is_monotonic
                )
            
            all_patch_boundaries[mode_name].append(patch_boundaries)
            all_thresholds[mode_name].append(threshold)
            
            patch_count = len(patch_boundaries) - 1
            avg_chunk_patch_size = args.context_length / patch_count if patch_count > 0 else args.context_length
            
            if optimal_thresholds[mode_name] is not None:
                print(f"    Entropy-based patches: {patch_count} with optimal threshold {threshold:.4f}")
                print(f"    Chunk patch size: {avg_chunk_patch_size:.1f}, Dataset avg: {dataset_avg_patch_sizes[mode_name]:.1f} (target: {args.target_patch_size})")
            else:
                print(f"    Entropy-based patches: {patch_count} with threshold {threshold:.4f}")
                print(f"    Chunk patch size: {avg_chunk_patch_size:.1f}")
            print(f"    Patch boundaries: {patch_boundaries}")
        
        all_ground_truths.append(ground_truth)
        all_predictions.append(preds)
        all_entropies.append(entropies)
    # Create results directory structure
    dataset_name = os.path.splitext(os.path.basename(args.csv_path))[0]
    results_dir = f"results_entropy_v2/{dataset_name}/col_{args.col_num}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create plots for each chunk and mode showing entropy-based patches
    modes = {'absolute': False, 'monotonic': True}
    
    for mode_name, is_monotonic in modes.items():
        print(f"\n=== Creating plots for {mode_name.upper()} mode ===")
        
        for i in range(3):
            fig, ax = plt.subplots(1, 1, figsize=(15, 6))
            
            # Create twin axis for entropy
            ax_twin = ax.twinx()
            
            # Plot ground truth and predictions
            ax.plot(all_ground_truths[i], label="Ground Truth", linestyle='-', 
                    linewidth=2, alpha=0.8, color='blue')
            ax.plot(all_predictions[i], label="Autoregressive Predictions", linestyle='--', 
                    linewidth=2, alpha=0.8, color='green')
            
            # Plot entropy or entropy difference based on mode
            if is_monotonic:
                entropy_diff = np.diff(all_entropies[i])
                ax_twin.plot(range(1, len(all_entropies[i])), entropy_diff, 
                             label="Entropy Difference", color='red', linewidth=1.5, alpha=0.7)
            else:
                ax_twin.plot(all_entropies[i], label="Token Entropy", 
                             color='red', linewidth=1.5, alpha=0.7)
            
            # Plot threshold
            threshold_val = all_thresholds[mode_name][i]
            if is_monotonic:
                ax_twin.axhline(y=threshold_val, color='orange', linestyle=':', 
                               linewidth=2, alpha=0.8, label=f'Entropy Diff Threshold ({threshold_val:.3f})')
            else:
                ax_twin.axhline(y=threshold_val, color='orange', linestyle=':', 
                               linewidth=2, alpha=0.8, label=f'Entropy Threshold ({threshold_val:.3f})')
            
            # Plot patch boundaries
            patch_boundaries = all_patch_boundaries[mode_name][i]
            for boundary in patch_boundaries:
                ax.axvline(x=boundary, color='purple', linestyle=':', 
                           linewidth=2, alpha=0.6)
            
            # Add patch boundary legend entry
            ax.axvline(x=-1, color='purple', linestyle=':', linewidth=2, 
                       alpha=0.6, label='Patch Boundaries')
            
            # Set labels and title
            patch_count = len(patch_boundaries) - 1
            context_avg_patch_size = args.context_length / patch_count if patch_count > 0 else args.context_length
            
            if optimal_thresholds[mode_name] is not None:
                # Include both dataset-level and context-level average patch sizes
                dataset_avg = dataset_avg_patch_sizes[mode_name]
                mode_str = f"Optimal Threshold: Target {args.target_patch_size}, Dataset Avg {dataset_avg:.1f}, Context Avg {context_avg_patch_size:.1f}"
            else:
                mode_str = f"{'Monotonic Entropy-Diff' if is_monotonic else 'Absolute Entropy'}, Context Avg {context_avg_patch_size:.1f}"
            
            ax.set_title(f"Chunk {i+1} - {mode_name.capitalize()} Mode ({mode_str}) - {patch_count} patches")
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Value", color='black')
            
            if is_monotonic:
                ax_twin.set_ylabel("Entropy Difference (bits)", color='red')
            else:
                ax_twin.set_ylabel("Token Entropy (bits)", color='red')
            ax_twin.tick_params(axis='y', labelcolor='red')
            
            # Add legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_twin.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(results_dir, f"chunk_{i+1}_entropy_patches_{mode_name}.pdf")
            plt.savefig(plot_path, format='pdf', bbox_inches='tight')
            print(f"Plot for chunk {i+1} ({mode_name} mode) saved to: {plot_path}")
            plt.show()

    # Calculate and print metrics for both modes
    has_optimal = any(optimal_thresholds[mode] is not None for mode in optimal_thresholds)
    if has_optimal:
        search_info = f"k={args.threshold_search_k}" if args.threshold_search_k is not None else "full dataset"
        threshold_method = f"Optimal (target: {args.target_patch_size}, search: {search_info})"
    else:
        threshold_method = "Traditional thresholding"
    
    print(f"\n=== PREDICTION METRICS ({threshold_method}) ===")
    
    for mode_name in ['absolute', 'monotonic']:
        print(f"\n--- {mode_name.upper()} MODE ---")
        
        for i in range(3):
            mae = np.mean(np.abs(all_ground_truths[i] - all_predictions[i]))
            mse = np.mean((all_ground_truths[i] - all_predictions[i]) ** 2)
            rmse = np.sqrt(mse)
            
            # Entropy method metrics
            avg_entropy = np.mean(all_entropies[i])
            patch_boundaries = all_patch_boundaries[mode_name][i]
            threshold_val = all_thresholds[mode_name][i]
            patch_count = len(patch_boundaries) - 1
            avg_patch_size = args.context_length / patch_count if patch_count > 0 else args.context_length
            
            print(f"Chunk {i+1}: MAE={mae:.4f}, RMSE={rmse:.4f}")
            print(f"         Entropy: Avg={avg_entropy:.4f} bits, Patches={patch_count}, Avg Patch Size={avg_patch_size:.1f}, Threshold={threshold_val:.4f}")


if __name__ == '__main__':
    main()
