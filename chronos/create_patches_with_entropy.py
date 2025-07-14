import argparse
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from chronos import BaseChronosPipeline
import random
import os
from tqdm import tqdm


def count_series_in_tsf_file(file_path):
    """Count the number of time series in a TSF file.
    
    Args:
        file_path: Path to the TSF file
    
    Returns:
        Number of time series found in the file
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the @data section
    data_section_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('@data'):
            data_section_start = i + 1
            break
    
    if data_section_start is None:
        return 0
    
    series_count = 0
    for i in range(data_section_start, len(lines)):
        line = lines[i].strip()
        if not line:  # Skip empty lines
            continue
            
        if ':' in line:
            parts = line.split(':')
            if len(parts) >= 4:  # TSF format: series_id:location:timestamp:data
                data_part = parts[-1]
                try:
                    # Check if we can parse at least one value
                    values = []
                    for x in data_part.split(','):
                        x = x.strip()
                        if x:
                            values.append(float(x))
                    
                    if len(values) > 0:
                        series_count += 1
                except ValueError:
                    continue
    
    return series_count


def parse_tsf_file(file_path, series_index=0):
    """Parse a TSF (Time Series Format) file and extract time series data.
    
    Args:
        file_path: Path to the TSF file
        series_index: Index of the time series to extract (0-based)
    
    Returns:
        numpy array containing the time series values
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the @data section
    data_section_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('@data'):
            data_section_start = i + 1
            break
    
    if data_section_start is None:
        raise ValueError("Could not find @data section in TSF file")
    
    # First, count total series to provide better error messages
    total_series = count_series_in_tsf_file(file_path)
    print(f"Found {total_series} time series in TSF file")
    
    if series_index >= total_series:
        raise ValueError(f"Requested series index {series_index} is out of range. "
                        f"File contains {total_series} time series (indices 0 to {total_series-1})")
    
    # Parse time series data
    series_data = []
    current_series = 0
    
    for i in range(data_section_start, len(lines)):
        line = lines[i].strip()
        if not line:  # Skip empty lines
            continue
            
        # Each line contains: series_id:location:timestamp:data_values
        # Example: T1:NSW:2002-01-01 00-00-00:5714.045004,5360.189078,...
        if ':' in line:
            # Split line to get the data part (last part after final colon)
            parts = line.split(':')
            if len(parts) >= 4:  # TSF format: series_id:location:timestamp:data
                # The time series values are in the last part, comma-separated
                data_part = parts[-1]
                try:
                    # Parse comma-separated values, filter out empty strings
                    values = []
                    for x in data_part.split(','):
                        x = x.strip()
                        if x:  # Only process non-empty strings
                            values.append(float(x))
                    
                    if len(values) > 0:  # Only count series with actual data
                        if current_series == series_index:
                            series_data = values
                            break
                        current_series += 1
                except ValueError as e:
                    # Skip lines that don't contain valid numeric data
                    print(f"Warning: Could not parse line {i}: {e}")
                    continue
    if not series_data:
        raise ValueError(f"Could not find time series at index {series_index} in TSF file")
    
    return np.array(series_data)


def load_time_series_data(file_path, series_index=0):
    """Load time series data from either CSV or TSF file format.
    
    Args:
        file_path: Path to the data file (.csv or .tsf)
        series_index: For CSV files, this is the column index (0-based).
                     For TSF files, this is the time series index (0-based).
    
    Returns:
        numpy array containing the time series values
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.csv':
        # Load CSV file using pandas
        df = pd.read_csv(file_path)
        if series_index >= len(df.columns):
            raise ValueError(f"Column index {series_index} is out of range for the CSV file")
        return df.iloc[:, series_index].values
    
    elif file_extension == '.tsf':
        # Load TSF file using custom parser
        return parse_tsf_file(file_path, series_index)
    
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Supported formats: .csv, .tsf")


def autoregressive_predict_context_simplified(pipeline, ground_truth_context, dataset_mean, max_context_length=None, entropy_batch=128):
    """Predict context autoregressively and compute entropy from token probabilities.
    
    Args:
        pipeline: The Chronos pipeline for prediction
        ground_truth_context: The ground truth time series data
        dataset_mean: Mean value of the dataset for initial context
        max_context_length: Maximum context length to use (if None, uses unbounded context until reaching model's max context, then sliding window)
        entropy_batch: Number of samples to process in each batch
    
    Returns:
        predictions: Array of predicted values
        entropies: Array of entropy values
        all_logits: List of logits for each timestep (for vocabulary analysis)
    """
    predictions = []
    entropies = []
    all_logits = []
    context_length = len(ground_truth_context)
    
    # If max_context_length is None, use model's default max context length (typically 1024 for Chronos)
    # This ensures we use sliding window approach for threshold optimization too
    effective_max_context = max_context_length if max_context_length is not None else 1024
    
    # Create all contexts for batch processing
    contexts = []
    for i in range(context_length):
        if i == 0:
            # Use dataset mean for first prediction context
            context = torch.tensor([dataset_mean], dtype=torch.float32)
        else:
            # Use ground truth values as context with sliding window approach
            if i > effective_max_context:
                # Keep only the latest effective_max_context samples (sliding window)
                start_idx = i - effective_max_context
                context = torch.tensor(ground_truth_context[start_idx:i], dtype=torch.float32)
            else:
                # Use all available context from 0 to i-1 (growing window)
                context = torch.tensor(ground_truth_context[:i], dtype=torch.float32)
        contexts.append(context)
    
    # Process contexts in batches
    for batch_start in tqdm(range(0, context_length, entropy_batch), desc="Processing batches"):
        batch_end = min(batch_start + entropy_batch, context_length)
        batch_contexts = contexts[batch_start:batch_end]
        
        # Get predictions with logits for entropy computation
        batch_preds, batch_logits = pipeline.predict(
            context=batch_contexts,
            prediction_length=1,  # Predict one step ahead
            num_samples=1,
            return_logits=True
        )
        
        # Process each sample in the batch
        for i in range(len(batch_contexts)):
            # Extract logits for the predicted token
            token_logits = batch_logits[i, 0, 0, :]  # Shape: [vocab_size]
            probs = torch.softmax(token_logits, dim=-1)

            # Store logits for vocabulary analysis
            all_logits.append(token_logits.cpu().numpy())
            
            # Compute entropy: H(X) = -sum(p(x) * log2(p(x)))
            entropy = -torch.sum(probs * torch.log2(probs + 1e-10)).item()
            entropies.append(entropy)
            
            # Get the predicted token for this timestep
            predicted_token = batch_preds[i, 0, 0].item()
            predictions.append(predicted_token)
    
    return np.array(predictions), np.array(entropies), all_logits


def detect_patches_from_entropy(entropies, threshold_multiplier=1.5, min_patch_size=5, 
                               relative_entropy_mode=False, fixed_threshold=None, max_patch_size=None):
    """
    Detect patch boundaries based on entropy spikes.
    
    Args:
        entropies: Array of entropy values
        threshold_multiplier: Multiplier for mean entropy to determine spike threshold
        min_patch_size: Minimum size of a patch
        relative_entropy_mode: If True, use Relative Entropy (diff-based)
        fixed_threshold: If not None, use this fixed threshold instead of calculating one
        max_patch_size: Maximum size of a patch (if None, no maximum limit)
    
    Returns:
        patch_boundaries: List of patch boundary indices
        threshold: The threshold value used for spike detection
    """
    if relative_entropy_mode:
        # Relative Entropy: use differences between consecutive entropy values
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
    
    # Enforce maximum patch size by adding additional boundaries if needed
    if max_patch_size is not None:
        final_boundaries = [patch_boundaries[0]]  # Start with first boundary
        
        for i in range(1, len(patch_boundaries)):
            current_start = final_boundaries[-1]
            current_end = patch_boundaries[i]
            
            # If the patch is too large, split it
            while current_end - current_start > max_patch_size:
                # Add a boundary at max_patch_size from current start
                new_boundary = current_start + max_patch_size
                final_boundaries.append(new_boundary)
                current_start = new_boundary
            
            # Add the original boundary
            final_boundaries.append(current_end)
        
        patch_boundaries = final_boundaries
    
    # Add final boundary
    if len(entropies) - 1 not in patch_boundaries:
        patch_boundaries.append(len(entropies) - 1)
    
    return patch_boundaries, threshold


def find_optimal_threshold(entropies, target_patch_size, min_patch_size=5, relative_entropy_mode=False, max_iterations=1000, max_patch_size=None):
    """
    Find the optimal threshold that produces patches with the target average size.
    
    Args:
        entropies: Array of entropy values for the entire dataset
        target_patch_size: Desired average patch size
        min_patch_size: Minimum size of a patch
        relative_entropy_mode: If True, use Relative Entropy (diff-based)
        max_iterations: Maximum number of iterations to find optimal threshold
        max_patch_size: Maximum size of a patch (if None, no maximum limit)
    
    Returns:
        optimal_threshold: The threshold that produces patches closest to target size
        final_avg_patch_size: The actual average patch size achieved
        iterations_used: Number of iterations used
    """
    print(f"\nFinding optimal threshold for target patch size: {target_patch_size}")
    
    # Initial threshold estimate using standard method
    if relative_entropy_mode:
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
            relative_entropy_mode=relative_entropy_mode, fixed_threshold=current_threshold,
            max_patch_size=max_patch_size
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
        relative_entropy_mode=relative_entropy_mode, fixed_threshold=best_threshold,
        max_patch_size=max_patch_size
    )
    num_patches = len(patch_boundaries) - 1
    final_avg_patch_size = len(entropies) / num_patches if num_patches > 0 else len(entropies)
    
    print(f"  Best threshold found: {best_threshold:.4f} "
          f"(avg patch size: {final_avg_patch_size:.1f}, diff: {best_diff:.1f})")
    
    return best_threshold, final_avg_patch_size, max_iterations


def create_patch_size_histogram(entropies, optimal_thresholds, dataset_avg_patch_sizes, args):
    """
    Create and save histograms showing the distribution of patch sizes for both modes.
    
    Args:
        entropies: Single entropy array or list of entropy arrays from threshold search samples
        optimal_thresholds: Dict with optimal thresholds for both modes
        dataset_avg_patch_sizes: Dict with average patch sizes for both modes
        args: Command line arguments
    """
    
    
    # Ensure entropies is a list
    if isinstance(entropies, np.ndarray):
        entropy_arrays = [entropies]
    else:
        entropy_arrays = entropies
    
    # Collect patch sizes for both modes
    patch_sizes = {'absolute_entropy': [], 'relative_entropy': []}
    
    for entropy_array in entropy_arrays:
        for mode_name, is_relative in [('absolute_entropy', False), ('relative_entropy', True)]:
            patch_boundaries, _ = detect_patches_from_entropy(
                entropy_array, threshold_multiplier=None, min_patch_size=args.min_patch_size,
                relative_entropy_mode=is_relative, fixed_threshold=optimal_thresholds[mode_name],
                max_patch_size=args.max_patch_size
            )
            
            # Calculate individual patch sizes
            for j in range(len(patch_boundaries) - 1):
                size = patch_boundaries[j+1] - patch_boundaries[j]
                patch_sizes[mode_name].append(size)
    
    # Create histogram plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = {'absolute_entropy': 'blue', 'relative_entropy': 'red'}
    
    for i, (mode_name, sizes) in enumerate(patch_sizes.items()):
        ax = ax1 if i == 0 else ax2
        
        if len(sizes) > 0:
            # Create histogram
            n, bins, patches = ax.hist(sizes, bins=30, alpha=0.7, color=colors[mode_name], 
                                     edgecolor='black', linewidth=0.5)
            
            # Add statistics text
            mean_size = np.mean(sizes)
            median_size = np.median(sizes)
            std_size = np.std(sizes)
            
            stats_text = f'Mean: {mean_size:.1f}\nMedian: {median_size:.1f}\nStd: {std_size:.1f}\nCount: {len(sizes)}'
            ax.text(0.7, 0.8, stats_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   verticalalignment='top', fontsize=16)
            
            # Add vertical line for target patch size
            ax.axvline(args.target_patch_size, color='green', linestyle='--', linewidth=2, 
                      label=f'Target: {args.target_patch_size}')
            
            # Add vertical line for actual dataset average
            ax.axvline(dataset_avg_patch_sizes[mode_name], color='orange', linestyle=':', linewidth=2,
                      label=f'Dataset Avg: {dataset_avg_patch_sizes[mode_name]:.1f}')
            
            ax.set_xlabel('Patch Size', fontsize=18)
            ax.set_ylabel('Frequency', fontsize=18)
            
            # Format title based on mode
            if mode_name == 'absolute_entropy':
                title = f'Patch Size Distribution - Absolute Entropy H(x_i)\nThreshold: {optimal_thresholds[mode_name]:.4f}'
            else:
                title = f'Patch Size Distribution - Relative Entropy H(x_i) - H(x_{{i-1}})\nThreshold: {optimal_thresholds[mode_name]:.4f}'
            
            ax.set_title(title, fontsize=18)
            ax.legend(fontsize=16)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=16)
        else:
            display_name = 'Absolute Entropy H(x_i)' if mode_name == 'absolute_entropy' else 'Relative Entropy H(x_i) - H(x_{i-1})'
            ax.text(0.5, 0.5, f'No patches found\nfor {display_name}', 
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"), fontsize=16)
            ax.set_title(f'Patch Size Distribution - {display_name}', fontsize=18)
    
    plt.tight_layout()
    
    # Get dataset name from data path
    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
    
    # Save histogram
    base_name = f"patch_size_histogram_{dataset_name}_series{args.series_index}"
    
    # Save as PNG
    png_path = f"results_entropy/{dataset_name}/series_{args.series_index}/{base_name}.png"
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    
    # Save as PDF
    pdf_path = f"results_entropy/{dataset_name}/series_{args.series_index}/{base_name}.pdf"
    plt.savefig(pdf_path, bbox_inches='tight')
    
    plt.close()
    
    print(f"Patch size histogram saved:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")
    
    # Print summary statistics
    print(f"\nPatch size statistics:")
    for mode_name, sizes in patch_sizes.items():
        if len(sizes) > 0:
            print(f"  {mode_name.title()} mode: mean={np.mean(sizes):.1f}, "
                  f"median={np.median(sizes):.1f}, std={np.std(sizes):.1f}, count={len(sizes)}")
        else:
            print(f"  {mode_name.title()} mode: No patches found")


def create_vocabulary_distribution_plots(entropies, all_logits, args, ground_truth_series=None, pipeline=None):
    """
    Create plots showing vocabulary probability distributions for high and low entropy points.
    
    Args:
        entropies: Array of entropy values
        all_logits: List of logits for each timestep
        args: Command line arguments
        ground_truth_series: Optional ground truth series to show actual values
        pipeline: The Chronos pipeline for tokenization
    """
    
    
    # Find 3 points with lowest entropy and 3 points with highest entropy
    entropy_indices = np.argsort(entropies)
    low_entropy_indices = entropy_indices[:3]
    high_entropy_indices = entropy_indices[-3:]
    
    # Get dataset name from data path
    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
    
    # Create plots for low entropy points
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Vocabulary Distributions at Low Entropy Points', fontsize=22)
    
    for i, idx in enumerate(low_entropy_indices):
        logits = all_logits[idx]
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        
        # Find peak and plot 100 values around it
        peak_idx = np.argmax(probs)
        start_idx = max(0, peak_idx - 50)
        end_idx = min(len(probs), peak_idx + 50)
        
        x_range = np.arange(start_idx, end_idx)
        y_values = probs[start_idx:end_idx]
        
        axes[i].bar(x_range, y_values, alpha=0.7, color='blue')
        
        # Add ground truth information if available
        if ground_truth_series is not None and idx < len(ground_truth_series):
            gt_value = ground_truth_series[idx]
            # Convert ground truth value to vocabulary token using the pipeline's tokenizer
            gt_token_tensor = torch.tensor([gt_value], dtype=torch.float32).unsqueeze(0)
            try:
                # Use the pipeline to get the vocabulary token for the ground truth value
                gt_tokens = pipeline.tokenizer.encode(gt_token_tensor)
                gt_token_idx = gt_tokens[0, 0].item() if len(gt_tokens.shape) > 1 else gt_tokens[0].item()
                title = f'Point {idx}: Entropy = {entropies[idx]:.3f}\nGT Value = {gt_value:.4f}, GT Token = {gt_token_idx}'
                
                # Mark the ground truth token on the plot
                axes[i].axvline(x=gt_token_idx, color='green', linestyle='-', linewidth=3, alpha=0.8, 
                               label=f'GT Token {gt_token_idx}')
            except:
                # Fallback if tokenization fails
                title = f'Point {idx}: Entropy = {entropies[idx]:.3f}\nGT Value = {gt_value:.4f}'
        else:
            title = f'Point {idx}: Entropy = {entropies[idx]:.3f}'
            
        axes[i].set_title(title, fontsize=18)
        axes[i].set_xlabel('Vocabulary Index', fontsize=18)
        axes[i].set_ylabel('Probability', fontsize=18)
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(labelsize=16)
        
        # Mark the peak
        axes[i].axvline(x=peak_idx, color='red', linestyle='--', alpha=0.8, label=f'Peak at {peak_idx}')
        axes[i].legend(fontsize=16)
    
    plt.tight_layout()
    
    # Save low entropy plot
    low_entropy_path_png = f"results_entropy/{dataset_name}/series_{args.series_index}/vocab_dist_low_entropy.png"
    low_entropy_path_pdf = f"results_entropy/{dataset_name}/series_{args.series_index}/vocab_dist_low_entropy.pdf"
    os.makedirs(os.path.dirname(low_entropy_path_png), exist_ok=True)
    plt.savefig(low_entropy_path_png, dpi=600, bbox_inches='tight')
    plt.savefig(low_entropy_path_pdf, bbox_inches='tight')
    plt.close()
    
    # Create plots for high entropy points
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Vocabulary Distributions at High Entropy Points', fontsize=22)
    
    for i, idx in enumerate(high_entropy_indices):
        logits = all_logits[idx]
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        
        # Find peak and plot 100 values around it
        peak_idx = np.argmax(probs)
        start_idx = max(0, peak_idx - 50)
        end_idx = min(len(probs), peak_idx + 50)
        
        x_range = np.arange(start_idx, end_idx)
        y_values = probs[start_idx:end_idx]
        
        axes[i].bar(x_range, y_values, alpha=0.7, color='red')
        
        # Add ground truth information if available
        if ground_truth_series is not None and idx < len(ground_truth_series):
            gt_value = ground_truth_series[idx]
            # Convert ground truth value to vocabulary token using the pipeline's tokenizer
            gt_token_tensor = torch.tensor([gt_value], dtype=torch.float32).unsqueeze(0)
            try:
                # Use the pipeline to get the vocabulary token for the ground truth value
                gt_tokens = pipeline.tokenizer.encode(gt_token_tensor)
                gt_token_idx = gt_tokens[0, 0].item() if len(gt_tokens.shape) > 1 else gt_tokens[0].item()
                title = f'Point {idx}: Entropy = {entropies[idx]:.3f}\nGT Value = {gt_value:.4f}, GT Token = {gt_token_idx}'
                
                # Mark the ground truth token on the plot
                axes[i].axvline(x=gt_token_idx, color='green', linestyle='-', linewidth=3, alpha=0.8, 
                               label=f'GT Token {gt_token_idx}')
            except:
                # Fallback if tokenization fails
                title = f'Point {idx}: Entropy = {entropies[idx]:.3f}\nGT Value = {gt_value:.4f}'
        else:
            title = f'Point {idx}: Entropy = {entropies[idx]:.3f}'
            
        axes[i].set_title(title, fontsize=18)
        axes[i].set_xlabel('Vocabulary Index', fontsize=18)
        axes[i].set_ylabel('Probability', fontsize=18)
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(labelsize=16)
        
        # Mark the peak
        axes[i].axvline(x=peak_idx, color='blue', linestyle='--', alpha=0.8, label=f'Peak at {peak_idx}')
        axes[i].legend(fontsize=16)
    
    plt.tight_layout()
    
    # Save high entropy plot
    high_entropy_path_png = f"results_entropy/{dataset_name}/series_{args.series_index}/vocab_dist_high_entropy.png"
    high_entropy_path_pdf = f"results_entropy/{dataset_name}/series_{args.series_index}/vocab_dist_high_entropy.pdf"
    plt.savefig(high_entropy_path_png, dpi=600, bbox_inches='tight')
    plt.savefig(high_entropy_path_pdf, bbox_inches='tight')
    plt.close()
    
    print(f"Vocabulary distribution plots saved:")
    print(f"  Low entropy: {low_entropy_path_png} and {low_entropy_path_pdf}")
    print(f"  High entropy: {high_entropy_path_png} and {high_entropy_path_pdf}")


def create_multi_resolution_plots(all_ground_truths, all_predictions, all_entropies, 
                                all_patch_boundaries, all_thresholds, optimal_thresholds, 
                                dataset_avg_patch_sizes, args, results_dir):
    """
    Create plots at multiple resolutions (1/4, 1/2, full context) for each chunk.
    """
    
    
    modes = {'absolute_entropy': False, 'relative_entropy': True}
    resolutions = [('quarter', 0.25), ('half', 0.5), ('full', 1.0)]
    
    for chunk_idx in range(3):
        for res_name, res_fraction in resolutions:
            end_point = int(args.context_length * res_fraction)
            
            # Create figure with 2 subplots stacked vertically
            fig, (ax_abs, ax_rel) = plt.subplots(2, 1, figsize=(16, 12))
            axes = {'absolute_entropy': ax_abs, 'relative_entropy': ax_rel}
            
            # Calculate y-axis limits for this resolution
            all_values = []
            all_values.extend(all_ground_truths[chunk_idx][:end_point])
            all_values.extend(all_predictions[chunk_idx][:end_point])
            left_y_min, left_y_max = min(all_values), max(all_values)
            left_y_range = left_y_max - left_y_min
            left_y_margin = max(left_y_range * 0.1, 0.01)
            left_y_limits = (left_y_min - left_y_margin, left_y_max + left_y_margin)
            
            # Calculate right axis limits
            all_entropy_values = []
            all_threshold_values = []
            for mode_name, is_relative in modes.items():
                if is_relative:
                    entropy_diff = np.diff(all_entropies[chunk_idx][:end_point])
                    all_entropy_values.extend(entropy_diff)
                else:
                    all_entropy_values.extend(all_entropies[chunk_idx][:end_point])
                all_threshold_values.append(all_thresholds[mode_name][chunk_idx])
            
            right_y_min = min(min(all_entropy_values), min(all_threshold_values))
            right_y_max = max(max(all_entropy_values), max(all_threshold_values))
            right_y_range = right_y_max - right_y_min
            right_y_margin = max(right_y_range * 0.1, 0.01)
            right_y_limits = (right_y_min - right_y_margin, right_y_max + right_y_margin)
            
            for mode_name, is_relative in modes.items():
                ax = axes[mode_name]
                ax_twin = ax.twinx()
                
                # Plot ground truth and predictions (limited to resolution)
                ax.plot(all_ground_truths[chunk_idx][:end_point], label="Ground Truth", 
                       linestyle='-', linewidth=2, alpha=0.8, color='blue')
                ax.plot(all_predictions[chunk_idx][:end_point], label="Autoregressive Predictions", 
                       linestyle='--', linewidth=2, alpha=0.8, color='green')
                
                # Plot entropy or entropy difference based on mode
                if is_relative:
                    entropy_diff = np.diff(all_entropies[chunk_idx][:end_point])
                    ax_twin.plot(range(1, len(entropy_diff) + 1), entropy_diff, 
                               label="Relative Entropy", color='red', linewidth=1.5, alpha=0.7)
                else:
                    ax_twin.plot(all_entropies[chunk_idx][:end_point], label="Absolute Entropy", 
                               color='red', linewidth=1.5, alpha=0.7)
                
                # Plot threshold
                threshold_val = all_thresholds[mode_name][chunk_idx]
                if is_relative:
                    ax_twin.axhline(y=threshold_val, color='orange', linestyle=':', 
                                   linewidth=2, alpha=0.8, label=f'Relative Entropy Threshold ({threshold_val:.3f})')
                else:
                    ax_twin.axhline(y=threshold_val, color='orange', linestyle=':', 
                                   linewidth=2, alpha=0.8, label=f'Absolute Entropy Threshold ({threshold_val:.3f})')
                
                # Plot patch boundaries (only those within the resolution range)
                patch_boundaries = all_patch_boundaries[mode_name][chunk_idx]
                boundaries_in_range = [b for b in patch_boundaries if b <= end_point]
                for boundary in boundaries_in_range:
                    ax.axvline(x=boundary, color='purple', linestyle=':', 
                              linewidth=2, alpha=0.6)
                
                # Add patch boundary legend entry
                if boundaries_in_range:
                    ax.axvline(x=-1, color='purple', linestyle=':', linewidth=2, 
                              alpha=0.6, label='Patch Boundaries')
                
                # Calculate patch statistics for this resolution
                patches_in_range = 0
                for i in range(len(boundaries_in_range) - 1):
                    if boundaries_in_range[i+1] <= end_point:
                        patches_in_range += 1
                avg_patch_size_res = end_point / patches_in_range if patches_in_range > 0 else end_point
                
                # Set labels and title
                if optimal_thresholds[mode_name] is not None:
                    dataset_avg = dataset_avg_patch_sizes[mode_name]
                    mode_str = f"Target {args.target_patch_size}, Dataset Avg {dataset_avg:.1f}, {res_name.title()} Avg {avg_patch_size_res:.1f}"
                else:
                    mode_str = f"{res_name.title()} Avg {avg_patch_size_res:.1f}"
                
                # Format title and labels based on mode
                if mode_name == 'absolute_entropy':
                    mode_display = "Absolute Entropy H(x_i)"
                    y_label = "$H(x_i)$ (bits)"
                else:
                    mode_display = "Relative Entropy H(x_i) - H(x_{i-1})"
                    y_label = "$H(x_i) - H(x_{i-1})$ (bits)"
                
                ax.set_title(f"{mode_display} ({mode_str}) - {patches_in_range} patches", fontsize=18)
                ax.set_xlabel("Sample Index", fontsize=18)
                ax.set_ylabel("Value", color='black', fontsize=18)
                
                ax_twin.set_ylabel(y_label, color='red', fontsize=18)
                ax_twin.tick_params(axis='y', labelcolor='red', labelsize=16)
                ax.tick_params(labelsize=16)
                
                # Set consistent y-axis limits
                ax.set_ylim(left_y_limits)
                ax_twin.set_ylim(right_y_limits)
                
                # Add legends
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax_twin.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=14)
                ax.grid(True, alpha=0.3)
            
            # Set overall figure title
            fig.suptitle(f"Chunk {chunk_idx+1} - {res_name.title()} Resolution ({end_point} points)", fontsize=20, y=0.98)
            
            plt.tight_layout()
            
            # Save plots
            plot_path_pdf = os.path.join(results_dir, f"chunk_{chunk_idx+1}_{res_name}_resolution.pdf")
            plot_path_png = os.path.join(results_dir, f"chunk_{chunk_idx+1}_{res_name}_resolution.png")
            plt.savefig(plot_path_pdf, format='pdf', bbox_inches='tight', dpi=300)
            plt.savefig(plot_path_png, format='png', dpi=600, bbox_inches='tight')
            print(f"Chunk {chunk_idx+1} {res_name} resolution plot saved to: {plot_path_pdf} and {plot_path_png}")
            plt.close()


def create_full_dataset_plots(series, entropies, all_logits, optimal_thresholds, dataset_avg_patch_sizes, args):
    """
    Create plots showing the entire dataset with entropy and patch boundaries.
    Creates two versions: one with the signal overlapped, one without.
    
    Args:
        series: The full time series data
        entropies: Array of entropy values for the dataset
        all_logits: List of logits for each timestep
        optimal_thresholds: Dict with optimal thresholds for both modes
        dataset_avg_patch_sizes: Dict with average patch sizes for both modes
        args: Command line arguments
    """
    
    
    # Get dataset name from data path
    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
    
    modes = {'absolute_entropy': False, 'relative_entropy': True}
    
    # Determine how much of the dataset we have entropy data for
    entropy_length = len(entropies)
    series_length = len(series)
    
    print(f"  Full dataset length: {series_length}, Entropy data length: {entropy_length}")
    
    # Create plot WITH signal overlay
    fig, axes = plt.subplots(2, 1, figsize=(20, 12))
    
    for i, (mode_name, is_relative) in enumerate(modes.items()):
        ax = axes[i]
        ax_twin = ax.twinx()
        
        # Plot the FULL time series signal
        ax.plot(series, label="Time Series Signal", linestyle='-', linewidth=1.5, alpha=0.8, color='blue')
        
        # Plot entropy or entropy difference based on mode (only for the portion we have data)
        if is_relative:
            entropy_diff = np.diff(entropies)
            ax_twin.plot(range(1, len(entropy_diff) + 1), entropy_diff, 
                        label="Relative Entropy $H(x_i) - H(x_{i-1})$", color='red', linewidth=1, alpha=0.7)
            y_label = "$H(x_i) - H(x_{i-1})$ (bits)"
        else:
            ax_twin.plot(range(len(entropies)), entropies, label="Absolute Entropy $H(x_i)$", 
                        color='red', linewidth=1, alpha=0.7)
            y_label = "$H(x_i)$ (bits)"
        
        # Plot threshold if available
        if optimal_thresholds[mode_name] is not None:
            threshold_val = optimal_thresholds[mode_name]
            ax_twin.axhline(y=threshold_val, color='orange', linestyle=':', 
                           linewidth=2, alpha=0.8, label=f'Threshold ({threshold_val:.3f})')
            
            # Detect and plot patch boundaries
            patch_boundaries, _ = detect_patches_from_entropy(
                entropies, threshold_multiplier=None, min_patch_size=args.min_patch_size,
                relative_entropy_mode=is_relative, fixed_threshold=threshold_val,
                max_patch_size=args.max_patch_size
            )
            
            # Plot patch boundaries
            for boundary in patch_boundaries[1:-1]:  # Skip first and last
                ax.axvline(x=boundary, color='purple', linestyle=':', linewidth=1.5, alpha=0.6)
            
            # Add patch boundary legend entry
            if len(patch_boundaries) > 2:
                ax.axvline(x=-1, color='purple', linestyle=':', linewidth=1.5, 
                          alpha=0.6, label='Patch Boundaries')
            
            patch_count = len(patch_boundaries) - 1
            avg_patch_size = len(entropies) / patch_count if patch_count > 0 else len(entropies)
            
            mode_display = "Absolute Entropy" if mode_name == 'absolute_entropy' else "Relative Entropy"
            title = f"{mode_display} with Signal Overlay\nPatches: {patch_count}, Avg Size: {avg_patch_size:.1f}, Target: {args.target_patch_size}"
        else:
            mode_display = "Absolute Entropy" if mode_name == 'absolute_entropy' else "Relative Entropy"
            title = f"{mode_display} with Signal Overlay"
        
        # Set labels and formatting
        ax.set_title(title, fontsize=18)
        ax.set_xlabel("Sample Index", fontsize=16)
        ax.set_ylabel("Signal Value", color='blue', fontsize=16)
        ax_twin.set_ylabel(y_label, color='red', fontsize=16)
        
        # Set tick label sizes
        ax.tick_params(axis='y', labelcolor='blue', labelsize=14)
        ax_twin.tick_params(axis='y', labelcolor='red', labelsize=14)
        ax.tick_params(axis='x', labelsize=14)
        
        # Add legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot with signal overlay
    with_signal_png = f"results_entropy/{dataset_name}/series_{args.series_index}/full_dataset_with_signal.png"
    with_signal_pdf = f"results_entropy/{dataset_name}/series_{args.series_index}/full_dataset_with_signal.pdf"
    os.makedirs(os.path.dirname(with_signal_png), exist_ok=True)
    plt.savefig(with_signal_png, dpi=600, bbox_inches='tight')
    plt.savefig(with_signal_pdf, bbox_inches='tight')
    plt.close()
    
    # Create plot WITHOUT signal overlay (entropy only)
    fig, axes = plt.subplots(2, 1, figsize=(20, 10))
    
    for i, (mode_name, is_relative) in enumerate(modes.items()):
        ax = axes[i]
        
        # Plot entropy or entropy difference based on mode
        if is_relative:
            entropy_diff = np.diff(entropies)
            ax.plot(range(1, len(entropy_diff) + 1), entropy_diff, 
                   label="Relative Entropy $H(x_i) - H(x_{i-1})$", color='red', linewidth=1.5, alpha=0.8)
            y_label = "$H(x_i) - H(x_{i-1})$ (bits)"
        else:
            ax.plot(range(len(entropies)), entropies, label="Absolute Entropy $H(x_i)$", 
                   color='red', linewidth=1.5, alpha=0.8)
            y_label = "$H(x_i)$ (bits)"
        
        # Set x-axis to span the full signal length
        ax.set_xlim(0, series_length - 1)
        
        # Set x-axis to span the full signal length (not just entropy length)
        ax.set_xlim(0, series_length - 1)
        
        # Plot threshold if available
        if optimal_thresholds[mode_name] is not None:
            threshold_val = optimal_thresholds[mode_name]
            ax.axhline(y=threshold_val, color='orange', linestyle=':', 
                      linewidth=2, alpha=0.8, label=f'Threshold ({threshold_val:.3f})')
            
            # Detect and plot patch boundaries
            patch_boundaries, _ = detect_patches_from_entropy(
                entropies, threshold_multiplier=None, min_patch_size=args.min_patch_size,
                relative_entropy_mode=is_relative, fixed_threshold=threshold_val,
                max_patch_size=args.max_patch_size
            )
            
            # Plot patch boundaries
            for boundary in patch_boundaries[1:-1]:  # Skip first and last
                ax.axvline(x=boundary, color='purple', linestyle=':', linewidth=1.5, alpha=0.6)
            
            # Add patch boundary legend entry
            if len(patch_boundaries) > 2:
                ax.axvline(x=-1, color='purple', linestyle=':', linewidth=1.5, 
                          alpha=0.6, label='Patch Boundaries')
            
            patch_count = len(patch_boundaries) - 1
            avg_patch_size = len(entropies) / patch_count if patch_count > 0 else len(entropies)
            
            mode_display = "Absolute Entropy" if mode_name == 'absolute_entropy' else "Relative Entropy"
            title = f"{mode_display} - Entropy Only\nPatches: {patch_count}, Avg Size: {avg_patch_size:.1f}, Target: {args.target_patch_size}"
        else:
            mode_display = "Absolute Entropy" if mode_name == 'absolute_entropy' else "Relative Entropy"
            title = f"{mode_display} - Entropy Only"
        
        # Set labels and formatting
        ax.set_title(title, fontsize=18)
        ax.set_xlabel("Sample Index", fontsize=16)
        ax.set_ylabel(y_label, color='red', fontsize=16)
        
        # Set tick label sizes
        ax.tick_params(axis='y', labelcolor='red', labelsize=14)
        ax.tick_params(axis='x', labelsize=14)
        
        # Add legend
        ax.legend(loc='upper left', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot without signal overlay
    without_signal_png = f"results_entropy/{dataset_name}/series_{args.series_index}/full_dataset_entropy_only.png"
    without_signal_pdf = f"results_entropy/{dataset_name}/series_{args.series_index}/full_dataset_entropy_only.pdf"
    plt.savefig(without_signal_png, dpi=600, bbox_inches='tight')
    plt.savefig(without_signal_pdf, bbox_inches='tight')
    plt.close()
    
    print(f"Full dataset plots saved:")
    print(f"  With signal overlay: {with_signal_png} and {with_signal_pdf}")
    print(f"  Entropy only: {without_signal_png} and {without_signal_pdf}")


def get_data_files(data_path):
    """Get list of data files to process.
    
    Args:
        data_path: Path to either a file or directory
        
    Returns:
        List of file paths to process
    """
    if os.path.isfile(data_path):
        # Single file
        return [data_path]
    elif os.path.isdir(data_path):
        # Directory - find all CSV and TSF files
        files = []
        for filename in os.listdir(data_path):
            filepath = os.path.join(data_path, filename)
            if os.path.isfile(filepath) and filename.lower().endswith(('.csv', '.tsf')):
                files.append(filepath)
        return sorted(files)  # Sort for consistent ordering
    else:
        raise FileNotFoundError(f"Path not found: {data_path}")


def process_single_file(file_path, args, pipeline, dataset_mean=None):
    """Process a single data file.
    
    Args:
        file_path: Path to the data file
        args: Command line arguments
        pipeline: The loaded Chronos pipeline
        dataset_mean: Precomputed dataset mean (if None, will compute from file)
        
    Returns:
        None (saves results to disk)
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING FILE: {file_path}")
    print(f"{'='*80}")
    
    try:
        series = load_time_series_data(file_path, args.series_index)
        file_type = "TSF" if file_path.lower().endswith('.tsf') else "CSV"
        print(f"Loaded {file_type} file: {file_path}")
        print(f"Using series index: {args.series_index}")
        print(f"Time series length: {len(series)}")
    except Exception as e:
        print(f"ERROR: Could not load file {file_path}: {str(e)}")
        return

    # Calculate dataset mean if not provided
    if dataset_mean is None:
        dataset_mean = np.mean(series)
    print(f"Dataset mean: {dataset_mean:.4f}")

    # Validate data length
    if len(series) < args.context_length:
        print(f"WARNING: Time series too short ({len(series)}) for context length {args.context_length}. Skipping file.")
        return
    
    # Select 3 random chunks of context_length
    max_start = len(series) - args.context_length
    if max_start < 3:
        print(f"WARNING: Time series too short for 3 chunks of length {args.context_length}. Using fewer chunks.")
        num_chunks = max(1, max_start + 1)
        if num_chunks == 1:
            random_starts = [0]
        else:
            random_starts = random.sample(range(max_start + 1), min(3, num_chunks))
    else:
        random_starts = random.sample(range(max_start), 3)

    # If target patch size is specified, compute entropy for dataset to find optimal thresholds
    optimal_thresholds = {'absolute_entropy': None, 'relative_entropy': None}
    dataset_avg_patch_sizes = {'absolute_entropy': None, 'relative_entropy': None}
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
        
        # Compute entropy for the search series (using sliding window approach for threshold optimization)
        _, search_entropies, search_logits = autoregressive_predict_context_simplified(
            pipeline, search_series, dataset_mean, max_context_length=args.context_length, entropy_batch=args.entropy_batch)

        # Find optimal thresholds for both modes
        print("\n--- Finding optimal threshold for Absolute Entropy mode ---")
        optimal_thresholds['absolute_entropy'], dataset_avg_patch_sizes['absolute_entropy'], abs_iterations = find_optimal_threshold(
            search_entropies, args.target_patch_size, args.min_patch_size, 
            relative_entropy_mode=False, max_iterations=args.max_iterations, max_patch_size=args.max_patch_size
        )
        
        print("\n--- Finding optimal threshold for Relative Entropy mode ---")
        optimal_thresholds['relative_entropy'], dataset_avg_patch_sizes['relative_entropy'], rel_iterations = find_optimal_threshold(
            search_entropies, args.target_patch_size, args.min_patch_size, 
            relative_entropy_mode=True, max_iterations=args.max_iterations, max_patch_size=args.max_patch_size
        )
        
        print(f"\nOptimal thresholds found:")
        print(f"  Absolute Entropy mode: {optimal_thresholds['absolute_entropy']:.4f} (dataset avg size: {dataset_avg_patch_sizes['absolute_entropy']:.1f})")
        print(f"  Relative Entropy mode: {optimal_thresholds['relative_entropy']:.4f} (dataset avg size: {dataset_avg_patch_sizes['relative_entropy']:.1f})")
        print(f"Threshold search performed on: {search_description}")
        
        # Update args.data_path temporarily for this file to ensure correct paths in plots
        original_data_path = args.data_path
        args.data_path = file_path
        
        # Create histogram of patch size distribution for both modes
        print("\nCreating patch size distribution histograms...")
        create_patch_size_histogram(search_entropies, optimal_thresholds, 
                                   dataset_avg_patch_sizes, args)
        
        # Create vocabulary distribution plots
        print("\nCreating vocabulary distribution plots...")
        create_vocabulary_distribution_plots(search_entropies, search_logits, args, search_series, pipeline)
        
        # Create full dataset plots
        print("\nCreating full dataset plots...")
        create_full_dataset_plots(series, search_entropies, search_logits, 
                                optimal_thresholds, dataset_avg_patch_sizes, args)
        
        # Restore original data_path
        args.data_path = original_data_path

    # Generate autoregressive predictions for each chunk
    all_ground_truths = []
    all_predictions = []
    all_entropies = []
    all_logits = []
    all_patch_boundaries = {'absolute_entropy': [], 'relative_entropy': []}
    all_thresholds = {'absolute_entropy': [], 'relative_entropy': []}

    for i, start in enumerate(random_starts):
        print(f"\nProcessing chunk {i+1}/{len(random_starts)} (start index {start})...")
        ground_truth = series[start:start+args.context_length]
        preds, entropies, logits = autoregressive_predict_context_simplified(
            pipeline, ground_truth, dataset_mean, max_context_length=args.context_length, entropy_batch=args.entropy_batch)
        
        # Process both modes
        modes = {'absolute_entropy': False, 'relative_entropy': True}
        
        for mode_name, is_relative in modes.items():
            mode_display = "Absolute Entropy" if mode_name == 'absolute_entropy' else "Relative Entropy"
            print(f"\n  --- {mode_display} Mode ---")
            
            # Detect patches based on entropy spikes
            if optimal_thresholds[mode_name] is not None:
                # Use the optimal threshold found from full dataset analysis
                patch_boundaries, threshold = detect_patches_from_entropy(
                    entropies, threshold_multiplier=None, min_patch_size=args.min_patch_size, 
                    relative_entropy_mode=is_relative, fixed_threshold=optimal_thresholds[mode_name],
                    max_patch_size=args.max_patch_size
                )
            else:
                # Use traditional threshold calculation
                patch_boundaries, threshold = detect_patches_from_entropy(
                    entropies, args.threshold_multiplier, args.min_patch_size, is_relative,
                    max_patch_size=args.max_patch_size
                )
            
            all_patch_boundaries[mode_name].append(patch_boundaries)
            all_thresholds[mode_name].append(threshold)
            
            patch_count = len(patch_boundaries) - 1
            avg_chunk_patch_size = len(entropies) / patch_count if patch_count > 0 else len(entropies)
            
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
        all_logits.append(logits)
    
    # Create results directory structure
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    results_dir = f"results_entropy/{dataset_name}/series_{args.series_index}"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n=== Creating multi-resolution plots ===")
    
    # Update args.data_path temporarily for this file to ensure correct paths in plots
    original_data_path = args.data_path
    args.data_path = file_path
    
    # Create plots at multiple resolutions for each chunk
    create_multi_resolution_plots(all_ground_truths, all_predictions, all_entropies,
                                all_patch_boundaries, all_thresholds, optimal_thresholds,
                                dataset_avg_patch_sizes, args, results_dir)

    # Calculate and print metrics for both modes
    has_optimal = any(optimal_thresholds[mode] is not None for mode in optimal_thresholds)
    if has_optimal:
        search_info = f"k={args.threshold_search_k}" if args.threshold_search_k is not None else "full dataset"
        threshold_method = f"Optimal (target: {args.target_patch_size}, search: {search_info})"
    else:
        threshold_method = "Traditional thresholding"
    
    print(f"\n=== PREDICTION METRICS ({threshold_method}) ===")
    
    for mode_name in ['absolute_entropy', 'relative_entropy']:
        mode_display = "ABSOLUTE ENTROPY" if mode_name == 'absolute_entropy' else "RELATIVE ENTROPY"
        print(f"\n--- {mode_display} MODE ---")
        
        for i in range(len(random_starts)):
            mae = np.mean(np.abs(all_ground_truths[i] - all_predictions[i]))
            mse = np.mean((all_ground_truths[i] - all_predictions[i]) ** 2)
            rmse = np.sqrt(mse)
            
            # Entropy method metrics
            avg_entropy = np.mean(all_entropies[i])
            patch_boundaries = all_patch_boundaries[mode_name][i]
            threshold_val = all_thresholds[mode_name][i]
            patch_count = len(patch_boundaries) - 1
            avg_patch_size = len(all_entropies[i]) / patch_count if patch_count > 0 else len(all_entropies[i])
            
            print(f"Chunk {i+1}: MAE={mae:.4f}, RMSE={rmse:.4f}")
            print(f"         Entropy: Avg={avg_entropy:.4f} bits, Patches={patch_count}, Avg Patch Size={avg_patch_size:.1f}, Threshold={threshold_val:.4f}")
    
    # Restore original data_path
    args.data_path = original_data_path
    
    print(f"\nCompleted processing file: {file_path}")


def main():
    parser = argparse.ArgumentParser(description='Autoregressive prediction with entropy-based patches for 3 random chunks. Creates Absolute Entropy H(x_i) and Relative Entropy H(x_i) - H(x_{i-1}) visualizations including full dataset plots.')
    parser.add_argument('--data_path', type=str, 
                        default='../datasets/time-moe-eval/ETT-small/ETTm2.csv',
                        help='Path to a data file (CSV or TSF format) or directory containing data files')
    parser.add_argument('--series_index', type=int, default=1, 
                        help='Series index (0-based): For CSV files, this is the column number. For TSF files, this is the time series index.')
    parser.add_argument('--context_length', type=int, default=1024, 
                        help='Length of context to predict autoregressively')
    parser.add_argument('--model', type=str, default='../model_weights/chronos/chronos-t5-small',
                        help='Chronos model to use for prediction')
    parser.add_argument('--threshold_multiplier', type=float, default=1.5,
                        help='Multiplier for mean+std to determine spike threshold (ignored if target_patch_size is set)')
    parser.add_argument('--target_patch_size', type=int, default=8,
                        help='Target average patch size. If set, will automatically find optimal threshold.')
    parser.add_argument('--threshold_search_k', type=int, default=None,
                        help='Number of samples to use for threshold search. If not set, uses entire dataset.')
    parser.add_argument('--min_patch_size', type=int, default=4,
                        help='Minimum size of a patch')
    parser.add_argument('--max_patch_size', type=int, default=16,
                        help='Maximum size of a patch')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU ID to use. If not specified, uses CPU. Example: --gpu 0')
    parser.add_argument('--max_iterations', type=int, default=1000,
                        help='Maximum number of iterations for threshold optimization (default: 1000)')
    parser.add_argument('--entropy_batch', type=int, default=128,
                        help='Number of samples to process in each batch for entropy computation (default: 128)')
    
    # Add backward compatibility for old argument names
    parser.add_argument('--csv_path', type=str, 
                        help='Deprecated: Use --data_path instead. Path to the CSV file containing the time series data')
    parser.add_argument('--col_num', type=int, 
                        help='Deprecated: Use --series_index instead. Column number (0-indexed) to process')
    
    args = parser.parse_args()

    # Handle backward compatibility
    if args.csv_path is not None:
        print("Warning: --csv_path is deprecated. Use --data_path instead.")
        args.data_path = args.csv_path
    if args.col_num is not None:
        print("Warning: --col_num is deprecated. Use --series_index instead.")
        args.series_index = args.col_num

    # Get list of files to process
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Path not found: {args.data_path}")
    
    try:
        data_files = get_data_files(args.data_path)
        if not data_files:
            raise ValueError(f"No CSV or TSF files found in: {args.data_path}")
        
        print(f"Found {len(data_files)} file(s) to process:")
        for i, file_path in enumerate(data_files, 1):
            print(f"  {i}. {file_path}")
        
    except Exception as e:
        raise ValueError(f"Error finding data files: {str(e)}")

    # Validate threshold_search_k argument (will be validated per file)
    if args.threshold_search_k is not None and args.threshold_search_k <= 0:
        raise ValueError("threshold_search_k must be positive")
    if args.target_patch_size is None and args.threshold_search_k is not None:
        print("Warning: threshold_search_k specified but target_patch_size not set. threshold_search_k will be ignored.")

    # Load model - default to CPU unless GPU ID is explicitly provided
    if args.gpu is not None:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            if args.gpu < torch.cuda.device_count():
                device = f"cuda:{args.gpu}"
                print(f"Using GPU {args.gpu} as requested")
            else:
                print(f"Warning: GPU {args.gpu} not available. Available GPUs: 0-{torch.cuda.device_count()-1}. Using CPU instead.")
                device = "cpu"
        else:
            print("Warning: CUDA not available. Using CPU instead.")
            device = "cpu"
    else:
        device = "cpu"
        print("Using CPU (default). Specify --gpu <id> to use GPU.")
    print(f"Loading model {args.model} on {device}...")
    pipeline = BaseChronosPipeline.from_pretrained(
        args.model,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )

    # Process each file
    for file_idx, file_path in enumerate(data_files, 1):
        print(f"\n{'#'*80}")
        print(f"PROCESSING FILE {file_idx}/{len(data_files)}: {os.path.basename(file_path)}")
        print(f"{'#'*80}")
        
        try:
            process_single_file(file_path, args, pipeline)
        except Exception as e:
            print(f"ERROR processing file {file_path}: {str(e)}")
            print(f"Continuing with next file...")
            continue
    
    print(f"\n{'='*80}")
    print(f"COMPLETED PROCESSING ALL FILES")
    print(f"Successfully processed files from: {args.data_path}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
