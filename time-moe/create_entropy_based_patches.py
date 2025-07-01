#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import os
import glob
try:
    from datasets import load_from_disk
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    print("Warning: datasets library not available. HuggingFace dataset support disabled.")

def compute_entropy_and_scaled(x, B, pooled_scaled_data=None, context_length=None):
    """Compute scaled signal and per-sample entropy."""
    s = np.mean(np.abs(x))
    if s == 0:
        raise ValueError("All-zero signal")
    scaled = x / s
    
    if context_length is not None:
        # Contextual entropy
        per_sample_entropy = np.zeros(len(scaled))
        for i in range(len(scaled)):
            start = max(0, i - context_length // 2)
            end = min(len(scaled), start + context_length)
            if end - start < context_length and start > 0:
                start = max(0, end - context_length)
            
            context = scaled[start:end]
            c_min, c_max = context.min(), context.max()
            if c_min == c_max:
                per_sample_entropy[i] = 0.0
                continue
                
            edges = np.linspace(c_min, c_max, B + 1)[1:-1]
            sample_bin = np.clip(np.digitize([scaled[i]], edges)[0], 0, B-1)
            context_bins = np.clip(np.digitize(context, edges), 0, B-1)
            counts = np.bincount(context_bins, minlength=B)
            probs = counts / len(context)
            per_sample_entropy[i] = -np.log2(probs[sample_bin] + np.finfo(float).eps)
        return scaled, per_sample_entropy
    
    # Local/global entropy
    prob_data = pooled_scaled_data if pooled_scaled_data is not None else scaled
    c_min, c_max = prob_data.min(), prob_data.max()
    edges = np.linspace(c_min, c_max, B + 1)[1:-1]
    
    bins = np.clip(np.digitize(scaled, edges), 0, B-1)
    prob_bins = np.clip(np.digitize(prob_data, edges), 0, B-1)
    counts = np.bincount(prob_bins, minlength=B)
    probs = counts / len(prob_data)
    per_sample_entropy = -np.log2(probs[bins] + np.finfo(float).eps)
    
    return scaled, per_sample_entropy, (edges, probs, counts)

def plot_scaling_effect(original_data, scaled_data, dataset_name, output_dir, chunk_size=128):
    """
    Plot a random chunk of data before and after scaling to show the effect.
    
    Args:
        original_data: Original unscaled data
        scaled_data: Scaled data (divided by mean absolute value)
        dataset_name: Name of the dataset for the plot title
        output_dir: Directory to save the plot
        chunk_size: Number of samples to plot (default 128)
    """
    if len(original_data) < chunk_size:
        print(f"Warning: Dataset {dataset_name} has only {len(original_data)} samples, using all available")
        chunk_size = len(original_data)
    
    # Select a random chunk
    np.random.seed(42)  # For reproducible results
    max_start = len(original_data) - chunk_size
    start_idx = np.random.randint(0, max_start + 1) if max_start > 0 else 0
    end_idx = start_idx + chunk_size
    
    # Extract chunks
    orig_chunk = original_data[start_idx:end_idx]
    scaled_chunk = scaled_data[start_idx:end_idx]
    indices = np.arange(start_idx, end_idx)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Original data plot
    ax1.plot(indices, orig_chunk, '.-', color='C0', markersize=4, alpha=0.8)
    ax1.set_title(f'Original Data - {dataset_name}\nSamples {start_idx}-{end_idx-1}')
    ax1.set_ylabel('Original Value')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text
    orig_stats = f'Mean: {np.mean(orig_chunk):.4f}, Std: {np.std(orig_chunk):.4f}, Range: [{np.min(orig_chunk):.4f}, {np.max(orig_chunk):.4f}]'
    ax1.text(0.02, 0.98, orig_stats, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Scaled data plot
    ax2.plot(indices, scaled_chunk, '.-', color='C1', markersize=4, alpha=0.8)
    ax2.set_title(f'Scaled Data - {dataset_name}\n(Divided by mean absolute value: {np.mean(np.abs(original_data)):.4f})')
    ax2.set_ylabel('Scaled Value')
    ax2.set_xlabel('Sample Index')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    scaled_stats = f'Mean: {np.mean(scaled_chunk):.4f}, Std: {np.std(scaled_chunk):.4f}, Range: [{np.min(scaled_chunk):.4f}, {np.max(scaled_chunk):.4f}]'
    ax2.text(0.02, 0.98, scaled_stats, transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle(f'Scaling Effect Demonstration - {dataset_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, f"scaling_effect_{dataset_name.replace('/', '_').replace('.', '_')}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Scaling effect plot saved to {output_path}")
    return output_path

def load_huggingface_dataset(dataset_path: str, max_samples_per_split: int = None):
    """Load numeric data from multiple HuggingFace dataset directories."""
    if not HF_DATASETS_AVAILABLE:
        raise ImportError("datasets library is required for HuggingFace dataset support. Install with: pip install datasets")
    
    print(f"Loading HuggingFace datasets from {dataset_path}")
    
    all_scaled_data = []
    total_samples = 0
    
    # Find all subdirectories that contain dataset files
    dataset_dirs = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            # Check if this directory contains dataset files
            has_arrow = any(f.endswith('.arrow') for f in os.listdir(item_path))
            has_state = os.path.exists(os.path.join(item_path, 'state.json'))
            if has_arrow or has_state:
                dataset_dirs.append(item_path)
    
    print(f"Found {len(dataset_dirs)} dataset directories")
    
    for dataset_dir in dataset_dirs:
        try:
            print(f"Loading dataset from {dataset_dir}")
            dataset = load_from_disk(dataset_dir)
            
            # Handle both single Dataset and DatasetDict
            if hasattr(dataset, 'keys'):
                # DatasetDict - process each split
                for split_name in dataset.keys():
                    split_data = dataset[split_name]
                    processed_samples = process_dataset_split(split_data, max_samples_per_split)
                    all_scaled_data.extend(processed_samples)
                    total_samples += sum(len(data) for data in processed_samples)
            else:
                # Single Dataset
                processed_samples = process_dataset_split(dataset, max_samples_per_split)
                all_scaled_data.extend(processed_samples)
                total_samples += sum(len(data) for data in processed_samples)
                
        except Exception as e:
            print(f"Warning: Failed to load dataset from {dataset_dir}: {e}")
            continue
    
    if not all_scaled_data:
        raise ValueError("No valid numeric data found in any HuggingFace datasets")
    
    pooled_scaled = np.concatenate(all_scaled_data)
    print(f"Total pooled scaled samples from HuggingFace datasets: {len(pooled_scaled)}")
    return pooled_scaled

def load_huggingface_dataset_with_plots(dataset_path: str, max_samples_per_split: int = None, scaling_plots_dir: str = None):
    """Load numeric data from multiple HuggingFace dataset directories with scaling plots."""
    if not HF_DATASETS_AVAILABLE:
        raise ImportError("datasets library is required for HuggingFace dataset support. Install with: pip install datasets")
    
    print(f"Loading HuggingFace datasets from {dataset_path}")
    
    all_scaled_data = []
    total_samples = 0
    
    # Find all subdirectories that contain dataset files
    dataset_dirs = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            # Check if this directory contains dataset files
            has_arrow = any(f.endswith('.arrow') for f in os.listdir(item_path))
            has_state = os.path.exists(os.path.join(item_path, 'state.json'))
            if has_arrow or has_state:
                dataset_dirs.append(item_path)
    
    print(f"Found {len(dataset_dirs)} dataset directories")
    
    for dataset_dir in dataset_dirs:
        try:
            print(f"Loading dataset from {dataset_dir}")
            dataset = load_from_disk(dataset_dir)
            dataset_name = os.path.basename(dataset_dir)
            
            # Handle both single Dataset and DatasetDict
            if hasattr(dataset, 'keys'):
                # DatasetDict - process each split
                for split_name in dataset.keys():
                    split_data = dataset[split_name]
                    processed_samples = process_dataset_split_with_plots(
                        split_data, max_samples_per_split, 
                        f"{dataset_name}_{split_name}", scaling_plots_dir
                    )
                    all_scaled_data.extend(processed_samples)
                    total_samples += sum(len(data) for data in processed_samples)
            else:
                # Single Dataset
                processed_samples = process_dataset_split_with_plots(
                    dataset, max_samples_per_split, 
                    dataset_name, scaling_plots_dir
                )
                all_scaled_data.extend(processed_samples)
                total_samples += sum(len(data) for data in processed_samples)
                
        except Exception as e:
            print(f"Warning: Failed to load dataset from {dataset_dir}: {e}")
            continue
    
    if not all_scaled_data:
        raise ValueError("No valid numeric data found in any HuggingFace datasets")
    
    pooled_scaled = np.concatenate(all_scaled_data)
    print(f"Total pooled scaled samples from HuggingFace datasets: {len(pooled_scaled)}")
    return pooled_scaled

def process_dataset_split(split_data, max_samples_per_split):
    """Process a single dataset split and extract numeric data."""
    processed_data = []
    
    split_limit = len(split_data)
    if max_samples_per_split is not None:
        split_limit = min(split_limit, max_samples_per_split)
    
    for i, row in enumerate(split_data.select(range(split_limit))):
        if i >= split_limit:
            break
            
        # Extract numeric values from all fields in the row
        for field_name, field_value in row.items():
            try:
                if isinstance(field_value, (list, tuple)):
                    numeric_data = np.array(field_value, dtype=float)
                    if numeric_data.size > 0 and not np.all(np.isnan(numeric_data)):
                        clean_data = numeric_data[~np.isnan(numeric_data)]
                        if len(clean_data) > 0:
                            s = np.mean(np.abs(clean_data))
                            if s > 0:
                                processed_data.append(clean_data / s)
                elif isinstance(field_value, (int, float)):
                    if not np.isnan(float(field_value)):
                        val = float(field_value)
                        if val != 0:
                            processed_data.append(np.array([val / abs(val)]))
                elif isinstance(field_value, str):
                    try:
                        val = float(field_value)
                        if not np.isnan(val) and val != 0:
                            processed_data.append(np.array([val / abs(val)]))
                    except ValueError:
                        continue
            except (ValueError, TypeError, OverflowError):
                continue
    
    return processed_data

def process_dataset_split_with_plots(split_data, max_samples_per_split, dataset_name, scaling_plots_dir):
    """Process a single dataset split and extract numeric data with scaling plots."""
    processed_data = []
    field_data_for_plots = {}  # Store original data for plotting
    
    split_limit = len(split_data)
    if max_samples_per_split is not None:
        split_limit = min(split_limit, max_samples_per_split)
    
    for i, row in enumerate(split_data.select(range(split_limit))):
        if i >= split_limit:
            break
            
        # Extract numeric values from all fields in the row
        for field_name, field_value in row.items():
            try:
                if isinstance(field_value, (list, tuple)):
                    numeric_data = np.array(field_value, dtype=float)
                    if numeric_data.size > 0 and not np.all(np.isnan(numeric_data)):
                        clean_data = numeric_data[~np.isnan(numeric_data)]
                        if len(clean_data) > 0:
                            s = np.mean(np.abs(clean_data))
                            if s > 0:
                                scaled_data = clean_data / s
                                processed_data.append(scaled_data)
                                
                                # Store for plotting (only first occurrence of each field)
                                field_key = f"{dataset_name}_{field_name}"
                                if field_key not in field_data_for_plots and len(clean_data) >= 128:
                                    field_data_for_plots[field_key] = (clean_data, scaled_data)
                                
                elif isinstance(field_value, (int, float)):
                    if not np.isnan(float(field_value)):
                        val = float(field_value)
                        if val != 0:
                            processed_data.append(np.array([val / abs(val)]))
                elif isinstance(field_value, str):
                    try:
                        val = float(field_value)
                        if not np.isnan(val) and val != 0:
                            processed_data.append(np.array([val / abs(val)]))
                    except ValueError:
                        continue
            except (ValueError, TypeError, OverflowError):
                continue
    
    # Create scaling plots for fields with sufficient data
    if scaling_plots_dir:
        for field_key, (original_data, scaled_data) in field_data_for_plots.items():
            plot_scaling_effect(original_data, scaled_data, field_key, scaling_plots_dir)
    
    return processed_data

def load_and_scale_all_data(datasets_dir: str, use_huggingface: bool = False, max_samples_per_split: int = None):
    """Load and scale all columns from all CSV files or HuggingFace dataset."""
    if use_huggingface:
        return load_huggingface_dataset(datasets_dir, max_samples_per_split)
    
    # Original CSV loading logic
    all_scaled_data = []
    csv_files = glob.glob(os.path.join(datasets_dir, "**", "*.csv"), recursive=True)
    
    print(f"Found {len(csv_files)} CSV files")
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            for col_idx in range(df.shape[1]):
                try:
                    col_data = df.iloc[:, col_idx].to_numpy(dtype=float)
                    col_data = col_data[~np.isnan(col_data)]
                    
                    if len(col_data) > 0:
                        s = np.mean(np.abs(col_data))
                        if s > 0:
                            all_scaled_data.append(col_data / s)
                except (ValueError, TypeError):
                    continue
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if not all_scaled_data:
        raise ValueError("No valid numeric data found")
    
    pooled_scaled = np.concatenate(all_scaled_data)
    print(f"Total pooled scaled samples: {len(pooled_scaled)}")
    return pooled_scaled

def load_and_scale_all_data_with_plots(datasets_dir: str, use_huggingface: bool = False, max_samples_per_split: int = None):
    """
    Modified version that creates scaling effect plots for each dataset.
    Load and scale all columns from all CSV files or HuggingFace dataset.
    """
    # Create output directory for scaling plots
    scaling_plots_dir = os.path.join("scaling_plots")
    os.makedirs(scaling_plots_dir, exist_ok=True)
    
    all_scaled_data = []
    
    if use_huggingface:
        all_scaled_data = load_huggingface_dataset_with_plots(datasets_dir, max_samples_per_split, scaling_plots_dir)
    else:
        # Original CSV loading logic with plots
        csv_files = glob.glob(os.path.join(datasets_dir, "**", "*.csv"), recursive=True)
        print(f"Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                dataset_name = os.path.relpath(csv_file, datasets_dir)
                
                for col_idx in range(df.shape[1]):
                    try:
                        col_data = df.iloc[:, col_idx].to_numpy(dtype=float)
                        col_data = col_data[~np.isnan(col_data)]
                        
                        if len(col_data) > 0:
                            s = np.mean(np.abs(col_data))
                            if s > 0:
                                scaled_data = col_data / s
                                all_scaled_data.append(scaled_data)
                                
                                # Create scaling effect plot for this column
                                col_dataset_name = f"{dataset_name}_col{col_idx}"
                                plot_scaling_effect(col_data, scaled_data, col_dataset_name, scaling_plots_dir)
                                
                    except (ValueError, TypeError):
                        continue
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
    
    if not all_scaled_data:
        raise ValueError("No valid numeric data found")
    
    pooled_scaled = np.concatenate(all_scaled_data)
    print(f"Total pooled scaled samples: {len(pooled_scaled)}")
    return pooled_scaled

def create_self_information_patches(scaled, per_sample_entropy, min_size=2, max_size=64, 
                                  high_info_threshold=None, adaptive_threshold=True):
    """
    Create variable-size patches based on self-information (entropy).
    
    Following the BLT paper approach:
    - Keep growing patches until encountering high self-information samples
    - Self-information is the per-sample entropy (negative log probability)
    - High self-information indicates complexity/unpredictability
    
    Args:
        scaled: Scaled signal data
        per_sample_entropy: Per-sample self-information (entropy) values
        min_size: Minimum patch size
        max_size: Maximum patch size
        high_info_threshold: Threshold for high self-information (if None, use adaptive)
        adaptive_threshold: Whether to use adaptive thresholding
    """
    patches = []
    i = 0
    
    # Determine threshold for high self-information
    if high_info_threshold is None:
        if adaptive_threshold:
            # Use percentile-based adaptive threshold (similar to BLT)
            high_info_threshold = np.percentile(per_sample_entropy, 75)  # 75th percentile
        else:
            # Use mean + std as threshold
            high_info_threshold = np.mean(per_sample_entropy) + np.std(per_sample_entropy)
    
    print(f"Using high self-information threshold: {high_info_threshold:.4f}")
    print(f"Self-information stats - mean: {np.mean(per_sample_entropy):.4f}, "
          f"std: {np.std(per_sample_entropy):.4f}, "
          f"75th percentile: {np.percentile(per_sample_entropy, 75):.4f}")
    
    while i < len(scaled):
        patch_start = i
        current_size = 1
        
        # Start with minimum size
        while current_size < min_size and i + current_size < len(scaled):
            current_size += 1
        
        # Keep growing until we hit high self-information or max size
        while (current_size < max_size and 
               i + current_size < len(scaled)):
            
            # Check if current sample has high self-information
            current_sample_info = per_sample_entropy[i + current_size - 1]
            
            # If we encounter high self-information, stop growing
            if current_sample_info > high_info_threshold:
                break
            
            # Look ahead to next sample
            if i + current_size < len(scaled):
                next_sample_info = per_sample_entropy[i + current_size]
                
                # If next sample has high self-information, include it and stop
                if next_sample_info > high_info_threshold:
                    current_size += 1
                    break
                
                # Otherwise, continue growing
                current_size += 1
            else:
                break
        
        # Ensure we don't exceed bounds
        patch_end = min(i + current_size, len(scaled))
        actual_size = patch_end - patch_start
        
        if actual_size <= 0:
            break
        
        # Create patch
        patch_entropy = per_sample_entropy[patch_start:patch_end]
        patch_scaled = scaled[patch_start:patch_end]
        
        # Calculate patch statistics
        mean_info = np.mean(patch_entropy)
        max_info = np.max(patch_entropy)
        info_variance = np.var(patch_entropy)
        
        patches.append({
            'start': patch_start,
            'end': patch_end,
            'size': actual_size,
            'scaled': patch_scaled,
            'entropy': patch_entropy,
            'mean_self_info': mean_info,
            'max_self_info': max_info,
            'self_info_variance': info_variance,
            'has_high_info': max_info > high_info_threshold,
            'boundary_info': per_sample_entropy[patch_end - 1] if patch_end > patch_start else 0.0
        })
        
        i = patch_end
    
    return patches, high_info_threshold

def print_patch_stats(patches, high_info_threshold):
    """Print comprehensive patch statistics for self-information based patching."""
    sizes = [p['size'] for p in patches]
    mean_infos = [p['mean_self_info'] for p in patches]
    max_infos = [p['max_self_info'] for p in patches]
    high_info_patches = [p for p in patches if p['has_high_info']]
    
    print(f"\n=== Self-Information Based Patch Statistics ===")
    print(f"High self-information threshold: {high_info_threshold:.4f}")
    print(f"Total patches: {len(patches)}")
    print(f"Patches with high self-information: {len(high_info_patches)} ({len(high_info_patches)/len(patches)*100:.1f}%)")
    
    print(f"\n--- Size Statistics ---")
    print(f"Size - mean: {np.mean(sizes):.2f}, median: {np.median(sizes):.1f}, range: {np.min(sizes)}-{np.max(sizes)}")
    
    print(f"\n--- Self-Information Statistics ---")
    print(f"Mean self-info per patch - mean: {np.mean(mean_infos):.4f}, range: {np.min(mean_infos):.4f}-{np.max(mean_infos):.4f}")
    print(f"Max self-info per patch - mean: {np.mean(max_infos):.4f}, range: {np.min(max_infos):.4f}-{np.max(max_infos):.4f}")
    
    # Size distribution
    unique_sizes = sorted(set(sizes))[:10]
    print(f"\n--- Size Distribution (top 10) ---")
    print(f"Sizes: {', '.join(f'{s}({sum(1 for x in sizes if x == s)})' for s in unique_sizes)}")
    
    # High vs low information patch sizes
    high_info_sizes = [p['size'] for p in patches if p['has_high_info']]
    low_info_sizes = [p['size'] for p in patches if not p['has_high_info']]
    
    if high_info_sizes and low_info_sizes:
        print(f"\n--- High vs Low Information Patches ---")
        print(f"High-info patches - mean size: {np.mean(high_info_sizes):.2f}, count: {len(high_info_sizes)}")
        print(f"Low-info patches - mean size: {np.mean(low_info_sizes):.2f}, count: {len(low_info_sizes)}")

def plot_signal_overview(scaled, per_sample_entropy, patches, portion_length, title_suffix, output_path, high_info_threshold):
    """Plot 3 random signal portions with patch boundaries and self-information threshold."""
    if len(scaled) < portion_length:
        raise ValueError(f"Signal too short ({len(scaled)}) for portion length {portion_length}")
    
    np.random.seed(42)
    max_start = len(scaled) - portion_length
    starts = np.sort(np.random.choice(max_start + 1, size=3, replace=False))
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Calculate global y-limits for all portions
    all_scaled_portions = []
    all_entropy_portions = []
    for start in starts:
        end = start + portion_length
        all_scaled_portions.append(scaled[start:end])
        all_entropy_portions.append(per_sample_entropy[start:end])
    
    scaled_ylim = (np.min(np.concatenate(all_scaled_portions)), 
                   np.max(np.concatenate(all_scaled_portions)))
    entropy_ylim = (np.min(np.concatenate(all_entropy_portions)), 
                    np.max(np.concatenate(all_entropy_portions)))
    
    for i, start in enumerate(starts):
        end = start + portion_length
        indices = np.arange(start, end)
        
        ax1 = axes[i]
        # Plot data with markers and dash-dot line style
        ax1.plot(indices, scaled[start:end], '-.o', color='C0', alpha=0.8, markersize=3)
        ax1.set_ylabel("Scaled value", color="C0")
        ax1.tick_params(axis="y", labelcolor="C0")
        ax1.set_ylim(scaled_ylim)
        
        # Add patch boundaries as dotted vertical lines
        relevant_patches = [p for p in patches if p['start'] < end and p['end'] > start]
        for patch in relevant_patches:
            patch_start = max(patch['start'], start)
            patch_end = min(patch['end'], end)
            
            # Color-code patch boundaries based on whether they contain high self-info
            boundary_color = 'red' if patch['has_high_info'] else 'black'
            boundary_alpha = 0.8 if patch['has_high_info'] else 0.5
            
            ax1.axvline(patch_start, color=boundary_color, linestyle=':', alpha=boundary_alpha, linewidth=1.5)
            if patch_end < end:
                ax1.axvline(patch_end, color=boundary_color, linestyle=':', alpha=boundary_alpha, linewidth=1.5)
        
        ax2 = ax1.twinx()
        ax2.plot(indices, per_sample_entropy[start:end], 'C1', alpha=0.8, linewidth=1.5)
        
        # Add horizontal line for high self-information threshold
        ax2.axhline(high_info_threshold, color='red', linestyle='--', alpha=0.7, linewidth=2, 
                   label=f'High Self-Info Threshold ({high_info_threshold:.3f})')
        
        ax2.set_ylabel("Self-Information (bits)", color="C1")
        ax2.tick_params(axis="y", labelcolor="C1")
        ax2.set_ylim(entropy_ylim)
        
        if i == 0:
            ax2.legend(loc='upper right')
        
        ax1.set_title(f"Signal portion {i+1}: samples {start}-{end-1}")
        
        if i == 2:
            ax1.set_xlabel("Sample index")
    
    plt.suptitle(f"Self-Information Based Patching {title_suffix}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Overview plot saved to {output_path}")

def plot_probability_distribution(quantization_info, output_path, title_suffix):
    """Plot the probability distribution of quantized values."""
    edges, probs, counts = quantization_info
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Probability distribution
    bin_centers = np.concatenate([[-np.inf], (edges[:-1] + edges[1:]) / 2, [np.inf]])
    non_zero_probs = probs[probs > 0]
    non_zero_bins = np.arange(len(probs))[probs > 0]
    
    ax1.bar(non_zero_bins, non_zero_probs, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Quantization Bin')
    ax1.set_ylabel('Probability')
    ax1.set_title(f'Quantized Value Probability Distribution\n{len(non_zero_bins)} non-zero bins out of {len(probs)} total')
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Log probability for better visualization
    log_probs = np.log10(probs + np.finfo(float).eps)
    ax2.bar(range(len(probs)), log_probs, alpha=0.7, color='darkgreen')
    ax2.set_xlabel('Quantization Bin')
    ax2.set_ylabel('Log10(Probability)')
    ax2.set_title('Log Probability Distribution\n(All bins including zeros)')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f"Global Vocabulary Distribution {title_suffix}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Probability distribution plot saved to {output_path}")

def find_entropy_regions(scaled, per_sample_entropy, window_size=50):
    """Find regions with consistently low and high entropy."""
    # Calculate windowed mean entropy
    windowed_entropy = np.array([
        np.mean(per_sample_entropy[max(0, i-window_size//2):min(len(per_sample_entropy), i+window_size//2)])
        for i in range(len(per_sample_entropy))
    ])
    
    # Find regions with consistent low/high entropy
    low_threshold = np.percentile(windowed_entropy, 25)
    high_threshold = np.percentile(windowed_entropy, 75)
    
    # Find the best low and high entropy regions
    low_entropy_candidates = []
    high_entropy_candidates = []
    
    i = 0
    while i < len(windowed_entropy) - window_size:
        if np.all(windowed_entropy[i:i+window_size] < low_threshold):
            mean_entropy = np.mean(per_sample_entropy[i:i+window_size])
            low_entropy_candidates.append((i, i+window_size, mean_entropy))
            i += window_size
        elif np.all(windowed_entropy[i:i+window_size] > high_threshold):
            mean_entropy = np.mean(per_sample_entropy[i:i+window_size])
            high_entropy_candidates.append((i, i+window_size, mean_entropy))
            i += window_size
        else:
            i += 1
    
    # Select best candidates
    low_region = min(low_entropy_candidates, key=lambda x: x[2]) if low_entropy_candidates else None
    high_region = max(high_entropy_candidates, key=lambda x: x[2]) if high_entropy_candidates else None
    
    return low_region, high_region, low_threshold, high_threshold

def plot_entropy_regions(scaled, per_sample_entropy, output_path, title_suffix, window_size=50):
    """Plot zoomed-in views of low and high entropy regions."""
    low_region, high_region, low_thresh, high_thresh = find_entropy_regions(
        scaled, per_sample_entropy, window_size
    )
    
    if not low_region or not high_region:
        print("Warning: Could not find suitable low/high entropy regions for detailed plots")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Low entropy region
    low_start, low_end, low_mean = low_region
    low_indices = np.arange(low_start, low_end)
    
    # Left: Low entropy signal
    axes[0, 0].plot(low_indices, scaled[low_start:low_end], '.-', color='C0', markersize=4)
    axes[0, 0].set_title(f'Low Entropy Region (samples {low_start}-{low_end-1})\nMean entropy: {low_mean:.3f} bits')
    axes[0, 0].set_ylabel('Scaled Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Right: Low entropy self-information
    axes[0, 1].plot(low_indices, per_sample_entropy[low_start:low_end], '.-', color='C1', markersize=4)
    axes[0, 1].axhline(low_thresh, color='blue', linestyle='--', alpha=0.7, 
                      label=f'25th percentile ({low_thresh:.3f})')
    axes[0, 1].set_title('Self-Information in Low Entropy Region')
    axes[0, 1].set_ylabel('Self-Information (bits)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # High entropy region
    high_start, high_end, high_mean = high_region
    high_indices = np.arange(high_start, high_end)
    
    # Left: High entropy signal
    axes[1, 0].plot(high_indices, scaled[high_start:high_end], '.-', color='C0', markersize=4)
    axes[1, 0].set_title(f'High Entropy Region (samples {high_start}-{high_end-1})\nMean entropy: {high_mean:.3f} bits')
    axes[1, 0].set_ylabel('Scaled Value')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Right: High entropy self-information
    axes[1, 1].plot(high_indices, per_sample_entropy[high_start:high_end], '.-', color='C1', markersize=4)
    axes[1, 1].axhline(high_thresh, color='red', linestyle='--', alpha=0.7, 
                      label=f'75th percentile ({high_thresh:.3f})')
    axes[1, 1].set_title('Self-Information in High Entropy Region')
    axes[1, 1].set_ylabel('Self-Information (bits)')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f"Low vs High Entropy Regions Analysis {title_suffix}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Entropy regions plot saved to {output_path}")
    
    # Print region statistics
    print(f"\n=== Entropy Regions Analysis ===")
    print(f"Low entropy region: samples {low_start}-{low_end-1}, mean entropy: {low_mean:.4f} bits")
    print(f"High entropy region: samples {high_start}-{high_end-1}, mean entropy: {high_mean:.4f} bits")
    print(f"Entropy difference: {high_mean - low_mean:.4f} bits")

def main():
    parser = argparse.ArgumentParser(description="Self-information based entropy analysis with variable-size patches")
    parser.add_argument("--csv_path", default="./datasets/ETT-small/ETTm2.csv", help="CSV file path")
    parser.add_argument("--column", type=int, default=1, help="Column index (0-based)")
    parser.add_argument("--bins", type=int, default=4096, help="Quantization bins")
    parser.add_argument("--use_global", action="store_true", help="Use global vocabulary")
    parser.add_argument("--use_contextual", action="store_true", help="Use contextual vocabulary")
    parser.add_argument("--context_length", type=int, default=1024, help="Context window length")
    
    # Global vocabulary options
    parser.add_argument("--global_data_path", default="./datasets/ETT-small", 
                       help="Path to directory containing CSV files or HuggingFace dataset")
    parser.add_argument("--use_huggingface", action="store_true", 
                       help="Use HuggingFace dataset format instead of CSV files for global vocabulary")
    parser.add_argument("--max_samples_per_split", type=int, default=None,
                       help="Maximum samples to load per dataset split (for large HuggingFace datasets)")
    
    # Self-information based patch parameters
    parser.add_argument("--min_patch_size", type=int, default=2, help="Minimum patch size")
    parser.add_argument("--max_patch_size", type=int, default=64, help="Maximum patch size")
    parser.add_argument("--high_info_threshold", type=float, default=None, 
                       help="Threshold for high self-information (if None, use adaptive)")
    parser.add_argument("--adaptive_threshold", action="store_true", default=True,
                       help="Use adaptive thresholding (75th percentile)")
    parser.add_argument("--plot_portion_length", type=int, default=256, help="Plot portion length")
    
    # New argument for scaling plots
    parser.add_argument("--create_scaling_plots", action="store_true", 
                       help="Create scaling effect plots for each dataset")

    args = parser.parse_args()

    # Load data
    print(f"Loading {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    if args.column >= df.shape[1]:
        raise IndexError(f"Column {args.column} not found (file has {df.shape[1]} columns)")
    
    x = df.iloc[:, args.column].to_numpy(dtype=float)
    print(f"Loaded {len(x)} samples")

    # Determine vocabulary mode and compute entropy
    quantization_info = None
    if args.use_global and args.use_contextual:
        raise ValueError("Cannot use both global and contextual modes")
    
    if args.use_global:
        if args.use_huggingface:
            print(f"Using global vocabulary from HuggingFace dataset at {args.global_data_path}...")
            if args.max_samples_per_split:
                print(f"Limiting to {args.max_samples_per_split} samples per split")
        else:
            print(f"Using global vocabulary from CSV files in {args.global_data_path}...")
        
        # Use the new function that creates scaling plots
        if args.create_scaling_plots:
            pooled_data = load_and_scale_all_data_with_plots(
                args.global_data_path, 
                use_huggingface=args.use_huggingface,
                max_samples_per_split=args.max_samples_per_split
            )
        else:
            pooled_data = load_and_scale_all_data(
                args.global_data_path, 
                use_huggingface=args.use_huggingface,
                max_samples_per_split=args.max_samples_per_split
            )
            
        scaled, entropy, quantization_info = compute_entropy_and_scaled(x, args.bins, pooled_scaled_data=pooled_data)
        vocab_type = "global_hf" if args.use_huggingface else "global"
    elif args.use_contextual:
        print(f"Using contextual vocabulary (length {args.context_length})...")
        scaled, entropy, quantization_info = compute_entropy_and_scaled(x, args.bins, context_length=args.context_length)
        vocab_type = f"contextual_ctx{args.context_length}"
    else:
        print("Using local vocabulary...")
        scaled, entropy, quantization_info = compute_entropy_and_scaled(x, args.bins)
        vocab_type = "local"

    # Create scaling plot for the main dataset being analyzed
    if args.create_scaling_plots:
        main_dataset_name = os.path.splitext(os.path.basename(args.csv_path))[0]
        scaling_plots_dir = "scaling_plots"
        os.makedirs(scaling_plots_dir, exist_ok=True)
        
        # Get original data for the specific column
        original_data = df.iloc[:, args.column].to_numpy(dtype=float)
        original_data = original_data[~np.isnan(original_data)]
        
        plot_scaling_effect(original_data, scaled, f"{main_dataset_name}_col{args.column}", scaling_plots_dir)

    print(f"Self-information (entropy) stats - mean: {np.mean(entropy):.4f}, "
          f"std: {np.std(entropy):.4f}, range: {np.min(entropy):.4f}-{np.max(entropy):.4f}")

    # Create patches using self-information based approach
    print("\n=== Creating Self-Information Based Patches ===")
    patches, threshold_used = create_self_information_patches(
        scaled, entropy, 
        min_size=args.min_patch_size,
        max_size=args.max_patch_size,
        high_info_threshold=args.high_info_threshold,
        adaptive_threshold=args.adaptive_threshold
    )
    
    print_patch_stats(patches, threshold_used)

    # Setup output
    dataset_name = os.path.splitext(os.path.basename(args.csv_path))[0]
    output_dir = f"si_patches/{dataset_name}/col_{args.column}"
    os.makedirs(output_dir, exist_ok=True)
    
    title_suffix = f"({vocab_type}) - col {args.column}, B={args.bins}, Self-Info Patching"
    filename_suffix = f"{vocab_type}_col{args.column}_bins{args.bins}_selfinfo"
    
    # Generate overview plot with self-information based patches
    overview_plot_path = os.path.join(output_dir, f"sample_{filename_suffix}.png")
    plot_signal_overview(scaled, entropy, patches, args.plot_portion_length, 
                        title_suffix, overview_plot_path, threshold_used)
    
    # Plot probability distribution (only for global vocabulary)
    if args.use_global and quantization_info is not None:
        prob_dist_path = os.path.join(output_dir, f"probability_distribution_{filename_suffix}.png")
        plot_probability_distribution(quantization_info, prob_dist_path, title_suffix)
    
    # Plot entropy regions analysis
    entropy_regions_path = os.path.join(output_dir, f"entropy_regions_{filename_suffix}.png")
    plot_entropy_regions(scaled, entropy, entropy_regions_path, title_suffix)
    
    print(f"\nMean self-information: {np.mean(entropy):.4f} bits")
    print(f"Patches created: {len(patches)}")
    print(f"Average patch size: {np.mean([p['size'] for p in patches]):.2f}")
    
    if args.create_scaling_plots:
        print(f"\nScaling effect plots saved to: scaling_plots/")

if __name__ == "__main__":
    main()
