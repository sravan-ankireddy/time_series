#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import tempfile
import os
import glob

def load_and_scale_all_data(datasets_dir: str):
    """Load and individually scale ALL columns from all CSV files for vocabulary creation."""
    all_scaled_data = []
    csv_files = glob.glob(os.path.join(datasets_dir, "**", "*.csv"), recursive=True)
    
    print(f"Found {len(csv_files)} CSV files in {datasets_dir}")
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            for col_idx in range(df.shape[1]):
                try:
                    col_data = df.iloc[:, col_idx].to_numpy(dtype=float)
                    col_data = col_data[~np.isnan(col_data)]  # remove NaN
                    
                    if len(col_data) > 0:
                        # Scale each time series individually (mean-absolute scaling)
                        s = np.mean(np.abs(col_data))
                        if s > 0:
                            scaled_col = col_data / s
                            all_scaled_data.append(scaled_col)
                            print(f"  Scaled {len(col_data)} samples from {csv_file} col {col_idx}")
                
                except (ValueError, TypeError):
                    continue  # Skip non-numeric columns
        except Exception as e:
            print(f"  Error loading {csv_file}: {e}")
    
    if not all_scaled_data:
        raise ValueError("No valid numeric data found")
    
    # Pool all individually-scaled time series for vocabulary
    pooled_scaled = np.concatenate(all_scaled_data)
    print(f"Total pooled scaled samples: {len(pooled_scaled)}")
    return pooled_scaled

def compute_scaled_and_entropy(x: np.ndarray, B: int, pooled_scaled_data: np.ndarray = None):
    """Compute scaled signal and per-sample entropy using global vocabulary or local statistics."""
    
    # Scale the input signal
    s = np.mean(np.abs(x))
    if s == 0:
        raise ValueError("All-zero signal")
    scaled = x / s

    # Determine quantization bounds
    if pooled_scaled_data is not None:
        # Use full range from pooled scaled data
        c_min, c_max = pooled_scaled_data.min(), pooled_scaled_data.max()
        print(f"Global vocab range: [{c_min:.4f}, {c_max:.4f}]")
        prob_data = pooled_scaled_data
    else:
        # Fallback to local statistics
        c_min, c_max = scaled.min(), scaled.max()
        prob_data = scaled

    # Create uniform quantization bins
    edges = np.linspace(c_min, c_max, B + 1)[1:-1]  # B-1 edges for B bins
    bins = np.digitize(scaled, edges, right=False)
    bins = np.clip(bins, 0, B-1)  # Ensure valid range

    # Compute probabilities from appropriate data
    prob_bins = np.digitize(prob_data, edges, right=False)
    prob_bins = np.clip(prob_bins, 0, B-1)
    counts = np.bincount(prob_bins, minlength=B)
    probs = counts / len(prob_data)
    
    # Per-sample entropy (self-information in bits)
    eps = np.finfo(float).eps
    per_sample_entropy = -np.log2(probs[bins] + eps)
    
    # Overall entropy of the distribution
    overall_entropy = -np.sum(probs * np.log2(probs + eps))
    print(f"Overall entropy: {overall_entropy:.4f} bits")

    return scaled, per_sample_entropy

def main():
    parser = argparse.ArgumentParser(
        description="Compute mean-scaled signal and entropy from global vocabulary."
    )
    parser.add_argument("--csv_path", type=str, default="./datasets/ETT-small/ETTm2.csv", help="path to your CSV file")
    parser.add_argument("--column",   type=int, default=1,
                        help="which column index (0-based) to load")
    parser.add_argument("--bins",   type=int, default=4096,
                        help="number of quantization bins (default: 4096)")
    args = parser.parse_args()

    # Load and scale all dataset columns for vocabulary creation
    print("Loading and scaling all datasets for vocabulary...")
    pooled_scaled_data = load_and_scale_all_data("./datasets/ETT-small")

    # Load specific file for plotting
    print(f"\nLoading file for plotting: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    if args.column < 0 or args.column >= df.shape[1]:
        raise IndexError(f"Column {args.column} not found (file has {df.shape[1]} columns)")
    
    x = df.iloc[:, args.column].to_numpy(dtype=float)
    print(f"Loaded {len(x)} samples for analysis")

    # Compute entropy using global vocabulary
    scaled, per_sample_entropy = compute_scaled_and_entropy(x, args.bins, pooled_scaled_data)

    # randomly select 3 patches of length 256 for visualization
    patch_length = 256
    if len(scaled) < patch_length:
        raise ValueError(f"Signal too short ({len(scaled)} samples) for patch length {patch_length}")
    
    # randomly select starting indices for 3 patches
    np.random.seed(42)  # for reproducibility
    max_start = len(scaled) - patch_length
    patch_starts = np.random.choice(max_start + 1, size=3, replace=False)
    patch_starts = np.sort(patch_starts)  # sort for better visualization
    
    # plot 3 subplots for the patches
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    for i, start_idx in enumerate(patch_starts):
        end_idx = start_idx + patch_length
        patch_scaled = scaled[start_idx:end_idx]
        patch_entropy = per_sample_entropy[start_idx:end_idx]
        patch_indices = np.arange(start_idx, end_idx)
        
        ax1 = axes[i]
        ax1.plot(patch_indices, patch_scaled, linewidth=1.2, color="C0")
        ax1.set_ylabel("Scaled value", color="C0")
        ax1.tick_params(axis="y", labelcolor="C0")
        
        ax2 = ax1.twinx()
        ax2.plot(patch_indices, patch_entropy, linewidth=1.2, color="C1")
        ax2.set_ylabel("Entropy (bits)", color="C1")
        ax2.tick_params(axis="y", labelcolor="C1")
        
        ax1.set_title(f"Sample patch {i+1}")
        
        if i == 2:  # only add x-label to bottom subplot
            ax1.set_xlabel("Sample index")

    plt.suptitle(f"Per-sample Entropy Analysis - col {args.column}, B={args.bins}")
    fig.tight_layout()

    # save to temp/ folder
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    plot_filename = f"entropy_analysis_col{args.column}_bins{args.bins}.png"
    plot_path = os.path.join(temp_dir, plot_filename)
    fig.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    print(f"Mean per-sample entropy: {np.mean(per_sample_entropy):.4f} bits")

if __name__ == "__main__":
    main()
