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
                               monotonic_mode=False):
    """
    Detect patch boundaries based on entropy spikes.
    
    Args:
        entropies: Array of entropy values
        threshold_multiplier: Multiplier for mean entropy to determine spike threshold
        min_patch_size: Minimum size of a patch
        monotonic_mode: If True, use Approximate Monotonic Constraint (diff-based)
    
    Returns:
        patch_boundaries: List of patch boundary indices
        threshold: The threshold value used for spike detection
    """
    if monotonic_mode:
        # Approximate Monotonic Constraint: use differences between consecutive entropy values
        entropy_diffs = np.diff(entropies)
        # Only consider positive differences (increases in entropy)
        positive_diffs = entropy_diffs[entropy_diffs > 0]
        
        if len(positive_diffs) > 0:
            mean_diff = np.mean(positive_diffs)
            std_diff = np.std(positive_diffs)
            threshold = mean_diff + threshold_multiplier * std_diff
            
            # Find significant increases
            spike_indices = []
            for i, diff in enumerate(entropy_diffs):
                if diff > threshold:
                    spike_indices.append(i + 1)  # +1 because diff[i] is between point i and i+1
        else:
            threshold = 0
            spike_indices = []
        
        spikes = np.array(spike_indices)
    else:
        # Original method: absolute entropy values
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





def main():
    parser = argparse.ArgumentParser(description='Autoregressive prediction with entropy-based patches for 3 random chunks.')
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
                        help='Multiplier for mean+std to determine spike threshold')
    parser.add_argument('--min_patch_size', type=int, default=1,
                        help='Minimum size of a patch')
    parser.add_argument('--monotonic_mode', action='store_true',
                        help='Use Approximate Monotonic Constraint (threshold on differences)')
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
    all_entropies = []
    all_patch_boundaries = []
    all_thresholds = []

    for i, start in enumerate(random_starts):
        print(f"\nProcessing chunk {i+1}/3 (start index {start})...")
        ground_truth = series[start:start+args.context_length]
        preds, entropies = autoregressive_predict_context_simplified(
            pipeline, ground_truth, dataset_mean)
        
        # Detect patches based on entropy spikes
        patch_boundaries, threshold = detect_patches_from_entropy(
            entropies, args.threshold_multiplier, args.min_patch_size, args.monotonic_mode
        )
        
        all_ground_truths.append(ground_truth)
        all_predictions.append(preds)
        all_entropies.append(entropies)
        all_patch_boundaries.append(patch_boundaries)
        all_thresholds.append(threshold)
        
        print(f"  Entropy-based patches: {len(patch_boundaries)-1} with threshold {threshold:.4f}")
        print(f"  Mode: {'Monotonic (entropy diff)' if args.monotonic_mode else 'Absolute entropy'}")
        print(f"  Patch boundaries: {patch_boundaries}")

    # Create results directory structure
    dataset_name = os.path.splitext(os.path.basename(args.csv_path))[0]
    results_dir = f"results_entropy/{dataset_name}/col_{args.col_num}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create plots for each chunk showing entropy-based patches
    for i in range(3):
        fig, ax = plt.subplots(1, 1, figsize=(15, 6))
        
        # Create twin axis for entropy
        ax_twin = ax.twinx()
        
        # Plot ground truth and predictions
        ax.plot(all_ground_truths[i], label="Ground Truth", linestyle='-', 
                linewidth=2, alpha=0.8, color='blue')
        ax.plot(all_predictions[i], label="Autoregressive Predictions", linestyle='--', 
                linewidth=2, alpha=0.8, color='green')
        
        # Plot entropy
        ax_twin.plot(all_entropies[i], label="Token Entropy", 
                     color='red', linewidth=1.5, alpha=0.7)
        
        # Plot threshold
        if args.monotonic_mode:
            ax_twin.axhline(y=all_thresholds[i], color='orange', linestyle=':', 
                           linewidth=2, alpha=0.8, label=f'Entropy Diff Threshold ({all_thresholds[i]:.3f})')
        else:
            ax_twin.axhline(y=all_thresholds[i], color='orange', linestyle=':', 
                           linewidth=2, alpha=0.8, label=f'Entropy Threshold ({all_thresholds[i]:.3f})')
        
        # Plot patch boundaries
        for boundary in all_patch_boundaries[i]:
            ax.axvline(x=boundary, color='purple', linestyle=':', 
                       linewidth=2, alpha=0.6)
        
        # Add patch boundary legend entry
        ax.axvline(x=-1, color='purple', linestyle=':', linewidth=2, 
                   alpha=0.6, label='Patch Boundaries')
        
        # Set labels and title
        patch_count = len(all_patch_boundaries[i]) - 1
        mode_str = "Monotonic Entropy-Diff" if args.monotonic_mode else "Absolute Entropy"
        ax.set_title(f"Chunk {i+1} - Entropy-based Patches ({mode_str}) - {patch_count} patches")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Value", color='black')
        ax_twin.set_ylabel("Token Entropy (bits)", color='red')
        ax_twin.tick_params(axis='y', labelcolor='red')
        
        # Add legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        mode_suffix = "_monotonic" if args.monotonic_mode else "_absolute"
        plot_path = os.path.join(results_dir, f"chunk_{i+1}_entropy_patches{mode_suffix}.pdf")
        plt.savefig(plot_path, format='pdf', bbox_inches='tight')
        print(f"Plot for chunk {i+1} saved to: {plot_path}")
        plt.show()

    # Calculate and print metrics
    print(f"\nPrediction Metrics ({'Monotonic Mode' if args.monotonic_mode else 'Absolute Mode'}):")
    for i in range(3):
        mae = np.mean(np.abs(all_ground_truths[i] - all_predictions[i]))
        mse = np.mean((all_ground_truths[i] - all_predictions[i]) ** 2)
        rmse = np.sqrt(mse)
        
        # Entropy method metrics
        avg_entropy = np.mean(all_entropies[i])
        patch_count = len(all_patch_boundaries[i]) - 1
        avg_patch_size = args.context_length / patch_count if patch_count > 0 else args.context_length
        
        print(f"Chunk {i+1}: MAE={mae:.4f}, RMSE={rmse:.4f}")
        print(f"         Entropy: Avg={avg_entropy:.4f} bits, Patches={patch_count}, Avg Patch Size={avg_patch_size:.1f}, Threshold={all_thresholds[i]:.4f}")


if __name__ == '__main__':
    main()
