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
    """Simplified version that extracts probabilities from quantile predictions."""
    predictions = []
    self_informations = []
    context_length = len(ground_truth_context)
    
    for i in tqdm(range(context_length), desc="Predicting samples"):
        if i == 0:
            # Use dataset mean for first prediction context
            context = torch.tensor([dataset_mean], dtype=torch.float32)
        else:
            # Use ground truth values from 0 to i-1 as context
            context = torch.tensor(ground_truth_context[:i], dtype=torch.float32)
        
        # Get quantile predictions with more quantiles to estimate distribution
        quantiles, mean = pipeline.predict_quantiles(
            context=context,
            prediction_length=1,
            quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        )

        pred = mean[0, 0].item()
        predictions.append(pred)
        
        # Estimate self-information from quantile spread
        # Higher spread = higher uncertainty = higher self-information
        q_values = quantiles[0, 0, :].cpu().numpy()
        spread = np.std(q_values)
        
        # Approximate self-information based on prediction uncertainty
        # This is a heuristic since we don't have direct access to token probabilities
        self_info = np.log2(1 + spread)  # Heuristic mapping
        self_informations.append(self_info)
    
    return np.array(predictions), np.array(self_informations)


def main():
    parser = argparse.ArgumentParser(description='Autoregressive prediction with self-information on context itself for 3 random chunks.')
    parser.add_argument('--csv_path', type=str, 
                        default='../datasets/time-moe-eval/ETT-small/ETTm2.csv',
                        help='Path to the CSV file containing the time series data')
    parser.add_argument('--col_num', type=int, default=1, 
                        help='Column number (0-indexed) to process')
    parser.add_argument('--context_length', type=int, default=256, 
                        help='Length of context to predict autoregressively')
    parser.add_argument('--model', type=str, default='amazon/chronos-bolt-small',
                        help='Chronos model to use for prediction')
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
    all_self_informations = []

    for i, start in enumerate(random_starts):
        print(f"\nProcessing chunk {i+1}/3 (start index {start})...")
        ground_truth = series[start:start+args.context_length]
        preds, self_infos = autoregressive_predict_context_simplified(pipeline, ground_truth, dataset_mean)
        all_ground_truths.append(ground_truth)
        all_predictions.append(preds)
        all_self_informations.append(self_infos)

    # Plot results with self-information
    fig, axs = plt.subplots(3, 1, figsize=(15, 12))
    
    for i in range(3):
        # Create twin axis for self-information
        ax2 = axs[i].twinx()
        
        # Plot ground truth and predictions on left axis
        axs[i].plot(all_ground_truths[i], label="Ground Truth", linestyle='-.', 
                   marker='o', linewidth=2, alpha=0.8, markersize=3, color='blue')
        axs[i].plot(all_predictions[i], label="Autoregressive Predictions", linestyle='-.', 
                   marker='x', linewidth=2, alpha=0.8, markersize=3, color='green')
        
        # Plot self-information on right axis
        ax2.plot(all_self_informations[i], label="Self-Information", 
                color='red', linewidth=1.5, alpha=0.7)
        
        # Set labels and titles
        axs[i].set_title(f"Chunk {i+1} - Autoregressive Prediction with Self-Information (start index {random_starts[i]})")
        axs[i].set_xlabel("Sample Index")
        axs[i].set_ylabel("Value", color='black')
        ax2.set_ylabel("Self-Information (bits)", color='red')
        
        # Color the right y-axis labels red
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Add legends
        lines1, labels1 = axs[i].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axs[i].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        axs[i].grid(True, alpha=0.3)

    # Create results directory structure
    dataset_name = os.path.splitext(os.path.basename(args.csv_path))[0]
    results_dir = f"results_si/{dataset_name}/col_{args.col_num}"
    os.makedirs(results_dir, exist_ok=True)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "si_with_quantiles.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.show()

    # Calculate and print metrics
    print("\nPrediction Metrics:")
    for i in range(3):
        mae = np.mean(np.abs(all_ground_truths[i] - all_predictions[i]))
        mse = np.mean((all_ground_truths[i] - all_predictions[i]) ** 2)
        rmse = np.sqrt(mse)
        avg_self_info = np.mean(all_self_informations[i])
        print(f"Chunk {i+1}: MAE={mae:.4f}, RMSE={rmse:.4f}, Avg Self-Info={avg_self_info:.4f} bits")


if __name__ == '__main__':
    main()
