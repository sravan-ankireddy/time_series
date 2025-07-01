import argparse
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from chronos import BaseChronosPipeline
import random
import os
from tqdm import tqdm


def autoregressive_predict_context(pipeline, ground_truth_context, dataset_mean):
    """Perform autoregressive prediction on the context itself, one sample at a time."""
    predictions = []
    context_length = len(ground_truth_context)
    
    for i in tqdm(range(context_length), desc="Predicting samples"):
        if i == 0:
            # Use dataset mean for first prediction context
            context = torch.tensor([dataset_mean], dtype=torch.float32)
        else:
            # Use ground truth values from 0 to i-1 as context
            context = torch.tensor(ground_truth_context[:i], dtype=torch.float32)
        
        quantiles, mean = pipeline.predict_quantiles(
            context=context,
            prediction_length=1,
            quantile_levels=[0.5],
        )
        pred = mean[0, 0].item()
        predictions.append(pred)
    
    return np.array(predictions)


def main():
    parser = argparse.ArgumentParser(description='Autoregressive prediction on context itself for 3 random chunks.')
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

    for i, start in enumerate(random_starts):
        print(f"\nProcessing chunk {i+1}/3 (start index {start})...")
        ground_truth = series[start:start+args.context_length]
        preds = autoregressive_predict_context(pipeline, ground_truth, dataset_mean)
        all_ground_truths.append(ground_truth)
        all_predictions.append(preds)

    # Plot results
    fig, axs = plt.subplots(3, 1, figsize=(15, 12))
    for i in range(3):
        axs[i].plot(all_ground_truths[i], label="Ground Truth", linestyle='-.', 
                   marker='o', linewidth=2, alpha=0.8, markersize=3)
        axs[i].plot(all_predictions[i], label="Autoregressive Predictions", linestyle='-.', 
                   marker='x', linewidth=2, alpha=0.8, markersize=3)
        axs[i].set_title(f"Chunk {i+1} - Autoregressive Prediction (start index {random_starts[i]})")
        axs[i].set_xlabel("Sample Index")
        axs[i].set_ylabel("Value")
        axs[i].legend()
        axs[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("autoregressive_prediction_results.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Calculate and print metrics
    print("\nPrediction Metrics:")
    for i in range(3):
        mae = np.mean(np.abs(all_ground_truths[i] - all_predictions[i]))
        mse = np.mean((all_ground_truths[i] - all_predictions[i]) ** 2)
        rmse = np.sqrt(mse)
        print(f"Chunk {i+1}: MAE={mae:.4f}, RMSE={rmse:.4f}")


if __name__ == '__main__':
    main()
