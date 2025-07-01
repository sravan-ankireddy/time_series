#!/usr/bin/env python
# -*- coding:utf-8 _*-
import json
import os
import argparse
import numpy as np
import logging
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from transformers import AutoModelForCausalLM

from time_moe.datasets.benchmark_dataset import BenchmarkEvalDataset, GeneralEvalDataset


def setup_nccl(rank, world_size, master_addr='127.0.0.1', master_port=9899):
    dist.init_process_group("nccl", init_method='tcp://{}:{}'.format(master_addr, master_port), rank=rank,
                            world_size=world_size)


def count_num_tensor_elements(tensor):
    n = 1
    for s in tensor.shape:
        n = n * s
    return n


# ------------------ Self-Information Computation ------------------
def compute_self_information(model, context_data, device, si_context_length=512, stride=1):
    """
    Compute self-information for each sample in the context using autoregressive prediction with stride.
    Self-information is computed as MSE between predicted and actual values.
    Uses a fixed context length for each prediction and slides the window by stride steps.
    
    Args:
        model: TimeMoE model for prediction
        context_data: numpy array of shape (context_length,) - the context sequence
        device: device to run computation on
        si_context_length: fixed number of context samples to use for each prediction
        stride: number of steps to predict at once and slide the window
    
    Returns:
        si_scores: numpy array of self-information scores (MSE) for each position
        predictions: numpy array of predicted values for each position
    """
    context_length = len(context_data)
    si_scores = []
    predictions = []
    
    # We can only compute SI for positions that have enough previous context
    # Start from si_context_length to ensure we always have exactly si_context_length samples
    total_positions = context_length - si_context_length
    
    # Calculate number of stride windows needed
    num_windows = (total_positions + stride - 1) // stride  # Ceiling division
    
    for window_idx in tqdm(range(num_windows), desc="Computing SI with stride", leave=False):
        # Calculate the starting position for this window
        window_start = si_context_length + window_idx * stride
        window_end = min(window_start + stride, context_length)
        actual_stride = window_end - window_start
        
        if actual_stride <= 0:
            break
            
        # Use exactly si_context_length samples as context to predict the next stride samples
        context_start = window_start - si_context_length
        context_subset = context_data[context_start:window_start]
        true_values = context_data[window_start:window_end]
        
        # Prepare input tensor
        context_tensor = torch.tensor(context_subset, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Generate stride-step prediction
        predicted_tensor = model.predict_with_stride(context_tensor, stride=actual_stride)
        predicted_values = predicted_tensor.cpu().float().numpy().flatten()
        
        # Ensure we have the right number of predictions
        if len(predicted_values) != actual_stride:
            # Handle case where model returns different number of predictions
            predicted_values = predicted_values[:actual_stride]
        
        # Compute self-information as MSE for each predicted step
        for i in range(actual_stride):
            if i < len(predicted_values) and i < len(true_values):
                mse = (predicted_values[i] - true_values[i]) ** 2
                si_scores.append(mse)
                predictions.append(predicted_values[i])
    
    return np.array(si_scores), np.array(predictions)


# ------------------ Metrics ------------------
class SumEvalMetric:
    def __init__(self, name, init_val: float = 0.0):
        self.name = name
        self.value = init_val

    def push(self, preds, labels, **kwargs):
        self.value += self._calculate(preds, labels, **kwargs)

    def _calculate(self, preds, labels, **kwargs):
        pass


class MSEMetric(SumEvalMetric):
    def _calculate(self, preds, labels, **kwargs):
        return torch.sum((preds - labels) ** 2)


class MAEMetric(SumEvalMetric):
    def _calculate(self, preds, labels, **kwargs):
        return torch.sum(torch.abs(preds - labels))


class TimeMoE:
    def __init__(self, model_path, device, context_length, prediction_length, downsample_rate=1, **kwargs):
        self.downsample_rate = downsample_rate
        self.original_prediction_length = prediction_length
        self.context_length = context_length
        self.prediction_length = prediction_length
        
        try:
            from time_moe.models.modeling_time_moe import TimeMoeForPrediction
            model = TimeMoeForPrediction.from_pretrained(
                model_path,
                device_map=device,
                attn_implementation='flash_attention_2',
                torch_dtype='auto',
            )
        except:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                attn_implementation='flash_attention_2',
                torch_dtype='auto',
                trust_remote_code=True,
            )

        logging.info(f'>>> Model dtype: {model.dtype}; Attention:{model.config._attn_implementation}')

        self.model = model
        self.device = device
        self.model.eval()

    def predict_with_stride(self, context, stride=1):
        """
        Predict a single time step given context.
        
        Args:
            context: tensor of shape (batch_size, context_length) or (context_length,)
            stride: the number of steps to move the context window for each prediction

        Returns:
            prediction: tensor of shape (batch_size, stride) or scalar
        """
        if len(context.shape) == 1:
            context = context.unsqueeze(0)
        
        context = context.to(self.device).to(self.model.dtype)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs=context,
                max_new_tokens=stride,
            )
            prediction = outputs[:, -stride:]  # Get the last predicted tokens
        
        return prediction

    def predict(self, batch):
        model = self.model
        device = self.device
        prediction_length = self.prediction_length
        downsample_rate = self.downsample_rate

        inputs = batch['inputs'].to(device).to(model.dtype)
        labels = batch['labels'].to(device)
        
        # Apply downsampling
        if downsample_rate > 1:
            inputs = inputs[:, ::downsample_rate]
            labels = labels[:, ::downsample_rate]

        outputs = model.generate(
            inputs=inputs,
            max_new_tokens=prediction_length,
        )
        preds = outputs[:, -prediction_length:]

        if len(preds.shape) > len(labels.shape):
            labels = labels[..., None]
        return preds, labels


def evaluate(args):
    batch_size = args.batch_size
    context_length = args.context_length
    prediction_length = args.prediction_length
    downsample_rates = args.downsample_rates

    # Handle column extraction if specified
    temp_csv_file = None
    data_file = args.data
    
    if args.date_col_idx is not None and args.data_col_idx is not None:
        if not args.data.endswith('.csv'):
            raise ValueError("Column extraction is only supported for CSV files")
        temp_csv_file = create_temp_csv(args.data, args.date_col_idx, args.data_col_idx)
        data_file = temp_csv_file
        print(f"Using temporary extracted CSV: {temp_csv_file}")
    elif args.date_col_idx is not None or args.data_col_idx is not None:
        raise ValueError("Both date_col_idx and data_col_idx must be specified together")

    master_addr = os.getenv('MASTER_ADDR', '127.0.0.1')
    master_port = os.getenv('MASTER_PORT', 9899)
    world_size = int(os.getenv('WORLD_SIZE') or 1)
    rank = int(os.getenv('RANK') or 0)
    local_rank = int(os.getenv('LOCAL_RANK') or 0)
    if torch.cuda.is_available():
        world_size = int(os.getenv("WORLD_SIZE") or 1)
        if world_size > 1:
            # only initialize NCCL if we're truly doing >1 process
            setup_nccl(rank, world_size, master_addr, master_port)
            device = f"cuda:{local_rank}"
            is_dist = True
        else:
            # single-GPU / CPU fallback
            device = "cuda:1" if torch.cuda.is_available() else "cpu"
            is_dist = False
    else:
        device = 'cpu'
        is_dist = False

    # Store results for all downsampling rates
    all_results = {}
    # Store sample data for plotting
    sample_data = {}
    
    for downsample_rate in downsample_rates:
        print(f"Evaluating with downsample_rate: {downsample_rate}")
        
        # Apply downsampling to context and prediction lengths
        downsampled_context_length = context_length // downsample_rate
        downsampled_prediction_length = prediction_length // downsample_rate
        
        # evaluation
        metric_list = [
            MSEMetric(name='mse'),
            MAEMetric(name='mae'),
        ]

        model = TimeMoE(
            args.model,
            device,
            context_length=downsampled_context_length,
            prediction_length=downsampled_prediction_length,
            downsample_rate=downsample_rate
        )
        
        if data_file.endswith('.csv'):
            dataset = BenchmarkEvalDataset(
                data_file,
                context_length=context_length,
                prediction_length=prediction_length,
            )
        else:
            dataset = GeneralEvalDataset(
                data_file,
                context_length=context_length,
                prediction_length=prediction_length,
            )

        # Limit dataset size if max_samples is specified
        if args.max_samples is not None and args.max_samples < len(dataset):
            # Create indices for subset
            subset_indices = list(range(min(args.max_samples, len(dataset))))
            dataset = Subset(dataset, subset_indices)
            print(f"Using subset of {len(dataset)} samples (max_samples={args.max_samples})")
        else:
            print(f"Using full dataset with {len(dataset)} samples")

        if torch.cuda.is_available() and dist.is_initialized():
            sampler = DistributedSampler(dataset=dataset, shuffle=False)
        else:
            sampler = None
        test_dl = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=2,
            prefetch_factor=2,
            drop_last=False,
        )

        acc_count = 0
        collected_samples = []
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(test_dl, desc=f"Evaluating downsample_rate={downsample_rate}")):
                preds, labels = model.predict(batch)

                # Collect first 3 samples for plotting
                if len(collected_samples) < 3:
                    batch_size_actual = preds.shape[0]
                    for b in range(min(batch_size_actual, 3 - len(collected_samples))):
                        # Get original context and apply same downsampling
                        original_context = batch['inputs'][b].cpu().numpy()
                        downsampled_context = original_context[::downsample_rate]
                        
                        # Labels and preds are already downsampled by the model
                        collected_samples.append({
                            'context': downsampled_context,
                            'ground_truth': labels[b].cpu().float().numpy().flatten(),
                            'prediction': preds[b].cpu().float().numpy().flatten(),
                            'rate': downsample_rate  # Store the rate with each sample
                        })

                for metric in metric_list:
                    metric.push(preds, labels)

                acc_count += count_num_tensor_elements(preds)

                if args.max_batches is not None and idx >= args.max_batches - 1:
                    break
        
        # Store samples for this downsampling rate
        sample_data[downsample_rate] = collected_samples

        ret_metric = {}
        for metric in metric_list:
            ret_metric[metric.name] = metric.value / acc_count
        print(f'{rank} - {ret_metric}')

        metric_tensors = [metric.value for metric in metric_list] + [acc_count]
        if is_dist:
            stat_tensor = torch.tensor(metric_tensors).to(model.device)
            gathered_results = [torch.zeros_like(stat_tensor) for _ in range(world_size)]
            dist.all_gather(gathered_results, stat_tensor)
            all_stat = torch.stack(gathered_results, dim=0).sum(dim=0)
        else:
            all_stat = metric_tensors

        if rank == 0:
            item = {
                'model': args.model,
                'data': args.data,
                'context_length': args.context_length,
                'prediction_length': args.prediction_length,
                'downsample_rate': downsample_rate,
            }

            count = all_stat[-1]
            for i, metric in enumerate(metric_list):
                val = all_stat[i] / count
                item[metric.name] = float(val.cpu().numpy())
            logging.info(item)
            
            # Store results for plotting
            all_results[downsample_rate] = {
                'mse': item['mse'],
                'mae': item['mae']
            }
    
    # Generate plot if we have multiple rates and we're on rank 0
    if rank == 0 and len(downsample_rates) > 1:
        plot_results(all_results, args)
        plot_sample_predictions(sample_data, args)
    
    # Self-Information based evaluation
    if args.enable_si and rank == 0:
        print("\n" + "="*60)
        print("SELF-INFORMATION BASED EVALUATION")
        print("Computing self-information scores for 3 random context samples...")
        print(f"Using fixed SI context length: {args.si_context_length}")
        print(f"Using stride for SI computation: {args.si_stride}")
        print("="*60)
        
        # Use the model with downsample_rate=1 for SI computation
        si_model = TimeMoE(
            args.model,
            device,
            context_length=args.si_context_length,  # Use SI context length for model
            prediction_length=args.si_stride,  # Use stride as prediction length for efficiency
            downsample_rate=1
        )
        
        # Get the dataset - we need to create a new dataset with extended context for SI
        # Load dataset with extended context length to ensure we have enough previous samples
        extended_context_length = context_length + args.si_context_length
        
        if data_file.endswith('.csv'):
            extended_dataset = BenchmarkEvalDataset(
                data_file,
                context_length=extended_context_length,
                prediction_length=prediction_length,
            )
        else:
            extended_dataset = GeneralEvalDataset(
                data_file,
                context_length=extended_context_length,
                prediction_length=prediction_length,
            )
        
        # Randomly select 3 samples from the extended dataset
        import random
        random.seed(42)  # For reproducible results
        selected_indices = random.sample(range(len(extended_dataset)), min(3, len(extended_dataset)))
        
        print(f"Selected samples: {selected_indices}")
        print(f"Extended context length for SI: {extended_context_length}")
        
        # Compute SI for selected samples and plot
        plot_si_for_samples(si_model, extended_dataset, selected_indices, device, args)
    
    # Clean up temporary file if created
    if temp_csv_file and os.path.exists(temp_csv_file):
        try:
            os.remove(temp_csv_file)
            print(f"Cleaned up temporary file: {temp_csv_file}")
        except OSError as e:
            print(f"Warning: Could not remove temporary file {temp_csv_file}: {e}")


def plot_results(results, args):
    """Generate bar plot comparing MSE and MAE across downsampling rates."""
    rates = sorted(results.keys())
    mse_values = [results[rate]['mse'] for rate in rates]
    mae_values = [results[rate]['mae'] for rate in rates]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # MSE bar plot
    bars1 = ax1.bar([str(rate) for rate in rates], mse_values, color='skyblue', alpha=0.7, edgecolor='black')
    ax1.set_title('MSE vs Downsampling Rate')
    ax1.set_xlabel('Downsampling Rate')
    ax1.set_ylabel('MSE')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars1, mse_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{val:.4f}', ha='center', va='bottom')
    
    # MAE bar plot
    bars2 = ax2.bar([str(rate) for rate in rates], mae_values, color='lightcoral', alpha=0.7, edgecolor='black')
    ax2.set_title('MAE vs Downsampling Rate')
    ax2.set_xlabel('Downsampling Rate')
    ax2.set_ylabel('MAE')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars2, mae_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{val:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Create structured folder and save plot
    dataset_name = os.path.basename(args.data).split('.')[0]
    results_dir = f"results_with_si/{dataset_name}/col_{args.data_col_idx}"
    os.makedirs(results_dir, exist_ok=True)
    
    plot_filename = f"{results_dir}/p{args.prediction_length}_c{args.context_length}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {plot_filename}")
    plt.show()


def plot_sample_predictions(sample_data, args):
    """Plot 3 randomly selected samples showing input context, ground truth, and predictions."""
    if not sample_data:
        return
        
    # Create base directory structure - same as SI plots
    dataset_name = os.path.basename(args.data).split('.')[0]
    base_results_dir = f"results_with_si/{dataset_name}/col_{args.data_col_idx}"
    samples_dir = f"{base_results_dir}/p{args.prediction_length}_c{args.context_length}_stride{args.si_stride}"
    os.makedirs(samples_dir, exist_ok=True)
    
    # Select sample keys
    sample_keys = list(sample_data.keys())
    selected_samples = [rate for rate in args.downsample_rates if rate in sample_keys]
    
    # Create individual plots for each sample (0, 1, 2)
    for sample_idx in range(3):
        # Check if we have enough samples
        if any(sample_idx >= len(sample_data[rate]) for rate in selected_samples):
            break
            
        # Create subplot for this sample across all rates
        fig, axes = plt.subplots(len(selected_samples), 1, figsize=(32, 4*len(selected_samples)))
        if len(selected_samples) == 1:
            axes = [axes]
        
        for row, rate in enumerate(selected_samples):
            samples = sample_data[rate]
            
            if sample_idx >= len(samples):
                continue
                
            sample = samples[sample_idx]
            context = sample['context']
            ground_truth = sample['ground_truth'] 
            prediction = sample['prediction']

            # Create time indices with proper spacing based on the downsample rate
            context_len = len(context)
            pred_len = len(prediction)
            
            # Time indices should show the original positions of the downsampled data
            context_time = np.arange(-context_len * rate, 0, rate)
            pred_time = np.arange(0, pred_len * rate, rate)
            
            ax = axes[row]
            
            # Plot context with markers only
            ax.plot(context_time, context, 'b-.o', label='Context', markersize=4)
            
            # Plot ground truth with markers only
            ax.plot(pred_time, ground_truth, 'g-.s', label='Ground Truth', markersize=4)

            # Plot prediction with markers only
            ax.plot(pred_time, prediction, 'r-.^', label='Prediction', markersize=4)
            
            ax.axvline(x=0, color='black', linestyle=':', alpha=0.7, linewidth=1)
            ax.set_title(f'Downsample Rate {rate}', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=12)
            if row == len(selected_samples) - 1:
                ax.set_xlabel('Time Steps', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
        
        plt.suptitle(f'Sample {sample_idx} - Prediction Comparison Across Downsample Rates', fontsize=16)
        plt.tight_layout()
        
        # Save individual plot with relative sample number
        plot_filename = f"{samples_dir}/prediction_sample_{sample_idx}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()  # Close to free memory
        
    print(f"Prediction plots saved in: {samples_dir}/")
    print(f"Generated {min(3, min(len(sample_data[rate]) for rate in selected_samples))} prediction sample plots")


def create_temp_csv(csv_file, date_col_idx, data_col_idx):
    """
    Extract specified date and data columns from CSV and create a temporary file.
    
    Args:
        csv_file: Path to the source CSV file
        date_col_idx: Index of the date column (0-based)
        data_col_idx: Index of the data column (0-based)
    
    Returns:
        Path to the temporary CSV file
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract the specified columns
    date_col = df.iloc[:, date_col_idx]
    data_col = df.iloc[:, data_col_idx]
    
    # Create a new dataframe with extracted columns
    temp_df = pd.DataFrame({
        df.columns[date_col_idx]: date_col,
        df.columns[data_col_idx]: data_col
    })
    
    # Create temp directory if it doesn't exist
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create temporary file in the temp directory
    import time
    timestamp = int(time.time())
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    temp_filename = f"{temp_dir}/{base_name}_extracted_{timestamp}.csv"
    
    temp_df.to_csv(temp_filename, index=False)
    
    print(f"Created temporary CSV with {len(temp_df)} rows: {temp_filename}")
    print(f"Columns: {list(temp_df.columns)}")
    
    return temp_filename


def plot_si_for_samples(model, dataset, selected_indices, device, args):
    """Plot self-information analysis for selected samples - create separate plots for each sample."""
    
    # Create directory for SI plots - same as prediction plots
    dataset_name = os.path.basename(args.data).split('.')[0]
    base_results_dir = f"results_with_si/{dataset_name}/col_{args.data_col_idx}"
    si_dir = f"{base_results_dir}/p{args.prediction_length}_c{args.context_length}_stride{args.si_stride}"
    os.makedirs(si_dir, exist_ok=True)
    
    # Process each sample with progress bar using relative sample numbers (0, 1, 2)
    for plot_idx, sample_idx in enumerate(tqdm(selected_indices, desc="Processing samples", unit="sample")):
        sample = dataset[sample_idx]
        context_data = sample['inputs']
        
        # Convert to numpy array if it's a tensor
        if hasattr(context_data, 'numpy'):
            context_data = context_data.numpy()
        elif not isinstance(context_data, np.ndarray):
            context_data = np.array(context_data)
        
        print(f"Processing sample {plot_idx}: context length = {len(context_data)}")
        
        # Compute self-information scores
        si_scores, si_predictions = compute_self_information(
            model, context_data, device, args.si_context_length, args.si_stride
        )
        
        # Extract only the portion of context for which we computed SI
        si_start_idx = args.si_context_length  # Start of SI computation
        si_end_idx = len(context_data)  # End of the data
        
        # Get the context portion for which we have SI scores
        relevant_context = context_data[si_start_idx:si_end_idx]
        
        # Create individual plot for this sample using relative sample number
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        
        # Plot only the relevant context signal (for which we computed SI)
        context_positions = np.arange(len(relevant_context))
        ax.plot(context_positions, relevant_context, 'b-.o', alpha=0.7, linewidth=1, markersize=4, label='Context Signal')
        
        # Plot predictions (starts at position 0 in the relevant context)
        pred_positions = np.arange(len(si_predictions))
        ax.plot(pred_positions, si_predictions, 'r-.o', alpha=0.8, linewidth=1, markersize=4, label='Predictions')
        
        # Create a second y-axis for self-information scores
        ax2 = ax.twinx()
        si_positions = np.arange(len(si_scores))
        ax2.plot(si_positions, si_scores, 'g-.o', alpha=0.6, markersize=6, linewidth=1, label='Self-Information (MSE)')
        
        # Formatting
        ax.set_title(f'Sample {plot_idx}: Context Signal, Predictions, and Self-Information (Stride={args.si_stride})', fontsize=16)
        ax.set_xlabel('Position in Context Window', fontsize=12)
        ax.set_ylabel('Signal Value', color='b', fontsize=12)
        ax2.set_ylabel('Self-Information (MSE)', color='g', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add legends
        ax.legend(loc='upper left', fontsize=10)
        ax2.legend(loc='upper right', fontsize=10)
        
        # Color the y-axis labels
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='g')
        
        plt.tight_layout()
        
        # Save individual plot with relative sample number
        plot_filename = f"{si_dir}/si_sample_{plot_idx}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()  # Close to free memory
        
        # Create zoomed-in plot showing only the last 256 samples
        zoom_samples = 256
        if len(relevant_context) > zoom_samples:
            # Create zoomed plot
            fig_zoom, ax_zoom = plt.subplots(1, 1, figsize=(15, 8))
            
            # Get the last 256 samples
            zoom_context = relevant_context[-zoom_samples:]
            zoom_predictions = si_predictions[-zoom_samples:] if len(si_predictions) >= zoom_samples else si_predictions
            zoom_si_scores = si_scores[-zoom_samples:] if len(si_scores) >= zoom_samples else si_scores
            
            # Plot zoomed context signal
            zoom_context_positions = np.arange(len(zoom_context))
            ax_zoom.plot(zoom_context_positions, zoom_context, 'b-.o', alpha=0.7, linewidth=1, markersize=4, label='Context Signal')
            
            # Plot zoomed predictions
            zoom_pred_positions = np.arange(len(zoom_predictions))
            ax_zoom.plot(zoom_pred_positions, zoom_predictions, 'r-.o', alpha=0.8, linewidth=1, markersize=4, label='Predictions')
            
            # Create second y-axis for zoomed SI scores
            ax2_zoom = ax_zoom.twinx()
            zoom_si_positions = np.arange(len(zoom_si_scores))
            ax2_zoom.plot(zoom_si_positions, zoom_si_scores, 'g-.o', alpha=0.6, markersize=6, linewidth=1, label='Self-Information (MSE)')
            
            # Formatting for zoomed plot
            ax_zoom.set_title(f'Sample {plot_idx}: Zoomed View (Last {zoom_samples} samples, Stride={args.si_stride})', fontsize=16)
            ax_zoom.set_xlabel('Position in Context Window', fontsize=12)
            ax_zoom.set_ylabel('Signal Value', color='b', fontsize=12)
            ax2_zoom.set_ylabel('Self-Information (MSE)', color='g', fontsize=12)
            ax_zoom.grid(True, alpha=0.3)
            
            # Add legends for zoomed plot
            ax_zoom.legend(loc='upper left', fontsize=10)
            ax2_zoom.legend(loc='upper right', fontsize=10)
            
            # Color the y-axis labels for zoomed plot
            ax_zoom.tick_params(axis='y', labelcolor='b')
            ax2_zoom.tick_params(axis='y', labelcolor='g')
            
            plt.tight_layout()
            
            # Save zoomed plot with relative sample number
            zoom_plot_filename = f"{si_dir}/si_sample_{plot_idx}_zoom.png"
            plt.savefig(zoom_plot_filename, dpi=300, bbox_inches='tight')
            plt.close()  # Close to free memory
            
            print(f"  Zoomed plot saved as: {zoom_plot_filename}")
        
        print(f"  Mean SI score: {np.mean(si_scores):.4f}")
        print(f"  SI score range: [{np.min(si_scores):.4f}, {np.max(si_scores):.4f}]")
        print(f"  Relevant context length: {len(relevant_context)}")
        print(f"  SI predictions length: {len(si_predictions)}")
        print(f"  Full plot saved as: {plot_filename}")
    
    print(f"\nAll self-information plots saved in: {si_dir}/")
    print(f"Generated {len(selected_indices)} individual SI sample plots")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('TimeMoE Evaluate')
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='/home/sa53869/time-series/model_weights/time-moe-200m',
        help='Model path'
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        help='Benchmark data path'
    )

    parser.add_argument(
        '--batch_size', '-b',
        type=int,
        default=32,
        help='Batch size of evaluation'
    )
    parser.add_argument(
        '--context_length', '-c',
        type=int,
        help='Context length'
    )
    parser.add_argument(
        '--prediction_length', '-p',
        type=int,
        default=96,
        help='Prediction length'
    )
    parser.add_argument(
        '--downsample_rates',
        type=int,
        nargs='+',
        default=[1, 2, 4, 8],
        help='List of downsampling rates to evaluate (e.g., 1 2 4 8)'
    )
    parser.add_argument(
        '--max_batches',
        type=int,
        default=None,
        help='Maximum number of batches to evaluate (default: all batches)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to evaluate (default: all samples). Useful for quick testing.'
    )
    parser.add_argument(
        '--date_col_idx',
        type=int,
        default=0,
        help='Index of the date column in the CSV file (0-based). If specified, will extract columns and create temp CSV.'
    )
    parser.add_argument(
        '--data_col_idx', 
        type=int,
        default=1,
        help='Index of the data column in the CSV file (0-based). Required if date_col_idx is specified.'
    )
    parser.add_argument(
        '--enable_si',
        action='store_true',
        help='Enable self-information based evaluation'
    )
    parser.add_argument(
        '--si_context_length',
        type=int,
        default=None,
        help='Fixed context length for SI computation (default: same as context_length)'
    )
    parser.add_argument(
        '--si_stride',
        type=int,
        default=1,
        help='Number of steps to predict at once for SI computation (default: 1)'
    )
    args = parser.parse_args()
    if args.context_length is None:
        if args.prediction_length == 96:
            args.context_length = 512
        elif args.prediction_length == 192:
            args.context_length = 1024
        elif args.prediction_length == 336:
            args.context_length = 2048
        elif args.prediction_length == 720:
            args.context_length = 3072
        else:
            args.context_length = args.prediction_length * 4
    
    # Set default si_context_length to be same as context_length
    if args.si_context_length is None:
        args.si_context_length = args.context_length
    
    evaluate(args)
