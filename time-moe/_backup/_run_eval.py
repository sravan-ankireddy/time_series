#!/usr/bin/env python
# -*- coding:utf-8 _*-
import json
import os
import argparse
import numpy as np
import logging
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

from transformers import AutoModelForCausalLM

from time_moe.datasets.benchmark_dataset import BenchmarkEvalDataset, GeneralEvalDataset


def setup_nccl(rank, world_size, master_addr='127.0.0.1', master_port=9899):
    dist.init_process_group("nccl", init_method='tcp://{}:{}'.format(master_addr, master_port), rank=rank,
                            world_size=world_size)


def downsample_data(data, downsample_factor):
    """Downsample data by taking every nth point"""
    if downsample_factor == 1:
        return data
    
    breakpoint()
    return data[::downsample_factor]


def upsample_predictions(predictions, original_length, downsample_factor):
    """Upsample predictions back to original length using linear interpolation"""
    if downsample_factor == 1:
        return predictions
    
    # Create indices for interpolation
    downsampled_indices = np.arange(0, original_length, downsample_factor)
    original_indices = np.arange(original_length)
    
    # Interpolate to get back to original length
    upsampled = np.interp(original_indices, downsampled_indices, predictions)
    return upsampled


def list_csv_columns(csv_path):
    """List available columns in a CSV file with their indices"""
    try:
        df = pd.read_csv(csv_path, nrows=1)  # Read only first row to get columns
        columns = [col for col in df.columns if col.lower() != 'date']
        return columns
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []

from time_moe.datasets.benchmark_dataset import BenchmarkEvalDataset, GeneralEvalDataset


def setup_nccl(rank, world_size, master_addr='127.0.0.1', master_port=9899):
    dist.init_process_group("nccl", init_method='tcp://{}:{}'.format(master_addr, master_port), rank=rank,
                            world_size=world_size)


def downsample_data(data, downsample_factor):
    """Downsample data by taking every nth point"""
    if downsample_factor == 1:
        return data
    return data[::downsample_factor]


def upsample_predictions(predictions, original_length, downsample_factor):
    """Upsample predictions back to original length using linear interpolation"""
    if downsample_factor == 1:
        return predictions
    
    # Create indices for interpolation
    downsampled_indices = np.arange(0, original_length, downsample_factor)
    original_indices = np.arange(original_length)
    
    # Interpolate to get back to original length
    upsampled = np.interp(original_indices, downsampled_indices, predictions)
    return upsampled


class DownsampledBenchmarkEvalDataset(BenchmarkEvalDataset):
    def __init__(self, csv_path, context_length: int, prediction_length: int, downsample_factor: int = 1, max_windows: int = None, column_index: int = None):
        # Calculate adjusted lengths for downsampling
        adjusted_context_length = context_length // downsample_factor
        adjusted_prediction_length = prediction_length // downsample_factor
        
        # Store column index before initialization
        self.column_index = column_index
        
        # Initialize parent with adjusted lengths
        super().__init__(csv_path, adjusted_context_length, adjusted_prediction_length)
        
        # Apply column selection if specified
        if column_index is not None:
            df = pd.read_csv(csv_path)
            # Get columns excluding 'date'
            data_columns = [col for col in df.columns if col.lower() != 'date']
            
            if column_index < 0 or column_index >= len(data_columns):
                raise ValueError(f"Column index {column_index} out of range. Available columns (0-{len(data_columns)-1}): {data_columns}")
            
            # Get the column name at the specified index
            selected_column = data_columns[column_index]
            
            # Keep only the specified column
            cols = [selected_column]
            df_values = df[cols].values
            
            # Reprocess the data with selected column
            base_name = os.path.basename(csv_path).lower()
            if 'etth' in base_name:
                border1s = [0, 12 * 30 * 24 - adjusted_context_length, 12 * 30 * 24 + 4 * 30 * 24 - adjusted_context_length]
                border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
            elif 'ettm' in base_name:
                border1s = [0, 12 * 30 * 24 * 4 - adjusted_context_length, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - adjusted_context_length]
                border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
            else:
                num_train = int(len(df) * 0.7)
                num_test = int(len(df) * 0.2)
                num_vali = len(df) - num_train - num_test
                border1s = [0, num_train - adjusted_context_length, len(df) - num_test - adjusted_context_length]
                border2s = [num_train, num_train + num_vali, len(df)]
            
            train_data = df_values[border1s[0]:border2s[0]]
            test_data = df_values[border1s[2]:border2s[2]]
            
            # scaling
            scaler = StandardScaler()
            scaler.fit(train_data)
            scaled_test_data = scaler.transform(test_data)
            
            # Update dataset with single column
            self.hf_dataset = scaled_test_data.transpose(1, 0)
            self.num_sequences = len(self.hf_dataset)
            
            print(f"Using column index {column_index} ('{selected_column}') from CSV dataset")
        
        # Store original lengths and downsample factor
        self.original_context_length = context_length
        self.original_prediction_length = prediction_length
        self.downsample_factor = downsample_factor
        
        # Apply downsampling to the dataset
        if downsample_factor > 1:
            self.hf_dataset = np.array([downsample_data(seq, downsample_factor) for seq in self.hf_dataset])
            
            # Recalculate sub_seq_indexes with downsampled data
            self.sub_seq_indexes = []
            for seq_idx, seq in enumerate(self.hf_dataset):
                n_points = len(seq)
                if n_points < self.window_length:
                    continue
                for offset_idx in range(self.window_length, n_points):
                    self.sub_seq_indexes.append((seq_idx, offset_idx))
        
        # Limit number of windows if specified
        if max_windows is not None and max_windows > 0:
            self.sub_seq_indexes = self.sub_seq_indexes[:max_windows]
            print(f"Limited to {len(self.sub_seq_indexes)} windows (max_windows={max_windows})")


class DownsampledGeneralEvalDataset(GeneralEvalDataset):
    def __init__(self, data_path, context_length: int, prediction_length: int, downsample_factor: int = 1, onfly_norm: bool = False, max_windows: int = None, column_index: int = None):
        # Calculate adjusted lengths for downsampling
        adjusted_context_length = context_length // downsample_factor
        adjusted_prediction_length = prediction_length // downsample_factor
        
        # Initialize parent with adjusted lengths
        super().__init__(data_path, adjusted_context_length, adjusted_prediction_length, onfly_norm)
        
        # Store original lengths and downsample factor
        self.original_context_length = context_length
        self.original_prediction_length = prediction_length
        self.downsample_factor = downsample_factor
        self.column_index = column_index
        
        # Note: column_index is not used for GeneralEvalDataset as it doesn't use CSV format
        if column_index is not None:
            print(f"Warning: column_index {column_index} is ignored for non-CSV datasets")
        
        # Apply downsampling to the dataset if needed
        if downsample_factor > 1:
            # Downsample each sequence in the dataset
            downsampled_sequences = []
            for seq in self.dataset:
                downsampled_seq = downsample_data(seq, downsample_factor)
                downsampled_sequences.append(downsampled_seq)
            self.dataset.data = downsampled_sequences
            
            # Recalculate sub_seq_indexes with downsampled data
            self.sub_seq_indexes = []
            for seq_idx, seq in enumerate(self.dataset):
                n_points = len(seq)
                if n_points < self.window_length:
                    continue
                for offset_idx in range(self.window_length, n_points):
                    self.sub_seq_indexes.append((seq_idx, offset_idx))
        
        # Limit number of windows if specified
        if max_windows is not None and max_windows > 0:
            self.sub_seq_indexes = self.sub_seq_indexes[:max_windows]
            print(f"Limited to {len(self.sub_seq_indexes)} windows (max_windows={max_windows})")


def list_csv_columns(csv_path):
    """List available columns in a CSV file"""
    try:
        df = pd.read_csv(csv_path, nrows=1)  # Read only first row to get columns
        columns = [col for col in df.columns if col.lower() != 'date']
        return columns
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []


def count_num_tensor_elements(tensor):
    n = 1
    for s in tensor.shape:
        n = n * s
    return n


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
    def __init__(self, model_path, device, context_length, prediction_length, **kwargs):
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
        self.prediction_length = prediction_length
        self.model.eval()

    def predict(self, batch):
        model = self.model
        device = self.device
        prediction_length = self.prediction_length

        outputs = model.generate(
            inputs=batch['inputs'].to(device).to(model.dtype),
            max_new_tokens=prediction_length,
        )
        preds = outputs[:, -prediction_length:]
        labels = batch['labels'].to(device)
        if len(preds.shape) > len(labels.shape):
            labels = labels[..., None]
        return preds, labels


def evaluate_single(args, downsample_factor=1, experiment_name="Original"):
    """Evaluate model with optional downsampling"""
    batch_size = args.batch_size
    
    # Adjust context and prediction lengths for downsampling
    if downsample_factor > 1:
        context_length = args.context_length // downsample_factor
        prediction_length = args.prediction_length // downsample_factor
    else:
        context_length = args.context_length
        prediction_length = args.prediction_length

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
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            is_dist = False
    else:
        device = 'cpu'
        is_dist = False

    # evaluation
    metric_list = [
        MSEMetric(name='mse'),
        MAEMetric(name='mae'),
    ]

    model = TimeMoE(
        args.model,
        device,
        context_length=context_length,
        prediction_length=prediction_length
    )
    
    # Create appropriate dataset with downsampling
    if args.data.endswith('.csv'):
        dataset = DownsampledBenchmarkEvalDataset(
            args.data,
            context_length=args.context_length,  # Use original lengths
            prediction_length=args.prediction_length,
            downsample_factor=downsample_factor,
            max_windows=args.max_windows,
            column_index=args.column_index
        )
    else:
        dataset = DownsampledGeneralEvalDataset(
            args.data,
            context_length=args.context_length,  # Use original lengths
            prediction_length=args.prediction_length,
            downsample_factor=downsample_factor,
            max_windows=args.max_windows,
            column_index=args.column_index
        )

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
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_dl, desc=f"Evaluating {experiment_name}")):
            preds, labels = model.predict(batch)

            # Store predictions and labels for plotting
            if rank == 0:
                all_predictions.append(preds.cpu().float().numpy())
                all_labels.append(labels.cpu().float().numpy())

            for metric in metric_list:
                metric.push(preds, labels)

            acc_count += count_num_tensor_elements(preds)

    ret_metric = {}
    for metric in metric_list:
        ret_metric[metric.name] = metric.value / acc_count
    print(f'{rank} - {experiment_name} - {ret_metric}')

    metric_tensors = [metric.value for metric in metric_list] + [acc_count]
    if is_dist:
        stat_tensor = torch.tensor(metric_tensors).to(model.device)
        gathered_results = [torch.zeros_like(stat_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_results, stat_tensor)
        all_stat = torch.stack(gathered_results, dim=0).sum(dim=0)
    else:
        all_stat = metric_tensors

    result = None
    if rank == 0:
        item = {
            'model': args.model,
            'data': args.data,
            'column_index': args.column_index,
            'context_length': context_length,
            'prediction_length': prediction_length,
            'original_context_length': args.context_length,
            'original_prediction_length': args.prediction_length,
            'downsample_factor': downsample_factor,
            'experiment_name': experiment_name,
            'max_windows': args.max_windows,
            'actual_windows': len(dataset)
        }

        count = all_stat[-1]
        for i, metric in enumerate(metric_list):
            val = all_stat[i] / count
            item[metric.name] = float(val.cpu().float().numpy())
        
        logging.info(item)
        
        # Concatenate all predictions and labels for plotting
        if all_predictions:
            all_predictions = np.concatenate(all_predictions, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            item['predictions'] = all_predictions
            item['labels'] = all_labels
        
        result = item
    
    return result


def plot_results(results, output_dir="./results"):
    """Plot comparison results between original and downsampled experiments"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract metrics for plotting
    experiments = []
    mse_values = []
    mae_values = []
    downsample_factors = []
    
    for result in results:
        experiments.append(result['experiment_name'])
        mse_values.append(result['mse'])
        mae_values.append(result['mae'])
        downsample_factors.append(result['downsample_factor'])
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # MSE comparison
    axes[0, 0].bar(experiments, mse_values, color=['blue', 'red', 'green', 'orange'][:len(experiments)])
    axes[0, 0].set_title('Mean Squared Error Comparison')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # MAE comparison
    axes[0, 1].bar(experiments, mae_values, color=['blue', 'red', 'green', 'orange'][:len(experiments)])
    axes[0, 1].set_title('Mean Absolute Error Comparison')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Sample predictions comparison (first few samples)
    if 'predictions' in results[0] and 'labels' in results[0]:
        sample_idx = 0
        for i, result in enumerate(results[:2]):  # Show only first 2 experiments to avoid clutter
            if 'predictions' in result:
                pred_sample = result['predictions'][sample_idx]
                label_sample = result['labels'][sample_idx]
                
                x_pred = np.arange(len(pred_sample))
                
                axes[1, 0].plot(x_pred, pred_sample, 
                               label=f'{result["experiment_name"]} Pred', 
                               linestyle='--', alpha=0.7)
                if i == 0:  # Only plot labels once
                    axes[1, 0].plot(x_pred, label_sample, 
                                   label='Ground Truth', 
                                   color='black', linewidth=2)
        
        axes[1, 0].set_title('Sample Prediction Comparison')
        axes[1, 0].set_xlabel('Time Steps')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].legend()
    
    # Downsample factor vs performance
    if len(downsample_factors) > 1:
        axes[1, 1].plot(downsample_factors, mse_values, 'o-', label='MSE', color='blue')
        axes[1, 1].set_xlabel('Downsample Factor')
        axes[1, 1].set_ylabel('MSE', color='blue')
        axes[1, 1].tick_params(axis='y', labelcolor='blue')
        
        ax2 = axes[1, 1].twinx()
        ax2.plot(downsample_factors, mae_values, 's-', label='MAE', color='red')
        ax2.set_ylabel('MAE', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        axes[1, 1].set_title('Performance vs Downsample Factor')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Multiple downsample factors\nneeded for this plot', 
                       transform=axes[1, 1].transAxes, ha='center', va='center')
        axes[1, 1].set_title('Performance vs Downsample Factor')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'evaluation_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Results plot saved to: {plot_path}")
    
    # Create results summary table
    results_df = pd.DataFrame([
        {
            'Experiment': result['experiment_name'],
            'Column Index': result.get('column_index', 'All'),
            'Downsample Factor': result['downsample_factor'],
            'Context Length': result['context_length'],
            'Prediction Length': result['prediction_length'],
            'Windows Tested': result['actual_windows'],
            'MSE': f"{result['mse']:.6f}",
            'MAE': f"{result['mae']:.6f}"
        }
        for result in results
    ])
    
    csv_path = os.path.join(output_dir, 'evaluation_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Results summary saved to: {csv_path}")
    print("\nResults Summary:")
    print(results_df.to_string(index=False))


def evaluate(args):
    """Main evaluation function that runs both original and downsampled experiments"""
    # Create structured results directory
    if args.data.endswith('.csv'):
        dataset_name = os.path.splitext(os.path.basename(args.data))[0]
    else:
        dataset_name = os.path.basename(args.data.rstrip('/'))
    
    if args.column_index is not None:
        dataset_name = f"{dataset_name}/COL_{args.column_index}"
    
    structured_output_dir = os.path.join(args.output_dir, dataset_name)
    os.makedirs(structured_output_dir, exist_ok=True)
    print(f"Results will be saved to: {structured_output_dir}")
    
    # List available columns if CSV and no column index specified
    if args.data.endswith('.csv') and args.column_index is None:
        available_columns = list_csv_columns(args.data)
        print(f"Available columns in CSV (0-{len(available_columns)-1}): {available_columns}")
        print("Using all columns (default behavior). Use --column_index to focus on a specific column.")
    
    results = []
    
    # Run original experiment
    print("Running original experiment...")
    original_result = evaluate_single(args, downsample_factor=1, experiment_name="Original")
    if original_result:
        results.append(original_result)
    
    # Run downsampled experiments
    for factor in args.downsample_factors:
        if factor > 1:
            print(f"Running downsampled experiment with factor {factor}...")
            downsampled_result = evaluate_single(
                args, 
                downsample_factor=factor, 
                experiment_name=f"Downsampled {factor}x"
            )
            if downsampled_result:
                results.append(downsampled_result)
    
    # Plot and save results (only on rank 0)
    rank = int(os.getenv('RANK') or 0)
    if rank == 0 and results:
        plot_results(results, structured_output_dir)


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
        '--downsample_factors',
        type=int,
        nargs='+',
        default=[2, 4, 8],
        help='List of downsampling factors to test (e.g., --downsample_factors 2 4 8)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Directory to save results and plots'
    )
    parser.add_argument(
        '--max_windows',
        type=int,
        default=None,
        help='Maximum number of windows to test (None for all windows). Useful for quick experiments.'
    )
    parser.add_argument(
        '--column_index',
        type=int,
        default=None,
        help='Column index to focus on in CSV files (e.g., 0, 1, 2...). If not specified, uses all columns. Use --column_index with the 0-based index of the column you want to analyze.'
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
    evaluate(args)
