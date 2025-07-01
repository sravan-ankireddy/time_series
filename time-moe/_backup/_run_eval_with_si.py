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

from transformers import AutoModelForCausalLM

from time_moe.datasets.benchmark_dataset import BenchmarkEvalDataset, GeneralEvalDataset


# Simple wrapper to add max_samples support
class LimitedBenchmarkEvalDataset(BenchmarkEvalDataset):
    def __init__(self, csv_path, context_length: int, prediction_length: int, max_samples: int = None):
        super().__init__(csv_path, context_length, prediction_length)
        if max_samples is not None and max_samples > 0:
            self.sub_seq_indexes = self.sub_seq_indexes[:max_samples]
            print(f"Limited to {len(self.sub_seq_indexes)} samples (max_samples={max_samples})")


def setup_nccl(rank, world_size, master_addr='127.0.0.1', master_port=9899):
    dist.init_process_group("nccl", init_method='tcp://{}:{}'.format(master_addr, master_port), rank=rank,
                            world_size=world_size)


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
    def __init__(self, model_path, device, context_length, prediction_length, downsample_rate=1, **kwargs):
        # Adjust prediction length for downsampling - use ceiling division to ensure we get enough tokens
        self.downsample_rate = downsample_rate
        self.original_prediction_length = prediction_length
        # Use ceiling division to ensure we generate enough downsampled tokens
        self.prediction_length = (prediction_length + downsample_rate - 1) // downsample_rate
        
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

    def predict(self, batch):
        model = self.model
        device = self.device
        prediction_length = self.prediction_length
        downsample_rate = self.downsample_rate
        original_prediction_length = self.original_prediction_length

        # Get inputs and labels
        inputs = batch['inputs'].to(device).to(model.dtype)
        labels = batch['labels'].to(device)
        
        # Downsample inputs and labels
        if downsample_rate > 1:
            inputs = inputs[:, ::downsample_rate]
            labels = labels[:, ::downsample_rate]
            
            # Ensure labels match the expected original prediction length after downsampling
            expected_downsampled_length = (original_prediction_length + downsample_rate - 1) // downsample_rate
            if labels.shape[1] > expected_downsampled_length:
                labels = labels[:, :expected_downsampled_length]

        outputs = model.generate(
            inputs=inputs,
            max_new_tokens=prediction_length,
        )
        preds = outputs[:, -prediction_length:]
        
        # Ensure predictions match labels shape
        min_length = min(preds.shape[1], labels.shape[1])
        preds = preds[:, :min_length]
        labels = labels[:, :min_length]
            
        if len(preds.shape) > len(labels.shape):
            labels = labels[..., None]
        return preds, labels


def evaluate_single_rate(args, downsample_rate):
    """Evaluate model with a single downsample rate"""
    batch_size = args.batch_size
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
        prediction_length=prediction_length,
        downsample_rate=downsample_rate
    )
    if args.data.endswith('.csv'):
        dataset = LimitedBenchmarkEvalDataset(
            args.data,
            context_length=context_length,
            prediction_length=prediction_length,
            max_samples=args.max_samples
        )
    else:
        dataset = GeneralEvalDataset(
            args.data,
            context_length=context_length,
            prediction_length=prediction_length,
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
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_dl, desc=f"Evaluating downsample_rate={downsample_rate}")):
            preds, labels = model.predict(batch)

            for metric in metric_list:
                metric.push(preds, labels)

            acc_count += count_num_tensor_elements(preds)

    ret_metric = {}
    for metric in metric_list:
        ret_metric[metric.name] = metric.value / acc_count
    print(f'{rank} - downsample_rate={downsample_rate} - {ret_metric}')

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
        
        # Save results to specified folder structure
        dataset_name = os.path.basename(args.data).split('.')[0]
        results_dir = f"results/{dataset_name}"
        os.makedirs(results_dir, exist_ok=True)
        
        result_file = f"{results_dir}/downsample_{downsample_rate}.json"
        with open(result_file, 'w') as f:
            json.dump(item, f, indent=2)
        
        return item
    
    return None


def plot_performance_comparison(results, dataset_name, output_dir="results"):
    """Plot performance comparison across different downsample rates"""
    if not results:
        print("No results to plot")
        return
    
    downsample_rates = [r['downsample_rate'] for r in results]
    mse_values = [r['mse'] for r in results]
    mae_values = [r['mae'] for r in results]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # MSE plot
    ax1.plot(downsample_rates, mse_values, 'o-', linewidth=2, markersize=8, label='MSE')
    ax1.set_xlabel('Downsample Rate')
    ax1.set_ylabel('MSE')
    ax1.set_title(f'MSE vs Downsample Rate\n{dataset_name}')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # MAE plot
    ax2.plot(downsample_rates, mae_values, 'o-', color='orange', linewidth=2, markersize=8, label='MAE')
    ax2.set_xlabel('Downsample Rate')
    ax2.set_ylabel('MAE')
    ax2.set_title(f'MAE vs Downsample Rate\n{dataset_name}')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    
    # Save plot
    plot_dir = f"{output_dir}/{dataset_name}"
    os.makedirs(plot_dir, exist_ok=True)
    plot_file = f"{plot_dir}/performance_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Performance comparison plot saved to: {plot_file}")
    
    # Also save summary table
    summary_df = pd.DataFrame(results)
    summary_file = f"{plot_dir}/performance_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Performance summary saved to: {summary_file}")
    
    plt.show()


def evaluate(args):
    """Main evaluation function that handles multiple downsample rates"""
    downsample_rates = args.downsample_rates
    dataset_name = os.path.basename(args.data).split('.')[0]
    
    print(f"Evaluating with downsample rates: {downsample_rates}")
    print(f"Dataset: {dataset_name}")
    
    results = []
    
    for rate in downsample_rates:
        print(f"\n{'='*50}")
        print(f"Evaluating downsample rate: {rate}")
        print(f"{'='*50}")
        
        result = evaluate_single_rate(args, rate)
        if result:
            results.append(result)
    
    # Plot performance comparison
    if results:
        print(f"\n{'='*50}")
        print("Generating performance comparison plot...")
        print(f"{'='*50}")
        plot_performance_comparison(results, dataset_name)
        
        # Print summary
        print("\nPerformance Summary:")
        print("-" * 60)
        print(f"{'Rate':<8} {'MSE':<15} {'MAE':<15}")
        print("-" * 60)
        for r in results:
            print(f"{r['downsample_rate']:<8} {r['mse']:<15.6f} {r['mae']:<15.6f}")
    else:
        print("No results collected for plotting.")


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
        default=[1, 2, 3, 4, 5, 6, 7, 8],
        help='List of downsampling rates to evaluate (e.g., 1 2 4 8)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to evaluate (default: all samples)'
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
