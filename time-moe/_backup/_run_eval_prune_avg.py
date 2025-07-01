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


def calculate_effective_downsampling(original_length, downsample_factor, recent_fraction):
    """
    Calculate the effective downsampling factor needed for the older portion
    to maintain the desired effective context length.
    
    Args:
        original_length: Original context length
        downsample_factor: Desired overall downsampling factor
        recent_fraction: Fraction of recent samples to preserve
    
    Returns:
        effective_downsample_factor: Factor to apply to older portion
        effective_length: Resulting effective context length
    """
    target_effective_length = original_length // downsample_factor
    recent_samples_count = int(original_length * recent_fraction)
    older_samples_count = original_length - recent_samples_count
    
    # We want: recent_samples_count + (older_samples_count / effective_factor) = target_effective_length
    # So: effective_factor = older_samples_count / (target_effective_length - recent_samples_count)
    
    target_older_effective = target_effective_length - recent_samples_count
    if target_older_effective <= 0 or older_samples_count <= 0:
        # If recent samples already exceed target, just keep recent samples
        return 1, recent_samples_count
    
    effective_downsample_factor = older_samples_count / target_older_effective
    
    # Round to nearest integer (must be >= 1)
    effective_downsample_factor = max(1, round(effective_downsample_factor))
    
    # Calculate actual effective length
    actual_older_effective = older_samples_count // effective_downsample_factor
    actual_effective_length = recent_samples_count + actual_older_effective
    
    return effective_downsample_factor, actual_effective_length


def downsample_and_interpolate_with_region(context, downsample_factor=2, recent_fraction=0.25):
    """
    Downsample context by keeping recent fraction unchanged and adaptively downsampling 
    the rest to maintain effective context length.
    
    Args:
        context: Input tensor of shape (batch_size, context_length, ...) or (context_length, ...)
        downsample_factor: Desired overall downsampling factor
        recent_fraction: Fraction of most recent samples to keep unchanged
    
    Returns:
        Modified context with same shape but with selective downsampling
    """
    if downsample_factor == 1:
        return context  # No downsampling
    
    original_shape = context.shape
    context_flat = context.clone()
    
    # Handle different tensor dimensions
    if len(original_shape) == 1:
        # 1D tensor: (context_length,)
        length = original_shape[0]
        recent_samples_count = int(length * recent_fraction)
        older_samples_count = length - recent_samples_count
        
        effective_factor, _ = calculate_effective_downsampling(length, downsample_factor, recent_fraction)
        
        for i in range(1, older_samples_count, effective_factor):
            if i + 1 < older_samples_count:
                # Replace with average of neighbors
                left_idx = max(0, i - 1)
                right_idx = min(older_samples_count - 1, i + 1)
                context_flat[i] = (context_flat[left_idx] + context_flat[right_idx]) / 2
    
    elif len(original_shape) == 2:
        # 2D tensor: (batch_size, context_length) or (context_length, features)
        if original_shape[1] > original_shape[0]:  # Assume (batch_size, context_length)
            batch_size, length = original_shape
            recent_samples_count = int(length * recent_fraction)
            older_samples_count = length - recent_samples_count
            
            effective_factor, _ = calculate_effective_downsampling(length, downsample_factor, recent_fraction)
            
            for b in range(batch_size):
                for i in range(1, older_samples_count, effective_factor):
                    if i + 1 < older_samples_count:
                        left_idx = max(0, i - 1)
                        right_idx = min(older_samples_count - 1, i + 1)
                        context_flat[b, i] = (context_flat[b, left_idx] + context_flat[b, right_idx]) / 2
        else:  # Assume (context_length, features)
            length, features = original_shape
            recent_samples_count = int(length * recent_fraction)
            older_samples_count = length - recent_samples_count
            
            effective_factor, _ = calculate_effective_downsampling(length, downsample_factor, recent_fraction)
            
            for i in range(1, older_samples_count, effective_factor):
                if i + 1 < older_samples_count:
                    left_idx = max(0, i - 1)
                    right_idx = min(older_samples_count - 1, i + 1)
                    context_flat[i, :] = (context_flat[left_idx, :] + context_flat[right_idx, :]) / 2
    
    elif len(original_shape) == 3:
        # 3D tensor: (batch_size, context_length, features)
        batch_size, length, features = original_shape
        recent_samples_count = int(length * recent_fraction)
        older_samples_count = length - recent_samples_count
        
        effective_factor, _ = calculate_effective_downsampling(length, downsample_factor, recent_fraction)
        
        for b in range(batch_size):
            for i in range(1, older_samples_count, effective_factor):
                if i + 1 < older_samples_count:
                    left_idx = max(0, i - 1)
                    right_idx = min(older_samples_count - 1, i + 1)
                    context_flat[b, i, :] = (context_flat[b, left_idx, :] + context_flat[b, right_idx, :]) / 2
    
    return context_flat


def downsample_by_deletion_with_region(context, downsample_factor=2, recent_fraction=0.25):
    """
    Downsample context by deleting samples instead of interpolating.
    Keeps recent fraction unchanged and deletes from older portion.
    
    Args:
        context: Input tensor
        downsample_factor: Desired overall downsampling factor
        recent_fraction: Fraction of most recent samples to keep unchanged
    
    Returns:
        Downsampled context with reduced length
    """
    if downsample_factor == 1:
        return context
    
    original_shape = context.shape
    
    # Handle different tensor dimensions
    if len(original_shape) == 1:
        # 1D tensor: (context_length,)
        length = original_shape[0]
        recent_samples_count = int(length * recent_fraction)
        older_samples_count = length - recent_samples_count
        
        effective_factor, _ = calculate_effective_downsampling(length, downsample_factor, recent_fraction)
        
        # Keep every effective_factor-th sample from older portion
        older_indices = list(range(0, older_samples_count, effective_factor))
        # Keep all recent samples
        recent_indices = list(range(older_samples_count, length))
        
        # Combine indices
        keep_indices = older_indices + recent_indices
        keep_indices.sort()
        
        return context[keep_indices]
    
    elif len(original_shape) == 2:
        if original_shape[1] > original_shape[0]:  # Assume (batch_size, context_length)
            batch_size, length = original_shape
            recent_samples_count = int(length * recent_fraction)
            older_samples_count = length - recent_samples_count
            
            effective_factor, _ = calculate_effective_downsampling(length, downsample_factor, recent_fraction)
            
            # Keep every effective_factor-th sample from older portion
            older_indices = list(range(0, older_samples_count, effective_factor))
            # Keep all recent samples
            recent_indices = list(range(older_samples_count, length))
            
            # Combine indices
            keep_indices = older_indices + recent_indices
            keep_indices.sort()
            
            return context[:, keep_indices]
        else:  # Assume (context_length, features)
            length, features = original_shape
            recent_samples_count = int(length * recent_fraction)
            older_samples_count = length - recent_samples_count
            
            effective_factor, _ = calculate_effective_downsampling(length, downsample_factor, recent_fraction)
            
            # Keep every effective_factor-th sample from older portion
            older_indices = list(range(0, older_samples_count, effective_factor))
            # Keep all recent samples
            recent_indices = list(range(older_samples_count, length))
            
            # Combine indices
            keep_indices = older_indices + recent_indices
            keep_indices.sort()
            
            return context[keep_indices, :]
    
    elif len(original_shape) == 3:
        # 3D tensor: (batch_size, context_length, features)
        batch_size, length, features = original_shape
        recent_samples_count = int(length * recent_fraction)
        older_samples_count = length - recent_samples_count
        
        effective_factor, _ = calculate_effective_downsampling(length, downsample_factor, recent_fraction)
        
        # Keep every effective_factor-th sample from older portion
        older_indices = list(range(0, older_samples_count, effective_factor))
        # Keep all recent samples
        recent_indices = list(range(older_samples_count, length))
        
        # Combine indices
        keep_indices = older_indices + recent_indices
        keep_indices.sort()
        
        return context[:, keep_indices, :]
    
    return context

def downsample_recent_only(context, downsample_factor=2, recent_fraction=0.25):
    """
    Keep only the most recent effective_length samples and delete the rest.
    
    Args:
        context: Input tensor
        downsample_factor: Desired overall downsampling factor
        recent_fraction: Not used in this method, kept for consistency
    
    Returns:
        Context with only the most recent effective_length samples
    """
    if downsample_factor == 1:
        return context
    
    original_shape = context.shape
    
    # Calculate effective length (target length after downsampling)
    if len(original_shape) == 1:
        original_length = original_shape[0]
        effective_length = original_length // downsample_factor
        return context[-effective_length:]
    
    elif len(original_shape) == 2:
        if original_shape[1] > original_shape[0]:  # Assume (batch_size, context_length)
            batch_size, original_length = original_shape
            effective_length = original_length // downsample_factor
            return context[:, -effective_length:]
        else:  # Assume (context_length, features)
            original_length, features = original_shape
            effective_length = original_length // downsample_factor
            return context[-effective_length:, :]
    
    elif len(original_shape) == 3:
        # 3D tensor: (batch_size, context_length, features)
        batch_size, original_length, features = original_shape
        effective_length = original_length // downsample_factor
        return context[:, -effective_length:, :]
    
    return context

def downsample_uniform(context, downsample_factor=2, recent_fraction=0.25):
    """
    Uniformly downsample the entire signal by the downsample factor.
    
    Args:
        context: Input tensor
        downsample_factor: Factor to downsample by
        recent_fraction: Not used in this method, kept for consistency
    
    Returns:
        Uniformly downsampled context
    """
    if downsample_factor == 1:
        return context
    
    original_shape = context.shape
    
    if len(original_shape) == 1:
        # 1D tensor: (context_length,)
        return context[::downsample_factor]
    
    elif len(original_shape) == 2:
        if original_shape[1] > original_shape[0]:  # Assume (batch_size, context_length)
            return context[:, ::downsample_factor]
        else:  # Assume (context_length, features)
            return context[::downsample_factor, :]
    
    elif len(original_shape) == 3:
        # 3D tensor: (batch_size, context_length, features)
        return context[:, ::downsample_factor, :]
    
    return context

def upsample_predictions(predictions, upsample_factor):
    """
    Upsample predictions using linear interpolation.
    
    Args:
        predictions: Tensor of shape (batch_size, prediction_length)
        upsample_factor: Factor to upsample by
    
    Returns:
        Upsampled predictions
    """
    if upsample_factor == 1:
        return predictions
    
    batch_size, pred_length = predictions.shape
    target_length = pred_length * upsample_factor
    
    # Use linear interpolation
    upsampled = torch.nn.functional.interpolate(
        predictions.unsqueeze(1),  # Add channel dimension
        size=target_length,
        mode='linear',
        align_corners=False
    ).squeeze(1)  # Remove channel dimension
    
    return upsampled

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
    def __init__(self, model_path, device, context_length, prediction_length, 
                 downsample_factor=1, recent_fraction=0.25, downsample_method='interpolate', **kwargs):
        try:
            from time_moe.models.modeling_time_moe import TimeMoeForPrediction
            model = TimeMoeForPrediction.from_pretrained(
                model_path,
                device_map=device,
                # attn_implementation='flash_attention_2',
                torch_dtype='auto',
            )
        except:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                # attn_implementation='flash_attention_2',
                torch_dtype='auto',
                trust_remote_code=True,
            )

        logging.info(f'>>> Model dtype: {model.dtype}; Attention:{model.config._attn_implementation}')
        logging.info(f'>>> Downsampling factor: {downsample_factor}')
        logging.info(f'>>> Recent fraction preserved: {recent_fraction:.2%}')
        logging.info(f'>>> Downsampling method: {downsample_method}')
        
        if downsample_factor > 1:
            if downsample_method == 'recent_only':
                effective_length = context_length // downsample_factor
                logging.info(f'>>> Recent-only method: keeping last {effective_length} samples out of {context_length}')
            elif downsample_method == 'uniform':
                effective_length = context_length // downsample_factor
                effective_pred_length = prediction_length // downsample_factor
                logging.info(f'>>> Uniform method: downsampling to {effective_length} context samples and {effective_pred_length} prediction samples')
            else:
                effective_factor, effective_length = calculate_effective_downsampling(
                    context_length, downsample_factor, recent_fraction
                )
                logging.info(f'>>> Effective downsampling factor for older portion: {effective_factor}')
                logging.info(f'>>> Target effective context length: {context_length // downsample_factor}')
                logging.info(f'>>> Actual effective context length: {effective_length}')

        self.model = model
        self.device = device
        self.prediction_length = prediction_length
        self.downsample_factor = downsample_factor
        self.recent_fraction = recent_fraction
        self.downsample_method = downsample_method
        self.model.eval()

    def predict(self, batch):
        model = self.model
        device = self.device
        prediction_length = self.prediction_length

        # Apply region-selective downsampling to inputs
        inputs = batch['inputs'].to(device).to(model.dtype)
        
        # For uniform method, also adjust prediction length
        if self.downsample_method == 'uniform' and self.downsample_factor > 1:
            effective_prediction_length = prediction_length // self.downsample_factor
        else:
            effective_prediction_length = prediction_length
        
        if self.downsample_factor > 1:
            if self.downsample_method == 'interpolate':
                inputs = downsample_and_interpolate_with_region(
                    inputs, self.downsample_factor, self.recent_fraction
                )
            elif self.downsample_method == 'delete':
                inputs = downsample_by_deletion_with_region(
                    inputs, self.downsample_factor, self.recent_fraction
                )
            elif self.downsample_method == 'recent_only':
                inputs = downsample_recent_only(
                    inputs, self.downsample_factor, self.recent_fraction
                )
            elif self.downsample_method == 'uniform':
                inputs = downsample_uniform(
                    inputs, self.downsample_factor, self.recent_fraction
                )

        outputs = model.generate(
            inputs=inputs,
            max_new_tokens=effective_prediction_length,
        )
        preds = outputs[:, -effective_prediction_length:]
        
        # For uniform method, upsample predictions back to original length
        if self.downsample_method == 'uniform' and self.downsample_factor > 1:
            preds = upsample_predictions(preds, self.downsample_factor)
        
        labels = batch['labels'].to(device)
        if len(preds.shape) > len(labels.shape):
            labels = labels[..., None]
        return preds, labels


def evaluate_single_config(args, downsample_factor=1, recent_fraction=0.25, downsample_method='interpolate'):
    """Evaluate with a specific downsampling configuration"""
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
        downsample_factor=downsample_factor,
        recent_fraction=recent_fraction,
        downsample_method=downsample_method
    )
    
    if args.data.endswith('.csv'):
        dataset = BenchmarkEvalDataset(
            args.data,
            context_length=context_length,
            prediction_length=prediction_length,
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
        for idx, batch in enumerate(tqdm(test_dl, desc=f"Evaluating {downsample_method} downsample_factor={downsample_factor}, recent_fraction={recent_fraction:.1%}")):
            preds, labels = model.predict(batch)

            for metric in metric_list:
                metric.push(preds, labels)

            acc_count += count_num_tensor_elements(preds)

            # if idx >=100: break
            

    ret_metric = {}
    for metric in metric_list:
        ret_metric[metric.name] = metric.value / acc_count
    print(f'{rank} - {downsample_method.title()} downsample factor {downsample_factor}, Recent fraction {recent_fraction:.1%}: {ret_metric}')

    metric_tensors = [metric.value for metric in metric_list] + [acc_count]
    if is_dist:
        stat_tensor = torch.tensor(metric_tensors).to(model.device)
        gathered_results = [torch.zeros_like(stat_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_results, stat_tensor)
        all_stat = torch.stack(gathered_results, dim=0).sum(dim=0)
    else:
        all_stat = metric_tensors

    result_item = None
    if rank == 0:
        item = {
            'model': args.model,
            'data': args.data,
            'context_length': args.context_length,
            'prediction_length': args.prediction_length,
            'downsample_factor': downsample_factor,
            'recent_fraction': recent_fraction,
            'downsample_method': downsample_method,
        }

        count = all_stat[-1]
        for i, metric in enumerate(metric_list):
            val = all_stat[i] / count
            item[metric.name] = float(val.cpu().numpy())
        logging.info(item)
        result_item = item

    return result_item


def evaluate(args):
    """Main evaluation function that compares different downsampling strategies"""
    
    # Test different configurations
    configs = []
    
    # Always test original (no downsampling)
    configs.append((1, 0.25, 'interpolate'))  # downsample_factor=1, method doesn't matter
    
    # Test different downsample factors with specified recent fractions and methods
    for factor in args.downsample_factors:
        if factor > 1:  # Only apply region selection for actual downsampling
            for recent_frac in args.recent_fractions:
                for method in args.downsample_methods:
                    if method in ['recent_only', 'uniform']:
                        # For recent_only and uniform, we only need to test once per factor (recent_frac doesn't matter)
                        if recent_frac == args.recent_fractions[0]:  # Only add once
                            configs.append((factor, recent_frac, method))
                    else:
                        configs.append((factor, recent_frac, method))
    
    results = []
    
    print("="*120)
    print("COMPARATIVE EVALUATION: ORIGINAL vs REGION-SELECTIVE DOWNSAMPLED (INTERPOLATE vs DELETE vs RECENT-ONLY vs UNIFORM)")
    print("="*120)
    
    for downsample_factor, recent_fraction, method in configs:
        print(f"\n{'='*80}")
        if downsample_factor == 1:
            print(f"EVALUATING: ORIGINAL (No Downsampling)")
        else:
            if method == 'recent_only':
                effective_length = args.context_length // downsample_factor
                print(f"EVALUATING: {method.upper().replace('_', '-')} (Factor = {downsample_factor})")
                print(f"  - Keeping only last {effective_length} samples out of {args.context_length}")
                print(f"  - Discarding oldest {args.context_length - effective_length} samples")
            elif method == 'uniform':
                effective_context_length = args.context_length // downsample_factor
                effective_pred_length = args.prediction_length // downsample_factor
                print(f"EVALUATING: {method.upper()} (Factor = {downsample_factor})")
                print(f"  - Uniformly downsampling context: {args.context_length} → {effective_context_length}")
                print(f"  - Downsampled prediction length: {args.prediction_length} → {effective_pred_length}")
                print(f"  - Will upsample predictions back to {args.prediction_length}")
            else:
                preserved_percent = recent_fraction * 100
                downsampled_percent = (1 - recent_fraction) * 100
                target_effective = args.context_length // downsample_factor
                effective_factor, actual_effective = calculate_effective_downsampling(
                    args.context_length, downsample_factor, recent_fraction
                )
                print(f"EVALUATING: {method.upper()} DOWNSAMPLED (Factor = {downsample_factor})")
                print(f"  - Recent {preserved_percent:.0f}% preserved ({int(args.context_length * recent_fraction)} samples)")
                print(f"  - Older {downsampled_percent:.0f}% downsampled by factor {effective_factor}")
                print(f"  - Target effective length: {target_effective}")
                print(f"  - Actual effective length: {actual_effective}")
        print(f"{'='*80}")
        
        result = evaluate_single_config(args, downsample_factor, recent_fraction, method)
        if result:
            results.append(result)
    
    # Compare results
    if len(results) >= 2 and int(os.getenv('RANK', 0)) == 0:
        print("\n" + "="*120)
        print("PERFORMANCE COMPARISON SUMMARY")
        print("="*120)
        
        original_result = next((r for r in results if r['downsample_factor'] == 1), None)
        
        print(f"{'Method':<40} {'MSE':<15} {'MAE':<15} {'MSE Change':<15} {'MAE Change':<15}")
        print("-" * 120)
        
        for result in results:
            factor = result['downsample_factor']
            recent_frac = result['recent_fraction']
            method = result['downsample_method']
            mse = result['mse']
            mae = result['mae']
            
            if factor == 1:
                method_name = "Original"
                mse_change = "Baseline"
                mae_change = "Baseline"
            else:
                if method in ['recent_only', 'uniform']:
                    method_name = f"{method.replace('_', '-').title()}-DS{factor}"
                else:
                    method_name = f"{method.title()}-DS{factor} (R{recent_frac:.0%})"
                
                if original_result:
                    mse_change = f"{((mse - original_result['mse']) / original_result['mse'] * 100):+.2f}%"
                    mae_change = f"{((mae - original_result['mae']) / original_result['mae'] * 100):+.2f}%"
                else:
                    mse_change = "N/A"
                    mae_change = "N/A"
            
            print(f"{method_name:<40} {mse:<15.6f} {mae:<15.6f} {mse_change:<15} {mae_change:<15}")
        
        # Analysis
        print("\n" + "="*120)
        print("DETAILED ANALYSIS")
        print("="*120)
        
        if original_result and len(results) > 1:
            downsampled_results = [r for r in results if r['downsample_factor'] != 1]
            
            best_mse_result = min(results, key=lambda x: x['mse'])
            best_mae_result = min(results, key=lambda x: x['mae'])
            
            if best_mse_result['downsample_method'] in ['recent_only', 'uniform']:
                best_mse_desc = f"{best_mse_result['downsample_method'].replace('_', '-').title()}-{best_mse_result['downsample_factor']}x"
            else:
                best_mse_desc = f"{best_mse_result['downsample_method'].title()}-{best_mse_result['downsample_factor']}x with {best_mse_result['recent_fraction']:.0%} recent"
            
            if best_mae_result['downsample_method'] in ['recent_only', 'uniform']:
                best_mae_desc = f"{best_mae_result['downsample_method'].replace('_', '-').title()}-{best_mae_result['downsample_factor']}x"
            else:
                best_mae_desc = f"{best_mae_result['downsample_method'].title()}-{best_mae_result['downsample_factor']}x with {best_mae_result['recent_fraction']:.0%} recent"
            
            print(f"Best MSE: {best_mse_desc} (MSE: {best_mse_result['mse']:.6f})")
            print(f"Best MAE: {best_mae_desc} (MAE: {best_mae_result['mae']:.6f})")
            
            print("\nPer-configuration analysis:")
            for ds_result in downsampled_results:
                factor = ds_result['downsample_factor']
                recent_frac = ds_result['recent_fraction']
                method = ds_result['downsample_method']
                mse_improvement = (original_result['mse'] - ds_result['mse']) / original_result['mse'] * 100
                mae_improvement = (original_result['mae'] - ds_result['mae']) / original_result['mae'] * 100
                
                if method in ['recent_only', 'uniform']:
                    config_name = f"{method.replace('_', '-').title()}-{factor}x"
                else:
                    config_name = f"{method.title()}-{factor}x with {recent_frac:.0%} recent"
                
                if mse_improvement > 0 and mae_improvement > 0:
                    print(f"✓ {config_name}: improves both MSE ({mse_improvement:.2f}%) and MAE ({mae_improvement:.2f}%)")
                elif mse_improvement > 0:
                    print(f"~ {config_name}: improves MSE ({mse_improvement:.2f}%) but degrades MAE ({mae_improvement:.2f}%)")
                elif mae_improvement > 0:
                    print(f"~ {config_name}: improves MAE ({mae_improvement:.2f}%) but degrades MSE ({mae_improvement:.2f}%)")
                else:
                    print(f"✗ {config_name}: degrades both MSE ({mse_improvement:.2f}%) and MAE ({mae_improvement:.2f}%)")
            
            # Method comparison
            print("\nMethod comparison (Interpolate vs Delete vs Recent-Only vs Uniform):")
            interpolate_results = [r for r in downsampled_results if r['downsample_method'] == 'interpolate']
            delete_results = [r for r in downsampled_results if r['downsample_method'] == 'delete']
            recent_only_results = [r for r in downsampled_results if r['downsample_method'] == 'recent_only']
            uniform_results = [r for r in downsampled_results if r['downsample_method'] == 'uniform']
            
            method_stats = []
            
            if interpolate_results:
                avg_mse_interp = np.mean([r['mse'] for r in interpolate_results])
                avg_mae_interp = np.mean([r['mae'] for r in interpolate_results])
                method_stats.append(("Interpolate", avg_mse_interp, avg_mae_interp))
                print(f"Average MSE/MAE - Interpolate: {avg_mse_interp:.6f} / {avg_mae_interp:.6f}")
            
            if delete_results:
                avg_mse_delete = np.mean([r['mse'] for r in delete_results])
                avg_mae_delete = np.mean([r['mae'] for r in delete_results])
                method_stats.append(("Delete", avg_mse_delete, avg_mae_delete))
                print(f"Average MSE/MAE - Delete: {avg_mse_delete:.6f} / {avg_mae_delete:.6f}")
            
            if recent_only_results:
                avg_mse_recent = np.mean([r['mse'] for r in recent_only_results])
                avg_mae_recent = np.mean([r['mae'] for r in recent_only_results])
                method_stats.append(("Recent-Only", avg_mse_recent, avg_mae_recent))
                print(f"Average MSE/MAE - Recent-Only: {avg_mse_recent:.6f} / {avg_mae_recent:.6f}")
            
            if uniform_results:
                avg_mse_uniform = np.mean([r['mse'] for r in uniform_results])
                avg_mae_uniform = np.mean([r['mae'] for r in uniform_results])
                method_stats.append(("Uniform", avg_mse_uniform, avg_mae_uniform))
                print(f"Average MSE/MAE - Uniform: {avg_mse_uniform:.6f} / {avg_mae_uniform:.6f}")
            
            if len(method_stats) > 1:
                best_mse_method = min(method_stats, key=lambda x: x[1])
                best_mae_method = min(method_stats, key=lambda x: x[2])
                print(f"→ Best method for MSE: {best_mse_method[0]}")
                print(f"→ Best method for MAE: {best_mae_method[0]}")
        
        # Save results to file
        results_file = f"effective_length_comparison_{os.path.basename(args.data)}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {results_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TimeMoE Evaluate with Effective Context Length Control')
    parser.add_argument(
        '--model', '-m',
        type=str,
        # default='Maple728/TimeMoE-50M',
        default='/home/sa53869/time_series/time-moe/model_weights/time-moe-50m',
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
        default=[1, 2],
        help='List of downsampling factors to test (default: [1, 2])'
    )
    parser.add_argument(
        '--recent_fractions', '-r',
        type=float,
        nargs='+',
        default=[0.25],
        help='List of recent fractions to preserve (default: [0.25] for 25%%)'
    )
    parser.add_argument(
        '--downsample_methods',
        type=str,
        nargs='+',
        choices=['interpolate', 'delete', 'recent_only', 'uniform'],
        default=['interpolate', 'delete', 'recent_only', 'uniform'],
        help='Downsampling methods to test (default: all four methods)'
    )
    
    args = parser.parse_args()
    
    # Validate recent_fractions
    for frac in args.recent_fractions:
        if not 0.0 <= frac <= 1.0:
            raise ValueError(f"Recent fraction must be between 0.0 and 1.0, got {frac}")
    
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