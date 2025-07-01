#!/usr/bin/env python
# -*- coding:utf-8 _*-
import json
import os
import argparse
import numpy as np
import logging
import torch
import torch.distributed as dist
import torch.nn.functional as F
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


def count_unique_samples(context):
    """Count remaining original samples after pruning (not true unique values)."""
    if len(context.shape) == 1:
        return context.shape[0]
    elif len(context.shape) == 2:
        if context.shape[1] > context.shape[0]:  # (batch_size, context_length)
            return context.shape[1]
        else:  # (context_length, features)
            return context.shape[0]
    elif len(context.shape) == 3:  # (batch_size, context_length, features)
        return context.shape[1]
    return context.shape[-1]


def calculate_effective_downsampling(original_length, downsample_factor, recent_fraction):
    """
    Calculate the effective downsampling factor needed for the older portion
    to maintain the desired effective context length.
    """
    target_effective_length = original_length // downsample_factor
    recent_samples_count = int(original_length * recent_fraction)
    older_samples_count = original_length - recent_samples_count
    
    target_older_effective = target_effective_length - recent_samples_count
    if target_older_effective <= 0 or older_samples_count <= 0:
        return 1, recent_samples_count
    
    effective_downsample_factor = older_samples_count / target_older_effective
    effective_downsample_factor = max(1, round(effective_downsample_factor))
    
    actual_older_effective = older_samples_count // effective_downsample_factor
    actual_effective_length = recent_samples_count + actual_older_effective
    
    return effective_downsample_factor, actual_effective_length


def downsample_and_interpolate_with_region(context, downsample_factor=2, recent_fraction=0.25):
    """Downsample context by keeping recent fraction unchanged and adaptively downsampling the rest."""
    if downsample_factor == 1:
        return context
    
    original_shape = context.shape
    context_flat = context.clone()
    
    if len(original_shape) == 1:
        length = original_shape[0]
        recent_samples_count = int(length * recent_fraction)
        older_samples_count = length - recent_samples_count
        
        effective_factor, _ = calculate_effective_downsampling(length, downsample_factor, recent_fraction)
        
        for i in range(1, older_samples_count, effective_factor):
            if i + 1 < older_samples_count:
                left_idx = max(0, i - 1)
                right_idx = min(older_samples_count - 1, i + 1)
                context_flat[i] = (context_flat[left_idx] + context_flat[right_idx]) / 2
    
    elif len(original_shape) == 2:
        if original_shape[1] > original_shape[0]:  # (batch_size, context_length)
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
        else:  # (context_length, features)
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
    """Downsample context by deleting samples instead of interpolating."""
    if downsample_factor == 1:
        return context
    
    original_shape = context.shape
    
    if len(original_shape) == 1:
        length = original_shape[0]
        recent_samples_count = int(length * recent_fraction)
        older_samples_count = length - recent_samples_count
        
        effective_factor, _ = calculate_effective_downsampling(length, downsample_factor, recent_fraction)
        
        older_indices = list(range(0, older_samples_count, effective_factor))
        recent_indices = list(range(older_samples_count, length))
        
        keep_indices = older_indices + recent_indices
        keep_indices.sort()
        
        return context[keep_indices]
    
    elif len(original_shape) == 2:
        if original_shape[1] > original_shape[0]:  # (batch_size, context_length)
            batch_size, length = original_shape
            recent_samples_count = int(length * recent_fraction)
            older_samples_count = length - recent_samples_count
            
            effective_factor, _ = calculate_effective_downsampling(length, downsample_factor, recent_fraction)
            
            older_indices = list(range(0, older_samples_count, effective_factor))
            recent_indices = list(range(older_samples_count, length))
            
            keep_indices = older_indices + recent_indices
            keep_indices.sort()
            
            return context[:, keep_indices]
        else:  # (context_length, features)
            length, features = original_shape
            recent_samples_count = int(length * recent_fraction)
            older_samples_count = length - recent_samples_count
            
            effective_factor, _ = calculate_effective_downsampling(length, downsample_factor, recent_fraction)
            
            older_indices = list(range(0, older_samples_count, effective_factor))
            recent_indices = list(range(older_samples_count, length))
            
            keep_indices = older_indices + recent_indices
            keep_indices.sort()
            
            return context[keep_indices, :]
    
    elif len(original_shape) == 3:
        batch_size, length, features = original_shape
        recent_samples_count = int(length * recent_fraction)
        older_samples_count = length - recent_samples_count
        
        effective_factor, _ = calculate_effective_downsampling(length, downsample_factor, recent_fraction)
        
        older_indices = list(range(0, older_samples_count, effective_factor))
        recent_indices = list(range(older_samples_count, length))
        
        keep_indices = older_indices + recent_indices
        keep_indices.sort()
        
        return context[:, keep_indices, :]
    
    return context


def downsample_recent_only(context, downsample_factor=2, recent_fraction=0.25):
    """Keep only the most recent effective_length samples and delete the rest."""
    if downsample_factor == 1:
        return context
    
    original_shape = context.shape
    
    if len(original_shape) == 1:
        original_length = original_shape[0]
        effective_length = original_length // downsample_factor
        return context[-effective_length:]
    
    elif len(original_shape) == 2:
        if original_shape[1] > original_shape[0]:  # (batch_size, context_length)
            batch_size, original_length = original_shape
            effective_length = original_length // downsample_factor
            return context[:, -effective_length:]
        else:  # (context_length, features)
            original_length, features = original_shape
            effective_length = original_length // downsample_factor
            return context[-effective_length:, :]
    
    elif len(original_shape) == 3:
        batch_size, original_length, features = original_shape
        effective_length = original_length // downsample_factor
        return context[:, -effective_length:, :]
    
    return context


def downsample_recent_only_zero_prepended(context, downsample_factor=2, recent_fraction=0.25):
    """Keep only the most recent effective_length samples and prepend zeros to match original context length."""
    if downsample_factor == 1:
        return context
    
    original_shape = context.shape
    
    if len(original_shape) == 1:
        original_length = original_shape[0]
        effective_length = original_length // downsample_factor
        recent_only = context[-effective_length:]
        # Prepend zeros
        zeros = torch.zeros(original_length - effective_length, dtype=context.dtype, device=context.device)
        return torch.cat([zeros, recent_only])
    
    elif len(original_shape) == 2:
        if original_shape[1] > original_shape[0]:  # (batch_size, context_length)
            batch_size, original_length = original_shape
            effective_length = original_length // downsample_factor
            recent_only = context[:, -effective_length:]
            # Prepend zeros
            zeros = torch.zeros(batch_size, original_length - effective_length, dtype=context.dtype, device=context.device)
            return torch.cat([zeros, recent_only], dim=1)
        else:  # (context_length, features)
            original_length, features = original_shape
            effective_length = original_length // downsample_factor
            recent_only = context[-effective_length:, :]
            # Prepend zeros
            zeros = torch.zeros(original_length - effective_length, features, dtype=context.dtype, device=context.device)
            return torch.cat([zeros, recent_only], dim=0)
    
    elif len(original_shape) == 3:
        batch_size, original_length, features = original_shape
        effective_length = original_length // downsample_factor
        recent_only = context[:, -effective_length:, :]
        # Prepend zeros
        zeros = torch.zeros(batch_size, original_length - effective_length, features, dtype=context.dtype, device=context.device)
        return torch.cat([zeros, recent_only], dim=1)
    
    return context


def downsample_adaptive(context, downsample_factor=2, recent_fraction=0.25):
    """Adaptive sampling preserving important patterns"""
    if downsample_factor == 1:
        return context
    
    original_shape = context.shape
    target_length = original_shape[-1] // downsample_factor
    
    if len(original_shape) == 1:
        # Calculate importance scores
        variance_weight = torch.var(context).item()
        trend_changes = torch.abs(torch.diff(context))
        
        # Recency bias
        recency_weights = torch.linspace(0.5, 1.0, len(context), device=context.device)
        
        # Trend change importance
        trend_weights = F.pad(trend_changes, (1, 0), value=0)
        trend_weights = F.normalize(trend_weights, dim=0)
        
        # Deviation from mean
        deviation_weights = torch.abs(context - torch.mean(context))
        deviation_weights = F.normalize(deviation_weights, dim=0)
        
        # Combined importance score
        importance = 0.4 * recency_weights + 0.3 * trend_weights + 0.3 * deviation_weights
        
        # Select top-k important indices
        _, top_indices = torch.topk(importance, target_length)
        top_indices, _ = torch.sort(top_indices)
        
        return context[top_indices]
    
    elif len(original_shape) == 2:
        if original_shape[1] > original_shape[0]:  # (batch_size, context_length)
            result = torch.zeros(original_shape[0], target_length, dtype=context.dtype, device=context.device)
            for b in range(original_shape[0]):
                result[b] = downsample_adaptive(context[b], downsample_factor, recent_fraction)
            return result
        else:  # (context_length, features)
            # Apply to first feature as reference
            reference_feature = context[:, 0]
            selected_context = downsample_adaptive(reference_feature, downsample_factor, recent_fraction)
            
            # Find indices of selected samples
            indices = []
            for val in selected_context:
                idx = torch.where(torch.abs(reference_feature - val) < 1e-6)[0]
                if len(idx) > 0:
                    indices.append(idx[0].item())
            
            return context[indices, :]
    
    elif len(original_shape) == 3:
        result = torch.zeros(original_shape[0], target_length, original_shape[2], dtype=context.dtype, device=context.device)
        for b in range(original_shape[0]):
            result[b] = downsample_adaptive(context[b], downsample_factor, recent_fraction)
        return result
    
    return context


def downsample_multiscale(context, downsample_factor=2, recent_fraction=0.25):
    """Multi-scale sampling: recent fine-grained + distant coarse-grained"""
    if downsample_factor == 1:
        return context
    
    original_shape = context.shape
    target_length = original_shape[-1] // downsample_factor
    
    if len(original_shape) == 1:
        original_length = len(context)
        recent_count = max(1, target_length // 4)  # 25% for recent samples
        recent_samples = context[-recent_count:]
        
        # Remaining from older portion
        older_portion = context[:-recent_count]
        older_needed = target_length - recent_count
        
        if older_needed > 0 and len(older_portion) > 0:
            stride = max(1, len(older_portion) // older_needed)
            older_indices = list(range(0, len(older_portion), stride))[:older_needed]
            older_samples = older_portion[older_indices]
            
            return torch.cat([older_samples, recent_samples])
        else:
            return recent_samples
    
    elif len(original_shape) == 2:
        if original_shape[1] > original_shape[0]:  # (batch_size, context_length)
            result = torch.zeros(original_shape[0], target_length, dtype=context.dtype, device=context.device)
            for b in range(original_shape[0]):
                result[b] = downsample_multiscale(context[b], downsample_factor, recent_fraction)
            return result
        else:  # (context_length, features)
            sampled_1d = downsample_multiscale(context[:, 0], downsample_factor, recent_fraction)
            
            # Find corresponding indices in original context
            indices = []
            for val in sampled_1d:
                idx = torch.where(torch.abs(context[:, 0] - val) < 1e-6)[0]
                if len(idx) > 0:
                    indices.append(idx[0].item())
            
            return context[indices, :]
    
    elif len(original_shape) == 3:
        result = torch.zeros(original_shape[0], target_length, original_shape[2], dtype=context.dtype, device=context.device)
        for b in range(original_shape[0]):
            result[b] = downsample_multiscale(context[b], downsample_factor, recent_fraction)
        return result
    
    return context


def downsample_pattern_aware(context, downsample_factor=2, recent_fraction=0.25):
    """Pattern-aware sampling preserving critical points"""
    if downsample_factor == 1:
        return context
    
    original_shape = context.shape
    target_length = original_shape[-1] // downsample_factor
    
    if len(original_shape) == 1:
        # Find critical points
        gradients = torch.diff(context)
        trend_changes = F.pad(torch.abs(torch.diff(gradients)), (2, 0), value=0)
        
        # Score each point
        scores = torch.abs(context - torch.mean(context))  # Deviation importance
        scores += trend_changes  # Trend change importance
        
        # Add recency bias
        recency = torch.linspace(0.1, 1.0, len(context), device=context.device)
        scores *= recency
        
        # Select top scoring points
        _, indices = torch.topk(scores, target_length)
        indices, _ = torch.sort(indices)
        
        return context[indices]
    
    elif len(original_shape) == 2:
        if original_shape[1] > original_shape[0]:  # (batch_size, context_length)
            result = torch.zeros(original_shape[0], target_length, dtype=context.dtype, device=context.device)
            for b in range(original_shape[0]):
                result[b] = downsample_pattern_aware(context[b], downsample_factor, recent_fraction)
            return result
        else:  # (context_length, features)
            sampled_1d = downsample_pattern_aware(context[:, 0], downsample_factor, recent_fraction)
            
            # Find corresponding indices
            indices = []
            for val in sampled_1d:
                idx = torch.where(torch.abs(context[:, 0] - val) < 1e-6)[0]
                if len(idx) > 0:
                    indices.append(idx[0].item())
            
            return context[indices, :]
    
    elif len(original_shape) == 3:
        result = torch.zeros(original_shape[0], target_length, original_shape[2], dtype=context.dtype, device=context.device)
        for b in range(original_shape[0]):
            result[b] = downsample_pattern_aware(context[b], downsample_factor, recent_fraction)
        return result
    
    return context


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
                torch_dtype='auto',
            )
        except:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype='auto',
                trust_remote_code=True,
            )

        logging.info(f'>>> Model dtype: {model.dtype}; Attention:{model.config._attn_implementation}')
        logging.info(f'>>> Downsampling factor: {downsample_factor}')
        logging.info(f'>>> Recent fraction preserved: {recent_fraction:.2%}')
        logging.info(f'>>> Downsampling method: {downsample_method}')
        
        if downsample_factor > 1:
            if downsample_method in ['recent_only', 'recent_only_zero_prepended']:
                effective_length = context_length // downsample_factor
                if downsample_method == 'recent_only':
                    logging.info(f'>>> Recent-only method: keeping last {effective_length} samples out of {context_length}')
                else:
                    logging.info(f'>>> Recent-only-zero-prepended method: keeping last {effective_length} samples, padding to {context_length}')
            elif downsample_method in ['adaptive', 'multiscale', 'pattern_aware']:
                effective_length = context_length // downsample_factor
                logging.info(f'>>> {downsample_method.title()} method: targeting {effective_length} samples out of {context_length}')
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

        # Apply downsampling to inputs
        inputs = batch['inputs'].to(device).to(model.dtype)
        original_samples = inputs.shape[-1]
        original_unique = count_unique_samples(inputs)
        
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
            elif self.downsample_method == 'recent_only_zero_prepended':
                inputs = downsample_recent_only_zero_prepended(
                    inputs, self.downsample_factor, self.recent_fraction
                )
            elif self.downsample_method == 'adaptive':
                inputs = downsample_adaptive(
                    inputs, self.downsample_factor, self.recent_fraction
                )
            elif self.downsample_method == 'multiscale':
                inputs = downsample_multiscale(
                    inputs, self.downsample_factor, self.recent_fraction
                )
            elif self.downsample_method == 'pattern_aware':
                inputs = downsample_pattern_aware(
                    inputs, self.downsample_factor, self.recent_fraction
                )

        processed_samples = inputs.shape[-1]
        processed_unique = count_unique_samples(inputs)

        outputs = model.generate(
            inputs=inputs,
            max_new_tokens=prediction_length,
        )
        preds = outputs[:, -prediction_length:]
        labels = batch['labels'].to(device)
        if len(preds.shape) > len(labels.shape):
            labels = labels[..., None]
        
        return preds, labels, {
            'original_samples': original_samples,
            'original_unique': original_unique,
            'processed_samples': processed_samples,
            'processed_unique': processed_unique
        }


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
            setup_nccl(rank, world_size, master_addr, master_port)
            device = f"cuda:{local_rank}"
            is_dist = True
        else:
            device = "cuda:1" if torch.cuda.is_available() else "cpu"
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
    total_original_samples = 0
    total_original_unique = 0
    total_processed_samples = 0
    total_processed_unique = 0
    batch_count = 0
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_dl, desc=f"Evaluating {downsample_method} downsample_factor={downsample_factor}, recent_fraction={recent_fraction:.1%}")):
            preds, labels, sample_info = model.predict(batch)

            for metric in metric_list:
                metric.push(preds, labels)

            acc_count += count_num_tensor_elements(preds)
            
            # Accumulate sample information
            total_original_samples += sample_info['original_samples']
            total_original_unique += sample_info['original_unique']
            total_processed_samples += sample_info['processed_samples']
            total_processed_unique += sample_info['processed_unique']
            batch_count += 1

            if idx >= 10: 
                break

    # Calculate averages
    avg_original_samples = total_original_samples / batch_count
    avg_original_unique = total_original_unique / batch_count
    avg_processed_samples = total_processed_samples / batch_count
    avg_processed_unique = total_processed_unique / batch_count

    ret_metric = {}
    for metric in metric_list:
        ret_metric[metric.name] = metric.value / acc_count
    
    print(f'{rank} - {downsample_method.title()} downsample factor {downsample_factor}, Recent fraction {recent_fraction:.1%}:')
    print(f'    Metrics: {ret_metric}')
    print(f'    Sample Info: Original={avg_original_samples:.0f} (Unique={avg_original_unique:.0f}), '
          f'Processed={avg_processed_samples:.0f} (Unique={avg_processed_unique:.0f})')

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
            'avg_original_samples': float(avg_original_samples),
            'avg_original_unique': float(avg_original_unique),
            'avg_processed_samples': float(avg_processed_samples),
            'avg_processed_unique': float(avg_processed_unique),
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
    
    configs = []
    
    # Always test original (no downsampling)
    configs.append((1, 0.25, 'interpolate'))
    
    # Test different downsample factors with specified recent fractions and methods
    for factor in args.downsample_factors:
        if factor > 1:
            for recent_frac in args.recent_fractions:
                for method in args.downsample_methods:
                    if method in ['recent_only', 'recent_only_zero_prepended', 'adaptive', 'multiscale', 'pattern_aware']:
                        # For these methods, we only need to test once per factor
                        if recent_frac == args.recent_fractions[0]:
                            configs.append((factor, recent_frac, method))
                    else:
                        configs.append((factor, recent_frac, method))

    results = []
    
    print("="*140)
    print("COMPARATIVE EVALUATION: ORIGINAL vs ALL DOWNSAMPLING METHODS")
    print("="*140)
    
    for downsample_factor, recent_fraction, method in configs:
        print(f"\n{'='*80}")
        if downsample_factor == 1:
            print(f"EVALUATING: ORIGINAL (No Downsampling)")
            print(f"  - Full context length: {args.context_length}")
        else:
            if method in ['recent_only', 'recent_only_zero_prepended', 'adaptive', 'multiscale', 'pattern_aware']:
                effective_length = args.context_length // downsample_factor
                print(f"EVALUATING: {method.upper().replace('_', '-')} (Factor = {downsample_factor})")
                print(f"  - Target samples: {effective_length} out of {args.context_length}")
                if method == 'recent_only':
                    print(f"  - Strategy: Keep only last {effective_length} samples")
                elif method == 'recent_only_zero_prepended':
                    print(f"  - Strategy: Keep only last {effective_length} samples, prepend with zeros to match original context length")
                elif method == 'adaptive':
                    print(f"  - Strategy: Importance-based sampling")
                elif method == 'multiscale':
                    print(f"  - Strategy: Multi-resolution sampling")
                elif method == 'pattern_aware':
                    print(f"  - Strategy: Critical pattern preservation")
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
        print("\n" + "="*140)
        print("PERFORMANCE COMPARISON SUMMARY")
        print("="*140)
        
        original_result = next((r for r in results if r['downsample_factor'] == 1), None)
        
        print(f"{'Method':<35} {'Samples (Unique)':<18} {'MSE':<12} {'MAE':<12} {'MSE Change':<12} {'MAE Change':<12}")
        print("-" * 140)
        
        for result in results:
            factor = result['downsample_factor']
            recent_frac = result['recent_fraction']
            method = result['downsample_method']
            mse = result['mse']
            mae = result['mae']
            proc_samples = result['avg_processed_samples']
            proc_unique = result['avg_processed_unique']
            
            if factor == 1:
                method_name = "Original"
                mse_change = "Baseline"
                mae_change = "Baseline"
            else:
                if method in ['recent_only', 'recent_only_zero_prepended', 'adaptive', 'multiscale', 'pattern_aware']:
                    method_name = f"{method.replace('_', '-').title()}-DS{factor}"
                else:
                    method_name = f"{method.title()}-DS{factor} (R{recent_frac:.0%})"
                
                if original_result:
                    mse_change = f"{((mse - original_result['mse']) / original_result['mse'] * 100):+.2f}%"
                    mae_change = f"{((mae - original_result['mae']) / original_result['mae'] * 100):+.2f}%"
                else:
                    mse_change = "N/A"
                    mae_change = "N/A"
            
            sample_info = f"{proc_samples:.0f} ({proc_unique:.0f})"
            print(f"{method_name:<35} {sample_info:<18} {mse:<12.6f} {mae:<12.6f} {mse_change:<12} {mae_change:<12}")
        
        # Save results to file
        results_file = f"all_methods_comparison_{os.path.basename(args.data)}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {results_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TimeMoE Evaluate with All Downsampling Methods')
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='/home/sa53869/time_series/time-moe/model_weights/time-moe-50m',
        help='Model path'
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='dataset/ETT-small/ETTm2_10k.csv',
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
        choices=['interpolate', 'delete', 'recent_only', 'recent_only_zero_prepended', 'adaptive', 'multiscale', 'pattern_aware'],
        default=['interpolate', 'delete', 'recent_only', 'recent_only_zero_prepended', 'adaptive', 'multiscale', 'pattern_aware'],
        help='Downsampling methods to test (default: all methods)'
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