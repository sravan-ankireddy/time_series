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
        self.context_length = context_length
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

    def predict_single_step(self, context):
        """Predict a single next token given context"""
        model = self.model
        device = self.device
        
        with torch.no_grad():
            outputs = model.generate(
                inputs=context.to(device).to(model.dtype),
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=model.config.eos_token_id if hasattr(model.config, 'eos_token_id') else 0
            )
            pred = outputs[:, -1:]  # Get only the last predicted token
        return pred

    def compute_importance_scores(self, inputs, preserve_ratio=0.25):
        """
        Compute importance scores for each sample in the context based on prediction error.
        
        Args:
            inputs: Input tensor of shape [batch_size, context_length] or [batch_size, context_length, features]
            preserve_ratio: Ratio of recent context to preserve (default 0.25)
            
        Returns:
            importance_scores: Tensor of shape [batch_size, context_length] with importance scores
        """
        batch_size, context_length = inputs.shape[0], inputs.shape[1]
        preserve_length = int(preserve_ratio * context_length)
        
        # Ensure inputs are on the correct device
        inputs = inputs.to(self.device)
        
        # Initialize importance scores
        importance_scores = torch.zeros(batch_size, context_length, device=self.device)
        
        # Recent samples (last preserve_length) are always preserved with high importance
        importance_scores[:, -preserve_length:] = 1.0
        
        # For older samples, compute importance based on prediction error
        for i in range(preserve_length, context_length):
            # Start position for sliding window
            start_pos = i - preserve_length
            end_pos = i
            
            # Extract context window (from start_pos to end_pos)
            context_window = inputs[:, start_pos:end_pos]
            
            # Get the true next value
            true_next = inputs[:, end_pos:end_pos+1].to(self.device)
            
            # Predict the next value using the context window
            pred_next = self.predict_single_step(context_window)
            
            # Ensure both tensors are on the same device
            pred_next = pred_next.to(self.device)
            true_next = true_next.to(self.device)
            
            # Compute prediction error (using relative/percentage error)
            # Add small epsilon to avoid division by zero
            epsilon = 1e-8
            abs_error = torch.abs(pred_next - true_next)
            relative_error = abs_error / (torch.abs(true_next) + epsilon)
            
            # Store the importance score for the sample at position start_pos
            # Higher error means higher importance
            importance_scores[:, start_pos] = relative_error.squeeze()
        
        return importance_scores

    def prune_context(self, inputs, pruning_ratio=0.5, preserve_ratio=0.25):
        """
        Prune the context based on importance scores.
        
        Args:
            inputs: Input tensor of shape [batch_size, context_length]
            pruning_ratio: Ratio of samples to remove (default 0.5)
            preserve_ratio: Ratio of recent context to preserve (default 0.25)
            
        Returns:
            pruned_inputs: Pruned input tensor
            selected_indices: Indices of selected samples
        """
        batch_size, context_length = inputs.shape[0], inputs.shape[1]
        preserve_length = int(preserve_ratio * context_length)
        
        # Ensure inputs are on the correct device
        inputs = inputs.to(self.device)
        
        # Compute importance scores
        importance_scores = self.compute_importance_scores(inputs, preserve_ratio)
        
        # Calculate how many samples to keep
        total_keep = int(context_length * (1 - pruning_ratio))
        older_keep = max(0, total_keep - preserve_length)
        
        # Always keep recent samples
        recent_indices = torch.arange(context_length - preserve_length, context_length, device=self.device)
        
        # For older samples, select based on importance scores
        if older_keep > 0:
            older_scores = importance_scores[:, :context_length - preserve_length]
            # Get top-k most important older samples
            _, older_top_indices = torch.topk(older_scores, k=min(older_keep, older_scores.shape[1]), dim=1)
            
            # Combine indices for each batch item
            selected_indices_list = []
            for b in range(batch_size):
                batch_older_indices = older_top_indices[b]
                batch_selected = torch.cat([batch_older_indices, recent_indices])
                batch_selected = torch.sort(batch_selected)[0]  # Sort to maintain order
                selected_indices_list.append(batch_selected)
            
            # Pad to same length and stack
            max_len = max(len(indices) for indices in selected_indices_list)
            padded_indices = []
            for indices in selected_indices_list:
                if len(indices) < max_len:
                    # Pad with the last index (will be masked later)
                    padding = torch.full((max_len - len(indices),), indices[-1], device=self.device)
                    indices = torch.cat([indices, padding])
                padded_indices.append(indices)
            
            selected_indices = torch.stack(padded_indices, dim=0)
        else:
            # Only keep recent samples
            selected_indices = recent_indices.unsqueeze(0).expand(batch_size, -1)
        
        # Extract selected samples
        batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1)
        if len(inputs.shape) == 3:  # [batch_size, context_length, features]
            pruned_inputs = inputs[batch_indices, selected_indices]
        else:  # [batch_size, context_length]
            pruned_inputs = inputs[batch_indices, selected_indices]
        
        return pruned_inputs, selected_indices

    def prune_context_random(self, inputs, pruning_ratio=0.5, preserve_ratio=0.25):
        """
        Prune the context randomly (for comparison with importance-based pruning).
        
        Args:
            inputs: Input tensor of shape [batch_size, context_length]
            pruning_ratio: Ratio of samples to remove (default 0.5)
            preserve_ratio: Ratio of recent context to preserve (default 0.25)
            
        Returns:
            pruned_inputs: Pruned input tensor
            selected_indices: Indices of selected samples
        """
        batch_size, context_length = inputs.shape[0], inputs.shape[1]
        preserve_length = int(preserve_ratio * context_length)
        
        # Ensure inputs are on the correct device
        inputs = inputs.to(self.device)
        
        # Calculate how many samples to keep
        total_keep = int(context_length * (1 - pruning_ratio))
        older_keep = max(0, total_keep - preserve_length)
        
        # Always keep recent samples
        recent_indices = torch.arange(context_length - preserve_length, context_length, device=self.device)
        
        # For older samples, select randomly
        if older_keep > 0:
            older_available = context_length - preserve_length
            
            # Randomly select older samples for each batch item
            selected_indices_list = []
            for b in range(batch_size):
                # Create random permutation of older indices
                older_indices = torch.randperm(older_available, device=self.device)[:older_keep]
                
                # Combine with recent indices
                batch_selected = torch.cat([older_indices, recent_indices])
                batch_selected = torch.sort(batch_selected)[0]  # Sort to maintain order
                selected_indices_list.append(batch_selected)
            
            # Pad to same length and stack
            max_len = max(len(indices) for indices in selected_indices_list)
            padded_indices = []
            for indices in selected_indices_list:
                if len(indices) < max_len:
                    # Pad with the last index
                    padding = torch.full((max_len - len(indices),), indices[-1], device=self.device)
                    indices = torch.cat([indices, padding])
                padded_indices.append(indices)
            
            selected_indices = torch.stack(padded_indices, dim=0)
        else:
            # Only keep recent samples
            selected_indices = recent_indices.unsqueeze(0).expand(batch_size, -1)
        
        # Extract selected samples
        batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1)
        if len(inputs.shape) == 3:  # [batch_size, context_length, features]
            pruned_inputs = inputs[batch_indices, selected_indices]
        else:  # [batch_size, context_length]
            pruned_inputs = inputs[batch_indices, selected_indices]
        
        return pruned_inputs, selected_indices

    def predict_with_pruning(self, batch, pruning_ratio=0.5, preserve_ratio=0.25, use_random=False):
        """Predict with context pruning based on importance scores or random selection"""
        inputs = batch['inputs'].to(self.device)
        
        # Apply pruning (either importance-based or random)
        if use_random:
            pruned_inputs, selected_indices = self.prune_context_random(
                inputs, pruning_ratio=pruning_ratio, preserve_ratio=preserve_ratio
            )
        else:
            pruned_inputs, selected_indices = self.prune_context(
                inputs, pruning_ratio=pruning_ratio, preserve_ratio=preserve_ratio
            )
        
        # Create new batch with pruned inputs
        pruned_batch = {'inputs': pruned_inputs}
        
        # Copy any other keys from original batch (like labels)
        for key, value in batch.items():
            if key != 'inputs':
                pruned_batch[key] = value
        
        # Predict using pruned context
        preds, labels = self.predict(pruned_batch)
        
        return preds, labels


def evaluate(args):
    batch_size = args.batch_size
    context_length = args.context_length
    prediction_length = args.prediction_length
    pruning_ratio = getattr(args, 'pruning_ratio', 0.0)
    preserve_ratio = getattr(args, 'preserve_ratio', 0.25)
    compare_random = getattr(args, 'compare_random', False)

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
            device = "cuda:3" if torch.cuda.is_available() else "cpu"
            is_dist = False
    else:
        device = 'cpu'
        is_dist = False

    # evaluation metrics
    metric_list = [
        MSEMetric(name='mse'),
        MAEMetric(name='mae'),
    ]

    # If comparing with random, create separate metric lists
    if compare_random and pruning_ratio > 0:
        metric_list_random = [
            MSEMetric(name='mse_random'),
            MAEMetric(name='mae_random'),
        ]
    else:
        metric_list_random = []

    model = TimeMoE(
        args.model,
        device,
        context_length=context_length,
        prediction_length=prediction_length
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
    acc_count_random = 0
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_dl)):
            if pruning_ratio > 0:
                # Use importance-based pruning
                preds, labels = model.predict_with_pruning(
                    batch, 
                    pruning_ratio=pruning_ratio, 
                    preserve_ratio=preserve_ratio,
                    use_random=False
                )
                
                # If comparing with random, also run random pruning
                if compare_random:
                    preds_random, labels_random = model.predict_with_pruning(
                        batch, 
                        pruning_ratio=pruning_ratio, 
                        preserve_ratio=preserve_ratio,
                        use_random=True
                    )
                    
                    for metric in metric_list_random:
                        metric.push(preds_random, labels_random)
                    acc_count_random += count_num_tensor_elements(preds_random)
                    
            else:
                # Use original method without pruning
                preds, labels = model.predict(batch)

            for metric in metric_list:
                metric.push(preds, labels)

            acc_count += count_num_tensor_elements(preds)

            if idx > 1000: break

    # Print results for importance-based pruning
    ret_metric = {}
    for metric in metric_list:
        ret_metric[metric.name] = metric.value / acc_count
    print(f'{rank} - Importance-based: {ret_metric}')

    # Print results for random pruning if enabled
    if compare_random and pruning_ratio > 0:
        ret_metric_random = {}
        for metric in metric_list_random:
            ret_metric_random[metric.name.replace('_random', '')] = metric.value / acc_count_random
        print(f'{rank} - Random pruning: {ret_metric_random}')

    # Gather results across processes
    all_metrics = metric_list + metric_list_random
    all_counts = [acc_count] + ([acc_count_random] if compare_random and pruning_ratio > 0 else [])
    
    metric_tensors = [metric.value for metric in all_metrics] + all_counts
    if is_dist:
        stat_tensor = torch.tensor(metric_tensors).to(model.device)
        gathered_results = [torch.zeros_like(stat_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_results, stat_tensor)
        all_stat = torch.stack(gathered_results, dim=0).sum(dim=0)
    else:
        all_stat = metric_tensors

    if rank == 0:
        # Log importance-based results
        item = {
            'model': args.model,
            'data': args.data,
            'context_length': args.context_length,
            'prediction_length': args.prediction_length,
            'pruning_ratio': pruning_ratio,
            'preserve_ratio': preserve_ratio,
            'method': 'importance-based' if pruning_ratio > 0 else 'no-pruning'
        }

        count = all_stat[len(metric_list)]
        for i, metric in enumerate(metric_list):
            val = all_stat[i] / count
            item[metric.name] = float(val.cpu().numpy())
        logging.info(f"Importance-based results: {item}")
        
        # Log random pruning results if enabled
        if compare_random and pruning_ratio > 0:
            item_random = item.copy()
            item_random['method'] = 'random'
            
            count_random = all_stat[len(metric_list) + len(metric_list_random)]
            for i, metric in enumerate(metric_list_random):
                val = all_stat[len(metric_list) + i] / count_random
                metric_name = metric.name.replace('_random', '')
                item_random[metric_name] = float(val.cpu().numpy())
            logging.info(f"Random pruning results: {item_random}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TimeMoE Evaluate with Pruning')
    parser.add_argument(
        '--model', '-m',
        type=str,
        # default='Maple728/TimeMoE-50M',
        default='/home/sa53869/time-series/time-moe/model_weights/time-moe-50m',
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
        '--pruning_ratio',
        type=float,
        default=0.0,
        help='Ratio of context to prune (0.0 means no pruning, 0.5 means remove 50% of context)'
    )
    parser.add_argument(
        '--preserve_ratio',
        type=float,
        default=0.25,
        help='Ratio of recent context to always preserve (default 0.25 means keep last 25%)'
    )
    parser.add_argument(
        '--compare_random',
        action='store_true',
        help='Also evaluate random pruning for comparison'
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