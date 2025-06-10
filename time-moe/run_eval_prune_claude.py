#!/usr/bin/env python
# -*- coding:utf-8 _*-
import json
import os
import argparse
import numpy as np
import logging
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader
from tqdm import tqdm
import copy

from transformers import AutoModelForCausalLM

from time_moe.datasets.benchmark_dataset import BenchmarkEvalDataset, GeneralEvalDataset

# Configure CUDA device
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def setup_nccl(rank, world_size, master_addr='127.0.0.1', master_port=9899):
    dist.init_process_group("nccl", init_method='tcp://{}:{}'.format(master_addr, master_port), rank=rank,
                            world_size=world_size)


def count_num_tensor_elements(tensor):
    n = 1
    for s in tensor.shape:
        n = n * s
    return n


class InferenceTimeCompressor:
    """Plug-and-play inference-time compression methods"""
    
    def __init__(self, compression_ratio=0.5, baseline_length=512):
        self.compression_ratio = compression_ratio
        self.baseline_length = baseline_length
        
    def baseline_recent(self, x):
        """Baseline method - take only the most recent baseline_length tokens"""
        if x.shape[-1] > self.baseline_length:
            return x[..., -self.baseline_length:]
        else:
            return x
        
    def selective_context_pruning(self, x):
        """Selective Context method - removes redundant information"""
        # Compute local variance to identify information density
        window_size = min(x.shape[-1] // 10, 16)
        if window_size > 1:
            local_variance = F.avg_pool1d(
                (x - x.mean(dim=-1, keepdim=True)).pow(2).unsqueeze(1),
                kernel_size=window_size, stride=1, 
                padding=window_size//2
            ).squeeze(1)
        else:
            local_variance = torch.var(x, dim=-1, keepdim=True).expand_as(x)
        
        # Always keep recent context (last 25%)
        recent_len = max(x.shape[-1] // 4, 1)
        historical_len = x.shape[-1] - recent_len
        target_historical = int(historical_len * (1 - self.compression_ratio))
        
        if historical_len > target_historical and target_historical > 0:
            # Select most informative historical points
            historical_variance = local_variance[..., :historical_len]
            _, top_indices = torch.topk(historical_variance, target_historical, dim=-1)
            top_indices = torch.sort(top_indices, dim=-1)[0]
            
            # Gather selected points
            batch_indices = torch.arange(x.shape[0]).unsqueeze(1).expand(-1, target_historical)
            selected_historical = x[batch_indices, top_indices]
            recent_context = x[..., -recent_len:]
            
            return torch.cat([selected_historical, recent_context], dim=-1)
        
        return x
    
    def attention_token_elimination(self, x):
        """Remove tokens with low attention importance"""
        # Simplified attention scoring without learnable parameters
        # Compute self-similarity as proxy for attention importance
        x_norm = F.normalize(x.unsqueeze(-1), dim=1)
        similarities = torch.matmul(x_norm, x_norm.transpose(-2, -1))
        importance_scores = similarities.mean(dim=-1)  # Average similarity
        
        target_len = int(x.shape[-1] * (1 - self.compression_ratio))
        target_len = max(target_len, 1)  # Ensure at least 1 token
        
        _, top_indices = torch.topk(importance_scores, target_len, dim=-1)
        top_indices = torch.sort(top_indices, dim=-1)[0]
        
        # Gather important tokens
        batch_indices = torch.arange(x.shape[0]).unsqueeze(1).expand(-1, target_len)
        return x[batch_indices, top_indices]
    
    def cycle_aware_compression(self, x):
        """Compress based on cyclical patterns in time series"""
        seq_len = x.shape[-1]
        
        # For simple uniform sampling when cycle detection is complex
        target_len = int(seq_len * (1 - self.compression_ratio))
        target_len = max(target_len, 1)
        
        # Simple uniform sampling with slight bias toward recent data
        recent_weight = 0.6  # 60% from recent half, 40% from older half
        recent_half = seq_len // 2
        recent_samples = int(target_len * recent_weight)
        older_samples = target_len - recent_samples
        
        compressed_sequences = []
        for i in range(x.shape[0]):
            # Sample from recent half
            if recent_samples > 0 and recent_half > 0:
                recent_step = max(recent_half // recent_samples, 1)
                recent_indices = torch.arange(recent_half, seq_len, recent_step)[:recent_samples]
            else:
                recent_indices = torch.tensor([])
                
            # Sample from older half  
            if older_samples > 0 and recent_half > 0:
                older_step = max(recent_half // older_samples, 1)
                older_indices = torch.arange(0, recent_half, older_step)[:older_samples]
            else:
                older_indices = torch.tensor([])
            
            # Combine and sort indices
            if len(recent_indices) > 0 and len(older_indices) > 0:
                all_indices = torch.cat([older_indices, recent_indices])
            elif len(recent_indices) > 0:
                all_indices = recent_indices
            elif len(older_indices) > 0:
                all_indices = older_indices
            else:
                # Fallback to uniform sampling
                all_indices = torch.linspace(0, seq_len-1, target_len).long()
                
            all_indices = torch.sort(all_indices)[0]
            compressed_sequences.append(x[i, all_indices])
        
        return torch.stack(compressed_sequences)


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
                 compression_method='mixed', compression_ratio=0.5, **kwargs):
        try:
            from time_moe.models.modeling_time_moe import TimeMoeForPrediction
            model = TimeMoeForPrediction.from_pretrained(
                model_path,
                device_map=device,
                # attn_implementation='flash_attention_2',
                torch_dtype='auto',
                local_files_only=True,
            )
        except:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                # attn_implementation='flash_attention_2',
                torch_dtype='auto',
                trust_remote_code=True,
                local_files_only=True,
            )

        logging.info(f'>>> Model dtype: {model.dtype}; Attention:{model.config._attn_implementation}')

        self.model = model
        self.device = device
        self.prediction_length = prediction_length
        self.model.eval()
        
        self.compression_method = compression_method
        self.compression_ratio = compression_ratio
        
        # Initialize inference-time compressor
        self.inference_compressor = InferenceTimeCompressor(compression_ratio=compression_ratio)
        
        logging.info(f'>>> Compression method: {compression_method}, ratio: {compression_ratio}')

    def predict(self, batch):
        model = self.model
        device = self.device
        prediction_length = self.prediction_length

        #### Plug-and-play inference-time compression ####
        original_inputs = batch['inputs'].to(device).to(model.dtype)
        
        # Apply compression method based on configuration
        if self.compression_method == 'baseline':
            compressed_inputs = self.inference_compressor.baseline_recent(original_inputs)
        elif self.compression_method == 'selective':
            compressed_inputs = self.inference_compressor.selective_context_pruning(original_inputs)
        elif self.compression_method == 'attention':
            compressed_inputs = self.inference_compressor.attention_token_elimination(original_inputs)
        elif self.compression_method == 'cycle':
            compressed_inputs = self.inference_compressor.cycle_aware_compression(original_inputs)
        elif self.compression_method == 'mixed':
            # Apply selective pruning first, then attention-based filtering
            temp_compressed = self.inference_compressor.selective_context_pruning(original_inputs)
            # Adjust compression ratio for second stage
            current_ratio = temp_compressed.shape[-1] / original_inputs.shape[-1]
            if current_ratio > (1 - self.compression_ratio):
                remaining_compression = 1 - ((1 - self.compression_ratio) / current_ratio)
                original_ratio = self.inference_compressor.compression_ratio
                self.inference_compressor.compression_ratio = remaining_compression
                compressed_inputs = self.inference_compressor.attention_token_elimination(temp_compressed)
                self.inference_compressor.compression_ratio = original_ratio
            else:
                compressed_inputs = temp_compressed
        else:
            # Fallback to baseline
            compressed_inputs = self.inference_compressor.baseline_recent(original_inputs)
        
        # Log compression statistics
        original_length = original_inputs.shape[-1]
        compressed_length = compressed_inputs.shape[-1]
        compression_achieved = 1 - (compressed_length / original_length)
        
        if not hasattr(self, '_compression_logged'):
            if self.compression_method == 'baseline':
                logging.info(f'>>> Baseline method: {original_length} -> {compressed_length} '
                           f'(recent {compressed_length} tokens)')
            else:
                logging.info(f'>>> Compression method {self.compression_method}: {original_length} -> {compressed_length} '
                           f'({compression_achieved:.1%} reduction)')
            self._compression_logged = True

        outputs = model.generate(
            inputs=compressed_inputs,
            max_new_tokens=prediction_length,
        )
        preds = outputs[:, -prediction_length:]
        labels = batch['labels'].to(device)
        if len(preds.shape) > len(labels.shape):
            labels = labels[..., None]
        return preds, labels


def evaluate_single_config(model_path, data_path, batch_size, context_length, prediction_length, 
                          compression_method, compression_ratio):
    """Single evaluation configuration wrapper"""
    
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
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            is_dist = False
    else:
        device = 'cpu'
        is_dist = False

    # evaluation metrics
    metric_list = [
        MSEMetric(name='mse'),
        MAEMetric(name='mae'),
    ]

    model = TimeMoE(
        model_path,
        device,
        context_length=context_length,
        prediction_length=prediction_length,
        compression_method=compression_method,
        compression_ratio=compression_ratio
    )
    
    if data_path.endswith('.csv'):
        dataset = BenchmarkEvalDataset(
            data_path,
            context_length=context_length,
            prediction_length=prediction_length,
        )
    else:
        dataset = GeneralEvalDataset(
            data_path,
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
        for idx, batch in enumerate(tqdm(test_dl, desc=f"Evaluating {compression_method}-{context_length}")):
            preds, labels = model.predict(batch)

            for metric in metric_list:
                metric.push(preds, labels)

            acc_count += count_num_tensor_elements(preds)

            if idx == 10:
                # Early exit for testing purposes
                break

    metric_tensors = [metric.value for metric in metric_list] + [acc_count]
    if is_dist:
        stat_tensor = torch.tensor(metric_tensors).to(model.device)
        gathered_results = [torch.zeros_like(stat_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_results, stat_tensor)
        all_stat = torch.stack(gathered_results, dim=0).sum(dim=0)
    else:
        all_stat = metric_tensors

    count = all_stat[-1]
    result_metrics = {}
    for i, metric in enumerate(metric_list):
        val = all_stat[i] / count
        result_metrics[metric.name] = float(val.cpu().numpy())
    
    return result_metrics


def dynamic_summary(results):
    """Print dynamic summary of all evaluations so far"""
    print("\n" + "="*70)
    print("         DYNAMIC SUMMARY OF EVALUATIONS SO FAR")
    print("="*70)
    for method, ctx_dict in results.items():
        print(f"\nCompression Method: {method.upper()}")
        print("-" * 40)
        for ctx_len, metrics in ctx_dict.items():
            if method == 'baseline':
                print(f"  Context {ctx_len} -> 512 (baseline: recent 512 tokens)")
            else:
                compression_ratio = 1 - (512 / ctx_len) if ctx_len > 512 else 0.0
                print(f"  Context {ctx_len} -> 512 (compression: {compression_ratio:.1%})")
            print(f"    MSE: {metrics.get('mse', 'N/A'):.6f}, MAE: {metrics.get('mae', 'N/A'):.6f}")
    print("="*70 + "\n")


def summarize_and_compare(results):
    """Final performance comparison and summary"""
    print("\n" + "="*80)
    print("                FINAL PERFORMANCE COMPARISON")
    print("="*80)
    
    best_per_method = {}
    all_results = []
    
    for method, ctx_dict in results.items():
        print(f"\nCompression Method: {method.upper()}")
        print("-" * 60)
        best_ctx = None
        best_mse = float('inf')
        
        for ctx_len, metrics in ctx_dict.items():
            mse = metrics.get('mse', float('inf'))
            mae = metrics.get('mae', float('inf'))
            
            if method == 'baseline':
                description = f"baseline: recent 512 tokens"
            else:
                compression_ratio = 1 - (512 / ctx_len) if ctx_len > 512 else 0.0
                description = f"compression: {compression_ratio:5.1%}"
            
            print(f"  Context {ctx_len:4d} -> 512 ({description:25s}) | "
                  f"MSE: {mse:.6f}, MAE: {mae:.6f}")
            
            all_results.append({
                'method': method,
                'context_length': ctx_len,
                'description': description,
                'mse': mse,
                'mae': mae
            })
            
            if mse < best_mse:
                best_mse = mse
                best_ctx = ctx_len
                
        if best_ctx is not None:
            best_per_method[method] = (best_ctx, best_mse)
    
    print("\n" + "="*80)
    print("               BEST CONFIGURATION PER METHOD (by MSE)")
    print("="*80)
    
    sorted_methods = sorted(best_per_method.items(), key=lambda x: x[1][1])
    for method, (ctx, mse) in sorted_methods:
        if method == 'baseline':
            description = "baseline: recent 512 tokens"
        else:
            compression_ratio = 1 - (512 / ctx) if ctx > 512 else 0.0
            description = f"compression: {compression_ratio:5.1%}"
            
        print(f"  {method:10s}: Context {ctx:4d} -> 512 ({description:25s}) | "
              f"Best MSE: {mse:.6f}")
    
    # Overall best
    if sorted_methods:
        best_method, (best_ctx, best_mse) = sorted_methods[0]
        if best_method == 'baseline':
            best_description = "baseline: recent 512 tokens"
        else:
            best_compression = 1 - (512 / best_ctx) if best_ctx > 512 else 0.0
            best_description = f"compression: {best_compression:.1%}"
            
        print(f"\n🏆 OVERALL BEST: {best_method.upper()} with context {best_ctx} -> 512 "
              f"({best_description}) | MSE: {best_mse:.6f}")
    
    print("="*80 + "\n")
    return best_per_method


def run_evaluation_for_all_methods_and_contexts(model_path, data_path, batch_size, prediction_length):
    """Run systematic evaluation across all compression methods and context lengths"""
    
    # Define the compression methods to test
    compression_methods = ['baseline', 'selective', 'attention', 'cycle', 'mixed']
    
    # Define the context lengths to test
    context_lengths = [512, 768, 1024, 2048]
    
    # Target compressed length (max context that can be processed)
    compressed_length_target = 512
    
    results = {}
    
    # Calculate total evaluations: baseline once + compression methods for all context lengths
    baseline_evaluations = 1
    compression_evaluations = (len(compression_methods) - 1) * len(context_lengths)
    total_evaluations = baseline_evaluations + compression_evaluations
    current_evaluation = 0
    
    print(f"\n🚀 Starting systematic evaluation:")
    print(f"   - Baseline method: 1 evaluation (recent {compressed_length_target} tokens)")
    print(f"   - Compression methods: {len(compression_methods)-1} methods × {len(context_lengths)} context lengths")
    print(f"   Total evaluations: {total_evaluations}\n")
    
    for method in compression_methods:
        results[method] = {}
        print(f"\n📊 Testing compression method: {method.upper()}")
        print("-" * 50)
        
        if method == 'baseline':
            # Run baseline only once with context length 512
            current_evaluation += 1
            ctx_len = 512
            compression_ratio = 0.0  # Not used for baseline
            description = f"baseline: recent {compressed_length_target} tokens"
            
            print(f"\n[{current_evaluation}/{total_evaluations}] Running: {method} | "
                  f"Context {ctx_len} -> {compressed_length_target} | "
                  f"{description}")
            
            # Run evaluation
            metrics = evaluate_single_config(
                model_path=model_path,
                data_path=data_path,
                batch_size=batch_size,
                context_length=ctx_len,
                prediction_length=prediction_length,
                compression_method=method,
                compression_ratio=compression_ratio
            )
            
            results[method][ctx_len] = metrics
            
            print(f"✅ Completed: MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}")
            
        else:
            # Run compression methods across all context lengths
            for ctx_len in context_lengths:
                current_evaluation += 1
                
                # Calculate compression ratio to achieve target length
                compression_ratio = 1 - (compressed_length_target / ctx_len) if ctx_len > compressed_length_target else 0.0
                description = f"compression: {compression_ratio:.1%}"
                
                print(f"\n[{current_evaluation}/{total_evaluations}] Running: {method} | "
                      f"Context {ctx_len} -> {compressed_length_target} | "
                      f"{description}")
                
                # Run evaluation
                metrics = evaluate_single_config(
                    model_path=model_path,
                    data_path=data_path,
                    batch_size=batch_size,
                    context_length=ctx_len,
                    prediction_length=prediction_length,
                    compression_method=method,
                    compression_ratio=compression_ratio
                )
                
                results[method][ctx_len] = metrics
                
                print(f"✅ Completed: MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}")
                
                # Show dynamic summary after each method completion
                if current_evaluation % len(context_lengths) == 0 or current_evaluation == total_evaluations:
                    dynamic_summary(results)
    
    # Final comparison and summary
    best_configs = summarize_and_compare(results)
    
    return results, best_configs


def dynamic_summary(results):
    """Print dynamic summary of all evaluations so far"""
    print("\n" + "="*70)
    print("         DYNAMIC SUMMARY OF EVALUATIONS SO FAR")
    print("="*70)
    for method, ctx_dict in results.items():
        print(f"\nCompression Method: {method.upper()}")
        print("-" * 40)
        if method == 'baseline':
            # Baseline only has one entry
            for ctx_len, metrics in ctx_dict.items():
                print(f"  Context {ctx_len} -> 512 (baseline: recent 512 tokens)")
                print(f"    MSE: {metrics.get('mse', 'N/A'):.6f}, MAE: {metrics.get('mae', 'N/A'):.6f}")
        else:
            # Compression methods have multiple context lengths
            for ctx_len, metrics in ctx_dict.items():
                compression_ratio = 1 - (512 / ctx_len) if ctx_len > 512 else 0.0
                print(f"  Context {ctx_len} -> 512 (compression: {compression_ratio:.1%})")
                print(f"    MSE: {metrics.get('mse', 'N/A'):.6f}, MAE: {metrics.get('mae', 'N/A'):.6f}")
    print("="*70 + "\n")


def summarize_and_compare(results):
    """Final performance comparison and summary"""
    print("\n" + "="*80)
    print("                FINAL PERFORMANCE COMPARISON")
    print("="*80)
    
    best_per_method = {}
    all_results = []
    
    for method, ctx_dict in results.items():
        print(f"\nCompression Method: {method.upper()}")
        print("-" * 60)
        best_ctx = None
        best_mse = float('inf')
        
        for ctx_len, metrics in ctx_dict.items():
            mse = metrics.get('mse', float('inf'))
            mae = metrics.get('mae', float('inf'))
            
            if method == 'baseline':
                description = f"baseline: recent 512 tokens"
            else:
                compression_ratio = 1 - (512 / ctx_len) if ctx_len > 512 else 0.0
                description = f"compression: {compression_ratio:5.1%}"
            
            print(f"  Context {ctx_len:4d} -> 512 ({description:25s}) | "
                  f"MSE: {mse:.6f}, MAE: {mae:.6f}")
            
            all_results.append({
                'method': method,
                'context_length': ctx_len,
                'description': description,
                'mse': mse,
                'mae': mae
            })
            
            if mse < best_mse:
                best_mse = mse
                best_ctx = ctx_len
                
        if best_ctx is not None:
            best_per_method[method] = (best_ctx, best_mse)
    
    print("\n" + "="*80)
    print("               BEST CONFIGURATION PER METHOD (by MSE)")
    print("="*80)
    
    # Sort methods, but put baseline first for reference
    baseline_result = None
    compression_results = []
    
    for method, (ctx, mse) in best_per_method.items():
        if method == 'baseline':
            baseline_result = (method, ctx, mse, "baseline: recent 512 tokens")
        else:
            compression_ratio = 1 - (512 / ctx) if ctx > 512 else 0.0
            description = f"compression: {compression_ratio:5.1%}"
            compression_results.append((method, ctx, mse, description))
    
    # Sort compression methods by performance
    compression_results.sort(key=lambda x: x[2])
    
    # Display baseline first, then sorted compression methods
    if baseline_result:
        method, ctx, mse, description = baseline_result
        print(f"  {method:10s}: Context {ctx:4d} -> 512 ({description:25s}) | "
              f"MSE: {mse:.6f} [BASELINE]")
    
    for method, ctx, mse, description in compression_results:
        print(f"  {method:10s}: Context {ctx:4d} -> 512 ({description:25s}) | "
              f"MSE: {mse:.6f}")
    
    # Overall best compression method (excluding baseline)
    if compression_results:
        best_method, best_ctx, best_mse, best_description = compression_results[0]
        baseline_mse = baseline_result[2] if baseline_result else float('inf')
        
        print(f"\n🏆 BEST COMPRESSION METHOD: {best_method.upper()} with context {best_ctx} -> 512 "
              f"({best_description}) | MSE: {best_mse:.6f}")
        
        if baseline_result:
            improvement = ((baseline_mse - best_mse) / baseline_mse) * 100
            if improvement > 0:
                print(f"📈 IMPROVEMENT OVER BASELINE: {improvement:.2f}% better MSE")
            else:
                print(f"📉 BASELINE PERFORMANCE: {abs(improvement):.2f}% better than best compression")
    
    print("="*80 + "\n")
    return best_per_method



def evaluate(args):
    """Main evaluation function - now runs systematic comparison"""
    
    if args.systematic_eval:
        print("🔬 Running systematic evaluation across all compression methods and context lengths...")
        results, best_configs = run_evaluation_for_all_methods_and_contexts(
            model_path=args.model,
            data_path=args.data,
            batch_size=args.batch_size,
            prediction_length=args.prediction_length
        )
        
        # Save results to file
        if args.output_file:
            output_data = {
                'results': results,
                'best_configs': best_configs,
                'config': {
                    'model': args.model,
                    'data': args.data,
                    'batch_size': args.batch_size,
                    'prediction_length': args.prediction_length,
                    'max_context_length': 512
                }
            }
            with open(args.output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"📝 Results saved to: {args.output_file}")
        
    else:
        # Single evaluation (original behavior)
        print("🎯 Running single evaluation configuration...")
        metrics = evaluate_single_config(
            model_path=args.model,
            data_path=args.data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            compression_method=args.compression_method,
            compression_ratio=args.compression_ratio
        )
        
        print(f"\n📊 Results: MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TimeMoE Evaluate')
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='/home/sa53869/time_series/time-moe/model_weights/time-moe-200m',
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
        help='Context length (for single evaluation)'
    )
    parser.add_argument(
        '--prediction_length', '-p',
        type=int,
        default=96,
        help='Prediction length'
    )
    parser.add_argument(
        '--compression_method',
        type=str,
        default='mixed',
        choices=['baseline', 'selective', 'attention', 'cycle', 'mixed'],
        help='Compression method to use (for single evaluation)'
    )
    parser.add_argument(
        '--compression_ratio',
        type=float,
        default=0.5,
        help='Compression ratio (for single evaluation, not used for baseline)'
    )
    parser.add_argument(
        '--systematic_eval',
        action='store_true',
        help='Run systematic evaluation across all methods and context lengths'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='compression_evaluation_results.json',
        help='Output file for systematic evaluation results'
    )
    
    args = parser.parse_args()
    
    # Set default context length for single evaluation (max 512)
    if not args.systematic_eval and args.context_length is None:
        args.context_length = 512
    
    evaluate(args)
