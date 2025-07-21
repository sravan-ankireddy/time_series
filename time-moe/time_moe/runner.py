import os
import math
import random
from functools import reduce
from operator import mul

import torch

from time_moe.datasets.time_moe_dataset import TimeMoEDataset
from time_moe.datasets.time_moe_window_dataset import TimeMoEWindowDataset
from time_moe.models.modeling_time_moe import TimeMoeForPrediction, TimeMoeConfig
from time_moe.trainer.hf_trainer import TimeMoETrainingArguments, TimeMoeTrainer
from time_moe.utils.dist_util import get_world_size
from time_moe.utils.log_util import logger, log_in_local_rank_0


import gc
import psutil

class TimeMoeRunner:
    def __init__(
            self,
            model_path: str = None,
            output_path: str = 'logs/time_moe',
            seed: int = 9899
    ):
        self.model_path = model_path
        self.output_path = output_path
        self.seed = seed

    def load_model(self, model_path: str = None, from_scatch: bool = False, **kwargs):
        if model_path is None:
            model_path = self.model_path
        attn = kwargs.pop('attn_implementation', None)
        if attn is None:
            attn = 'eager'
        elif attn == 'auto':
            # try to use flash-attention
            try:
                from flash_attn import flash_attn_func, flash_attn_varlen_func
                from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
                attn = 'flash_attention_2'
            except:
                log_in_local_rank_0('Flash attention import failed, switching to eager attention.', type='warn')
                attn = 'eager'

        if attn == 'eager':
            log_in_local_rank_0('Use Eager Attention')
        elif attn == 'flash_attention_2':
            log_in_local_rank_0('Use Flash Attention 2')
        else:
            raise ValueError(f'Unknown attention method: {attn}')
        kwargs['attn_implementation'] = attn

        if from_scatch:
            config = TimeMoeConfig.from_pretrained(model_path, _attn_implementation=attn)
            original_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.bfloat16)
            model = TimeMoeForPrediction(config)
            torch.set_default_dtype(original_dtype)
        else:
            model = TimeMoeForPrediction.from_pretrained(model_path, **kwargs)
        return model

    def load_training_state(self, checkpoint_path: str, model=None, trainer=None):
        """
        Load full training state from a checkpoint directory.
        
        Args:
            checkpoint_path: Path to the checkpoint directory (e.g., 'logs/time_moe/checkpoint-1000')
            model: Optional model instance to load weights into
            trainer: Optional trainer instance to restore training state
            
        Returns:
            Dictionary containing loaded state information
        """
        import json
        
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
        
        log_in_local_rank_0(f'Loading training state from: {checkpoint_path}')
        
        # Load trainer state (optimizer, scheduler, RNG states, etc.)
        trainer_state_path = os.path.join(checkpoint_path, 'trainer_state.json')
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path, 'r') as f:
                trainer_state = json.load(f)
            log_in_local_rank_0('Loaded trainer state')
        else:
            trainer_state = None
            log_in_local_rank_0('No trainer state found', type='warn')
        
        # Load model weights
        if model is not None:
            model_path = os.path.join(checkpoint_path, 'pytorch_model.bin')
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
                model.load_state_dict(state_dict)
                log_in_local_rank_0('Loaded model weights')
            else:
                # Try safetensors format
                model_path = os.path.join(checkpoint_path, 'model.safetensors')
                if os.path.exists(model_path):
                    from safetensors.torch import load_file
                    state_dict = load_file(model_path)
                    model.load_state_dict(state_dict)
                    log_in_local_rank_0('Loaded model weights from safetensors')
                else:
                    log_in_local_rank_0('No model weights found in checkpoint', type='warn')
        
        # Load optimizer state
        optimizer_state = None
        optimizer_path = os.path.join(checkpoint_path, 'optimizer.pt')
        if os.path.exists(optimizer_path):
            optimizer_state = torch.load(optimizer_path, map_location='cpu', weights_only=False)
            log_in_local_rank_0('Loaded optimizer state')
        
        # Load scheduler state
        scheduler_state = None
        scheduler_path = os.path.join(checkpoint_path, 'scheduler.pt')
        if os.path.exists(scheduler_path):
            scheduler_state = torch.load(scheduler_path, map_location='cpu', weights_only=False)
            log_in_local_rank_0('Loaded scheduler state')
        
        # Load RNG states - handle multiple file patterns
        rng_state = None
        rng_paths = [
            os.path.join(checkpoint_path, 'rng_state_0.pth'),
            os.path.join(checkpoint_path, 'rng_state_1.pth'),
        ]
        
        # Find any RNG state files
        for rng_path in rng_paths:
            if os.path.exists(rng_path):
                try:
                    rng_state = torch.load(rng_path, map_location='cpu', weights_only=False)
                    # Restore RNG states
                    if 'python' in rng_state:
                        random.setstate(rng_state['python'])
                    if 'numpy' in rng_state:
                        try:
                            import numpy as np
                            np.random.set_state(rng_state['numpy'])
                        except ImportError:
                            pass
                    if 'torch' in rng_state:
                        torch.set_rng_state(rng_state['torch'])
                    if 'torch_cuda' in rng_state and torch.cuda.is_available():
                        torch.cuda.set_rng_state_all(rng_state['torch_cuda'])
                    log_in_local_rank_0(f'Restored RNG states from {rng_path}')
                    break
                except Exception as e:
                    log_in_local_rank_0(f'Failed to load RNG state from {rng_path}: {e}', type='warn')
                    continue
        
        if rng_state is None:
            log_in_local_rank_0('No RNG state files found or all failed to load', type='warn')
        
        # If trainer is provided, restore its state
        if trainer is not None and trainer_state is not None:
            # Restore step counts and progress information
            global_step = trainer_state.get('global_step', 0)
            trainer.state.global_step = global_step
            trainer.state.epoch = trainer_state.get('epoch', 0.0)
            trainer.state.total_flos = trainer_state.get('total_flos', 0)
            trainer.state.log_history = trainer_state.get('log_history', [])
            trainer.state.best_metric = trainer_state.get('best_metric', None)
            trainer.state.best_model_checkpoint = trainer_state.get('best_model_checkpoint', None)
            trainer.state.max_steps = trainer_state.get('max_steps', -1)
            trainer.state.num_train_epochs = trainer_state.get('num_train_epochs', 0.0)
            
            # CRITICAL: Set max_steps from training args to ensure proper progress tracking
            if trainer.args.max_steps > 0:
                trainer.state.max_steps = trainer.args.max_steps
            
            # For progress bar: ensure the trainer knows where to start
            if global_step > 0:
                log_in_local_rank_0(f'Will resume training from step {global_step} to step {trainer.state.max_steps}')
                # Force update of internal counters that affect progress bars
                trainer.state.is_world_process_zero = trainer.is_world_process_zero()
            
            # Restore optimizer state
            if optimizer_state is not None and trainer.optimizer is not None:
                trainer.optimizer.load_state_dict(optimizer_state)
                log_in_local_rank_0('Restored optimizer state')
            
            # Restore scheduler state
            if scheduler_state is not None and trainer.lr_scheduler is not None:
                trainer.lr_scheduler.load_state_dict(scheduler_state)
                log_in_local_rank_0('Restored scheduler state')
            
            log_in_local_rank_0(f'Restored trainer state - Global step: {trainer.state.global_step}, Epoch: {trainer.state.epoch}')
        
        return {
            'trainer_state': trainer_state,
            'optimizer_state': optimizer_state,
            'scheduler_state': scheduler_state,
            'rng_state': rng_state,
            'checkpoint_path': checkpoint_path
        }

    def train_model(self, from_scratch: bool = False, **kwargs):
        setup_seed(self.seed)

        train_config = kwargs

        num_devices = get_world_size()

        global_batch_size = train_config.get('global_batch_size', None)
        micro_batch_size = train_config.get('micro_batch_size', None)

        if global_batch_size is None and micro_batch_size is None:
            raise ValueError('Must set at lease one argument: "global_batch_size" or "micro_batch_size"')
        elif global_batch_size is None:
            gradient_accumulation_steps = 1
            global_batch_size = micro_batch_size * num_devices
        elif micro_batch_size is None:
            micro_batch_size = math.ceil(global_batch_size / num_devices)
            gradient_accumulation_steps = 1
        else:
            if micro_batch_size * num_devices > global_batch_size:
                if num_devices > global_batch_size:
                    micro_batch_size = 1
                    global_batch_size = num_devices
                else:
                    micro_batch_size = math.ceil(global_batch_size / num_devices)
            gradient_accumulation_steps = math.ceil(global_batch_size / num_devices / micro_batch_size)
            global_batch_size = int(gradient_accumulation_steps * num_devices * micro_batch_size)

        if ('train_steps' in train_config
                and train_config['train_steps'] is not None
                and train_config['train_steps'] > 0):
            train_steps = int(train_config["train_steps"])
            num_train_epochs = -1
        else:
            train_steps = -1
            num_train_epochs = _safe_float(train_config.get("num_train_epochs", 1))

        precision = train_config.get('precision', 'bf16')
        if precision not in ['bf16', 'fp16', 'fp32']:
            log_in_local_rank_0(f'Precision {precision} is not set, use fp32 default!', type='warn')
            precision = 'fp32'

        if precision == 'bf16':
            torch_dtype = torch.bfloat16
        elif precision == 'fp16':
            # use fp32 to load model but uses fp15 to train model
            torch_dtype = torch.float32
        elif precision == 'fp32':
            torch_dtype = torch.float32
        else:
            raise ValueError(f'Unsupported precision {precision}')

        log_in_local_rank_0(f'Set global_batch_size to {global_batch_size}')
        log_in_local_rank_0(f'Set micro_batch_size to {micro_batch_size}')
        log_in_local_rank_0(f'Set gradient_accumulation_steps to {gradient_accumulation_steps}')
        log_in_local_rank_0(f'Set precision to {precision}')
        log_in_local_rank_0(f'Set normalization to {train_config["normalization_method"]}')

        training_args = TimeMoETrainingArguments(
            output_dir=self.output_path,
            num_train_epochs=num_train_epochs,
            # use_cpu=True,
            max_steps=train_steps,
            evaluation_strategy=train_config.get("evaluation_strategy", 'no'),
            eval_steps=_safe_float(train_config.get("eval_steps", None)),
            save_strategy=train_config.get("save_strategy", "no"),
            save_steps=_safe_float(train_config.get("save_steps", None)),
            learning_rate=float(train_config.get("learning_rate", 1e-5)),
            min_learning_rate=float(train_config.get("min_learning_rate", 0)),
            adam_beta1=float(train_config.get("adam_beta1", 0.9)),
            adam_beta2=float(train_config.get("adam_beta2", 0.95)),
            adam_epsilon=float(train_config.get("adam_epsilon", 1e-8)),
            lr_scheduler_type=train_config.get("lr_scheduler_type", 'constant'),
            warmup_ratio=float(train_config.get("warmup_ratio") or 0.0),
            warmup_steps=int(train_config.get("warmup_steps", 0)),
            weight_decay=float(train_config.get("weight_decay", 0.1)),
            per_device_train_batch_size=int(micro_batch_size),
            per_device_eval_batch_size=int(micro_batch_size * 2),
            gradient_accumulation_steps=int(gradient_accumulation_steps),
            gradient_checkpointing=train_config.get("gradient_checkpointing", False),
            bf16=True if precision == 'bf16' else False,
            fp16=True if precision == 'fp16' else False,
            deepspeed=train_config.get("deepspeed"),
            push_to_hub=False,
            logging_first_step=True,
            log_on_each_node=False,
            logging_steps=int(train_config.get('logging_steps', 1)),
            seed=self.seed,
            data_seed=self.seed,
            max_grad_norm=train_config.get('max_grad_norm', 1.0),
            optim=train_config.get('optim', 'adamw_torch'),
            torch_compile=train_config.get('torch_compile', False),
            dataloader_num_workers=train_config.get('dataloader_num_workers') or 2,
            ddp_find_unused_parameters=False,

            logging_dir=os.path.join(self.output_path, 'tb_logs'),
            save_only_model=train_config.get('save_only_model', True),
            save_total_limit=train_config.get('save_total_limit'),
        )

        model_path = train_config.pop('model_path', None) or self.model_path
        if model_path is not None:
            model = self.load_model(
                model_path=model_path,
                from_scatch=from_scratch,
                torch_dtype=torch_dtype,
                attn_implementation=train_config.get('attn_implementation', 'eager'),
            )
            log_in_local_rank_0(f'Load model parameters from: {model_path}')
        else:
            raise ValueError('Model path is None')

        num_total_params = 0
        for p in model.parameters():
            num_total_params += reduce(mul, p.shape)

        # print statistics info
        log_in_local_rank_0(train_config)
        log_in_local_rank_0(training_args)
        log_in_local_rank_0(model.config)
        log_in_local_rank_0(f'Number of the model parameters: {length_to_str(num_total_params)}')

        if train_steps > 0:
            total_train_tokens = train_steps * global_batch_size * train_config['max_length']
            log_in_local_rank_0(f'Tokens will consume: {length_to_str(total_train_tokens)}')

        # Training
        train_ds = self.get_train_dataset(
            train_config["data_path"],
            max_length=train_config["max_length"],
            stride=train_config["stride"],
            normalization_method=train_config["normalization_method"],
        )
        # breakpoint()
        # # train a 1000 samples from train_ds at .pt file
        # from torch.utils.data import Subset
        # sample_ds = Subset(train_ds, list(range(1000)))
        # torch.save(sample_ds, "../datasets/time-300b-sample.pt")
        # breakpoint()

        # load the sample dataset
        # train_ds = torch.load("../datasets/time-300b-sample.pt",weights_only=False)
        # print(f'Loaded {len(train_ds)} samples from the sample dataset')

        trainer = TimeMoeTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
        )
        
        # Resume from checkpoint if not training from scratch
        if not from_scratch and model_path is not None:
            # Check if model_path is a checkpoint directory (contains trainer_state.json)
            checkpoint_files = ['trainer_state.json', 'optimizer.pt', 'scheduler.pt']
            is_checkpoint = any(os.path.exists(os.path.join(model_path, f)) for f in checkpoint_files)
            
            if is_checkpoint:
                log_in_local_rank_0(f'Resuming training from checkpoint: {model_path}')
                
                # Temporarily monkey-patch torch.load to handle weights_only issue
                original_torch_load = torch.load
                def patched_torch_load(*args, **kwargs):
                    # For RNG state files and optimizer/scheduler files, disable weights_only
                    if len(args) > 0:
                        file_path = str(args[0])
                        if any(pattern in file_path for pattern in ['rng_state', 'optimizer.pt', 'scheduler.pt']):
                            kwargs['weights_only'] = False
                    return original_torch_load(*args, **kwargs)
                
                # Apply the patch
                torch.load = patched_torch_load
                
                try:
                    # Use the trainer's built-in resume mechanism
                    trainer.train(resume_from_checkpoint=model_path)
                finally:
                    # Always restore the original torch.load
                    torch.load = original_torch_load
            else:
                log_in_local_rank_0('Model path is not a checkpoint directory, starting fresh training')
                trainer.train()
        else:
            trainer.train()
            
        # Add cleanup after training
        del train_ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        trainer.save_model(self.output_path)
        log_in_local_rank_0(f'Saving model to {self.output_path}')

        return trainer.model

    def get_train_dataset(self, data_path, max_length, stride, normalization_method):
        log_in_local_rank_0('Loading dataset...')
        dataset = TimeMoEDataset(data_path, normalization_method=normalization_method)
        log_in_local_rank_0('Processing dataset to fixed-size sub-sequences...')
        window_dataset = TimeMoEWindowDataset(dataset, context_length=max_length, prediction_length=0, stride=stride, shuffle=False)
        return window_dataset

    # def get_train_dataset(self, data_path, max_length, stride, normalization_method):
    #     log_in_local_rank_0('Loading dataset...')
        
    #     # Monitor memory before dataset loading
    #     mem_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    #     log_in_local_rank_0(f'Memory before dataset loading: {mem_before:.2f} MB')
        
    #     dataset = TimeMoEDataset(data_path, normalization_method=normalization_method)
        
    #     log_in_local_rank_0('Processing dataset to fixed-size sub-sequences...')
    #     window_dataset = TimeMoEWindowDataset(dataset, context_length=max_length, prediction_length=0, stride=stride, shuffle=False)
        
    #     # Wrap dataset with memory cleanup
    #     class MemoryEfficientWrapper:
    #         def __init__(self, dataset):
    #             self.dataset = dataset
    #             self.access_count = 0
                
    #         def __len__(self):
    #             return len(self.dataset)
                
    #         def __getitem__(self, idx):
    #             item = self.dataset[idx]
    #             self.access_count += 1
                
    #             # Cleanup every 1000 data accesses
    #             if self.access_count % 1000 == 0:
    #                 gc.collect()
                    
    #             return item
        
    #     # Monitor memory after dataset creation
    #     mem_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    #     log_in_local_rank_0(f'Memory after dataset creation: {mem_after:.2f} MB (+{mem_after-mem_before:.2f})')
        
    #     return MemoryEfficientWrapper(window_dataset)

def setup_seed(seed: int = 9899):
    """
    Setup seed for all known operations.

    Args:
        seed (int): seed number.

    Returns:

    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def length_to_str(length):
    if length >= 1e12:
        return f'{length / 1e12:.3f}T'
    if length >= 1e9:
        return f'{length / 1e9:.3f}B'
    elif length >= 1e6:
        return f'{length / 1e6:.3f}M'
    else:
        return f'{length / 1e3:.3f}K'


def _safe_float(number):
    if number is None:
        return None
    else:
        return float(number)
