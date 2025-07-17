import math
from typing import Optional, Tuple, List, Union, Any
import warnings

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel, Cache, DynamicCache, StaticCache
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import MoeModelOutputWithPast, MoeCausalLMOutputWithPast
from transformers.utils import logging, is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10

from .configuration_time_moe import TimeMoeConfig
from .ts_generation_mixin import TSGenerationMixin

logger = logging.get_logger(__name__)

# if is_flash_attn_2_available():
#     from flash_attn import flash_attn_func, flash_attn_varlen_func
#     from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
except:
    pass

import sys
import os
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'chronos'))

from chronos import BaseChronosPipeline


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def load_balancing_loss_func(
        gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
        top_k: int,
        num_experts: int = None,
        attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor], List[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        top_k (`int`)
            Selected Top k over the experts.
        attention_mask (`torch.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, (tuple, list)) or gate_logits[0] is None:
        return 0.0

    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each expert
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, 2, num_experts))
            .reshape(-1, 2, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(dim=0))

    return overall_loss * num_experts


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int, dim: int = 1) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=dim, repeats=n_rep). 
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    when dim=1, or can repeat along other dimensions when specified.
    """
    if n_rep == 1:
        return hidden_states
    
    if dim == 1:
        # Original behavior for transformer attention
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    elif dim == 2:
        # For cross-attention where we might want to repeat along the head dimension
        return torch.repeat_interleave(hidden_states, n_rep, dim=dim)
    else:
        return torch.repeat_interleave(hidden_states, n_rep, dim=dim)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def compute_entropies(
    input_ids: torch.FloatTensor,
    chronos_pipeline,
    config
) -> Tuple[List[List[float]], List[List[float]], List[List[float]], List[List[np.ndarray]]]:
    """
    Compute entropies and relative entropies for time series sequences using autoregressive prediction.
    
    This function performs causal cropping to create sequences suitable for autoregressive
    prediction, processes them in batches for memory efficiency, and computes both absolute
    entropies and relative entropies from the prediction logits.
    
    Args:
        input_ids: Input time series tensor of shape [batch_size, seq_length, input_size]
        chronos_pipeline: Chronos pipeline for making predictions
        config: Model configuration containing entropy_batch_size parameter
    
    Returns:
        Tuple of (batch_predictions, batch_entropies, batch_relative_entropies, batch_logits):
        - batch_predictions: List of predictions for each sequence in the batch
        - batch_entropies: List of absolute entropy values for each sequence in the batch
        - batch_relative_entropies: List of relative entropy values for each sequence in the batch
        - batch_logits: List of logits arrays for each sequence in the batch
    """
    # Create causally cropped sequences for parallel processing
    # Pattern: [[mean], [x0], [x0, x1], [x0, x1, x2], ..., [mean], [y0], [y0, y1], ...]
    causal_context_list = []
    sequence_metadata = []  # Track which sequence and timestep each context belongs to
    
    batch_size, seq_length, _ = input_ids.shape
    
    for batch_idx in range(batch_size):
        sequence = input_ids[batch_idx].squeeze(-1).to("cpu")  # [seq_length]
        sequence_mean = torch.mean(sequence).item()
        
        # Create causally cropped versions starting with mean for first prediction
        for timestep in range(seq_length):  # 0 to seq_length-1
            if timestep == 0:
                # Use sequence mean as context to predict x0
                causal_crop = torch.tensor([sequence_mean], dtype=torch.float32)
            else:
                # Use x0, x1, ..., x_{timestep-1} as context to predict x_timestep
                causal_crop = sequence[:timestep]  # Progressive causal cropping
            
            causal_context_list.append(causal_crop)
            sequence_metadata.append((batch_idx, timestep))  # (sequence_idx, target_timestep)
    
    # Process in batches to avoid memory issues
    all_preds = []
    all_logits = []

    for i in range(0, len(causal_context_list), config.entropy_batch_size):
        batch_end = min(i + config.entropy_batch_size, len(causal_context_list))
        batch_contexts = causal_context_list[i:batch_end]
        
        # Process this batch
        batch_preds, batch_logits = chronos_pipeline.predict(
            context=batch_contexts,
            prediction_length=1,
            num_samples=1,
            return_logits=True
        )
        
        all_preds.append(batch_preds)
        all_logits.append(batch_logits)
    
    # Concatenate all batches
    preds = torch.cat(all_preds, dim=0)
    logits = torch.cat(all_logits, dim=0)
    
    # Reorganize results back to batch format
    batch_predictions = [[] for _ in range(batch_size)]
    batch_entropies = [[] for _ in range(batch_size)]
    batch_relative_entropies = [[] for _ in range(batch_size)]
    batch_logits = [[] for _ in range(batch_size)]
    
    for ctx_idx, (batch_idx, timestep) in enumerate(sequence_metadata):
        # Extract logits and compute entropy for this prediction
        token_logits = logits[ctx_idx, 0, 0, :]  # Shape: [vocab_size]
        probs = torch.softmax(token_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log2(probs + 1e-10)).item()
        predicted_token = preds[ctx_idx, 0, 0].item()
        
        # Store results organized by original batch sequence
        batch_predictions[batch_idx].append(predicted_token)
        batch_entropies[batch_idx].append(entropy)
        batch_logits[batch_idx].append(token_logits.cpu().numpy())

        # Compute relative entropies for each sequence
        if timestep == 0:
            # First timestep has no previous entropy to compare with
            relative_entropy = 0.0
        else:
            # Relative entropy = current_entropy - previous_entropy
            relative_entropy = entropy - batch_entropies[batch_idx][timestep - 1]
        batch_relative_entropies[batch_idx].append(relative_entropy)

    return batch_predictions, batch_entropies, batch_relative_entropies, batch_logits


def compute_threshold_patches(
    input_sequences: torch.Tensor,
    threshold_factor: float = 1.0,
    min_patch_size: int = 1,
    max_patch_size: Optional[int] = None,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create patches by starting a new patch whenever the variance of the accumulated
    points exceeds threshold_factor * var(full batch), with optional max size.
    Uses prefix sums to compute variances in O(1) per check.

    Args:
        input_sequences: Tensor of shape [B, S]
        threshold_factor: multiple of the global variance at which to cut a new patch
        min_patch_size: smallest allowed patch length
        max_patch_size: largest allowed patch length (None to disable)
        seed: optional RNG seed

    Returns:
        mask: Boolean tensor [B, S] marking each patch start
        avg_patch_sizes: Float tensor [B] with the mean patch length per sequence
    """
    if seed is not None:
        torch.manual_seed(seed)

    if input_sequences.ndim != 2:
        input_sequences = torch.squeeze(input_sequences)

    # Ensure shape [B, S]
    if input_sequences.ndim != 2:
        input_sequences = input_sequences.squeeze(0)

    B, S = input_sequences.shape
    device = input_sequences.device

    # 1) compute global variance
    global_var = input_sequences.var(unbiased=False)
    threshold = threshold_factor * global_var

    # 2) build prefix sums for O(1) variance queries
    x = input_sequences
    cum_x = torch.cat([torch.zeros(B, 1, device=device), x.cumsum(dim=1)], dim=1)        # [B, S+1]
    cum_x2 = torch.cat([torch.zeros(B, 1, device=device), (x * x).cumsum(dim=1)], dim=1) # [B, S+1]

    mask = torch.zeros(B, S, dtype=torch.bool, device=device)
    avg_sizes = torch.zeros(B, device=device)

    # 3) iterate per batch (inner loop is now O(1) variance checks)
    for b in range(B):
        idx = 0
        sizes = []
        seq_cum_x = cum_x[b]
        seq_cum_x2 = cum_x2[b]

        while idx < S:
            mask[b, idx] = True
            start = idx
            end = min(start + min_patch_size, S)

            # grow until variance exceeds threshold or hit max size
            while end < S:
                length = end - start
                sum_x  = seq_cum_x[end]  - seq_cum_x[start]
                sum_x2 = seq_cum_x2[end] - seq_cum_x2[start]
                var    = sum_x2 / length - (sum_x / length) ** 2

                if var > threshold:
                    break
                if max_patch_size is not None and length >= max_patch_size:
                    break
                end += 1

            size = end - start
            sizes.append(size)
            idx += size

        avg_sizes[b] = float(sum(sizes)) / len(sizes)

    return mask, avg_sizes



def compute_random_patches(
    batch_size: int,
    seq_length: int,
    min_patch_size: int = 4,
    max_patch_size: int = 8,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    This version generates all patch sizes at once and uses cumsum to find positions.
    """
    if seed is not None:
        torch.manual_seed(seed)

    batch_patch_starts = torch.zeros(batch_size, seq_length, dtype=torch.bool)
    
    # Fixed patch size case
    if min_patch_size == max_patch_size:
        patch_size = min_patch_size
        patch_indices = torch.arange(0, seq_length, patch_size)
        batch_patch_starts[:, patch_indices] = 1
        return batch_patch_starts
    
    # Variable patch size case - generate more patches than needed, then truncate
    for batch_idx in range(batch_size):
        # Estimate maximum number of patches needed
        max_patches = seq_length // min_patch_size + 1
        
        # Generate random patch sizes
        patch_sizes = torch.randint(min_patch_size, max_patch_size + 1, (max_patches,))
        
        # Use cumulative sum to get patch start positions
        patch_positions = torch.cumsum(patch_sizes, dim=0) - patch_sizes[0]
        
        # Filter positions that are within sequence length
        valid_positions = patch_positions[patch_positions < seq_length]
        
        # Set patch starts
        if len(valid_positions) > 0:
            batch_patch_starts[batch_idx, valid_positions] = 1
    
    return batch_patch_starts

def compute_patches(
    entropies: List[List[float]],
    use_relative_entropies: bool = True,
    threshold: Optional[float] = None,
    min_patch_size: int = 4,
    max_patch_size: int = 8
) -> Tuple[List[torch.Tensor], List[float]]:
    """
    Compute variable-length patches based on entropy thresholds.
    
    Creates patches by monitoring entropy values and starting new patches when 
    entropy exceeds a threshold, indicating potential pattern changes or 
    increased uncertainty in the time series.
    
    Args:
        entropies: List of entropy sequences, either absolute or relative entropies
                  Shape: [batch_size][seq_length]
        use_relative_entropies: If True, uses relative entropies; if False, uses absolute entropies
        threshold: Entropy threshold for patch boundary detection. If None, computed automatically
        min_patch_size: Minimum allowed patch size
        max_patch_size: Maximum allowed patch size
    
    Returns:
        Tuple of (batch_patch_starts, thresholds_used):
        - batch_patch_starts: List of binary vectors for each sequence 
                             Each vector is shape [seq_length], 1 at patch start, 0 elsewhere
        - thresholds_used: List of thresholds used for each sequence
    """
    batch_patch_starts = []
    thresholds_used = []
    
    for sequence_entropies in entropies:
        if len(sequence_entropies) == 0:
            # Empty sequence - create single patch
            batch_patch_starts.append(torch.tensor([1], dtype=torch.long))
            thresholds_used.append(0.0)
            continue
        
        seq_length = len(sequence_entropies)
        patch_starts = torch.zeros(seq_length, dtype=torch.long)
        
        # Convert to numpy for easier computation
        entropy_array = np.array(sequence_entropies)
        
        # Determine threshold for this sequence
        if threshold is None:
            if use_relative_entropies:
                # For relative entropies: Use robust statistics (percentiles)
                # Skip the first element which is always 0.0 for relative entropies
                relevant_entropies = entropy_array[1:] if len(entropy_array) > 1 else entropy_array
                if len(relevant_entropies) > 0:
                    q75 = np.percentile(relevant_entropies, 75)
                    q25 = np.percentile(relevant_entropies, 25)
                    iqr = q75 - q25
                    # Use 75th percentile + 0.5 * IQR as threshold
                    sequence_threshold = q75 + 0.5 * iqr
                    # Ensure minimum threshold to avoid too many patches
                    sequence_threshold = max(sequence_threshold, 0.2)
                else:
                    sequence_threshold = 0.5  # Default fallback
            else:
                # For absolute entropies: Use mean + standard deviation
                mean_entropy = np.mean(entropy_array)
                std_entropy = np.std(entropy_array)
                sequence_threshold = mean_entropy + 1.0 * std_entropy
                sequence_threshold = max(sequence_threshold, 1.0)
        else:
            sequence_threshold = threshold
        
        thresholds_used.append(sequence_threshold)
        
        # Always start with first patch
        patch_starts[0] = 1
        current_patch_start = 0
        
        i = 1
        while i < seq_length:
            current_patch_size = i - current_patch_start
            
            # Check if we should start a new patch due to high entropy
            if (sequence_entropies[i] > sequence_threshold and 
                current_patch_size >= min_patch_size):
                patch_starts[i] = 1
                current_patch_start = i
            
            # Force patch boundary if current patch reaches max size
            elif current_patch_size >= max_patch_size:
                patch_starts[i] = 1
                current_patch_start = i
            
            i += 1
        
        batch_patch_starts.append(patch_starts)
    
    return batch_patch_starts, thresholds_used



def compute_initial_patch_embeddings_with_pooling(
    input_embeddings: torch.Tensor,
    batch_patch_starts: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    hidden_size: Optional[int] = None,
    config=None,
    pooling_type: str = "max",
) -> torch.Tensor:
    """
    Compute patch embeddings using pooling over variable-length or fixed-size patches.
    Fully vectorized and handles both Tensor and List[Tensor] inputs for batch_patch_starts.
    """
    B, S, D = input_embeddings.shape
    hidden_size = hidden_size or D
    device = input_embeddings.device
    dtype = input_embeddings.dtype

    if batch_patch_starts is not None:
        # --- build a [B, S] boolean mask of patch-starts ---
        if isinstance(batch_patch_starts, torch.Tensor):
            batch_mask = batch_patch_starts.to(device).bool()          # [B, S]
        else:
            batch_mask = torch.stack(batch_patch_starts, dim=0).to(device).bool()

        # how many patches per sequence?
        num_patches_per_seq = batch_mask.sum(dim=1)                    # [B]
        P = int(num_patches_per_seq.max().item())                      # max patches

        # assign each position a patch-id: cumsum(start_flags)-1
        patch_ids = torch.cumsum(batch_mask.long(), dim=1) - 1        # [B, S]
        # now values ∈ [0 .. num_patches_i-1]

        # build one-hot over patches: [B, S, P]
        patch_ids_exp = patch_ids.unsqueeze(-1)                       # [B, S, 1]
        range_p = torch.arange(P, device=device).view(1, 1, P)        # [1, 1, P]
        one_hot = (patch_ids_exp == range_p)                          # [B, S, P]

        # expand embeddings to [B, S, P, D]
        emb_exp = input_embeddings.unsqueeze(2).expand(-1, -1, P, -1)

        if pooling_type == "max":
            min_val = torch.finfo(dtype).min
            masked = emb_exp.masked_fill(~one_hot.unsqueeze(-1), min_val)
            # max over S → [B, P, D]
            patch_embeddings = masked.max(dim=1)[0]
            # zero out empty patches (where no start was present)
            empty = one_hot.sum(dim=1) == 0                             # [B, P]
            patch_embeddings = patch_embeddings.masked_fill(
                empty.unsqueeze(-1), 0.0
            )
        else:  # mean pooling
            masked = emb_exp * one_hot.unsqueeze(-1).float()            # [B, S, P, D]
            summed = masked.sum(dim=1)                                  # [B, P, D]
            counts = one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)      # [B, P, 1]
            patch_embeddings = summed / counts                          # [B, P, D]

        return patch_embeddings

    else:
        # Fixed-size patches (unchanged)
        if config is None:
            raise ValueError("config must be provided when batch_patch_starts is None")
        patch_size = getattr(config, "patch_size", 1)
        if patch_size <= 1:
            return input_embeddings

        P = S // patch_size
        if P == 0:
            return torch.zeros(B, 1, hidden_size, device=device, dtype=dtype)

        seq_trunc = P * patch_size
        x = input_embeddings[:, :seq_trunc, :].view(B, P, patch_size, D)
        if pooling_type == "max":
            return x.max(dim=2)[0]
        else:
            return x.mean(dim=2)




class TimeMoeInputEmbedding(nn.Module):
    """
    Use a mlp layer to embedding the time-series.
    """

    def __init__(self, config: TimeMoeConfig):
        super().__init__()
        self.config = config
        self.input_size = config.input_size  # default 1
        self.hidden_size = config.hidden_size
        self.emb_layer = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.gate_layer = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        emb = self.act_fn(self.gate_layer(x)) * self.emb_layer(x)
        return emb

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.dim = dim
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w3 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B S D
        x1 = self.w1(x.view_as(x))
        x3 = self.w3(x.view_as(x))
        output = self.w2(F.silu(x1) * x3)
        return output

    def reset_parameters(self, init_std=None, factor=1.0):
        in_init_std = init_std or (self.dim ** (-0.5)) / factor
        out_init_std = init_std or (self.hidden_dim ** (-0.5)) / factor

        nn.init.trunc_normal_(
            self.w1.weight,
            mean=0.0,
            std=in_init_std,
            a=-3 * in_init_std,
            b=3 * in_init_std,
        )
        nn.init.trunc_normal_(
            self.w2.weight,
            mean=0.0,
            std=out_init_std,
            a=-3 * out_init_std,
            b=3 * out_init_std,
        )
        nn.init.trunc_normal_(
            self.w3.weight,
            mean=0.0,
            std=in_init_std,
            a=-3 * in_init_std,
            b=3 * in_init_std,
        )

class TimeMoeLocalSelfAttention(nn.Module):
    """
    Local self-attention module for point embeddings with sliding window.
    """
    
    def __init__(self, config: TimeMoeConfig, window_size: int = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_patch_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.attention_dropout = config.attention_dropout
        self.window_size = config.local_attention_window_size

        # Self-attention projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=False)

        # Layer norms
        self.attention_norm = TimeMoeRMSNorm(self.hidden_size, eps=getattr(config, 'rms_norm_eps', 1e-6))
        self.ffn_norm = TimeMoeRMSNorm(self.hidden_size, eps=getattr(config, 'rms_norm_eps', 1e-6))

        # Feedforward layer (use FeedForward instead of TimeMoeMLP)
        self.feed_forward = FeedForward(
            dim=self.hidden_size,
            hidden_dim=4 * self.hidden_size,
        )
    
    def create_local_causal_attention_mask(self, seq_len, window_size, device):
        """
        Create causal local attention mask for sliding window attention.
        Each position can only attend to itself and previous positions within the window.
        """
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        
        for i in range(seq_len):
            # Causal: only attend to current and previous positions
            start = max(0, i - window_size + 1)  # Include current position
            end = i + 1  # Only up to current position (causal)
            mask[i, start:end] = 0.0
            
        return mask
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
        Returns:
            [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = x.shape

        # Attention block
        attn_input = self.attention_norm(x)
        query_states = self.q_proj(attn_input)
        key_states = self.k_proj(attn_input)
        value_states = self.v_proj(attn_input)

        query_states = query_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        local_mask = self.create_local_causal_attention_mask(seq_len, self.window_size, x.device)
        attn_weights = attn_weights + local_mask.unsqueeze(0).unsqueeze(0)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=value_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        h = x + attn_output

        # Feedforward block
        h_norm = self.ffn_norm(h)
        ff_out = self.feed_forward(h_norm)
        out = h + ff_out
        return out


class TimeMoeCrossAttention(nn.Module):
    def __init__(self, config: TimeMoeConfig):
        super().__init__()
        self.dim = config.hidden_size
        self.head_dim = self.dim // config.num_patch_attention_heads
        self.n_heads = config.num_patch_attention_heads
        self.n_kv_heads = config.num_patch_attention_heads
        self.heads_per_group = self.n_heads // self.n_kv_heads
        self.attention_dropout = getattr(config, 'attention_dropout', 0.0)
        
        self.cross_attn_norm_q = TimeMoeRMSNorm(self.dim, eps=getattr(config, 'rms_norm_eps', 1e-6))
        self.cross_attn_norm_kv = TimeMoeRMSNorm(self.dim, eps=getattr(config, 'rms_norm_eps', 1e-6))
        
        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        kv: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len_q, _ = x.shape
        _, seq_len_kv, _ = kv.shape
        
        x_norm = self.cross_attn_norm_q(x)
        kv_norm = self.cross_attn_norm_kv(kv)
        
        xq = self.wq(x_norm)
        xk = self.wk(kv_norm)
        xv = self.wv(kv_norm)
        
        xq = xq.view(bsz, seq_len_q, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len_kv, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len_kv, self.n_kv_heads, self.head_dim)
        
        xk = repeat_kv(xk, self.heads_per_group)
        xv = repeat_kv(xv, self.heads_per_group)
        
        xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
        
        attn_weights = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        query_padding_mask = None
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            
            # Find rows that are completely masked (padded queries)
            fully_masked = (mask == float('-inf')).all(dim=-1)
            query_padding_mask = fully_masked.any(dim=1)  # [batch_size, seq_len_q]
            
            # For fully masked rows, unmask the first position to prevent NaN
            safe_mask = mask.clone()
            safe_mask[fully_masked.unsqueeze(-1).expand_as(mask)] = 0.0
            safe_mask = safe_mask.scatter(-1, torch.zeros_like(fully_masked.unsqueeze(-1), dtype=torch.long), 0.0)
            
            attn_weights = attn_weights + safe_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=xv.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        output = torch.matmul(attn_weights, xv)
        output = output.transpose(1, 2).contiguous()
        
        output = self.wo(output.view(bsz, seq_len_q, -1))
        
        output = x + output

        # Post-processing cleanup: zero out padded query positions
        if query_padding_mask is not None:
            output = output.masked_fill(query_padding_mask.unsqueeze(-1), 0.0)

        return output
    
    def create_cross_attention_mask(self, num_patches, patch_size, device, reverse=False):
        """
        Create mask for cross-attention between patches and points.
        
        Args:
            num_patches: Number of patches
            patch_size: Size of each patch
            device: Device to create tensors on
            reverse: If False (encoder), patches attend to points. If True (decoder), points attend to patches.
        
        Returns:
            mask: Attention mask where 0.0 means attend and -inf means mask out
        """
        if reverse:
            # Decoder: points attend to patches (causal)
            total_points = num_patches * patch_size
            mask = torch.full((total_points, num_patches), float('-inf'), device=device)
            
            for point_idx in range(total_points):
                # Calculate which patch this point belongs to
                current_patch = point_idx // patch_size
                # Allow attention to current and previous patches (causal)
                mask[point_idx, :current_patch + 1] = 0.0
        else:
            # Encoder: patches attend to points
            point_seq_len = num_patches * patch_size
            mask = torch.zeros(num_patches, point_seq_len, device=device)
            
            for patch_idx in range(num_patches):
                # Each patch can only attend to points from its own corresponding patch
                start_point = patch_idx * patch_size
                end_point = (patch_idx + 1) * patch_size
                mask[patch_idx, start_point:end_point] = 1.0
            
            # Convert to attention mask format (0 for attend, -inf for mask)
            mask = mask.masked_fill(mask == 0, float('-inf'))
            mask = mask.masked_fill(mask == 1, 0.0)
        
        return mask


class TimeMoePatchify(nn.Module):
    """
    Patchify module similar to BLT's LocalEncoder.
    
    This module converts point embeddings to patch embeddings through:
    1. Cross-attention where patches (queries) attend to points (keys/values)
    2. Local self-attention for point embeddings (like BLT's LocalEncoder)
    
    Following BLT's design, this combines cross-attention and local self-attention
    without additional feedforward networks.
    """
    
    def __init__(self, config: TimeMoeConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.num_layers = config.num_patchify_layers
        
        # Cross-attention layers for patches attending to points
        self.cross_attention_layers = nn.ModuleList([
            TimeMoeCrossAttention(config) for _ in range(self.num_layers)
        ])
        
        # Local self-attention layers for point embeddings
        self.num_attention_layers = config.num_patchify_layers if config.use_unpatchify else config.num_patchify_layers - 1

        self.local_self_attention_layers = nn.ModuleList([
            TimeMoeLocalSelfAttention(config) for _ in range(self.num_attention_layers)
        ])
        
        # Layer norms for cross-attention
        self.cross_attn_norms = nn.ModuleList([
            TimeMoeRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
            for _ in range(self.num_layers)
        ])
    
    def forward(self, point_embeddings, initial_patch_embeds, patch_attention_masks=None):
        """
        Args:
            point_embeddings: [batch_size, seq_len, hidden_size] - Input point embeddings
            initial_patch_embeds: [batch_size, num_patches, hidden_size] - Initial patch representations
            patch_attention_masks: Optional cross-attention masks
        
        Returns:
            patch_embeddings: [batch_size, num_patches, hidden_size] - Refined patch embeddings
            point_embeddings: [batch_size, seq_len, hidden_size] - Updated point embeddings (for unpatchify)
        """
        patch_embeddings = initial_patch_embeds
        current_point_embeddings = point_embeddings
        
        for layer_idx in range(self.num_layers):
            # Cross-attention: patches attend to points
            residual = patch_embeddings
            patch_embeddings_norm = self.cross_attn_norms[layer_idx](patch_embeddings)
            
            # print(patch_attention_masks.shape)
            # breakpoint()

            cross_attn_output = self.cross_attention_layers[layer_idx](
                x=patch_embeddings_norm,    # queries (patches)
                kv=current_point_embeddings, # keys/values (points)
                mask=patch_attention_masks
            )
            patch_embeddings = residual + cross_attn_output
            
            # Local self-attention for point embeddings (like BLT's LocalEncoder)
            # This updates point embeddings using local context
            if layer_idx < self.num_attention_layers:
                current_point_embeddings = self.local_self_attention_layers[layer_idx](
                    current_point_embeddings
                )

            # if patch_attention_masks.shape[-1] != 1024:
            #     breakpoint()
        
        return patch_embeddings, current_point_embeddings


class TimeMoeLinearPatchify(nn.Module):
    """
    Combined input embedding and patch embedding module that directly processes raw input.
    This is a more efficient alternative to the transformer-based TimeMoePatchify.
    
    The approach:
    1. Groups consecutive time points into patches of size `patch_size`
    2. Flattens each patch: [patch_size, input_size] -> [patch_size * input_size]
    3. Projects via linear layer: [patch_size * input_size] -> [hidden_size]
    4. Applies layer normalization for stability
    
    This combines input embedding and patch embedding in one step, reducing computation
    and memory usage compared to first embedding each point then combining patches.
    
    Compared to TimeMoePatchify:
    - 95%+ fewer parameters (no cross-attention layers)
    - Much faster computation (single matrix multiplication vs. multiple attention layers)
    - Simpler and more stable training
    - More efficient by combining input and patch embedding steps
    - Still captures patch-level information effectively for many time series tasks
    """

    def __init__(self, config: TimeMoeConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        
        # Combined input + patch embedding: directly from raw input to patch embeddings
        # Takes patch_size * input_size input and outputs hidden_size (one embedding per patch)
        self.patch_projection = nn.Linear(
            self.patch_size * self.input_size, 
            self.hidden_size, 
            bias=True
        )
        
        # Gate projection for gating mechanism (similar to TimeMoeInputEmbedding)
        self.gate_projection = nn.Linear(
            self.patch_size * self.input_size, 
            self.hidden_size, 
            bias=True
        )
        
        # Activation function
        self.act_fn = ACT2FN[config.hidden_act]
        
        # Optional layer norm for stability
        self.layer_norm = TimeMoeRMSNorm(self.hidden_size, eps=getattr(config, 'rms_norm_eps', 1e-6))
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Raw input of shape [batch_size, seq_len, input_size]
        
        Returns:
            torch.Tensor: Patch embeddings of shape [batch_size, num_patches, hidden_size]
        """
        batch_size, seq_len, input_size = x.shape
        
        # Calculate number of patches
        num_patches = seq_len // self.patch_size
        
        if num_patches == 0:
            # If sequence length is smaller than patch size, pad to at least one patch
            pad_length = self.patch_size - seq_len
            x_padded = F.pad(x, (0, 0, 0, pad_length), mode='constant', value=0)
            num_patches = 1
            truncated_seq_len = self.patch_size
            x_truncated = x_padded
        else:
            # Truncate sequence to make it divisible by patch_size
            truncated_seq_len = num_patches * self.patch_size
            x_truncated = x[:, :truncated_seq_len, :]
        
        # Reshape to group sequences into patches
        # [batch_size, num_patches, patch_size, input_size]
        patches = x_truncated.view(batch_size, num_patches, self.patch_size, input_size)
        
        # Flatten each patch: [batch_size, num_patches, patch_size * input_size]
        flattened_patches = patches.view(batch_size, num_patches, self.patch_size * input_size)
        
        # Combined input + patch embedding with gating mechanism
        # Similar to TimeMoeInputEmbedding but for patches
        gate_output = self.act_fn(self.gate_projection(flattened_patches))
        embed_output = self.patch_projection(flattened_patches)
        patch_embeddings = gate_output * embed_output
        
        # Apply layer normalization
        patch_embeddings = self.layer_norm(patch_embeddings)
        
        return patch_embeddings


class TimeMoeLinearUnpatchify(nn.Module):
    """
    Unpatchify module that projects patch embeddings back to point embeddings.
    
    This module reverses the patching operation by expanding each patch embedding
    into multiple point embeddings, enabling fine-grained temporal predictions.
    
    For the linear patch embedding case, this module:
    1. Takes patch embeddings of shape [batch_size, num_patches, hidden_size]
    2. Projects each patch embedding to patch_size point embeddings
    3. Outputs point embeddings of shape [batch_size, seq_len, hidden_size]
    
    This allows the model to make predictions at the original temporal resolution
    rather than just at the patch level.
    """
    
    def __init__(self, config: TimeMoeConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        
        # Project each patch embedding to patch_size point embeddings
        # Input: [hidden_size] -> Output: [patch_size * hidden_size]
        self.unpatch_projection = nn.Linear(
            self.hidden_size,
            self.patch_size * self.hidden_size,
            bias=True
        )
        
        # Optional layer norm for stability
        self.layer_norm = TimeMoeRMSNorm(self.hidden_size, eps=getattr(config, 'rms_norm_eps', 1e-6))
        
    def forward(self, patch_embeddings, original_seq_len=None):
        """
        Args:
            patch_embeddings (torch.Tensor): Patch embeddings of shape [batch_size, num_patches, hidden_size]
            original_seq_len (int, optional): Original sequence length before patching
                                            If None, uses num_patches * patch_size
        
        Returns:
            torch.Tensor: Point embeddings of shape [batch_size, seq_len, hidden_size]
        """
        batch_size, num_patches, hidden_size = patch_embeddings.shape
        
        # Project patch embeddings to point embeddings
        # [batch_size, num_patches, patch_size * hidden_size]
        unpatch_output = self.unpatch_projection(patch_embeddings)
        
        # Reshape to separate patch_size dimension
        # [batch_size, num_patches, patch_size, hidden_size]
        unpatch_reshaped = unpatch_output.view(
            batch_size, num_patches, self.patch_size, hidden_size
        )
        
        # Flatten to get point embeddings
        # [batch_size, num_patches * patch_size, hidden_size]
        point_embeddings = unpatch_reshaped.view(
            batch_size, num_patches * self.patch_size, hidden_size
        )
        
        # Truncate to original sequence length if specified
        if original_seq_len is not None and original_seq_len < point_embeddings.size(1):
            point_embeddings = point_embeddings[:, :original_seq_len, :]
        
        # Apply layer normalization
        point_embeddings = self.layer_norm(point_embeddings)
        
        return point_embeddings


# Copied from transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding with Mistral->TimeMOE
class TimeMoeRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->TimeMOE
class TimeMoeRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class TimeMoeTemporalBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class TimeMoeMLP(TimeMoeTemporalBlock):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__(hidden_size, intermediate_size, hidden_act)

    def forward(self, hidden_state):
        return super().forward(hidden_state), None


class TimeMoeSparseExpertsLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.norm_topk_prob = False

        moe_intermediate_size = self.config.intermediate_size // self.top_k

        # gating
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [TimeMoeTemporalBlock(
                hidden_size=self.config.hidden_size,
                intermediate_size=moe_intermediate_size,
                hidden_act=self.config.hidden_act,
            ) for _ in range(self.num_experts)]
        )

        self.shared_expert = TimeMoeTemporalBlock(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            hidden_act=self.config.hidden_act,
        )
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits -> (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        shared_expert_output = self.shared_expert(hidden_states)
        # Clamp gate logits to prevent extreme sigmoid outputs
        gate_logits = torch.clamp(self.shared_expert_gate(hidden_states), min=-10.0, max=10.0)
        shared_expert_output = F.sigmoid(gate_logits) * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


# Copied from transformers.models.qwen2.modeling_qwen2.Qwen2Attention with Qwen2->TimeMoe
class TimeMoeAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: TimeMoeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = TimeMoeRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class TimeMoeFlashAttention2(TimeMoeAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:

            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
            self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        origin_dtype = query_states.dtype
        if origin_dtype not in [torch.bfloat16, torch.float16]:
            query_states = query_states.to(dtype=torch.bfloat16)
            key_states = key_states.to(dtype=torch.bfloat16)
            value_states = value_states.to(dtype=torch.bfloat16)

        # without attention mask to faster speed
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout,
            softmax_scale=softmax_scale,
            causal=causal
        )
        if origin_dtype not in [torch.bfloat16, torch.float16]:
            return attn_output.to(origin_dtype)
        else:
            return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

TIME_MOE_ATTENTION_CLASSES = {
    "eager": TimeMoeAttention,
    'flash_attention_2': TimeMoeFlashAttention2,
}

class TimeMoeDecoderLayer(nn.Module):
    def __init__(self, config: TimeMoeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.self_attn = TIME_MOE_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        if self.config.use_dense:
            self.ffn_layer = TimeMoeMLP(
                hidden_size=self.config.hidden_size,
                intermediate_size=self.config.intermediate_size,
                hidden_act=self.config.hidden_act,
            )
        else:
            self.ffn_layer = TimeMoeSparseExpertsLayer(config)
        self.input_layernorm = TimeMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = TimeMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.ffn_layer(hidden_states)
        hidden_states = residual + hidden_states

        if not output_attentions:
            self_attn_weights = None

        if not use_cache:
            present_key_value = None
        return hidden_states, self_attn_weights, present_key_value, router_logits


class TimeMoePreTrainedModel(PreTrainedModel):
    config_class = TimeMoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TimeMoeDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = False
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class TimeMoeModel(TimeMoePreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`TimeMoeDecoderLayer`]

    Args:
        config: TimeMoeConfig
    """

    def __init__(self, config: TimeMoeConfig):
        super().__init__(config)
        
        self.patch_size = config.patch_size
        patch_embedding_type = getattr(config, 'patch_embedding_type', 'transformer')
        
        # Only create input embedding layer when needed
        # Linear patch embedding processes raw input directly, so no input embedding needed
        if self.patch_size <= 1 or patch_embedding_type != 'linear':
            self.embed_layer = TimeMoeInputEmbedding(config)
        else:
            self.embed_layer = None

        # if self.config.patching_strategy == "adaptive":

        #     # Use Chronos for computing entropy
        #     self.chronos_pipeline = BaseChronosPipeline.from_pretrained(
        #         "../model_weights/chronos/chronos-t5-small",
        #         device_map="cuda:0",#self.device,
        #         torch_dtype=torch.bfloat16,
        #     )

        if self.patch_size > 1:
            # Choose patch embedding type based on configuration
            if patch_embedding_type == 'linear':
                self.patch_embedding = TimeMoeLinearPatchify(config)
            else:
                self.patch_embedding = TimeMoePatchify(config)
        
        # Create unpatchify module based on configuration
        if config.use_unpatchify and self.patch_size > 1:
            if patch_embedding_type == 'linear':
                # Add unpatchify module for linear patch embedding to project back to point embeddings
                self.unpatchify = TimeMoeLinearUnpatchify(config)
            else:
                # Add transformer-based unpatchify for sophisticated reconstruction
                self.unpatchify = TimeMoeUnpatchify(config)
        else:
            self.unpatchify = None

        self.layers = nn.ModuleList(
            [TimeMoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = TimeMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: torch.FloatTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        # input_ids is the input of time series, its shape is [batch_size, seq_len, input_size]
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            if len(input_ids.shape) == 2:
                input_ids.unsqueeze_(dim=-1)
            batch_size, seq_length, _ = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if inputs_embeds is None:
            # For linear patch embedding, process raw input directly
            # For transformer patch embedding, use traditional input embedding first
            if self.patch_size > 1 and self.config.patch_embedding_type == 'linear':
                # Skip input embedding - patch embedding will handle raw input directly
                inputs_embeds = input_ids
            else:
                # Traditional approach: input embedding first
                if self.embed_layer is not None:
                    inputs_embeds = self.embed_layer(input_ids)
                else:
                    raise ValueError("Input embedding layer is not available but required for this configuration")
        
        if self.config.patch_embedding_type == 'transformer':
            if self.config.patching_strategy == "adaptive":

                # # Get predictions with logits for entropy computation
                # batch_predictions, batch_entropies, batch_relative_entropies, batch_logits = compute_entropies(
                #     input_ids, self.chronos_pipeline, self.config
                # )

                # # Compute patches based on entropies
                # batch_patches, thresholds = compute_patches(
                #     batch_relative_entropies, 
                #     use_relative_entropies=True,
                #     min_patch_size=4,
                #     max_patch_size=8
                # )

                # batch_patch_starts = compute_random_patches(
                #     batch_size=batch_size,
                #     seq_length=seq_length,
                #     min_patch_size=4,
                #     max_patch_size=8,
                #     seed=42  # For reproducible results during testing
                # )
                batch_patch_starts, avg_patch_size = compute_threshold_patches(input_ids, threshold_factor=0.01, min_patch_size=2, max_patch_size=8)

                # Compute patch embeddings using max pooling over variable-length patches
                initial_patch_embeds = compute_initial_patch_embeddings_with_pooling(
                    inputs_embeds, batch_patch_starts, self.config.hidden_size, self.config
                )

                # Create attention masks for variable-length patches
                patch_attention_masks = compute_patch_attention_masks(
                    batch_patch_starts=batch_patch_starts,
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype
                )

                # Store batch patches for loss computation
                self._batch_patches = batch_patch_starts
                # print(patch_attention_masks.shape)
                # breakpoint()
            else:
                # compute maxpooled patches with fixed patch size
                initial_patch_embeds = compute_initial_patch_embeddings_with_pooling(
                    inputs_embeds, None, self.config.hidden_size, self.config
                )
                
                # Create attention masks for fixed-size patches
                patch_attention_masks = compute_patch_attention_masks(
                    batch_patches=None,
                    batch_size=batch_size,
                    seq_length=seq_length,
                    patch_size=self.patch_size,
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype
                )

                # Store batch patches for loss computation
                self._batch_patches = None

        # Store original sequence length for attention mask computation
        original_seq_length = seq_length
        
        # Initialize encoder point embeddings (needed for transformer unpatchify)
        encoder_point_embeddings = None

        if self.patch_size > 1:
            # Apply patch embedding
            if self.config.patch_embedding_type== 'linear':
                # Linear patch embedding expects raw input [batch_size, seq_len, input_size]
                inputs_embeds = self.patch_embedding(inputs_embeds)
            else:
                # Transformer patch embedding expects embedded input [batch_size, seq_len, hidden_size]
                # and returns both patch embeddings and final point embeddings
                inputs_embeds, encoder_point_embeddings = self.patch_embedding(inputs_embeds, initial_patch_embeds, patch_attention_masks)
        
        # Store encoder point embeddings for transformer unpatchify
        self._encoder_point_embeddings = encoder_point_embeddings

        # Update sequence length after patching
        batch_size, seq_length, _ = inputs_embeds.shape

        # Update attention mask for patching BEFORE computing position_ids
        if self.patch_size > 1 and attention_mask is not None:
            # Truncate attention mask to match the truncated sequence length used in patching
            num_patches = original_seq_length // self.patch_size
            if num_patches > 0:
                truncated_seq_len = num_patches * self.patch_size
                attention_mask = attention_mask[:, :truncated_seq_len]
            
            # Create patch-level attention mask
            # Each patch gets attention if any of its constituent tokens have attention
            patch_attention_mask = attention_mask.view(batch_size, num_patches, self.patch_size)
            patch_attention_mask = patch_attention_mask.max(dim=2)[0]  # [batch_size, num_patches]
            attention_mask = patch_attention_mask
        elif self.patch_size > 1 and attention_mask is None:
            # Create default attention mask for patches
            attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long, device=inputs_embeds.device)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            # position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            position_ids = position_ids.view(-1, seq_length)
        else:
            # FIX ME: update the position_ids to match the new sequence length by truncating
            position_ids = position_ids[:, past_key_values_length:past_key_values_length + seq_length]
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=None,
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = ()
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            all_router_logits += (layer_outputs[-1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if use_cache:
                next_decoder_cache = layer_outputs[2]

        hidden_states = self.norm(hidden_states)
        
        # Apply unpatchify if enabled
        # This projects patch embeddings back to point embeddings for fine-grained predictions
        if self.unpatchify is not None:
            # Store the patch-level sequence length
            patch_seq_len = hidden_states.shape[1]
            
            # Calculate original sequence length before patching
            original_seq_len = getattr(self, '_original_seq_length', patch_seq_len * self.patch_size)
            
            # Apply appropriate unpatchify based on patch embedding type
            if self.config.patch_embedding_type == 'linear':
                hidden_states = self.unpatchify(hidden_states, patch_seq_len * self.patch_size)
            else:
                # transpose of encoder patch_attention_masks
                hidden_states = self.unpatchify(hidden_states, self._encoder_point_embeddings, patch_attention_masks.transpose(1, 2))

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits]
                if v is not None
            )
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits
        )


class TimeMoeOutputLayer(nn.Module):

    def __init__(self, hidden_size: int, horizon_length: int, input_size: int = 1):
        super().__init__()

        self.out_layer = nn.Linear(
            hidden_size,
            input_size * horizon_length,
            bias=False,
        )

    def forward(self, x):
        """

        Args:
            x (torch.FloatTensor): with shape [B, seq_len, hidden_size]

        Returns:
    `       torch.FloatTensor: final prediction with shape [B, seq_len, input_size]
        """
        return self.out_layer(x)


class TimeMoeForPrediction(TimeMoePreTrainedModel, TSGenerationMixin):

    def __init__(self, config: TimeMoeConfig):
        super().__init__(config)
        self.config = config
        self.apply_aux_loss = config.apply_aux_loss
        self.num_experts_per_tok = config.num_experts_per_tok
        self.router_aux_loss_factor = config.router_aux_loss_factor

        self.model = TimeMoeModel(config)
        # output layer
        lm_head_list = []
        self.horizon_length_map = {}
        for i, horizon_length in enumerate(config.horizon_lengths):
            lm_head_list.append(
                TimeMoeOutputLayer(
                    hidden_size=self.config.hidden_size,
                    input_size=self.config.input_size,
                    horizon_length=horizon_length,
                )
            )
            self.horizon_length_map[horizon_length] = i
        self.lm_heads = nn.ModuleList(lm_head_list)

        self.loss_function = torch.nn.HuberLoss(reduction='none', delta=2.0)

        # Initialize weights and apply final processing
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
            self,
            input_ids: torch.FloatTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.FloatTensor] = None,
            loss_masks: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            max_horizon_length: Optional[int] = None,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        predictions = None

        loss = None
        aux_loss = None
        if labels is not None:
            # AutoRegressive loss
            ar_loss = 0.0

            for lm_head, horizon_length in zip(self.lm_heads, self.config.horizon_lengths):
                one_predictions = lm_head(hidden_states)
                one_loss = self.calc_ar_loss(one_predictions, labels, loss_masks, horizon_length, patch_starts=self.model._batch_patches)
                ar_loss += one_loss
                if predictions is None:
                    predictions = one_predictions
            loss = ar_loss / len(self.config.horizon_lengths)

            if self.apply_aux_loss:
                router_logits = outputs.router_logits if return_dict else outputs[-1]

                temporal_aux_loss = load_balancing_loss_func(
                    router_logits,
                    top_k=self.num_experts_per_tok,
                    num_experts=self.config.num_experts,
                    attention_mask=attention_mask
                )
                loss += self.router_aux_loss_factor * temporal_aux_loss.to(loss.device)
        else:
            if max_horizon_length is None:
                horizon_length = self.config.horizon_lengths[0]
                max_horizon_length = horizon_length
            else:
                horizon_length = self.config.horizon_lengths[0]
                for h in self.config.horizon_lengths[1:]:
                    if h > max_horizon_length:
                        break
                    else:
                        horizon_length = h
            lm_head = self.lm_heads[self.horizon_length_map[horizon_length]]
            predictions = lm_head(hidden_states)
            if horizon_length > max_horizon_length:
                predictions = predictions[:, :, : self.config.input_size * max_horizon_length]

        if not return_dict:
            output = (predictions,) + outputs[1:]
            return (loss, aux_loss) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=predictions,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def calc_ar_loss(self, predictions, labels, loss_masks, horizon_length, patch_starts=None):
        if len(labels.shape) == 2:
            labels.unsqueeze_(dim=-1)
            labels = labels.to(predictions.device)
        if loss_masks is not None and len(loss_masks.shape) == 2:
            loss_masks.unsqueeze_(dim=-1)
            loss_masks = loss_masks.to(predictions.device)

        # Get patch size from config
        patch_size = getattr(self.config, 'patch_size', 1)

        if horizon_length > 1:
            batch_size, seq_len, output_size = predictions.shape
            shift_predictions = predictions.view(batch_size, seq_len, horizon_length, -1)

            # pad to the same length with predictions
            labels = F.pad(labels.transpose(-1, -2), (0, horizon_length - 1), mode='constant', value=0)
            shift_labels = labels.unfold(dimension=-1, size=horizon_length, step=1)
            shift_labels = shift_labels.permute(0, 2, 3, 1)

            if loss_masks is not None:
                loss_masks = F.pad(loss_masks.transpose(-1, -2), (0, horizon_length - 1), mode='constant', value=0)
                loss_masks = loss_masks.unfold(dimension=-1, size=horizon_length, step=1)
                loss_masks = loss_masks.permute(0, 2, 3, 1)

        else:
            shift_predictions = predictions
            shift_labels = labels

        # If patch_size > 1, we need to sample labels at patch boundaries
        # Skip this sampling if unpatchify is enabled, as the model outputs point-level predictions
        if patch_size > 1 and self.config.use_unpatchify == False:
            if patch_starts is not None:
                # Adaptive patching: patch_starts is a list of binary vectors
                max_patches = max([starts.sum().item() for starts in patch_starts])
                padded_labels = []
                padded_masks = []
                
                for i in range(shift_labels.shape[0]):
                    # Get patch boundaries from binary vector
                    patch_indices = torch.nonzero(patch_starts[i], as_tuple=False).flatten()
                    sampled_labels = shift_labels[i, patch_indices, :]  # [num_patches_i, ...]
                    
                    # Pad to max_patches
                    pad_len = max_patches - sampled_labels.shape[0]
                    if pad_len > 0:
                        pad = torch.zeros(pad_len, *sampled_labels.shape[1:], 
                                        device=sampled_labels.device, dtype=sampled_labels.dtype)
                        sampled_labels = torch.cat([sampled_labels, pad], dim=0)
                    padded_labels.append(sampled_labels)
                    
                    if loss_masks is not None:
                        sampled_masks = loss_masks[i, patch_indices, :]
                        if pad_len > 0:
                            pad_mask = torch.zeros(pad_len, *sampled_masks.shape[1:], 
                                                device=sampled_masks.device, dtype=sampled_masks.dtype)
                            sampled_masks = torch.cat([sampled_masks, pad_mask], dim=0)
                        padded_masks.append(sampled_masks)
                
                shift_labels = torch.stack(padded_labels, dim=0)
                if loss_masks is not None:
                    loss_masks = torch.stack(padded_masks, dim=0)
                
                # Pad predictions to max_patches
                shift_predictions = shift_predictions[:, :max_patches, :]
            else:
                # Fixed patching: sample at regular intervals
                seq_len = predictions.shape[1]
                max_start_pos = seq_len * patch_size - 1 
                patch_starts_indices = torch.arange(0, max_start_pos + 1, patch_size, device=labels.device)
                shift_labels = shift_labels[:, patch_starts_indices, :]
                loss_masks = loss_masks[:, patch_starts_indices, :] if loss_masks is not None else None

        # Calculate loss with mask
        losses = self.loss_function(shift_predictions, shift_labels)

        if loss_masks is not None:
            losses = losses * loss_masks
            loss = losses.sum() / loss_masks.sum()
        else:
            loss = torch.mean(losses)

        return loss


    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                if isinstance(past_key_values, DynamicCache):
                    past_length = past_key_values.seen_tokens
                else:
                    past_length = cache_length

                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                    max_cache_length is not None
                    and attention_mask is not None
                    and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            logger.info('Use input_embedding')
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


class TimeMoeUnpatchify(nn.Module):
    """
    Unpatchify module similar to BLT's LocalDecoder.
    
    This module converts patch embeddings back to point embeddings through:
    1. Cross-attention where points (queries) attend to patches (keys/values)
    """
    
    def __init__(self, config: TimeMoeConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.num_layers = config.num_unpatchify_layers

        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            TimeMoeCrossAttention(config) for _ in range(self.num_layers)
        ])

        # Local self-attention layers for point embeddings
        self.local_self_attention_layers = nn.ModuleList([
            TimeMoeLocalSelfAttention(config) for _ in range(self.num_layers)
        ])
        
        # Layer norms for cross-attention
        self.cross_attn_norms = nn.ModuleList([
            TimeMoeRMSNorm(self.hidden_size, config.rms_norm_eps)
            for _ in range(self.num_layers)
        ])

    def forward(self, patch_embeddings, encoder_point_embeddings, cross_attention_mask=None):
        """
        Args:
            patch_embeddings: [batch_size, num_patches, hidden_size] - Input patch embeddings from main model
            encoder_point_embeddings: [batch_size, seq_len, hidden_size] - Point embeddings from encoder
            batch_patch_starts: Optional list of binary vectors for adaptive patching
        
        Returns:
            point_embeddings: [batch_size, seq_len, hidden_size] - Refined point embeddings
        """

        # Use encoder point embeddings as starting point
        point_embeddings = encoder_point_embeddings
        
        for layer_idx in range(self.num_layers):
            # Cross-attention: points attend to patches
            residual = point_embeddings
            point_embeddings_norm = self.cross_attn_norms[layer_idx](point_embeddings)
            
            cross_attn_output = self.cross_attention_layers[layer_idx](
                x=point_embeddings_norm,  # queries (points)
                kv=patch_embeddings,      # keys/values (patches)
                mask=cross_attention_mask
            )
            point_embeddings = residual + cross_attn_output

            # Local self-attention: points attend to each other
            point_embeddings = self.local_self_attention_layers[layer_idx](
                point_embeddings
            )

        return point_embeddings


def compute_patch_attention_masks(
    batch_patch_starts: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    batch_size: Optional[int] = None,
    seq_length: Optional[int] = None,
    patch_size: Optional[int] = None,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Create attention masks for cross-attention between patch embeddings and point embeddings.
    Fully vectorized, no Python loops.
    
    Returns a tensor of shape [B, P, L] with 0.0 where attend is allowed and -inf elsewhere.
    """
    dtype = dtype or torch.float32
    if batch_patch_starts is not None:
        # --- Adaptive (variable-length) ---
        # Accept either a [B, L] tensor or a list of [L] tensors
        if isinstance(batch_patch_starts, torch.Tensor):
            start_mask = batch_patch_starts.to(device).bool()  # [B, L]
        else:
            start_mask = torch.stack(batch_patch_starts, dim=0).to(device).bool()  # [B, L]

        B, L = start_mask.shape
        # compute patch IDs: cumsum start flags minus 1 → values in [-1..], clamp to [0..P-1]
        raw_ids = torch.cumsum(start_mask.long(), dim=1) - 1  # [B, L]
        P = int(start_mask.sum(dim=1).max().item())
        patch_ids = raw_ids.clamp(min=0, max=P - 1)         # [B, L]

        # build a [B, P, L] boolean mask: True where patch_ids == p
        patch_ids_exp = patch_ids.unsqueeze(1)               # [B, 1, L]
        range_p = torch.arange(P, device=device).view(1, P, 1)  # [1, P, 1]
        valid = (patch_ids_exp == range_p)                   # [B, P, L]

        # create final attention mask
        attn_mask = torch.full((B, P, L), float('-inf'),
                               device=device, dtype=dtype)
        attn_mask = attn_mask.masked_fill(valid, 0.0)
        return attn_mask

    else:
        # --- Fixed-size patches ---
        if batch_size is None or seq_length is None or patch_size is None:
            raise ValueError("For fixed patches, must provide batch_size, seq_length, and patch_size")
        B = batch_size
        # how many patches?
        P = seq_length // patch_size
        if P == 0:
            P = 1
        L = P * patch_size

        # build [P, L] mask once
        patch_idx = torch.arange(P, device=device).view(P, 1)      # [P, 1]
        point_idx = torch.arange(L, device=device).view(1, L)      # [1, L]
        valid = (point_idx >= patch_idx * patch_size) & \
                (point_idx < (patch_idx + 1) * patch_size)         # [P, L]

        # expand to [B, P, L]
        attn_mask = torch.full((B, P, L), float('-inf'),
                               device=device, dtype=dtype)
        attn_mask = attn_mask.masked_fill(valid.unsqueeze(0), 0.0)
        return attn_mask