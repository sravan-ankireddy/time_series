import math
from typing import Optional, Tuple, List, Union
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

# from chronos import BaseChronosPipeline


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
    entropy_batch_size = getattr(config, 'entropy_batch_size', 32)  # Default batch size
    
    all_preds = []
    all_logits = []
    
    for i in range(0, len(causal_context_list), entropy_batch_size):
        batch_end = min(i + entropy_batch_size, len(causal_context_list))
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


def compute_patches(
    entropies: List[List[float]],
    use_relative_entropies: bool = True,
    threshold: Optional[float] = None,
    min_patch_size: int = 4,
    max_patch_size: int = 8
) -> Tuple[List[List[Tuple[int, int]]], List[float]]:
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
        Tuple of (batch_patches, thresholds_used):
        - batch_patches: List of patch boundaries for each sequence 
                        Each patch is (start_idx, end_idx) where end_idx is exclusive
        - thresholds_used: List of thresholds used for each sequence
    
    """
    batch_patches = []
    thresholds_used = []
    
    for sequence_entropies in entropies:
        if len(sequence_entropies) == 0:
            batch_patches.append([])
            thresholds_used.append(0.0)
            continue
        
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
                    # This captures significant entropy increases while being robust to outliers
                    sequence_threshold = q75 + 0.5 * iqr
                    # Ensure minimum threshold to avoid too many patches
                    sequence_threshold = max(sequence_threshold, 0.2)
                else:
                    sequence_threshold = 0.5  # Default fallback
            else:
                # For absolute entropies: Use mean + standard deviation
                mean_entropy = np.mean(entropy_array)
                std_entropy = np.std(entropy_array)
                # Use mean + 1.0 * std as threshold
                sequence_threshold = mean_entropy + 1.0 * std_entropy
                # Ensure reasonable bounds
                sequence_threshold = max(sequence_threshold, 1.0)
        else:
            sequence_threshold = threshold
        
        thresholds_used.append(sequence_threshold)
        
        # Generate patches based on entropy threshold
        patches = []
        current_patch_start = 0
        seq_length = len(sequence_entropies)
        
        i = 0
        while i < seq_length:
            # Check if we should start a new patch due to high entropy
            if (i > current_patch_start and 
                sequence_entropies[i] > sequence_threshold and 
                (i - current_patch_start) >= min_patch_size):
                
                # End current patch
                patches.append((current_patch_start, i))
                current_patch_start = i
            
            # Force patch boundary if current patch reaches max size
            elif (i - current_patch_start) >= max_patch_size:
                patches.append((current_patch_start, i))
                current_patch_start = i
            
            i += 1
        
        # Add the final patch
        if current_patch_start < seq_length:
            final_patch_end = seq_length
            final_patch_size = final_patch_end - current_patch_start
            
            # If final patch is too small, merge with previous patch
            if final_patch_size < min_patch_size and len(patches) > 0:
                # Remove last patch and extend it
                last_start, _ = patches.pop()
                patches.append((last_start, final_patch_end))
            else:
                patches.append((current_patch_start, final_patch_end))
        
        # Ensure we have at least one patch
        if not patches:
            patches.append((0, seq_length))
        
        # Validate patches
        validated_patches = []
        for start, end in patches:
            patch_size = end - start
            if patch_size >= min_patch_size:
                validated_patches.append((start, end))
            elif validated_patches:
                # Merge small patch with previous one
                prev_start, _ = validated_patches[-1]
                validated_patches[-1] = (prev_start, end)
            else:
                # First patch, keep it even if small
                validated_patches.append((start, end))
        
        batch_patches.append(validated_patches)
    
    return batch_patches, thresholds_used


def compute_initial_patch_embeddings_with_pooling(
    input_embeddings: torch.Tensor,
    batch_patches: Optional[List[List[Tuple[int, int]]]] = None,
    hidden_size: int = None,
    config = None,
    pooling_type: str = "max"
) -> torch.Tensor:
    """
    Compute patch embeddings using pooling over variable-length or fixed-size patches.
    
    This function supports two modes:
    1. Variable-length patches: Uses provided batch_patches for adaptive patching
    2. Fixed-size patches: Uses config.patch_size to create uniform patches
    
    Args:
        input_embeddings: Input embeddings tensor of shape [batch_size, seq_length, hidden_size]
        batch_patches: Optional list of patch boundaries for each sequence in the batch
                      Each sequence contains patches as (start_idx, end_idx) tuples
                      where end_idx is exclusive. If None, uses fixed patch size from config.
        hidden_size: Hidden dimension size for output embeddings (inferred if None)
        config: Model configuration containing patch_size for fixed patching mode
        pooling_type: Type of pooling to use, either "max" or "avg" (default: "max")
    
    Returns:
        torch.Tensor: Patch embeddings of shape [batch_size, num_patches, hidden_size]
                     For variable patches: num_patches = max patches across sequences
                     For fixed patches: num_patches = seq_length // patch_size
    """
    batch_size, seq_length, embed_dim = input_embeddings.shape
    
    # Infer hidden_size if not provided
    if hidden_size is None:
        hidden_size = embed_dim
    
    # Determine patching mode
    if batch_patches is not None:
        # Variable-length patches mode (adaptive)
        max_num_patches = max(len(patches) for patches in batch_patches) if batch_patches else 1
        
        # Initialize output tensor with zeros
        patch_embeddings = torch.zeros(
            batch_size, max_num_patches, hidden_size,
            dtype=input_embeddings.dtype,
            device=input_embeddings.device
        )
        
        for batch_idx, patches in enumerate(batch_patches):
            for patch_idx, (start_idx, end_idx) in enumerate(patches):
                # Ensure indices are within bounds
                start_idx = max(0, min(start_idx, seq_length - 1))
                end_idx = max(start_idx + 1, min(end_idx, seq_length))
                
                # Extract embeddings for this patch
                patch_embeddings_slice = input_embeddings[batch_idx, start_idx:end_idx, :]  # [patch_length, hidden_size]
                
                # Apply pooling
                if patch_embeddings_slice.size(0) > 0:
                    if pooling_type == "max":
                        patch_embedding = torch.max(patch_embeddings_slice, dim=0)[0]  # [hidden_size]
                    else:  # avg pooling
                        patch_embedding = torch.mean(patch_embeddings_slice, dim=0)  # [hidden_size]
                    patch_embeddings[batch_idx, patch_idx, :] = patch_embedding
    
    else:
        # Fixed-size patches mode
        if config is None:
            raise ValueError("Config must be provided when batch_patches is None for fixed-size patching")
        
        patch_size = getattr(config, 'patch_size', 1)
        if patch_size <= 1:
            # No patching, return original embeddings
            return input_embeddings
        
        # Calculate number of patches
        num_patches = seq_length // patch_size
        
        if num_patches == 0:
            # Sequence too short, return zero tensor
            patch_embeddings = torch.zeros(
                batch_size, 1, hidden_size,
                dtype=input_embeddings.dtype,
                device=input_embeddings.device
            )
        else:
            # Truncate sequence to multiple of patch_size
            truncated_seq_len = num_patches * patch_size
            truncated_embeddings = input_embeddings[:, :truncated_seq_len, :]
            
            # Reshape to group into patches: [batch_size, num_patches, patch_size, hidden_size]
            reshaped_embeddings = truncated_embeddings.view(
                batch_size, num_patches, patch_size, hidden_size
            )
            
            # Apply pooling
            if pooling_type == "max":
                patch_embeddings = torch.max(reshaped_embeddings, dim=2)[0]  # [batch_size, num_patches, hidden_size]
            else:  # avg pooling
                patch_embeddings = torch.mean(reshaped_embeddings, dim=2)  # [batch_size, num_patches, hidden_size]
    
    return patch_embeddings


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


class TimeMoeLocalSelfAttention(nn.Module):
    """
    Local self-attention module for point embeddings with sliding window.
    Following BLT's pattern, this only does self-attention without feedforward.
    """
    
    def __init__(self, config: TimeMoeConfig, window_size: int = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_patch_attention_heads  # Use num_patch_attention_heads for local attention
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.attention_dropout = getattr(config, 'attention_dropout', 0.0)
        self.window_size = window_size if window_size is not None else getattr(config, 'local_attention_window_size', 4)
        
        # Self-attention projections for point embeddings
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Layer norm for self-attention (no feedforward in BLT pattern)
        self.layer_norm = TimeMoeRMSNorm(self.hidden_size, eps=getattr(config, 'rms_norm_eps', 1e-6))
    
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
    
    def forward(self, point_embeddings):
        """
        Args:
            point_embeddings: [batch_size, seq_len, hidden_size]
        
        Returns:
            Enhanced point embeddings: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = point_embeddings.shape
        
        # Self-attention + residual connection (BLT pattern: no feedforward)
        residual = point_embeddings
        point_embeddings = self.layer_norm(point_embeddings)
        
        # Self-attention computation
        query_states = self.q_proj(point_embeddings)
        key_states = self.k_proj(point_embeddings)
        value_states = self.v_proj(point_embeddings)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # Apply causal local attention mask
        local_mask = self.create_local_causal_attention_mask(seq_len, self.window_size, point_embeddings.device)
        attn_weights = attn_weights + local_mask.unsqueeze(0).unsqueeze(0) # Broadcast for batch and heads
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=value_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        # Output projection and residual connection
        attn_output = self.o_proj(attn_output)
        point_embeddings = residual + attn_output
        
        return point_embeddings


class TimeMoeCrossAttention(nn.Module):
    """
    Unified cross-attention block similar to BLT's CrossAttention.
    Can handle both directions:
    - Encoder: Patch embeddings (queries) attend to point embeddings (keys/values)
    - Decoder: Point embeddings (queries) attend to patch embeddings (keys/values)
    """
    def __init__(self, config: TimeMoeConfig):
        super().__init__()
        self.dim = config.hidden_size
        self.head_dim = self.dim // config.num_patch_attention_heads
        self.n_heads = config.num_patch_attention_heads
        self.n_kv_heads = config.num_patch_attention_heads # Same as n_heads for simplicity
        self.heads_per_group = self.n_heads // self.n_kv_heads
        self.attention_dropout = getattr(config, 'attention_dropout', 0.0)
        
        # Cross-attention layer norms (BLT pattern)
        self.cross_attn_norm_q = TimeMoeRMSNorm(self.dim, eps=getattr(config, 'rms_norm_eps', 1e-6))
        self.cross_attn_norm_kv = TimeMoeRMSNorm(self.dim, eps=getattr(config, 'rms_norm_eps', 1e-6))
        
        # Cross-attention projections
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
        """
        Unified cross-attention forward pass.
        
        Args:
            x: query embeddings [batch_size, seq_len_q, dim] - Can be patches or points
            kv: key/value embeddings [batch_size, seq_len_kv, dim] - Can be points or patches
            mask: attention mask for controlling attention pattern
                  Shape: [batch_size, seq_len_q, seq_len_kv] for per-sample masks
                  OR [seq_len_q, seq_len_kv] for shared mask across batch
        """
        bsz, seq_len_q, _ = x.shape
        _, seq_len_kv, _ = kv.shape
        
        # Apply layer norms following BLT pattern
        x_norm = self.cross_attn_norm_q(x)
        kv_norm = self.cross_attn_norm_kv(kv)
        
        # Cross-attention projections
        xq = self.wq(x_norm)  # queries
        xk = self.wk(kv_norm)  # keys
        xv = self.wv(kv_norm)  # values
        
        # Reshape for multi-head attention
        xq = xq.view(bsz, seq_len_q, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len_kv, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len_kv, self.n_kv_heads, self.head_dim)
        
        # Repeat k,v heads if needed
        xk = repeat_kv(xk, self.heads_per_group)
        xv = repeat_kv(xv, self.heads_per_group)
        
        # Transpose for attention computation
        xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
        # Now: xq is [batch_size, n_heads, seq_len_q, head_dim]
        # xk, xv are [batch_size, n_heads, seq_len_kv, head_dim]
        
        # Compute attention scores
        attn_weights = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        # Shape: [batch_size, n_heads, seq_len_q, seq_len_kv]
        
        # Apply attention mask if provided
        if mask is not None:
            if mask.dim() == 2:
                # Shared mask across batch: [seq_len_q, seq_len_kv]
                # Broadcast to [batch_size, n_heads, seq_len_q, seq_len_kv]
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len_q, seq_len_kv]
            elif mask.dim() == 3:
                # Per-sample mask: [batch_size, seq_len_q, seq_len_kv]
                # Broadcast to [batch_size, n_heads, seq_len_q, seq_len_kv]
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len_q, seq_len_kv]
            else:
                raise ValueError(f"Mask should be 2D or 3D, got {mask.dim()}D")
            
            attn_weights = attn_weights + mask
        
        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=xv.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, xv)
        output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D
        
        # Output projection
        output = self.wo(output.view(bsz, seq_len_q, -1))
        
        return x + output
    
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
        self.num_layers = getattr(config, 'num_patchify_layers', 2)
        
        # Cross-attention layers for patches attending to points
        self.cross_attention_layers = nn.ModuleList([
            TimeMoeCrossAttention(config) for _ in range(self.num_layers)
        ])
        
        # Local self-attention layers for point embeddings (like BLT's LocalEncoder)
        self.local_self_attention_layers = nn.ModuleList([
            TimeMoeLocalSelfAttention(config) for _ in range(self.num_layers - 1)
        ])
        
        # Layer norms for cross-attention
        self.cross_attn_norms = nn.ModuleList([
            TimeMoeRMSNorm(self.hidden_size, eps=getattr(config, 'rms_norm_eps', 1e-6))
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
            
            cross_attn_output = self.cross_attention_layers[layer_idx](
                x=patch_embeddings_norm,    # queries (patches)
                kv=current_point_embeddings, # keys/values (points)
                mask=patch_attention_masks
            )
            patch_embeddings = residual + cross_attn_output
            
            # Local self-attention for point embeddings (like BLT's LocalEncoder)
            # This updates point embeddings using local context
            if layer_idx < self.num_layers - 1:  # No local self-attention on last layer
                current_point_embeddings = self.local_self_attention_layers[layer_idx](
                    current_point_embeddings
                )
        
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


class TimeMoeTransformerUnpatchify(nn.Module):
    """
    Transformer-based unpatchify module using TimeMoeUnpatchify.
    Similar to BLT's LocalDecoder approach.
    """
    
    def __init__(self, config: TimeMoeConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        
        # Use the new unpatchify module
        self.unpatchify = TimeMoeUnpatchify(config)
        
    def forward(self, patch_embeddings, encoder_point_embeddings, original_seq_len=None):
        """
        Args:
            patch_embeddings: [batch_size, num_patches, hidden_size] - Input patch embeddings from main model
            encoder_point_embeddings: [batch_size, seq_len, hidden_size] - Point embeddings from encoder
            original_seq_len: Optional original sequence length for reshaping
        
        Returns:
            point_embeddings: [batch_size, seq_len, hidden_size] - Refined point embeddings
        """
        return self.unpatchify(patch_embeddings, encoder_point_embeddings, original_seq_len)


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

        if self.config.patching_strategy == "adaptive":

            # Use Chronos for computing entropy
            self.chronos_pipeline = BaseChronosPipeline.from_pretrained(
                "../model_weights/chronos/chronos-t5-small",
                device_map="cuda:0",#self.device,
                torch_dtype=torch.bfloat16,
            )

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
                self.unpatchify = TimeMoeTransformerUnpatchify(config)
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

                # Get predictions with logits for entropy computation
                batch_predictions, batch_entropies, batch_relative_entropies, batch_logits = compute_entropies(
                    input_ids, self.chronos_pipeline, self.config
                )

                # Compute patches based on entropies
                batch_patches, thresholds = compute_patches(
                    batch_relative_entropies, 
                    use_relative_entropies=True,
                    min_patch_size=4,
                    max_patch_size=8
                )

                # Compute patch embeddings using max pooling over variable-length patches
                initial_patch_embeds = compute_initial_patch_embeddings_with_pooling(
                    inputs_embeds, batch_patches, self.config.hidden_size, self.config
                )
                
                # Create attention masks for variable-length patches
                patch_attention_masks = compute_patch_attention_masks(
                    batch_patches=batch_patches,
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype
                )

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
            if getattr(self.config, 'patch_embedding_type', 'transformer') == 'linear':
                # Linear unpatchify: only needs patch embeddings
                hidden_states = self.unpatchify(hidden_states, original_seq_len)
            else:
                # Transformer unpatchify: needs both patch embeddings and encoder point embeddings
                encoder_point_embeddings = getattr(self, '_encoder_point_embeddings', None)
                if encoder_point_embeddings is not None:
                    hidden_states = self.unpatchify(hidden_states, encoder_point_embeddings, original_seq_len)
                else:
                    # Fallback: create dummy point embeddings if not available
                    batch_size, num_patches, hidden_size = hidden_states.shape
                    dummy_point_embeddings = torch.zeros(
                        batch_size, num_patches, self.patch_size, hidden_size,
                        device=hidden_states.device, dtype=hidden_states.dtype
                    )
                    hidden_states = self.unpatchify(hidden_states, dummy_point_embeddings, original_seq_len)

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
                one_loss = self.calc_ar_loss(one_predictions, labels, loss_masks, horizon_length)
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

    def calc_ar_loss(self, predictions, labels, loss_masks, horizon_length):
        if len(labels.shape) == 2:
            labels.unsqueeze_(dim=-1)
            # enable model parallelism
            labels = labels.to(predictions.device)
        if loss_masks is not None and len(loss_masks.shape) == 2:
            loss_masks.unsqueeze_(dim=-1)
            # enable model parallelism
            loss_masks = loss_masks.to(predictions.device)

        # Get patch size from config
        patch_size = getattr(self.config, 'patch_size', 1)

        if horizon_length > 1:
            batch_size, seq_len, output_size = predictions.shape
            shift_predictions = predictions.view(batch_size, seq_len, horizon_length, -1)

            # pad to the same length with predictions
            # shape -> [B, input_size, seq_len + horizon_length -1]
            labels = F.pad(labels.transpose(-1, -2), (0, horizon_length - 1), mode='constant', value=0)

            # shape -> [B, input_size, seq_len, horizon_length]
            shift_labels = labels.unfold(dimension=-1, size=horizon_length, step=1)
            shift_labels = shift_labels.permute(0, 2, 3, 1)

            if loss_masks is not None:
                # pad to the same length with predictions
                loss_masks = F.pad(loss_masks.transpose(-1, -2), (0, horizon_length - 1), mode='constant', value=0)

                loss_masks = loss_masks.unfold(dimension=-1, size=horizon_length, step=1)
                loss_masks = loss_masks.permute(0, 2, 3, 1)

        else:
            shift_predictions = predictions
            shift_labels = labels

        # If patch_size > 1, we need to sample labels at patch boundaries
        # Skip this sampling if unpatchify is enabled, as the model outputs point-level predictions
        if patch_size > 1 and not getattr(self.config, 'use_unpatchify', False):
            seq_len = predictions.shape[1]
            max_start_pos = seq_len * patch_size - 1 
            patch_starts = torch.arange(0, max_start_pos + 1, patch_size, device=labels.device)
            shift_labels = shift_labels[:, patch_starts, :]
            loss_masks = loss_masks[:, patch_starts, :] if loss_masks is not None else None
                
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
    
    Following BLT's LocalDecoder design, this only uses cross-attention
    without self-attention or feedforward networks.
    """
    
    def __init__(self, config: TimeMoeConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.num_layers = getattr(config, 'num_unpatchify_layers', 2)

        # Only cross-attention layers (like BLT's LocalDecoder)
        self.cross_attention_layers = nn.ModuleList([
            TimeMoeCrossAttention(config) for _ in range(self.num_layers)
        ])
        
        # Layer norms for cross-attention only
        self.cross_attn_norms = nn.ModuleList([
            TimeMoeRMSNorm(self.hidden_size, eps=getattr(config, 'rms_norm_eps', 1e-6))
            for _ in range(self.num_layers)
        ])
    
    def forward(self, patch_embeddings, encoder_point_embeddings, original_seq_len=None):
        """
        Args:
            patch_embeddings: [batch_size, num_patches, hidden_size] - Input patch embeddings from main model
            encoder_point_embeddings: [batch_size, seq_len, hidden_size] - Point embeddings from encoder
            original_seq_len: Optional original sequence length for reshaping
        
        Returns:
            point_embeddings: [batch_size, seq_len, hidden_size] - Refined point embeddings
        """
        batch_size, num_patches, hidden_size = patch_embeddings.shape
        
        # Use encoder point embeddings as starting point
        point_embeddings = encoder_point_embeddings
        
        for layer_idx in range(self.num_layers):
            # Cross-attention: points attend to patches (like BLT's LocalDecoder)
            residual = point_embeddings
            point_embeddings_norm = self.cross_attn_norms[layer_idx](point_embeddings)
            
            # Create reverse cross-attention mask
            seq_len = point_embeddings.shape[1]
            patch_size = seq_len // num_patches
            
            # Create mask where each point can attend to its corresponding patch
            cross_mask = self._create_reverse_cross_mask(
                num_patches=num_patches,
                patch_size=patch_size,
                device=point_embeddings.device
            )
            
            cross_attn_output = self.cross_attention_layers[layer_idx](
                x=point_embeddings_norm,  # queries (points)
                kv=patch_embeddings,      # keys/values (patches)
                mask=cross_mask
            )
            point_embeddings = residual + cross_attn_output
        
        return point_embeddings
    
    def _create_reverse_cross_mask(self, num_patches, patch_size, device):
        """Create mask for reverse cross-attention where points attend to patches."""
        seq_len = num_patches * patch_size
        
        # Create mask: [seq_len, num_patches]
        mask = torch.full((seq_len, num_patches), float('-inf'), device=device)
        
        for patch_idx in range(num_patches):
            start_idx = patch_idx * patch_size
            end_idx = (patch_idx + 1) * patch_size
            # Points in this patch can attend to this patch
            mask[start_idx:end_idx, patch_idx] = 0.0
        
        return mask.unsqueeze(0)  # Add batch dimension


def compute_patch_attention_masks(
    batch_patches: Optional[List[List[Tuple[int, int]]]] = None,
    batch_size: int = None,
    seq_length: int = None,
    patch_size: int = None,
    device: torch.device = None,
    dtype: torch.dtype = None
) -> torch.Tensor:
    """
    Create attention masks for cross-attention between patch embeddings and point embeddings.
    
    This function creates masks that ensure each patch embedding only attends to point embeddings
    within its corresponding patch boundaries. Supports both variable-length patches (adaptive)
    and fixed-size patches.
    
    Args:
        batch_patches: Optional list of patch boundaries for each sequence in the batch
                      Each sequence contains patches as (start_idx, end_idx) tuples
                      If None, uses fixed patch size
        batch_size: Batch size for fixed patching mode
        seq_length: Sequence length for fixed patching mode  
        patch_size: Fixed patch size for uniform patching mode
        device: Device to create tensors on
    
    Returns:
        torch.Tensor: Attention mask of shape [batch_size, max_num_patches, padded_seq_length]
                     where 0.0 means attend and -inf means mask out
    
    Example usage:
        ```python
        # For adaptive patches
        patch_masks = compute_patch_attention_masks(
            batch_patches=batch_patches,
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype
        )
        
        # For fixed patches
        patch_masks = compute_patch_attention_masks(
            batch_patches=None,
            batch_size=batch_size,
            seq_length=seq_length,
            patch_size=patch_size,
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype
        )
        ```
    """
    
    if batch_patches is not None:
        # Variable-length patches mode (adaptive)
        max_num_patches = max(len(patches) for patches in batch_patches) if batch_patches else 1
        max_seq_length = max(
            max(end for start, end in patches) if patches else 0 
            for patches in batch_patches
        )
        batch_size = len(batch_patches)
        
        # Initialize attention mask with large negative values instead of -inf for better gradients
        attention_mask = torch.full(
            (batch_size, max_num_patches, max_seq_length),
            -1e4,  # Use -1e4 instead of -inf for better gradient flow
            device=device,
            dtype=dtype or torch.float32
        )
        
        # Fill in 0.0 for valid patch-point correspondences
        for batch_idx, patches in enumerate(batch_patches):
            for patch_idx, (start_idx, end_idx) in enumerate(patches):
                # Each patch can only attend to points within its boundaries
                attention_mask[batch_idx, patch_idx, start_idx:end_idx] = 0.0
                
    else:
        # Fixed-size patches mode
        if batch_size is None or seq_length is None or patch_size is None:
            raise ValueError("For fixed patching, batch_size, seq_length, and patch_size must be provided")
        
        num_patches = seq_length // patch_size
        if num_patches == 0:
            num_patches = 1
        
        # Truncate sequence to multiple of patch_size
        truncated_seq_len = num_patches * patch_size
        
        # Initialize attention mask with large negative values instead of -inf for better gradients
        attention_mask = torch.full(
            (batch_size, num_patches, truncated_seq_len),
            -1e4,  # Use -1e4 instead of -inf for better gradient flow
            device=device,
            dtype=dtype or torch.float32
        )
        
        # Fill in 0.0 for fixed patch boundaries
        for patch_idx in range(num_patches):
            start_idx = patch_idx * patch_size
            end_idx = (patch_idx + 1) * patch_size
            # Each patch attends only to its own points
            attention_mask[:, patch_idx, start_idx:end_idx] = 0.0
    
    return attention_mask


def apply_gradient_clipping(model, max_norm=1.0):
    """
    Apply gradient clipping to prevent gradient explosion.
    
    Args:
        model: The model to apply gradient clipping to
        max_norm: Maximum norm for gradients
    
    Returns:
        Total norm of gradients before clipping
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def scale_layer_weights_init(layer, scale_factor=0.1):
    """
    Initialize layer weights with a smaller scale to prevent exploding gradients.
    
    Args:
        layer: Neural network layer to initialize
        scale_factor: Scale factor for weight initialization
    """
    if hasattr(layer, 'weight') and layer.weight is not None:
        with torch.no_grad():
            layer.weight.data *= scale_factor
    if hasattr(layer, 'bias') and layer.bias is not None:
        with torch.no_grad():
            layer.bias.data.zero_()


def stable_layer_norm(x, normalized_shape, eps=1e-6):
    """
    More stable layer normalization that prevents numerical issues.
    
    Args:
        x: Input tensor
        normalized_shape: Shape over which to normalize
        eps: Small constant for numerical stability
    
    Returns:
        Normalized tensor
    """
    # Clamp input to prevent extreme values
    x = torch.clamp(x, min=-10.0, max=10.0)
    return F.layer_norm(x, normalized_shape, eps=eps)


def stable_softmax_attention(attn_weights, dim=-1, temperature=1.0):
    """
    More stable softmax for attention computation.
    
    Args:
        attn_weights: Attention weight logits
        dim: Dimension to apply softmax over
        temperature: Temperature scaling factor
    
    Returns:
        Attention probabilities
    """
    # Store original dtype to preserve it
    original_dtype = attn_weights.dtype
    
    # Apply temperature scaling
    attn_weights = attn_weights / temperature
    
    # Clip extreme values to prevent overflow/underflow
    attn_weights = torch.clamp(attn_weights, min=-15.0, max=15.0)
    
    # Subtract max for numerical stability
    max_vals = torch.max(attn_weights, dim=dim, keepdim=True)[0]
    attn_weights = attn_weights - max_vals
    
    # Apply softmax and preserve original dtype
    return F.softmax(attn_weights, dim=dim, dtype=original_dtype)
