import math
from typing import Optional, Tuple, List, Union, Literal
import warnings

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel, Cache, DynamicCache, StaticCache
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import MoeModelOutputWithPast, MoeCausalLMOutputWithPast
from transformers.utils import logging, is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10

from mamba_ssm import Mamba2
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

from dataclasses import dataclass
from einops import repeat, rearrange
import torch.nn.init as init

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

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_output
        return grad_x

def ste_func(x):
    return STE.apply(x)

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
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


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

class Downsampling(nn.Module):
    """
    Causal downsampling module using various pooling strategies over causal windows.
    """
    def __init__(
        self,
        sampling_factor: int,
        pooling_type: Literal["max", "min", "avg", "boundary"] = "boundary"
    ):
        super().__init__()
        self.sampling_factor = sampling_factor
        self.pooling_type = pooling_type

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_length, dim)
        Returns:
            Output tensor of shape (batch_size, output_length, dim)
            where output_length = input_length // sampling_factor
        """
        if self.sampling_factor == 1:
            return x

        batch_size, input_length, dim = x.shape
        output_length = input_length // self.sampling_factor

        if self.pooling_type == "boundary":
            # For boundary pooling, we want elements at positions:
            # sampling_factor-1, 2*sampling_factor-1, 3*sampling_factor-1, etc.
            indices = torch.arange(
                self.sampling_factor - 1, 
                output_length * self.sampling_factor, 
                self.sampling_factor,
                device=x.device
            )
            return x[:, indices, :]
        
        # For other pooling types
        output = []
        for i in range(output_length):
            end_pos = min((i + 1) * self.sampling_factor, input_length)
            window = x[:, :end_pos, :]
            
            if self.pooling_type == "max":
                pooled = window.max(dim=1)[0]
            elif self.pooling_type == "min":
                pooled = window.min(dim=1)[0]
            elif self.pooling_type == "avg":
                pooled = window.mean(dim=1)
            else:
                raise ValueError(f"Unknown pooling type: {self.pooling_type}")
            
            output.append(pooled)
        
        return torch.stack(output, dim=1)


class Upsampling(nn.Module):
    """
    Upsampling module that repeats embeddings to restore original length.
    """
    def __init__(
        self,
        sampling_factor: int
    ):
        super().__init__()
        self.sampling_factor = sampling_factor
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_length, dim)
        Returns:
            Output tensor of shape (batch_size, output_length, dim)
            where output_length = input_length * sampling_factor
        """
        # Repeat each embedding by the sampling factor
        upsampled = x.repeat_interleave(self.sampling_factor, dim=1)
        return upsampled

class LinearEncoder(nn.Module):
    """
    Simple linear encoder that reduces sequence length by grouping consecutive embeddings.
    
    The approach:
    1. Groups consecutive embeddings into patches of size `patch_size`
    2. Applies simple linear transformation to combine embeddings within each patch
    3. Reduces sequence length by factor of `patch_size` while preserving feature dimension
    
    For example: input [4, 1024, 384] with patch_size=4 -> output [4, 256, 384]
    Projects 4x384 -> 1x384 (combining 4 consecutive embeddings into 1)
    """

    def __init__(self, config: TimeMoeConfig, patch_size: int = 4):
        super().__init__()
        self.config = config
        self.patch_size = patch_size
        self.hidden_size = config.hidden_size  # Use hidden_size for embedding dimension
        
        # Simple linear projection to combine patch_size embeddings into one
        # Takes patch_size * hidden_size and outputs hidden_size (preserving feature dimension)
        self.patch_projection = nn.Linear(
            self.patch_size * self.hidden_size, 
            self.hidden_size,  # Keep same feature dimension
            bias=True
        )
        
        # Optional layer norm for stability
        self.layer_norm = TimeMoeRMSNorm(self.hidden_size, eps=getattr(config, 'rms_norm_eps', 1e-6))
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input embeddings of shape [batch_size, seq_len, hidden_size]
                             e.g., [4, 1024, 384]
        
        Returns:
            torch.Tensor: Reduced embeddings of shape [batch_size, seq_len//patch_size, hidden_size]
                         e.g., [4, 256, 384]
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Calculate number of patches
        num_patches = seq_len // self.patch_size
        
        # Truncate sequence to make it divisible by patch_size
        truncated_seq_len = num_patches * self.patch_size
        x_truncated = x[:, :truncated_seq_len, :]
        
        # Reshape to group consecutive embeddings into patches
        # [batch_size, num_patches, patch_size, hidden_size]
        # e.g., [4, 256, 4, 384]
        patches = x_truncated.view(batch_size, num_patches, self.patch_size, hidden_size)
        
        # Flatten each patch: [batch_size, num_patches, patch_size * hidden_size]
        # e.g., [4, 256, 1536] (4 * 384 = 1536)
        flattened_patches = patches.view(batch_size, num_patches, self.patch_size * hidden_size)
        
        # Apply simple linear transformation
        # [4, 256, 1536] -> [4, 256, 384]
        reduced_embeddings = self.patch_projection(flattened_patches)
        
        # Apply layer normalization
        reduced_embeddings = self.layer_norm(reduced_embeddings)
        
        return reduced_embeddings


class LinearDecoder(nn.Module):
    """
    Simple decoder module that expands patch embeddings back to point embeddings.
    
    This module reverses the patching operation by expanding each patch embedding
    into multiple point embeddings, enabling fine-grained temporal predictions.
    
    For example: input [4, 256, 384] with patch_size=4 -> output [4, 1024, 384]
    Projects 1x384 -> 4x384 (expanding 1 embedding into 4 consecutive embeddings)
    """
    
    def __init__(self, config: TimeMoeConfig, patch_size: int = 4):
        super().__init__()
        self.config = config
        self.patch_size = patch_size
        self.hidden_size = config.hidden_size  # Use hidden_size for embedding dimension
        
        # Project each patch embedding to patch_size point embeddings
        # Input: [hidden_size] -> Output: [patch_size * hidden_size]
        # e.g., [384] -> [1536] (4 * 384 = 1536)
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
                                            e.g., [4, 256, 384]
            original_seq_len (int, optional): Original sequence length before patching
                                            If None, uses num_patches * patch_size
        
        Returns:
            torch.Tensor: Point embeddings of shape [batch_size, seq_len, hidden_size]
                         e.g., [4, 1024, 384]
        """
        batch_size, num_patches, hidden_size = patch_embeddings.shape
        
        # Project patch embeddings to point embeddings
        # [4, 256, 384] -> [4, 256, 1536] (1536 = 4 * 384)
        unpatch_output = self.unpatch_projection(patch_embeddings)
        
        # Reshape to separate patch_size dimension
        # [4, 256, 1536] -> [4, 256, 4, 384]
        unpatch_reshaped = unpatch_output.view(
            batch_size, num_patches, self.patch_size, hidden_size
        )
        
        # Flatten to get point embeddings
        # [4, 256, 4, 384] -> [4, 1024, 384] (256 * 4 = 1024)
        point_embeddings = unpatch_reshaped.view(
            batch_size, num_patches * self.patch_size, hidden_size
        )
        
        # Truncate to original sequence length if specified
        if original_seq_len is not None and original_seq_len < point_embeddings.size(1):
            point_embeddings = point_embeddings[:, :original_seq_len, :]
        
        # Apply layer normalization
        point_embeddings = self.layer_norm(point_embeddings)
        
        return point_embeddings


class MambaEncoder(nn.Module):
    """
    Mamba2-based encoder for Time-MoE models.
    """
    def __init__(self, config: TimeMoeConfig):
        super().__init__()
        self.config = config

        # # Input embedding: project from input_dim to d_model
        # self.input_embedding = nn.Linear(config.input_size, config.hidden_size)

        # Build Mamba2 layers + RMS‐norms
        self.mamba_layers = nn.ModuleList()
        self.layer_norms  = nn.ModuleList()

        for _ in range(config.num_encoder_layers):
            self.mamba_layers.append(
                Mamba2(
                    d_model=config.hidden_size,
                    d_state=config.d_state,
                    d_conv=config.d_conv,
                    expand=config.expand,
                )
            )
            self.layer_norms.append(
                TimeMoeRMSNorm(config.hidden_size, eps=1e-6)
            )

        # Final layer norm
        self.final_layer_norm = TimeMoeRMSNorm(config.hidden_size, eps=1e-6)

        # Re‑init all Conv1d and Linear weights to a sane uniform range
        self._init_conv1d()

    def _init_conv1d(self):
        """
        FIX ME: Reinitialize only Conv1d layers inside Mamba layers with normal initialization
        """
        for mamba_layer in self.mamba_layers:
            for module in mamba_layer.modules():
                if isinstance(module, nn.Conv1d):
                    with torch.no_grad():
                        fan_in = module.weight.size(1) * module.weight.size(2)  # in_channels * kernel_size
                        std = (2.0 / fan_in) ** 0.5  # Kaiming std for ReLU/SiLU
                        new_weights = torch.randn_like(module.weight) * std
                        module.weight.copy_(new_weights)

                        # nn.init.normal_(module.weight, mean=0.0, std=0.02)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_length, input_dim)
        Returns:
            (batch_size, seq_length, d_model)
        """
        # # 1) Input embedding
        # x = self.input_embedding(x)  # → (B, L, d_model)

        # 2) Mamba2 stacks with residuals
        for mamba_layer, layer_norm in zip(self.mamba_layers, self.layer_norms):
            residual = x
            x = layer_norm(x)
            x = mamba_layer(x)
            x = x + residual

        # 3) Final normalization
        x = self.final_layer_norm(x)
        return x


@dataclass
class RoutingOutput:
    """Output from the routing module following H-Net's structure"""
    boundary_prob: torch.Tensor      # [batch_size, seq_len, 2] - probability distribution
    boundary_mask: torch.Tensor      # [batch_size, seq_len] - binary boundary indicators  
    selected_probs: torch.Tensor     # [batch_size, seq_len, 1] - selected probabilities

class RoutingModule(nn.Module):
    """
    Simplified routing module following H-Net's dynamic chunking approach.
    Creates boundary masks for downstream pooling operations.
    """

    def __init__(self, config: TimeMoeConfig, device=None, dtype=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # factory_kwargs = {"device": device, "dtype": dtype}
        
        # Query and key projections following H-Net design
        self.q_proj_layer = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj_layer = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Initialize as identity matrices for stable training
        with torch.no_grad():
            self.q_proj_layer.weight.copy_(torch.eye(self.hidden_size))
            self.k_proj_layer.weight.copy_(torch.eye(self.hidden_size))
            
        # Prevent reinitialization
        self.q_proj_layer.weight._no_reinit = True
        self.k_proj_layer.weight._no_reinit = True

    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5
    ) -> RoutingOutput:
        """
        Create boundary masks using H-Net's cosine similarity approach.
        
        Args:
            hidden_states: Input embeddings [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch_size, seq_len]
            threshold: Fixed threshold for boundary detection (default: 0.5)
            
        Returns:
            RoutingOutput with boundary information
        """        
        # Compute cosine similarity between adjacent representations
        cos_sim = torch.einsum(
            "b l d, b l d -> b l",
            F.normalize(self.q_proj_layer(hidden_states[:, :-1]), dim=-1),
            F.normalize(self.k_proj_layer(hidden_states[:, 1:]), dim=-1),
        )

        boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)

        # Force boundary probability of the first element to 1.0
        PAD_PROB = 1.0
        boundary_prob = F.pad(boundary_prob, (1, 0), "constant", PAD_PROB)

        boundary_prob = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)

        selected_idx = torch.argmax(boundary_prob, dim=-1)

        boundary_mask = selected_idx == 1  # (shape hidden_states.shape[:-1])
        
        selected_probs = boundary_prob.gather(
            dim=-1, index=selected_idx.unsqueeze(-1)
        )
        return RoutingOutput(
            boundary_prob=boundary_prob,
            boundary_mask=boundary_mask,
            selected_probs=selected_probs
        )

class ChunkLayer(nn.Module):

    def forward(self, hidden_states, boundary_mask):

        num_tokens = boundary_mask.sum(dim=-1)
        next_max_seqlen = int(num_tokens.max())

        device = hidden_states.device
        L = hidden_states.shape[1]
        token_idx = (
            torch.arange(L, device=device)[None, :] + (~boundary_mask).long() * L
        )
        seq_sorted_indices = torch.argsort(token_idx, dim=1)

        next_hidden_states = torch.gather(
            hidden_states,
            dim=1,
            index=seq_sorted_indices[:, :next_max_seqlen, None].expand(
                -1, -1, hidden_states.shape[-1]
            ),
        )

        # mask to indicate padded positions; to be used by the latent transformer
        embed_masks = (torch.arange(next_max_seqlen, device=device)[None, :] < num_tokens[:, None])

        return next_hidden_states, embed_masks

def compute_ratio_loss(boundary_mask, boundary_prob, N=4.0):
    """
    Compute the auxiliary ratio loss L_ratio for controlling compression ratios.
    This should be called immediately after the chunking layer.
    
    Args:
        boundary_mask (torch.Tensor): Binary boundary indicators [batch_size, seq_len]  
        boundary_prob (torch.Tensor): Boundary probabilities [batch_size, seq_len]
        N (float): Target compression ratio (e.g., 6.0 for 6:1 compression)
        
    Returns:
        torch.Tensor: Scalar ratio loss
    """
    L = boundary_mask.shape[1]  # sequence length
    
    # F = fraction of vectors actually selected (discrete)
    F = boundary_mask.float().sum(dim=-1) / L  # [batch_size]
    F = F.mean()  # average across batch
    
    # G = average boundary probability (continuous)
    G = boundary_prob.mean()
    
    # Compute ratio loss: L_ratio = N/(N-1) * ((N-1)*F*G + (1-F)*(1-G))
    L_ratio = (N / (N - 1)) * ((N - 1) * F * G + (1 - F) * (1 - G))
    
    return L_ratio

class DeChunkLayer(nn.Module):

    def __init__(self, config: TimeMoeConfig):
        super().__init__()
        self.config = config
        self.d_model = config.hidden_size

        # Just for Mamba2 kernel.
        self.dtype = torch.bfloat16 #config.mamba_dtype
        self.block_size = config.block_size
        self.headdim = config.headdim
        assert config.hidden_size % self.headdim == 0
        self.nheads = config.hidden_size // self.headdim

    def forward(
        self,
        hidden_states,
        boundary_mask,
        boundary_prob,
    ):

        p = torch.clamp(boundary_prob[..., -1].float(), min=1e-4, max=1 - (1e-4))

        B, L = boundary_mask.shape
        seq_idx = None

        token_idx = (
            torch.arange(L, device=hidden_states.device)[None, :]
            + (~boundary_mask).long() * L
        )
        seq_sorted_indices = torch.argsort(token_idx, dim=1)

        p = torch.gather(
            p, dim=1, index=seq_sorted_indices[:, : hidden_states.shape[1]]
        )  # (B, M)

        original_dtype = hidden_states.dtype
        # Reuse Mamba2 kernel for EMA Deaggregator.
        dt = torch.log(1 / (1 - p)).to(self.dtype)
        x = (hidden_states / dt[..., None]).to(self.dtype)
        A = -torch.ones(
            (self.nheads,), device=hidden_states.device, dtype=torch.float32
        )
        b = p.to(self.dtype)
        c = torch.ones_like(b)

        out = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            repeat(dt, "b l -> b l h", h=self.nheads),
            A,
            rearrange(b, "b l -> b l 1 1"),
            rearrange(c, "b l -> b l 1 1"),
            chunk_size=self.block_size,
            seq_idx=seq_idx,
        )
        out = rearrange(out, "b l h p -> b l (h p)")

        plug_back_idx = torch.cumsum(boundary_mask, dim=1) - 1  # (B, L)
        out = torch.gather(
            out,
            dim=1,
            index=plug_back_idx.unsqueeze(-1).expand(-1, -1, self.d_model),
        )

        return out.to(original_dtype)

class MambaDecoder(nn.Module):
    """
    Mamba2-based decoder for Time-MoE models.
    """
    def __init__(self, config: TimeMoeConfig):

        super().__init__()
        self.config = config
        
        # Multiple Mamba2 layers with layer normalization
        self.mamba_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(config.num_decoder_layers):
            self.mamba_layers.append(Mamba2(
                d_model=config.hidden_size,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
            ))
            self.layer_norms.append(TimeMoeRMSNorm(config.hidden_size, eps=1e-6))

        self.final_layer_norm = TimeMoeRMSNorm(config.hidden_size, eps=1e-6)
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.final_layer_norm = TimeMoeRMSNorm(config.hidden_size, eps=1e-6)

        # Re‑init all Conv1d and Linear weights to a sane uniform range
        self._init_conv1d()

    def _init_conv1d(self):
        """
        FIX ME: Reinitialize only Conv1d layers inside Mamba layers with normal initialization
        """
        for mamba_layer in self.mamba_layers:
            for module in mamba_layer.modules():
                if isinstance(module, nn.Conv1d):
                    with torch.no_grad():
                        fan_in = module.weight.size(1) * module.weight.size(2)  # in_channels * kernel_size
                        std = (2.0 / fan_in) ** 0.5  # Kaiming std for ReLU/SiLU
                        new_weights = torch.randn_like(module.weight) * std
                        module.weight.copy_(new_weights)

                        # nn.init.normal_(module.weight, mean=0.0, std=0.02)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Args:
            x: Decoder input tensor of shape (batch_size, seq_length, d_model)
        Returns:
            Output tensor of shape (batch_size, seq_length, output_dim)
        """
        # Process through multiple Mamba2 layers with residual connections
        for mamba_layer, layer_norm in zip(self.mamba_layers, self.layer_norms):
            residual = x
            x = layer_norm(x)
            x = mamba_layer(x)
            x = x + residual
        
        # Final layer norm and output projection
        x = self.final_layer_norm(x)
        output = self.output_projection(x)
        return output

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
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

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
        self.embed_layer = TimeMoeInputEmbedding(config)
        self.config = config

        if config.patch_embedding_type == "linear" and config.patch_size > 1:
            self.encoder_1 = LinearEncoder(config, config.patch_size)
            self.decoder_1 = LinearDecoder(config, config.patch_size)

            if config.num_stages > 1:
                self.encoder_2 = LinearEncoder(config, config.patch_size_2)
                self.decoder_2 = LinearDecoder(config, config.patch_size_2)

        elif config.patch_embedding_type == "mamba":
            self.encoder_1 = MambaEncoder(config)
            self.decoder_1 = MambaDecoder(config)

            if config.num_stages > 1:
                self.encoder_2 = MambaEncoder(config)
                self.decoder_2 = MambaDecoder(config)

            if config.patching_strategy == "fixed":
                self.downsample_1 = Downsampling(config.patch_size)
                self.upsample_1 = Upsampling(config.patch_size)
                if config.num_stages > 1:
                    self.downsample_2 = Downsampling(config.patch_size_2)
                    self.upsample_2 = Upsampling(config.patch_size_2)
            else:
                self.routing_module_1 = RoutingModule(config)
                if config.num_stages > 1:
                    self.routing_module_2 = RoutingModule(config)
                self.chunk_layer = ChunkLayer()
                self.dechunk_layer = DeChunkLayer(config)

        # do the residual in fp32
        self.residual_proj_1 = nn.Linear(config.hidden_size, config.hidden_size, dtype=torch.float32)
        nn.init.zeros_(self.residual_proj_1.weight)
        self.residual_proj_1.weight._no_reinit = True

        if config.num_stages > 1:
            self.residual_proj_2 = nn.Linear(config.hidden_size, config.hidden_size, dtype=torch.float32)
            nn.init.zeros_(self.residual_proj_2.weight)
            self.residual_proj_2.weight._no_reinit = True

        if config.patch_embedding_type == "mamba" and config.patching_strategy == "adaptive":
            self.residual_func = lambda out, residual, p: out * ste_func(p) + residual
        else:
            self.residual_func = lambda out, residual: out + residual

        # main network
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

        loss_ratio_total = None
        embed_masks_1, embed_masks_2 = None, None
        residual_1, residual_2 = None, None
        if inputs_embeds is None:
            # project inputs to the embedding space
            point_inputs_embeds = self.embed_layer(input_ids)

            # pass the inputs through the encoder
            if self.config.patch_embedding_type == "linear" and self.config.patch_size > 1:
                encoder_embeddings_1 = self.encoder_1(point_inputs_embeds)
                encoder_embeddings_for_residual_1 = point_inputs_embeds.to(dtype=self.residual_proj_1.weight.dtype)
                residual_1 = self.residual_proj_1(encoder_embeddings_for_residual_1)

                if self.config.num_stages > 1:
                    encoder_embeddings_2 = self.encoder_2(encoder_embeddings_1)
                    encoder_embeddings_for_residual_2 = encoder_embeddings_1.to(dtype=self.residual_proj_2.weight.dtype)
                    residual_2 = self.residual_proj_2(encoder_embeddings_for_residual_2)

                    inputs_embeds = encoder_embeddings_2
                else:
                    inputs_embeds = encoder_embeddings_1
            else:
                if self.config.patching_strategy == "fixed":
                    encoder_embeddings_1 = self.encoder_1(point_inputs_embeds)

                    encoder_embeddings_for_residual_1 = encoder_embeddings_1.to(dtype=self.residual_proj_1.weight.dtype)
                    residual_1 = self.residual_proj_1(encoder_embeddings_for_residual_1)

                    inputs_embeds = self.downsample_1(encoder_embeddings_1)

                    if self.config.num_stages > 1:
                        encoder_embeddings_2 = self.encoder_2(inputs_embeds)

                        encoder_embeddings_for_residual_2 = encoder_embeddings_2.to(dtype=self.residual_proj_2.weight.dtype)
                        residual_2 = self.residual_proj_2(encoder_embeddings_for_residual_2)

                        inputs_embeds = self.downsample_2(encoder_embeddings_2)

                elif self.config.patching_strategy == "adaptive":
                    encoder_embeddings_1 = self.encoder_1(point_inputs_embeds)
                    routing_output_1 = self.routing_module_1(encoder_embeddings_1)

                    encoder_embeddings_for_residual_1 = encoder_embeddings_1.to(dtype=self.residual_proj_1.weight.dtype)
                    residual_1 = self.residual_proj_1(encoder_embeddings_for_residual_1)

                    inputs_embeds, embed_masks_1 = self.chunk_layer(encoder_embeddings_1, routing_output_1.boundary_mask)

                    if self.config.num_stages > 1:
                        encoder_embeddings_2 = self.encoder_2(inputs_embeds)

                        encoder_embeddings_for_residual_2 = encoder_embeddings_2.to(dtype=self.residual_proj_2.weight.dtype)
                        residual_2 = self.residual_proj_2(encoder_embeddings_for_residual_2)

                        routing_output_2 = self.routing_module_2(encoder_embeddings_2)
                        inputs_embeds, embed_masks_2 = self.chunk_layer(encoder_embeddings_2, routing_output_2.boundary_mask)

                    # compute auxiliary loss for controlling compression ratio
                    loss_ratio_total = compute_ratio_loss(routing_output_1.boundary_mask, routing_output_1.boundary_prob[:,:,-1], 4)
                    if self.config.num_stages > 1:
                        loss_ratio_2 = compute_ratio_loss(routing_output_2.boundary_mask, routing_output_2.boundary_prob[:,:,-1], 4)
                        loss_ratio_total += loss_ratio_2

                    # from .extra_utils import plot_signal_and_mask
                    # plot_signal_and_mask(input_ids, routing_output.boundary_mask)
                    # breakpoint()
                # encoder_embeddings_for_residual_1 = encoder_embeddings_1.to(dtype=self.residual_proj_1.weight.dtype)
                # residual_1 = self.residual_proj_1(encoder_embeddings_for_residual_1)

                # encoder_embeddings_for_residual_2 = encoder_embeddings_2.to(dtype=self.residual_proj_2.weight.dtype)
                # residual_2 = self.residual_proj_2(encoder_embeddings_for_residual_2)

        batch_size, seq_length, _ = inputs_embeds.shape
    
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            # position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            position_ids = position_ids.view(-1, seq_length)
        else:
            position_ids = position_ids[:, :seq_length]
            position_ids = position_ids.view(-1, seq_length).long()

        # truncate the attention mask to match the input sequence length
        if attention_mask is not None:
            attention_mask = attention_mask[:seq_length, :seq_length]

        if embed_masks_1 is not None:
            attention_mask = embed_masks_1.to(attention_mask.dtype) if attention_mask is not None else embed_masks_1

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

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # project back to the input space
        if self.config.patch_embedding_type == "linear":
            if self.config.num_stages > 1:
                hidden_states = self.decoder_2(hidden_states)
                hidden_states = self.residual_func(hidden_states.to(dtype=residual_2.dtype), residual_2).to(hidden_states.dtype)
 
            hidden_states = self.decoder_1(hidden_states)
            hidden_states = self.residual_func(hidden_states.to(dtype=residual_1.dtype), residual_1).to(hidden_states.dtype)

        else:
            if self.config.patching_strategy == "fixed":
                if self.config.num_stages > 1:
                    hidden_states = self.upsample_2(hidden_states)
                    hidden_states = self.residual_func(hidden_states.to(dtype=residual_2.dtype), residual_2).to(hidden_states.dtype)
                    hidden_states = self.decoder_2(hidden_states)

                hidden_states = self.upsample_1(hidden_states)
                hidden_states = self.residual_func(hidden_states.to(dtype=residual_1.dtype), residual_1).to(hidden_states.dtype)
                hidden_states = self.decoder_1(hidden_states)
            else:
                if self.config.num_stages > 1:
                    hidden_states = self.dechunk_layer(hidden_states, routing_output_2.boundary_mask, routing_output_2.boundary_prob)
                    hidden_states = self.residual_func(hidden_states.to(dtype=residual_2.dtype), residual_2, routing_output_2.selected_probs).to(hidden_states.dtype)
                    hidden_states = self.decoder_2(hidden_states)

                hidden_states = self.dechunk_layer(hidden_states, routing_output_1.boundary_mask, routing_output_1.boundary_prob)
                hidden_states = self.residual_func(hidden_states.to(dtype=residual_1.dtype), residual_1, routing_output_1.selected_probs).to(hidden_states.dtype)
                hidden_states = self.decoder_1(hidden_states)
    
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
        ), loss_ratio_total


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
        outputs, loss_ratio = self.model(
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

        ## add loss_ratio
        if loss is not None and loss_ratio is not None:
            loss = loss + 0.01 * loss_ratio
    
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
