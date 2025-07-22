import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

def plot_signal_and_mask(input_ids, boundary_mask, save_dir="routing_examples"):
    """
    Plots the signal (input_ids) and boundary mask for 3 random samples in the batch.
    Each plot has 3 subplots: full signal, first half, first quarter.
    Patch boundaries are indicated, and average patch size is shown in subplot titles.
    """
    os.makedirs(save_dir, exist_ok=True)
    batch_size, seq_len = boundary_mask.shape
    # Convert to float32 for plotting
    input_ids = input_ids.float().cpu().numpy()
    boundary_mask = boundary_mask.float().cpu().numpy()

    # Remove last dim if present
    if input_ids.ndim == 3 and input_ids.shape[-1] == 1:
        input_ids = input_ids[..., 0]

    indices = random.sample(range(batch_size), min(3, batch_size))

    for i, idx in enumerate(indices, 1):
        signal = input_ids[idx]
        mask = boundary_mask[idx].astype(bool)
        patch_indices = np.where(mask)[0]
        # Ensure patch starts at 0 and ends at seq_len
        patch_boundaries = np.concatenate(([0], patch_indices, [seq_len]))
        patch_sizes = np.diff(patch_boundaries)
        avg_patch_size = patch_sizes.mean() if len(patch_sizes) > 0 else 0

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=False)
        zooms = [1.0, 0.5, 0.25]
        for j, zoom in enumerate(zooms):
            end = int(seq_len * zoom)
            ax = axes[j]
            ax.plot(np.arange(end), signal[:end], label="Signal", color="tab:blue")
            ax2 = ax.twinx()
            ax2.step(np.arange(end), mask[:end], label="Boundary Mask", color="tab:red", alpha=0.5, where='post')
            # Mark patch boundaries and annotate patch sizes
            shown_boundaries = patch_boundaries[patch_boundaries < end]
            for k, p in enumerate(shown_boundaries):
                if p < end:
                    ax.axvline(p, color="green", linestyle="--", alpha=0.3)
                    if k < len(shown_boundaries) - 1:
                        patch_size = shown_boundaries[k+1] - p
                        ax.annotate(f"{patch_size}", xy=(p + patch_size/2, np.max(signal[:end])), 
                                    xytext=(0, 5), textcoords='offset points', ha='center', fontsize=8, color="green")
            ax.set_xlim(0, end)  # Each subplot zooms to its region
            ax.set_ylabel("Signal")
            ax2.set_ylabel("Mask")
            ax.set_title(f"Sample {i} | Zoom: {int(zoom*100)}% | Avg Patch Size: {avg_patch_size:.1f}")
            ax.legend(loc="upper left")
            ax2.legend(loc="upper right")
        axes[-1].set_xlabel("Time Step")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"sample_{i}.png"))
        plt.close(fig)