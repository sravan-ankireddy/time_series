#!/usr/bin/env python3
import argparse
import random
import torch
import torch.multiprocessing as mp
import numpy as np
from chronos import BaseChronosPipeline
from tqdm import tqdm
import os
import sys
import matplotlib.pyplot as plt

# If your chronos package is in a non‑standard location, adjust this:
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'chronos'))


def compute_sequence_entropies(seq_tensor, pipeline, max_context_length, batch_size):
    """
    Given a single sequence (NumPy array or 1D torch.Tensor),
    compute per-step autoregressive entropy H(x_t | x_<t>) for t=0..L-1.
    Returns a NumPy array of shape (L,).
    """
    # Convert to NumPy array if needed
    if isinstance(seq_tensor, torch.Tensor):
        seq = seq_tensor.cpu().numpy()
    else:
        seq = seq_tensor
    L = len(seq)

    # Build contexts on CPU
    contexts = []
    for t in range(L):
        if t == 0:
            contexts.append(torch.tensor([float(seq.mean())], dtype=torch.float32))
        else:
            start = max(0, t - max_context_length)
            contexts.append(torch.tensor(seq[start:t], dtype=torch.float32))

    # Predict in batches and compute entropies
    entropies = []
    for i in range(0, L, batch_size):
        batch_ctx = contexts[i : i + batch_size]
        with torch.no_grad():
            # Chronos will handle moving inputs to the correct device
            _, logits = pipeline.predict(
                context=batch_ctx,
                prediction_length=1,
                num_samples=1,
                return_logits=True,
            )
        # logits shape = [B,1,1,vocab_size]
        for b in range(len(batch_ctx)):
            p = torch.softmax(logits[b, 0, 0, :], dim=-1)
            ent = float((-p * torch.log2(p + 1e-12)).sum().item())
            entropies.append(ent)

    return np.array(entropies, dtype=np.float32)


def plot_sequence_entropy(input_ids, ent, seq_idx, plots_dir):
    """
    Plot a single sequence's signal vs entropy overlay in 3 vertical panels:
    full, first half, first quarter. Saves one PNG per sequence.
    """
    L = len(ent)
    zooms = [("full", L), ("half", L // 2), ("quarter", L // 4)]
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    for ax, (name, end) in zip(axes, zooms):
        ax_sig = ax
        ax_ent = ax_sig.twinx()
        ax_sig.plot(input_ids[:end], label="Signal")
        ax_ent.plot(ent[:end], color="red", label="Entropy")
        ax_sig.set_ylabel("Signal")
        ax_ent.set_ylabel("Entropy", color="red")
        ax_sig.set_title(f"Sequence {seq_idx} – {name} ({end} pts)")
        h1, l1 = ax_sig.get_legend_handles_labels()
        h2, l2 = ax_ent.get_legend_handles_labels()
        ax_sig.legend(h1 + h2, l1 + l2, loc="upper right")
        ax_sig.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, f"seq_{seq_idx}_signal_entropy.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main_worker(rank, world_size, args):
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    print(f"[{rank}] → using device {device}")

    # 1) Load the original dataset
    original = torch.load(args.dataset_pt, weights_only=False)
    total_N = len(original)
    N = args.max_samples if (args.max_samples and args.max_samples < total_N) else total_N

    # assign every world_size‑th index to this worker
    indices = list(range(rank, N, world_size))
    print(f"[{rank}] → processing {len(indices)}/{N} sequences")

    # 2) Load the Chronos pipeline onto this GPU
    pipeline = BaseChronosPipeline.from_pretrained(
        args.model,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )

    # 3) Compute entropies for our shard
    partial = []
    for idx in tqdm(indices, desc=f"worker {rank}", unit="seq"):
        sample = original[idx]
        input_ids = sample["input_ids"]
        labels     = sample["labels"]
        loss_masks = sample["loss_masks"]

        ent_array = compute_sequence_entropies(
            seq_tensor=input_ids,
            pipeline=pipeline,
            max_context_length=args.max_context_length,
            batch_size=args.batch_size,
        )

        partial.append((
            idx,
            {
                "input_ids":  input_ids,
                "labels":     labels,
                "loss_masks": loss_masks,
                "entropy":    ent_array
            }
        ))

    # 4) Save our shard
    part_path = f"{args.output_dataset}.part{rank}"
    torch.save(partial, part_path)
    print(f"[{rank}] → saved {len(partial)} entries to {part_path}")


def merge_and_plot(args, world_size):
    # 1) Load & concat all shards
    all_parts = []
    for r in range(world_size):
        part = torch.load(f"{args.output_dataset}.part{r}")
        all_parts.extend(part)

    # 2) Restore original order and strip indices
    all_parts.sort(key=lambda x: x[0])
    final = [entry for _, entry in all_parts]

    # 3) Save merged dataset
    torch.save(final, args.output_dataset)
    print(f"→ merged dataset saved to {args.output_dataset}")

    # 4) Clean up part files
    for r in range(world_size):
        os.remove(f"{args.output_dataset}.part{r}")

    # 5) Plot 3 random sequences from the full set
    os.makedirs(args.plots_dir, exist_ok=True)
    for seq_idx in random.sample(range(len(final)), k=min(3, len(final))):
        entry = final[seq_idx]
        ids = entry["input_ids"].cpu().numpy() if isinstance(entry["input_ids"], torch.Tensor) else entry["input_ids"]
        plot_sequence_entropy(ids, entry["entropy"], seq_idx, args.plots_dir)
        print(f"  → plotted sequence {seq_idx}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-step entropies in parallel across all GPUs"
    )
    parser.add_argument(
        "--dataset_pt",
        default="/home/sa53869/datasets/time-moe-300b.pt",
        help="Path to the pickled TimeMoE dataset (.pt)",
    )
    parser.add_argument(
        "--model",
        default="../model_weights/chronos/chronos-t5-small",
        help="Chronos model ID or local path",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size when predicting multiple contexts at once",
    )
    parser.add_argument(
        "--max_context_length",
        type=int,
        default=1024,
        help="Maximum context length for autoregressive prediction",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of sequences to process (default: all)",
    )
    parser.add_argument(
        "--output_dataset",
        default="/home/sa53869/datasets/time-300b-with-entropy.pt",
        help="Where to save the new .pt dataset",
    )
    parser.add_argument(
        "--plots_dir",
        default="plots",
        help="Directory to save the sequence+entropy overlay plots",
    )
    args = parser.parse_args()

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        world_size = torch.cuda.device_count()
        mp.set_start_method("spawn", force=True)
        mp.spawn(
            main_worker,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
        merge_and_plot(args, world_size)
    else:
        # Single‑GPU or CPU fallback
        main_worker(0, 1, args)
        merge_and_plot(args, 1)


if __name__ == "__main__":
    main()
