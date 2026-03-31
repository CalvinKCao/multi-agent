"""
Full probe pipeline: load collected episodes → label → train probes → evaluate.

Usage:
    # Full probe run
    python scripts/run_probes.py \\
        --data data/probe_data/train.pkl \\
        --results-dir results/

    # Quick check (5 positions, 500 episodes)
    python scripts/run_probes.py \\
        --data data/probe_data/smoke_test.pkl \\
        --n-positions 5 \\
        --quick-check

    # Plot results from saved file
    python scripts/run_probes.py \\
        --load-results results/probe_results.pkl \\
        --plot-only
"""

import argparse
import os
import sys
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from drc_sokoban.probing.concept_labeler import label_episodes_parallel, label_episode_fast
from drc_sokoban.probing.train_probes import (
    prepare_probe_dataset,
    split_episodes_by_index,
    train_all_probes,
)
from drc_sokoban.probing.evaluate_probes import (
    print_results_table,
    save_results,
    load_results,
    plot_f1_heatmap,
    plot_spatial_f1,
    check_probe_sanity,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train and evaluate Bush et al. probes")
    p.add_argument("--data",          type=str, default=None,
                   help="Path to collected probe data .pkl file")
    p.add_argument("--results-dir",   type=str, default="results")
    p.add_argument("--n-positions",   type=int, default=None,
                   help="Subset of positions to probe (None = all 64)")
    p.add_argument("--n-episodes",    type=int, default=None,
                   help="Limit number of episodes loaded (None = all)")
    p.add_argument("--n-workers",     type=int, default=4,
                   help="Parallel workers for episode labeling")
    p.add_argument("--quick-check",   action="store_true",
                   help="Quick sanity check: subset of positions and episodes")
    p.add_argument("--load-results",  type=str, default=None,
                   help="Load existing results file instead of recomputing")
    p.add_argument("--plot-only",     action="store_true",
                   help="Only generate plots from loaded results")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    # ── Load or recompute results ──────────────────────────────────────────────
    if args.load_results or args.plot_only:
        results_path = args.load_results or os.path.join(args.results_dir, "probe_results.pkl")
        print(f"Loading results from {results_path}")
        results = load_results(results_path)
    else:
        results = _run_full_pipeline(args)

    # ── Display ────────────────────────────────────────────────────────────────
    print("\n=== Probe Results ===")
    print_results_table(results)
    check_probe_sanity(results)

    # ── Plots ─────────────────────────────────────────────────────────────────
    for concept in ["CA", "CB"]:
        plot_f1_heatmap(
            results, concept=concept,
            save_path=os.path.join(args.results_dir, f"{concept}_heatmap.png"),
        )
        plot_spatial_f1(
            results, concept=concept,
            save_path=os.path.join(args.results_dir, f"{concept}_spatial_f1.png"),
        )


def _run_full_pipeline(args):
    if not args.data:
        raise ValueError("--data is required unless using --load-results")

    # ── Load episodes ──────────────────────────────────────────────────────────
    print(f"Loading episodes from {args.data}...")
    with open(args.data, "rb") as f:
        episodes = pickle.load(f)
    print(f"  Loaded {len(episodes)} episodes")

    if args.n_episodes:
        episodes = episodes[:args.n_episodes]
        print(f"  Using first {len(episodes)} episodes")

    if args.quick_check:
        episodes = episodes[:min(500, len(episodes))]
        print(f"  Quick-check mode: using {len(episodes)} episodes")

    # ── Split by episode index ─────────────────────────────────────────────────
    train_eps, val_eps, test_eps = split_episodes_by_index(episodes)
    print(f"  Split: {len(train_eps)} train / {len(val_eps)} val / {len(test_eps)} test")

    # ── Label episodes (CA and CB concepts) ───────────────────────────────────
    print("Labeling episodes (CA and CB concepts)...")
    n_workers = min(args.n_workers, len(train_eps))

    if n_workers > 1:
        ca_list, cb_list = label_episodes_parallel(
            train_eps + val_eps, n_workers=n_workers
        )
        split_at = len(train_eps)
        ca_train, ca_val = ca_list[:split_at], ca_list[split_at:]
        cb_train, cb_val = cb_list[:split_at], cb_list[split_at:]
    else:
        ca_train = [label_episode_fast(ep["agent_positions"], ep["box_positions"])[0]
                    for ep in train_eps]
        cb_train = [label_episode_fast(ep["agent_positions"], ep["box_positions"])[1]
                    for ep in train_eps]
        ca_val   = [label_episode_fast(ep["agent_positions"], ep["box_positions"])[0]
                    for ep in val_eps]
        cb_val   = [label_episode_fast(ep["agent_positions"], ep["box_positions"])[1]
                    for ep in val_eps]

    print("  Labeling complete.")

    # ── Prepare flat arrays ────────────────────────────────────────────────────
    hs_train, obs_train, ca_arr_train, cb_arr_train = prepare_probe_dataset(
        train_eps, ca_train, cb_train
    )
    print(f"  Train data shape: hs={hs_train.shape}, obs={obs_train.shape}")

    # ── Determine grid size and architecture params ────────────────────────────
    num_ticks  = hs_train.shape[1]
    num_layers = hs_train.shape[2]

    # ── Choose probe positions ─────────────────────────────────────────────────
    if args.n_positions:
        all_positions = [(x, y) for y in range(8) for x in range(8)]
        import random
        random.seed(42)
        positions = random.sample(all_positions, min(args.n_positions, 64))
    elif args.quick_check:
        positions = [(x, y) for y in range(4) for x in range(4)]  # 16 positions
    else:
        positions = None  # all 64

    # ── Train probes ───────────────────────────────────────────────────────────
    print(f"\nTraining probes (ticks={num_ticks}, layers={num_layers}, "
          f"positions={len(positions) if positions else 64})...")
    results = train_all_probes(
        hidden_states = hs_train,
        observations  = obs_train,
        ca_labels     = ca_arr_train,
        cb_labels     = cb_arr_train,
        num_ticks     = num_ticks,
        num_layers    = num_layers,
        positions     = positions,
        verbose       = True,
    )

    # ── Save results ───────────────────────────────────────────────────────────
    save_path = os.path.join(args.results_dir, "probe_results.pkl")
    save_results(results, save_path)

    return results


if __name__ == "__main__":
    main()
