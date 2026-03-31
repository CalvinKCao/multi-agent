"""
Spatial logistic-regression probe training following Bush et al. 2025.

Each probe is a 1×1 probe: it only sees the 32-dimensional activation at a
single spatial position (x, y) in the ConvLSTM hidden state.  This spatial
locality is the core methodological contribution of Bush et al. — it shows that
planning information is spatially localised within the hidden state.

Probe inputs:
    hidden_states[:, tick, layer, :, y, x]   shape: (N, 32)

Probe targets:
    labels[:, y, x]                           shape: (N,)  values in 0..4

Metric: macro F1 (not accuracy — class imbalance from NEVER class).
"""

import numpy as np
import pickle
from typing import Dict, List, Optional, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from drc_sokoban.probing.concept_labeler import NEVER, N_CLASSES


def _build_probe(seed: int = 0) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=seed,
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])


def train_spatial_probe(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    x: int,
    y: int,
    layer: int,
    tick: int,
    n_seeds: int = 3,
    val_split: float = 0.2,
    max_samples: int = 8000,
) -> Tuple[Pipeline, float]:
    """
    Train a 1×1 spatial probe at position (x, y) for one (layer, tick).

    Args:
        hidden_states: (N, num_ticks, num_layers, 32, 8, 8)
        labels:        (N, 8, 8) int concept labels
        x, y:          grid position to probe (x=col, y=row)
        layer:         ConvLSTM layer index
        tick:          tick index
        n_seeds:       number of restarts; best on val F1 is kept
        val_split:     fraction of data held out for model selection
        max_samples:   subsample to this many timesteps for speed

    Returns:
        best_probe: fitted sklearn Pipeline
        best_val_f1: macro F1 on validation split
    """
    # Extract 32-dim features at position (x, y)
    X = hidden_states[:, tick, layer, :, y, x]   # (N, 32)
    y_labels = labels[:, y, x]                    # (N,)

    # Subsample for speed while keeping class balance
    if len(X) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X, y_labels = X[idx], y_labels[idx]

    # Filter positions with at most one class (uninformative)
    unique_classes = np.unique(y_labels)
    if len(unique_classes) < 2:
        return _build_probe(), 0.0

    # Use stratify only when every class has ≥2 samples
    counts = np.bincount(y_labels, minlength=int(y_labels.max()) + 1)
    stratify = y_labels if np.all(counts[counts > 0] >= 2) else None

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_labels, test_size=val_split, random_state=42, stratify=stratify
    )

    # Skip if training set has only one class (trivial / degenerate position)
    if len(np.unique(y_train)) < 2:
        return _build_probe(), 0.0

    best_f1 = -1.0
    best_probe = None

    for seed in range(n_seeds):
        probe = _build_probe(seed)
        probe.fit(X_train, y_train)
        preds = probe.predict(X_val)
        f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_probe = probe

    return best_probe, float(best_f1)


def train_observation_baseline(
    observations: np.ndarray,
    labels: np.ndarray,
    x: int,
    y: int,
    n_seeds: int = 3,
    max_samples: int = 8000,
) -> Tuple[Pipeline, float]:
    """
    Observation-only baseline probe: uses the 7-dim observation at (x, y).

    This is the lower-bound — if the hidden-state probe doesn't beat this,
    something is wrong.
    """
    # observations: (N, 7, 8, 8) → extract channel vector at (y, x)
    X = observations[:, :, y, x]   # (N, 7)
    y_labels = labels[:, y, x]

    if len(X) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X, y_labels = X[idx], y_labels[idx]

    unique_classes = np.unique(y_labels)
    if len(unique_classes) < 2:
        return _build_probe(), 0.0

    counts = np.bincount(y_labels, minlength=int(y_labels.max()) + 1)
    stratify = y_labels if np.all(counts[counts > 0] >= 2) else None

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_labels, test_size=0.2, random_state=42, stratify=stratify
    )

    if len(np.unique(y_train)) < 2:
        return _build_probe(), 0.0

    best_f1 = -1.0
    best_probe = None
    for seed in range(n_seeds):
        probe = _build_probe(seed)
        probe.fit(X_train, y_train)
        preds = probe.predict(X_val)
        f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_probe = probe

    return best_probe, float(best_f1)


def train_all_probes(
    hidden_states: np.ndarray,
    observations: np.ndarray,
    ca_labels: np.ndarray,
    cb_labels: np.ndarray,
    num_ticks: int = 3,
    num_layers: int = 3,
    grid_size: int = 8,
    positions: Optional[List[Tuple[int, int]]] = None,
    verbose: bool = True,
) -> Dict:
    """
    Train probes for all positions, all layers, all ticks.

    Args:
        hidden_states: (N, num_ticks, num_layers, 32, 8, 8)
        observations:  (N, 7, 8, 8)
        ca_labels:     (N, 8, 8) CA concept labels
        cb_labels:     (N, 8, 8) CB concept labels
        positions:     optional list of (x, y) to probe (subset for quick checks)
        verbose:       print progress

    Returns dict:
        {
          'CA':          {(layer, tick): mean_macro_f1},
          'CB':          {(layer, tick): mean_macro_f1},
          'CA_baseline': float,
          'CB_baseline': float,
          'CA_per_pos':  {(layer, tick): {(x,y): f1}},
          'CB_per_pos':  {(layer, tick): {(x,y): f1}},
        }
    """
    if positions is None:
        positions = [(x, y) for y in range(grid_size) for x in range(grid_size)]

    results = {
        "CA": {}, "CB": {},
        "CA_per_pos": {}, "CB_per_pos": {},
        "CA_baseline": 0.0, "CB_baseline": 0.0,
    }

    # ── Observation-only baseline ──────────────────────────────────────────────
    ca_bl, cb_bl = [], []
    for x, y in positions:
        _, f1_ca = train_observation_baseline(observations, ca_labels, x, y)
        _, f1_cb = train_observation_baseline(observations, cb_labels, x, y)
        ca_bl.append(f1_ca)
        cb_bl.append(f1_cb)
    results["CA_baseline"] = float(np.mean(ca_bl))
    results["CB_baseline"] = float(np.mean(cb_bl))
    if verbose:
        print(f"Observation baseline — CA: {results['CA_baseline']:.3f} | "
              f"CB: {results['CB_baseline']:.3f}")

    # ── Hidden-state probes ────────────────────────────────────────────────────
    for layer in range(num_layers):
        for tick in range(num_ticks):
            ca_f1s, cb_f1s = {}, {}
            for x, y in positions:
                _, f1_ca = train_spatial_probe(
                    hidden_states, ca_labels, x, y, layer, tick
                )
                _, f1_cb = train_spatial_probe(
                    hidden_states, cb_labels, x, y, layer, tick
                )
                ca_f1s[(x, y)] = f1_ca
                cb_f1s[(x, y)] = f1_cb

            key = (layer, tick)
            results["CA"][key] = float(np.mean(list(ca_f1s.values())))
            results["CB"][key] = float(np.mean(list(cb_f1s.values())))
            results["CA_per_pos"][key] = ca_f1s
            results["CB_per_pos"][key] = cb_f1s

            if verbose:
                print(f"Layer {layer}, Tick {tick}: "
                      f"CA F1={results['CA'][key]:.3f} | "
                      f"CB F1={results['CB'][key]:.3f}")

    return results


def prepare_probe_dataset(
    episodes: list,
    ca_labels_list: list,
    cb_labels_list: list,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Flatten episodes into arrays suitable for probe training.

    Episodes must contain 'hidden_states' (T, num_ticks, num_layers, 32, 8, 8)
    and 'observations' (T, 7, 8, 8) arrays.

    Returns:
        hidden_states: (N_total, num_ticks, num_layers, 32, 8, 8)
        observations:  (N_total, 7, 8, 8)
        ca_labels:     (N_total, 8, 8)
        cb_labels:     (N_total, 8, 8)
    """
    all_hs, all_obs, all_ca, all_cb = [], [], [], []
    for ep, ca, cb in zip(episodes, ca_labels_list, cb_labels_list):
        T = len(ep["hidden_states"])
        all_hs.append(ep["hidden_states"])       # (T, ticks, layers, 32, 8, 8)
        all_obs.append(ep["observations"])        # (T, 7, 8, 8)
        all_ca.append(ca)
        all_cb.append(cb)

    return (
        np.concatenate(all_hs,  axis=0),
        np.concatenate(all_obs, axis=0),
        np.concatenate(all_ca,  axis=0),
        np.concatenate(all_cb,  axis=0),
    )


def split_episodes_by_index(
    episodes: list, train_frac: float = 0.70, val_frac: float = 0.15
) -> Tuple[list, list, list]:
    """
    Split episodes into train / val / test by episode index (not timestep).
    Splitting by timestep would cause data leakage due to temporal correlation.
    """
    N = len(episodes)
    n_train = int(N * train_frac)
    n_val   = int(N * val_frac)
    return (
        episodes[:n_train],
        episodes[n_train : n_train + n_val],
        episodes[n_train + n_val :],
    )
