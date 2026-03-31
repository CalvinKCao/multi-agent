"""
Spatial probe training for ToM concepts (TA, TB, TC).

Reuses the 1x1 spatial probe machinery from train_probes.py but adapts
it for the three Theory of Mind concepts:
  TA / TB: 5-class logistic probe (same as CA / CB)
  TC:      binary logistic probe (2 classes)

The probe input is always agent A's hidden state at position (x, y) — we
are asking whether A's representation encodes information about B's future.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from drc_sokoban.probing.train_probes import _build_probe, split_episodes_by_index


def _build_binary_probe(seed: int = 0) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000, random_state=seed, C=1.0,
            class_weight="balanced", solver="lbfgs",
        )),
    ])


def train_tom_probe(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    x: int,
    y: int,
    layer: int,
    tick: int,
    binary: bool = False,
    n_seeds: int = 3,
    max_samples: int = 8000,
) -> Tuple[Pipeline, float]:
    """
    Train a 1x1 spatial probe for one ToM concept at one (x, y, layer, tick).

    Args:
        hidden_states: (N, num_ticks, num_layers, 32, 8, 8)  -- agent A's hidden state
        labels:        (N, 8, 8) int concept labels
        binary:        True for TC (2-class), False for TA/TB (5-class)

    Returns:
        (probe, val_f1)
    """
    X        = hidden_states[:, tick, layer, :, y, x]   # (N, 32)
    y_labels = labels[:, y, x]

    if len(X) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), max_samples, replace=False)
        X, y_labels = X[idx], y_labels[idx]

    unique = np.unique(y_labels)
    if len(unique) < 2:
        return _build_binary_probe() if binary else _build_probe(), 0.0

    counts  = np.bincount(y_labels, minlength=int(y_labels.max()) + 1)
    stratify = y_labels if np.all(counts[counts > 0] >= 2) else None

    X_tr, X_vl, y_tr, y_vl = train_test_split(
        X, y_labels, test_size=0.2, random_state=42, stratify=stratify
    )
    if len(np.unique(y_tr)) < 2:
        return _build_binary_probe() if binary else _build_probe(), 0.0

    build_fn  = _build_binary_probe if binary else _build_probe
    best_f1, best_probe = -1.0, None

    for seed in range(n_seeds):
        probe = build_fn(seed)
        probe.fit(X_tr, y_tr)
        preds = probe.predict(X_vl)
        f1 = f1_score(y_vl, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_probe = f1, probe

    return best_probe, float(best_f1)


def train_obs_baseline_tom(
    observations: np.ndarray,
    labels: np.ndarray,
    x: int,
    y: int,
    binary: bool = False,
    n_seeds: int = 2,
    max_samples: int = 8000,
) -> float:
    """Raw-observation baseline for one concept at (x, y)."""
    X        = observations[:, :, y, x]   # (N, 10)
    y_labels = labels[:, y, x]

    if len(X) > max_samples:
        idx = np.random.default_rng(42).choice(len(X), max_samples, replace=False)
        X, y_labels = X[idx], y_labels[idx]

    unique = np.unique(y_labels)
    if len(unique) < 2:
        return 0.0

    counts   = np.bincount(y_labels, minlength=int(y_labels.max()) + 1)
    stratify = y_labels if np.all(counts[counts > 0] >= 2) else None

    X_tr, X_vl, y_tr, y_vl = train_test_split(
        X, y_labels, test_size=0.2, random_state=42, stratify=stratify
    )
    if len(np.unique(y_tr)) < 2:
        return 0.0

    build_fn = _build_binary_probe if binary else _build_probe
    best_f1  = 0.0
    for seed in range(n_seeds):
        p = build_fn(seed)
        p.fit(X_tr, y_tr)
        f1 = f1_score(y_vl, p.predict(X_vl), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
    return best_f1


def train_all_tom_probes(
    hidden_states: np.ndarray,
    observations: np.ndarray,
    ta_labels: np.ndarray,
    tb_labels: np.ndarray,
    tc_labels: np.ndarray,
    num_ticks: int = 3,
    num_layers: int = 2,
    grid_size: int = 8,
    positions: Optional[List[Tuple[int, int]]] = None,
    verbose: bool = True,
) -> Dict:
    """
    Train probes for all three ToM concepts at all (layer, tick, position).

    Returns dict with keys:
        TA, TB, TC: {(layer, tick): mean_f1}
        TA_baseline, TB_baseline, TC_baseline: float
        TA_per_pos, TB_per_pos, TC_per_pos: {(layer, tick): {(x,y): f1}}
    """
    if positions is None:
        positions = [(x, y) for y in range(grid_size) for x in range(grid_size)]

    results = {
        "TA": {}, "TB": {}, "TC": {},
        "TA_per_pos": {}, "TB_per_pos": {}, "TC_per_pos": {},
        "TA_baseline": 0.0, "TB_baseline": 0.0, "TC_baseline": 0.0,
    }

    # Observation baselines
    ta_bl, tb_bl, tc_bl = [], [], []
    for x, y in positions:
        ta_bl.append(train_obs_baseline_tom(observations, ta_labels, x, y, binary=False))
        tb_bl.append(train_obs_baseline_tom(observations, tb_labels, x, y, binary=False))
        tc_bl.append(train_obs_baseline_tom(observations, tc_labels, x, y, binary=True))
    results["TA_baseline"] = float(np.mean(ta_bl))
    results["TB_baseline"] = float(np.mean(tb_bl))
    results["TC_baseline"] = float(np.mean(tc_bl))
    if verbose:
        print(f"  Obs baseline: TA={results['TA_baseline']:.3f}  "
              f"TB={results['TB_baseline']:.3f}  TC={results['TC_baseline']:.3f}")

    # Hidden-state probes
    for layer in range(num_layers):
        for tick in range(num_ticks):
            ta_f1s, tb_f1s, tc_f1s = {}, {}, {}
            for x, y in positions:
                _, f1_ta = train_tom_probe(hidden_states, ta_labels, x, y, layer, tick, binary=False)
                _, f1_tb = train_tom_probe(hidden_states, tb_labels, x, y, layer, tick, binary=False)
                _, f1_tc = train_tom_probe(hidden_states, tc_labels, x, y, layer, tick, binary=True)
                ta_f1s[(x, y)] = f1_ta
                tb_f1s[(x, y)] = f1_tb
                tc_f1s[(x, y)] = f1_tc

            key = (layer, tick)
            results["TA"][key] = float(np.mean(list(ta_f1s.values())))
            results["TB"][key] = float(np.mean(list(tb_f1s.values())))
            results["TC"][key] = float(np.mean(list(tc_f1s.values())))
            results["TA_per_pos"][key] = ta_f1s
            results["TB_per_pos"][key] = tb_f1s
            results["TC_per_pos"][key] = tc_f1s

            if verbose:
                print(f"  L{layer} T{tick}: "
                      f"TA={results['TA'][key]:.3f}  "
                      f"TB={results['TB'][key]:.3f}  "
                      f"TC={results['TC'][key]:.3f}")

    return results


def prepare_tom_dataset(
    episodes: list,
    ta_list: list,
    tb_list: list,
    tc_list: list,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Flatten episodes into arrays for probe training.

    Episodes must contain:
        hidden_states_a: (T, num_ticks, num_layers, 32, 8, 8) -- agent A's states
        observations_a:  (T, 10, 8, 8)

    Returns:
        hs_a, obs_a, ta, tb, tc  each of shape (N_total, ...)
    """
    hs_l, obs_l, ta_l, tb_l, tc_l = [], [], [], [], []
    for ep, ta, tb, tc in zip(episodes, ta_list, tb_list, tc_list):
        hs_l.append(ep["hidden_states_a"])
        obs_l.append(ep["observations_a"])
        ta_l.append(ta); tb_l.append(tb); tc_l.append(tc)
    return (
        np.concatenate(hs_l,  axis=0),
        np.concatenate(obs_l, axis=0),
        np.concatenate(ta_l,  axis=0),
        np.concatenate(tb_l,  axis=0),
        np.concatenate(tc_l,  axis=0),
    )
