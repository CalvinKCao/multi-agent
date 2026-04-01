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
from typing import Any, Dict, List, Optional, Tuple, Union

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from drc_sokoban.probing.train_probes import _build_probe


def _build_binary_probe(seed: int = 0) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000, random_state=seed, C=1.0,
            class_weight="balanced", solver="lbfgs",
        )),
    ])


def _split_train_val(
    X: np.ndarray,
    y_labels: np.ndarray,
    episode_ids: Optional[np.ndarray],
    stratify,
    random_state: int = 42,
    test_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if episode_ids is None or len(np.unique(episode_ids)) < 2:
        return train_test_split(
            X, y_labels, test_size=test_size, random_state=random_state,
            stratify=stratify,
        )
    gss = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state,
    )
    tr_idx, vl_idx = next(gss.split(X, y_labels, groups=episode_ids))
    return X[tr_idx], X[vl_idx], y_labels[tr_idx], y_labels[vl_idx]


def _subsample_preserving_groups(
    X: np.ndarray,
    y_labels: np.ndarray,
    episode_ids: Optional[np.ndarray],
    max_samples: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if len(X) <= max_samples:
        return X, y_labels, episode_ids
    rng = np.random.default_rng(seed)
    if episode_ids is not None:
        uniq = np.unique(episode_ids)
        rng.shuffle(uniq)
        picked: List[int] = []
        n = 0
        for e in uniq:
            picked.append(int(e))
            n += int(np.sum(episode_ids == e))
            if n >= max_samples:
                break
        mask = np.isin(episode_ids, np.array(picked, dtype=episode_ids.dtype))
        return X[mask], y_labels[mask], episode_ids[mask]
    idx = rng.choice(len(X), max_samples, replace=False)
    ep_out = episode_ids[idx] if episode_ids is not None else None
    return X[idx], y_labels[idx], ep_out


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
    episode_ids: Optional[np.ndarray] = None,
) -> Tuple[Pipeline, float]:
    """
    Train a 1x1 spatial probe for one ToM concept at one (x, y, layer, tick).

    Args:
        hidden_states: (N, num_ticks, num_layers, 32, 8, 8)  -- agent A's hidden state
        labels:        (N, 8, 8) int concept labels
        episode_ids:   optional (N,) episode index per timestep; train/val split
                       is by episode (GroupShuffleSplit) to reduce temporal leakage.

    Returns:
        (probe, val_f1)
    """
    X        = hidden_states[:, tick, layer, :, y, x]   # (N, 32)
    y_labels = labels[:, y, x]

    X, y_labels, episode_ids = _subsample_preserving_groups(
        X, y_labels, episode_ids, max_samples,
    )

    unique = np.unique(y_labels)
    if len(unique) < 2:
        return _build_binary_probe() if binary else _build_probe(), 0.0

    counts  = np.bincount(y_labels, minlength=int(y_labels.max()) + 1)
    stratify = y_labels if np.all(counts[counts > 0] >= 2) else None

    X_tr, X_vl, y_tr, y_vl = _split_train_val(
        X, y_labels, episode_ids, stratify=stratify,
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
    episode_ids: Optional[np.ndarray] = None,
) -> float:
    """Raw-observation baseline for one concept at (x, y)."""
    X        = observations[:, :, y, x]   # (N, 10)
    y_labels = labels[:, y, x]

    X, y_labels, episode_ids = _subsample_preserving_groups(
        X, y_labels, episode_ids, max_samples,
    )

    unique = np.unique(y_labels)
    if len(unique) < 2:
        return 0.0

    counts   = np.bincount(y_labels, minlength=int(y_labels.max()) + 1)
    stratify = y_labels if np.all(counts[counts > 0] >= 2) else None

    X_tr, X_vl, y_tr, y_vl = _split_train_val(
        X, y_labels, episode_ids, stratify=stratify,
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


def evaluate_tom_probe_f1(
    probe: Pipeline,
    hidden_states: np.ndarray,
    labels: np.ndarray,
    x: int,
    y: int,
    layer: int,
    tick: int,
    binary: bool = False,
) -> float:
    """Macro-F1 of an already-fitted probe on a (possibly new) dataset."""
    X = hidden_states[:, tick, layer, :, y, x]
    y_labels = labels[:, y, x]
    if len(np.unique(y_labels)) < 2:
        return 0.0
    try:
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(probe[-1])
    except Exception:
        return 0.0
    preds = probe.predict(X)
    return float(f1_score(y_labels, preds, average="macro", zero_division=0))


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
    episode_ids: Optional[np.ndarray] = None,
    return_probes: bool = False,
) -> Union[Dict, Tuple[Dict, Dict[str, Dict[Tuple[int, int], Dict[Tuple[int, int], Pipeline]]]]]:
    """
    Train probes for all three ToM concepts at all (layer, tick, position).

    If ``return_probes`` is True, also returns fitted sklearn Pipelines keyed by
    concept -> (layer, tick) -> (x, y) -> probe.
    """
    if positions is None:
        positions = [(x, y) for y in range(grid_size) for x in range(grid_size)]

    results: Dict[str, Any] = {
        "TA": {}, "TB": {}, "TC": {},
        "TA_per_pos": {}, "TB_per_pos": {}, "TC_per_pos": {},
        "TA_baseline": 0.0, "TB_baseline": 0.0, "TC_baseline": 0.0,
    }

    fitted: Dict[str, Dict[Tuple[int, int], Dict[Tuple[int, int], Pipeline]]] = {
        "TA": {}, "TB": {}, "TC": {},
    }

    # Observation baselines
    ta_bl, tb_bl, tc_bl = [], [], []
    for x, y in positions:
        ta_bl.append(train_obs_baseline_tom(
            observations, ta_labels, x, y, binary=False, episode_ids=episode_ids,
        ))
        tb_bl.append(train_obs_baseline_tom(
            observations, tb_labels, x, y, binary=False, episode_ids=episode_ids,
        ))
        tc_bl.append(train_obs_baseline_tom(
            observations, tc_labels, x, y, binary=True, episode_ids=episode_ids,
        ))
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
            ta_p, tb_p, tc_p = {}, {}, {}
            for x, y in positions:
                p_ta, f1_ta = train_tom_probe(
                    hidden_states, ta_labels, x, y, layer, tick,
                    binary=False, episode_ids=episode_ids,
                )
                p_tb, f1_tb = train_tom_probe(
                    hidden_states, tb_labels, x, y, layer, tick,
                    binary=False, episode_ids=episode_ids,
                )
                p_tc, f1_tc = train_tom_probe(
                    hidden_states, tc_labels, x, y, layer, tick,
                    binary=True, episode_ids=episode_ids,
                )
                ta_f1s[(x, y)] = f1_ta
                tb_f1s[(x, y)] = f1_tb
                tc_f1s[(x, y)] = f1_tc
                ta_p[(x, y)] = p_ta
                tb_p[(x, y)] = p_tb
                tc_p[(x, y)] = p_tc

            key = (layer, tick)
            results["TA"][key] = float(np.mean(list(ta_f1s.values())))
            results["TB"][key] = float(np.mean(list(tb_f1s.values())))
            results["TC"][key] = float(np.mean(list(tc_f1s.values())))
            results["TA_per_pos"][key] = ta_f1s
            results["TB_per_pos"][key] = tb_f1s
            results["TC_per_pos"][key] = tc_f1s
            if return_probes:
                fitted["TA"][key] = ta_p
                fitted["TB"][key] = tb_p
                fitted["TC"][key] = tc_p

            if verbose:
                print(f"  L{layer} T{tick}: "
                      f"TA={results['TA'][key]:.3f}  "
                      f"TB={results['TB'][key]:.3f}  "
                      f"TC={results['TC'][key]:.3f}")

    if return_probes:
        return results, fitted
    return results


def evaluate_fitted_tom_probes(
    fitted: Dict[str, Dict[Tuple[int, int], Dict[Tuple[int, int], Pipeline]]],
    hidden_states: np.ndarray,
    ta_labels: np.ndarray,
    tb_labels: np.ndarray,
    tc_labels: np.ndarray,
    num_ticks: int,
    num_layers: int,
    positions: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """
    Macro-F1 on ``hidden_states`` using probes fitted on another dataset (e.g. v1 → v2).
    Returns the same mean / per_pos structure as train_all_tom_probes (no baselines).
    """
    if positions is None:
        positions = [(x, y) for y in range(8) for x in range(8)]

    out: Dict[str, Any] = {"TA": {}, "TB": {}, "TC": {},
                           "TA_per_pos": {}, "TB_per_pos": {}, "TC_per_pos": {}}
    spec = [
        ("TA", ta_labels, False),
        ("TB", tb_labels, False),
        ("TC", tc_labels, True),
    ]
    for layer in range(num_layers):
        for tick in range(num_ticks):
            key = (layer, tick)
            for concept, lab, binary in spec:
                f1s = {}
                for x, y in positions:
                    probe = fitted[concept][key][(x, y)]
                    f1s[(x, y)] = evaluate_tom_probe_f1(
                        probe, hidden_states, lab, x, y, layer, tick, binary=binary,
                    )
                out[concept][key] = float(np.mean(list(f1s.values())))
                out[f"{concept}_per_pos"][key] = f1s
    return out


def prepare_tom_dataset(
    episodes: list,
    ta_list: list,
    tb_list: list,
    tc_list: list,
    return_episode_ids: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Flatten episodes into arrays for probe training.

    Episodes must contain:
        hidden_states_a: (T, num_ticks, num_layers, 32, 8, 8) -- agent A's states
        observations_a:  (T, 10, 8, 8)

    Returns:
        hs_a, obs_a, ta, tb, tc  each of shape (N_total, ...)
        If return_episode_ids: also episode_ids (N,) with episode index per row.
    """
    hs_l, obs_l, ta_l, tb_l, tc_l, ep_l = [], [], [], [], [], []
    for ep_i, (ep, ta, tb, tc) in enumerate(zip(episodes, ta_list, tb_list, tc_list)):
        T = ep["hidden_states_a"].shape[0]
        hs_l.append(ep["hidden_states_a"])
        obs_l.append(ep["observations_a"])
        ta_l.append(ta)
        tb_l.append(tb)
        tc_l.append(tc)
        ep_l.append(np.full(T, ep_i, dtype=np.int64))
    hs = np.concatenate(hs_l, axis=0)
    obs = np.concatenate(obs_l, axis=0)
    ta_a = np.concatenate(ta_l, axis=0)
    tb_a = np.concatenate(tb_l, axis=0)
    tc_a = np.concatenate(tc_l, axis=0)
    ep_ids = np.concatenate(ep_l, axis=0)
    if return_episode_ids:
        return hs, obs, ta_a, tb_a, tc_a, ep_ids
    return hs, obs, ta_a, tb_a, tc_a
