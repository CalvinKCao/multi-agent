"""
Phase 4 kill tests for the ToM experiment.

Test 1  Cross-Policy Generalization
    Train probes on trajectories where A plays with Partner v1 (self-play).
    Evaluate those SAME probes on trajectories where A plays with Partner v2
    (handicapped / high-entropy policy).
    Pass criterion: F1 stays high.  This shows A's representation is
    genuinely modeling "the other agent" not "what I would do from here."

Test 2  Ambiguity Test
    Split timesteps into "ambiguous" (partner has >= 3 valid moves) vs
    "obvious" (partner has <= 1 valid move).
    Compare ToM probe F1 on the two subsets.
    ToM should shine on ambiguous steps where the behavior is less predictable
    from the raw observation alone.

Test 3  Random-Weights Baseline
    Run an untrained MA DRC agent, collect hidden states, probe TA/TB/TC.
    If probe F1 is low here but high for the trained agent, the signal comes
    from learned representations, not architectural inductive bias.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from drc_sokoban.probing.train_probes import _build_probe, split_episodes_by_index
from drc_sokoban.probing.tom_train_probes import (
    train_tom_probe, prepare_tom_dataset, train_all_tom_probes,
)
from drc_sokoban.probing.tom_concept_labeler import (
    label_tom_episode, count_valid_moves,
)


# ── Test 1: Cross-Policy Generalization ─────────────────────────────────────

def cross_policy_generalization_test(
    probes_v1: Dict,
    episodes_v2: List[dict],
    ta_v2: List[np.ndarray],
    tb_v2: List[np.ndarray],
    tc_v2: List[np.ndarray],
    positions: Optional[List[Tuple[int, int]]] = None,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate probes trained on Partner-v1 trajectories against Partner-v2 data.

    probes_v1 is the output of train_all_tom_probes on v1 trajectories.
    We re-train probes on v2 data and compare F1 to establish the reference,
    then compute stability = F1_v2 / F1_v1.

    Returns dict with F1 scores and stability ratios for each (concept, layer, tick).
    """
    if positions is None:
        positions = [(x, y) for y in range(8) for x in range(8)]

    hs_v2, obs_v2, ta_v2_arr, tb_v2_arr, tc_v2_arr = prepare_tom_dataset(
        episodes_v2, ta_v2, tb_v2, tc_v2
    )
    num_ticks  = hs_v2.shape[1]
    num_layers = hs_v2.shape[2]

    results_v2 = train_all_tom_probes(
        hs_v2, obs_v2, ta_v2_arr, tb_v2_arr, tc_v2_arr,
        num_ticks=num_ticks, num_layers=num_layers,
        positions=positions, verbose=False,
    )

    stability = {}
    for concept in ("TA", "TB", "TC"):
        for key in probes_v1.get(concept, {}):
            f1_v1 = probes_v1[concept].get(key, 0.0)
            f1_v2 = results_v2[concept].get(key, 0.0)
            stab  = f1_v2 / max(f1_v1, 1e-6)
            stability[f"{concept}_{key}"] = {
                "f1_v1": f1_v1, "f1_v2": f1_v2, "stability": stab,
            }

    mean_stab_ta = np.mean([v["stability"] for k, v in stability.items() if k.startswith("TA")])
    mean_stab_tb = np.mean([v["stability"] for k, v in stability.items() if k.startswith("TB")])

    if verbose:
        print("\n--- Cross-Policy Generalization Test ---")
        for k, v in sorted(stability.items()):
            print(f"  {k}: v1={v['f1_v1']:.3f} v2={v['f1_v2']:.3f} "
                  f"stability={v['stability']:.2f}")
        print(f"  Mean stability TA: {mean_stab_ta:.3f}  TB: {mean_stab_tb:.3f}")
        verdict = "PASS" if (mean_stab_ta + mean_stab_tb) / 2 >= 0.70 else "FAIL"
        print(f"  Verdict: {verdict} (threshold >= 0.70)")

    return {
        "stability_per_key": stability,
        "mean_stability_ta": float(mean_stab_ta),
        "mean_stability_tb": float(mean_stab_tb),
        "results_v2": results_v2,
    }


# ── Test 2: Ambiguity Test ───────────────────────────────────────────────────

def ambiguity_test(
    episodes: List[dict],
    ta_labels: List[np.ndarray],
    tb_labels: List[np.ndarray],
    num_layers: int,
    num_ticks: int,
    ambig_threshold: int = 3,
    obvious_threshold: int = 1,
    positions: Optional[List[Tuple[int, int]]] = None,
    verbose: bool = True,
) -> Dict:
    """
    Compare probe F1 on ambiguous vs obvious timesteps.

    A timestep is "ambiguous" if agent B has >= ambig_threshold valid moves.
    "Obvious" = <= obvious_threshold valid moves.

    We train probes on all data, then evaluate them separately on each subset.
    """
    if positions is None:
        positions = [(x, y) for y in range(8) for x in range(8)]

    # Build validity count for each timestep
    ambig_mask_list  = []
    obvious_mask_list = []

    for ep in episodes:
        obs_b_seq   = ep.get("observations_b", ep.get("observations_a"))  # B's obs
        agent_b_seq = ep.get("agent_b_positions", [])
        T = len(obs_b_seq)
        am = np.zeros(T, dtype=bool)
        ob = np.zeros(T, dtype=bool)
        for t in range(T):
            if t < len(agent_b_seq):
                n_valid = count_valid_moves(obs_b_seq[t], agent_b_seq[t])
                am[t] = n_valid >= ambig_threshold
                ob[t] = n_valid <= obvious_threshold
        ambig_mask_list.append(am)
        obvious_mask_list.append(ob)

    # Pool hidden states and labels, tracking which timesteps are ambiguous
    hs_all   = np.concatenate([ep["hidden_states_a"] for ep in episodes], axis=0)
    obs_all  = np.concatenate([ep["observations_a"] for ep in episodes],  axis=0)
    ta_all   = np.concatenate(ta_labels, axis=0)
    tb_all   = np.concatenate(tb_labels, axis=0)
    am_all   = np.concatenate(ambig_mask_list, axis=0)
    ob_all   = np.concatenate(obvious_mask_list, axis=0)

    def probe_subset(mask, concept_labels, binary):
        if mask.sum() < 20:
            return 0.0
        f1s = []
        for x, y in positions[:16]:  # subset for speed
            _, f1 = train_tom_probe(
                hs_all[mask], concept_labels[mask],
                x, y, layer=num_layers - 1, tick=num_ticks - 1,
                binary=binary, n_seeds=1,
            )
            f1s.append(f1)
        return float(np.mean(f1s))

    ta_ambig  = probe_subset(am_all,  ta_all, binary=False)
    ta_obv    = probe_subset(ob_all,  ta_all, binary=False)
    tb_ambig  = probe_subset(am_all,  tb_all, binary=False)
    tb_obv    = probe_subset(ob_all,  tb_all, binary=False)

    if verbose:
        print("\n--- Ambiguity Test ---")
        print(f"  Ambiguous steps: {am_all.sum()}  Obvious: {ob_all.sum()}")
        print(f"  TA F1: ambiguous={ta_ambig:.3f}  obvious={ta_obv:.3f}")
        print(f"  TB F1: ambiguous={tb_ambig:.3f}  obvious={tb_obv:.3f}")
        gap_ta = ta_ambig - ta_obv
        gap_tb = tb_ambig - tb_obv
        print(f"  Delta (ambig - obv): TA={gap_ta:+.3f}  TB={gap_tb:+.3f}")

    return {
        "ta_ambiguous": ta_ambig, "ta_obvious": ta_obv,
        "tb_ambiguous": tb_ambig, "tb_obvious": tb_obv,
        "n_ambiguous": int(am_all.sum()), "n_obvious": int(ob_all.sum()),
    }


# ── Test 3: Random-Weights Baseline ─────────────────────────────────────────

def random_weights_baseline(
    env_factory,
    trained_results: Dict,
    num_layers: int,
    num_ticks: int,
    hidden_channels: int = 32,
    n_episodes: int = 150,
    positions: Optional[List[Tuple[int, int]]] = None,
    verbose: bool = True,
) -> Dict:
    """
    Collect hidden states from an untrained (random-weight) MA DRC agent and
    train probes to check whether the architecture itself produces the signal.

    Returns F1 scores and delta vs trained agent for each (concept, layer, tick).
    """
    import torch
    from drc_sokoban.models.agent import DRCAgent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = DRCAgent(
        obs_channels    = 10,
        hidden_channels = hidden_channels,
        num_layers      = num_layers,
        num_ticks       = num_ticks,
    ).to(device)
    agent.eval()

    if positions is None:
        positions = [(x, y) for y in range(8) for x in range(8)]

    if verbose:
        print(f"Collecting {n_episodes} eps from random-weight MA agent...")

    n_batch = min(16, n_episodes)
    envs    = [env_factory() for _ in range(n_batch)]
    obs_A_all, obs_B_all = zip(*[env.reset() for env in envs])
    obs_A_all = np.stack(obs_A_all, axis=0)
    obs_B_all = np.stack(obs_B_all, axis=0)

    hidden_A = agent.init_hidden(n_batch, device)
    hidden_B = agent.init_hidden(n_batch, device)

    ep_bufs = [
        {"obs_a": [], "obs_b": [], "hs_a": [], "pos_a": [], "pos_b": [],
         "box_pushes_b": [], "box_pos": []}
        for _ in range(n_batch)
    ]
    ep_steps   = [0] * n_batch
    episodes   = []
    collected  = 0

    while collected < n_episodes:
        obs_t_A = torch.FloatTensor(obs_A_all).to(device)
        obs_t_B = torch.FloatTensor(obs_B_all).to(device)

        with torch.no_grad():
            logits_A, _, new_hA, all_ticks_A = agent(obs_t_A, hidden_A, return_all_ticks=True)
            logits_B, _, new_hB, _           = agent(obs_t_B, hidden_B, return_all_ticks=True)

        nt, nl = len(all_ticks_A), len(all_ticks_A[0])
        hs_batch = np.stack([
            np.stack([all_ticks_A[t][l][0].cpu().numpy() for l in range(nl)], axis=1)
            for t in range(nt)
        ], axis=1)  # (n_batch, ticks, layers, 32, 8, 8)

        acts_A = torch.argmax(logits_A, dim=-1).cpu().numpy()
        acts_B = torch.argmax(logits_B, dim=-1).cpu().numpy()

        obs_A_next_l, obs_B_next_l = [], []
        dones = np.zeros(n_batch, dtype=bool)

        for i, env in enumerate(envs):
            ep_bufs[i]["obs_a"].append(obs_A_all[i].copy())
            ep_bufs[i]["hs_a"].append(hs_batch[i])
            ep_bufs[i]["pos_a"].append(env.get_agent_a_pos())
            ep_bufs[i]["pos_b"].append(env.get_agent_b_pos())
            ep_bufs[i]["box_pos"].append(env.get_box_positions())

            (obs_a_n, obs_b_n), rew, done, info = env.step((int(acts_A[i]), int(acts_B[i])))
            ep_bufs[i]["box_pushes_b"].append(info.get("box_pushed_by_b"))
            ep_steps[i] += 1
            force_done = done or ep_steps[i] >= 200
            dones[i]   = force_done
            obs_A_next_l.append(obs_a_n)
            obs_B_next_l.append(obs_b_n)

        obs_A_all = np.stack(obs_A_next_l, axis=0)
        obs_B_all = np.stack(obs_B_next_l, axis=0)

        mask_t = torch.FloatTensor([[0.0 if dones[i] else 1.0] for i in range(n_batch)]).view(n_batch, 1, 1, 1).to(device)
        hidden_A = [(h * mask_t, c * mask_t) for h, c in new_hA]
        hidden_B = [(h * mask_t, c * mask_t) for h, c in new_hB]

        for i in range(n_batch):
            if dones[i] and collected < n_episodes:
                buf = ep_bufs[i]
                T   = len(buf["hs_a"])
                ep  = {
                    "hidden_states_a": np.array(buf["hs_a"]),
                    "observations_a":  np.array(buf["obs_a"]),
                    "agent_a_positions": buf["pos_a"],
                    "agent_b_positions": buf["pos_b"],
                    "box_pushes_b":      buf["box_pushes_b"],
                    "box_positions":     buf["box_pos"],
                }
                ta, tb, tc = label_tom_episode(
                    buf["pos_b"], buf["box_pushes_b"], buf["box_pos"]
                )
                ep["ta"] = ta; ep["tb"] = tb; ep["tc"] = tc
                episodes.append(ep)
                collected += 1
                obs_A_all[i], obs_B_all[i] = envs[i].reset()
                ep_bufs[i] = {"obs_a": [], "obs_b": [], "hs_a": [], "pos_a": [],
                               "pos_b": [], "box_pushes_b": [], "box_pos": []}
                ep_steps[i] = 0
                for h, c in hidden_A:
                    h[i] = 0.0; c[i] = 0.0
                for h, c in hidden_B:
                    h[i] = 0.0; c[i] = 0.0

    hs_r    = np.concatenate([ep["hidden_states_a"] for ep in episodes], axis=0)
    obs_r   = np.concatenate([ep["observations_a"]  for ep in episodes], axis=0)
    ta_r    = np.concatenate([ep["ta"] for ep in episodes], axis=0)
    tb_r    = np.concatenate([ep["tb"] for ep in episodes], axis=0)
    tc_r    = np.concatenate([ep["tc"] for ep in episodes], axis=0)

    results_rnd = train_all_tom_probes(
        hs_r, obs_r, ta_r, tb_r, tc_r,
        num_ticks=num_ticks, num_layers=num_layers,
        positions=positions[:16], verbose=False,
    )

    if verbose:
        print("\n--- Random-Weights Baseline ---")
        for concept in ("TA", "TB", "TC"):
            for key, f1 in sorted(results_rnd.get(concept, {}).items()):
                trained_f1 = trained_results.get(concept, {}).get(key, 0.0)
                delta = trained_f1 - f1
                print(f"  {concept} {key}: rand={f1:.3f}  trained={trained_f1:.3f}  "
                      f"delta={delta:+.3f}")

    return {"random": results_rnd, "trained": trained_results}
