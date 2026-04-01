"""
Full Theory of Mind probing pipeline for the multi-agent DRC experiment.

Phases:
    1. Collect trajectories from trained IPPO agent (v1 and v2 partners)
    2. Label ToM concepts: TA, TB, TC
    3. Train spatial probes on Agent A's hidden state
    4. Run kill tests: cross-policy, ambiguity, random-weights
    5. Save all results + heatmaps for report generation

Usage:
    python -m drc_sokoban.scripts.run_tom_experiment \\
        --checkpoint checkpoints/ma_selfplay_final.pt \\
        --partner-v2-ckpt checkpoints/ma_handicap_final.pt \\
        --data-dir data/boxoban_levels \\
        --results-dir results/tom/

    # Quick mode (fewer episodes)
    python -m drc_sokoban.scripts.run_tom_experiment \\
        --checkpoint checkpoints/ma_selfplay_final.pt \\
        --data-dir data/boxoban_levels \\
        --results-dir results/tom_quick/ \\
        --quick
"""

import argparse
import os
import sys
import json
import pickle
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from drc_sokoban.envs.ma_boxoban_env import MABoxobanEnv
from drc_sokoban.models.agent import DRCAgent
from drc_sokoban.probing.tom_concept_labeler import label_tom_episode
from drc_sokoban.probing.tom_train_probes import (
    train_all_tom_probes, prepare_tom_dataset,
)
from drc_sokoban.probing.tom_kill_tests import (
    cross_policy_generalization_test,
    ambiguity_test,
    random_weights_baseline,
)
from drc_sokoban.wandb_env import load_wandb_local_env

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",       type=str, required=True)
    p.add_argument("--partner-v2-ckpt",  type=str, default=None,
                   help="Handicapped partner checkpoint for cross-policy test")
    p.add_argument("--data-dir",         type=str, default=None)
    p.add_argument("--results-dir",      type=str, default="results/tom/")
    p.add_argument("--device",           type=str, default="auto")
    p.add_argument("--seed",             type=int, default=0)
    p.add_argument("--quick",            action="store_true")
    p.add_argument("--n-train-eps",      type=int, default=None)
    p.add_argument("--n-val-eps",        type=int, default=None)
    p.add_argument("--n-positions",      type=int, default=None)
    p.add_argument("--skip-kill",        action="store_true")
    p.add_argument("--wandb-project",    type=str, default="drc-sokoban-ma-tom")
    return p.parse_args()


def load_agent(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt.get("cfg", {})
    agent = DRCAgent(
        obs_channels    = cfg.get("obs_channels",    10),
        hidden_channels = cfg.get("hidden_channels", 32),
        num_layers      = cfg.get("num_layers",       2),
        num_ticks       = cfg.get("num_ticks",        3),
    ).to(device)
    agent.load_state_dict(ckpt["model_state"])
    agent.eval()
    step = ckpt.get("global_step", 0)
    print(f"Loaded {ckpt_path} (step {step:,})")
    return agent, cfg, step


def collect_ma_episodes(
    agent_a,
    agent_b,
    env_factory,
    n_episodes: int,
    device,
    greedy: bool = True,
    max_steps: int = 200,
) -> list:
    """
    Run MA agent pair and collect episode data for ToM probing.

    Stores per episode:
        observations_a:    (T, 10, 8, 8)
        observations_b:    (T, 10, 8, 8)
        hidden_states_a:   (T, num_ticks, num_layers, 32, 8, 8)
        actions_a, actions_b: (T,)
        agent_a_positions, agent_b_positions: list of (x,y) per step
        box_pushes_b: list of None or push dict {from_xy, to_xy, onto_target} per step
        box_positions: list of [(x,y),...] per step
        rewards: (T,)
        solved: bool
    """
    from tqdm import tqdm

    n_batch   = min(16, n_episodes)
    episodes  = []
    collected = 0
    solve_n   = 0

    envs  = [env_factory() for _ in range(n_batch)]
    pairs = [e.reset() for e in envs]
    ob_A  = np.stack([p[0] for p in pairs], axis=0)
    ob_B  = np.stack([p[1] for p in pairs], axis=0)

    hid_A = agent_a.init_hidden(n_batch, device)
    hid_B = agent_b.init_hidden(n_batch, device)

    bufs = [
        {"obs_a": [], "obs_b": [], "hs_a": [], "act_a": [], "act_b": [],
         "pos_a": [], "pos_b": [], "box_b": [], "box_pos": [], "rews": []}
        for _ in range(n_batch)
    ]
    ep_steps = [0] * n_batch

    with tqdm(total=n_episodes, desc="Collecting MA eps") as pbar:
        while collected < n_episodes:
            obs_t_A = torch.FloatTensor(ob_A).to(device)
            obs_t_B = torch.FloatTensor(ob_B).to(device)

            with torch.no_grad():
                logits_A, _, new_hA, all_ticks = agent_a(obs_t_A, hid_A, return_all_ticks=True)
                logits_B, _, new_hB            = agent_b(obs_t_B, hid_B)

            nt = len(all_ticks); nl = len(all_ticks[0])
            hs_batch = np.stack([
                np.stack([all_ticks[t][l][0].cpu().numpy() for l in range(nl)], axis=1)
                for t in range(nt)
            ], axis=1)  # (n_batch, ticks, layers, 32, 8, 8)

            if greedy:
                act_A = torch.argmax(logits_A, dim=-1).cpu().numpy()
                act_B = torch.argmax(logits_B, dim=-1).cpu().numpy()
            else:
                act_A = torch.distributions.Categorical(logits=logits_A).sample().cpu().numpy()
                act_B = torch.distributions.Categorical(logits=logits_B).sample().cpu().numpy()

            ob_A_next_l, ob_B_next_l = [], []
            dones = np.zeros(n_batch, dtype=bool)

            for i, env in enumerate(envs):
                bufs[i]["obs_a"].append(ob_A[i].copy())
                bufs[i]["obs_b"].append(ob_B[i].copy())
                bufs[i]["hs_a"].append(hs_batch[i])
                bufs[i]["act_a"].append(int(act_A[i]))
                bufs[i]["act_b"].append(int(act_B[i]))
                bufs[i]["pos_a"].append(env.get_agent_a_pos())
                bufs[i]["pos_b"].append(env.get_agent_b_pos())
                bufs[i]["box_pos"].append(env.get_box_positions())

                (ob_a_n, ob_b_n), rew, done, info = env.step((int(act_A[i]), int(act_B[i])))
                bufs[i]["box_b"].append(info.get("box_push_b"))
                bufs[i]["rews"].append(rew)
                ep_steps[i] += 1
                force_done = done or ep_steps[i] >= max_steps
                dones[i]   = force_done
                ob_A_next_l.append(ob_a_n); ob_B_next_l.append(ob_b_n)

            ob_A = np.stack(ob_A_next_l, axis=0)
            ob_B = np.stack(ob_B_next_l, axis=0)

            mask_t = torch.FloatTensor([[0.0 if dones[i] else 1.0] for i in range(n_batch)]).view(n_batch, 1, 1, 1).to(device)
            hid_A  = [(h * mask_t, c * mask_t) for h, c in new_hA]
            hid_B  = [(h * mask_t, c * mask_t) for h, c in new_hB]

            for i in range(n_batch):
                if dones[i] and collected < n_episodes:
                    buf = bufs[i]
                    ep  = {
                        "observations_a":  np.array(buf["obs_a"]),
                        "observations_b":  np.array(buf["obs_b"]),
                        "hidden_states_a": np.array(buf["hs_a"]),
                        "actions_a":       np.array(buf["act_a"]),
                        "actions_b":       np.array(buf["act_b"]),
                        "agent_a_positions": buf["pos_a"],
                        "agent_b_positions": buf["pos_b"],
                        "box_pushes_b":      buf["box_b"],
                        "box_positions":     buf["box_pos"],
                        "rewards":           np.array(buf["rews"]),
                        "solved":            bool(np.any(np.array(buf["rews"]) > 5.0)),
                    }
                    if ep["solved"]:
                        solve_n += 1
                    episodes.append(ep)
                    collected += 1
                    pbar.update(1)

                    pairs_i = envs[i].reset()
                    ob_A[i] = pairs_i[0]; ob_B[i] = pairs_i[1]
                    bufs[i] = {"obs_a": [], "obs_b": [], "hs_a": [], "act_a": [], "act_b": [],
                               "pos_a": [], "pos_b": [], "box_b": [], "box_pos": [], "rews": []}
                    ep_steps[i] = 0
                    for h, c in hid_A: h[i] = 0.0; c[i] = 0.0
                    for h, c in hid_B: h[i] = 0.0; c[i] = 0.0

    print(f"  Solve rate: {solve_n / max(collected, 1):.3f}")
    return episodes


def save_heatmap(per_pos_dict, concept, layer, tick, results_dir):
    if not _MPL:
        return
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    key = (layer, tick)
    if key not in per_pos_dict:
        return
    grid = np.zeros((8, 8))
    for (x, y), f1 in per_pos_dict[key].items():
        grid[y, x] = f1
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(grid, vmin=0, vmax=1, cmap="hot")
    ax.set_title(f"{concept} F1 — L{layer} T{tick}")
    plt.colorbar(im, ax=ax)
    path = os.path.join(fig_dir, f"{concept}_heatmap_L{layer}_T{tick}.png")
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    n_train = args.n_train_eps or (200 if args.quick else 3000)
    n_val   = args.n_val_eps   or (50  if args.quick else 1000)
    n_rand  = 100 if args.quick else 300
    n_pos   = args.n_positions or (16  if args.quick else 64)

    positions = [(x, y) for y in range(8) for x in range(8)]
    if n_pos < 64:
        rng = np.random.default_rng(args.seed)
        positions = [positions[i] for i in rng.choice(64, n_pos, replace=False)]

    # ── Load agents ────────────────────────────────────────────────────────────
    agent_a, cfg, step = load_agent(args.checkpoint, device)
    num_layers = agent_a.drc.num_layers
    num_ticks  = agent_a.drc.num_ticks

    # For v1 condition, partner B is the same model (self-play)
    agent_b_v1 = agent_a

    def env_fn():
        return MABoxobanEnv(
            data_dir=args.data_dir, split="train",
            difficulty="unfiltered", max_steps=400,
            seed=int(np.random.randint(10000)),
        )

    # ── WandB init ─────────────────────────────────────────────────────────────
    wrun = None
    if _WANDB:
        load_wandb_local_env()
        try:
            wrun = wandb.init(
                project = args.wandb_project,
                name    = f"tom-probes-step{step//1_000_000}M",
                config  = {"checkpoint": args.checkpoint, "n_train": n_train,
                           "n_positions": n_pos, **cfg},
            )
        except Exception as e:
            print(
                f"wandb.init failed ({e}); continuing without W&B.\n"
                f"  Repo root: wandb.local.example -> wandb.local (gitignored)\n"
                f"  Or:  wandb login  on login node\n"
                f"  Or:  export WANDB_API_KEY=...  |  WANDB_MODE=offline",
                flush=True,
            )
            wrun = None

    # ── Phase 3: Collect episodes ──────────────────────────────────────────────
    print("\n=== PHASE 3: COLLECTING PROBE DATA ===")
    ep_cache = os.path.join(args.results_dir, "episodes_train_v1.pkl")
    if os.path.exists(ep_cache):
        print(f"Loading cached: {ep_cache}")
        with open(ep_cache, "rb") as f:
            train_eps_v1 = pickle.load(f)
    else:
        train_eps_v1 = collect_ma_episodes(agent_a, agent_b_v1, env_fn, n_train, device)
        with open(ep_cache, "wb") as f:
            pickle.dump(train_eps_v1, f, protocol=4)

    # ── Phase 3: Label ─────────────────────────────────────────────────────────
    print("\nLabelling ToM concepts...")
    ta_v1, tb_v1, tc_v1 = [], [], []
    for ep in train_eps_v1:
        ta, tb, tc = label_tom_episode(
            ep["agent_b_positions"], ep["box_pushes_b"], ep["box_positions"]
        )
        ta_v1.append(ta); tb_v1.append(tb); tc_v1.append(tc)

    # ── Phase 3: Train probes ──────────────────────────────────────────────────
    probe_cache = os.path.join(args.results_dir, "probe_results_v1.pkl")
    models_cache = os.path.join(args.results_dir, "probe_models_v1.pkl")
    need_fitted = (args.partner_v2_ckpt is not None) and (not args.skip_kill)
    fitted_v1 = None

    if os.path.exists(probe_cache):
        print(f"Loading cached probes: {probe_cache}")
        with open(probe_cache, "rb") as f:
            probe_results_v1 = pickle.load(f)
    else:
        hs_tr, obs_tr, ta_arr, tb_arr, tc_arr, ep_ids = prepare_tom_dataset(
            train_eps_v1, ta_v1, tb_v1, tc_v1, return_episode_ids=True,
        )
        print(f"  Data shape: hs={hs_tr.shape}  obs={obs_tr.shape}")
        print("\nTraining ToM probes (train/val split by episode)...")
        if need_fitted:
            probe_results_v1, fitted_v1 = train_all_tom_probes(
                hs_tr, obs_tr, ta_arr, tb_arr, tc_arr,
                num_ticks=num_ticks, num_layers=num_layers,
                positions=positions, verbose=True,
                episode_ids=ep_ids, return_probes=True,
            )
            with open(models_cache, "wb") as f:
                pickle.dump(fitted_v1, f, protocol=4)
        else:
            probe_results_v1 = train_all_tom_probes(
                hs_tr, obs_tr, ta_arr, tb_arr, tc_arr,
                num_ticks=num_ticks, num_layers=num_layers,
                positions=positions, verbose=True,
                episode_ids=ep_ids, return_probes=False,
            )
        with open(probe_cache, "wb") as f:
            pickle.dump(probe_results_v1, f, protocol=4)

    if need_fitted and fitted_v1 is None:
        if os.path.exists(models_cache):
            with open(models_cache, "rb") as f:
                fitted_v1 = pickle.load(f)
        else:
            hs_tr, obs_tr, ta_arr, tb_arr, tc_arr, ep_ids = prepare_tom_dataset(
                train_eps_v1, ta_v1, tb_v1, tc_v1, return_episode_ids=True,
            )
            print("Fitting probe models for cross-policy test (no models cache)...")
            _, fitted_v1 = train_all_tom_probes(
                hs_tr, obs_tr, ta_arr, tb_arr, tc_arr,
                num_ticks=num_ticks, num_layers=num_layers,
                positions=positions, verbose=False,
                episode_ids=ep_ids, return_probes=True,
            )
            with open(models_cache, "wb") as f:
                pickle.dump(fitted_v1, f, protocol=4)

    print("\n--- ToM Probe Results (v1 partner) ---")
    for concept in ("TA", "TB", "TC"):
        bl = probe_results_v1.get(f"{concept}_baseline", 0.0)
        best_key = max(probe_results_v1.get(concept, {(0,0): 0.0}),
                       key=probe_results_v1.get(concept, {}).get)
        best_f1  = probe_results_v1.get(concept, {}).get(best_key, 0.0)
        delta    = best_f1 - bl
        print(f"  {concept}: best_f1={best_f1:.3f}  obs_baseline={bl:.3f}  "
              f"delta={delta:+.3f}")

    # Save heatmaps
    best_key_ta = max(probe_results_v1.get("TA", {(0,0): 0.0}),
                      key=probe_results_v1.get("TA", {}).get)
    bl_ta, bt_ta = best_key_ta
    for concept in ("TA", "TB", "TC"):
        save_heatmap(probe_results_v1.get(f"{concept}_per_pos", {}),
                     concept, bl_ta, bt_ta, args.results_dir)

    if wrun:
        for concept in ("TA", "TB", "TC"):
            bl = probe_results_v1.get(f"{concept}_baseline", 0.0)
            for key, f1 in probe_results_v1.get(concept, {}).items():
                wrun.log({f"probe/{concept}_L{key[0]}_T{key[1]}": f1,
                          f"probe/{concept}_baseline": bl})

    # ── Phase 4: Kill tests ────────────────────────────────────────────────────
    kill_results = {}
    if not args.skip_kill:
        print("\n=== PHASE 4: KILL TESTS ===")

        # Test 1: Cross-policy (requires partner v2)
        if args.partner_v2_ckpt is not None:
            print("\n--- Kill Test 1: Cross-Policy Generalization ---")
            agent_b_v2, _, _ = load_agent(args.partner_v2_ckpt, device)
            v2_cache = os.path.join(args.results_dir, "episodes_train_v2.pkl")
            if os.path.exists(v2_cache):
                with open(v2_cache, "rb") as f:
                    train_eps_v2 = pickle.load(f)
            else:
                n_v2 = max(n_train // 3, 100)
                train_eps_v2 = collect_ma_episodes(agent_a, agent_b_v2, env_fn, n_v2, device)
                with open(v2_cache, "wb") as f:
                    pickle.dump(train_eps_v2, f, protocol=4)

            ta_v2, tb_v2, tc_v2 = [], [], []
            for ep in train_eps_v2:
                ta, tb, tc = label_tom_episode(
                    ep["agent_b_positions"], ep["box_pushes_b"], ep["box_positions"]
                )
                ta_v2.append(ta); tb_v2.append(tb); tc_v2.append(tc)

            cp_results = cross_policy_generalization_test(
                probe_results_v1, fitted_v1, train_eps_v2, ta_v2, tb_v2, tc_v2,
                positions=positions, verbose=True,
            )
            kill_results["cross_policy"] = cp_results
            if wrun:
                wrun.log({"kill/mean_stability_ta": cp_results["mean_stability_ta"],
                          "kill/mean_stability_tb": cp_results["mean_stability_tb"]})
        else:
            print("  [Skipping cross-policy test — no --partner-v2-ckpt provided]")

        # Test 2: Ambiguity
        print("\n--- Kill Test 2: Ambiguity ---")
        ambig_results = ambiguity_test(
            train_eps_v1, ta_v1, tb_v1,
            num_layers=num_layers, num_ticks=num_ticks,
            positions=positions, verbose=True,
        )
        kill_results["ambiguity"] = ambig_results
        if wrun:
            wrun.log({
                "kill/ta_ambiguous": ambig_results["ta_ambiguous"],
                "kill/ta_obvious": ambig_results["ta_obvious"],
                "kill/tb_ambiguous": ambig_results["tb_ambiguous"],
                "kill/tb_obvious": ambig_results["tb_obvious"],
            })

        # Test 3: Random-weights
        print("\n--- Kill Test 3: Random-Weights Baseline ---")
        rand_results = random_weights_baseline(
            env_fn, probe_results_v1,
            num_layers=num_layers, num_ticks=num_ticks,
            n_episodes=n_rand, positions=positions, verbose=True,
        )
        kill_results["random_weights"] = rand_results
        with open(os.path.join(args.results_dir, "kill_results.pkl"), "wb") as f:
            pickle.dump(kill_results, f, protocol=4)

    # ── Save summary ───────────────────────────────────────────────────────────
    summary = {
        "checkpoint":   args.checkpoint,
        "global_step":  step,
        "n_train_eps":  n_train,
        "probe_results": {
            concept: {str(k): v for k, v in probe_results_v1.get(concept, {}).items()}
            for concept in ("TA", "TB", "TC")
        },
        "obs_baselines": {
            concept: probe_results_v1.get(f"{concept}_baseline", 0.0)
            for concept in ("TA", "TB", "TC")
        },
        "kill_tests": {
            "cross_policy_mean_stability_ta": kill_results.get("cross_policy", {}).get("mean_stability_ta"),
            "cross_policy_mean_stability_tb": kill_results.get("cross_policy", {}).get("mean_stability_tb"),
            "ambiguity_ta_ambiguous": kill_results.get("ambiguity", {}).get("ta_ambiguous"),
            "ambiguity_ta_obvious":   kill_results.get("ambiguity", {}).get("ta_obvious"),
            "ambiguity_tb_ambiguous": kill_results.get("ambiguity", {}).get("tb_ambiguous"),
            "ambiguity_tb_obvious":   kill_results.get("ambiguity", {}).get("tb_obvious"),
        },
    }
    summary_path = os.path.join(args.results_dir, "tom_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")

    if wrun:
        wrun.finish()

    print("\nDone.  Run generate_tom_report.py to produce TOM_RESULTS.md")


if __name__ == "__main__":
    main()
