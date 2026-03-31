"""
Full experiment pipeline: Phase 3 (probes) → Phase 4 (kill tests) →
Phase 5 (causal intervention) → Phase 6 (visualisation + report).

Usage:
    # Full pipeline on a trained checkpoint
    python scripts/run_full_experiment.py \\
        --checkpoint checkpoints/agent_10M.pt \\
        --data-dir data/boxoban_levels \\
        --results-dir results/

    # Quick run on any available checkpoint (fewer episodes/positions)
    python scripts/run_full_experiment.py \\
        --checkpoint /tmp/test_ckpt_final.pt \\
        --data-dir data/boxoban_levels \\
        --quick \\
        --results-dir results/quick/
"""

import argparse
import os
import sys
import pickle
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from drc_sokoban.envs.boxoban_env import BoxobanEnv
from drc_sokoban.models.agent import DRCAgent
from drc_sokoban.probing.concept_labeler import (
    label_episode_fast, label_episodes_parallel, NEVER
)
from drc_sokoban.probing.train_probes import (
    prepare_probe_dataset, split_episodes_by_index, train_all_probes,
    train_spatial_probe,
)
from drc_sokoban.probing.evaluate_probes import (
    print_results_table, save_results, check_probe_sanity,
    plot_f1_heatmap, plot_spatial_f1,
)
from drc_sokoban.probing.kill_tests import (
    run_window_baseline, run_random_network_baseline,
    run_cross_level_test, print_kill_test_summary,
)
from drc_sokoban.probing.causal_intervention import (
    extract_plan_vector, HiddenStateInjector, NullInjector,
    run_dose_response, print_dose_response_table, measure_action_shift,
)
from drc_sokoban.probing.visualize import (
    ascii_probe_overlay, plot_probe_confidence_grid,
    plot_smoking_gun, plot_tick_progression,
)


ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--data-dir",    type=str, default=None)
    p.add_argument("--results-dir", type=str, default="results/")
    p.add_argument("--device",      type=str, default="auto")
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--quick",       action="store_true",
                   help="Fast mode: fewer episodes and positions")
    p.add_argument("--n-train-eps", type=int, default=None,
                   help="Override number of probe training episodes")
    p.add_argument("--n-positions", type=int, default=None,
                   help="Override number of board positions probed")
    p.add_argument("--skip-causal", action="store_true")
    p.add_argument("--skip-kill",   action="store_true")
    return p.parse_args()


def load_agent(checkpoint_path: str, device: torch.device) -> DRCAgent:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt.get("cfg", {})
    agent = DRCAgent(
        hidden_channels = cfg.get("hidden_channels", 32),
        num_layers      = cfg.get("num_layers",      2),
        num_ticks       = cfg.get("num_ticks",       3),
    ).to(device)
    agent.load_state_dict(ckpt["model_state"])
    agent.eval()
    step = ckpt.get("global_step", 0)
    print(f"Loaded agent: {checkpoint_path}  (step {step:,})")
    return agent, cfg, step


def collect_episodes(
    agent: DRCAgent,
    env_factory,
    n_episodes: int,
    device: torch.device,
    greedy: bool = True,
    max_steps_per_ep: int = 200,
) -> list:
    """
    Run agent and collect full episode data for probing.
    Uses a batch of parallel environments (n_batch) for speed.
    """
    from tqdm import tqdm

    n_batch  = min(16, n_episodes)   # run 16 envs in parallel (CPU inference)
    episodes = []
    solve_count = 0

    # Build n_batch envs
    envs    = [env_factory() for _ in range(n_batch)]
    obs_all = np.stack([e.reset() for e in envs], axis=0)
    hidden_all = agent.init_hidden(n_batch, device)
    ep_bufs = [{k: [] for k in ("observations", "hidden_states", "actions",
                                "agent_positions", "box_positions",
                                "rewards", "dones")}
               for _ in range(n_batch)]
    ep_steps = [0] * n_batch

    collected = 0
    with tqdm(total=n_episodes, desc="Collecting") as pbar:
        while collected < n_episodes:
            obs_t = torch.FloatTensor(obs_all).to(device)
            with torch.no_grad():
                logits, _, new_hidden_all, all_ticks_all = agent(
                    obs_t, hidden_all, return_all_ticks=True
                )

            # all_ticks_all[tick][layer] = (h, c) of shape (n_batch, 32, 8, 8)
            num_ticks  = len(all_ticks_all)
            num_layers = len(all_ticks_all[0])
            tick_arr_batch = np.stack([
                np.stack([all_ticks_all[tick][layer][0].cpu().numpy()
                          for layer in range(num_layers)], axis=1)
                for tick in range(num_ticks)
            ], axis=1)  # (n_batch, num_ticks, num_layers, 32, 8, 8)

            if greedy:
                actions = torch.argmax(logits, dim=-1).cpu().numpy()
            else:
                actions = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()

            new_obs_all = []
            new_hidden_list = list(new_hidden_all)
            dones = np.zeros(n_batch, dtype=bool)

            for i, env in enumerate(envs):
                ep_bufs[i]["observations"].append(obs_all[i].copy())
                ep_bufs[i]["hidden_states"].append(tick_arr_batch[i])
                ep_bufs[i]["agent_positions"].append(env.get_agent_pos())
                ep_bufs[i]["box_positions"].append(env.get_box_positions())
                ep_bufs[i]["actions"].append(int(actions[i]))

                obs_i, rew, done, info = env.step(int(actions[i]))
                ep_bufs[i]["rewards"].append(rew)
                ep_bufs[i]["dones"].append(done)
                ep_steps[i] += 1

                force_done = done or ep_steps[i] >= max_steps_per_ep
                dones[i] = force_done
                new_obs_all.append(obs_i)

            obs_all = np.stack(new_obs_all, axis=0)
            mask_t = torch.FloatTensor(
                [[0.0 if dones[i] else 1.0] for i in range(n_batch)]
            ).view(n_batch, 1, 1, 1).to(device)
            hidden_all = [(h * mask_t, c * mask_t) for h, c in new_hidden_list]

            # Collect finished episodes
            for i in range(n_batch):
                if dones[i] and collected < n_episodes:
                    ep = ep_bufs[i]
                    for k in ("observations","hidden_states","actions","rewards","dones"):
                        ep[k] = np.array(ep[k])
                    ep["solved"] = bool(np.any(ep["rewards"] > 5.0))
                    if ep["solved"]:
                        solve_count += 1
                    episodes.append(ep)
                    collected += 1
                    pbar.update(1)

                    # Reset env
                    obs_all[i] = envs[i].reset()
                    ep_bufs[i] = {k: [] for k in ("observations", "hidden_states",
                                                   "actions", "agent_positions",
                                                   "box_positions", "rewards", "dones")}
                    ep_steps[i] = 0
                    # Zero hidden state for this env slot
                    new_ha = []
                    for h, c in hidden_all:
                        h2 = h.clone(); h2[i] = 0.0
                        c2 = c.clone(); c2[i] = 0.0
                        new_ha.append((h2, c2))
                    hidden_all = new_ha

    print(f"  Solve rate: {solve_count/max(collected,1):.3f}")
    return episodes


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # ── Quick-mode overrides ────────────────────────────────────────────────────
    n_train_eps  = args.n_train_eps  or (200  if args.quick else 2000)
    n_val_eps    = 50   if args.quick else 500
    n_rand_eps   = 100  if args.quick else 300
    n_alpha_eps  = 20   if args.quick else 50
    n_pos        = args.n_positions or (16   if args.quick else 64)

    positions = [(x, y) for y in range(8) for x in range(8)]
    if n_pos < 64:
        np.random.seed(args.seed)
        idx = np.random.choice(64, size=n_pos, replace=False)
        positions = [positions[i] for i in idx]

    # ── Load agent ──────────────────────────────────────────────────────────────
    agent, cfg, global_step = load_agent(args.checkpoint, device)
    num_layers = agent.drc.num_layers
    num_ticks  = agent.drc.num_ticks

    def env_fn():
        return BoxobanEnv(
            data_dir=args.data_dir,
            split="train",
            difficulty="unfiltered",
            max_steps=400,
            seed=np.random.randint(10000),
        )

    def env_fn_medium():
        return BoxobanEnv(
            data_dir=args.data_dir,
            split="train",
            difficulty="medium",
            max_steps=400,
            seed=np.random.randint(10000),
        )

    # ─────────────────────────────────────────────────────────────────────────────
    # PHASE 3: Base probe training
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 3 — COLLECTING PROBE DATA")
    print("="*60)

    probe_cache = os.path.join(args.results_dir, "episodes_train.pkl")
    if os.path.exists(probe_cache):
        print(f"Loading cached episodes from {probe_cache}")
        with open(probe_cache, "rb") as f:
            train_episodes = pickle.load(f)
    else:
        train_episodes = collect_episodes(agent, env_fn, n_train_eps, device)
        with open(probe_cache, "wb") as f:
            pickle.dump(train_episodes, f, protocol=4)

    val_cache = os.path.join(args.results_dir, "episodes_val.pkl")
    if os.path.exists(val_cache):
        with open(val_cache, "rb") as f:
            val_episodes = pickle.load(f)
    else:
        val_episodes = collect_episodes(agent, env_fn, n_val_eps, device)
        with open(val_cache, "wb") as f:
            pickle.dump(val_episodes, f, protocol=4)

    # Cache check: skip expensive probe training if results already exist
    probe_cache_path = os.path.join(args.results_dir, "probe_results.pkl")
    best_probe_path  = os.path.join(args.results_dir, "best_probe_ca.pkl")

    if os.path.exists(probe_cache_path) and os.path.exists(best_probe_path):
        print("\nLoading cached probe results...")
        with open(probe_cache_path, "rb") as f:
            probe_results = pickle.load(f)
        with open(best_probe_path, "rb") as f:
            best_probe_ca = pickle.load(f)
        print_results_table(probe_results)
    else:
        print("\nLabeling episodes...")
        ca_train = [label_episode_fast(ep["agent_positions"], ep["box_positions"])[0]
                    for ep in train_episodes]
        cb_train = [label_episode_fast(ep["agent_positions"], ep["box_positions"])[1]
                    for ep in train_episodes]

        hs_tr, obs_tr, ca_arr, cb_arr = prepare_probe_dataset(
            train_episodes, ca_train, cb_train
        )
        print(f"  Data shape: hs={hs_tr.shape}")

        print("\nTraining probes...")
        probe_results = train_all_probes(
            hs_tr, obs_tr, ca_arr, cb_arr,
            num_ticks=num_ticks, num_layers=num_layers,
            positions=positions, verbose=True,
        )
        save_results(probe_results, probe_cache_path)

        print("\n--- Base probe results ---")
        print_results_table(probe_results)
        check_probe_sanity(probe_results)

        # Train a single high-quality probe for causal use
        probe_pos_x, probe_pos_y = 3, 3
        best_key = max(probe_results["CA"], key=probe_results["CA"].get)
        best_layer, best_tick = best_key
        best_probe_ca, _ = train_spatial_probe(
            hs_tr, ca_arr, probe_pos_x, probe_pos_y, best_layer, best_tick, n_seeds=5
        )
        with open(best_probe_path, "wb") as f:
            pickle.dump(best_probe_ca, f)

    # Need ca_train for window baseline — label if not already done
    if "ca_train" not in dir():
        print("\nLabeling episodes for kill tests...")
        ca_train = [label_episode_fast(ep["agent_positions"], ep["box_positions"])[0]
                    for ep in train_episodes]
        cb_train = [label_episode_fast(ep["agent_positions"], ep["box_positions"])[1]
                    for ep in train_episodes]

    check_probe_sanity(probe_results)

    # Pick the best (layer, tick) for downstream experiments
    best_key = max(probe_results["CA"], key=probe_results["CA"].get)
    best_layer, best_tick = best_key
    print(f"\nBest (layer, tick) for CA: ({best_layer}, {best_tick})  "
          f"F1={probe_results['CA'][best_key]:.3f}")
    probe_pos_x, probe_pos_y = 3, 3

    # Free large episode arrays from memory before Phase 4 (avoids OOM)
    import gc
    try:
        del hs_tr, obs_tr, ca_arr, cb_arr
    except NameError:
        pass
    # Keep only labels and episode metadata (small) for window probe
    gc.collect()

    # ─────────────────────────────────────────────────────────────────────────────
    # PHASE 4: Kill Tests
    # ─────────────────────────────────────────────────────────────────────────────
    if not args.skip_kill:
        print("\n" + "="*60)
        print("PHASE 4 — KILL TESTS")
        print("="*60)

        # 4.1 K-Frame Window Probe
        _window_cache = os.path.join(args.results_dir, "window_results.pkl")
        if os.path.exists(_window_cache):
            print("\n--- 4.1  K=5 Window Probe (cached) ---")
            with open(_window_cache, "rb") as f:
                window_results = pickle.load(f)
            print(f"CA_window={window_results['CA_window']:.3f} | CB_window={window_results['CB_window']:.3f}")
        else:
            print("\n--- 4.1  K=5 Window Probe ---")
            window_results = run_window_baseline(
                train_episodes, ca_train, cb_train, k=5,
                positions=positions, verbose=True,
            )
            save_results(window_results, _window_cache)

        # Free episode data now that window probe is done
        try:
            del train_episodes, val_episodes, ca_train, cb_train
        except NameError:
            pass
        gc.collect()

        # 4.2 Random-weights baseline
        _rand_cache = os.path.join(args.results_dir, "random_net_results.pkl")
        if os.path.exists(_rand_cache):
            print("\n--- 4.2  Random Network Baseline (cached) ---")
            with open(_rand_cache, "rb") as f:
                random_results = pickle.load(f)
            print(f"Random net CA: {random_results.get('CA', {})}")
        else:
            print("\n--- 4.2  Random Network Baseline ---")
            random_results = run_random_network_baseline(
                probe_results, env_fn,
                num_layers=num_layers, num_ticks=num_ticks,
                hidden_channels=cfg.get("hidden_channels", 32),
                n_episodes=n_rand_eps,
                positions=positions, verbose=True,
            )
            save_results(random_results, _rand_cache)

        # 4.3 Cross-level generalisation
        ood_results = None
        medium_dir = (os.path.join(args.data_dir, "medium", "train") if args.data_dir else None)
        if medium_dir and os.path.isdir(medium_dir):
            print("\n--- 4.3  Cross-Level Generalisation (medium levels) ---")
            ood_cache = os.path.join(args.results_dir, "ood_episodes.pkl")
            if os.path.exists(ood_cache):
                with open(ood_cache, "rb") as f:
                    ood_episodes = pickle.load(f)
            else:
                ood_episodes = collect_episodes(agent, env_fn_medium, 100, device)
                with open(ood_cache, "wb") as f:
                    pickle.dump(ood_episodes, f, protocol=4)

            ca_ood = [label_episode_fast(ep["agent_positions"], ep["box_positions"])[0]
                      for ep in ood_episodes]
            cb_ood = [label_episode_fast(ep["agent_positions"], ep["box_positions"])[1]
                      for ep in ood_episodes]

            ood_results = run_cross_level_test(
                probe_results, ood_episodes, ca_ood, cb_ood,
                positions=positions, verbose=True,
            )
            save_results(ood_results, os.path.join(args.results_dir, "ood_results.pkl"))
        else:
            print("  [Skipping 4.3 — no medium levels directory found]")

        print_kill_test_summary(probe_results, window_results, random_results, ood_results)
        kill_summary = {
            "window":    window_results,
            "random":    random_results,
            "ood":       ood_results,
        }
        save_results(kill_summary, os.path.join(args.results_dir, "kill_tests.pkl"))

    # ─────────────────────────────────────────────────────────────────────────────
    # PHASE 5: Causal Intervention
    # ─────────────────────────────────────────────────────────────────────────────
    if not args.skip_causal:
        print("\n" + "="*60)
        print("PHASE 5 — CAUSAL INTERVENTION (ACTIVATION STEERING)")
        print("="*60)

        # Extract UP plan vector from the best probe
        plan_vec = extract_plan_vector(best_probe_ca, target_class=0)  # UP
        print(f"Plan vector (UP) norm: {np.linalg.norm(plan_vec):.4f}  "
              f"shape: {plan_vec.shape}")

        # Dose-response
        _causal_cache = os.path.join(args.results_dir, "causal_results.pkl")
        if os.path.exists(_causal_cache):
            print("\n--- 5.3  Dose-Response Analysis (cached) ---")
            with open(_causal_cache, "rb") as f:
                _c = pickle.load(f)
            action_dists = {float(k): np.array(v) for k, v in _c.get("action_dists", {}).items()}
        else:
            print("\n--- 5.3  Dose-Response Analysis ---")
            alphas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
            action_dists = measure_action_shift(
                agent, env_fn, plan_vec,
                layer=best_layer, x=probe_pos_x, y=probe_pos_y,
                alphas=alphas,
                n_episodes=n_alpha_eps,
                device=str(device),
            )

        alphas = sorted(action_dists.keys())
        print("\nAction distribution (soft) under plan-UP injection:")
        print(f"{'α':>5}  {'UP':>6}  {'DOWN':>6}  {'LEFT':>6}  {'RIGHT':>6}  {'Shift↑':>8}")
        baseline_up = action_dists.get(0.0, np.ones(4) / 4)[0]
        for alpha in alphas:
            d = action_dists.get(alpha, np.ones(4) / 4)
            shift = d[0] - baseline_up
            print(f"{alpha:>5.1f}  {d[0]:>6.3f}  {d[1]:>6.3f}  {d[2]:>6.3f}  "
                  f"{d[3]:>6.3f}  {shift:>+8.3f}")

        save_results({
            "plan_vector": plan_vec.tolist(),
            "action_dists": {str(a): d.tolist() for a, d in action_dists.items()},
            "target_class": 0,
            "target_class_name": "UP",
            "layer": best_layer,
            "tick": best_tick,
            "position": (probe_pos_x, probe_pos_y),
        }, os.path.join(args.results_dir, "causal_results.pkl"))

    # ─────────────────────────────────────────────────────────────────────────────
    # PHASE 6: Visualisation
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 6 — VISUALISATION")
    print("="*60)

    # Collect a few solved episodes for vis
    vis_episodes = collect_episodes(agent, env_fn, 50, device, greedy=True)
    solved_eps = [ep for ep in vis_episodes if ep["solved"]]
    use_ep = solved_eps[0] if solved_eps else vis_episodes[0]
    ca_ep, cb_ep = label_episode_fast(use_ep["agent_positions"], use_ep["box_positions"])

    # ASCII overlay at t=0
    t0_hs  = use_ep["hidden_states"][0]
    t0_obs = use_ep["observations"][0]
    ascii_vis = ascii_probe_overlay(t0_obs, t0_hs, best_probe_ca, best_layer, best_tick, ca_ep[0])
    print("\nASCII overlay (t=0) — board | CA probe prediction | true CA label:")
    print(ascii_vis)

    # Save vis with matplotlib if available
    try:
        plot_probe_confidence_grid(
            best_probe_ca, t0_hs, best_layer, best_tick,
            target_class=0,
            title=f"CA probe (UP) confidence — step 0",
            save_path=os.path.join(args.results_dir, "figures", "ca_confidence_t0.png"),
            obs=t0_obs,
        )
        plot_tick_progression(
            best_probe_ca, t0_hs, best_layer,
            x=probe_pos_x, y=probe_pos_y,
            save_path=os.path.join(args.results_dir, "figures", "tick_progression.png"),
        )
        plot_smoking_gun(
            use_ep, best_probe_ca, best_layer, best_tick,
            concept="CA", t_early=0, t_late=min(20, len(use_ep["observations"])-1),
            save_path=os.path.join(args.results_dir, "figures", "smoking_gun.png"),
        )
        plot_f1_heatmap(
            probe_results, concept="CA",
            save_path=os.path.join(args.results_dir, "figures", "CA_heatmap.png"),
        )
        plot_f1_heatmap(
            probe_results, concept="CB",
            save_path=os.path.join(args.results_dir, "figures", "CB_heatmap.png"),
        )
        plot_spatial_f1(
            probe_results, concept="CA",
            save_path=os.path.join(args.results_dir, "figures", "CA_spatial.png"),
        )
    except Exception as e:
        print(f"  [matplotlib plots skipped: {e}]")

    # ─────────────────────────────────────────────────────────────────────────────
    # Compile all metrics for the report
    # ─────────────────────────────────────────────────────────────────────────────
    causal_data = {}
    if not args.skip_causal:
        causal_data = {str(a): d.tolist() for a, d in action_dists.items()}

    kill_data = {}
    if not args.skip_kill:
        kill_data = {
            "window_CA":  window_results.get("CA_window", 0.0),
            "window_CB":  window_results.get("CB_window", 0.0),
            "random_CA":  {str(k): v for k, v in random_results.get("CA", {}).items()},
            "random_CB":  {str(k): v for k, v in random_results.get("CB", {}).items()},
            "ood_stability": ood_results.get("mean_stability", None) if ood_results else None,
        }

    summary_metrics = {
        "global_step":      global_step,
        "solve_rate":       float(np.mean([ep["solved"] for ep in vis_episodes])),
        "probe_CA":         {str(k): v for k, v in probe_results.get("CA", {}).items()},
        "probe_CB":         {str(k): v for k, v in probe_results.get("CB", {}).items()},
        "probe_CA_baseline":probe_results.get("CA_baseline", 0.0),
        "probe_CB_baseline":probe_results.get("CB_baseline", 0.0),
        "kill_tests":       kill_data,
        "causal":           causal_data,
    }
    with open(os.path.join(args.results_dir, "summary_metrics.json"), "w") as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"\nMetrics saved to {args.results_dir}/summary_metrics.json")
    print("Done. Run generate_report.py to produce the final report.")


if __name__ == "__main__":
    main()
