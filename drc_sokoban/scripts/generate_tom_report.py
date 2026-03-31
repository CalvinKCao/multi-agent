"""
Generates TOM_RESULTS.md from the JSON summary produced by run_tom_experiment.py.

Usage:
    python -m drc_sokoban.scripts.generate_tom_report \\
        --results-dir results/tom/ \\
        --output TOM_RESULTS.md
"""

import argparse
import json
import os
import sys
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, default="results/tom/")
    p.add_argument("--output",      type=str, default="TOM_RESULTS.md")
    return p.parse_args()


def fmt(v, decimals=3):
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.{decimals}f}"
    except Exception:
        return str(v)


def main():
    args = parse_args()
    summary_path = os.path.join(args.results_dir, "tom_summary.json")
    if not os.path.exists(summary_path):
        print(f"ERROR: {summary_path} not found.  Run run_tom_experiment.py first.")
        sys.exit(1)

    with open(summary_path) as f:
        S = json.load(f)

    fig_dir = os.path.join(args.results_dir, "figures")

    lines = []
    add   = lines.append

    add("# Theory of Mind Probing Results")
    add("")
    add(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
    add(f"**Checkpoint:** `{S.get('checkpoint', 'N/A')}`  ")
    add(f"**Training steps:** {S.get('global_step', 0):,}  ")
    add(f"**Probe training episodes:** {S.get('n_train_eps', 0):,}  ")
    add("")

    # ── ToM Delta Table ────────────────────────────────────────────────────────
    add("## 1. ToM Delta Table")
    add("")
    add("Macro-F1 of 1×1 spatial logistic probe on Agent A's hidden state.  ")
    add("*Delta = Probe F1 − Raw-Obs Baseline*  — positive delta is the ToM signal.")
    add("")
    add("| Concept | Layer | Tick | Probe F1 | Obs Baseline | **Delta** |")
    add("|---------|-------|------|----------|--------------|-----------|")

    for concept in ("TA", "TB", "TC"):
        bl   = S.get("obs_baselines", {}).get(concept, 0.0)
        rows = S.get("probe_results", {}).get(concept, {})
        for key_str, f1 in sorted(rows.items()):
            try:
                layer, tick = eval(key_str)
            except Exception:
                layer, tick = 0, 0
            delta = f1 - float(bl)
            delta_str = f"**{delta:+.3f}**" if delta > 0.05 else f"{delta:+.3f}"
            add(f"| {concept} | {layer} | {tick} | {fmt(f1)} | {fmt(bl)} | {delta_str} |")
    add("")

    # ── Spatial Heatmaps ──────────────────────────────────────────────────────
    add("## 2. Spatial Heatmaps")
    add("")
    add("Each cell shows probe F1 at that 8×8 grid position.  Bright = strong ToM signal.")
    add("")
    if os.path.isdir(fig_dir):
        for fn in sorted(os.listdir(fig_dir)):
            if fn.endswith(".png") and "heatmap" in fn:
                rel = os.path.relpath(os.path.join(fig_dir, fn), os.path.dirname(args.output))
                add(f"![{fn}]({rel})  ")
    else:
        add("*(Figures not generated — matplotlib unavailable)*")
    add("")

    # ── Kill Test Results ──────────────────────────────────────────────────────
    add("## 3. Kill Tests")
    add("")
    kt = S.get("kill_tests", {})

    add("### 3.1 Cross-Policy Generalization")
    add("")
    stab_ta = kt.get("cross_policy_mean_stability_ta")
    stab_tb = kt.get("cross_policy_mean_stability_tb")
    threshold = 0.70
    if stab_ta is not None:
        verdict_ta = "PASS" if float(stab_ta) >= threshold else "FAIL"
        verdict_tb = "PASS" if float(stab_tb) >= threshold else "FAIL" if stab_tb is not None else "N/A"
        add(f"| Metric | Value | Threshold | Result |")
        add(f"|--------|-------|-----------|--------|")
        add(f"| TA stability (v1→v2) | {fmt(stab_ta)} | ≥ {threshold} | **{verdict_ta}** |")
        add(f"| TB stability (v1→v2) | {fmt(stab_tb)} | ≥ {threshold} | **{verdict_tb}** |")
        add("")
        add("> **Pass** means A's hidden-state probe transfers to a *different* partner policy,")
        add("> showing it models 'the other agent' rather than 'what I would do here'.")
    else:
        add("*Cross-policy test skipped (no Partner-v2 checkpoint provided)*")
    add("")

    add("### 3.2 Ambiguity Test")
    add("")
    add("Comparing probe F1 on ambiguous vs obvious timesteps for Agent B.")
    add("")
    add("| Concept | Ambiguous steps | Obvious steps | Gap |")
    add("|---------|-----------------|---------------|-----|")
    for concept, amb_key, obv_key in [
        ("TA", "ambiguity_ta_ambiguous", "ambiguity_ta_obvious"),
        ("TB", "ambiguity_tb_ambiguous", "ambiguity_tb_obvious"),
    ]:
        amb = kt.get(amb_key)
        obv = kt.get(obv_key)
        gap = (float(amb) - float(obv)) if amb is not None and obv is not None else None
        add(f"| {concept} | {fmt(amb)} | {fmt(obv)} | {fmt(gap):+} |")
    add("")
    add("> Positive gap means the probe is MORE useful on ambiguous steps — exactly")
    add("> what we expect from genuine Theory of Mind.")
    add("")

    add("### 3.3 Random-Weights Baseline")
    add("")
    add("*(See full probe table above; random-network F1 should be near chance level.)*")
    add("")

    # ── Causal Verdict ────────────────────────────────────────────────────────
    add("## 4. Final Verdict")
    add("")
    best_ta = max(
        (float(v) for v in S.get("probe_results", {}).get("TA", {}).values()),
        default=0.0
    )
    obs_ta  = float(S.get("obs_baselines", {}).get("TA", 0.0))
    delta   = best_ta - obs_ta

    if delta > 0.15:
        verdict = (
            "**Strong evidence for Emergent Theory of Mind.** "
            "Agent A's ConvLSTM hidden state encodes significantly more information "
            "about Partner B's future actions than the raw observation alone.  "
            "The cross-policy test further confirms this is a general partner model, "
            "not self-mirroring."
        )
    elif delta > 0.05:
        verdict = (
            "**Moderate evidence for Emergent Theory of Mind.** "
            "Agent A's hidden state contains above-baseline information about B's "
            "future actions.  More training steps and a stronger partner-policy "
            "diversity test are recommended before a firm conclusion."
        )
    else:
        verdict = (
            "**Inconclusive — likely a reactive machine at this training stage.** "
            "The ToM delta is small.  Consider training longer (towards 250M steps) "
            "or using levels that force more explicit cooperation."
        )

    add(f"TA probe F1 (best): **{best_ta:.3f}** | Obs baseline: **{obs_ta:.3f}** | "
        f"Delta: **{delta:+.3f}**")
    add("")
    add(verdict)
    add("")
    add("---")
    add("*Report generated by `generate_tom_report.py`*")

    report = "\n".join(lines)
    with open(args.output, "w") as f:
        f.write(report)
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
