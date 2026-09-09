"""
Stratified analysis by prompt subtlety level.

Breaks down the projection gap (ΔP) by subtlety:
- explicit: Clearly harmful intent in the words
- contextual: Harm is implied by context, not explicit words
- framed: Professional/legitimate framing hiding harmful action

This explains why the global effect size (Cohen's d) may appear small:
explicit prompts likely show a large effect, while framed prompts show little,
and averaging them together dilutes the signal.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from scipy import stats


@dataclass
class StratifiedResult:
    """Result for one subtlety level."""
    subtlety: str
    n_samples: int
    delta_p: float
    cohens_d: float
    p_value: float
    ci_lower: float
    ci_upper: float
    mean_chat: float
    mean_agent: float
    std_chat: float
    std_agent: float
    significant: bool


def run_stratified_analysis(
    strata: List[str],
    activations: Dict[str, Dict[int, np.ndarray]],
    refusal_direction: np.ndarray,
    layer: int,
    n_permutations: int = 10000,
    seed: int = 42,
) -> List[StratifiedResult]:
    """
    Run stratified projection gap analysis by subtlety level.

    Args:
        strata: Per-row subtlety label, ALIGNED with the rows of
            activations['chat_harmful'] (and therefore with the agent arm,
            which the extraction stage guarantees is row-aligned by pair_id).
            This used to be re-derived by re-filtering the raw dataset, which
            silently assumed the file's variant ordering matched the
            activation ordering; when it did not, the routine compared one
            prompt's chat activation against a different prompt's agent
            activation and reported the result without complaint.
        activations: Dict of variant -> {layer -> (n, hidden_dim)}.
        refusal_direction: Direction vector for the target layer.
        layer: Layer index to analyze.
        n_permutations: For permutation test.
        seed: Random seed.

    Returns:
        List of StratifiedResult, one per subtlety level.
    """
    from .robust_estimation import (
        paired_bootstrap_ci,
        paired_permutation_test,
        pooled_cohens_d,
    )

    agent_key = "agent_harmful" if "agent_harmful" in activations else "agent_full_harmful"

    if "chat_harmful" not in activations or agent_key not in activations:
        print("  WARNING: Required activations not found for stratification.")
        return []

    chat_acts = activations["chat_harmful"][layer]  # (n, hidden_dim)
    agent_acts = activations[agent_key][layer]      # (n, hidden_dim)

    if len(chat_acts) != len(agent_acts):
        raise ValueError(
            f"chat ({len(chat_acts)}) and agent ({len(agent_acts)}) arms are not "
            "paired row-wise; stratified DeltaP would be meaningless."
        )
    if len(strata) != len(chat_acts):
        raise ValueError(
            f"{len(strata)} stratum labels for {len(chat_acts)} activations; "
            "the labels must be row-aligned, not merely the same multiset."
        )

    # Project all onto refusal direction
    proj_chat = chat_acts @ refusal_direction
    proj_agent = agent_acts @ refusal_direction

    results = []
    for subtlety in sorted(set(strata)):
        idx = np.array([i for i, s in enumerate(strata) if s == subtlety])
        if len(idx) < 2:
            continue

        # One index vector: within a stratum the two conditions are the SAME
        # prompts, so the pairing is preserved by construction.
        proj_c, proj_a = proj_chat[idx], proj_agent[idx]

        delta_p = float(proj_c.mean() - proj_a.mean())

        # Sign-flip null (exchangeable within pair) and a paired bootstrap that
        # resamples prompts, not conditions independently.
        perm = paired_permutation_test(
            proj_c, proj_a, n_permutations=n_permutations,
            alternative="two-sided", seed=seed,
        )
        boot = paired_bootstrap_ci(proj_c, proj_a, n_boot=10000, seed=seed)

        results.append(StratifiedResult(
            subtlety=subtlety,
            n_samples=len(idx),
            delta_p=delta_p,
            cohens_d=pooled_cohens_d(proj_c, proj_a),
            p_value=perm["p_value"],
            ci_lower=boot["ci_lower"],
            ci_upper=boot["ci_upper"],
            mean_chat=float(proj_c.mean()),
            mean_agent=float(proj_a.mean()),
            std_chat=float(proj_c.std(ddof=1)),
            std_agent=float(proj_a.std(ddof=1)),
            significant=perm["p_value"] < 0.05,
        ))

    return results


def print_stratified_summary(results: List[StratifiedResult], layer: int):
    """Print formatted summary of stratified analysis."""
    print("\n" + "=" * 70)
    print(f"STRATIFIED ANALYSIS BY SUBTLETY (Layer {layer})")
    print("=" * 70)

    print(f"\n{'Subtlety':<12} {'N':<5} {'ΔP':<9} {'Cohen d':<9} {'p-value':<10} "
          f"{'95% CI':<22} {'Sig?'}")
    print("-" * 78)

    for r in sorted(results, key=lambda x: -x.delta_p):
        sig = "✓" if r.significant else "✗"
        ci = f"[{r.ci_lower:.3f}, {r.ci_upper:.3f}]"
        print(f"{r.subtlety:<12} {r.n_samples:<5} {r.delta_p:<9.4f} {r.cohens_d:<9.3f} "
              f"{r.p_value:<10.4f} {ci:<22} {sig}")

    print(f"\n  Interpretation:")
    sig_results = [r for r in results if r.significant]
    if sig_results:
        strongest = max(sig_results, key=lambda r: r.cohens_d)
        print(f"  → Strongest effect in '{strongest.subtlety}' prompts "
              f"(d={strongest.cohens_d:.3f}, ΔP={strongest.delta_p:.4f})")
    else:
        print(f"  → No subtlety level shows significant effect individually")

    # Is the gap monotone in explicitness? Check the whole ordering, not just
    # its endpoints: comparing explicit against framed and then announcing
    # "explicit > contextual > framed" asserts an ordering the test never
    # examined, and the middle stratum can sit anywhere.
    order = ["explicit", "contextual", "framed"]
    present = [s for s in order if s in {r.subtlety for r in results}]
    by_sub = {r.subtlety: r for r in results}
    if len(present) >= 2:
        deltas = [by_sub[s].delta_p for s in present]
        if all(a >= b for a, b in zip(deltas, deltas[1:])):
            print(f"  → Monotone in explicitness: "
                  f"{' > '.join(present)} ({', '.join(f'{d:.3f}' for d in deltas)})")
            print( "    The gap shrinks as harm becomes less explicit, so a pooled")
            print( "    effect size averages heterogeneous strata.")
        else:
            print(f"  → NOT monotone: {', '.join(f'{s}={by_sub[s].delta_p:.3f}' for s in present)}")
            print( "    The gap does not track explicitness in the expected order.")

    print("  NOTE: DeltaP measures INTENSITY of the projection, not detection.")
    print("        Objective 4 asks about detection capability, which is AUROC.")
    print("        See format_effect_results.json -> by_stratum.")
    print("=" * 70)


def plot_stratified_results(
    results: List[StratifiedResult],
    layer: int,
    output_dir: str = "results/figures",
):
    """Generate bar chart of ΔP by subtlety with CI error bars."""
    import matplotlib.pyplot as plt
    from pathlib import Path

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Sort by ΔP
    results_sorted = sorted(results, key=lambda r: -r.delta_p)
    labels = [r.subtlety for r in results_sorted]
    deltas = [r.delta_p for r in results_sorted]
    cohens = [r.cohens_d for r in results_sorted]
    ci_low = [r.delta_p - r.ci_lower for r in results_sorted]
    ci_high = [r.ci_upper - r.delta_p for r in results_sorted]
    p_vals = [r.p_value for r in results_sorted]

    # Colors by significance
    colors = ['#e74c3c' if p < 0.05 else '#95a5a6' for p in p_vals]

    # --- Left: ΔP bars ---
    bars = ax1.bar(range(len(labels)), deltas, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=0.5,
                   yerr=[ci_low, ci_high], capsize=8)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylabel('ΔP (chat - agent)', fontsize=11)
    ax1.set_title(f'Projection Gap by Subtlety (Layer {layer})\n'
                  f'Red = p<0.05, Gray = ns', fontsize=12)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.grid(axis='y', alpha=0.3)

    # Add p-value annotations
    for i, (p, d) in enumerate(zip(p_vals, deltas)):
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        y_pos = d + ci_high[i] + 0.05
        ax1.text(i, y_pos, f'{star}\np={p:.3f}', ha='center', fontsize=9,
                 color='#c0392b' if p < 0.05 else '#7f8c8d')

    # --- Right: Cohen's d bars ---
    colors_d = ['#27ae60' if d > 0.8 else '#f39c12' if d > 0.5 else '#3498db'
                for d in cohens]
    ax2.barh(range(len(labels)), cohens, color=colors_d, alpha=0.8,
             edgecolor='black', linewidth=0.5)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=11)
    ax2.set_xlabel("Cohen's d", fontsize=11)
    ax2.set_title("Effect Size by Subtlety\n"
                  "Green>0.8, Orange>0.5, Blue<0.5", fontsize=12)
    ax2.axvline(0.2, color='gray', linestyle='--', alpha=0.5, label='Small (0.2)')
    ax2.axvline(0.5, color='gray', linestyle='-.', alpha=0.5, label='Medium (0.5)')
    ax2.axvline(0.8, color='gray', linestyle=':', alpha=0.5, label='Large (0.8)')
    ax2.legend(fontsize=8)
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{output_dir}/stratified_by_subtlety.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_dir}/stratified_by_subtlety.png")
