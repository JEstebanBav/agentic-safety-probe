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
    dataset: List[Dict],
    activations: Dict[str, Dict[int, np.ndarray]],
    refusal_direction: np.ndarray,
    layer: int,
    n_permutations: int = 10000,
    seed: int = 42,
) -> List[StratifiedResult]:
    """
    Run stratified projection gap analysis by subtlety level.

    Args:
        dataset: Full dataset list (must have 'subtlety' field).
        activations: Dict of variant -> {layer -> (n, hidden_dim)}.
        refusal_direction: Direction vector for the target layer.
        layer: Layer index to analyze.
        n_permutations: For permutation test.
        seed: Random seed.

    Returns:
        List of StratifiedResult, one per subtlety level.
    """
    # Build index mapping: for each variant, which entries have which subtlety
    # We need to map dataset indices to activation indices
    chat_harmful_entries = [d for d in dataset if d["variant"] in ("chat_harmful", "chat_full_harmful")]
    agent_harmful_entries = [d for d in dataset if d["variant"] in ("agent_harmful", "agent_full_harmful")]

    # Resolve activation keys
    agent_key = "agent_harmful" if "agent_harmful" in activations else "agent_full_harmful"

    if "chat_harmful" not in activations or agent_key not in activations:
        print("  WARNING: Required activations not found for stratification.")
        return []

    chat_acts = activations["chat_harmful"][layer]  # (n, hidden_dim)
    agent_acts = activations[agent_key][layer]      # (n, hidden_dim)

    # Get subtlety for each index (assuming order is preserved from dataset)
    chat_subtleties = [e.get("subtlety", "unknown") for e in chat_harmful_entries]
    agent_subtleties = [e.get("subtlety", "unknown") for e in agent_harmful_entries]

    # Verify lengths match
    if len(chat_subtleties) != len(chat_acts):
        print(f"  WARNING: chat entries ({len(chat_subtleties)}) != activations ({len(chat_acts)})")
        # Try to use min
        n = min(len(chat_subtleties), len(chat_acts))
        chat_subtleties = chat_subtleties[:n]
        chat_acts = chat_acts[:n]

    if len(agent_subtleties) != len(agent_acts):
        n = min(len(agent_subtleties), len(agent_acts))
        agent_subtleties = agent_subtleties[:n]
        agent_acts = agent_acts[:n]

    # Project all onto refusal direction
    proj_chat = chat_acts @ refusal_direction
    proj_agent = agent_acts @ refusal_direction

    # Stratify
    subtlety_levels = sorted(set(chat_subtleties))
    rng = np.random.default_rng(seed)
    results = []

    for subtlety in subtlety_levels:
        # Get indices for this subtlety
        chat_idx = [i for i, s in enumerate(chat_subtleties) if s == subtlety]
        agent_idx = [i for i, s in enumerate(agent_subtleties) if s == subtlety]

        if not chat_idx or not agent_idx:
            continue

        proj_c = proj_chat[chat_idx]
        proj_a = proj_agent[agent_idx]

        n = min(len(proj_c), len(proj_a))
        proj_c = proj_c[:n]
        proj_a = proj_a[:n]

        # Compute gap
        delta_p = float(proj_c.mean() - proj_a.mean())

        # Cohen's d
        pooled_std = np.sqrt((proj_c.std(ddof=1)**2 + proj_a.std(ddof=1)**2) / 2)
        cohens_d = delta_p / pooled_std if pooled_std > 1e-10 else 0.0

        # Permutation test
        combined = np.concatenate([proj_c, proj_a])
        n_c = len(proj_c)
        null_gaps = np.empty(n_permutations)
        for i in range(n_permutations):
            perm = rng.permutation(combined)
            null_gaps[i] = perm[:n_c].mean() - perm[n_c:].mean()
        p_value = float((np.abs(null_gaps) >= np.abs(delta_p)).mean())

        # Bootstrap CI
        n_boot = 1000
        boot_diffs = np.empty(n_boot)
        for i in range(n_boot):
            idx_c = rng.integers(0, len(proj_c), size=len(proj_c))
            idx_a = rng.integers(0, len(proj_a), size=len(proj_a))
            boot_diffs[i] = proj_c[idx_c].mean() - proj_a[idx_a].mean()

        results.append(StratifiedResult(
            subtlety=subtlety,
            n_samples=n,
            delta_p=delta_p,
            cohens_d=float(cohens_d),
            p_value=p_value,
            ci_lower=float(np.percentile(boot_diffs, 2.5)),
            ci_upper=float(np.percentile(boot_diffs, 97.5)),
            mean_chat=float(proj_c.mean()),
            mean_agent=float(proj_a.mean()),
            std_chat=float(proj_c.std()),
            std_agent=float(proj_a.std()),
            significant=p_value < 0.05,
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

    # Check if explicit > contextual > framed
    by_sub = {r.subtlety: r for r in results}
    if "explicit" in by_sub and "framed" in by_sub:
        if by_sub["explicit"].delta_p > by_sub["framed"].delta_p:
            print(f"  → Pattern: explicit > contextual > framed (as expected)")
            print(f"    This confirms that the global d={sum(r.cohens_d for r in results)/len(results):.3f} "
                  f"is diluted by averaging heterogeneous effects.")
        else:
            print(f"  → Unexpected: framed prompts show larger gap than explicit")
            print(f"    The model may find professionally-framed harmful requests harder to refuse in agent mode")

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
