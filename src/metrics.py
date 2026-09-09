"""
Statistical metrics and tests module.

Implements all evaluation metrics for the experiment:
- Projection Gap (ΔP)
- AUROC by format
- Paired statistical tests
- Bootstrap confidence intervals
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from sklearn.metrics import roc_auc_score


@dataclass
class StatisticalTestResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    effect_size_name: str
    ci_lower: float
    ci_upper: float
    significant: bool  # at alpha=0.01
    interpretation: str


def paired_ttest(
    projections_chat: np.ndarray,
    projections_agent: np.ndarray,
    alpha: float = 0.01,
) -> StatisticalTestResult:
    """
    Paired t-test: Do projections differ between chat and agent format?
    
    H₀: mean(proj_chat) = mean(proj_agent)
    H₁: mean(proj_chat) > mean(proj_agent) (one-sided)
    
    Args:
        projections_chat: Projections for harmful prompts in chat format.
        projections_agent: Projections for SAME prompts in agent format.
        alpha: Significance level.
        
    Returns:
        StatisticalTestResult with t-statistic, p-value, Cohen's d.
    """
    assert len(projections_chat) == len(projections_agent), \
        "Must have paired observations (same prompts in both formats)"

    # Paired t-test, one-sided (chat > agent). The p-value must be SIGN-AWARE:
    # halving the two-sided p is only valid when the observed effect points the
    # way H1 predicts. Halving unconditionally reports a small p for an effect
    # running in the WRONG direction, which is how a reversal gets published as
    # a confirmation. Delegated to the canonical implementation.
    from .robust_estimation import paired_cohens_d, paired_ttest_directional

    t_stat, p_value = paired_ttest_directional(
        projections_chat, projections_agent, alternative="greater"
    )

    # Cohen's d_z for paired samples (ddof=1: the sample SD is the estimator,
    # and ddof=0 biases every effect size upward at n=42).
    differences = projections_chat - projections_agent
    d = paired_cohens_d(projections_chat, projections_agent)

    # Confidence interval for the mean difference, at the SAME level as the
    # significance test (previously labelled 95% while being built from alpha).
    n = len(differences)
    se = differences.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    ci_lower = differences.mean() - t_crit * se
    ci_upper = differences.mean() + t_crit * se

    # Interpretation
    if p_value < alpha and d > 0:
        if d > 0.8:
            interp = "LARGE effect: refusal direction is STRONGLY reduced in agent format"
        elif d > 0.5:
            interp = "MEDIUM effect: refusal direction is moderately reduced in agent format"
        else:
            interp = "SMALL effect: refusal direction is slightly reduced in agent format"
    elif p_value >= alpha:
        interp = "NO significant difference: refusal direction activates similarly in both formats"
    else:
        interp = "Unexpected: agent projections are HIGHER than chat (direction activates MORE)"
    
    return StatisticalTestResult(
        test_name="Paired t-test (one-sided: chat > agent)",
        statistic=float(t_stat),
        p_value=float(p_value),
        effect_size=float(d),
        effect_size_name="Cohen's d_z (paired, ddof=1)",
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        significant=bool(p_value < alpha),
        interpretation=interp,
    )


def permutation_test(
    projections_chat: np.ndarray,
    projections_agent: np.ndarray,
    n_permutations: int = 10000,
    random_seed: int = 42,
) -> StatisticalTestResult:
    """
    Permutation test for robustness (no distributional assumptions).
    
    Randomly swaps chat/agent labels and measures how extreme
    the observed ΔP is compared to the null distribution.
    
    Args:
        projections_chat: Chat format projections.
        projections_agent: Agent format projections (paired).
        n_permutations: Number of permutations.
        random_seed: Random seed for reproducibility.
        
    Returns:
        StatisticalTestResult with empirical p-value.
    """
    rng = np.random.default_rng(random_seed)

    observed_diff = projections_chat.mean() - projections_agent.mean()
    n = len(projections_chat)

    # Swapping the two labels within a pair negates that pair's difference, so
    # the whole null is one (n_permutations x n) sign matrix. Vectorised: the
    # loop below used to dominate the runtime of every per-layer analysis.
    diff = np.asarray(projections_chat, dtype=float) - np.asarray(
        projections_agent, dtype=float
    )
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, n))
    null_diffs = (signs * diff).mean(axis=1)

    # Empirical p-value (one-sided). A value of 0.0 means p < 1/n_permutations,
    # never exactly zero.
    p_value = (null_diffs >= observed_diff).mean()

    # Effect size relative to null distribution
    null_sd = null_diffs.std(ddof=1)
    effect_size = (observed_diff - null_diffs.mean()) / null_sd if null_sd > 1e-12 else 0.0

    return StatisticalTestResult(
        test_name=f"Permutation test ({n_permutations} permutations)",
        statistic=float(observed_diff),
        p_value=float(p_value),
        effect_size=float(effect_size),
        effect_size_name="Z-score (vs null distribution)",
        ci_lower=float(np.percentile(null_diffs, 2.5)),
        ci_upper=float(np.percentile(null_diffs, 97.5)),
        significant=bool(p_value < 0.01),
        interpretation=f"Observed ΔP={observed_diff:.4f} vs null 95% CI [{np.percentile(null_diffs, 2.5):.4f}, {np.percentile(null_diffs, 97.5):.4f}]",
    )


def bootstrap_ci(
    projections_chat: np.ndarray,
    projections_agent: np.ndarray,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    random_seed: int = 42,
) -> Dict[str, float]:
    """
    Bootstrap confidence interval for ΔP — the headline uncertainty estimate.

    ΔP is a direct measurement, not a trained classifier, so it does not suffer
    the overfitting problem that makes an in-sample probe AUROC uninformative.
    What it does need is an interval: a point estimate cannot distinguish a real
    gap from noise. The decision rule is whether the interval excludes zero, and
    `excludes_zero` in the return value states that verdict explicitly rather
    than leaving it to be eyeballed off two numbers.

    Resampling is PAIRED: one index vector is drawn and applied to BOTH
    conditions, so a prompt's chat and agent observations always travel
    together. Drawing independent indices per condition would estimate the
    variance of a between-groups difference and yield an interval wider than
    this design warrants.

    Both interval types are returned. The percentile interval is the headline
    (`ci_lower`/`ci_upper`, unchanged keys). The BCa interval corrects for bias
    and skew in the bootstrap distribution and is the more accurate one at
    n ≈ 42; when the two disagree materially, report BCa and say so.

    Args:
        projections_chat: Chat projections.
        projections_agent: Agent projections (paired, row-aligned).
        n_bootstrap: Number of bootstrap resamples. Default 10,000, which
            resolves a 95% interval to ~0.5% of the tail mass.
        ci_level: Confidence level.
        random_seed: Random seed.

    Returns:
        Dict with the point estimate, percentile and BCa intervals, the
        bootstrap SE, a two-sided bootstrap p-value, and `excludes_zero`.
    """
    from .robust_estimation import paired_bootstrap_ci

    a = np.asarray(projections_chat, dtype=float)
    b = np.asarray(projections_agent, dtype=float)
    if len(a) != len(b):
        raise ValueError(
            "bootstrap_ci requires paired inputs; got "
            f"{len(a)} chat vs {len(b)} agent observations"
        )

    res = paired_bootstrap_ci(
        a, b, n_boot=n_bootstrap, ci_level=ci_level, seed=random_seed, bca=True
    )

    return {
        # Observed ΔP, not the bootstrap mean: the bootstrap distribution is
        # centred on the observed value and its mean only adds resampling noise.
        "mean_delta_p": float(a.mean() - b.mean()),
        "bootstrap_mean": res["mean"],
        "ci_lower": res["ci_lower"],
        "ci_upper": res["ci_upper"],
        "ci_lower_bca": res["ci_lower_bca"],
        "ci_upper_bca": res["ci_upper_bca"],
        "std": res["std"],                       # bootstrap SE of ΔP
        "excludes_zero": res["excludes_zero"],
        "excludes_zero_bca": res["excludes_zero_bca"],
        "p_bootstrap_two_sided": res["p_bootstrap_two_sided"],
        "n_bootstrap": n_bootstrap,
        "ci_level": ci_level,
        "n_pairs": int(len(a)),
    }


def compute_auroc_by_format(
    projections: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Compute AUROC for harmful vs benign separation in each format.
    
    Args:
        projections: Dict with keys 'chat_harmful', 'chat_benign',
                     'agent_harmful', 'agent_benign'
                     
    Returns:
        Dict with AUROC for each format.
    """
    def _auroc(harmful_key: str, benign_key: str) -> float:
        h, b = projections[harmful_key], projections[benign_key]
        labels = np.concatenate([np.ones(len(h)), np.zeros(len(b))])
        return float(roc_auc_score(labels, np.concatenate([h, b])))

    # In --decompose mode the agent variants are renamed to agent_full_*.
    # Indexing the bare name raised KeyError and aborted the run at the
    # reporting step, after the expensive extraction had already completed.
    agent_h = "agent_harmful" if "agent_harmful" in projections else "agent_full_harmful"
    agent_b = "agent_benign" if "agent_benign" in projections else "agent_full_benign"

    missing = [k for k in ("chat_harmful", "chat_benign", agent_h, agent_b)
               if k not in projections]
    if missing:
        raise KeyError(
            f"compute_auroc_by_format is missing {missing}; "
            f"available conditions: {sorted(projections)}"
        )

    results = {
        "auroc_chat": _auroc("chat_harmful", "chat_benign"),
        "auroc_agent": _auroc(agent_h, agent_b),
    }

    # Gap
    results["auroc_gap"] = results["auroc_chat"] - results["auroc_agent"]
    
    return results


def behavior_activation_concordance(
    projections: np.ndarray,
    behaviors: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Measure concordance between activation projections and model behavior.
    
    Does high projection (direction active) predict refusal?
    Does low projection (direction inactive) predict compliance?
    
    Args:
        projections: Projection values for each sample.
        behaviors: 1 = model refused, 0 = model complied.
        threshold: Projection threshold for "direction active".
        
    Returns:
        Dict with concordance metrics.
    """
    predicted_refuse = (projections > threshold).astype(int)
    
    # Concordance = how often projection predicts behavior
    concordance = (predicted_refuse == behaviors).mean()
    
    # When model complied with harmful request, was projection low?
    complied_mask = behaviors == 0
    if complied_mask.any():
        mean_proj_when_complied = projections[complied_mask].mean()
    else:
        mean_proj_when_complied = None
    
    # When model refused, was projection high?
    refused_mask = behaviors == 1
    if refused_mask.any():
        mean_proj_when_refused = projections[refused_mask].mean()
    else:
        mean_proj_when_refused = None
    
    # Point-biserial correlation
    if len(np.unique(behaviors)) > 1:
        correlation, corr_p = stats.pointbiserialr(behaviors, projections)
    else:
        correlation, corr_p = 0.0, 1.0
    
    return {
        "concordance": float(concordance),
        "mean_proj_when_complied": float(mean_proj_when_complied) if mean_proj_when_complied else None,
        "mean_proj_when_refused": float(mean_proj_when_refused) if mean_proj_when_refused else None,
        "point_biserial_r": float(correlation),
        "point_biserial_p": float(corr_p),
    }


def cross_tool_analysis(
    projections_by_tool: Dict[str, Dict[str, np.ndarray]],
    refusal_direction: np.ndarray,
) -> Dict[str, Dict]:
    """
    Analyze if the projection gap is consistent across tools.
    
    Args:
        projections_by_tool: Dict mapping tool_name -> {
            'chat_harmful': projections, 'agent_harmful': projections
        }
        refusal_direction: The refusal direction used.
        
    Returns:
        Per-tool ΔP and overall consistency metrics.
    """
    gaps = {}
    for tool_name, projs in projections_by_tool.items():
        chat_mean = projs["chat_harmful"].mean()
        agent_mean = projs["agent_harmful"].mean()
        gaps[tool_name] = {
            "delta_p": float(chat_mean - agent_mean),
            "chat_mean": float(chat_mean),
            "agent_mean": float(agent_mean),
            "n_samples": len(projs["chat_harmful"]),
        }
    
    # Overall consistency
    all_gaps = [g["delta_p"] for g in gaps.values()]
    
    return {
        "per_tool": gaps,
        "mean_gap": float(np.mean(all_gaps)),
        "std_gap": float(np.std(all_gaps)),
        "cv": float(np.std(all_gaps) / abs(np.mean(all_gaps))) if np.mean(all_gaps) != 0 else float('inf'),
        "consistent": bool(np.std(all_gaps) / abs(np.mean(all_gaps)) < 0.5) if np.mean(all_gaps) != 0 else False,
    }


def print_full_report(
    ttest_result: StatisticalTestResult,
    perm_result: StatisticalTestResult,
    bootstrap: Dict[str, float],
    auroc: Dict[str, float],
    auroc_heldout: Optional[Dict[str, float]] = None,
    benign: Optional[Dict[str, float]] = None,
):
    """
    Print comprehensive statistical report.

    Args:
        auroc: in-sample AUROC by format, i.e. scored with a direction fitted
            on these same prompts. Reported for comparison only.
        auroc_heldout: the same quantities scored out-of-fold, with keys
            'auroc_chat', 'auroc_agent' and optionally 'auroc_gap'. When
            present these are the figures the conclusion is drawn from.
        benign: the benign-arm control, with keys 'delta_p', 'p_value',
            'interaction', 'interaction_p' and 'common_shift_fraction'. DeltaP
            on the harmful arm alone cannot distinguish a weakened refusal
            representation from a rigid displacement of every representation,
            so the conclusion below is withheld when the control is absent.
    """
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS REPORT")
    print("=" * 70)
    
    print("\n--- PRIMARY METRIC: Projection Gap (ΔP) ---")
    n_boot = bootstrap.get("n_bootstrap", "?")
    level = int(100 * bootstrap.get("ci_level", 0.95))
    print(f"ΔP = {bootstrap['mean_delta_p']:.4f}   "
          f"(paired bootstrap, {n_boot} resamples, n = {bootstrap.get('n_pairs', '?')} pairs)")
    print(f"{level}% CI (percentile): [{bootstrap['ci_lower']:.4f}, "
          f"{bootstrap['ci_upper']:.4f}]")
    if bootstrap.get("ci_lower_bca") is not None and np.isfinite(bootstrap.get("ci_lower_bca", np.nan)):
        print(f"{level}% CI (BCa)       : [{bootstrap['ci_lower_bca']:.4f}, "
              f"{bootstrap['ci_upper_bca']:.4f}]   <- bias/skew corrected")
    if "excludes_zero" in bootstrap:
        verdict = ("EXCLUDES zero -> the gap is significant"
                   if bootstrap["excludes_zero"]
                   else "CONTAINS zero -> the gap is NOT distinguishable from noise")
        print(f"Interval {verdict}")
    if "p_bootstrap_two_sided" in bootstrap:
        print(f"Bootstrap two-sided p: {bootstrap['p_bootstrap_two_sided']:.5f}")
    print("ΔP is a direct measurement, not a fitted classifier, so it is not")
    print("subject to the probe's overfitting critique.")

    print(f"\n--- PAIRED T-TEST ---")
    print(f"t = {ttest_result.statistic:.4f}")
    print(f"p = {ttest_result.p_value:.6f}")
    print(f"Cohen's d = {ttest_result.effect_size:.4f}")
    print(f"Significant (α=0.01): {ttest_result.significant}")
    print(f"Interpretation: {ttest_result.interpretation}")
    
    print(f"\n--- PERMUTATION TEST ---")
    print(f"Observed ΔP = {perm_result.statistic:.4f}")
    print(f"Empirical p = {perm_result.p_value:.6f}")
    print(f"Null 95% CI: [{perm_result.ci_lower:.4f}, {perm_result.ci_upper:.4f}]")
    print(f"Significant: {perm_result.significant}")
    
    if benign is not None:
        print(f"\n--- BENIGN CONTROL (difference-in-differences) ---")
        print(f"ΔP harmful (headline): {bootstrap['mean_delta_p']:+.4f}")
        print(f"ΔP benign  (control) : {benign['delta_p']:+.4f}   "
              f"p = {benign['p_value']:.6f}")
        print(f"Interaction          : {benign['interaction']:+.4f}   "
              f"p = {benign['interaction_p']:.6f}")
        print(f"Share of the harmful shift also seen on benign prompts: "
              f"{benign['common_shift_fraction']:+.1%}")
        print("Only the interaction is specific to refusal. The rest is what")
        print("moving from chat to agent does to EVERY prompt.")

    print(f"\n--- AUROC BY FORMAT ---")
    if auroc_heldout is not None:
        # The held-out figures come first because they are the ones that mean
        # what AUROC is normally taken to mean. The in-sample pair is printed
        # beside them so the size of the optimism stays visible.
        print(f"{'':<16}{'held-out':>12}{'in-sample':>12}")
        print(f"{'AUROC (chat)':<16}{auroc_heldout['auroc_chat']:>12.4f}"
              f"{auroc['auroc_chat']:>12.4f}")
        print(f"{'AUROC (agent)':<16}{auroc_heldout['auroc_agent']:>12.4f}"
              f"{auroc['auroc_agent']:>12.4f}")
        gap_ho = auroc_heldout.get(
            "auroc_gap",
            auroc_heldout["auroc_chat"] - auroc_heldout["auroc_agent"],
        )
        print(f"{'AUROC gap':<16}{gap_ho:>12.4f}{auroc['auroc_gap']:>12.4f}")
        print("Held-out = each prompt scored by a direction fitted without it.")
        print("In-sample = the direction was fitted on these same prompts; that")
        print("column is a fit statistic, not a detection rate.")
    else:
        print(f"AUROC (chat):  {auroc['auroc_chat']:.4f}")
        print(f"AUROC (agent): {auroc['auroc_agent']:.4f}")
        print(f"AUROC gap:     {auroc['auroc_gap']:.4f}")
        print("IN-SAMPLE: d_chat was fitted on these prompts, so these values")
        print("are optimistic. Run the held-out evaluation before quoting them.")

    print("\n" + "=" * 70)
    shift_is_significant = ttest_result.significant and perm_result.significant
    if benign is not None and shift_is_significant and benign["interaction_p"] >= 0.05:
        # The harmful arm moved, but so did the benign arm by as much. That is
        # a change of origin, not a weakened refusal representation, and the
        # old wording ("behaves differently in agentic format") would have
        # reported a rigid translation as the thesis's finding.
        print("CONCLUSION: the projection shifts between formats, but NOT more")
        print("for harmful prompts than for benign ones (interaction "
              f"p = {benign['interaction_p']:.4f}).")
        print("→ The agentic format DISPLACES the representation; it does not")
        print("  selectively weaken refusal. Do not report ΔP as loss of safety.")
    elif shift_is_significant:
        print("CONCLUSION: STRONG EVIDENCE that refusal direction behaves")
        print("differently in agentic format.")
        if benign is not None:
            print(f"The benign control rules out a common displacement "
                  f"(interaction p = {benign['interaction_p']:.4f}).")
    elif ttest_result.significant or perm_result.significant:
        print("CONCLUSION: MODERATE EVIDENCE of format effect.")
        print("Tests disagree — investigate further.")
    else:
        print("CONCLUSION: NO significant evidence of format effect.")
        print("The refusal direction appears to activate similarly in both formats.")
        print("→ If model still complies in agent format, the problem is DOWNSTREAM.")
    print("=" * 70)


def _compute_single_effect(
    proj_a: np.ndarray,
    proj_b: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> Dict:
    """
    Compute a single pairwise effect with statistical tests.

    Args:
        proj_a: Projections for condition A (higher expected).
        proj_b: Projections for condition B (lower expected).
        n_permutations: Permutation count.
        seed: Random seed.

    Returns:
        Dict with delta_p, p_value, cohens_d, ci_lower, ci_upper, mean_a, mean_b.
    """
    # The decomposition contrasts the SAME prompts under adjacent conditions,
    # so every step of this chain is paired. The previous implementation shuffled
    # the concatenated samples (a null that assumes exchangeability ACROSS
    # prompts) and bootstrapped each condition with independent indices (which
    # estimates the variance of a between-groups difference). Both discard the
    # within-pair correlation the design was built to exploit, widening the
    # intervals and costing power. Paired estimators throughout.
    from .robust_estimation import (
        paired_bootstrap_ci,
        paired_cohens_d,
        paired_permutation_test,
        pooled_cohens_d,
    )

    proj_a = np.asarray(proj_a, dtype=float)
    proj_b = np.asarray(proj_b, dtype=float)
    if len(proj_a) != len(proj_b):
        raise ValueError(
            "decomposition steps must be paired; got "
            f"{len(proj_a)} vs {len(proj_b)} observations"
        )

    delta_p = float(proj_a.mean() - proj_b.mean())

    perm = paired_permutation_test(
        proj_a, proj_b, n_permutations=n_permutations,
        alternative="two-sided", seed=seed,
    )
    boot = paired_bootstrap_ci(proj_a, proj_b, n_boot=10000, seed=seed)

    return {
        "delta_p": delta_p,
        "mean_a": float(proj_a.mean()),
        "mean_b": float(proj_b.mean()),
        "p_value": perm["p_value"],
        "p_value_floor": perm["p_value_floor"],
        "cohens_d": pooled_cohens_d(proj_a, proj_b),
        "cohens_d_paired": paired_cohens_d(proj_a, proj_b),
        "ci_lower": boot["ci_lower"],
        "ci_upper": boot["ci_upper"],
        "significant": bool(perm["p_value"] < 0.05),
    }


def decompose_format_effects(
    projections: Dict[str, np.ndarray],
    n_permutations: int = 10000,
    seed: int = 42,
) -> Dict[str, Dict]:
    """
    Decompose the total format effect into 3 components.

    Requires projections for all 4 conditions:
    - chat_harmful
    - role_only_harmful
    - role_plus_tools_harmful
    - agent_full_harmful

    Computes:
    - role_effect = mean(chat) - mean(role_only)
    - tools_effect = mean(role_only) - mean(role_plus_tools)
    - json_format_effect = mean(role_plus_tools) - mean(agent_full)
    - total_effect = mean(chat) - mean(agent_full)

    Each effect includes: delta_p, p_value, cohens_d, ci_lower, ci_upper.

    Args:
        projections: Dict with keys 'chat_harmful', 'role_only_harmful',
                     'role_plus_tools_harmful', 'agent_full_harmful'.
        n_permutations: Permutations for significance tests.
        seed: Random seed.

    Returns:
        Dict with keys 'role_effect', 'tools_effect', 'json_format_effect',
        'total_effect', each containing statistical measures.
    """
    chat = projections["chat_harmful"]
    role_only = projections["role_only_harmful"]
    role_plus_tools = projections["role_plus_tools_harmful"]
    agent_full = projections["agent_full_harmful"]

    return {
        "role_effect": _compute_single_effect(
            chat, role_only, n_permutations, seed
        ),
        "tools_effect": _compute_single_effect(
            role_only, role_plus_tools, n_permutations, seed + 1
        ),
        "json_format_effect": _compute_single_effect(
            role_plus_tools, agent_full, n_permutations, seed + 2
        ),
        "total_effect": _compute_single_effect(
            chat, agent_full, n_permutations, seed + 3
        ),
    }
