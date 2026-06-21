"""
Validation module for the Agentic Safety Probe experiment.

Separates validation logic by dataset type:

1. Activation-space validation (custom paired dataset):
   - Uses paired prompts (same text in chat/agent)
   - Measures refusal direction projection gap (ΔP)
   - Statistical tests: permutation, t-test, bootstrap CI
   - PCA validation of direction extraction

2. Behavioral validation (HarmAgent datasets):
   - Uses grading functions and target_functions
   - Measures refusal rate, compliance rate, ASR (attack success rate)
   - Validates that activation patterns correspond to actual behavior

Both validation types can be combined to check concordance between
internal representations and external behavior.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ActivationValidationResult:
    """Result of activation-space validation on paired dataset."""
    layer: int
    delta_p: float
    p_value_permutation: float
    p_value_ttest: float
    cohens_d: float
    ci_lower: float
    ci_upper: float
    pca_cosine_similarity: float
    pca_explained_variance: float
    is_significant: bool
    n_samples: int


@dataclass
class BehavioralValidationResult:
    """Result of behavioral validation on HarmAgent dataset."""
    condition: str          # "agent" or "chat"
    split: str             # "test" or "validation"
    n_total: int
    n_refused: int
    n_complied: int
    refusal_rate: float
    compliance_rate: float
    # Per-category breakdown
    category_refusal_rates: Dict[str, float]


def validate_activations_paired(
    chat_projections: np.ndarray,
    agent_projections: np.ndarray,
    layer: int,
    n_permutations: int = 10000,
    seed: int = 42,
) -> ActivationValidationResult:
    """
    Validate activation-space difference using the PAIRED custom dataset.

    This is the primary validation for the research question:
    "Does agentic context reduce refusal direction activation?"

    Uses paired data where SAME prompt appears in both conditions.

    Args:
        chat_projections: Projections of chat_harmful onto refusal direction.
        agent_projections: Projections of agent_harmful onto refusal direction.
        layer: Layer index.
        n_permutations: For permutation test.
        seed: Random seed.

    Returns:
        ActivationValidationResult with all metrics.
    """
    from scipy import stats

    observed_gap = float(chat_projections.mean() - agent_projections.mean())

    # Permutation test
    rng = np.random.default_rng(seed)
    combined = np.concatenate([chat_projections, agent_projections])
    n_chat = len(chat_projections)

    null_gaps = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(combined)
        null_gaps[i] = perm[:n_chat].mean() - perm[n_chat:].mean()

    p_perm = float((np.abs(null_gaps) >= np.abs(observed_gap)).mean())

    # Welch's t-test
    t_stat, p_ttest = stats.ttest_ind(chat_projections, agent_projections, equal_var=False)

    # Cohen's d
    pooled_std = np.sqrt(
        (chat_projections.std(ddof=1) ** 2 + agent_projections.std(ddof=1) ** 2) / 2
    )
    cohens_d = observed_gap / pooled_std if pooled_std > 1e-10 else 0.0

    # Bootstrap CI
    n_boot = 1000
    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx_c = rng.integers(0, len(chat_projections), size=len(chat_projections))
        idx_a = rng.integers(0, len(agent_projections), size=len(agent_projections))
        boot_diffs[i] = chat_projections[idx_c].mean() - agent_projections[idx_a].mean()

    return ActivationValidationResult(
        layer=layer,
        delta_p=observed_gap,
        p_value_permutation=p_perm,
        p_value_ttest=float(p_ttest),
        cohens_d=float(cohens_d),
        ci_lower=float(np.percentile(boot_diffs, 2.5)),
        ci_upper=float(np.percentile(boot_diffs, 97.5)),
        pca_cosine_similarity=0.0,  # filled externally
        pca_explained_variance=0.0,  # filled externally
        is_significant=p_perm < 0.05,
        n_samples=len(chat_projections),
    )


def validate_behavior(
    predictions: List[bool],
    entries: List,
    condition: str = "agent",
    split: str = "test",
) -> BehavioralValidationResult:
    """
    Validate model behavior using HarmAgent grading criteria.

    This checks whether the model actually refuses/complies,
    independently of activation patterns.

    Args:
        predictions: List of booleans — True if model refused, False if complied.
        entries: Corresponding HarmAgentEntry list.
        condition: "agent" or "chat".
        split: "test" or "validation".

    Returns:
        BehavioralValidationResult.
    """
    from collections import Counter

    n_total = len(predictions)
    n_refused = sum(predictions)
    n_complied = n_total - n_refused

    # Per-category refusal rates
    category_results = {}
    category_entries = {}
    for pred, entry in zip(predictions, entries):
        cat = entry.category
        if cat not in category_entries:
            category_entries[cat] = []
        category_entries[cat].append(pred)

    for cat, preds in category_entries.items():
        category_results[cat] = sum(preds) / len(preds) if preds else 0.0

    return BehavioralValidationResult(
        condition=condition,
        split=split,
        n_total=n_total,
        n_refused=n_refused,
        n_complied=n_complied,
        refusal_rate=n_refused / n_total if n_total > 0 else 0.0,
        compliance_rate=n_complied / n_total if n_total > 0 else 0.0,
        category_refusal_rates=category_results,
    )


def validate_concordance(
    projections: np.ndarray,
    refused: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Check concordance between activation projections and actual behavior.

    High projection (refusal direction active) should predict refusal.
    Low projection should predict compliance.

    This bridges the two validation approaches.

    Args:
        projections: Refusal direction projections.
        refused: Binary array — 1 = refused, 0 = complied.
        threshold: Projection threshold. None = use median.

    Returns:
        Dict with concordance metrics.
    """
    from scipy import stats

    if threshold is None:
        threshold = float(np.median(projections))

    predicted_refuse = (projections > threshold).astype(int)
    concordance = float((predicted_refuse == refused).mean())

    # Point-biserial correlation
    if len(np.unique(refused)) > 1:
        corr, p_val = stats.pointbiserialr(refused, projections)
    else:
        corr, p_val = 0.0, 1.0

    return {
        "concordance": concordance,
        "threshold": threshold,
        "correlation": float(corr),
        "correlation_p_value": float(p_val),
        "mean_proj_refused": float(projections[refused == 1].mean()) if refused.sum() > 0 else None,
        "mean_proj_complied": float(projections[refused == 0].mean()) if (refused == 0).sum() > 0 else None,
    }
