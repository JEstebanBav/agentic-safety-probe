"""
Robust (bias-corrected) estimators for the agentic safety probe.

This module addresses three inference problems in the original pipeline that
inflate the reported quantities. Each corrected estimator is paired with the
naive one so the magnitude of the bias can be reported, not just removed.

PROBLEM 1 — Circularity in DeltaP (Observation 2)
    d_chat is estimated from chat_harmful, and chat_harmful is then projected
    onto it. In the d >> n regime the noise in the sample centroid enters the
    direction and is recovered on projection, biasing the chat term upward.
    The agent term carries no such bias, so DeltaP is inflated.
    FIX: cross-fitting. Estimate d on K-1 folds, project the held-out fold.

PROBLEM 2 — No train/test split for w_agent and the monitor (Observation 3)
    With N=84 in d=8192, Cover's theorem guarantees linear separability for
    ANY labelling. In-sample AUROC = 1.0 is a geometric certainty, not
    evidence. Threshold calibration on the evaluation set has the same defect.
    FIX: stratified K-fold. Fit and calibrate on train, score on test.

PROBLEM 3 — Unquantified attenuation in the rotation angle (Observation 9)
    cos(w_agent, d_chat) is attenuated by estimation noise in w_agent, so the
    reported angle is an upper bound. Without a reliability estimate the
    magnitude cannot be interpreted.
    FIX: report inter-fold cos(w_i, w_j) as the reliability ceiling.

PROBLEM 4 — AUROC scored on the prompts that DEFINED the direction
    d_chat is the difference between the chat_harmful and chat_benign
    centroids, and the reported AUROC then asks how well that same difference
    separates those same two groups. The direction absorbs whatever separates
    THIS sample, noise included, so the number is an in-sample fit statistic
    rather than a detection rate — and it cannot be compared honestly against
    the agent format, which never entered the estimation.
    FIX: cross-fitted projections by condition. Folds are assigned over PROMPT
    PAIRS, the direction is fitted on the training pairs only, and every
    prompt is scored by a direction that never saw it — nor saw its benign
    counterpart, because a pair travels to its fold whole. See
    `crossfit_condition_projections` and `held_out_detection`.

The module also provides single canonical implementations of the paired
statistics, replacing the inconsistent variants in metrics.py,
refusal_direction.py and stratified_analysis.py (Observations 5, 6, 7, 8).

All estimators preserve the paired structure of the design: fold assignment
and bootstrap resampling operate on PROMPT indices, so a prompt's chat and
agent observations always travel together.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# Minimum split-half reliability at which the attenuation correction is
# reported. The correction divides by sqrt(reliability), so this floor caps
# the noise amplification at 2x. Below it the corrected angle is not
# interpretable and the estimators return nan rather than a plausible-looking
# artefact. Value chosen so that the correction never inflates by more than a
# factor of two, matching the conventional psychometric guidance that
# correction for attenuation becomes unstable at low reliability.
MIN_RELIABILITY_FOR_CORRECTION = 0.25


# ============================================================
# RESULT CONTAINERS
# ============================================================

@dataclass
class PairedTestResult:
    """Outcome of the paired inference suite on one pair of projection vectors."""
    n_pairs: int
    delta_p: float
    mean_a: float
    mean_b: float
    # Paired t-test (sign-aware one-sided)
    t_statistic: float
    p_value_ttest: float
    alternative: str
    # Sign-flip permutation test
    p_value_permutation: float
    permutation_z: float
    n_permutations: int
    p_value_floor: float          # smallest p this permutation budget can express
    # Effect sizes (both, ddof=1)
    cohens_d_paired: float
    cohens_d_pooled: float
    implied_pair_correlation: float
    # Paired bootstrap
    ci_lower: float
    ci_upper: float
    # Distribution-free sign summary
    n_decreasing: int
    sign_test_p: float


@dataclass
class CrossFittedGapResult:
    """Cross-fitted DeltaP with the naive estimate retained for comparison."""
    layer_idx: int
    n_folds: int
    delta_p_naive: float
    delta_p_crossfit: float
    bias_estimate: float          # naive - crossfit
    bias_ratio: float             # bias / |crossfit|, inf when crossfit ~ 0
    naive_stats: PairedTestResult
    crossfit_stats: PairedTestResult
    proj_chat_oof: np.ndarray = field(repr=False, default=None)
    proj_agent_oof: np.ndarray = field(repr=False, default=None)


@dataclass
class CrossValidatedDirectionResult:
    """Out-of-fold evaluation of w_agent plus its reliability ceiling."""
    layer_idx: int
    n_folds: int
    auroc_in_sample: float        # the inflated number the old pipeline reported
    auroc_oof_mean: float
    auroc_oof_std: float
    auroc_oof_folds: List[float]
    # Alignment with d_chat
    cosine_with_d_chat_mean: float
    cosine_with_d_chat_std: float
    angle_deg_mean: float
    # Reliability from DISJOINT split-halves (see note in the estimator)
    split_half_reliability: float
    split_half_reliability_std: float
    attenuation_factor: float          # sqrt(reliability); 1.0 = no attenuation
    cosine_disattenuated: float        # measured / attenuation_factor, clipped
    angle_deg_disattenuated: float
    # Inter-fold cosine, retained for diagnosis only: it is INFLATED because
    # K-fold training sets overlap, so it must not be used as the ceiling.
    inter_fold_cosine_mean: float
    inter_fold_cosine_std: float
    # Chance scale in this dimension
    chance_cosine_sd: float
    cosine_in_chance_sd: float
    w_agent_full: np.ndarray = field(repr=False, default=None)


@dataclass
class CrossValidatedMonitorResult:
    """Out-of-fold monitor metrics with the threshold calibrated on train only."""
    layer_idx: int
    n_folds: int
    precision_oof: float
    recall_oof: float
    f1_oof: float
    fpr_oof: float
    thresholds_per_fold: List[float]
    f1_in_sample: float           # the inflated number the old pipeline reported
    n_harmful: int
    n_benign: int


# ============================================================
# 1. CANONICAL PAIRED STATISTICS
#    Single implementation; replaces the divergent copies elsewhere.
# ============================================================

def paired_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Standardised mean difference for paired samples (Cohen's d_z).

    d_z = mean(a - b) / sd(a - b), with sd using ddof=1.

    Note this is NOT interchangeable with the pooled variant; d_z answers
    "how consistent is the within-pair change", not "how separated are the
    two distributions". Cohen's conventional benchmarks were set for the
    latter. Report both.
    """
    diff = np.asarray(a) - np.asarray(b)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 1e-12 else 0.0


def pooled_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised mean difference treating the samples as independent (d_av)."""
    a, b = np.asarray(a), np.asarray(b)
    pooled_sd = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return float((a.mean() - b.mean()) / pooled_sd) if pooled_sd > 1e-12 else 0.0


def implied_pair_correlation(d_paired: float, d_pooled: float) -> float:
    """
    Recover the within-pair correlation implied by the two effect sizes.

    From d_paired = d_pooled / sqrt(2(1-r)):
        r = 1 - (1/2)(d_pooled / d_paired)^2

    A high r confirms the pairing did its job. Returns nan when d_paired is
    ~0, where the relation is undefined.
    """
    if abs(d_paired) < 1e-12:
        return float("nan")
    return float(1.0 - 0.5 * (d_pooled / d_paired) ** 2)


def paired_ttest_directional(
    a: np.ndarray,
    b: np.ndarray,
    alternative: str = "greater",
) -> Tuple[float, float]:
    """
    Paired t-test with a SIGN-AWARE one-sided p-value (Observation 8).

    The original code computed p_one_sided = p_two_sided / 2 unconditionally.
    That identity only holds when the observed effect points the way H1
    predicts; otherwise the correct value is 1 - p_two_sided/2, and the naive
    halving reports a small p for an effect in the WRONG direction.

    Args:
        a, b: paired observations. alternative='greater' tests mean(a) > mean(b).

    Returns:
        (t_statistic, one_sided_p_value)
    """
    if alternative not in ("greater", "less", "two-sided"):
        raise ValueError(f"alternative must be greater/less/two-sided, got {alternative}")

    t_stat, p_two = stats.ttest_rel(a, b)
    t_stat = float(t_stat)

    if alternative == "two-sided":
        return t_stat, float(p_two)

    # Halve only when the sign agrees with the alternative; otherwise complement.
    sign_agrees = (t_stat > 0) if alternative == "greater" else (t_stat < 0)
    p_one = p_two / 2.0 if sign_agrees else 1.0 - p_two / 2.0
    return t_stat, float(p_one)


def paired_permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    n_permutations: int = 10000,
    alternative: str = "greater",
    seed: int = 42,
) -> Dict[str, float]:
    """
    Sign-flip permutation test for paired data (Observation 5).

    Under H0 the two condition labels are exchangeable WITHIN each pair, so
    the null is generated by independently flipping each pair's sign. This is
    the correct null for this design. A global shuffle of the concatenated
    samples — used elsewhere in the original pipeline — assumes exchangeability
    ACROSS units, which the paired design violates; it discards the within-pair
    correlation and loses power.

    Reports p_value_floor = 1/n_permutations. An empirical p of 0.0 must be
    read as "p < floor", never as exactly zero.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("paired test requires equal-length inputs")

    rng = np.random.default_rng(seed)
    diff = a - b
    n = len(diff)
    observed = float(diff.mean())

    # Vectorised sign-flip: (n_permutations, n) matrix of +/-1
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, n))
    null = (signs * diff).mean(axis=1)

    if alternative == "greater":
        p = float((null >= observed).mean())
    elif alternative == "less":
        p = float((null <= observed).mean())
    else:
        p = float((np.abs(null) >= abs(observed)).mean())

    null_sd = null.std(ddof=1)
    z = float((observed - null.mean()) / null_sd) if null_sd > 1e-12 else 0.0

    return {
        "observed": observed,
        "p_value": p,
        "z_score": z,
        "null_mean": float(null.mean()),
        "null_sd": float(null_sd),
        "n_permutations": n_permutations,
        "p_value_floor": 1.0 / n_permutations,
    }


def paired_bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
    bca: bool = True,
) -> Dict[str, float]:
    """
    Bootstrap CI for the paired mean difference (Observation 6).

    A SINGLE index vector is drawn and applied to both conditions, preserving
    the pairing. Drawing independent indices per condition — as the stratified
    and decomposition routines did — estimates the variance of a difference
    between independent groups instead, yielding intervals wider than the
    design warrants.

    Two intervals are returned:

      PERCENTILE  the 2.5th and 97.5th percentiles of the resampled statistic.
                  Reported as ci_lower/ci_upper, unchanged from before.

      BCa         bias-corrected and accelerated. The percentile interval is
                  only first-order accurate and mis-covers when the bootstrap
                  distribution is biased or skewed, which is the common case at
                  n ~ 40. BCa shifts the percentiles by z0 (median bias) and
                  corrects for a variance that changes with the estimate
                  (acceleration, from the jackknife). It is second-order
                  accurate and transformation-respecting.

    `excludes_zero` states the decision the interval exists to support, so no
    caller has to re-derive it from two floats.

    Default n_boot is 10,000: with 1,000 resamples each 95% tail contains only
    ~25 values, so the endpoints carry visible Monte-Carlo noise and shift
    between seeds.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("paired bootstrap requires equal-length inputs")

    rng = np.random.default_rng(seed)
    n = len(a)
    diff = a - b                                     # the paired statistic
    observed = float(diff.mean())

    idx = rng.integers(0, n, size=(n_boot, n))       # shared across conditions
    boot = diff[idx].mean(axis=1)

    alpha = 1.0 - ci_level
    lo_pct, hi_pct = 100 * alpha / 2, 100 * (1 - alpha / 2)
    ci_lower = float(np.percentile(boot, lo_pct))
    ci_upper = float(np.percentile(boot, hi_pct))

    # Two-sided bootstrap p-value: how much of the resampled mass sits on the
    # far side of zero, doubled. Floored at 1/n_boot — a value of 0 means
    # "below the resolution of this budget", never exactly zero.
    prop_below = float((boot <= 0.0).mean())
    p_boot = 2.0 * min(prop_below, 1.0 - prop_below)
    p_boot = float(min(1.0, max(p_boot, 1.0 / n_boot)))

    out = {
        "mean": float(boot.mean()),
        "observed": observed,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std": float(boot.std(ddof=1)),
        "excludes_zero": bool(ci_lower > 0.0 or ci_upper < 0.0),
        "p_bootstrap_two_sided": p_boot,
        "n_boot": int(n_boot),
        "ci_level": float(ci_level),
        "ci_lower_bca": float("nan"),
        "ci_upper_bca": float("nan"),
        "excludes_zero_bca": None,
        "bias_correction_z0": float("nan"),
        "acceleration": float("nan"),
    }

    if not bca or n < 3:
        return out

    # ---- BCa ----
    # z0: median bias, from the share of resamples below the observed value.
    prop_less = float((boot < observed).mean())
    if prop_less <= 0.0 or prop_less >= 1.0:
        return out                                   # degenerate; percentile only
    z0 = float(stats.norm.ppf(prop_less))

    # Acceleration from the jackknife over PROMPTS (the resampling unit), so
    # the pairing is respected here too. For a mean, the leave-one-out values
    # are available in closed form.
    total = diff.sum()
    jack = (total - diff) / (n - 1)                   # leave-one-out means
    jack_mean = jack.mean()
    dev = jack_mean - jack
    denom = 6.0 * (dev ** 2).sum() ** 1.5
    if denom < 1e-300:
        return out
    acc = float((dev ** 3).sum() / denom)

    z_lo, z_hi = stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)
    adj_lo = z0 + (z0 + z_lo) / (1 - acc * (z0 + z_lo))
    adj_hi = z0 + (z0 + z_hi) / (1 - acc * (z0 + z_hi))
    p_lo = float(np.clip(stats.norm.cdf(adj_lo) * 100, 0.0, 100.0))
    p_hi = float(np.clip(stats.norm.cdf(adj_hi) * 100, 0.0, 100.0))

    bca_lower = float(np.percentile(boot, p_lo))
    bca_upper = float(np.percentile(boot, p_hi))

    out.update({
        "ci_lower_bca": bca_lower,
        "ci_upper_bca": bca_upper,
        "excludes_zero_bca": bool(bca_lower > 0.0 or bca_upper < 0.0),
        "bias_correction_z0": z0,
        "acceleration": acc,
    })
    return out


def paired_sign_test(a: np.ndarray, b: np.ndarray) -> Tuple[int, float]:
    """
    Exact sign test: how many pairs decrease, and how surprising is that.

    Fully distribution-free and immune to every bias discussed in this module,
    because it uses only the sign of each within-pair difference and never the
    magnitude. When the direction of an effect is contested but its magnitude
    is disputed, this is the statistic that survives.

    Returns:
        (n_decreasing, two_sided_exact_binomial_p)
    """
    diff = np.asarray(a) - np.asarray(b)
    nonzero = diff[diff != 0]
    n_dec = int((nonzero > 0).sum())
    n = len(nonzero)
    if n == 0:
        return 0, 1.0
    p = float(stats.binomtest(n_dec, n, 0.5, alternative="two-sided").pvalue)
    return n_dec, p


def run_paired_suite(
    a: np.ndarray,
    b: np.ndarray,
    n_permutations: int = 10000,
    n_boot: int = 10000,
    alternative: str = "greater",
    seed: int = 42,
) -> PairedTestResult:
    """Run the full corrected paired inference suite on two projection vectors."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)

    t_stat, p_t = paired_ttest_directional(a, b, alternative=alternative)
    perm = paired_permutation_test(a, b, n_permutations, alternative, seed)
    boot = paired_bootstrap_ci(a, b, n_boot, 0.95, seed)
    d_par = paired_cohens_d(a, b)
    d_pool = pooled_cohens_d(a, b)
    n_dec, p_sign = paired_sign_test(a, b)

    return PairedTestResult(
        n_pairs=len(a),
        delta_p=float(a.mean() - b.mean()),
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        t_statistic=t_stat,
        p_value_ttest=p_t,
        alternative=alternative,
        p_value_permutation=perm["p_value"],
        permutation_z=perm["z_score"],
        n_permutations=n_permutations,
        p_value_floor=perm["p_value_floor"],
        cohens_d_paired=d_par,
        cohens_d_pooled=d_pool,
        implied_pair_correlation=implied_pair_correlation(d_par, d_pool),
        ci_lower=boot["ci_lower"],
        ci_upper=boot["ci_upper"],
        n_decreasing=n_dec,
        sign_test_p=p_sign,
    )


# ============================================================
# 1b. PAIR-RESPECTING PARTITIONS
# ============================================================
#
# A harmful prompt and its benign counterpart are written from the same seed:
# same structure, same vocabulary, same tools, opposite intent. They are one
# unit of the design, not two observations. Any partition that sends one to
# training and the other to evaluation lets the fitted object see the very
# contrast it is then scored on.
#
# Every estimator in this module used to partition ROWS (StratifiedKFold over
# the 84 stacked rows, or an independent permutation of the benign arm), which
# is not what the methodology states and not what the held-out AUROC of d_chat
# already did. The two helpers below are the single place the partition is
# drawn, so the rule cannot drift between estimators again.

def pair_groups(n_harmful: int, n_benign: int) -> Optional[np.ndarray]:
    """
    Group id for each row of a stacked [harmful; benign] matrix.

    Row i of each arm is the same prompt pair (the pipeline asserts this in
    `_assert_pairing`), so both rows receive group i. Returns None when the
    arms differ in size, in which case no pairing can be assumed and callers
    fall back to a stratified partition over rows.
    """
    if n_harmful != n_benign:
        return None
    return np.concatenate([np.arange(n_harmful), np.arange(n_benign)])


def grouped_fold_indices(
    y: np.ndarray,
    n_folds: int,
    seed: int,
    groups: Optional[np.ndarray] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    (train_idx, test_idx) for K-fold cross-validation.

    With `groups`, whole groups are held out together: the group ids are
    shuffled and cut into K blocks, and a fold tests on every row whose group
    falls in its block. In the balanced design (one harmful and one benign row
    per pair) each fold is therefore class-balanced by construction, so no
    separate stratification is needed.

    Without `groups` this is StratifiedKFold over rows, the previous behaviour.
    """
    y = np.asarray(y)
    if groups is None:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return list(skf.split(np.zeros(len(y)), y))

    groups = np.asarray(groups)
    if len(groups) != len(y):
        raise ValueError(f"groups has {len(groups)} entries for {len(y)} rows.")
    unique = np.unique(groups)
    k = min(n_folds, len(unique))
    if k < 2:
        raise ValueError("Need at least two groups to cross-validate.")
    order = np.random.default_rng(seed).permutation(unique)

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for block in np.array_split(order, k):
        test = np.isin(groups, block)
        folds.append((np.flatnonzero(~test), np.flatnonzero(test)))
    return folds


# ============================================================
# 2. CROSS-FITTED PROJECTION GAP  (Observation 2)
# ============================================================

def _diff_in_means_direction(harmful: np.ndarray, benign: np.ndarray) -> np.ndarray:
    """Normalised difference-in-means direction. Raises on a degenerate contrast."""
    direction = harmful.mean(axis=0) - benign.mean(axis=0)
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        raise ValueError("Degenerate contrast: harmful and benign centroids coincide.")
    return direction / norm


def cross_fitted_projection_gap(
    chat_harmful: np.ndarray,
    chat_benign: np.ndarray,
    agent_harmful: np.ndarray,
    layer_idx: int = -1,
    n_folds: int = 5,
    n_permutations: int = 10000,
    n_boot: int = 10000,
    seed: int = 42,
    paired_arms: bool = True,
) -> CrossFittedGapResult:
    """
    Cross-fitted estimate of DeltaP, removing the own-sample bias.

    The naive pipeline estimates d_chat from chat_harmful and chat_benign, then
    projects chat_harmful onto it. Because d = 8192 >> n = 42, the sample
    centroid mu_H carries a substantial noise component that propagates into
    the direction; projecting mu_H back onto that direction recovers part of
    it. The chat term is therefore inflated. The agent condition never enters
    the estimation, so it carries no matching inflation, and DeltaP absorbs the
    whole difference.

    Cross-fitting breaks the circularity: for each fold k the direction is
    estimated WITHOUT fold k, and only fold k is projected. Every projection is
    out-of-sample with respect to the direction that produced it.

    Pairing is preserved: folds are assigned over PROMPT indices, and a
    prompt's chat and agent activations are always projected onto the same
    held-out direction d_k.

    The benign arm follows the SAME folds when it is matched to the harmful one
    (`paired_arms`, the default, and equal row counts). Folding it
    independently, as this function used to, left the benign twin of a
    held-out harmful prompt in the training set four times out of five, so the
    direction had already seen the contrast it was then scored on.

    Args:
        chat_harmful:  (n, hidden_dim) chat-format harmful activations.
        chat_benign:   (m, hidden_dim) chat-format benign activations.
        agent_harmful: (n, hidden_dim) agent-format harmful activations, paired
                       row-wise with chat_harmful.
        layer_idx:     recorded for reporting only.
        n_folds:       K in K-fold cross-fitting.

    Returns:
        CrossFittedGapResult carrying naive and cross-fitted estimates, the
        bias, and the full paired inference suite for each.
    """
    chat_harmful = np.asarray(chat_harmful, dtype=np.float64)
    chat_benign = np.asarray(chat_benign, dtype=np.float64)
    agent_harmful = np.asarray(agent_harmful, dtype=np.float64)

    n = len(chat_harmful)
    if len(agent_harmful) != n:
        raise ValueError(
            f"chat_harmful ({n}) and agent_harmful ({len(agent_harmful)}) must be "
            "paired row-wise."
        )
    if n_folds < 2 or n_folds > n:
        raise ValueError(f"n_folds must be in [2, {n}], got {n_folds}")

    # ---- Naive (in-sample) estimate: what the original pipeline reports ----
    d_full = _diff_in_means_direction(chat_harmful, chat_benign)
    proj_chat_naive = chat_harmful @ d_full
    proj_agent_naive = agent_harmful @ d_full
    naive_stats = run_paired_suite(
        proj_chat_naive, proj_agent_naive, n_permutations, n_boot, "greater", seed
    )

    # ---- Cross-fitted estimate ----
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds = np.array_split(order, n_folds)

    # A matched benign arm is held out with its harmful twin. Only an
    # unmatched arm (different row count) is folded on its own.
    m = len(chat_benign)
    if paired_arms and m == n:
        folds_b = folds
    else:
        order_b = rng.permutation(m)
        folds_b = np.array_split(order_b, n_folds)

    proj_chat_oof = np.empty(n)
    proj_agent_oof = np.empty(n)

    for k in range(n_folds):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        train_idx_b = np.concatenate([folds_b[j] for j in range(n_folds) if j != k])

        d_k = _diff_in_means_direction(
            chat_harmful[train_idx], chat_benign[train_idx_b]
        )
        # Both conditions of a held-out prompt use the SAME direction: pairing intact.
        proj_chat_oof[test_idx] = chat_harmful[test_idx] @ d_k
        proj_agent_oof[test_idx] = agent_harmful[test_idx] @ d_k

    crossfit_stats = run_paired_suite(
        proj_chat_oof, proj_agent_oof, n_permutations, n_boot, "greater", seed
    )

    bias = naive_stats.delta_p - crossfit_stats.delta_p
    denom = abs(crossfit_stats.delta_p)
    bias_ratio = float(bias / denom) if denom > 1e-12 else float("inf")

    return CrossFittedGapResult(
        layer_idx=layer_idx,
        n_folds=n_folds,
        delta_p_naive=naive_stats.delta_p,
        delta_p_crossfit=crossfit_stats.delta_p,
        bias_estimate=float(bias),
        bias_ratio=bias_ratio,
        naive_stats=naive_stats,
        crossfit_stats=crossfit_stats,
        proj_chat_oof=proj_chat_oof,
        proj_agent_oof=proj_agent_oof,
    )


# ============================================================
# 3. CROSS-VALIDATED AGENT DIRECTION  (Observations 3 and 9)
# ============================================================

DIRECTION_METHODS = ("logistic", "diff_means")


def _fit_agent_direction(X: np.ndarray, y: np.ndarray, C: float = 1.0,
                         seed: int = 42, method: str = "logistic") -> np.ndarray:
    """
    Unit direction separating the two classes of (X, y).

    method="diff_means": the normalised difference between the class
    centroids - the estimator that defines d_chat. No fitting, no
    hyperparameter, no rescaling of the axes.

    method="logistic": a logistic probe, returned as its normal in the
    ORIGINAL activation space. The probe trains on standardised features, so
    the raw coefficient lives in the scaled space. Since z_j = (a_j -
    mu_j)/s_j, the hyperplane normal in the original space is w_j / s_j
    (element-wise). Skipping this rescaling would make any cosine against
    d_chat meaningless, because the two vectors would inhabit different metric
    spaces.

    The two are different objects, not two estimates of one. With 84 points in
    thousands of dimensions infinitely many hyperplanes separate the classes;
    the probe returns one of them, reweighted by 1/s_j, and it need not point
    along the centroid difference at all. Compare a direction only with
    another obtained by the same method (see `format_rotation`).
    """
    if method == "diff_means":
        y = np.asarray(y)
        return _diff_in_means_direction(X[y == 1], X[y == 0])
    if method != "logistic":
        raise ValueError(f"method must be one of {DIRECTION_METHODS}, got {method!r}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, C=C, solver="lbfgs", random_state=seed)
    clf.fit(X_scaled, y)

    w = clf.coef_[0] / scaler.scale_
    norm = np.linalg.norm(w)
    if norm < 1e-12:
        raise ValueError("Probe returned a degenerate (zero-norm) direction.")
    return w / norm


def _split_half_reliability(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
    n_repeats: int = 20,
    seed: int = 42,
    groups: Optional[np.ndarray] = None,
    method: str = "logistic",
) -> Tuple[float, float]:
    """
    Reproducibility of w_agent measured on DISJOINT halves of the data.

    Why not the inter-fold cosine: in K-fold cross-validation the training
    partitions overlap (with K=5, any two share 60% of their samples), so the
    two fitted directions share a large part of their noise. Their agreement
    therefore overstates reproducibility, sometimes drastically.

    Splitting into two halves with NO shared samples makes the noise
    components independent, which is the condition under which

        cos(w_1, w_2) = b^2

    holds for w = b * w_true + noise. That identity is what licenses the
    attenuation correction, so the disjointness is not a refinement but a
    requirement.

    Returns:
        (mean reliability over repeats, sd over repeats)
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    cosines: List[float] = []

    for _ in range(n_repeats):
        if groups is not None:
            # Disjoint halves of PAIRS. Splitting the classes separately put a
            # prompt's harmful row in one half and its benign twin in the
            # other, so the two halves shared every prompt's content and their
            # noise was not independent - the condition the attenuation
            # correction rests on.
            order = rng.permutation(np.unique(groups))
            in_a = np.isin(groups, order[: len(order) // 2])
            half_a, half_b = np.flatnonzero(in_a), np.flatnonzero(~in_a)
        else:
            # Stratified disjoint halves: split each class separately so both
            # halves keep both classes.
            idx_pos = rng.permutation(np.flatnonzero(y == 1))
            idx_neg = rng.permutation(np.flatnonzero(y == 0))
            half_a = np.concatenate([idx_pos[: len(idx_pos) // 2],
                                     idx_neg[: len(idx_neg) // 2]])
            half_b = np.concatenate([idx_pos[len(idx_pos) // 2:],
                                     idx_neg[len(idx_neg) // 2:]])

        if len(np.unique(y[half_a])) < 2 or len(np.unique(y[half_b])) < 2:
            continue
        try:
            w_a = _fit_agent_direction(X[half_a], y[half_a], C=C, seed=seed,
                                       method=method)
            w_b = _fit_agent_direction(X[half_b], y[half_b], C=C, seed=seed,
                                       method=method)
        except ValueError:
            continue
        cosines.append(float(np.dot(w_a, w_b)))

    if not cosines:
        return float("nan"), float("nan")
    return (
        float(np.mean(cosines)),
        float(np.std(cosines, ddof=1)) if len(cosines) > 1 else 0.0,
    )


def cross_validated_agent_direction(
    agent_harmful: np.ndarray,
    agent_benign: np.ndarray,
    d_chat: Optional[np.ndarray] = None,
    layer_idx: int = -1,
    n_folds: int = 5,
    C: float = 1.0,
    seed: int = 42,
    method: str = "logistic",
) -> CrossValidatedDirectionResult:
    """
    Estimate w_agent with honest out-of-fold evaluation and a reliability ceiling.

    Why the original in-sample AUROC carries no information: with N = 84
    observations in d = 8192 dimensions, Cover's theorem states that N points
    in general position in R^d are linearly separable for ANY labelling when
    N <= d + 1. Perfect in-sample separation is therefore guaranteed a priori
    and would also be obtained from randomly permuted labels. It is a property
    of the geometry, not a finding about the data.

    The inter-fold cosine addresses a separate problem. w_agent estimated from
    84 points in 8192 dimensions is noise-dominated, and in high dimension the
    noise is near-orthogonal to any fixed direction, so cos(w_agent, d_chat) is
    attenuated toward zero and the reported angle is an UPPER BOUND. Comparing
    directions estimated from disjoint folds measures how much of w_agent is
    reproducible at all, which is the ceiling on how precisely its angle with
    d_chat can be known. If two disjoint estimates of w_agent agree only at
    cos = 0.5, a measured cos = 0.36 against d_chat cannot be reported to two
    decimal places.

    Returns:
        CrossValidatedDirectionResult with out-of-fold AUROC per fold, the
        in-sample value for comparison, alignment with d_chat, the inter-fold
        reliability, and the chance scale for this dimensionality.
    """
    agent_harmful = np.asarray(agent_harmful, dtype=np.float64)
    agent_benign = np.asarray(agent_benign, dtype=np.float64)

    X = np.concatenate([agent_harmful, agent_benign], axis=0)
    y = np.concatenate([np.ones(len(agent_harmful)), np.zeros(len(agent_benign))])
    hidden_dim = X.shape[1]

    # ---- In-sample value: the inflated number, kept for explicit comparison ----
    w_full = _fit_agent_direction(X, y, C=C, seed=seed, method=method)
    auroc_in_sample = float(roc_auc_score(y, X @ w_full))

    # ---- Out-of-fold evaluation ----
    groups = pair_groups(len(agent_harmful), len(agent_benign))
    fold_dirs: List[np.ndarray] = []
    fold_aurocs: List[float] = []
    fold_cosines: List[float] = []

    for train_idx, test_idx in grouped_fold_indices(y, n_folds, seed, groups):
        w_k = _fit_agent_direction(X[train_idx], y[train_idx], C=C, seed=seed,
                                   method=method)
        scores = X[test_idx] @ w_k
        # A fold with a single class present cannot yield an AUROC.
        if len(np.unique(y[test_idx])) > 1:
            fold_aurocs.append(float(roc_auc_score(y[test_idx], scores)))
        fold_dirs.append(w_k)
        if d_chat is not None:
            fold_cosines.append(float(np.dot(w_k, d_chat)))

    # ---- Inter-fold cosine: DIAGNOSTIC ONLY, do not use as the ceiling ----
    # K-fold training sets overlap heavily (with K=5, any two share 60% of
    # their samples), so their noise components are correlated and this
    # quantity is biased upward as a measure of reproducibility.
    inter = [
        float(np.dot(fold_dirs[i], fold_dirs[j]))
        for i in range(len(fold_dirs))
        for j in range(i + 1, len(fold_dirs))
    ]

    # ---- Reliability from DISJOINT split-halves ----
    sh_mean, sh_std = _split_half_reliability(
        X, y, C=C, seed=seed, groups=groups, method=method
    )

    # Correction for attenuation (Spearman). If the estimate is
    #     w_est = b * w_true + noise,  noise ~ orthogonal
    # then two estimates from DISJOINT data have independent noise, so
    #     cos(w_1, w_2) = b^2      and      cos(w_est, d_chat) = b * cos_true.
    # Hence b = sqrt(reliability), and the true cosine is recovered by
    # dividing the measured one by b.
    attenuation = (
        float(np.sqrt(sh_mean)) if np.isfinite(sh_mean) and sh_mean > 0.0
        else float("nan")
    )

    chance_sd = 1.0 / np.sqrt(hidden_dim)
    cos_mean = float(np.mean(fold_cosines)) if fold_cosines else 0.0

    # GUARD: the correction divides by the attenuation factor, so it amplifies
    # estimation noise by 1/b. Below MIN_RELIABILITY_FOR_CORRECTION the
    # amplification exceeds 2x and the "corrected" value is numerically
    # meaningless - it can exceed 1 and clip to a spurious 0 deg. Refusing to
    # report is the correct behaviour: a direction this irreproducible does
    # not have a recoverable angle at this sample size. Returning nan forces
    # that fact to surface instead of hiding behind a plausible number.
    if np.isfinite(attenuation) and sh_mean >= MIN_RELIABILITY_FOR_CORRECTION:
        cos_disatt = float(np.clip(cos_mean / attenuation, -1.0, 1.0))
    else:
        cos_disatt = float("nan")

    return CrossValidatedDirectionResult(
        layer_idx=layer_idx,
        n_folds=n_folds,
        auroc_in_sample=auroc_in_sample,
        auroc_oof_mean=float(np.mean(fold_aurocs)) if fold_aurocs else float("nan"),
        auroc_oof_std=float(np.std(fold_aurocs, ddof=1)) if len(fold_aurocs) > 1 else 0.0,
        auroc_oof_folds=fold_aurocs,
        cosine_with_d_chat_mean=cos_mean,
        cosine_with_d_chat_std=(
            float(np.std(fold_cosines, ddof=1)) if len(fold_cosines) > 1 else 0.0
        ),
        angle_deg_mean=float(np.degrees(np.arccos(np.clip(cos_mean, -1.0, 1.0)))),
        split_half_reliability=sh_mean,
        split_half_reliability_std=sh_std,
        attenuation_factor=attenuation,
        cosine_disattenuated=cos_disatt,
        angle_deg_disattenuated=(
            float(np.degrees(np.arccos(cos_disatt)))
            if np.isfinite(cos_disatt) else float("nan")
        ),
        inter_fold_cosine_mean=float(np.mean(inter)) if inter else float("nan"),
        inter_fold_cosine_std=float(np.std(inter, ddof=1)) if len(inter) > 1 else 0.0,
        chance_cosine_sd=float(chance_sd),
        cosine_in_chance_sd=float(cos_mean / chance_sd) if chance_sd > 0 else 0.0,
        w_agent_full=w_full,
    )


# ============================================================
# 4. CROSS-VALIDATED SAFETY MONITOR  (Observation 3)
# ============================================================

def _calibrate_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Pick the threshold maximising F1 over percentiles of the score set."""
    if len(scores) == 0:
        return 0.0
    candidates = np.percentile(scores, np.arange(5, 96, 1))
    best_t, best_f1 = float(np.median(scores)), -1.0
    for t in candidates:
        f1 = f1_score(labels, (scores > t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def cross_validated_monitor(
    agent_harmful: np.ndarray,
    agent_benign: np.ndarray,
    layer_idx: int = -1,
    n_folds: int = 5,
    C: float = 1.0,
    seed: int = 42,
    method: str = "logistic",
) -> CrossValidatedMonitorResult:
    """
    Evaluate the safety monitor with direction AND threshold fitted on train only.

    The original routine calibrated tau on the same scores it then scored,
    which guarantees an optimistic F1 for the same reason as the probe: with
    N << d the training scores are perfectly separable, so some threshold
    always achieves F1 = 1.0. Here each fold fits w and calibrates tau on its
    training partition, then predicts on the held-out partition. Predictions
    are pooled across folds before computing the metrics, so every sample is
    scored exactly once by a model that never saw it.
    """
    agent_harmful = np.asarray(agent_harmful, dtype=np.float64)
    agent_benign = np.asarray(agent_benign, dtype=np.float64)

    X = np.concatenate([agent_harmful, agent_benign], axis=0)
    y = np.concatenate([np.ones(len(agent_harmful)), np.zeros(len(agent_benign))])

    # In-sample reference (what the old pipeline reported)
    w_full = _fit_agent_direction(X, y, C=C, seed=seed, method=method)
    s_full = X @ w_full
    t_full = _calibrate_threshold(s_full, y)
    f1_in_sample = float(f1_score(y, (s_full > t_full).astype(int), zero_division=0))

    groups = pair_groups(len(agent_harmful), len(agent_benign))
    y_pred = np.zeros(len(y), dtype=int)
    thresholds: List[float] = []

    for train_idx, test_idx in grouped_fold_indices(y, n_folds, seed, groups):
        w_k = _fit_agent_direction(X[train_idx], y[train_idx], C=C, seed=seed,
                                   method=method)
        tau_k = _calibrate_threshold(X[train_idx] @ w_k, y[train_idx])
        thresholds.append(tau_k)
        y_pred[test_idx] = (X[test_idx] @ w_k > tau_k).astype(int)

    tp = int(((y_pred == 1) & (y == 1)).sum())
    fp = int(((y_pred == 1) & (y == 0)).sum())
    fn = int(((y_pred == 0) & (y == 1)).sum())
    tn = int(((y_pred == 0) & (y == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return CrossValidatedMonitorResult(
        layer_idx=layer_idx,
        n_folds=n_folds,
        precision_oof=float(precision),
        recall_oof=float(recall),
        f1_oof=float(f1),
        fpr_oof=float(fpr),
        thresholds_per_fold=thresholds,
        f1_in_sample=f1_in_sample,
        n_harmful=len(agent_harmful),
        n_benign=len(agent_benign),
    )


# ============================================================
# 4b. ROTATION BETWEEN FORMATS, LIKE FOR LIKE
# ============================================================

def format_rotation(
    chat_harmful: np.ndarray,
    chat_benign: np.ndarray,
    agent_harmful: np.ndarray,
    agent_benign: np.ndarray,
    method: str = "diff_means",
    n_repeats: int = 50,
    C: float = 1.0,
    seed: int = 42,
    layer_idx: int = -1,
) -> Dict:
    """
    How far does the harmful/benign direction turn between chat and agent?

    The number first reported for this was cos(w_agent, d_chat), with w_agent
    a logistic probe and d_chat a difference of centroids. That cosine cannot
    be read as an angle, for three separate reasons:

    1. DIFFERENT ESTIMATORS. A probe normal and a centroid difference are not
       the same object. Fitted on the SAME chat prompts, the probe normal and
       d_chat already disagree (cosine around 0.4-0.5 in the Llama-3.3 run), so
       a low cosine across formats mixes "the format changed" with "the method
       changed". Here both directions come from one method.

    2. BOTH DIRECTIONS ARE NOISY. Each is estimated from a few dozen prompts in
       thousands of dimensions, and noise pulls any cosine toward zero. The
       reference is therefore not 1 but what the method achieves against
       ITSELF on fresh prompts of the same format.

    3. SHARED PROMPTS. Chat and agent activations of one prompt share that
       prompt's content. Two directions fitted on the same prompts share its
       noise, which pulls their cosine UP. Fitting the two directions on
       disjoint halves of the pairs removes it.

    So the pairs are split in two halves A and B, a direction is fitted per
    format on each half, and three cosines are averaged over many splits:

        within chat   cos(c_A, c_B)   -> reliability of the chat direction
        within agent  cos(a_A, a_B)   -> reliability of the agent direction
        across        cos(c_A, a_B)   -> agreement between formats

    With d_est = b * d_true + noise and independent noise in the two halves,
    the within-format cosines estimate b_chat^2 and b_agent^2 and the across
    one estimates b_chat * b_agent * cos(true). Their ratio

        across / sqrt(within_chat * within_agent)

    is the cosine between the two TRUE directions: 1 if the format does not
    turn the axis, 0 if it turns it by a right angle. It is reported only when
    both reliabilities are high enough for the division to mean something.

    Args:
        the four arrays are (n_pairs, hidden_dim), row-aligned by pair.
        method: "diff_means" or "logistic" (see `_fit_agent_direction`).

    Returns:
        dict with the three mean cosines, the same-half cosine for reference,
        the full-sample cosine, and the normalised cosine with its angle.
    """
    arms = [np.asarray(a, dtype=np.float64) for a in
            (chat_harmful, chat_benign, agent_harmful, agent_benign)]
    n = len(arms[0])
    if any(len(a) != n for a in arms):
        raise ValueError("format_rotation needs four arms row-aligned by pair.")
    if n < 4:
        raise ValueError("format_rotation needs at least 4 pairs.")
    ch, cb, ah, ab = arms

    def fit(h: np.ndarray, b: np.ndarray, idx: np.ndarray) -> np.ndarray:
        X = np.concatenate([h[idx], b[idx]], axis=0)
        y = np.concatenate([np.ones(len(idx)), np.zeros(len(idx))])
        return _fit_agent_direction(X, y, C=C, seed=seed, method=method)

    everything = np.arange(n)
    cos_full = float(np.dot(fit(ch, cb, everything), fit(ah, ab, everything)))

    rng = np.random.default_rng(seed)
    within_c: List[float] = []
    within_a: List[float] = []
    across: List[float] = []
    same_half: List[float] = []
    for _ in range(n_repeats):
        perm = rng.permutation(n)
        A, B = perm[: n // 2], perm[n // 2:]
        try:
            c_a, c_b = fit(ch, cb, A), fit(ch, cb, B)
            a_a, a_b = fit(ah, ab, A), fit(ah, ab, B)
        except ValueError:
            continue
        within_c.append(float(c_a @ c_b))
        within_a.append(float(a_a @ a_b))
        across.append(0.5 * float(c_a @ a_b + c_b @ a_a))
        same_half.append(0.5 * float(c_a @ a_a + c_b @ a_b))

    nan = float("nan")
    if not across:
        return {"layer_idx": layer_idx, "method": method, "n_pairs": n,
                "cosine_full_sample": cos_full, "cosine_normalised": nan}

    rel_c, rel_a = float(np.mean(within_c)), float(np.mean(within_a))
    cross = float(np.mean(across))
    ceiling = float(np.sqrt(rel_c * rel_a)) if rel_c > 0 and rel_a > 0 else nan

    # Below the floor the division amplifies noise more than it corrects, and
    # the result would be a plausible-looking number with nothing behind it.
    if np.isfinite(ceiling) and ceiling >= MIN_RELIABILITY_FOR_CORRECTION:
        cos_norm = float(np.clip(cross / ceiling, -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cos_norm)))
    else:
        cos_norm, angle = nan, nan

    return {
        "layer_idx": layer_idx,
        "method": method,
        "n_pairs": n,
        "n_repeats": len(across),
        "cosine_full_sample": cos_full,
        "cosine_same_half": float(np.mean(same_half)),
        "cosine_across_formats": cross,
        "cosine_across_formats_sd": float(np.std(across, ddof=1)) if len(across) > 1 else nan,
        "reliability_chat": rel_c,
        "reliability_agent": rel_a,
        "ceiling": ceiling,
        "cosine_normalised": cos_norm,
        "angle_deg_normalised": angle,
    }


# ============================================================
# 5. LAYER SELECTION WITHOUT POST-HOC INFLATION  (Observation 4)
# ============================================================

def select_layer_and_test(
    chat_harmful: Dict[int, np.ndarray],
    chat_benign: Dict[int, np.ndarray],
    agent_harmful: Dict[int, np.ndarray],
    n_folds: int = 5,
    n_permutations: int = 10000,
    seed: int = 42,
    agent_benign: Optional[Dict[int, np.ndarray]] = None,
    n_splits: int = 200,
) -> Dict:
    """
    Split-sample layer selection, avoiding the winner's-curse in the reported p.

    Choosing the layer that maximises DeltaP and then testing the hypothesis on
    that same layer is post-hoc inference: the statistic is biased upward and
    the type-I error rate exceeds the nominal level.

    Here the prompt PAIRS are split in half. The selection half picks the
    layer; the inference half runs the test on the pre-committed layer only.
    The test is therefore performed on data that played no part in the
    selection, and every figure quoted for the selected layer - the count of
    pairs that decrease included - must be the one from the inference half.
    Quoting the full-sample count for a layer chosen on half of it re-admits
    the selection half through the back door.

    Two things changed relative to the first version of this routine:

      - The split is over pairs. The benign arm used to be permuted
        independently, so a harmful prompt in the inference half could have
        its benign twin in the selection half, where it helped fit the
        direction that ranked the layers.
      - One random split is one draw. The procedure is repeated over
        `n_splits` partitions, with the direction fitted on the selection half
        and the inference pairs projected onto it untouched, to report how
        often each layer is chosen and how the inference-half result varies.
        A conclusion that holds in one partition and not in most others is not
        a conclusion.

    The Bonferroni-adjusted alpha over all candidate layers is also returned,
    so the alternative correction is available without a second run.

    Args:
        agent_benign: optional, row-aligned with the other arms. When given,
            the harmful-vs-benign interaction is also tested on the inference
            half, out-of-fold, paired by prompt.
        n_splits: random partitions for the stability summary. 0 disables it.

    Returns a dict with the selected layer, the inference-half test result,
    the per-layer selection scores, the stability summary and the adjusted
    alpha.
    """
    layers = sorted(set(chat_harmful) & set(chat_benign) & set(agent_harmful))
    if not layers:
        raise ValueError("No layer is present in all three activation dicts.")

    n = len(chat_harmful[layers[0]])
    m = len(chat_benign[layers[0]])
    paired = m == n

    def _split(rng):
        perm = rng.permutation(n)
        sel, inf = perm[: n // 2], perm[n // 2:]
        if paired:
            return sel, inf, sel, inf
        perm_b = rng.permutation(m)
        return sel, inf, perm_b[: m // 2], perm_b[m // 2:]

    def _select(sel, sel_b):
        scores: Dict[int, float] = {}
        dirs: Dict[int, np.ndarray] = {}
        for l in layers:
            d_l = _diff_in_means_direction(
                chat_harmful[l][sel], chat_benign[l][sel_b]
            )
            dirs[l] = d_l
            scores[l] = float(
                ((chat_harmful[l][sel] - agent_harmful[l][sel]) @ d_l).mean()
            )
        best = max(scores, key=scores.get)
        return best, scores, dirs[best]

    # --- The reported split ---
    sel_idx, inf_idx, sel_idx_b, inf_idx_b = _split(np.random.default_rng(seed))
    chosen, selection_scores, _ = _select(sel_idx, sel_idx_b)
    k_inf = max(2, min(n_folds, len(inf_idx)))

    # --- Inference half: cross-fitted test on the pre-committed layer only ---
    result = cross_fitted_projection_gap(
        chat_harmful[chosen][inf_idx],
        chat_benign[chosen][inf_idx_b],
        agent_harmful[chosen][inf_idx],
        layer_idx=chosen,
        n_folds=k_inf,
        n_permutations=n_permutations,
        seed=seed,
    )

    # --- Same half, harmful-vs-benign interaction, paired by prompt ---
    interaction: Optional[PairedTestResult] = None
    if paired and agent_benign is not None and chosen in agent_benign:
        held = crossfit_condition_projections(
            {
                "chat_harmful": chat_harmful[chosen][inf_idx],
                "chat_benign": chat_benign[chosen][inf_idx],
                "agent_harmful": agent_harmful[chosen][inf_idx],
                "agent_benign": agent_benign[chosen][inf_idx],
            },
            n_folds=k_inf, seed=seed, layer_idx=chosen,
        ).projections
        interaction = run_paired_suite(
            held["chat_harmful"] - held["agent_harmful"],
            held["chat_benign"] - held["agent_benign"],
            n_permutations, n_permutations, "greater", seed,
        )

    # --- Stability across partitions ---
    stability: Dict = {}
    if n_splits > 0:
        rng = np.random.default_rng(seed + 1)
        picks: List[int] = []
        frac_dec: List[float] = []
        deltas: List[float] = []
        for _ in range(n_splits):
            s, i, s_b, _ = _split(rng)
            layer_s, _, d_s = _select(s, s_b)
            # d_s was fitted on the selection half alone, so the inference
            # pairs are out-of-sample for it without any further folding.
            diff = (chat_harmful[layer_s][i] - agent_harmful[layer_s][i]) @ d_s
            picks.append(int(layer_s))
            frac_dec.append(float((diff > 0).mean()))
            deltas.append(float(diff.mean()))
        counts = {l: picks.count(l) for l in sorted(set(picks))}
        stability = {
            "n_splits": int(n_splits),
            "selected_layer_frequency": {
                int(l): c / n_splits for l, c in counts.items()
            },
            "fraction_decreasing_mean": float(np.mean(frac_dec)),
            "fraction_decreasing_p05": float(np.percentile(frac_dec, 5)),
            "fraction_decreasing_min": float(np.min(frac_dec)),
            "delta_p_mean": float(np.mean(deltas)),
            "delta_p_p05": float(np.percentile(deltas, 5)),
            "share_splits_positive": float(np.mean(np.asarray(deltas) > 0)),
        }

    return {
        "selected_layer": int(chosen),
        "selection_scores": selection_scores,
        "n_selection": int(len(sel_idx)),
        "n_inference": int(len(inf_idx)),
        "split_over_pairs": bool(paired),
        "result": result,
        "interaction": interaction,
        "stability": stability,
        "bonferroni_alpha": 0.05 / len(layers),
        "n_layers_tested": len(layers),
    }


# ============================================================
# 6. REPORTING
# ============================================================

def format_gap_report(res: CrossFittedGapResult) -> str:
    """Human-readable comparison of naive and cross-fitted estimates."""
    n, c = res.naive_stats, res.crossfit_stats
    ratio = "n/a" if not np.isfinite(res.bias_ratio) else f"{res.bias_ratio:+.1%}"
    lines = [
        "=" * 72,
        f"CROSS-FITTED PROJECTION GAP  (layer {res.layer_idx}, K={res.n_folds})",
        "=" * 72,
        f"{'':<26}{'naive (in-sample)':>21}{'cross-fitted':>21}",
        "-" * 72,
        f"{'DeltaP':<26}{n.delta_p:>21.4f}{c.delta_p:>21.4f}",
        f"{'mean chat':<26}{n.mean_a:>21.4f}{c.mean_a:>21.4f}",
        f"{'mean agent':<26}{n.mean_b:>21.4f}{c.mean_b:>21.4f}",
        f"{'Cohen d (paired)':<26}{n.cohens_d_paired:>21.4f}{c.cohens_d_paired:>21.4f}",
        f"{'Cohen d (pooled)':<26}{n.cohens_d_pooled:>21.4f}{c.cohens_d_pooled:>21.4f}",
        f"{'implied pair corr r':<26}{n.implied_pair_correlation:>21.4f}"
        f"{c.implied_pair_correlation:>21.4f}",
        f"{'permutation p':<26}{n.p_value_permutation:>21.4f}{c.p_value_permutation:>21.4f}",
        f"{'bootstrap CI low':<26}{n.ci_lower:>21.4f}{c.ci_lower:>21.4f}",
        f"{'bootstrap CI high':<26}{n.ci_upper:>21.4f}{c.ci_upper:>21.4f}",
        f"{'pairs decreasing':<26}{str(n.n_decreasing) + '/' + str(n.n_pairs):>21}"
        f"{str(c.n_decreasing) + '/' + str(c.n_pairs):>21}",
        "-" * 72,
        f"Own-sample bias in DeltaP : {res.bias_estimate:+.4f}  ({ratio} of the "
        "cross-fitted estimate)",
        "",
        "  Report the CROSS-FITTED value as the estimate of the effect.",
        "  The naive value is retained only to quantify the bias.",
        f"  Permutation p of 0.0 means p < {n.p_value_floor:g}, not zero.",
        "=" * 72,
    ]
    return "\n".join(lines)


def format_direction_report(res: CrossValidatedDirectionResult) -> str:
    """Human-readable out-of-fold report for w_agent."""
    rel = res.split_half_reliability
    lines = [
        "=" * 72,
        f"CROSS-VALIDATED AGENT DIRECTION  (layer {res.layer_idx}, K={res.n_folds})",
        "=" * 72,
        f"  AUROC in-sample        : {res.auroc_in_sample:.4f}   "
        "<- guaranteed by Cover's theorem, NOT evidence",
        f"  AUROC out-of-fold      : {res.auroc_oof_mean:.4f} "
        f"(sd {res.auroc_oof_std:.4f})   <- report this one",
        f"  per-fold               : "
        + ", ".join(f"{a:.3f}" for a in res.auroc_oof_folds),
        "",
        "  --- alignment with d_chat ---",
        f"  cos measured           : {res.cosine_with_d_chat_mean:.4f} "
        f"(sd {res.cosine_with_d_chat_std:.4f})",
        f"  angle measured         : {res.angle_deg_mean:.1f} deg   "
        "<- UPPER BOUND, attenuated",
        f"  chance cosine sd       : {res.chance_cosine_sd:.5f}  "
        f"-> measured is {res.cosine_in_chance_sd:.1f} sd from chance",
        "",
        "  --- reliability and disattenuation ---",
        f"  split-half reliability : {rel:.4f} "
        f"(sd {res.split_half_reliability_std:.4f})   [DISJOINT halves]",
        f"  attenuation factor     : {res.attenuation_factor:.4f}   = sqrt(reliability)",
        f"  cos disattenuated      : {res.cosine_disattenuated:.4f}",
        f"  angle disattenuated    : {res.angle_deg_disattenuated:.1f} deg   "
        "<- bias-corrected estimate",
        "",
        f"  inter-fold cos(wi,wj)  : {res.inter_fold_cosine_mean:.4f}   "
        "[DIAGNOSTIC ONLY]",
        "    ^ inflated: K-fold training sets overlap, so their noise is",
        "      correlated. Never use this as the reliability ceiling.",
    ]
    if np.isfinite(rel):
        if rel < 0.10:
            lines += [
                "",
                "    VERDICT: w_agent is NOT reproducible at this sample size.",
                "             Report only that the alignment differs from chance.",
                "             Do not quote any angle, corrected or otherwise.",
            ]
        elif rel < 0.50:
            lines += [
                "",
                "    VERDICT: w_agent is weakly reproducible. The disattenuated",
                "             angle is a rough estimate with wide uncertainty;",
                "             quote it as a range, never to two decimals.",
            ]
        else:
            lines += [
                "",
                "    VERDICT: w_agent is reproducible. The disattenuated angle",
                "             is interpretable as a point estimate.",
            ]
    lines.append("=" * 72)
    return "\n".join(lines)


def format_monitor_report(res: CrossValidatedMonitorResult) -> str:
    """Human-readable out-of-fold report for the safety monitor."""
    return "\n".join([
        "=" * 72,
        f"CROSS-VALIDATED SAFETY MONITOR  (layer {res.layer_idx}, K={res.n_folds})",
        "=" * 72,
        f"  F1 in-sample   : {res.f1_in_sample:.4f}   <- threshold calibrated on the "
        "evaluation set",
        f"  F1 out-of-fold : {res.f1_oof:.4f}   <- report this one",
        f"  Precision      : {res.precision_oof:.4f}",
        f"  Recall         : {res.recall_oof:.4f}",
        f"  FPR            : {res.fpr_oof:.4f}",
        f"  thresholds     : " + ", ".join(f"{t:.3f}" for t in res.thresholds_per_fold),
        f"  samples        : {res.n_harmful} harmful + {res.n_benign} benign",
        "=" * 72,
    ])


# ============================================================
# 6. HELD-OUT EVALUATION OF d_chat  (Problem 4)
# ============================================================

@dataclass
class HeldOutProjections:
    """Projections of every condition onto directions that never saw the prompt."""
    layer_idx: int
    n_folds: int
    n_pairs: int
    projections: Dict[str, np.ndarray] = field(repr=False, default=None)
    projections_in_sample: Dict[str, np.ndarray] = field(repr=False, default=None)
    fold_of_pair: np.ndarray = field(repr=False, default=None)
    # How far the fitted direction moves between folds. A low value means the
    # held-out numbers below are noisy because d_chat itself is unstable at
    # this n, not because detection failed.
    inter_fold_cosine_mean: float = float("nan")
    inter_fold_cosine_min: float = float("nan")


@dataclass
class HeldOutDetectionResult:
    """AUROC of d_chat scored in-sample and out-of-fold, per format."""
    layer_idx: int
    n_folds: int
    n_pairs: int
    auroc_chat_in_sample: float
    auroc_chat_heldout: float
    auroc_agent_in_sample: float
    auroc_agent_heldout: float
    optimism_chat: float              # in-sample minus held-out
    optimism_agent: float
    auroc_gap_in_sample: float        # chat minus agent
    auroc_gap_heldout: float
    # Paired bootstrap over prompt pairs, on the held-out scores
    auroc_chat_ci: Tuple[float, float]
    auroc_agent_ci: Tuple[float, float]
    auroc_gap_ci: Tuple[float, float]
    p_value_gap: float
    inter_fold_cosine_mean: float

    @property
    def gap_excludes_zero(self) -> bool:
        """True when the chat-vs-agent AUROC difference survives resampling."""
        lo, hi = self.auroc_gap_ci
        return bool(lo > 0 or hi < 0)


def crossfit_condition_projections(
    conditions: Dict[str, np.ndarray],
    n_folds: int = 5,
    seed: int = 42,
    layer_idx: int = -1,
    fit_harmful_key: str = "chat_harmful",
    fit_benign_key: str = "chat_benign",
) -> HeldOutProjections:
    """
    Project every condition onto a d_chat fitted WITHOUT that prompt.

    The pipeline's headline AUROC asks how well d_chat separates chat_harmful
    from chat_benign - the two groups whose centroid difference IS d_chat. At
    n = 42 in d = 8192 that question has a guaranteed flattering answer, and
    the flattery does not transfer to the agent format, which never entered
    the fit. Comparing the two as though they were the same measurement
    attributes to the format what is an artefact of the estimation.

    Folds are assigned over PROMPT PAIRS, not over rows. A pair's harmful and
    benign variants are built from the same seed, so sending one to training
    while the other is scored would leak the very contrast the direction
    encodes. Holding the pair out whole is what turns the resulting AUROC into
    a detection rate.

    Every condition in `conditions` is scored - chat and agent, both arms - so
    the harmful/benign contrast and the chat/agent contrast are measured on
    the same held-out footing.

    Args:
        conditions: condition name -> (n_pairs, hidden_dim). All arrays must
            share one row order, one row per prompt pair, which is what
            `_assert_pairing` in the pipeline guarantees. Must contain the two
            fit arms.
        n_folds: K. Clamp it upstream for tiny dry runs.
        fit_harmful_key / fit_benign_key: the arms the direction is fitted
            from. The defaults reproduce d_chat.

    Returns:
        HeldOutProjections carrying out-of-fold and in-sample projections side
        by side, so the optimism of the in-sample number can be reported
        rather than silently corrected away.
    """
    for key in (fit_harmful_key, fit_benign_key):
        if key not in conditions:
            raise KeyError(
                f"crossfit_condition_projections needs '{key}' to fit the "
                f"direction; got {sorted(conditions)}"
            )

    arrays = {k: np.asarray(v, dtype=np.float64) for k, v in conditions.items()}
    n = len(arrays[fit_harmful_key])
    ragged = {k: len(v) for k, v in arrays.items() if len(v) != n}
    if ragged:
        raise ValueError(
            "crossfit_condition_projections requires every condition to be "
            f"row-aligned by prompt pair ({n} rows); got {ragged}. Fold "
            "assignment is by pair, so unpaired arms cannot be held out "
            "together."
        )
    if n_folds < 2 or n_folds > n:
        raise ValueError(f"n_folds must be in [2, {n}], got {n_folds}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds = np.array_split(order, n_folds)

    H, B = arrays[fit_harmful_key], arrays[fit_benign_key]
    oof = {name: np.empty(n, dtype=np.float64) for name in arrays}
    fold_of_pair = np.empty(n, dtype=int)
    directions: List[np.ndarray] = []

    for k, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        d_k = _diff_in_means_direction(H[train_idx], B[train_idx])
        directions.append(d_k)
        fold_of_pair[test_idx] = k
        # Both formats AND both arms of a held-out pair are scored with the
        # same d_k, so every downstream contrast stays inside one direction.
        for name, acts in arrays.items():
            oof[name][test_idx] = acts[test_idx] @ d_k

    d_full = _diff_in_means_direction(H, B)
    in_sample = {name: acts @ d_full for name, acts in arrays.items()}

    cosines = [
        float(np.dot(directions[i], directions[j]))
        for i in range(len(directions))
        for j in range(i + 1, len(directions))
    ]

    return HeldOutProjections(
        layer_idx=layer_idx,
        n_folds=n_folds,
        n_pairs=n,
        projections=oof,
        projections_in_sample=in_sample,
        fold_of_pair=fold_of_pair,
        inter_fold_cosine_mean=float(np.mean(cosines)) if cosines else float("nan"),
        inter_fold_cosine_min=float(np.min(cosines)) if cosines else float("nan"),
    )


def _auroc_from_projections(harmful: np.ndarray, benign: np.ndarray) -> float:
    """AUROC for harmful-vs-benign from two already-projected score vectors."""
    y = np.r_[np.ones(len(harmful)), np.zeros(len(benign))]
    return float(roc_auc_score(y, np.r_[harmful, benign]))


def held_out_detection(
    heldout: HeldOutProjections,
    chat_harmful_key: str = "chat_harmful",
    chat_benign_key: str = "chat_benign",
    agent_harmful_key: str = "agent_harmful",
    agent_benign_key: str = "agent_benign",
    n_boot: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> HeldOutDetectionResult:
    """
    Detection capability of d_chat per format, scored out-of-fold.

    The in-sample AUROC is reported alongside so the optimism is visible: the
    difference between the two columns is the part of the headline number that
    came from fitting and scoring on the same prompts. Only the held-out
    figures are comparable across formats.

    The bootstrap resamples PROMPT PAIRS and reuses the same resampled pairs
    for both formats, so the interval on the chat-minus-agent difference is
    paired - the two AUROCs move together under resampling, and an unpaired
    interval would be needlessly wide.
    """
    oof, ins = heldout.projections, heldout.projections_in_sample
    keys = (chat_harmful_key, chat_benign_key, agent_harmful_key, agent_benign_key)
    missing = [k for k in keys if k not in oof]
    if missing:
        raise KeyError(
            f"held_out_detection is missing {missing}; available: {sorted(oof)}"
        )

    ch, cb = oof[chat_harmful_key], oof[chat_benign_key]
    ah, ab = oof[agent_harmful_key], oof[agent_benign_key]

    auroc_chat = _auroc_from_projections(ch, cb)
    auroc_agent = _auroc_from_projections(ah, ab)
    auroc_chat_ins = _auroc_from_projections(
        ins[chat_harmful_key], ins[chat_benign_key]
    )
    auroc_agent_ins = _auroc_from_projections(
        ins[agent_harmful_key], ins[agent_benign_key]
    )

    n = heldout.n_pairs
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))

    boot_chat = np.empty(n_boot)
    boot_agent = np.empty(n_boot)
    for b in range(n_boot):
        take = idx[b]
        # Both classes are present in every resample by construction: a drawn
        # pair contributes one harmful row and one benign row.
        boot_chat[b] = _auroc_from_projections(ch[take], cb[take])
        boot_agent[b] = _auroc_from_projections(ah[take], ab[take])

    boot_gap = boot_chat - boot_agent
    lo_q, hi_q = (1 - ci_level) / 2 * 100, (1 + ci_level) / 2 * 100
    p_gap = 2 * min(float((boot_gap <= 0).mean()), float((boot_gap >= 0).mean()))

    return HeldOutDetectionResult(
        layer_idx=heldout.layer_idx,
        n_folds=heldout.n_folds,
        n_pairs=n,
        auroc_chat_in_sample=auroc_chat_ins,
        auroc_chat_heldout=auroc_chat,
        auroc_agent_in_sample=auroc_agent_ins,
        auroc_agent_heldout=auroc_agent,
        optimism_chat=float(auroc_chat_ins - auroc_chat),
        optimism_agent=float(auroc_agent_ins - auroc_agent),
        auroc_gap_in_sample=float(auroc_chat_ins - auroc_agent_ins),
        auroc_gap_heldout=float(auroc_chat - auroc_agent),
        auroc_chat_ci=(float(np.percentile(boot_chat, lo_q)),
                       float(np.percentile(boot_chat, hi_q))),
        auroc_agent_ci=(float(np.percentile(boot_agent, lo_q)),
                        float(np.percentile(boot_agent, hi_q))),
        auroc_gap_ci=(float(np.percentile(boot_gap, lo_q)),
                      float(np.percentile(boot_gap, hi_q))),
        p_value_gap=float(min(1.0, p_gap)),
        inter_fold_cosine_mean=heldout.inter_fold_cosine_mean,
    )


def format_heldout_detection_report(res: HeldOutDetectionResult) -> str:
    """Human-readable in-sample vs held-out detection report."""
    chat_ci = f"[{res.auroc_chat_ci[0]:.3f}, {res.auroc_chat_ci[1]:.3f}]"
    agent_ci = f"[{res.auroc_agent_ci[0]:.3f}, {res.auroc_agent_ci[1]:.3f}]"
    lines = [
        "=" * 72,
        f"DETECTION WITH d_chat: IN-SAMPLE vs HELD-OUT  "
        f"(layer {res.layer_idx}, K={res.n_folds})",
        "=" * 72,
        "  d_chat is fitted on the chat arms, so an AUROC scored on those same",
        "  prompts is a fit statistic. The held-out column scores every prompt",
        f"  with a direction fitted without it ({res.n_pairs} pairs).",
        "-" * 72,
        f"{'':<10}{'in-sample':>12}{'held-out':>12}{'optimism':>11}"
        f"{'95% CI (held-out)':>22}",
        f"{'chat':<10}{res.auroc_chat_in_sample:>12.4f}"
        f"{res.auroc_chat_heldout:>12.4f}{res.optimism_chat:>11.4f}{chat_ci:>22}",
        f"{'agent':<10}{res.auroc_agent_in_sample:>12.4f}"
        f"{res.auroc_agent_heldout:>12.4f}{res.optimism_agent:>11.4f}{agent_ci:>22}",
        "-" * 72,
        f"  AUROC chat - agent (held-out) : {res.auroc_gap_heldout:+.4f}  "
        f"[{res.auroc_gap_ci[0]:+.4f}, {res.auroc_gap_ci[1]:+.4f}]  "
        f"p={res.p_value_gap:.4f}",
        f"  same difference, in-sample    : {res.auroc_gap_in_sample:+.4f}",
        f"  inter-fold cos(d_i, d_j)      : {res.inter_fold_cosine_mean:.3f}  "
        "(stability of the fitted direction)",
        "-" * 72,
    ]
    max_optimism = max(res.optimism_chat, res.optimism_agent)
    if max_optimism > 0.02:
        lines += [
            f"  NOTE: the in-sample AUROC is inflated by up to {max_optimism:.3f}.",
            "        Report the held-out column; the in-sample one is kept only",
            "        to show the size of the inflation.",
        ]
    if res.gap_excludes_zero:
        lines += [
            "  VERDICT: detection is measurably weaker in the agentic format,",
            "           on prompts that never entered the direction's fit.",
        ]
    else:
        lines += [
            "  VERDICT: no detectable AUROC difference between formats once both",
            "           are scored out-of-fold. An in-sample AUROC gap of this",
            "           size is estimation optimism, not a format effect.",
        ]
    if res.inter_fold_cosine_mean < 0.5:
        lines += [
            "  CAUTION: directions from different folds disagree "
            f"(mean cos {res.inter_fold_cosine_mean:.2f}).",
            "           d_chat is unstable at this n, so the held-out figures",
            "           are attenuated: read them as a lower bound.",
        ]
    lines.append("=" * 72)
    return "\n".join(lines)
