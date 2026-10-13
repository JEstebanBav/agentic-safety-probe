"""
Linear and non-linear probing classifiers, evaluated by cross-validation.

Trains probes on activations to determine whether harmful/benign information
is linearly separable at each layer.

THE LEAKAGE RULE THAT GOVERNS THIS MODULE
-----------------------------------------
With n = 84 samples in d = 8192 dimensions, Cover's theorem guarantees that
ANY labelling is linearly separable. An in-sample AUROC of 1.0 is therefore a
property of the geometry, not a finding about the data — validate_corrections.py
reproduces it from pure noise. The only number that carries information is one
computed on samples the whole pipeline never saw.

"The whole pipeline" is the operative phrase. Every quantity a fold reports
must be derived from that fold's TRAINING partition only:

    - the standardiser        (fit on train)
    - the probe               (fit on train)
    - the refusal direction   (difference-in-means over train)
    - the threshold           (calibrated on train; see robust_estimation)

Computing the refusal direction once over the full dataset and then
cross-validating only the classifier is the failure mode that looks like
cross-validation and is not: the held-out fold contributed its centroid to the
direction, so it is no longer held out, and the AUROC comes back inflated. The
direction is re-estimated inside each fold here for exactly that reason.

A label-permutation control accompanies the linear probe. Re-running the same
cross-validation on shuffled labels gives the empirical chance level for this
sample geometry; it is the direct answer to "is the separation real or is it
Cover's theorem again". A cross-validated AUROC is only evidence to the extent
that it exceeds that control.

REPEATED K-FOLD AND WHAT ITS SPREAD MEANS
-----------------------------------------
At n = 84 and K = 5 a fold holds ~17 samples, so a single partition is a coarse
instrument and the answer moves with the seed. Repeating the whole K-fold pass
over several random partitions (`n_repeats`) averages that away.

What it does NOT do is shrink the uncertainty that matters. Two variance
sources are in play and they behave differently:

    PARTITION VARIANCE   which prompts landed in which fold. `auroc_repeat_std`
                         measures how far one partition's answer sits from
                         another's. That spread is a property of the data and
                         does NOT shrink with R; what shrinks is its
                         contribution to the REPORTED average, which falls as
                         auroc_repeat_std / sqrt(R). That is the whole benefit
                         of repeating.

    SAMPLE VARIANCE      having 84 prompts rather than 840. No amount of
                         repeating touches this, because every repeat reuses
                         the same 84 prompts.

Quoting either sd as "the uncertainty" is a trap. The partition sd answers
"would a different split have said something else", not "how well does this
generalise". The naive sd across folds has a further problem in the opposite
direction: K-fold training sets overlap heavily, so the fold scores are
positively correlated and their sample variance UNDERSTATES the variance of
the estimator.

The Nadeau-Bengio correction is the standard fix. It inflates the variance by
the overlap factor:

    Var_corrected = (1/(K*R) + n_test/n_train) * s^2_folds

and is what `auroc_ci_lower`/`auroc_ci_upper` are built from. That interval
answers "how precisely do I know this AUROC given 84 prompts", which is the
question a reviewer is actually asking.

Reference: Nadeau & Bengio (2003), "Inference for the Generalization Error",
Machine Learning 52(3).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from src.robust_estimation import grouped_fold_indices


@dataclass
class ProbeResult:
    """
    Result of probing at a single layer.

    `auroc` carries the CROSS-VALIDATED mean when the result comes from
    `cross_validate_probe`, so any downstream consumer that reads it (layer
    ranking, figures, results.json) reports the honest number by default. The
    in-sample value is retained separately, and only for comparison.
    """
    layer_idx: int
    accuracy: float
    auroc: float
    f1: float
    probe_type: str  # 'linear' or 'mlp'
    learned_direction: Optional[np.ndarray] = None  # For linear probes
    cosine_with_refusal: Optional[float] = None

    # ---- Cross-validation (populated by cross_validate_probe) ----
    n_folds: Optional[int] = None
    auroc_std: Optional[float] = None            # sd across ALL folds
    auroc_folds: Optional[List[float]] = None    # every fold of every repeat
    auroc_in_sample: Optional[float] = None      # the inflated reference
    accuracy_std: Optional[float] = None
    f1_std: Optional[float] = None

    # ---- Repeated K-fold ----
    n_repeats: Optional[int] = None
    auroc_repeat_means: Optional[List[float]] = None
    # Spread of the answer ACROSS PARTITIONS. A diagnostic, never a confidence
    # interval: it does not shrink with n_repeats (its contribution to the
    # reported mean does, as auroc_repeat_std/sqrt(R)). See module docstring.
    auroc_repeat_std: Optional[float] = None
    # Nadeau-Bengio corrected SE and interval: accounts for the overlap between
    # training folds, which makes the naive across-fold sd too small. THIS is
    # the uncertainty to quote.
    auroc_se_corrected: Optional[float] = None
    auroc_ci_lower: Optional[float] = None
    auroc_ci_upper: Optional[float] = None

    # Label-permutation control: same CV, shuffled labels -> empirical chance
    auroc_permuted_mean: Optional[float] = None
    auroc_permuted_std: Optional[float] = None
    n_label_permutations: Optional[int] = None

    # Difference-in-means direction re-estimated INSIDE each fold and scored on
    # the held-out fold. This is the fold-honest analogue of the refusal
    # direction used by the projection analysis.
    auroc_direction_oof_mean: Optional[float] = None
    auroc_direction_oof_std: Optional[float] = None

    # cos(probe normal, fold refusal direction), both fitted on the same
    # training partition, so neither has seen the evaluation fold.
    cosine_with_refusal_oof_mean: Optional[float] = None
    cosine_with_refusal_oof_std: Optional[float] = None

    n_samples: Optional[int] = None
    n_positive: Optional[int] = None
    # True when folds held out whole prompt pairs (harmful + benign twin)
    # rather than individual rows. Recorded so a figure produced under one
    # scheme is never compared with one produced under the other unknowingly.
    folds_by_pair: bool = False


@dataclass
class ProbingSuiteResult:
    """Results across all layers for one probe type."""
    per_layer: Dict[int, ProbeResult]
    best_layer: int
    best_auroc: float
    probe_type: str
    cross_validated: bool = False
    n_folds: Optional[int] = None
    n_repeats: Optional[int] = None


# ============================================================
# 0. SHARED PRIMITIVES
# ============================================================

def _probe_normal_original_space(clf, scaler: StandardScaler) -> np.ndarray:
    """
    Return the probe's hyperplane normal in the ORIGINAL activation space.

    The probe is fitted on standardised features, so `clf.coef_` lives in the
    scaled space. Since z_j = (a_j - mu_j) / s_j, the normal in the original
    space is w_j / s_j element-wise. Omitting this rescaling — as the earlier
    version of this module did — makes any cosine against the refusal direction
    meaningless, because the two vectors then inhabit different metric spaces.
    Matches robust_estimation._fit_agent_direction.
    """
    w = clf.coef_[0] / scaler.scale_
    norm = np.linalg.norm(w)
    if norm < 1e-12:
        raise ValueError("Probe returned a degenerate (zero-norm) direction.")
    return w / norm


def _diff_in_means_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Normalised difference-in-means direction from a LABELLED SUBSET.

    Called with a fold's training rows only. Same estimator as
    refusal_direction.compute_refusal_direction; duplicated here as a private
    helper so the fold loop cannot accidentally be handed a direction computed
    elsewhere over the full sample.
    """
    pos, neg = X[y == 1], X[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("Both classes must be present to define a direction.")
    d = pos.mean(axis=0) - neg.mean(axis=0)
    norm = np.linalg.norm(d)
    if norm < 1e-10:
        raise ValueError("Degenerate contrast: class centroids coincide.")
    return d / norm


# Smallest training partition on which the MLP's internal early-stopping split
# is viable. It holds out `validation_fraction` of the training rows and
# stratifies them, so the held-out part must contain at least one sample per
# class: n_train * 0.15 >= 2 gives n_train >= 14. Rounded up for margin.
_MIN_TRAIN_FOR_EARLY_STOPPING = 20


def _make_probe(probe_type: str, hidden_size: int, seed: int,
                n_train: Optional[int] = None):
    """
    Instantiate the estimator for a probe type.

    `n_train` governs whether the MLP may use early stopping. With a small
    training partition sklearn cannot carve a stratified validation split out
    of it and raises

        ValueError: The test_size = 1 should be greater or equal to the
        number of classes = 2

    which aborts the run. That is precisely the situation a `--limit-pairs`
    dry run creates: 8 samples over 5 folds leaves ~6 training rows, and 15%
    of 6 is one row. Since the dry run exists to validate the pipeline
    cheaply, it has to survive its own sample size. At the real n = 84 the
    training fold holds 67 rows and early stopping stays enabled, so nothing
    about the reported run changes.
    """
    if probe_type == "linear":
        return LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", random_state=seed
        )
    if probe_type == "mlp":
        early = n_train is None or n_train >= _MIN_TRAIN_FOR_EARLY_STOPPING
        return MLPClassifier(
            hidden_layer_sizes=(hidden_size,),
            max_iter=500,
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            early_stopping=early,
            validation_fraction=0.15 if early else 0.1,
            random_state=seed,
        )
    raise ValueError(f"probe_type must be 'linear' or 'mlp', got {probe_type!r}")


def _max_usable_folds(y: np.ndarray, n_folds: int) -> int:
    """K cannot exceed the size of the smallest class."""
    counts = np.bincount(y.astype(int), minlength=2)
    smallest = int(counts[counts > 0].min())
    return max(2, min(n_folds, smallest))


def _cv_fold_scores(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    probe_type: str,
    hidden_size: int,
    seed: int,
    with_direction: bool = True,
    groups: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    One complete K-fold pass with EVERY fitted object confined to the fold.

    Per fold, in this order:
        1. StandardScaler   fitted on train
        2. probe            fitted on train
        3. refusal direction (difference-in-means) computed on train
        4. all metrics evaluated on the held-out test rows

    Nothing derived from the test rows enters steps 1-3, which is the property
    that makes the resulting AUROC interpretable.

    `groups` carries the prompt-pair id of each row. When given, a fold holds
    out whole pairs: a harmful prompt and its benign twin share structure and
    vocabulary by construction, so testing on one while training on the other
    is not an out-of-sample evaluation of the harmful/benign contrast.

    Returns the per-fold statistics. Fold AUROCs are not pooled into a single
    number: each fold has its own scaler and probe, so scores from different
    folds are not on a common scale and concatenating them would compare
    incomparable quantities. The mean and sd across folds are the summary.
    """
    auroc_folds: List[float] = []
    acc_folds: List[float] = []
    f1_folds: List[float] = []
    dir_auroc_folds: List[float] = []
    cos_folds: List[float] = []

    for train_idx, test_idx in grouped_fold_indices(y, n_folds, seed, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # --- 1 & 2: standardiser and probe, TRAIN ONLY ---
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        clf = _make_probe(probe_type, hidden_size, seed, n_train=len(train_idx))
        clf.fit(X_tr_s, y_tr)

        y_pred = clf.predict(X_te_s)
        y_score = clf.predict_proba(X_te_s)[:, 1]

        acc_folds.append(float(accuracy_score(y_te, y_pred)))
        f1_folds.append(float(f1_score(y_te, y_pred, zero_division=0)))
        if len(np.unique(y_te)) > 1:
            auroc_folds.append(float(roc_auc_score(y_te, y_score)))

        # --- 3 & 4: refusal direction re-estimated on TRAIN ONLY ---
        if with_direction:
            try:
                d_k = _diff_in_means_direction(X_tr, y_tr)
            except ValueError:
                continue
            if len(np.unique(y_te)) > 1:
                dir_auroc_folds.append(float(roc_auc_score(y_te, X_te @ d_k)))
            if probe_type == "linear":
                try:
                    w_k = _probe_normal_original_space(clf, scaler)
                except ValueError:
                    pass
                else:
                    cos_folds.append(float(np.dot(w_k, d_k)))

    return {
        "auroc_folds": auroc_folds,
        "accuracy_folds": acc_folds,
        "f1_folds": f1_folds,
        "direction_auroc_folds": dir_auroc_folds,
        "cosine_folds": cos_folds,
    }


def _mean_sd(values: List[float]) -> Tuple[float, float]:
    """Mean and sample sd of a fold statistic; (nan, nan) when empty."""
    if not values:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, sd


def _nadeau_bengio_interval(
    fold_scores: List[float],
    n_train: int,
    n_test: int,
    ci_level: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Corrected standard error and interval for a (repeated) K-fold estimate.

    The naive standard error, s/sqrt(K*R), assumes the fold scores are
    independent. They are not: with K = 5 any two training folds share 60% of
    their samples, so the scores are positively correlated and the naive SE is
    too small — often by a factor of two or more. An interval built from it
    looks tight and under-covers.

    Nadeau and Bengio (2003) show the variance of the cross-validation estimate
    is approximated by

        Var = (1/(K*R) + n_test/n_train) * s^2

    where s^2 is the sample variance across all folds. The second term is the
    penalty for reusing the same data in overlapping training sets; it does not
    vanish as R grows, which is exactly the point — repeating the partition
    cannot buy information the 84 prompts do not contain.

    Args:
        fold_scores: every fold of every repeat.
        n_train: training-set size per fold.
        n_test: test-set size per fold.
        ci_level: confidence level.

    Returns:
        (corrected_se, ci_lower, ci_upper). AUROC is bounded in [0, 1], so the
        interval is clipped; a clipped endpoint means the estimate sits near
        the boundary and the normal approximation is straining.
    """
    m = len(fold_scores)
    if m < 2 or n_train <= 0:
        return float("nan"), float("nan"), float("nan")

    mean = float(np.mean(fold_scores))
    var = float(np.var(fold_scores, ddof=1))
    corrected_var = (1.0 / m + n_test / n_train) * var
    se = float(np.sqrt(max(corrected_var, 0.0)))

    from scipy import stats as _stats
    t_crit = float(_stats.t.ppf(0.5 + ci_level / 2.0, df=m - 1))

    return (
        se,
        float(np.clip(mean - t_crit * se, 0.0, 1.0)),
        float(np.clip(mean + t_crit * se, 0.0, 1.0)),
    )


# ============================================================
# 1. CROSS-VALIDATED PROBE  (the estimator to report)
# ============================================================

def cross_validate_probe(
    X: np.ndarray,
    y: np.ndarray,
    layer_idx: int = -1,
    probe_type: str = "linear",
    n_folds: int = 5,
    n_repeats: int = 1,
    hidden_size: int = 128,
    n_label_permutations: int = 0,
    ci_level: float = 0.95,
    seed: int = 42,
    groups: Optional[np.ndarray] = None,
) -> ProbeResult:
    """
    (Repeated) K-fold cross-validation with the FULL pipeline inside each fold.

    Args:
        X: Activations (n_samples, hidden_dim).
        y: Binary labels (n_samples,) — 1 = harmful, 0 = benign.
        layer_idx: Layer index, for reporting.
        probe_type: 'linear' or 'mlp'.
        n_folds: K. Clamped to the size of the smaller class.
        n_repeats: R — how many independent random partitions to average over.
            1 reproduces plain K-fold. At n = 84 a single partition puts ~17
            samples in each fold, so the estimate moves noticeably with the
            seed; 10 repeats stabilise it. This averages away the PARTITION
            variance only — see the module docstring on why that is not the
            same as reducing uncertainty.
        hidden_size: Hidden units, MLP only.
        n_label_permutations: Repeats of the label-permutation control. 0
            disables it. Each repeat shuffles y and re-runs the whole K-fold
            pass, so the resulting mean is the empirical chance level for this
            n and d — the number the reported AUROC must be read against.
        ci_level: Level for the Nadeau-Bengio corrected interval.
        seed: Random seed.
        groups: Prompt-pair id per row. When given, folds hold out whole
            pairs, and the label-permutation control shuffles labels WITHIN
            each pair - the exchangeability a matched design licenses.

    Returns:
        ProbeResult whose `auroc` is the cross-validated mean, with the sd,
        the per-fold values, the in-sample reference, the permutation control,
        the fold-honest refusal-direction metrics, and — when n_repeats > 1 —
        the per-repeat means and the corrected confidence interval.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)} labels.")
    if len(np.unique(y)) < 2:
        raise ValueError("Both classes must be present to probe.")

    k = _max_usable_folds(y, n_folds)
    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(y):
            raise ValueError(
                f"groups has {len(groups)} entries but y has {len(y)} labels."
            )
        k = max(2, min(k, len(np.unique(groups))))

    # ---- In-sample reference: the inflated number, kept for comparison ----
    scaler_full = StandardScaler()
    X_full_s = scaler_full.fit_transform(X)
    clf_full = _make_probe(probe_type, hidden_size, seed, n_train=len(y))
    clf_full.fit(X_full_s, y)
    auroc_in_sample = float(
        roc_auc_score(y, clf_full.predict_proba(X_full_s)[:, 1])
    )

    learned_direction = None
    if probe_type == "linear":
        try:
            learned_direction = _probe_normal_original_space(clf_full, scaler_full)
        except ValueError:
            learned_direction = None

    # ---- Cross-validated evaluation, averaged over R random partitions ----
    n_repeats = max(1, int(n_repeats))
    pooled: Dict[str, List[float]] = {
        "auroc_folds": [], "accuracy_folds": [], "f1_folds": [],
        "direction_auroc_folds": [], "cosine_folds": [],
    }
    repeat_means: List[float] = []

    for r in range(n_repeats):
        # A distinct seed per repeat is what makes the partitions independent;
        # reusing `seed` would draw the same split R times and report a
        # spuriously tiny spread.
        cv = _cv_fold_scores(
            X, y, k, probe_type, hidden_size,
            seed=seed + 1000 * r, with_direction=True, groups=groups,
        )
        for key in pooled:
            pooled[key].extend(cv[key])
        if cv["auroc_folds"]:
            repeat_means.append(float(np.mean(cv["auroc_folds"])))

    auroc_mean, auroc_sd = _mean_sd(pooled["auroc_folds"])
    acc_mean, acc_sd = _mean_sd(pooled["accuracy_folds"])
    f1_mean, f1_sd = _mean_sd(pooled["f1_folds"])
    dir_mean, dir_sd = _mean_sd(pooled["direction_auroc_folds"])
    cos_mean, cos_sd = _mean_sd(pooled["cosine_folds"])

    # With a single partition there is nothing to compare against, so the
    # partition spread is undefined. Reporting 0.0 would read as "perfectly
    # stable", which is the opposite of what one partition establishes.
    _, repeat_sd = _mean_sd(repeat_means) if n_repeats > 1 else (0.0, float("nan"))

    # Corrected interval. Fold sizes are near-equal under StratifiedKFold, so
    # the average test size is a faithful summary for the correction term.
    n_test = int(round(len(y) / k))
    se_corr, ci_lo, ci_hi = _nadeau_bengio_interval(
        pooled["auroc_folds"], n_train=len(y) - n_test, n_test=n_test,
        ci_level=ci_level,
    )

    # ---- Label-permutation control ----
    perm_mean = perm_sd = None
    if n_label_permutations > 0:
        rng = np.random.default_rng(seed)
        perm_scores: List[float] = []
        for r in range(n_label_permutations):
            if groups is None:
                y_shuffled = rng.permutation(y)
            else:
                # Shuffle within each pair: which of the two twins is called
                # harmful becomes a coin flip, while every fold stays balanced.
                y_shuffled = y.copy()
                for g in np.unique(groups):
                    idx = np.flatnonzero(groups == g)
                    y_shuffled[idx] = rng.permutation(y[idx])
            res = _cv_fold_scores(
                X, y_shuffled, k, probe_type, hidden_size,
                seed=seed + r + 1, with_direction=False, groups=groups,
            )
            if res["auroc_folds"]:
                perm_scores.append(float(np.mean(res["auroc_folds"])))
        perm_mean, perm_sd = _mean_sd(perm_scores)

    return ProbeResult(
        layer_idx=layer_idx,
        accuracy=acc_mean,
        auroc=auroc_mean,            # <- the cross-validated number
        f1=f1_mean,
        probe_type=probe_type,
        learned_direction=learned_direction,
        cosine_with_refusal=cos_mean if np.isfinite(cos_mean) else None,
        n_folds=k,
        auroc_std=auroc_sd,
        auroc_folds=[float(a) for a in pooled["auroc_folds"]],
        auroc_in_sample=auroc_in_sample,
        accuracy_std=acc_sd,
        f1_std=f1_sd,
        n_repeats=n_repeats,
        auroc_repeat_means=repeat_means,
        auroc_repeat_std=repeat_sd,
        auroc_se_corrected=se_corr,
        auroc_ci_lower=ci_lo,
        auroc_ci_upper=ci_hi,
        auroc_permuted_mean=perm_mean,
        auroc_permuted_std=perm_sd,
        n_label_permutations=n_label_permutations,
        auroc_direction_oof_mean=dir_mean,
        auroc_direction_oof_std=dir_sd,
        cosine_with_refusal_oof_mean=cos_mean,
        cosine_with_refusal_oof_std=cos_sd,
        n_samples=int(len(y)),
        n_positive=int((y == 1).sum()),
        folds_by_pair=groups is not None,
    )


def paired_fold_difference(
    scores_a: List[float],
    scores_b: List[float],
    n_train: int,
    n_test: int,
    ci_level: float = 0.95,
) -> Dict[str, float]:
    """
    Compare two probes fold by fold, on the SAME partitions.

    Two cross-validated AUROCs that look alike are not thereby equivalent, and
    two that differ are not thereby different: each carries a wide interval at
    this sample size, and the two intervals are computed on the same prompts,
    so they are strongly correlated. The quantity that answers the question is
    the per-fold DIFFERENCE, in which everything the two probes share - the
    difficulty of that particular held-out fold above all - cancels.

    The variance of the mean difference uses the Nadeau-Bengio correction for
    the same reason the single-probe interval does: training folds overlap, so
    the fold differences are not independent draws.

    Read the result as an interval, not as a verdict. An interval that
    contains zero says the data do not separate the probes; how tightly it
    brackets zero says how much of a difference could still be hiding. Only a
    narrow interval around zero supports "the probes perform the same".

    Args:
        scores_a, scores_b: per-fold AUROC of each probe. Entry j of both must
            come from the same train/test partition; only the first
            min(len) entries are used, which is what pairs a probe run with R
            repeats against one run with fewer.
        n_train, n_test: fold sizes, for the correction term.

    Returns:
        dict with the mean of each probe over the paired folds, the mean
        difference (a - b), its corrected SE, the interval and a two-sided p.
    """
    m = min(len(scores_a), len(scores_b))
    nan = float("nan")
    if m < 2 or n_train <= 0:
        return {"n_paired_folds": m, "mean_a": nan, "mean_b": nan,
                "difference": nan, "se_corrected": nan,
                "ci_lower": nan, "ci_upper": nan, "p_value": nan}

    a = np.asarray(scores_a[:m], dtype=float)
    b = np.asarray(scores_b[:m], dtype=float)
    d = a - b
    mean = float(d.mean())
    se = float(np.sqrt((1.0 / m + n_test / n_train) * np.var(d, ddof=1)))

    from scipy import stats as _stats
    t_crit = float(_stats.t.ppf(0.5 + ci_level / 2.0, df=m - 1))
    p = float(2 * _stats.t.sf(abs(mean) / se, df=m - 1)) if se > 0 else nan

    return {
        "n_paired_folds": int(m),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "difference": mean,
        "se_corrected": se,
        "ci_lower": mean - t_crit * se,
        "ci_upper": mean + t_crit * se,
        "p_value": p,
    }


def compare_linear_vs_mlp(
    linear_suite: "ProbingSuiteResult",
    mlp_suite: "ProbingSuiteResult",
) -> Dict[int, Dict[str, float]]:
    """
    Per-layer paired comparison MLP - linear on the folds both probes ran.

    Repeat r of either probe is drawn with seed + 1000*r, so the partitions
    coincide repeat for repeat. With fewer MLP repeats than linear ones only
    the shared repeats are paired, and the interval widens accordingly.
    """
    out: Dict[int, Dict[str, float]] = {}
    for layer, lin in linear_suite.per_layer.items():
        mlp = mlp_suite.per_layer.get(layer)
        if mlp is None or not lin.auroc_folds or not mlp.auroc_folds:
            continue
        n = lin.n_samples or 0
        k = lin.n_folds or 1
        n_test = int(round(n / k)) if n else 0
        out[layer] = paired_fold_difference(
            mlp.auroc_folds, lin.auroc_folds,
            n_train=n - n_test, n_test=n_test,
        )
    return out


# ============================================================
# 2. SINGLE-SPLIT PROBES  (legacy path, kept for comparison)
# ============================================================

def train_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    layer_idx: int,
    refusal_direction: Optional[np.ndarray] = None,
) -> ProbeResult:
    """
    Train a logistic regression probe on one fixed train/test split.

    Retained for comparison and for callers that supply their own split. Prefer
    `cross_validate_probe`: a single 80/20 split of 84 samples leaves a test set
    of 17, on which AUROC moves in steps of ~0.01 and a value of 1.0 carries no
    usable uncertainty.

    CAUTION on `refusal_direction`: if it was estimated over the full dataset,
    the returned cosine is contaminated — the test rows contributed to the
    direction. `cross_validate_probe` computes the honest per-fold cosine.

    Args:
        X_train: Training activations (n_train, hidden_dim)
        y_train: Training labels (n_train,) - 1=harmful, 0=benign
        X_test: Test activations (n_test, hidden_dim)
        y_test: Test labels (n_test,)
        layer_idx: Layer index for metadata.
        refusal_direction: Optional direction to compare against.

    Returns:
        ProbeResult with metrics and learned direction.
    """
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    # Learned direction, rescaled into the ORIGINAL activation space so that a
    # cosine against the refusal direction compares vectors in one metric.
    learned_direction = _probe_normal_original_space(clf, scaler)

    # Compare with refusal direction if provided
    cosine = None
    if refusal_direction is not None:
        cosine = float(np.dot(learned_direction, refusal_direction))

    return ProbeResult(
        layer_idx=layer_idx,
        accuracy=accuracy,
        auroc=auroc,
        f1=f1,
        probe_type="linear",
        learned_direction=learned_direction,
        cosine_with_refusal=cosine,
        n_samples=int(len(y_train) + len(y_test)),
    )


def train_mlp_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    layer_idx: int,
    hidden_size: int = 128,
) -> ProbeResult:
    """
    Train a non-linear (MLP) probe on one fixed train/test split.

    If MLP works but linear doesn't, the information exists
    but is NOT linearly separable (not a single direction).

    Args:
        X_train: Training activations (n_train, hidden_dim)
        y_train: Training labels (n_train,)
        X_test: Test activations (n_test, hidden_dim)
        y_test: Test labels (n_test,)
        layer_idx: Layer index for metadata.
        hidden_size: Hidden layer size for MLP.

    Returns:
        ProbeResult with metrics.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = MLPClassifier(
        hidden_layer_sizes=(hidden_size,),
        max_iter=500,
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    return ProbeResult(
        layer_idx=layer_idx,
        accuracy=accuracy,
        auroc=auroc,
        f1=f1,
        probe_type="mlp",
        n_samples=int(len(y_train) + len(y_test)),
    )


# ============================================================
# 3. SUITE ACROSS LAYERS
# ============================================================

def run_probing_suite(
    activations: Dict[int, np.ndarray],
    labels: np.ndarray,
    refusal_directions: Optional[Dict[int, np.ndarray]] = None,
    test_size: float = 0.2,
    n_folds: int = 5,
    n_repeats: int = 1,
    mlp_n_repeats: Optional[int] = None,
    use_cv: bool = True,
    n_label_permutations: int = 10,
    seed: int = 42,
    groups: Optional[np.ndarray] = None,
) -> Tuple[ProbingSuiteResult, ProbingSuiteResult]:
    """
    Run the probing suite (linear + MLP) across all layers.

    By default every layer is evaluated by K-fold cross-validation with the
    standardiser, the probe AND the difference-in-means direction all refitted
    inside each fold. `ProbeResult.auroc` therefore carries the cross-validated
    mean, and the in-sample value is kept alongside it for comparison rather
    than reported as the result.

    The `n_folds` argument used to be accepted and silently ignored: the suite
    ran a single 80/20 split. That is the discrepancy this function fixes.

    Args:
        activations: Dict mapping layer_idx -> (n_samples, hidden_dim)
        labels: Binary labels (n_samples,) - 1=harmful, 0=benign
        refusal_directions: Optional dict of full-sample directions per layer.
            Used ONLY for the legacy single-split path; the cross-validated
            path recomputes the direction inside each fold, which is the point.
        test_size: Fraction for the test set, single-split path only.
        n_folds: Number of cross-validation folds.
        n_repeats: Independent random partitions to average over. 1 is plain
            K-fold; 10 is the recommended setting for a reported run.
        mlp_n_repeats: Separate repeat count for the MLP, which costs far more
            per fit than the logistic probe. None means "same as n_repeats";
            set it to 1 to keep a high-repeat linear run affordable.
        use_cv: False restores the old single-split behaviour.
        n_label_permutations: Repeats of the label-permutation control for the
            linear probe (0 disables). The MLP control is skipped by default —
            it costs K fits per repeat and the linear probe is the one whose
            perfect AUROC is under dispute.
        seed: Random seed.
        groups: Prompt-pair id per row; folds then hold out whole pairs. Both
            probes receive the same groups and the same seeds, so their folds
            coincide and `compare_linear_vs_mlp` can pair them.

    Returns:
        Tuple of (linear_results, mlp_results)
    """
    from sklearn.model_selection import train_test_split

    labels = np.asarray(labels).astype(int)
    linear_results: Dict[int, ProbeResult] = {}
    mlp_results: Dict[int, ProbeResult] = {}

    for layer_idx in sorted(activations.keys()):
        X = activations[layer_idx]

        if use_cv:
            linear_results[layer_idx] = cross_validate_probe(
                X, labels, layer_idx=layer_idx, probe_type="linear",
                n_folds=n_folds, n_repeats=n_repeats,
                n_label_permutations=n_label_permutations, seed=seed,
                groups=groups,
            )
            mlp_results[layer_idx] = cross_validate_probe(
                X, labels, layer_idx=layer_idx, probe_type="mlp",
                n_folds=n_folds,
                n_repeats=n_repeats if mlp_n_repeats is None else mlp_n_repeats,
                n_label_permutations=0, seed=seed,
                groups=groups,
            )
            continue

        # ---- Legacy single-split path ----
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=test_size, random_state=seed, stratify=labels
        )

        ref_dir = None
        if refusal_directions and layer_idx in refusal_directions:
            ref_dir = refusal_directions[layer_idx]

        linear_results[layer_idx] = train_linear_probe(
            X_train, y_train, X_test, y_test, layer_idx, ref_dir
        )
        mlp_results[layer_idx] = train_mlp_probe(
            X_train, y_train, X_test, y_test, layer_idx
        )

    # Find best layers. With CV this ranks on the out-of-fold mean; ties are
    # far less likely than on the saturated in-sample column.
    def _score(res: ProbeResult) -> float:
        return res.auroc if np.isfinite(res.auroc) else -1.0

    best_linear_layer = max(linear_results, key=lambda l: _score(linear_results[l]))
    best_mlp_layer = max(mlp_results, key=lambda l: _score(mlp_results[l]))

    linear_suite = ProbingSuiteResult(
        per_layer=linear_results,
        best_layer=best_linear_layer,
        best_auroc=linear_results[best_linear_layer].auroc,
        probe_type="linear",
        cross_validated=use_cv,
        n_folds=n_folds if use_cv else None,
        n_repeats=n_repeats if use_cv else None,
    )

    mlp_suite = ProbingSuiteResult(
        per_layer=mlp_results,
        best_layer=best_mlp_layer,
        best_auroc=mlp_results[best_mlp_layer].auroc,
        probe_type="mlp",
        cross_validated=use_cv,
        n_folds=n_folds if use_cv else None,
        n_repeats=(n_repeats if mlp_n_repeats is None else mlp_n_repeats)
                  if use_cv else None,
    )

    return linear_suite, mlp_suite


def compare_probe_vs_direction(
    linear_suite: ProbingSuiteResult,
    refusal_directions: Dict[int, np.ndarray],
) -> Dict[int, float]:
    """
    Compare learned probe directions vs analytical refusal directions.

    High cosine similarity = the probe found the SAME direction as
    difference-in-means. Low similarity = probe found a DIFFERENT
    direction that also separates harmful/benign.

    When the suite was cross-validated, the per-fold cosine already on each
    ProbeResult is preferred: both vectors were fitted on the same training
    partition, so neither has seen the evaluation fold. The full-sample cosine
    is used only as a fallback for the legacy path.

    Args:
        linear_suite: Results from linear probing.
        refusal_directions: Analytical refusal directions per layer.

    Returns:
        Dict mapping layer_idx -> cosine similarity
    """
    cosines: Dict[int, float] = {}
    for layer_idx, result in linear_suite.per_layer.items():
        oof = result.cosine_with_refusal_oof_mean
        if oof is not None and np.isfinite(oof):
            cosines[layer_idx] = float(oof)
        elif result.learned_direction is not None and layer_idx in refusal_directions:
            cosines[layer_idx] = float(
                np.dot(result.learned_direction, refusal_directions[layer_idx])
            )
    return cosines


def print_probing_summary(
    linear_suite: ProbingSuiteResult,
    mlp_suite: ProbingSuiteResult,
):
    """Print a summary table of probing results."""
    cv = linear_suite.cross_validated
    reps = linear_suite.n_repeats or 1
    print("\n" + "=" * 96)
    if cv:
        scheme = (f"{linear_suite.n_folds}-fold" if reps <= 1
                  else f"{reps}x{linear_suite.n_folds}-fold repeated")
        print(f"PROBING RESULTS  ({scheme} cross-validated; "
              "scaler, probe AND direction refitted inside each fold)")
    else:
        print("PROBING RESULTS SUMMARY  (single split — NOT cross-validated)")
    print("=" * 96)

    if not cv:
        print(f"\n{'Layer':<8} {'Linear AUROC':<15} {'MLP AUROC':<15} {'Cosine w/ RefDir':<18}")
        print("-" * 56)
        for layer_idx in sorted(linear_suite.per_layer.keys()):
            lin = linear_suite.per_layer[layer_idx]
            mlp = mlp_suite.per_layer[layer_idx]
            cos = lin.cosine_with_refusal
            cosine_str = f"{cos:.3f}" if cos is not None else "N/A"
            print(f"{layer_idx:<8} {lin.auroc:<15.4f} {mlp.auroc:<15.4f} {cosine_str:<18}")
        print(f"\nBest linear: layer {linear_suite.best_layer} "
              f"(AUROC={linear_suite.best_auroc:.4f})")
        print(f"Best MLP:    layer {mlp_suite.best_layer} "
              f"(AUROC={mlp_suite.best_auroc:.4f})")
        return

    header = (f"{'Layer':<7}{'linear CV':>18}{'in-samp':>10}{'shuffled':>11}"
              f"{'MLP CV':>18}{'d_refusal CV':>18}{'cos(w,d) CV':>16}")
    print("\n" + header)
    print("-" * len(header))

    for layer_idx in sorted(linear_suite.per_layer.keys()):
        lin = linear_suite.per_layer[layer_idx]
        mlp = mlp_suite.per_layer[layer_idx]

        lin_str = f"{lin.auroc:.3f}+-{lin.auroc_std:.3f}"
        mlp_str = f"{mlp.auroc:.3f}+-{mlp.auroc_std:.3f}"
        dir_str = (f"{lin.auroc_direction_oof_mean:.3f}"
                   f"+-{lin.auroc_direction_oof_std:.3f}")
        cos_str = (f"{lin.cosine_with_refusal_oof_mean:.3f}"
                   f"+-{lin.cosine_with_refusal_oof_std:.3f}")
        perm_str = (f"{lin.auroc_permuted_mean:.3f}"
                    if lin.auroc_permuted_mean is not None
                    and np.isfinite(lin.auroc_permuted_mean) else "n/a")

        print(f"{layer_idx:<7}{lin_str:>18}{lin.auroc_in_sample:>10.3f}"
              f"{perm_str:>11}{mlp_str:>18}{dir_str:>18}{cos_str:>16}")

    best_lin = linear_suite.per_layer[linear_suite.best_layer]
    best_mlp = mlp_suite.per_layer[mlp_suite.best_layer]

    print()
    print(f"  Best linear : layer {linear_suite.best_layer}  "
          f"AUROC_cv = {best_lin.auroc:.4f} +- {best_lin.auroc_std:.4f}  "
          f"(in-sample {best_lin.auroc_in_sample:.4f})")
    print(f"  Best MLP    : layer {mlp_suite.best_layer}  "
          f"AUROC_cv = {best_mlp.auroc:.4f} +- {best_mlp.auroc_std:.4f}  "
          f"(in-sample {best_mlp.auroc_in_sample:.4f})")
    print(f"  n = {best_lin.n_samples} samples "
          f"({best_lin.n_positive} harmful), K = {best_lin.n_folds} folds"
          + (f", R = {best_lin.n_repeats} repeats "
             f"({len(best_lin.auroc_folds)} fold scores)"
             if (best_lin.n_repeats or 1) > 1 else ""))

    # --- The three numbers, and what each one is for ---
    if best_lin.auroc_se_corrected is not None and np.isfinite(best_lin.auroc_se_corrected):
        print()
        print(f"  Uncertainty at layer {linear_suite.best_layer}:")
        print(f"    sd across folds            : {best_lin.auroc_std:.4f}   "
              "spread of individual fold scores")
        if (best_lin.n_repeats or 1) > 1:
            r = best_lin.n_repeats
            print(f"    sd across repeat-means     : {best_lin.auroc_repeat_std:.4f}   "
                  "how much the answer moves")
            print( "                                          between partitions "
                  "(not a CI)")
            print(f"      -> its share of the reported mean: "
                  f"{best_lin.auroc_repeat_std / np.sqrt(r):.4f}  "
                  f"(= sd/sqrt({r}))")
        print(f"    Nadeau-Bengio 95% CI       : "
              f"[{best_lin.auroc_ci_lower:.4f}, {best_lin.auroc_ci_upper:.4f}]   "
              "<- quote this")
        print(f"    corrected SE               : {best_lin.auroc_se_corrected:.4f}   "
              "accounts for overlapping train folds")
        naive_se = (best_lin.auroc_std / np.sqrt(len(best_lin.auroc_folds))
                    if best_lin.auroc_folds else float("nan"))
        if np.isfinite(naive_se) and naive_se > 1e-9:
            print(f"    (the naive SE would be {naive_se:.4f}, i.e. "
                  f"{best_lin.auroc_se_corrected / naive_se:.1f}x too small)")

    # --- Label-permutation control: the reference the AUROC must beat ---
    perm = best_lin.auroc_permuted_mean
    if perm is not None and np.isfinite(perm):
        print()
        print(f"  Label-permutation control at layer {linear_suite.best_layer} "
              f"({best_lin.n_label_permutations} repeats):")
        print(f"    AUROC_cv on SHUFFLED labels : {perm:.4f} "
              f"+- {best_lin.auroc_permuted_std:.4f}   <- empirical chance here")
        print(f"    AUROC_cv on TRUE labels     : {best_lin.auroc:.4f} "
              f"+- {best_lin.auroc_std:.4f}")
        margin = best_lin.auroc - perm
        print(f"    margin over chance          : {margin:+.4f}")
        if margin < 0.05:
            print("    -> The probe does not beat its own chance level. The "
                  "separation is\n       NOT established at this sample size.")
        elif margin < 0.15:
            print("    -> Modest margin over chance. Report the CV mean with its "
                  "sd and\n       treat the effect as suggestive, not established.")
        else:
            print("    -> Clear margin over chance: the separation survives "
                  "cross-validation.")

    # --- In-sample inflation, stated explicitly ---
    inflation = best_lin.auroc_in_sample - best_lin.auroc
    print()
    print(f"  In-sample inflation at the best layer: {inflation:+.4f}")
    print("    In-sample AUROC is guaranteed high at n << d (Cover's theorem) "
          "and is\n    reported only to quantify this gap. Report the CV mean "
          "with its sd.")

    # --- Linear vs non-linear, judged on the cross-validated numbers ---
    gap = mlp_suite.best_auroc - linear_suite.best_auroc
    print()
    if not np.isfinite(gap):
        print("  Linear/MLP comparison unavailable (a probe failed to score).")
    elif gap < -0.05:
        # Out-of-fold the MLP can land BELOW the linear probe, which the old
        # in-sample comparison never showed. That is not evidence of linearity;
        # it means the MLP has too many parameters for this sample and its
        # excess capacity costs it generalisation. Read it as "no non-linear
        # component was recoverable at n = {n}", not as a property of the
        # representation.
        print(f"  -> MLP is WORSE than linear out-of-fold ({gap:+.3f}). At this "
              "sample size the\n     MLP cannot generalise; this says nothing "
              "about whether a non-linear\n     component exists. Read the "
              "linear result on its own.")
    elif gap < 0.05:
        print("  -> Linear probe matches MLP: information is LINEARLY SEPARABLE")
        print("     (exists as a direction in activation space)")
    elif gap < 0.15:
        print("  -> MLP slightly better: information mostly linear with some "
              "non-linear component")
    else:
        print("  -> MLP significantly better: information exists but is NOT a "
              "simple direction")
        print("     (more complex representation)")
    print("=" * 96)
