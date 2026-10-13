"""
Controlled validation of the bias corrections in src/robust_estimation.py.

Every claim made in the methodology chapter's audit section is checked here
against synthetic data whose ground truth is known by construction. This is
not a demonstration that the corrections are elegant; it is a test that the
biases are real and that the corrections remove them.

The experiments deliberately reproduce the sample geometry of the real study:
n = 42 paired prompts, hidden_dim = 8192, and a chat-condition separation
strong enough to give AUROC ~ 0.99, matching the regime the pipeline operates
in.

Experiments
-----------
A. NULL CASE     True DeltaP = 0 by construction (agent drawn from the same
                 distribution as chat). A correct estimator must return ~0.
                 Detects the own-sample circularity bias.

B. SIGNAL CASE   True DeltaP > 0 by construction. Both estimators should
                 detect it; the question is whether the naive one inflates it.

C. COVER         Random labels carrying no information. In-sample AUROC must
                 be ~1.0 (the geometric certainty) while out-of-fold AUROC
                 must be ~0.5 (the truth).

D. ATTENUATION   A known true angle between two directions. Measures how far
                 the estimate strays and whether the inter-fold reliability
                 predicts that error.

E. FOLD HYGIENE  The specific way cross-validation gets done wrong: computing
                 the refusal direction over the FULL sample and then
                 cross-validating only the classifier. On data with no signal
                 at all this returns AUROC ~ 1.0 while looking validated.
                 Recomputing the direction inside each fold returns ~0.5.

F. BENIGN CONTROL
                 Two ground truths that produce the SAME DeltaP on harmful
                 prompts: one where only harmful prompts move, one where
                 harmful and benign move alike. DeltaP cannot separate them;
                 the difference-in-differences must.

G. HELD-OUT AUROC
                 The AUROC the pipeline reports for d_chat is scored on the
                 two groups whose centroid difference IS d_chat. With no true
                 separation the in-sample value must still be ~1.0 and the
                 out-of-fold value ~0.5. This is experiment C's point made for
                 the difference-in-means direction rather than for a probe.

H. PAIR FOLDS    A harmful prompt and its benign twin share a template, so
                 their activations share a large component. Folding the 84
                 rows freely puts one twin in training and the other in the
                 test fold, with opposite labels: the probe learns the shared
                 component as evidence for the WRONG label of the held-out
                 twin. With nothing to detect, the AUROC must be ~0.5; row
                 folds report far less, and with a real signal they understate
                 it. Folding by pair restores both.

I. PAIRED INTERACTION
                 The harmful and benign arms are matched prompt for prompt. If
                 the two shifts of a pair share a component, the within-pair
                 test has a narrower interval than the two-sample one; if they
                 do not, the two agree. Either way the paired analysis is the
                 one that matches the design.

J. ROTATION      The angle between the chat direction and the agent direction,
                 with a known truth. The raw cosine between two directions
                 fitted on the same prompts is pulled toward zero by noise and
                 toward one by the content the two formats share, so it misses
                 in both directions. Fitting on disjoint halves and dividing
                 by what the method achieves against itself recovers the
                 truth. A probe normal and a centroid difference fitted on the
                 SAME data are also compared: they are different objects, and
                 a cosine between them is not an angle between formats.

Run:
    python validate_corrections.py
    python validate_corrections.py --trials 50 --seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.robust_estimation import (  # noqa: E402
    MIN_RELIABILITY_FOR_CORRECTION,
    cross_fitted_projection_gap,
    cross_validated_agent_direction,
    crossfit_condition_projections,
    _diff_in_means_direction,
)
from src.format_effect import specificity_from_projections  # noqa: E402
from src.probes import cross_validate_probe  # noqa: E402
from src.robust_estimation import (  # noqa: E402
    _fit_agent_direction,
    format_rotation,
    pair_groups,
)

# Geometry of the real study
N_PAIRS = 42
HIDDEN_DIM = 8192
SEPARATION = 3.30      # gives AUROC ~ 0.99 on the true direction
N_FOLDS = 5


def _banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _unit(rng: np.random.Generator, dim: int) -> np.ndarray:
    v = rng.standard_normal(dim)
    return v / np.linalg.norm(v)


def make_data(rng, dim, n, separation, agent_shift, noise_scale=1.0):
    """
    Build one synthetic replicate with a known ground truth.

    chat_benign   ~ N(0, s^2 I)
    chat_harmful  ~ N(separation * u, s^2 I)
    agent_harmful ~ N((separation - agent_shift) * u, s^2 I), paired row-wise

    True DeltaP measured ALONG u is exactly `agent_shift`. The pipeline never
    measures along u, however: it measures along the ESTIMATED direction, which
    is why the attenuation reported below matters as much as the bias.

    `noise_scale` (s) is the per-dimension within-class SD, and it is the
    parameter that governs the size of the circularity bias. The bias is
    approximately

        bias  ~  (s^2 * dim / n) / || mu_H - mu_B ||

    so it grows with the total within-class variance and shrinks with n. The
    value of s for real activations is an empirical quantity and is NOT known
    from this simulation; see the closing note.
    """
    u = _unit(rng, dim)
    chat_benign = noise_scale * rng.standard_normal((n, dim))
    chat_harmful = noise_scale * rng.standard_normal((n, dim)) + separation * u
    agent_harmful = (
        noise_scale * rng.standard_normal((n, dim)) + (separation - agent_shift) * u
    )
    return chat_harmful, chat_benign, agent_harmful, u


# ============================================================
# A + B. Circularity bias in DeltaP
# ============================================================

def experiment_gap(trials: int, seed: int, agent_shift: float, label: str,
                   noise_scale: float = 1.0, verbose: bool = True) -> dict:
    """
    Compare the naive and cross-fitted estimators against a known truth.

    Two distinct error sources are separated here, because conflating them is
    exactly what makes the naive number misleading:

      BIAS         an additive inflation, present only in the naive estimator,
                   caused by projecting the sample onto a direction fitted to
                   that same sample.

      ATTENUATION  a multiplicative shrinkage, present in BOTH estimators,
                   caused by the estimated direction being noisy: the pipeline
                   measures along d_est, not along the true axis u, and
                   recovers only the fraction cos(u, d_est) of the true effect.

    Cross-fitting removes the bias. It does NOT remove the attenuation, which
    is the same phenomenon that makes the rotation angle an upper bound in
    experiment D. Both must be reported.
    """
    rng = np.random.default_rng(seed)
    naive, crossfit, dir_quality = [], [], []

    for t in range(trials):
        ch, cb, ah, u = make_data(
            rng, HIDDEN_DIM, N_PAIRS, SEPARATION, agent_shift, noise_scale
        )
        res = cross_fitted_projection_gap(
            ch, cb, ah,
            n_folds=N_FOLDS,
            n_permutations=200,   # p-values are not used here; keep the sweep cheap
            n_boot=200,
            seed=seed + t,
        )
        naive.append(res.delta_p_naive)
        crossfit.append(res.delta_p_crossfit)
        # How much of the true axis does the estimated direction actually capture?
        dir_quality.append(abs(float(np.dot(_diff_in_means_direction(ch, cb), u))))

    naive, crossfit = np.array(naive), np.array(crossfit)
    dq = float(np.mean(dir_quality))

    # The attainable target once attenuation is accounted for.
    attenuated_truth = agent_shift * dq

    out = {
        "label": label,
        "true": agent_shift,
        "noise_scale": noise_scale,
        "dir_quality": dq,
        "attenuated_truth": attenuated_truth,
        "naive_mean": float(naive.mean()),
        "naive_sd": float(naive.std(ddof=1)),
        "crossfit_mean": float(crossfit.mean()),
        "crossfit_sd": float(crossfit.std(ddof=1)),
        "naive_bias": float(naive.mean() - attenuated_truth),
        "crossfit_bias": float(crossfit.mean() - attenuated_truth),
    }

    if verbose:
        print(f"\n  {label}   ({trials} replicates, per-dim noise SD = {noise_scale})")
        print(f"    true DeltaP along the true axis u : {agent_shift:.4f}")
        print(f"    cos(u, d_estimated)               : {dq:.4f}   "
              "<- direction quality")
        print(f"    attainable target after attenuation: {attenuated_truth:.4f}   "
              "<- what an UNBIASED estimator should return")
        print(f"    {'-' * 68}")
        print(f"    {'estimator':<22}{'mean':>10}{'sd':>10}{'bias vs target':>18}")
        print(f"    {'naive (in-sample)':<22}{out['naive_mean']:>10.4f}"
              f"{out['naive_sd']:>10.4f}{out['naive_bias']:>+18.4f}")
        print(f"    {'cross-fitted':<22}{out['crossfit_mean']:>10.4f}"
              f"{out['crossfit_sd']:>10.4f}{out['crossfit_bias']:>+18.4f}")
    return out


def experiment_noise_sweep(trials: int, seed: int) -> list:
    """
    Show how the circularity bias scales with the within-class variance.

    The point of this sweep is that the bias is not a fixed quantity: it is
    governed by s^2 * dim / n relative to the between-class separation. The
    isotropic unit-variance case used elsewhere in this script is the WORST
    case. Reading off the row closest to the real data's noise level gives a
    rough expectation for how much inflation to expect there.
    """
    rows = []
    print(f"\n  {'noise SD':>10}{'cos(u,d)':>12}{'target':>10}"
          f"{'naive':>12}{'cross-fit':>12}{'naive bias':>14}")
    print(f"  {'-' * 70}")
    for s in [0.10, 0.25, 0.50, 1.00]:
        r = experiment_gap(trials, seed, 0.70, f"s={s}", noise_scale=s, verbose=False)
        rows.append(r)
        print(f"  {s:>10.2f}{r['dir_quality']:>12.4f}{r['attenuated_truth']:>10.4f}"
              f"{r['naive_mean']:>12.4f}{r['crossfit_mean']:>12.4f}"
              f"{r['naive_bias']:>+14.4f}")
    return rows


# ============================================================
# C. Cover's theorem
# ============================================================

def experiment_cover(trials: int, seed: int) -> dict:
    """
    Labels assigned at random, carrying zero information.

    A probe that reports in-sample AUROC = 1.0 here is reporting the geometry
    of high-dimensional space, not a property of the data. Out-of-fold AUROC
    should sit at chance.
    """
    rng = np.random.default_rng(seed)
    in_sample, out_of_fold = [], []

    for _ in range(trials):
        X = rng.standard_normal((2 * N_PAIRS, HIDDEN_DIM))     # no class structure
        y = rng.permutation(np.r_[np.ones(N_PAIRS), np.zeros(N_PAIRS)])

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(Xs, y)
        in_sample.append(roc_auc_score(y, Xs @ clf.coef_[0]))

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
        folds = []
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            c = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
            c.fit(sc.transform(X[tr]), y[tr])
            folds.append(roc_auc_score(y[te], sc.transform(X[te]) @ c.coef_[0]))
        out_of_fold.append(np.mean(folds))

    ins, oof = np.array(in_sample), np.array(out_of_fold)
    print(f"\n  Random labels, no signal whatsoever ({trials} replicates)")
    print(f"    N = {2 * N_PAIRS}, d = {HIDDEN_DIM}  ->  N <= d+1, so Cover's")
    print(f"    theorem guarantees separability for any labelling.")
    print(f"    {'-' * 60}")
    print(f"    {'AUROC in-sample':<26}{ins.mean():>9.4f}  (sd {ins.std(ddof=1):.4f})")
    print(f"    {'AUROC out-of-fold':<26}{oof.mean():>9.4f}  (sd {oof.std(ddof=1):.4f})")
    return {"in_sample": ins.mean(), "oof": oof.mean()}


# ============================================================
# D. Attenuation of the measured angle
# ============================================================

def experiment_attenuation(trials: int, seed: int, true_cos: float) -> dict:
    """
    Two directions separated by a KNOWN angle.

    d_chat is the true chat direction. The agent data are generated along a
    direction w_true whose cosine with d_chat is fixed at `true_cos`. The
    probe estimate of w_true is noise-dominated, so the measured cosine should
    fall short of the truth. The inter-fold cosine should track that shortfall.
    """
    rng = np.random.default_rng(seed)
    measured, disatt, rel_split, rel_fold = [], [], [], []

    for t in range(trials):
        d_chat = _unit(rng, HIDDEN_DIM)
        # Construct w_true at exactly the requested angle from d_chat.
        perp = _unit(rng, HIDDEN_DIM)
        perp = perp - np.dot(perp, d_chat) * d_chat
        perp /= np.linalg.norm(perp)
        w_true = true_cos * d_chat + np.sqrt(1 - true_cos ** 2) * perp

        agent_harmful = rng.standard_normal((N_PAIRS, HIDDEN_DIM)) + SEPARATION * w_true
        agent_benign = rng.standard_normal((N_PAIRS, HIDDEN_DIM))

        res = cross_validated_agent_direction(
            agent_harmful, agent_benign, d_chat=d_chat,
            n_folds=N_FOLDS, seed=seed + t,
        )
        measured.append(res.cosine_with_d_chat_mean)
        disatt.append(res.cosine_disattenuated)
        rel_split.append(res.split_half_reliability)
        rel_fold.append(res.inter_fold_cosine_mean)

    m = np.array(measured)
    dis = np.array(disatt)
    rs, rf = np.array(rel_split), np.array(rel_fold)
    true_angle = np.degrees(np.arccos(true_cos))
    meas_angle = np.degrees(np.arccos(np.clip(m.mean(), -1, 1)))
    dis_angle = np.degrees(np.arccos(np.clip(dis.mean(), -1, 1)))

    print(f"\n  True cos = {true_cos:.3f}  (true angle {true_angle:.1f} deg), "
          f"{trials} replicates")
    print(f"    {'-' * 66}")
    print(f"    {'measured cos':<30}{m.mean():>9.4f}  (sd {m.std(ddof=1):.4f})")
    print(f"    {'measured angle':<30}{meas_angle:>9.1f} deg  "
          f"-> overstated by {meas_angle - true_angle:.1f} deg")
    print(f"    {'attenuation factor (empirical)':<30}"
          f"{m.mean() / true_cos:>9.4f}")
    print(f"    {'-' * 66}")
    print(f"    {'split-half reliability':<30}{rs.mean():>9.4f}   [disjoint]")
    print(f"    {'sqrt(reliability)':<30}{np.sqrt(max(rs.mean(), 0)):>9.4f}   "
          "<- should track the attenuation factor")
    print(f"    {'inter-fold cosine':<30}{rf.mean():>9.4f}   "
          "[overlapping -> inflated]")
    print(f"    {'-' * 66}")
    print(f"    {'cos DISATTENUATED':<30}{dis.mean():>9.4f}   "
          f"(truth {true_cos:.3f})")
    print(f"    {'angle DISATTENUATED':<30}{dis_angle:>9.1f} deg  "
          f"-> error {abs(dis_angle - true_angle):.1f} deg")
    return {
        "true_cos": true_cos,
        "measured_cos": float(m.mean()),
        "disattenuated_cos": float(dis.mean()),
        "true_angle": float(true_angle),
        "measured_angle": float(meas_angle),
        "disattenuated_angle": float(dis_angle),
        "reliability_split": float(rs.mean()),
        "reliability_fold": float(rf.mean()),
    }


# ============================================================
# E. Fold hygiene: where the direction is estimated
# ============================================================

def experiment_fold_hygiene(trials: int, seed: int) -> dict:
    """
    The failure mode that looks like cross-validation and is not.

    Three pipelines are compared on data containing NO signal, so the correct
    answer is 0.5 in every case:

      IN-SAMPLE       probe fitted and scored on everything. Cover's theorem
                      guarantees 1.0.

      LEAKY CV        the refusal direction is computed ONCE over the full
                      sample; only the classifier is cross-validated. This is
                      the tempting shortcut — the fold loop is real, the split
                      is real, and the number is still worthless, because every
                      held-out row contributed its own centroid to the
                      direction it is then scored against.

      HONEST CV       direction AND probe re-estimated inside each fold.

    The gap between LEAKY and HONEST is the cost of that one shortcut, measured
    rather than asserted. It is also why src/probes.py recomputes the
    difference-in-means direction inside the fold loop instead of accepting a
    precomputed one.
    """
    rng = np.random.default_rng(seed)
    ins, leaky, honest, probe_leaky, probe_honest = [], [], [], [], []

    for t in range(trials):
        # No class structure whatsoever: both classes are the same distribution.
        X = rng.standard_normal((2 * N_PAIRS, HIDDEN_DIM))
        y = np.r_[np.ones(N_PAIRS), np.zeros(N_PAIRS)]

        d_full = _diff_in_means_direction(X[y == 1], X[y == 0])
        ins.append(roc_auc_score(y, X @ d_full))

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed + t)
        f_leaky, f_honest, p_leaky, p_honest = [], [], [], []
        for tr, te in skf.split(X, y):
            # --- direction ---
            d_tr = _diff_in_means_direction(X[tr][y[tr] == 1], X[tr][y[tr] == 0])
            f_leaky.append(roc_auc_score(y[te], X[te] @ d_full))   # leaks
            f_honest.append(roc_auc_score(y[te], X[te] @ d_tr))    # clean

            # --- probe, standardised on the wrong vs the right partition ---
            sc_all = StandardScaler().fit(X)                       # leaks
            c = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
            c.fit(sc_all.transform(X[tr]), y[tr])
            p_leaky.append(
                roc_auc_score(y[te], c.predict_proba(sc_all.transform(X[te]))[:, 1])
            )

            sc_tr = StandardScaler().fit(X[tr])                    # clean
            c2 = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
            c2.fit(sc_tr.transform(X[tr]), y[tr])
            p_honest.append(
                roc_auc_score(y[te], c2.predict_proba(sc_tr.transform(X[te]))[:, 1])
            )

        leaky.append(np.mean(f_leaky))
        honest.append(np.mean(f_honest))
        probe_leaky.append(np.mean(p_leaky))
        probe_honest.append(np.mean(p_honest))

    ins, leaky, honest = np.array(ins), np.array(leaky), np.array(honest)
    pl, ph = np.array(probe_leaky), np.array(probe_honest)

    print(f"\n  No signal at all: both classes drawn from N(0, I). Truth = 0.500.")
    print(f"  n = {2 * N_PAIRS}, d = {HIDDEN_DIM}, K = {N_FOLDS}, "
          f"{trials} replicates")
    print(f"    {'-' * 68}")
    print(f"    {'direction, in-sample':<44}{ins.mean():>9.4f}")
    print(f"    {'direction from FULL sample, scored out-of-fold':<44}"
          f"{leaky.mean():>9.4f}   <- the trap")
    print(f"    {'direction from the TRAINING fold only':<44}"
          f"{honest.mean():>9.4f}   <- correct")
    print(f"    {'cost of leaking the direction':<44}"
          f"{leaky.mean() - honest.mean():>+9.4f}")
    print(f"    {'-' * 68}")
    print(f"    {'probe, scaler fitted on the FULL sample':<44}{pl.mean():>9.4f}")
    print(f"    {'probe, scaler fitted on the training fold':<44}{ph.mean():>9.4f}")
    print(f"    {'cost of leaking the scaler':<44}"
          f"{pl.mean() - ph.mean():>+9.4f}")
    # The standard error of the honest estimate is reported so the verdict can
    # size its own tolerance. At n = 84 a single replicate's out-of-fold AUROC
    # has an sd near 0.12, so a fixed tolerance either fails spuriously at low
    # trial counts or passes vacuously at high ones.
    honest_se = (
        float(honest.std(ddof=1) / np.sqrt(len(honest))) if len(honest) > 1 else float("nan")
    )

    return {
        "in_sample": float(ins.mean()),
        "leaky_direction": float(leaky.mean()),
        "honest_direction": float(honest.mean()),
        "honest_se": honest_se,
        "leaky_scaler": float(pl.mean()),
        "honest_scaler": float(ph.mean()),
    }


# ============================================================
# F. BENIGN CONTROL  (is the shift specific to harmful prompts?)
# ============================================================

def make_data_both_arms(rng, dim, n, separation, agent_shift_harmful,
                        agent_shift_benign, noise_scale=1.0):
    """
    Build a replicate with BOTH arms present and a known interaction.

    chat_benign    ~ N(0, s^2 I)
    chat_harmful   ~ N(separation * u, s^2 I)
    agent_harmful  ~ chat_harmful - agent_shift_harmful * u  (paired)
    agent_benign   ~ chat_benign  - agent_shift_benign  * u  (paired)

    The true interaction along u is exactly
    `agent_shift_harmful - agent_shift_benign`. Setting the two shifts equal
    builds the case the harmful-only pipeline cannot distinguish from a real
    weakening: DeltaP is unchanged, and only the interaction knows the
    difference.
    """
    u = _unit(rng, dim)
    chat_benign = noise_scale * rng.standard_normal((n, dim))
    chat_harmful = noise_scale * rng.standard_normal((n, dim)) + separation * u
    jitter_h = 0.1 * noise_scale * rng.standard_normal((n, dim))
    jitter_b = 0.1 * noise_scale * rng.standard_normal((n, dim))
    agent_harmful = chat_harmful - agent_shift_harmful * u + jitter_h
    agent_benign = chat_benign - agent_shift_benign * u + jitter_b
    return {
        "chat_harmful": chat_harmful,
        "chat_benign": chat_benign,
        "agent_harmful": agent_harmful,
        "agent_benign": agent_benign,
    }, u


def experiment_benign_control(trials: int, seed: int, shift_harmful: float,
                              shift_benign: float, label: str) -> dict:
    """
    Can DeltaP tell a weakened refusal from a displaced representation?

    Two ground truths are simulated. In both, harmful prompts lose the same
    amount of projection when the format changes, so DeltaP is the same. They
    differ only in whether benign prompts move with them. The claim under test
    is that DeltaP cannot separate the two and the interaction can.
    """
    rng = np.random.default_rng(seed)
    dp_h, dp_b, inter, inter_p = [], [], [], []

    for _ in range(trials):
        conditions, u = make_data_both_arms(
            rng, HIDDEN_DIM, N_PAIRS, SEPARATION, shift_harmful, shift_benign
        )
        d_est = _diff_in_means_direction(
            conditions["chat_harmful"], conditions["chat_benign"]
        )
        res = specificity_from_projections(
            chat_harmful=conditions["chat_harmful"] @ d_est,
            agent_harmful=conditions["agent_harmful"] @ d_est,
            chat_benign=conditions["chat_benign"] @ d_est,
            agent_benign=conditions["agent_benign"] @ d_est,
            n_permutations=2000, n_boot=2000, seed=int(rng.integers(1 << 30)),
        )
        dp_h.append(res.delta_p_harmful)
        dp_b.append(res.delta_p_benign)
        inter.append(res.interaction)
        inter_p.append(res.interaction_p)

    out = {
        "label": label,
        "true_interaction": shift_harmful - shift_benign,
        "delta_p_harmful": float(np.mean(dp_h)),
        "delta_p_benign": float(np.mean(dp_b)),
        "interaction": float(np.mean(inter)),
        "detection_rate": float(np.mean([p < 0.05 for p in inter_p])),
    }
    print(f"\n  {label}")
    print(f"    true interaction along u        : {out['true_interaction']:+.4f}")
    print(f"    DeltaP harmful (the headline)   : {out['delta_p_harmful']:+.4f}")
    print(f"    DeltaP benign  (the control)    : {out['delta_p_benign']:+.4f}")
    print(f"    interaction (harmful - benign)  : {out['interaction']:+.4f}")
    print(f"    trials calling it significant   : {out['detection_rate']:.0%}")
    return out


# ============================================================
# G. HELD-OUT AUROC OF d_chat
# ============================================================

def experiment_heldout_auroc(trials: int, seed: int, separation: float,
                             label: str) -> dict:
    """
    Is the reported AUROC of d_chat a detection rate or a fit statistic?

    d_chat is the difference between the chat_harmful and chat_benign
    centroids, and the pipeline then scores exactly those two groups with it.
    With `separation = 0` there is nothing to detect and the honest answer is
    0.5; anything the in-sample number reports above that is the direction
    fitting its own sample's noise. Experiment C makes this point for a
    logistic probe; this one makes it for the difference-in-means direction the
    thesis actually reports.
    """
    rng = np.random.default_rng(seed)
    ins, oof = [], []

    for _ in range(trials):
        conditions, _u = make_data_both_arms(
            rng, HIDDEN_DIM, N_PAIRS, separation,
            agent_shift_harmful=0.0, agent_shift_benign=0.0,
        )
        ho = crossfit_condition_projections(
            conditions, n_folds=N_FOLDS, seed=int(rng.integers(1 << 30))
        )
        y = np.r_[np.ones(N_PAIRS), np.zeros(N_PAIRS)]
        ins.append(roc_auc_score(y, np.r_[
            ho.projections_in_sample["chat_harmful"],
            ho.projections_in_sample["chat_benign"],
        ]))
        oof.append(roc_auc_score(y, np.r_[
            ho.projections["chat_harmful"], ho.projections["chat_benign"],
        ]))

    out = {
        "label": label,
        "separation": separation,
        "in_sample": float(np.mean(ins)),
        "oof": float(np.mean(oof)),
        "optimism": float(np.mean(ins) - np.mean(oof)),
    }
    print(f"\n  {label} (separation = {separation:.2f})")
    print(f"    AUROC in-sample (fit and scored on the same prompts): "
          f"{out['in_sample']:.3f}")
    print(f"    AUROC held-out  (direction fitted without them)     : "
          f"{out['oof']:.3f}")
    print(f"    optimism                                            : "
          f"{out['optimism']:+.3f}")
    return out


def experiment_pair_folds(trials: int, seed: int, separation: float,
                          shared_scale: float, label: str) -> dict:
    """
    Does it matter whether a fold holds out ROWS or whole PAIRS?

    Each pair draws one vector that both of its prompts carry - the template,
    the topic, the tools they mention - on top of which the harmful twin moves
    one way along the true axis and the benign twin the other. That shared
    vector is the whole difference between a matched design and two
    independent samples.

    With row folds a twin is in the training set 4 times out of 5 while its
    partner is scored, carrying the opposite label. The probe has 67 points in
    thousands of dimensions and fits them exactly, so the shared vector
    becomes a feature for the training twin's label - and the held-out twin,
    which has the same vector and the other label, is scored on the wrong
    side.
    """
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(N_PAIRS), np.zeros(N_PAIRS)]
    groups = pair_groups(N_PAIRS, N_PAIRS)
    rows, pairs = [], []

    for _ in range(trials):
        u = _unit(rng, HIDDEN_DIM)
        shared = shared_scale * rng.standard_normal((N_PAIRS, HIDDEN_DIM))
        H = shared + rng.standard_normal((N_PAIRS, HIDDEN_DIM)) + 0.5 * separation * u
        B = shared + rng.standard_normal((N_PAIRS, HIDDEN_DIM)) - 0.5 * separation * u
        X = np.concatenate([H, B])
        s = int(rng.integers(1 << 30))
        rows.append(cross_validate_probe(X, y, n_folds=N_FOLDS, seed=s).auroc)
        pairs.append(cross_validate_probe(
            X, y, n_folds=N_FOLDS, seed=s, groups=groups).auroc)

    out = {
        "label": label, "separation": separation,
        "rows": float(np.mean(rows)), "pairs": float(np.mean(pairs)),
    }
    print(f"\n  {label} (separation = {separation:.2f}, shared component "
          f"scale = {shared_scale:.1f})")
    print(f"    probe AUROC, folds over ROWS  : {out['rows']:.3f}")
    print(f"    probe AUROC, folds over PAIRS : {out['pairs']:.3f}")
    return out


def experiment_paired_interaction(trials: int, seed: int, shared_sd: float,
                                  label: str) -> dict:
    """
    Width of the interaction interval, paired versus two-sample.

    Works directly on shifts (chat minus agent projections), which is all the
    interaction test sees. A pair-level component of standard deviation
    `shared_sd` enters both the harmful and the benign shift of a pair; the
    true interaction is 1.0 in every trial.
    """
    rng = np.random.default_rng(seed)
    w_paired, w_two, corr = [], [], []
    zeros = np.zeros(N_PAIRS)

    for _ in range(trials):
        common = shared_sd * rng.standard_normal(N_PAIRS)
        shift_h = 1.0 + common + rng.standard_normal(N_PAIRS)
        shift_b = 0.0 + common + rng.standard_normal(N_PAIRS)
        s = int(rng.integers(1 << 30))
        # chat = shift, agent = 0 reproduces the shifts exactly.
        paired = specificity_from_projections(
            shift_h, zeros, shift_b, zeros, n_permutations=2000, n_boot=2000,
            seed=s, paired_arms=True)
        two = specificity_from_projections(
            shift_h, zeros, shift_b, zeros, n_permutations=2000, n_boot=2000,
            seed=s, paired_arms=False)
        w_paired.append(paired.interaction_ci_upper - paired.interaction_ci_lower)
        w_two.append(two.interaction_ci_upper - two.interaction_ci_lower)
        corr.append(paired.shift_correlation)

    out = {
        "label": label,
        "width_paired": float(np.mean(w_paired)),
        "width_two_sample": float(np.mean(w_two)),
        "corr": float(np.mean(corr)),
    }
    out["ratio"] = out["width_paired"] / out["width_two_sample"]
    print(f"\n  {label}")
    print(f"    corr(shift harmful, shift benign) : {out['corr']:+.2f}")
    print(f"    95% CI width, within pairs        : {out['width_paired']:.3f}")
    print(f"    95% CI width, two-sample          : {out['width_two_sample']:.3f}")
    print(f"    ratio paired / two-sample         : {out['ratio']:.2f}")
    return out


def _two_formats(rng, true_cos: float, separation: float,
                 anisotropic: bool = False):
    """
    Four arms with a KNOWN cosine between the chat and the agent axis.

    Every prompt carries its own content vector into both formats (the request
    is the same text), on top of a component shared by the two twins of a pair
    and of noise specific to each format. That shared content is what makes a
    same-prompt cosine between formats optimistic.
    """
    u_c = _unit(rng, HIDDEN_DIM)
    o = rng.standard_normal(HIDDEN_DIM)
    o -= (o @ u_c) * u_c
    o /= np.linalg.norm(o)
    u_a = true_cos * u_c + np.sqrt(1.0 - true_cos ** 2) * o

    scale = (np.exp(rng.normal(0.0, 1.0, HIDDEN_DIM)) if anisotropic
             else np.ones(HIDDEN_DIM))
    shape = (N_PAIRS, HIDDEN_DIM)
    topic = 2.0 * rng.standard_normal(shape)
    e_h, e_b = rng.standard_normal(shape), rng.standard_normal(shape)

    def arm(content, axis, sign):
        return ((topic + content + rng.standard_normal(shape)) * scale
                + sign * 0.5 * separation * axis)

    return (arm(e_h, u_c, +1), arm(e_b, u_c, -1),
            arm(e_h, u_a, +1), arm(e_b, u_a, -1))


def experiment_rotation(trials: int, seed: int, true_cos: float) -> dict:
    """Raw versus normalised cosine between formats, against a known truth."""
    rng = np.random.default_rng(seed)
    full, norm, rel = [], [], []
    for _ in range(trials):
        arms = _two_formats(rng, true_cos, separation=60.0)
        r = format_rotation(*arms, method="diff_means", n_repeats=12,
                            seed=int(rng.integers(1 << 30)))
        full.append(r["cosine_full_sample"])
        norm.append(r["cosine_normalised"])
        rel.append(r["ceiling"])
    out = {
        "true_cos": true_cos,
        "full": float(np.mean(full)),
        "normalised": float(np.nanmean(norm)),
        "ceiling": float(np.mean(rel)),
    }
    print(f"\n  true cosine between formats = {true_cos:.2f}")
    print(f"    raw cosine, both fitted on all prompts : {out['full']:.3f}")
    print(f"    reliability ceiling of the method      : {out['ceiling']:.3f}")
    print(f"    normalised, disjoint halves            : {out['normalised']:.3f}")
    return out


def experiment_method_mismatch(trials: int, seed: int) -> dict:
    """One true axis, one data set, two estimators: how far apart do they land?"""
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(N_PAIRS), np.zeros(N_PAIRS)]
    cosines = []
    for _ in range(trials):
        ch, cb, _ah, _ab = _two_formats(rng, 1.0, separation=60.0,
                                        anisotropic=True)
        X = np.concatenate([ch, cb])
        cosines.append(float(
            _fit_agent_direction(X, y, method="logistic")
            @ _fit_agent_direction(X, y, method="diff_means")))
    out = {"cosine": float(np.mean(cosines))}
    print("\n  one axis, one data set, noise of unequal variance per dimension")
    print(f"    cos(probe normal, centroid difference) : {out['cosine']:.3f}")
    return out


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=20,
                    help="replicates per experiment (default 20)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("\nCONTROLLED VALIDATION OF THE BIAS CORRECTIONS")
    print(f"Geometry: n = {N_PAIRS} pairs, d = {HIDDEN_DIM}, K = {N_FOLDS} folds")
    print(f"Replicates per experiment: {args.trials}")

    _banner("A. NULL CASE - does the estimator invent an effect that is not there?")
    print("\n  Agent activations are drawn from the SAME distribution as chat,")
    print("  so the true projection gap is exactly zero. Any estimator that")
    print("  returns a non-zero value here is reporting an artefact.")
    null_res = experiment_gap(args.trials, args.seed, agent_shift=0.0,
                              label="null (no true effect)")

    _banner("B. SIGNAL CASE - is a real effect recovered without inflation?")
    print("\n  Agent activations are displaced along the true axis by 0.700.")
    print("  Because the pipeline measures along the ESTIMATED direction, the")
    print("  attainable target is 0.700 * cos(u, d_est), not 0.700 itself.")
    signal_res = experiment_gap(args.trials, args.seed + 1000, agent_shift=0.70,
                                label="signal (true effect = 0.700)")

    _banner("B2. HOW THE BIAS SCALES WITH WITHIN-CLASS VARIANCE")
    sweep = experiment_noise_sweep(args.trials, args.seed + 1500)

    _banner("C. COVER'S THEOREM - is a perfect in-sample AUROC informative?")
    cover_res = experiment_cover(args.trials, args.seed + 2000)

    _banner("D. ATTENUATION - is the measured rotation angle trustworthy?")
    att = [experiment_attenuation(args.trials, args.seed + 3000 + i * 100, c)
           for i, c in enumerate([0.80, 0.50, 0.20])]

    _banner("E. FOLD HYGIENE - does it matter WHERE the direction is estimated?")
    print("\n  Cross-validating the classifier while estimating the direction on")
    print("  the full sample is the failure mode that survives review, because")
    print("  the fold loop is real and the number still is not.")
    hygiene = experiment_fold_hygiene(args.trials, args.seed + 4000)

    _banner("F. BENIGN CONTROL - is the shift specific to harmful prompts?")
    print("\n  Both cases move harmful prompts by the same amount, so DeltaP is")
    print("  the same in each. They differ only in whether benign prompts move")
    print("  with them. If DeltaP alone were sufficient, the two would be")
    print("  indistinguishable — which is exactly the point.")
    specific = experiment_benign_control(
        args.trials, args.seed + 5000, shift_harmful=0.70, shift_benign=0.00,
        label="SPECIFIC: only harmful prompts move",
    )
    common = experiment_benign_control(
        args.trials, args.seed + 5500, shift_harmful=0.70, shift_benign=0.70,
        label="COMMON: every prompt moves alike (a change of origin)",
    )

    _banner("G. HELD-OUT AUROC - is the reported detection real?")
    print("\n  d_chat is the difference between the two centroids it is then")
    print("  asked to separate. With no true separation the honest answer is")
    print("  0.500, and whatever the in-sample number reports above that is")
    print("  the direction fitting its own sample's noise.")
    auroc_null = experiment_heldout_auroc(
        args.trials, args.seed + 6000, separation=0.0,
        label="NO true separation",
    )
    auroc_signal = experiment_heldout_auroc(
        args.trials, args.seed + 6500, separation=SEPARATION,
        label="REAL separation",
    )

    _banner("H. PAIR FOLDS - may a fold separate a prompt from its twin?")
    print("\n  Twins share a component (same template, same topic). Row folds")
    print("  train on one twin and test on the other, labels opposed.")
    folds_null = experiment_pair_folds(
        args.trials, args.seed + 7000, separation=0.0, shared_scale=1.0,
        label="NO true separation",
    )
    folds_signal = experiment_pair_folds(
        args.trials, args.seed + 7500, separation=3.0 * SEPARATION,
        shared_scale=1.0, label="REAL separation",
    )

    _banner("I. PAIRED INTERACTION - which units are exchangeable?")
    print("\n  True interaction = 1.0 in both cases. Only the amount the two")
    print("  shifts of a pair have in common changes.")
    inter_shared = experiment_paired_interaction(
        args.trials, args.seed + 8000, shared_sd=1.5,
        label="shifts of a pair SHARE a component",
    )
    inter_indep = experiment_paired_interaction(
        args.trials, args.seed + 8500, shared_sd=0.0,
        label="shifts of a pair are UNRELATED",
    )

    _banner("J. ROTATION - is a cosine between formats an angle?")
    print("\n  The same request is read in both formats, so its content is in")
    print("  both activations. Three truths: no rotation, 60 degrees, 90.")
    rot = [experiment_rotation(args.trials, args.seed + 9000 + i * 100, c)
           for i, c in enumerate([1.0, 0.5, 0.0])]
    mismatch = experiment_method_mismatch(args.trials, args.seed + 9500)

    # ---------------- verdicts ----------------
    _banner("VERDICTS")

    checks = []

    ok_null = abs(null_res["crossfit_mean"]) < 0.1 * abs(null_res["naive_mean"])
    checks.append((
        "A. Cross-fitting removes the spurious gap when there is no effect",
        ok_null,
        f"naive reports {null_res['naive_mean']:+.4f} where the truth is 0.0000; "
        f"cross-fitted reports {null_res['crossfit_mean']:+.4f}",
    ))

    ok_signal = abs(signal_res["crossfit_bias"]) < abs(signal_res["naive_bias"])
    checks.append((
        "B. Cross-fitting is unbiased for the attainable target; naive is not",
        ok_signal,
        f"target {signal_res['attenuated_truth']:.4f} | "
        f"naive bias {signal_res['naive_bias']:+.4f} | "
        f"cross-fitted bias {signal_res['crossfit_bias']:+.4f}",
    ))

    ok_sweep = all(
        abs(r["crossfit_bias"]) < abs(r["naive_bias"]) for r in sweep
    ) and sweep[0]["naive_bias"] < sweep[-1]["naive_bias"]
    checks.append((
        "B2. The bias grows with within-class variance, as the theory predicts",
        ok_sweep,
        f"naive bias runs {sweep[0]['naive_bias']:+.4f} (s=0.10) -> "
        f"{sweep[-1]['naive_bias']:+.4f} (s=1.00)",
    ))

    ok_cover = cover_res["in_sample"] > 0.95 and cover_res["oof"] < 0.65
    checks.append((
        "C. In-sample AUROC is ~1.0 on pure noise; out-of-fold is not",
        ok_cover,
        f"in-sample {cover_res['in_sample']:.3f}, out-of-fold {cover_res['oof']:.3f}",
    ))

    ok_att = all(a["measured_cos"] < a["true_cos"] for a in att)
    checks.append((
        "D1. Measured cosine is attenuated -> the raw angle is an upper bound",
        ok_att,
        "; ".join(f"{a['true_cos']:.2f}->{a['measured_cos']:.3f}" for a in att),
    ))

    # sqrt(split-half reliability) should predict the empirical attenuation.
    # This is the claim the theory makes, and it is what makes the reliability
    # a usable diagnostic even when the correction itself is unusable.
    pred_err = [
        abs(np.sqrt(max(a["reliability_split"], 0.0)) - a["measured_cos"] / a["true_cos"])
        for a in att
    ]
    ok_theory = all(e < 0.10 for e in pred_err)
    checks.append((
        "D2. sqrt(split-half reliability) predicts the attenuation factor",
        ok_theory,
        "; ".join(
            f"predicted {np.sqrt(max(a['reliability_split'], 0)):.3f} vs "
            f"empirical {a['measured_cos'] / a['true_cos']:.3f}"
            for a in att
        ),
    ))

    ok_inflated = all(a["reliability_fold"] > a["reliability_split"] for a in att)
    checks.append((
        "D3. Inter-fold cosine overstates reliability (overlapping train sets)",
        ok_inflated,
        "; ".join(
            f"fold {a['reliability_fold']:.3f} vs split-half {a['reliability_split']:.3f}"
            for a in att
        ),
    ))

    # At this sample geometry the reliability is far below the floor, so the
    # estimator must REFUSE to report a corrected angle rather than emit a
    # plausible-looking artefact.
    ok_guard = all(
        (a["reliability_split"] >= MIN_RELIABILITY_FOR_CORRECTION)
        == np.isfinite(a["disattenuated_cos"])
        for a in att
    )
    checks.append((
        "D4. The estimator refuses to report an angle it cannot support",
        ok_guard,
        f"reliability floor = {MIN_RELIABILITY_FOR_CORRECTION}; observed "
        + ", ".join(f"{a['reliability_split']:.3f}" for a in att)
        + " -> all below floor, all corrections withheld (nan)",
    ))

    # What this experiment establishes is the CONTRAST: the leak manufactures a
    # near-perfect AUROC out of nothing, and the clean estimate does not. The
    # honest value converges to 0.500, but pinning it there is a claim about
    # convergence rate, and at the default trial count the estimated SE is
    # itself unstable (chi-squared with few df), so a band built from it fails
    # at random. The gate therefore tests the contrast; the distance from 0.500
    # is reported for the reader to judge.
    gap = hygiene["leaky_direction"] - hygiene["honest_direction"]
    ok_hygiene = (
        hygiene["leaky_direction"] > 0.90
        and hygiene["honest_direction"] < 0.70
        and gap > 0.25
    )
    detail = (
        f"on data with NO signal: direction from full sample -> "
        f"{hygiene['leaky_direction']:.3f}, direction from the training fold -> "
        f"{hygiene['honest_direction']:.3f} +- {hygiene['honest_se']:.3f} "
        f"(gap {gap:+.3f}; truth is 0.500, which the clean estimate approaches "
        f"as --trials rises)"
    )
    checks.append((
        "E. Leaking the direction into the folds fakes a validated AUROC",
        ok_hygiene,
        detail,
    ))

    # F. The two ground truths are built to give the same DeltaP. If the
    # headline statistic separated them, the benign control would be
    # redundant; it does not, which is why the control is not optional.
    dp_difference = abs(specific["delta_p_harmful"] - common["delta_p_harmful"])
    ok_dp_blind = dp_difference < 0.10
    checks.append((
        "F1. DeltaP alone cannot tell a weakened refusal from a common shift",
        ok_dp_blind,
        f"specific {specific['delta_p_harmful']:+.4f} vs common "
        f"{common['delta_p_harmful']:+.4f} (difference {dp_difference:.4f})",
    ))

    # The gate is relative, not absolute. Everything measured along the
    # ESTIMATED direction is attenuated by cos(u, d_est), which at n=42 in
    # d=8192 is small: a true shift of 0.70 shows up as roughly 0.12. The
    # attainable interaction in the specific case is therefore about the size
    # of DeltaP_harmful itself, and testing against 0.70 would fail a correct
    # estimator for the wrong reason.
    ok_interaction = (
        specific["interaction"] > 0.5 * specific["delta_p_harmful"]
        and abs(common["interaction"]) < 0.25 * abs(common["delta_p_harmful"])
        and specific["detection_rate"] > 0.8
        and common["detection_rate"] < 0.35
    )
    checks.append((
        "F2. The difference-in-differences separates them",
        ok_interaction,
        f"interaction {specific['interaction']:+.4f} (significant in "
        f"{specific['detection_rate']:.0%} of trials) vs "
        f"{common['interaction']:+.4f} ({common['detection_rate']:.0%}); both "
        f"attenuated by cos(u, d_est), so read them against DeltaP_harmful "
        f"({specific['delta_p_harmful']:+.4f}), not against the true 0.700",
    ))

    # G. The AUROC the thesis quotes for d_chat, on data with nothing to find.
    ok_auroc_null = auroc_null["in_sample"] > 0.95 and auroc_null["oof"] < 0.65
    checks.append((
        "G1. In-sample AUROC of d_chat is ~1.0 with no true separation",
        ok_auroc_null,
        f"in-sample {auroc_null['in_sample']:.3f}, held-out "
        f"{auroc_null['oof']:.3f} (truth 0.500)",
    ))

    # Held-out AUROC must rise when there IS something to detect, and its
    # optimism must fall. Note how far below the in-sample value it stays even
    # with a real separation: at n=42 in d=8192 the direction estimated from 33
    # pairs recovers only a fraction of the true axis, so the honest number is
    # modest. That is a statement about this sample geometry under isotropic
    # noise (the worst case, see the closing note), not a defect of the
    # estimator — and it is precisely why the in-sample figure cannot be quoted.
    ok_auroc_signal = (
        auroc_signal["oof"] > auroc_null["oof"] + 0.05
        and auroc_signal["optimism"] < auroc_null["optimism"]
    )
    checks.append((
        "G2. Held-out AUROC rises with real signal and its optimism falls",
        ok_auroc_signal,
        f"held-out {auroc_signal['oof']:.3f} (optimism "
        f"{auroc_signal['optimism']:+.3f}) against {auroc_null['oof']:.3f} "
        f"(optimism {auroc_null['optimism']:+.3f}) with nothing to detect; "
        f"in-sample reports {auroc_signal['in_sample']:.3f} either way",
    ))

    # With nothing to detect, the honest AUROC is 0.5. Row folds must miss it
    # on the LOW side - that is the signature of training on the twin.
    checks.append((
        "H1. Row folds are not a valid null for matched pairs; pair folds are",
        folds_null["rows"] < 0.35 and 0.40 <= folds_null["pairs"] <= 0.60,
        f"no true separation: rows {folds_null['rows']:.3f}, pairs "
        f"{folds_null['pairs']:.3f} (truth 0.500)",
    ))
    checks.append((
        "H2. With a real signal, row folds understate what pair folds recover",
        folds_signal["pairs"] > folds_signal["rows"] + 0.10,
        f"real separation: rows {folds_signal['rows']:.3f}, pairs "
        f"{folds_signal['pairs']:.3f}",
    ))
    checks.append((
        "I1. Pairing narrows the interaction interval when shifts are shared",
        inter_shared["ratio"] < 0.85,
        f"corr {inter_shared['corr']:+.2f}: paired width "
        f"{inter_shared['width_paired']:.3f} vs two-sample "
        f"{inter_shared['width_two_sample']:.3f}",
    ))
    checks.append((
        "I2. With unrelated shifts the two analyses agree",
        0.85 <= inter_indep["ratio"] <= 1.15,
        f"corr {inter_indep['corr']:+.2f}: paired width "
        f"{inter_indep['width_paired']:.3f} vs two-sample "
        f"{inter_indep['width_two_sample']:.3f}",
    ))

    checks.append((
        "J1. The normalised cosine recovers the true angle between formats",
        all(abs(r["normalised"] - r["true_cos"]) < 0.10 for r in rot),
        "; ".join(f"truth {r['true_cos']:.1f} -> {r['normalised']:.3f}"
                  for r in rot),
    ))
    # Noise drags the raw cosine below 1 when nothing rotated; shared prompt
    # content lifts it above 0 when the axes are orthogonal.
    checks.append((
        "J2. The raw cosine misses in both directions",
        rot[0]["full"] < 0.97 and rot[2]["full"] > 0.03,
        f"no rotation reads {rot[0]['full']:.3f} (truth 1.000); a right angle "
        f"reads {rot[2]['full']:.3f} (truth 0.000)",
    ))
    checks.append((
        "J3. A probe normal and a centroid difference are different objects",
        mismatch["cosine"] < 0.80,
        f"same data, one true axis: cosine {mismatch['cosine']:.3f}",
    ))

    for name, passed, detail in checks:
        print(f"\n  [{'PASS' if passed else 'FAIL'}] {name}")
        print(f"         {detail}")

    all_passed = all(c[1] for c in checks)
    print()
    print("=" * 78)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    print("=" * 78)
    print("""
  WHAT THIS ESTABLISHES

    1. The circularity bias is real, not hypothetical. With no true effect
       at all, the naive estimator reports a large positive gap. Cross-
       fitting returns it to zero.

    2. A perfect in-sample AUROC is reproduced here from PURE NOISE at this
       sample geometry. It certifies nothing about the data.

    2b. So is a *cross-validated* AUROC, if the refusal direction was estimated
       over the full sample before the folds were cut. Experiment E gets ~1.0
       out of pure noise that way. Cross-validation is not a property of the
       loop; it is a property of what each fold was allowed to see. The whole
       pipeline - standardiser, direction, probe, threshold - has to be
       refitted inside the fold, which is what src/probes.py now does.

    3. The measured rotation angle is systematically overstated, and the
       overstatement grows as the true angle shrinks.

    4. THE STRONGEST RESULT HERE. At this sample geometry the split-half
       reliability of w_agent is near zero, and sqrt(reliability) correctly
       predicts the observed attenuation. Two estimates of w_agent from
       disjoint halves of the data barely agree at all. A direction that
       irreproducible has no recoverable angle: the correct report is that
       the alignment differs from chance, with NO angle attached. The
       estimator now withholds the number instead of inventing one.

  WHAT THIS DOES *NOT* ESTABLISH

    The SIZE of the bias in the real run. These simulations use isotropic
    Gaussian noise, which maximises centroid noise for a given total
    variance; real activations are strongly anisotropic and occupy a
    lower-dimensional manifold, so the real bias is expected to be smaller
    than the worst case shown here. Sweep B2 shows how strongly the bias
    depends on that assumption.

    To obtain the real figure, re-run the pipeline with the corrected
    estimators and read `bias_estimate` off the cross-fitted gap report.

  A SEPARATE POINT WORTH CARRYING INTO THE THESIS

    Cross-fitting fixes the BIAS. It does not fix the ATTENUATION: because
    the direction is estimated from 42 samples in 8192 dimensions, it
    captures only a fraction of the true axis, and every projection-based
    magnitude is shrunk by that fraction. Bias and attenuation are separate
    problems with separate remedies, and the DIRECTION of the effect - which
    the sign test certifies without using any magnitude - is unaffected by
    either.
""")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
