"""
Specificity and scale of the format effect.

DeltaP alone cannot answer the research question. It measures how far the
harmful centroid moves along d_chat when the format changes, and three very
different mechanisms produce the same number:

  1. The refusal representation genuinely weakens for harmful requests.
     This is the thesis's claim.

  2. The agentic context displaces EVERY representation along d_chat by a
     common offset. The refusal mechanism is untouched; only the origin moved.
     A rigid translation leaves the harmful/benign contrast intact.

  3. The agentic context contracts the SCALE of the projection. Activation
     norms are arbitrary units, and if the whole distribution shrinks, both
     the between-class distance and the within-class spread shrink together.
     Nothing about detection has changed, but DeltaP is large.

The methodology chapter already argues that projections are not comparable
across layers without normalisation, because the norm of the residual stream
grows with depth. That same argument applies across FORMATS, and for the same
reason: the agentic prompt is longer and its activations live at a different
scale. Reporting a raw DeltaP across formats repeats, between conditions, the
error the chapter forbids between layers.

This module supplies the two contrasts that separate the three mechanisms:

  BENIGN CONTROL (difference-in-differences)
      Compare the paired shift of harmful prompts against the paired shift of
      benign ones. Mechanism 2 predicts the two shifts are equal and the
      interaction vanishes. Only a harmful-specific interaction supports the
      thesis. This is the first objection any examiner raises, and the corpus
      already contains the benign arm needed to answer it.

  SCALE-INVARIANT EFFECT
      Re-express the format effect in units that survive a change of scale:
      the standardised class separation, and the AUROC. If the raw separation
      falls by 40% while the standardised separation falls by 10%, then three
      quarters of the apparent weakening was a change of units.

AUROC is reported alongside because it is scale-invariant by construction, but
it saturates: once separation exceeds roughly three pooled standard
deviations, AUROC sits at the ceiling and is blind to further change. It
certifies that detection survives; it cannot measure how much margin was lost.
The standardised separation covers exactly that gap.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import roc_auc_score

from .robust_estimation import (
    paired_bootstrap_ci,
    paired_cohens_d,
    paired_permutation_test,
    paired_sign_test,
    paired_ttest_directional,
    pooled_cohens_d,
    run_paired_suite,
)


# ============================================================
# RESULT CONTAINERS
# ============================================================

@dataclass
class SpecificityResult:
    """Difference-in-differences: is the format shift specific to harmful prompts?"""
    layer_idx: int
    n_harmful: int
    n_benign: int
    delta_p_harmful: float          # mean(chat_h - agent_h), the reported effect
    delta_p_benign: float           # mean(chat_b - agent_b), the control
    interaction: float              # harmful shift minus benign shift
    # Is the benign arm moving at all?
    benign_shift_p: float
    # Is the interaction real?
    interaction_p: float
    interaction_cohens_d: float
    interaction_ci_lower: float
    interaction_ci_upper: float
    # Share of the headline effect that is NOT specific to harmfulness
    common_shift_fraction: float
    # 'in-sample' when the projections come from a direction fitted on these
    # same prompts, 'held-out' when each prompt was scored by a direction
    # fitted without it. The two are not interchangeable.
    evaluation: str = "in-sample"
    # True when harmful prompt i and benign prompt i are the two halves of one
    # pair and the interaction was tested WITHIN pairs (see
    # specificity_from_projections). interaction_p, the interval and the
    # effect size above then come from that paired analysis.
    arms_paired: bool = False
    effect_size_kind: str = "d (pooled, two-sample)"
    n_pairs_interaction_positive: Optional[int] = None
    shift_correlation: Optional[float] = None
    # The two-sample analysis, kept beside the paired one whenever both were
    # run, so the consequence of the pairing is reported rather than assumed.
    interaction_p_two_sample: Optional[float] = None
    interaction_ci_lower_two_sample: Optional[float] = None
    interaction_ci_upper_two_sample: Optional[float] = None

    @property
    def is_specific(self) -> bool:
        """True when the effect cannot be explained by a rigid common shift."""
        return self.interaction_p < 0.05 and self.interaction > 0


@dataclass
class ScaleResult:
    """The format effect expressed in raw, standardised and rank units."""
    layer_idx: int
    # Raw units (scale-dependent — what the pipeline reports today)
    separation_chat: float
    separation_agent: float
    separation_loss_raw: float          # fractional loss, raw units
    # Standardised units (scale-invariant)
    pooled_sd_chat: float
    pooled_sd_agent: float
    cohens_d_chat: float
    cohens_d_agent: float
    separation_loss_standardised: float
    # Rank units (scale-invariant, saturating)
    auroc_chat: float
    auroc_agent: float
    auroc_loss: float
    # How much of the raw loss was a change of units rather than of separability
    scale_contraction: float            # pooled_sd_agent / pooled_sd_chat
    evaluation: str = "in-sample"

    @property
    def raw_loss_is_inflated(self) -> bool:
        """True when the raw loss materially overstates the standardised one."""
        return self.separation_loss_raw > self.separation_loss_standardised + 0.10


@dataclass
class StratumDetection:
    """Detection capability within one explicitness stratum, per format."""
    stratum: str
    n_harmful: int
    n_benign: int
    auroc_chat: float
    auroc_agent: float
    auroc_loss: float
    cohens_d_chat: float
    cohens_d_agent: float
    separation_loss_standardised: float
    evaluation: str = "in-sample"


# ============================================================
# 1. BENIGN CONTROL  (difference-in-differences)
# ============================================================

def format_effect_specificity(
    chat_harmful: np.ndarray,
    agent_harmful: np.ndarray,
    chat_benign: np.ndarray,
    agent_benign: np.ndarray,
    direction: np.ndarray,
    layer_idx: int = -1,
    n_permutations: int = 10000,
    n_boot: int = 10000,
    seed: int = 42,
) -> SpecificityResult:
    """
    Test whether the agentic shift is specific to harmful requests.

    Both arms are paired within themselves (a prompt appears in both formats),
    so each arm's shift is a within-pair difference. The two arms are ALSO
    matched to each other: benign prompt i is the harmful prompt i rewritten
    with a legitimate intent, sharing its structure, vocabulary and tools. The
    interaction is therefore a within-pair contrast of the two shifts, and is
    tested as one (see `specificity_from_projections`).

    Interpretation:
        interaction ~ 0        the format translates everything equally; the
                               refusal mechanism is not selectively weakened
        interaction > 0        harmful prompts lose more projection than benign
                               ones, which is the thesis's claim
        interaction < 0        the direction responds MORE to harm under the
                               agentic format

    Args:
        chat_harmful, agent_harmful: paired row-wise, (n_h, hidden_dim).
        chat_benign, agent_benign:   paired row-wise, (n_b, hidden_dim).
        direction: the refusal direction for this layer, (hidden_dim,).

    Returns:
        SpecificityResult.
    """
    return specificity_from_projections(
        chat_harmful=chat_harmful @ direction,
        agent_harmful=agent_harmful @ direction,
        chat_benign=chat_benign @ direction,
        agent_benign=agent_benign @ direction,
        layer_idx=layer_idx,
        n_permutations=n_permutations,
        n_boot=n_boot,
        seed=seed,
        evaluation="in-sample",
    )


def specificity_from_projections(
    chat_harmful: np.ndarray,
    agent_harmful: np.ndarray,
    chat_benign: np.ndarray,
    agent_benign: np.ndarray,
    layer_idx: int = -1,
    n_permutations: int = 10000,
    n_boot: int = 10000,
    seed: int = 42,
    evaluation: str = "in-sample",
    paired_arms: Optional[bool] = None,
) -> SpecificityResult:
    """
    Difference-in-differences on projections that are already computed.

    WHICH UNITS ARE EXCHANGEABLE

    The corpus is built pair by pair: each harmful instruction has a benign
    counterpart on the same template. Row i of the harmful arm and row i of
    the benign arm are therefore not two independent draws but two halves of
    one unit, and the quantity under test is the within-pair double difference

        D_i = (chat_h,i - agent_h,i) - (chat_b,i - agent_b,i).

    Its null distribution comes from flipping the sign of D_i pair by pair,
    and its interval from resampling pairs. Treating the arms as independent
    groups - shuffling the 84 shifts across arms and bootstrapping each arm
    separately, as this function first did - is the analysis for a design in
    which the benign prompts were unrelated to the harmful ones. It is not
    wrong in expectation, but it is not the design: it ignores whatever the
    two shifts of a pair share, and so mis-states the uncertainty by an amount
    that depends on their correlation. Both analyses are reported, and
    `shift_correlation` says how far apart they can be.

    Separated from `format_effect_specificity` so the same contrast can be run
    on HELD-OUT projections, where each prompt was scored by a direction
    fitted without it. There is no single `direction` there - one per fold -
    so the analysis has to start from the scores, not from the activations.

    Args:
        the four arguments are projection vectors (n,), not activations.
        evaluation: label recorded on the result and printed in the report, so
            an in-sample and a held-out figure can never be confused.
        paired_arms: True tests the interaction within pairs and requires the
            arms to be row-aligned. False forces the two-sample analysis. None
            (default) pairs whenever the arms have the same number of rows,
            which in this pipeline means they are row-aligned by pair id.
    """
    chat_harmful = np.asarray(chat_harmful, dtype=float)
    agent_harmful = np.asarray(agent_harmful, dtype=float)
    chat_benign = np.asarray(chat_benign, dtype=float)
    agent_benign = np.asarray(agent_benign, dtype=float)

    if len(chat_harmful) != len(agent_harmful):
        raise ValueError("harmful arm is not paired row-wise")
    if len(chat_benign) != len(agent_benign):
        raise ValueError("benign arm is not paired row-wise")

    shift_h = chat_harmful - agent_harmful
    shift_b = chat_benign - agent_benign

    # Does the benign arm move at all? (paired test against zero)
    _, p_benign = paired_ttest_directional(
        chat_benign, agent_benign, alternative="two-sided"
    )

    # Interaction: the two shift vectors come from different prompts, so this
    # is a two-sample contrast on within-pair differences. A label-shuffling
    # permutation is the right null here — unlike the within-arm tests, where
    # exchangeability holds within pairs rather than across units.
    rng = np.random.default_rng(seed)
    observed = float(shift_h.mean() - shift_b.mean())
    combined = np.concatenate([shift_h, shift_b])
    n_h = len(shift_h)

    idx = np.argsort(rng.random((n_permutations, len(combined))), axis=1)
    permuted = combined[idx]
    null = permuted[:, :n_h].mean(axis=1) - permuted[:, n_h:].mean(axis=1)
    p_inter = float((np.abs(null) >= abs(observed)).mean())

    d_inter = pooled_cohens_d(shift_h, shift_b)

    # Two-sample bootstrap: each arm resampled on its own.
    bh = rng.integers(0, n_h, size=(n_boot, n_h))
    bb = rng.integers(0, len(shift_b), size=(n_boot, len(shift_b)))
    boot = shift_h[bh].mean(axis=1) - shift_b[bb].mean(axis=1)
    ci_two = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    dp_h, dp_b = float(shift_h.mean()), float(shift_b.mean())
    common = float(dp_b / dp_h) if abs(dp_h) > 1e-12 else float("nan")

    if paired_arms is None:
        paired_arms = len(shift_h) == len(shift_b)
    if paired_arms and len(shift_h) != len(shift_b):
        raise ValueError(
            "paired_arms=True needs row-aligned arms; got "
            f"{len(shift_h)} harmful and {len(shift_b)} benign rows."
        )

    common_fields = dict(
        layer_idx=layer_idx,
        n_harmful=len(shift_h),
        n_benign=len(shift_b),
        delta_p_harmful=dp_h,
        delta_p_benign=dp_b,
        interaction=observed,
        benign_shift_p=float(p_benign),
        common_shift_fraction=common,
        evaluation=evaluation,
    )

    if not paired_arms:
        return SpecificityResult(
            interaction_p=p_inter,
            interaction_cohens_d=d_inter,
            interaction_ci_lower=ci_two[0],
            interaction_ci_upper=ci_two[1],
            **common_fields,
        )

    # Within-pair test of D_i against zero: sign-flip permutation, paired
    # bootstrap, d_z. Two-sided, like the two-sample test it replaces.
    paired = run_paired_suite(
        shift_h, shift_b, n_permutations, n_boot, "two-sided", seed
    )
    if np.std(shift_h) > 0 and np.std(shift_b) > 0:
        corr = float(np.corrcoef(shift_h, shift_b)[0, 1])
    else:
        corr = float("nan")

    return SpecificityResult(
        # A permutation p cannot be smaller than 1/n_permutations; report the
        # floor rather than a literal zero.
        interaction_p=float(max(paired.p_value_permutation, paired.p_value_floor)),
        interaction_cohens_d=paired.cohens_d_paired,
        interaction_ci_lower=paired.ci_lower,
        interaction_ci_upper=paired.ci_upper,
        arms_paired=True,
        effect_size_kind="d_z (paired)",
        n_pairs_interaction_positive=int((shift_h - shift_b > 0).sum()),
        shift_correlation=corr,
        interaction_p_two_sample=p_inter,
        interaction_ci_lower_two_sample=ci_two[0],
        interaction_ci_upper_two_sample=ci_two[1],
        **common_fields,
    )


# ============================================================
# 2. SCALE-INVARIANT RE-EXPRESSION
# ============================================================

def format_effect_scale(
    chat_harmful: np.ndarray,
    agent_harmful: np.ndarray,
    chat_benign: np.ndarray,
    agent_benign: np.ndarray,
    direction: np.ndarray,
    layer_idx: int = -1,
) -> ScaleResult:
    """
    Express the format effect in raw, standardised and rank units.

    The three disagree whenever the agentic format changes the scale of the
    projection, and the disagreement is the finding: it says how much of the
    headline number is a change of units.
    """
    return scale_from_projections(
        chat_harmful=chat_harmful @ direction,
        agent_harmful=agent_harmful @ direction,
        chat_benign=chat_benign @ direction,
        agent_benign=agent_benign @ direction,
        layer_idx=layer_idx,
        evaluation="in-sample",
    )


def scale_from_projections(
    chat_harmful: np.ndarray,
    agent_harmful: np.ndarray,
    chat_benign: np.ndarray,
    agent_benign: np.ndarray,
    layer_idx: int = -1,
    evaluation: str = "in-sample",
) -> ScaleResult:
    """
    Raw / standardised / rank re-expression from already-computed projections.

    Same contrast as `format_effect_scale`, entered from the scores so it can
    also be run on held-out projections. The AUROCs it reports are in-sample
    whenever the caller passes in-sample projections - which is exactly the
    circularity the held-out path exists to remove, so pass the held-out
    scores when the number is going into the thesis.
    """
    ph_c = np.asarray(chat_harmful, dtype=float)
    ph_a = np.asarray(agent_harmful, dtype=float)
    pb_c = np.asarray(chat_benign, dtype=float)
    pb_a = np.asarray(agent_benign, dtype=float)

    sep_c = float(ph_c.mean() - pb_c.mean())
    sep_a = float(ph_a.mean() - pb_a.mean())

    sd_c = float(np.sqrt((ph_c.std(ddof=1) ** 2 + pb_c.std(ddof=1) ** 2) / 2))
    sd_a = float(np.sqrt((ph_a.std(ddof=1) ** 2 + pb_a.std(ddof=1) ** 2) / 2))

    d_c = sep_c / sd_c if sd_c > 1e-12 else 0.0
    d_a = sep_a / sd_a if sd_a > 1e-12 else 0.0

    y_c = np.r_[np.ones(len(ph_c)), np.zeros(len(pb_c))]
    y_a = np.r_[np.ones(len(ph_a)), np.zeros(len(pb_a))]
    auroc_c = float(roc_auc_score(y_c, np.r_[ph_c, pb_c]))
    auroc_a = float(roc_auc_score(y_a, np.r_[ph_a, pb_a]))

    return ScaleResult(
        layer_idx=layer_idx,
        separation_chat=sep_c,
        separation_agent=sep_a,
        separation_loss_raw=float(1.0 - sep_a / sep_c) if abs(sep_c) > 1e-12 else float("nan"),
        pooled_sd_chat=sd_c,
        pooled_sd_agent=sd_a,
        cohens_d_chat=float(d_c),
        cohens_d_agent=float(d_a),
        separation_loss_standardised=float(1.0 - d_a / d_c) if abs(d_c) > 1e-12 else float("nan"),
        auroc_chat=auroc_c,
        auroc_agent=auroc_a,
        auroc_loss=float(auroc_c - auroc_a),
        scale_contraction=float(sd_a / sd_c) if sd_c > 1e-12 else float("nan"),
        evaluation=evaluation,
    )


# ============================================================
# 3. DETECTION CAPABILITY BY EXPLICITNESS STRATUM
# ============================================================

def detection_by_stratum(
    strata_harmful: List[str],
    strata_benign: List[str],
    chat_harmful: np.ndarray,
    agent_harmful: np.ndarray,
    chat_benign: np.ndarray,
    agent_benign: np.ndarray,
    direction: np.ndarray,
    min_per_class: int = 4,
) -> List[StratumDetection]:
    """
    AUROC and standardised separation per explicitness stratum, per format.

    The fourth specific objective asks whether DETECTION CAPABILITY holds as
    harmful intent becomes less explicit. The methodology chapter defines
    detection capability as AUROC, and argues explicitly that DeltaP and AUROC
    answer different questions. The stratified analysis currently reports
    DeltaP per stratum only, which measures intensity, not detection — so the
    objective is not yet answered by the code that claims to answer it.

    AUROC needs both classes, so the benign arm is stratified too. The corpus
    labels benign prompts with the explicitness of the harmful prompt they were
    built from, which is what makes the within-stratum contrast meaningful.

    Args:
        strata_harmful / strata_benign: per-row stratum labels, aligned with
            the corresponding activation arrays.
        min_per_class: strata with fewer rows than this in either class are
            skipped; AUROC on a handful of points is noise.

    Returns:
        One StratumDetection per stratum with enough data, ordered by name.
    """
    return stratum_detection_from_projections(
        strata_harmful, strata_benign,
        chat_harmful=chat_harmful @ direction,
        agent_harmful=agent_harmful @ direction,
        chat_benign=chat_benign @ direction,
        agent_benign=agent_benign @ direction,
        min_per_class=min_per_class,
        evaluation="in-sample",
    )


def stratum_detection_from_projections(
    strata_harmful: List[str],
    strata_benign: List[str],
    chat_harmful: np.ndarray,
    agent_harmful: np.ndarray,
    chat_benign: np.ndarray,
    agent_benign: np.ndarray,
    min_per_class: int = 4,
    evaluation: str = "in-sample",
) -> List[StratumDetection]:
    """
    Per-stratum detection from already-computed projections.

    The stratified AUROC inherits the circularity of whatever produced the
    scores: within a stratum the numbers are smaller and the optimism is
    proportionally larger, because each stratum holds a handful of prompts.
    Pass held-out projections here and the per-stratum figures become
    detection rates too.
    """
    strata_harmful = list(strata_harmful)
    strata_benign = list(strata_benign)

    ph_c = np.asarray(chat_harmful, dtype=float)
    ph_a = np.asarray(agent_harmful, dtype=float)
    pb_c = np.asarray(chat_benign, dtype=float)
    pb_a = np.asarray(agent_benign, dtype=float)

    if len(strata_harmful) != len(ph_c):
        raise ValueError("harmful stratum labels are not aligned with projections")
    if len(strata_benign) != len(pb_c):
        raise ValueError("benign stratum labels are not aligned with projections")

    out: List[StratumDetection] = []
    for stratum in sorted(set(strata_harmful)):
        ih = np.array([i for i, s in enumerate(strata_harmful) if s == stratum])
        ib = np.array([i for i, s in enumerate(strata_benign) if s == stratum])
        if len(ih) < min_per_class or len(ib) < min_per_class:
            continue

        y = np.r_[np.ones(len(ih)), np.zeros(len(ib))]
        auroc_c = float(roc_auc_score(y, np.r_[ph_c[ih], pb_c[ib]]))
        auroc_a = float(roc_auc_score(y, np.r_[ph_a[ih], pb_a[ib]]))

        sd_c = np.sqrt((ph_c[ih].std(ddof=1) ** 2 + pb_c[ib].std(ddof=1) ** 2) / 2)
        sd_a = np.sqrt((ph_a[ih].std(ddof=1) ** 2 + pb_a[ib].std(ddof=1) ** 2) / 2)
        d_c = (ph_c[ih].mean() - pb_c[ib].mean()) / sd_c if sd_c > 1e-12 else 0.0
        d_a = (ph_a[ih].mean() - pb_a[ib].mean()) / sd_a if sd_a > 1e-12 else 0.0

        out.append(StratumDetection(
            stratum=stratum,
            n_harmful=len(ih),
            n_benign=len(ib),
            auroc_chat=auroc_c,
            auroc_agent=auroc_a,
            auroc_loss=auroc_c - auroc_a,
            cohens_d_chat=float(d_c),
            cohens_d_agent=float(d_a),
            separation_loss_standardised=(
                float(1.0 - d_a / d_c) if abs(d_c) > 1e-12 else float("nan")
            ),
            evaluation=evaluation,
        ))

    return out


# ============================================================
# 4. REPORTING
# ============================================================

def format_specificity_report(res: SpecificityResult) -> str:
    """Human-readable difference-in-differences report."""
    lines = [
        "=" * 72,
        f"FORMAT EFFECT SPECIFICITY  (layer {res.layer_idx})  "
        f"[benign control / diff-in-diff, {res.evaluation}]",
        "=" * 72,
        f"  DeltaP harmful (the headline effect) : {res.delta_p_harmful:+.4f}  "
        f"(n={res.n_harmful})",
        f"  DeltaP benign  (the control)         : {res.delta_p_benign:+.4f}  "
        f"(n={res.n_benign})",
        f"  benign arm moves? p                  : {res.benign_shift_p:.4f}",
        "-" * 72,
        f"  INTERACTION (harmful - benign)       : {res.interaction:+.4f}",
        f"  test                                 : "
        + ("within pairs (sign-flip on D_i)" if res.arms_paired
           else "two-sample (arms not matched)"),
        f"  permutation p                        : {res.interaction_p:.4f}",
        f"  effect size, {res.effect_size_kind:<24}: "
        f"{res.interaction_cohens_d:+.3f}",
        f"  95% CI                               : "
        f"[{res.interaction_ci_lower:+.4f}, {res.interaction_ci_upper:+.4f}]",
        f"  share of the effect common to benign : "
        f"{res.common_shift_fraction:+.1%}",
    ]
    if res.arms_paired:
        lines += [
            f"  pairs with D_i > 0                   : "
            f"{res.n_pairs_interaction_positive}/{res.n_harmful}",
            f"  corr(shift harmful, shift benign)    : "
            f"{res.shift_correlation:+.3f}",
            f"  two-sample reference (unpaired)      : "
            f"p={res.interaction_p_two_sample:.4f}  CI "
            f"[{res.interaction_ci_lower_two_sample:+.4f}, "
            f"{res.interaction_ci_upper_two_sample:+.4f}]",
        ]
    lines.append("-" * 72)
    if res.is_specific:
        lines += [
            "  VERDICT: the shift is SPECIFIC to harmful requests. A rigid",
            "           translation of the whole representation is ruled out,",
            "           so DeltaP cannot be dismissed as an offset.",
        ]
    elif res.interaction_p >= 0.05:
        lines += [
            "  VERDICT: no detectable interaction. The agentic format displaces",
            "           harmful and benign representations ALIKE, which is a",
            "           change of origin, not a weakening of refusal. DeltaP",
            "           must NOT be read as loss of safety representation.",
        ]
    else:
        lines += [
            "  VERDICT: interaction runs OPPOSITE to the hypothesis — the",
            "           direction responds more to harm under the agentic format.",
        ]
    lines.append("=" * 72)
    return "\n".join(lines)


def format_scale_report(res: ScaleResult) -> str:
    """Human-readable scale-invariance report."""
    lines = [
        "=" * 72,
        f"FORMAT EFFECT IN SCALE-INVARIANT UNITS  (layer {res.layer_idx}, "
        f"{res.evaluation})",
        "=" * 72,
        f"{'':<34}{'chat':>12}{'agent':>12}{'loss':>13}",
        "-" * 72,
        f"{'separation (raw units)':<34}{res.separation_chat:>12.4f}"
        f"{res.separation_agent:>12.4f}{res.separation_loss_raw:>12.1%}",
        f"{'pooled SD (raw units)':<34}{res.pooled_sd_chat:>12.4f}"
        f"{res.pooled_sd_agent:>12.4f}{'':>13}",
        f"{'separation (Cohen d)':<34}{res.cohens_d_chat:>12.4f}"
        f"{res.cohens_d_agent:>12.4f}{res.separation_loss_standardised:>12.1%}",
        f"{'AUROC':<34}{res.auroc_chat:>12.4f}"
        f"{res.auroc_agent:>12.4f}{res.auroc_loss:>12.1%}",
        "-" * 72,
        f"  scale contraction (SD_agent / SD_chat): {res.scale_contraction:.3f}",
    ]
    if res.raw_loss_is_inflated:
        lines += [
            "",
            "  WARNING: the raw loss materially exceeds the standardised one.",
            "           The agentic format compressed the projection scale, so",
            "           part of the headline DeltaP is a change of units and not",
            "           a loss of separability. Report the standardised figure",
            "           as the effect and the raw one as its scale-dependent",
            "           counterpart, exactly as the chapter already requires",
            "           when comparing across layers.",
        ]
    at_ceiling = res.auroc_chat > 0.97 and res.auroc_agent > 0.97
    if at_ceiling and abs(res.separation_loss_standardised) > 0.02:
        lines += [
            "",
            "  NOTE: AUROC is at ceiling in BOTH formats. It certifies that",
            "        detection survives, but it cannot resolve the "
            f"{abs(res.separation_loss_standardised):.0%} loss of margin the",
            "        standardised separation reports. Do not read a flat AUROC",
            "        as 'the mechanism is unaffected' — read it as 'the margin",
            "        shrank without crossing the decision boundary'.",
        ]
    lines.append("=" * 72)
    return "\n".join(lines)


def format_stratum_report(results: List[StratumDetection]) -> str:
    """Human-readable per-stratum detection report."""
    evaluation = results[0].evaluation if results else "in-sample"
    lines = [
        "=" * 72,
        "DETECTION CAPABILITY BY EXPLICITNESS  [objective 4: AUROC, not DeltaP]"
        f"  ({evaluation})",
        "=" * 72,
        f"{'stratum':<14}{'n_h':>5}{'n_b':>5}{'AUROC_chat':>12}{'AUROC_ag':>11}"
        f"{'d_chat':>9}{'d_agent':>9}{'d loss':>9}",
        "-" * 72,
    ]
    for r in results:
        lines.append(
            f"{r.stratum:<14}{r.n_harmful:>5}{r.n_benign:>5}{r.auroc_chat:>12.4f}"
            f"{r.auroc_agent:>11.4f}{r.cohens_d_chat:>9.3f}{r.cohens_d_agent:>9.3f}"
            f"{r.separation_loss_standardised:>8.1%}"
        )
    lines.append("-" * 72)
    if results:
        worst = max(results, key=lambda r: r.auroc_loss)
        lines += [
            f"  Detection degrades most in '{worst.stratum}' "
            f"(AUROC {worst.auroc_chat:.3f} -> {worst.auroc_agent:.3f}).",
        ]
        if all(r.auroc_agent > 0.9 for r in results):
            lines += [
                "  Detection holds above 0.90 in every stratum: the direction",
                "  still separates harm from benignity even when the intent is",
                "  covert. The objective's question is answered in the",
                "  affirmative, for this corpus.",
            ]
    lines.append("=" * 72)
    return "\n".join(lines)
