"""
Bootstrap confidence intervals for DeltaP, recomputed from saved results.

Why this script exists separately from the pipeline: the cross-validation of
the probes needs the full (n, 8192) activations, which are not saved and which
require a GPU pass to regenerate. The projections onto the refusal direction
ARE saved, in results.json under `projections_by_layer`, and DeltaP is a
function of nothing else. So the confidence interval for the headline result
can be recomputed on a laptop, for every layer, without re-running the model.

What it computes, per layer:

    DeltaP  = mean(proj chat_harmful) - mean(proj agent_harmful)

    a PAIRED bootstrap interval: one index vector is drawn and applied to BOTH
    conditions, because row i of each array is the SAME prompt in the two
    formats. Resampling the conditions independently would estimate the
    variance of a between-groups difference and inflate the interval.

    a percentile interval and a BCa interval. Percentile is first-order
    accurate; BCa corrects for median bias and for a variance that changes with
    the estimate, and is the one to quote at n ~ 42 when the two disagree.

    the sign test, which uses only the DIRECTION of each within-pair change and
    is therefore immune to every magnitude-related critique.

DeltaP is a direct measurement, not a fitted classifier. It is not subject to
the overfitting critique that applies to a probe's AUROC: there is no training
step whose in-sample fit could be mistaken for generalisation. Cross-fitting is
still worthwhile for a different reason — the DIRECTION is estimated from the
same chat activations that are then projected onto it — and that correction
lives in src/robust_estimation.cross_fitted_projection_gap, which needs the
activations. This script quantifies sampling uncertainty given the direction.

Run:
    python bootstrap_delta_p.py
    python bootstrap_delta_p.py --results results/results.json --n-boot 10000
    python bootstrap_delta_p.py --layer 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.robust_estimation import (  # noqa: E402
    paired_bootstrap_ci,
    paired_cohens_d,
    paired_permutation_test,
    paired_sign_test,
    paired_ttest_directional,
    pooled_cohens_d,
)

CHAT_KEY = "chat_harmful"
AGENT_KEYS = ("agent_harmful", "agent_full_harmful")


def _coerce_projection_array(value) -> Optional[np.ndarray]:
    """
    Accept either a JSON list or a numpy repr string.

    Runs produced before the `_jsonify` fix in run_experiment.py serialised the
    nested projection arrays through `json.dump(default=str)`, so they landed in
    results.json as the array's *repr*: "[-0.0638 -0.0992 ...]". Those files are
    still the record of what was measured, and parsing them back is preferable
    to declaring the run unusable — but only when the repr is COMPLETE. Numpy
    abbreviates arrays past its print threshold with a literal "...", and a
    silently truncated array would produce a confident interval over the wrong
    data. Truncated reprs are rejected rather than parsed.

    Returns None when the value cannot be recovered exactly.
    """
    if isinstance(value, list):
        try:
            return np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return None

    if not isinstance(value, str):
        return None

    text = value.strip()
    if "..." in text:                     # abbreviated repr: unrecoverable
        return None
    text = text.strip("[]").replace("\n", " ")
    try:
        parts = [float(tok) for tok in text.split()]
    except ValueError:
        return None
    return np.asarray(parts, dtype=float) if parts else None


def load_projections(path: Path) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict]:
    """
    Read `projections_by_layer` out of a results.json.

    Keys arrive as strings because JSON has no integer keys; they are converted
    back to int so layers sort numerically rather than lexicographically (which
    would order 10 before 5).

    The top-level `projections` block holds the selected layer at full float
    precision, while `projections_by_layer` may hold reprs rounded to 8
    significant digits. Where both are available for a layer, the full-precision
    copy wins — the rounding is far below the width of any interval computed
    here, but there is no reason to accept it when the exact values are present.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the experiment first, or pass --results."
        )

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    raw = data.get("projections_by_layer")
    if not raw:
        raise KeyError(
            f"{path} has no 'projections_by_layer'. Available top-level keys: "
            f"{sorted(data)}"
        )

    out: Dict[int, Dict[str, np.ndarray]] = {}
    n_reparsed = 0
    for layer_str, by_variant in raw.items():
        arrays: Dict[str, np.ndarray] = {}
        for variant, values in by_variant.items():
            arr = _coerce_projection_array(values)
            if arr is not None:
                arrays[variant] = arr
                if isinstance(values, str):
                    n_reparsed += 1
        if arrays:
            out[int(layer_str)] = arrays

    # Full-precision copy of the selected layer, when present.
    best_layer = data.get("best_layer")
    best = data.get("projections")
    if isinstance(best, dict) and best_layer is not None:
        exact = {
            variant: arr
            for variant, values in best.items()
            if (arr := _coerce_projection_array(values)) is not None
            and isinstance(values, list)
        }
        if exact:
            out.setdefault(int(best_layer), {}).update(exact)

    meta = {
        "best_layer": best_layer,
        "model": (data.get("model_info") or {}).get("model_type"),
        "hidden_size": (data.get("model_info") or {}).get("hidden_size"),
        "n_layers": (data.get("model_info") or {}).get("num_layers"),
        "n_reparsed_from_repr": n_reparsed,
    }
    return out, meta


def _resolve_agent_key(variants: Dict[str, np.ndarray]) -> Optional[str]:
    """The agent arm is named agent_harmful, or agent_full_harmful in --decompose runs."""
    for key in AGENT_KEYS:
        if key in variants:
            return key
    return None


def analyse_layer(
    proj_chat: np.ndarray,
    proj_agent: np.ndarray,
    layer: int,
    n_boot: int,
    n_permutations: int,
    ci_level: float,
    seed: int,
) -> Dict:
    """Full paired inference for one layer's projection pair."""
    if len(proj_chat) != len(proj_agent):
        raise ValueError(
            f"layer {layer}: {len(proj_chat)} chat vs {len(proj_agent)} agent "
            "projections. These must be paired row-wise."
        )

    boot = paired_bootstrap_ci(
        proj_chat, proj_agent, n_boot=n_boot, ci_level=ci_level, seed=seed, bca=True
    )
    perm = paired_permutation_test(
        proj_chat, proj_agent, n_permutations=n_permutations,
        alternative="two-sided", seed=seed,
    )
    t_stat, p_t = paired_ttest_directional(proj_chat, proj_agent, alternative="greater")
    n_dec, p_sign = paired_sign_test(proj_chat, proj_agent)

    return {
        "layer": int(layer),
        "n_pairs": int(len(proj_chat)),
        "delta_p": float(proj_chat.mean() - proj_agent.mean()),
        "mean_chat": float(proj_chat.mean()),
        "mean_agent": float(proj_agent.mean()),
        "sd_chat": float(proj_chat.std(ddof=1)),
        "sd_agent": float(proj_agent.std(ddof=1)),
        # Bootstrap
        "n_boot": int(n_boot),
        "ci_level": float(ci_level),
        "ci_lower": boot["ci_lower"],
        "ci_upper": boot["ci_upper"],
        "ci_lower_bca": boot["ci_lower_bca"],
        "ci_upper_bca": boot["ci_upper_bca"],
        "excludes_zero": boot["excludes_zero"],
        "excludes_zero_bca": boot["excludes_zero_bca"],
        "bootstrap_se": boot["std"],
        "p_bootstrap_two_sided": boot["p_bootstrap_two_sided"],
        "bias_correction_z0": boot["bias_correction_z0"],
        "acceleration": boot["acceleration"],
        # Companion tests
        "p_permutation": perm["p_value"],
        "p_value_floor": perm["p_value_floor"],
        "t_statistic": t_stat,
        "p_ttest_paired_onesided": p_t,
        "cohens_d_paired": paired_cohens_d(proj_chat, proj_agent),
        "cohens_d_pooled": pooled_cohens_d(proj_chat, proj_agent),
        "n_decreasing": int(n_dec),
        "sign_test_p": p_sign,
    }


def print_table(rows: List[Dict], best_layer: Optional[int], ci_level: float) -> None:
    """Per-layer table of DeltaP with its interval."""
    level = int(100 * ci_level)
    print()
    header = (f"{'layer':>6}{'n':>5}{'DeltaP':>10}"
              f"{f'{level}% CI (percentile)':>26}{f'{level}% CI (BCa)':>26}"
              f"{'excl. 0':>9}{'d_z':>8}{'signs':>9}")
    print(header)
    print("-" * len(header))
    for r in rows:
        mark = " *" if best_layer is not None and r["layer"] == best_layer else "  "
        pct = f"[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]"
        bca = f"[{r['ci_lower_bca']:+.4f}, {r['ci_upper_bca']:+.4f}]"
        excl = "yes" if r["excludes_zero_bca"] else "NO"
        signs = f"{r['n_decreasing']}/{r['n_pairs']}"
        print(f"{r['layer']:>4}{mark}{r['n_pairs']:>5}{r['delta_p']:>10.4f}"
              f"{pct:>26}{bca:>26}{excl:>9}{r['cohens_d_paired']:>8.2f}{signs:>9}")
    if best_layer is not None:
        print("\n  * layer selected by the pipeline")


def print_detail(r: Dict, ci_level: float) -> None:
    """Full report for the layer being reported in the write-up."""
    level = int(100 * ci_level)
    print()
    print("=" * 78)
    print(f"DeltaP AT LAYER {r['layer']}   (n = {r['n_pairs']} paired prompts)")
    print("=" * 78)
    print(f"  mean projection, chat harmful   : {r['mean_chat']:+.4f} "
          f"(sd {r['sd_chat']:.4f})")
    print(f"  mean projection, agent harmful  : {r['mean_agent']:+.4f} "
          f"(sd {r['sd_agent']:.4f})")
    print(f"  DeltaP                          : {r['delta_p']:+.4f}")
    print()
    print(f"  --- paired bootstrap, {r['n_boot']} resamples ---")
    print(f"  {level}% CI percentile : [{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]")
    print(f"  {level}% CI BCa        : [{r['ci_lower_bca']:+.4f}, "
          f"{r['ci_upper_bca']:+.4f}]   <- quote this one")
    print(f"  bootstrap SE      : {r['bootstrap_se']:.4f}")
    print(f"  bias correction z0: {r['bias_correction_z0']:+.4f}    "
          f"acceleration: {r['acceleration']:+.4f}")
    print(f"  two-sided p       : {r['p_bootstrap_two_sided']:.5f}")
    print()
    verdict = ("EXCLUDES zero" if r["excludes_zero_bca"] else "CONTAINS zero")
    print(f"  VERDICT: the {level}% BCa interval {verdict}.")
    if r["excludes_zero_bca"]:
        print("           The gap is not attributable to sampling noise.")
    else:
        print("           The gap is NOT distinguishable from noise at this n.")
    print()
    print("  --- companion tests ---")
    print(f"  sign-flip permutation p : {r['p_permutation']:.5f}   "
          f"(floor {r['p_value_floor']:g}; 0.0 means below the floor)")
    print(f"  paired t (one-sided)    : t = {r['t_statistic']:.3f}, "
          f"p = {r['p_ttest_paired_onesided']:.3g}")
    print(f"  Cohen d_z (paired)      : {r['cohens_d_paired']:.3f}")
    print(f"  Cohen d (pooled)        : {r['cohens_d_pooled']:.3f}")
    print(f"  pairs decreasing        : {r['n_decreasing']}/{r['n_pairs']}   "
          f"exact sign test p = {r['sign_test_p']:.5f}")
    print()
    print("  The sign test uses only the direction of each within-pair change,")
    print("  never a magnitude, so it is unaffected by the estimation of the")
    print("  direction or by any scaling of the projections.")
    print("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=str, default="results/llama31_hackathon/results.json",
                    help="results.json produced by run_experiment.py")
    ap.add_argument("--n-boot", type=int, default=10000,
                    help="bootstrap resamples (default 10000)")
    ap.add_argument("--n-permutations", type=int, default=10000,
                    help="sign-flip permutations (default 10000)")
    ap.add_argument("--ci-level", type=float, default=0.95,
                    help="confidence level (default 0.95)")
    ap.add_argument("--layer", type=int, default=None,
                    help="layer for the detailed report (default: the "
                         "pipeline's selected layer)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default=None,
                    help="where to write the JSON (default: alongside "
                         "--results as bootstrap_delta_p.json)")
    args = ap.parse_args()

    results_path = Path(args.results)
    projections, meta = load_projections(results_path)

    print("=" * 78)
    print("BOOTSTRAP CONFIDENCE INTERVALS FOR DeltaP")
    print("=" * 78)
    print(f"  source        : {results_path}")
    if meta.get("model"):
        print(f"  model         : {meta['model']}, {meta.get('n_layers')} layers, "
              f"hidden {meta.get('hidden_size')}")
    print(f"  resamples     : {args.n_boot} (paired: one index vector, both conditions)")
    print(f"  permutations  : {args.n_permutations} (sign-flip, exchangeable within pair)")
    if meta.get("n_reparsed_from_repr"):
        print(f"  note          : {meta['n_reparsed_from_repr']} projection arrays were "
              "stored as numpy reprs by an\n                  older _jsonify and have "
              "been re-parsed (8 significant digits).")

    rows: List[Dict] = []
    skipped: List[str] = []
    for layer in sorted(projections):
        variants = projections[layer]
        agent_key = _resolve_agent_key(variants)
        if CHAT_KEY not in variants or agent_key is None:
            skipped.append(f"layer {layer}: has {sorted(variants)}")
            continue
        rows.append(analyse_layer(
            variants[CHAT_KEY], variants[agent_key], layer,
            n_boot=args.n_boot, n_permutations=args.n_permutations,
            ci_level=args.ci_level, seed=args.seed,
        ))

    if not rows:
        print("\n  No layer carries both a chat_harmful and an agent harmful arm.")
        for s in skipped:
            print(f"    {s}")
        return 1

    if skipped:
        print(f"\n  Skipped {len(skipped)} layer(s) missing one of the two arms.")

    print_table(rows, meta.get("best_layer"), args.ci_level)

    target = args.layer if args.layer is not None else meta.get("best_layer")
    detail = next((r for r in rows if r["layer"] == target), rows[-1])
    print_detail(detail, args.ci_level)

    # --- Consistency across layers, which no single interval shows ---
    n_excl = sum(r["excludes_zero_bca"] for r in rows)
    print()
    print(f"  Across {len(rows)} layers, {n_excl} have a {int(100*args.ci_level)}% BCa "
          "interval excluding zero.")
    if n_excl == len(rows):
        print("  The gap is positive at every layer examined, not only at the")
        print("  selected one, so the result does not rest on the layer choice.")

    out_path = Path(args.output) if args.output else results_path.parent / "bootstrap_delta_p.json"
    payload = {
        "source": str(results_path),
        "n_boot": args.n_boot,
        "n_permutations": args.n_permutations,
        "ci_level": args.ci_level,
        "seed": args.seed,
        "selected_layer": meta.get("best_layer"),
        "reported_layer": detail["layer"],
        "per_layer": rows,
        "note": (
            "Paired bootstrap over prompts. DeltaP is a direct measurement, not "
            "a trained classifier, so it is not subject to the probe "
            "overfitting critique. The direction itself is estimated from the "
            "chat activations; the cross-fitted correction for that is in "
            "src/robust_estimation.cross_fitted_projection_gap and requires the "
            "activations, not just these projections."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  Saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
