"""
Re-analysis of a finished run, from the saved results only (no GPU, no model).

Three of the tutor's observations question HOW a number was computed rather
than what the model did, and the per-prompt projections needed to recompute
them are already on disk (results.json -> projections_by_layer). This script
answers them for an existing run instead of waiting for a new one.

1. INTERACTION WITH THE RIGHT PAIRING  (thesis sec. 3.3.1, "grupos
   independientes"). Harmful prompt i and benign prompt i are built from the
   same seed: same structure, same vocabulary, same tools. The two arms are
   therefore MATCHED, not independent, and the interaction is a within-pair
   double difference

        D_i = (chat_h,i - agent_h,i) - (chat_b,i - agent_b,i)

   tested against zero with a sign-flip permutation and a paired bootstrap.
   The two-sample version the pipeline used is reported beside it, so the cost
   or gain of respecting the pairing is visible rather than assumed.

2. LAYER SELECTION ON A SPLIT SAMPLE  (thesis sec. 4.2.1, "39 de 42 pares").
   The layer was chosen on one half of the pairs, but the count that was
   reported comes from all 42 - selection half included. Here the count is
   taken on the inference half only, the split respects pairs, and the whole
   procedure is repeated over many random splits so the answer does not hang
   on one partition.

3. LINEAR PROBE VERSUS MLP  (thesis table 4.1). Close means are not
   equivalence. The comparison is made fold by fold on the SAME partitions,
   with the Nadeau-Bengio variance correction, and reports the interval of the
   difference instead of two separate intervals.

LIMIT OF THIS SCRIPT. The saved projections were taken on a direction fitted
on all 42 pairs. Items 1 and 2 therefore inherit that direction's small
own-sample bias (1.4-4.4% of DeltaP in layers 24-56, see robust_results.json).
The exact, fully out-of-fold versions need the activations and run inside the
pipeline (src/robust_estimation.select_layer_and_test and
src/format_effect.specificity_from_projections).

Run:
    python scripts/reanalisis_observaciones.py
    python scripts/reanalisis_observaciones.py --results results/corridas/llama33_70b/v1/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.format_effect import specificity_from_projections  # noqa: E402
from src.probes import paired_fold_difference  # noqa: E402
from src.robust_estimation import paired_sign_test  # noqa: E402


def _arm(proj: Dict, harmful: bool, agent: bool) -> np.ndarray:
    """One condition's projections; tolerates the --decompose variant names."""
    suffix = "harmful" if harmful else "benign"
    names = ([f"agent_{suffix}", f"agent_full_{suffix}"] if agent
             else [f"chat_{suffix}"])
    for name in names:
        if name in proj:
            return np.asarray(proj[name], dtype=float)
    raise KeyError(f"none of {names} in {sorted(proj)}")


# ------------------------------------------------------------------
# 1. Interaction, paired by prompt
# ------------------------------------------------------------------

def paired_interaction(proj: Dict, n_resamples: int, seed: int) -> Dict:
    ch, ah = _arm(proj, True, False), _arm(proj, True, True)
    cb, ab = _arm(proj, False, False), _arm(proj, False, True)

    # The same function the pipeline now uses: it tests D_i within pairs and
    # carries the two-sample analysis alongside for comparison.
    res = specificity_from_projections(
        ch, ah, cb, ab, n_permutations=n_resamples, n_boot=n_resamples,
        seed=seed, paired_arms=True,
    )
    _, sign_p = paired_sign_test(ch - ah, cb - ab)
    return {
        "n_pairs": int(res.n_harmful),
        "interaction": res.interaction,
        "corr_shift_harmful_benign": res.shift_correlation,
        "paired": {
            "p_permutation": res.interaction_p,
            "ci95": [res.interaction_ci_lower, res.interaction_ci_upper],
            "cohens_dz": res.interaction_cohens_d,
            "n_pairs_positive": res.n_pairs_interaction_positive,
            "sign_test_p": float(sign_p),
        },
        "two_sample_as_reported": {
            "p_permutation": res.interaction_p_two_sample,
            "ci95": [res.interaction_ci_lower_two_sample,
                     res.interaction_ci_upper_two_sample],
        },
    }


# ------------------------------------------------------------------
# 2. Split-sample layer selection, inference half only
# ------------------------------------------------------------------

def _inference_stats(by_layer: Dict[int, Dict], layer: int,
                     idx: np.ndarray) -> Dict:
    proj = by_layer[layer]
    ch, ah = _arm(proj, True, False)[idx], _arm(proj, True, True)[idx]
    cb, ab = _arm(proj, False, False)[idx], _arm(proj, False, True)[idx]
    n_dec, p_sign = paired_sign_test(ch, ah)
    d = (ch - ah) - (cb - ab)
    return {
        "layer": int(layer),
        "n_inference": int(len(idx)),
        "delta_p": float((ch - ah).mean()),
        "n_decreasing": int(n_dec),
        "sign_test_p": float(p_sign),
        "interaction": float(d.mean()),
        "n_interaction_positive": int((d > 0).sum()),
    }


def split_sample_selection(by_layer: Dict[int, Dict], n_splits: int,
                           seed: int) -> Dict:
    layers = sorted(by_layer)
    n = len(_arm(by_layer[layers[0]], True, False))
    shift = {l: _arm(by_layer[l], True, False) - _arm(by_layer[l], True, True)
             for l in layers}

    def run(perm: np.ndarray) -> Dict:
        sel, inf = perm[: n // 2], perm[n // 2:]
        chosen = max(layers, key=lambda l: shift[l][sel].mean())
        return _inference_stats(by_layer, chosen, inf)

    # The partition the pipeline actually drew: default_rng(seed), first draw.
    pipeline_split = run(np.random.default_rng(seed).permutation(n))

    rng = np.random.default_rng(seed + 1)
    runs = [run(rng.permutation(n)) for _ in range(n_splits)]
    frac = np.array([r["n_decreasing"] / r["n_inference"] for r in runs])
    return {
        "note": ("direction fitted on all pairs; the split governs layer "
                 "choice and inference only"),
        "pipeline_split": pipeline_split,
        "n_splits": n_splits,
        "selected_layer_frequency": {
            str(l): c / n_splits
            for l, c in sorted(Counter(r["layer"] for r in runs).items())
        },
        "inference_fraction_decreasing": {
            "mean": float(frac.mean()),
            "p05": float(np.percentile(frac, 5)),
            "min": float(frac.min()),
        },
        "inference_delta_p": {
            "mean": float(np.mean([r["delta_p"] for r in runs])),
            "p05": float(np.percentile([r["delta_p"] for r in runs], 5)),
        },
        "share_splits_sign_test_below_bonferroni": float(np.mean(
            [r["sign_test_p"] < 0.05 / len(layers) for r in runs])),
    }


# ------------------------------------------------------------------
# 3. Linear probe vs MLP on the same folds
# ------------------------------------------------------------------

def probe_comparison(results: Dict) -> Dict:
    """
    Fold-wise paired difference MLP - linear, Nadeau-Bengio corrected.

    Both probes are cross-validated with seed + 1000*r for repeat r, so repeat
    r of one and repeat r of the other partition the data identically. Only
    the repeats both probes ran can be paired; with --mlp-n-repeats 1 that is
    the first K folds, and the interval is correspondingly wide.
    """
    lin = results["linear_suite"]["per_layer"]
    mlp = results["mlp_suite"]["per_layer"]
    out: Dict[str, Dict] = {}
    for layer in sorted(lin, key=int):
        a = mlp[layer].get("auroc_folds") or []
        b = lin[layer].get("auroc_folds") or []
        if min(len(a), len(b)) < 2:
            continue
        n = int(lin[layer].get("n_samples") or 84)
        k = int(lin[layer].get("n_folds") or 5)
        n_test = round(n / k)
        c = paired_fold_difference(a, b, n_train=n - n_test, n_test=n_test)
        out[layer] = {
            "n_paired_folds": c["n_paired_folds"],
            "auroc_linear": c["mean_b"],
            "auroc_mlp": c["mean_a"],
            "diff_mlp_minus_linear": c["difference"],
            "se_corrected": c["se_corrected"],
            "ci95": [c["ci_lower"], c["ci_upper"]],
            "p_two_sided": c["p_value"],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default="results/corridas/llama33_70b/v1/results.json")
    ap.add_argument("--n-resamples", type=int, default=10000)
    ap.add_argument("--n-splits", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layers", default="28,52,56",
                    help="layers to print in detail (all are saved)")
    args = ap.parse_args()

    path = Path(args.results)
    results = json.loads(path.read_text(encoding="utf-8"))
    by_layer = {int(l): p for l, p in results["projections_by_layer"].items()}
    show = [int(x) for x in args.layers.split(",") if int(x) in by_layer]

    print(f"Run: {path}   ({len(by_layer)} layers)")

    print("\n1. INTERACTION, PAIRED BY PROMPT")
    print(f"   {'layer':>5} {'D':>8} {'corr':>6} {'paired CI95':>17} "
          f"{'p':>8} {'dz':>6} {'D>0':>7} {'2-sample CI95':>17} {'p':>8}")
    inter: Dict[str, Dict] = {}
    for layer in sorted(by_layer):
        r = paired_interaction(by_layer[layer], args.n_resamples, args.seed)
        inter[str(layer)] = r
        if layer in show:
            p, t = r["paired"], r["two_sample_as_reported"]
            print(f"   {layer:>5} {r['interaction']:>8.3f} "
                  f"{r['corr_shift_harmful_benign']:>6.2f} "
                  f"[{p['ci95'][0]:>7.3f},{p['ci95'][1]:>7.3f}] "
                  f"{p['p_permutation']:>8.1e} "
                  f"{p['cohens_dz']:>6.2f} "
                  f"{p['n_pairs_positive']:>3}/{r['n_pairs']:<3} "
                  f"[{t['ci95'][0]:>7.3f},{t['ci95'][1]:>7.3f}] "
                  f"{t['p_permutation']:>8.1e}")
    n_sig = sum(r["paired"]["ci95"][0] > 0 for r in inter.values())
    print(f"   paired CI excludes zero in {n_sig}/{len(inter)} layers")

    print("\n2. LAYER SELECTION, INFERENCE HALF ONLY")
    sel = split_sample_selection(by_layer, args.n_splits, args.seed)
    ps = sel["pipeline_split"]
    print(f"   pipeline's own split -> layer {ps['layer']}: "
          f"{ps['n_decreasing']}/{ps['n_inference']} inference pairs decrease "
          f"(sign test p={ps['sign_test_p']:.1e}), DeltaP={ps['delta_p']:.3f}, "
          f"interaction={ps['interaction']:.3f} "
          f"({ps['n_interaction_positive']}/{ps['n_inference']} positive)")
    print(f"   over {sel['n_splits']} random pair splits, layer chosen: "
          + ", ".join(f"L{l} {f:.0%}"
                      for l, f in sel["selected_layer_frequency"].items()))
    f = sel["inference_fraction_decreasing"]
    print(f"   inference-half share of pairs decreasing: mean {f['mean']:.1%}, "
          f"5th pct {f['p05']:.1%}, worst split {f['min']:.1%}")
    print(f"   splits whose sign test clears Bonferroni: "
          f"{sel['share_splits_sign_test_below_bonferroni']:.1%}")

    print("\n3. LINEAR PROBE vs MLP, SAME FOLDS")
    cmp_ = probe_comparison(results) if "linear_suite" in results else {}
    for layer, r in cmp_.items():
        if int(layer) in show:
            print(f"   layer {layer:>2}: linear {r['auroc_linear']:.3f}  "
                  f"MLP {r['auroc_mlp']:.3f}  diff {r['diff_mlp_minus_linear']:+.3f} "
                  f"CI95 [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  "
                  f"p={r['p_two_sided']:.2f}  ({r['n_paired_folds']} paired folds)")
    if cmp_:
        prof = {int(l): v for l, v in results["linear_aurocs"].items()}
        peak, last = max(prof, key=prof.get), max(prof)
        print(f"   depth profile (linear, CV): layer {min(prof)} "
              f"{prof[min(prof)]:.3f} -> peak layer {peak} {prof[peak]:.3f} "
              f"-> layer {last} {prof[last]:.3f}  (rises, then falls)")

    out = {"source": str(path), "interaction_paired": inter,
           "layer_selection": sel, "probe_comparison": cmp_}
    out_path = path.with_name("reanalisis_observaciones.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
