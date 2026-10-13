"""
Main experiment pipeline.

Orchestrates the full experiment:
1. Load model
2. Build dataset
3. Extract activations (chat vs agent format)
4. Compute refusal direction (difference-in-means)
5. Project activations onto refusal direction
6. Run probing classifiers
7. Statistical tests (paired t-test, permutation, bootstrap)
8. Intervention experiments
9. Generate visualizations
10. Save results

Usage:
    python run_experiment.py --model mistralai/Mistral-7B-Instruct-v0.3
    python run_experiment.py --model meta-llama/Llama-3.1-8B-Instruct --no-quantize
    python run_experiment.py --skip-intervention  # Skip slow intervention step
"""

import argparse
import json
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# The reports print Greek deltas (ΔP) and other non-Latin-1 characters. A
# Windows console defaults to cp1252, which cannot encode them, so every run on
# Windows died with UnicodeEncodeError inside a print statement — after the
# expensive work had completed. Reconfiguring the streams keeps the symbols and
# fixes the crash everywhere at once; errors="replace" means an exotic terminal
# degrades a glyph instead of aborting the run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):        # non-reconfigurable stream
        pass

# The model stack (torch, transformers) is needed to extract activations and
# to generate. It is NOT needed to analyse activations saved by an earlier run
# (--load-activations), which is the path that lets every statistical question
# be re-asked on a laptop instead of on a rented GPU. Import it when present
# and fail only if a stage that needs it is actually reached.
try:
    from src.model_loader import (
        load_model_and_tokenizer,
        format_chat_prompt,
        format_agent_prompt,
        format_agent_prompt_with_history,
        format_role_only_prompt,
        format_role_plus_tools_prompt,
        get_model_info,
    )
    from src.activation_extractor import ActivationExtractor
    from src.intervention import (
        ActivationIntervention,
        compute_intervention_success_rate,
    )
    from src.agency import format_agency_prompt, expand_toolbox
    _MODEL_STACK_ERROR = None
except ImportError as _exc:
    _MODEL_STACK_ERROR = _exc
from src.refusal_direction import (
    compute_refusal_direction,
    compute_refusal_direction_pca,
    project_onto_direction,
    compute_gap_by_layer,
)
from src.probes import (
    run_probing_suite,
    compare_probe_vs_direction,
    print_probing_summary,
    compare_linear_vs_mlp,
)
from src.robust_estimation import pair_groups, run_paired_suite
from src.causal_test import (
    run_behavioral_baseline,
    run_causal_test,
    summarise_baseline,
    select_evenly,
    format_baseline_report,
    format_causal_report,
    write_generations,
)
from src.metrics import (
    paired_ttest,
    permutation_test,
    bootstrap_ci,
    compute_auroc_by_format,
    behavior_activation_concordance,
    cross_tool_analysis,
    print_full_report,
    decompose_format_effects,
)
from src.validation import validate_activations_paired
try:
    from src.visualizations import generate_all_figures
except ImportError:                     # matplotlib absent: skip the figures
    generate_all_figures = None
from data.loader import load_custom_dataset, load_harmagent_dataset, DatasetEntry
from data.build_dataset import build_dataset, HARMFUL_PROMPTS


def parse_args():
    parser = argparse.ArgumentParser(description="Run the agentic safety probe experiment")
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="HuggingFace model identifier",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="custom",
        choices=["custom", "harmagent", "both"],
        help="Dataset source: 'custom' (paired, for activation analysis), "
             "'harmagent' (benchmark, for behavioral validation), "
             "'both' (run both pipelines)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Paired dataset file for --dataset custom. Default: "
             "data/dataset_full.jsonl. Use data/dataset_expanded.jsonl for the "
             "corpus extended with AgentHarm (data/build_agentharm_pairs.py).",
    )
    parser.add_argument(
        "--agency-level",
        type=str,
        default=None,
        choices=["tools", "workflow", "autonomous", "trajectory"],
        help="Agentic scaffolding for the agent arm (src/agency.py). Default: "
             "the system prompt stored in each dataset entry, i.e. the original "
             "condition, which 'tools' reproduces. Takes precedence over "
             "--include-history.",
    )
    parser.add_argument(
        "--toolbox",
        type=int,
        default=0,
        help="Offer the model a toolbox of N tools instead of only the "
             "2-8 the task needs: the case tools plus distractors, identical "
             "in both arms of a pair. Selecting exactly the needed tools hands "
             "the model the plan; an autonomous agent chooses. 0 = off. "
             "12-16 is comfortable; 26 still fits the token budget.",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable 4-bit quantization (requires more VRAM)",
    )

    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Force 4-bit even when FP16 fits. Quantization control: same "
             "model and same prompts, only the precision changes",
    )

    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices to analyze (default: all)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20,
        help="Prompts used by the optional w_agent stage (--agent-intervention) "
             "only. It does not reduce activation extraction, and the causal "
             "test is sized by --intervention-pairs. Use --limit-pairs for a "
             "cheap dry run",
    )
    parser.add_argument(
        "--limit-pairs",
        type=int,
        default=None,
        help="Keep only the first N prompt PAIRS, across every variant. This is "
             "the flag that makes a dry run cheap: it cuts the forward passes, "
             "which are what the GPU time is spent on. Selection is by pair_id, "
             "so the chat/agent pairing every paired statistic depends on stays "
             "intact. Results from a limited run are a plumbing check, NOT "
             "science — the statistics are meaningless at N=4",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--skip-intervention",
        action="store_true",
        help="Skip intervention experiments (faster)",
    )
    parser.add_argument(
        "--skip-probes",
        action="store_true",
        help="Skip probing classifiers (faster)",
    )
    parser.add_argument(
        "--include-history",
        action="store_true",
        help="Also test agent format with tool-use history",
    )
    parser.add_argument(
        "--decompose",
        action="store_true",
        help="Run decomposition analysis with 4 intermediate conditions "
             "(chat → role_only → role_plus_tools → agent_full)",
    )
    parser.add_argument(
        "--skip-robust",
        action="store_true",
        help="Skip the bias-corrected re-analysis (cross-fitting, out-of-fold "
             "evaluation of w_agent and the monitor)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Folds for cross-fitting and cross-validation (default: 5)",
    )
    parser.add_argument(
        "--skip-heldout",
        action="store_true",
        help="Skip the held-out evaluation of d_chat. Without it every AUROC "
             "in the main analysis is scored with a direction fitted on the "
             "same prompts, which is a fit statistic rather than a detection "
             "rate. For debugging only",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=1,
        help="Independent random partitions for the probe cross-validation. "
             "1 is plain K-fold; 10 is recommended for a reported run, since "
             "at n=84 a single partition puts ~17 samples per fold and the "
             "estimate moves with the seed (default: 1)",
    )
    parser.add_argument(
        "--mlp-n-repeats",
        type=int,
        default=None,
        help="Separate repeat count for the MLP probe, which is far more "
             "expensive per fit. Defaults to --n-repeats; set to 1 to keep a "
             "high-repeat linear run affordable",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Bootstrap resamples for the ΔP confidence interval (default: 10000)",
    )
    parser.add_argument(
        "--n-label-permutations",
        type=int,
        default=10,
        help="Repeats of the label-permutation control for the linear probe. "
             "Each repeat re-runs the whole K-fold pass on shuffled labels to "
             "measure the empirical chance level at this n and d. 0 disables "
             "(default: 10)",
    )
    parser.add_argument(
        "--no-probe-cv",
        action="store_true",
        help="Restore the old single 80/20 split for the probes instead of "
             "K-fold cross-validation. For comparison only — the single-split "
             "AUROC is what the cross-validation replaces",
    )
    parser.add_argument(
        "--save-activations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the extracted activations to <output-dir>/activations.npz "
             "(a few hundred MB). On by default: with them on disk, every "
             "analysis stage can be re-run with --load-activations and no GPU",
    )
    parser.add_argument(
        "--load-activations",
        type=str,
        default=None,
        help="Path to an activations.npz written by an earlier run. Skips the "
             "model entirely and re-runs the analysis stages on the saved "
             "activations; the generation stages are skipped",
    )
    parser.add_argument(
        "--factorial",
        action="store_true",
        help="Also extract two intermediate conditions per prompt - the agentic "
             "system message WITHOUT tool schemas, and the chat system message "
             "WITH them - so the format effect can be split into what the "
             "system message contributes and what the tools contribute",
    )
    parser.add_argument(
        "--intervention-layers",
        type=str,
        default=None,
        help="Comma-separated layers for the causal test. Default: the layer "
             "chosen by split-sample selection (falls back to the best layer)",
    )
    parser.add_argument(
        "--intervention-doses",
        type=str,
        default="0.5,1,2,4",
        help="Doses for the causal test, as multiples of the projection gap "
             "measured at the intervention layer. 1 restores exactly the "
             "projection the agentic format removed (default: 0.5,1,2,4)",
    )
    parser.add_argument(
        "--intervention-pairs",
        type=int,
        default=None,
        help="Prompt pairs used by the behavioural baseline and the causal "
             "test, spread evenly over the corpus. Default: all of them",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Tokens generated per response in the behavioural and causal "
             "stages. A refusal is stated in the first sentence, so 64 is "
             "enough to label it (default: 64)",
    )
    parser.add_argument(
        "--agent-intervention",
        action="store_true",
        help="Also run the older stage that adds w_agent at its own layer over "
             "a fixed alpha grid. Off by default: its alphas are not calibrated "
             "to the measured effect",
    )
    parser.add_argument(
        "--n-selection-splits",
        type=int,
        default=200,
        help="Random pair partitions used to report how stable the split-sample "
             "layer selection is (default: 200, 0 disables)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def _pair_id(entry: Dict) -> str:
    return str(entry.get("pair_id", entry.get("id", "")))


def format_entry(
    tokenizer,
    entry: Dict,
    include_history: bool = False,
    agency_level: Optional[str] = None,
    toolbox: int = 0,
) -> str:
    """
    Turn one corpus entry into the prompt string the model receives.

    This is the ONLY place that mapping is defined. The extraction stage and
    the generation stages both call it, so the prompt whose activation is
    measured and the prompt the model is asked to answer cannot differ. They
    used to be built by separate copies of this logic; one copy dropped the
    per-prompt tools, the full 26-tool catalogue overflowed the token limit,
    the user message was truncated away, and the intervention stage of a paid
    run steered a model that was completing a JSON tool list.
    """
    fmt = entry["format"]
    user = entry["base_prompt"]

    if fmt == "chat":
        # Use the corpus system prompt, not the module default. The module
        # default reads "helpful, harmless, and honest"; that 'harmless' is a
        # safety-priming token present in only one arm of the contrast, and it
        # loads the very direction being measured. The corpus prompt is the
        # neutral one the methodology describes.
        return format_chat_prompt(
            tokenizer, user, system_message=entry.get("system_prompt") or None,
        )
    if fmt == "role_only":
        return format_role_only_prompt(tokenizer, user)
    if fmt == "role_plus_tools":
        return format_role_plus_tools_prompt(tokenizer, user)

    if fmt in ("agent_full", "agent", "sys_only", "tools_only"):
        # Per-entry tools, not the full 26-tool catalogue (which inflates the
        # prompt and risks truncating the final token).
        sys_prompt = entry.get("system_prompt") or None
        entry_tools = entry.get("tool_definitions") or None

        if fmt == "sys_only":
            # Agentic system message, NO tool schemas. `tools=[]` is the
            # explicit way to ask format_agent_prompt for that.
            return format_agent_prompt(
                tokenizer, user, tools=[], system_message=sys_prompt,
            )
        if fmt == "tools_only":
            # Tool schemas under the CHAT system message.
            chat_sys = entry.get("chat_system_prompt")
            if not chat_sys or not entry_tools:
                raise ValueError(
                    f"tools_only needs the pair's chat system prompt and its "
                    f"tools; entry {_pair_id(entry)} lacks one of them."
                )
            return format_agent_prompt(
                tokenizer, user, tools=entry_tools, system_message=chat_sys,
            )

        if toolbox:
            # Un agente elige entre lo que tiene: se completa la caja
            # con distractores, identicos en los dos brazos del par.
            entry_tools = expand_toolbox(_pair_id(entry), entry_tools, toolbox)
        if agency_level:
            return format_agency_prompt(
                tokenizer, user, tools=entry_tools, level=agency_level,
            )
        if include_history:
            return format_agent_prompt_with_history(tokenizer, user)
        return format_agent_prompt(
            tokenizer, user, tools=entry_tools, system_message=sys_prompt,
        )

    return format_chat_prompt(tokenizer, user)


def _factorial_variants(variants: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """
    Two intermediate conditions per prompt, derived from the agentic entries.

    The agentic format differs from the chat format in two things at once: the
    system message (which announces tools and asks for tool calls) and the tool
    schemas themselves. The chat-vs-agent contrast cannot say which of the two
    moves the projection. Crossing them gives a 2x2 per prompt:

                          no tool schemas        tool schemas
        chat system       chat_*   (corpus)      toolsonly_*  (derived)
        agent system      sysonly_* (derived)    agent_*      (corpus)

    The instruction text is identical in all four cells.
    """
    out: Dict[str, List[Dict]] = {}
    for label in ("harmful", "benign"):
        agent = variants.get(f"agent_{label}") or variants.get(f"agent_full_{label}")
        chat = variants.get(f"chat_{label}")
        if not agent or not chat:
            continue
        chat_sys = {_pair_id(e): e.get("system_prompt") for e in chat}
        out[f"sysonly_{label}"] = [
            dict(e, format="sys_only", variant=f"sysonly_{label}") for e in agent
        ]
        out[f"toolsonly_{label}"] = [
            dict(e, format="tools_only", variant=f"toolsonly_{label}",
                 chat_system_prompt=chat_sys.get(_pair_id(e)))
            for e in agent
        ]
    return out


def extract_all_activations(
    extractor: "ActivationExtractor",
    tokenizer,
    dataset: List[Dict],
    layers: List[int],
    include_history: bool = False,
    decompose: bool = False,
    agency_level: Optional[str] = None,
    toolbox: int = 0,
    factorial: bool = False,
) -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict[str, List[str]]]:
    """
    Extract activations for all dataset entries.

    Every downstream paired statistic assumes row i of `chat_harmful` and row i
    of `agent_harmful` are the SAME prompt. That alignment used to rest on the
    dataset file happening to list the variants in matching order — true for
    the current file, silently false the moment anyone reorders or filters it,
    and the failure would show up as a plausible-looking wrong number rather
    than an error. Entries are therefore sorted by `pair_id` here, which makes
    the alignment a property of the code instead of a property of the file.

    Returns:
        (activations, pair_order) where
          activations: variant -> {layer_idx -> (n_samples, hidden_dim)}
          pair_order:  variant -> list of pair_id, row-aligned with the arrays

    Prompt lengths per variant are left on `extractor.token_stats`.
    """
    print("\n" + "=" * 60)
    print("EXTRACTING ACTIVATIONS")
    print("=" * 60)

    # Determine variants present in dataset, rows ordered by pair_id
    variant_names = sorted(set(d["variant"] for d in dataset))
    variants = {
        v: sorted((d for d in dataset if d["variant"] == v), key=_pair_id)
        for v in variant_names
    }
    if factorial:
        if include_history or agency_level or toolbox:
            raise ValueError(
                "--factorial splits the corpus system message from the corpus "
                "tools; it cannot be combined with --include-history, "
                "--agency-level or --toolbox, which replace them."
            )
        variants.update(_factorial_variants(variants))

    all_activations = {}
    pair_order = {v: [_pair_id(e) for e in entries]
                  for v, entries in variants.items()}
    token_stats: Dict[str, Dict[str, float]] = {}

    for variant_name, entries in variants.items():
        print(f"\n  Processing {variant_name} ({len(entries)} prompts)...")
        prompts = [
            format_entry(tokenizer, entry, include_history=include_history,
                         agency_level=agency_level, toolbox=toolbox)
            for entry in entries
        ]

        # Extract activations for all prompts in this variant
        variant_activations = {layer: [] for layer in layers}
        lengths: List[int] = []

        for i, prompt in enumerate(prompts):
            if (i + 1) % 10 == 0:
                print(f"    {i + 1}/{len(prompts)}")

            acts = extractor.extract_single(prompt, layers=layers)
            for layer in layers:
                variant_activations[layer].append(acts[layer])

            ids = extractor.last_input_ids
            lengths.append(len(ids))
            if i == 0:
                # Show where the sequence starts and which token the
                # activation is read from. The last tokens must be the
                # assistant header in EVERY variant - that is what makes the
                # final position comparable across formats - and the sequence
                # must open with a single BOS.
                head = tokenizer.convert_ids_to_tokens(ids[:3])
                tail = tokenizer.convert_ids_to_tokens(ids[-5:])
                print(f"    first tokens: {head}")
                print(f"    last tokens : {tail}   <- activation read at the last one")

        # Stack into arrays
        all_activations[variant_name] = {
            layer: np.stack(variant_activations[layer])
            for layer in layers
        }
        token_stats[variant_name] = {
            "mean": float(np.mean(lengths)),
            "min": int(np.min(lengths)),
            "max": int(np.max(lengths)),
        }
        print(f"    Shape: {all_activations[variant_name][layers[0]].shape}   "
              f"prompt length: mean {token_stats[variant_name]['mean']:.0f} "
              f"tokens (min {token_stats[variant_name]['min']}, "
              f"max {token_stats[variant_name]['max']})")

    _assert_pairing(pair_order)
    extractor.token_stats = token_stats
    return all_activations, pair_order


def save_activations(
    path: str,
    activations: Dict[str, Dict[int, np.ndarray]],
    pair_order: Dict[str, List[str]],
    meta: Dict,
) -> None:
    """
    Write every extracted activation to one .npz.

    The forward passes are the only part of this pipeline that needs a GPU.
    Everything after them is arithmetic on (n_prompts, hidden_dim) arrays that
    a laptop handles. With the arrays saved, a question about the statistics -
    a different partition, another control, a probe comparison - is answered
    by re-running the analysis with --load-activations, not by renting the
    hardware again.
    """
    arrays = {
        f"{variant}__L{layer}": np.asarray(acts, dtype=np.float32)
        for variant, by_layer in activations.items()
        for layer, acts in by_layer.items()
    }
    arrays["__meta__"] = np.array(json.dumps(dict(meta, pair_order=pair_order)))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    size_mb = Path(path).stat().st_size / 1e6
    print(f"\n  Activations saved to {path}  ({size_mb:.0f} MB)")


def load_activations(path: str) -> Tuple[Dict, Dict, Dict]:
    """Inverse of save_activations: (activations, pair_order, meta)."""
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["__meta__"]))
    activations: Dict[str, Dict[int, np.ndarray]] = {}
    for key in data.files:
        if key == "__meta__":
            continue
        variant, layer = key.rsplit("__L", 1)
        activations.setdefault(variant, {})[int(layer)] = data[key]
    pair_order = meta.pop("pair_order")
    _assert_pairing(pair_order)
    return activations, pair_order, meta


def _assert_pairing(pair_order: Dict[str, List[str]]) -> None:
    """Fail loudly if the harmful/benign arms of a format are not row-aligned."""
    for fmt in ("chat", "agent", "agent_full", "role_only", "role_plus_tools",
                "sysonly", "toolsonly"):
        h, b = f"{fmt}_harmful", f"{fmt}_benign"
        if h in pair_order and b in pair_order and pair_order[h] != pair_order[b]:
            raise ValueError(
                f"{h} and {b} are not row-aligned by pair_id; paired statistics "
                "would compare different prompts."
            )
    harmful = [v for v in pair_order if v.endswith("_harmful")]
    if len(harmful) > 1:
        ref = pair_order[harmful[0]]
        for v in harmful[1:]:
            if pair_order[v] != ref:
                raise ValueError(
                    f"{v} is not row-aligned with {harmful[0]}; the cross-format "
                    "pairing that DeltaP depends on is broken."
                )


def run_main_analysis(
    activations: Dict[str, Dict[int, np.ndarray]],
    layers: List[int],
    n_bootstrap: int = 10000,
    seed: int = 42,
    n_folds: int = 5,
    use_heldout: bool = True,
) -> Dict:
    """
    Run the main analysis pipeline:
    - Compute refusal direction
    - Project activations
    - Compute gaps, on the harmful arm AND on the benign control arm
    - Statistical tests, including the harmful-vs-benign interaction
    - Detection (AUROC), scored both in-sample and out-of-fold

    Two questions the earlier version could not answer are answered here.

    First, whether the chat-to-agent shift belongs to refusal at all. The
    agentic prompt is longer, carries a tool schema and a different system
    message, and any of that can translate the whole residual stream along
    d_chat. A DeltaP measured only on harmful prompts cannot tell a weakened
    refusal representation from a rigid displacement of every representation.
    The benign arm is therefore measured with the same paired statistics, and
    the interaction between the two arms is what is specific to harm.

    Second, whether the reported AUROC is a detection rate. d_chat is the
    difference between the chat_harmful and chat_benign centroids, so asking
    how well it separates those same two groups is asking how well a line fits
    the points it was fitted to. The held-out evaluation scores every prompt
    with a direction estimated without it; both numbers are reported, and
    their difference is the optimism.
    """
    from src.format_effect import (
        specificity_from_projections,
        format_specificity_report,
    )

    print("\n" + "=" * 60)
    print("COMPUTING REFUSAL DIRECTION")
    print("=" * 60)

    results = {}

    # Resolve variant names (backward compat: agent_harmful or agent_full_harmful)
    agent_harmful_key = "agent_harmful" if "agent_harmful" in activations else "agent_full_harmful"
    agent_benign_key = "agent_benign" if "agent_benign" in activations else "agent_full_benign"

    # Compute refusal direction at each layer
    refusal_directions = {}
    for layer in layers:
        direction = compute_refusal_direction(
            harmful_activations=activations["chat_harmful"][layer],
            benign_activations=activations["chat_benign"][layer],
        )
        refusal_directions[layer] = direction

    results["refusal_directions"] = refusal_directions
    print(f"  Computed refusal directions for {len(layers)} layers")

    # Project all activations onto refusal direction
    print("\n  Projecting activations...")
    projections = {}
    for layer in layers:
        projections[layer] = {}
        for variant in activations.keys():
            if layer in activations[variant]:
                proj = project_onto_direction(
                    activations[variant][layer],
                    refusal_directions[layer],
                )
                projections[layer][variant] = proj

    results["projections_by_layer"] = projections

    # Compute gap at each layer (chat vs agent_full)
    layer_results = compute_gap_by_layer(
        harmful_activations={l: activations["chat_harmful"][l] for l in layers},
        benign_activations={l: activations["chat_benign"][l] for l in layers},
        chat_harmful_acts={l: activations["chat_harmful"][l] for l in layers},
        agent_harmful_acts={l: activations[agent_harmful_key][l] for l in layers},
        # The methodology states the contrast is validated by PCA. The sweep
        # ran with it switched off, so `pca_validation` was None at every layer
        # and there was nothing to report. A randomized SVD on (84, 8192) costs
        # milliseconds; there is no reason to skip it.
        run_pca=True,
    )
    results["pca_by_layer"] = {
        l: r["pca_validation"] for l, r in layer_results.items()
        if r.get("pca_validation")
    }
    if results["pca_by_layer"]:
        print("\n  PCA validation of the chat contrast (pooled, globally centred):")
        print(f"    {'layer':>7}{'cos(PC1,d)':>12}{'best PC':>9}{'cos':>7}"
              f"{'var PC1':>9}{'var along d':>13}{'between/d':>11}")
        for l in sorted(results["pca_by_layer"]):
            v = results["pca_by_layer"][l]
            print(f"    {l:>7}{v['cosine_similarity_with_diff_means']:>12.3f}"
                  f"{v['best_aligned_component']:>9d}"
                  f"{v['cosine_best_component']:>7.3f}"
                  f"{v['explained_variance_ratio_pc1']:>9.1%}"
                  f"{v['variance_fraction_along_direction']:>13.1%}"
                  f"{v['between_class_share_along_direction']:>11.1%}")
        print("    cos(PC1,d) near 1: the contrast is the dominant axis. A low")
        print("    value with a high 'between/d' means the direction separates")
        print("    the classes cleanly but is not the largest source of variance.")
    # Extract gaps from layer results
    gaps = {l: r["gap_analysis"]["delta_p"] for l, r in layer_results.items()}
    results["gaps_by_layer"] = gaps
    results["layer_results"] = layer_results
    print(f"  Gaps computed. Max gap at layer {max(gaps, key=gaps.get)}: {max(gaps.values()):.4f}")

    # ---- BENIGN CONTROL: the same shift, measured on benign prompts ----
    # Computed for every layer, because a shift that is common to both arms is
    # a property of the format and should be visible as such across depth, not
    # only at the layer the harmful arm happens to peak at.
    gaps_benign = {}
    for layer in sorted(gaps):
        p_layer = projections[layer]
        gaps_benign[layer] = float(
            p_layer["chat_benign"].mean() - p_layer[agent_benign_key].mean()
        )
    interaction_by_layer = {
        layer: float(gaps[layer] - gaps_benign[layer]) for layer in gaps_benign
    }
    results["gaps_by_layer_benign"] = gaps_benign
    results["interaction_by_layer"] = interaction_by_layer

    print("\n  Format shift by layer, harmful arm vs benign control:")
    print(f"    {'layer':>7}{'DP_harmful':>12}{'DP_benign':>12}{'interaction':>13}")
    for layer in sorted(gaps_benign):
        print(f"    {layer:>7}{gaps[layer]:>12.4f}{gaps_benign[layer]:>12.4f}"
              f"{interaction_by_layer[layer]:>13.4f}")

    # Find best layer (largest gap)
    best_layer = max(gaps, key=gaps.get)
    results["best_layer"] = best_layer
    print(f"\n  Best layer for analysis: {best_layer}")

    # Selection is unchanged — the largest harmful shift — but the layer where
    # the effect is most SPECIFIC to harm need not be the same one, and if it
    # is not, the reported layer is showing the format's common displacement
    # at its strongest rather than refusal's weakening at its strongest.
    best_layer_interaction = max(interaction_by_layer, key=interaction_by_layer.get)
    results["best_layer_by_interaction"] = int(best_layer_interaction)
    if best_layer_interaction != best_layer:
        print(f"  NOTE: the harmful-vs-benign interaction peaks at layer "
              f"{best_layer_interaction} "
              f"({interaction_by_layer[best_layer_interaction]:.4f}), not at "
              f"layer {best_layer} ({interaction_by_layer[best_layer]:.4f}).")
        print("        Layer selection still follows the harmful gap; the two")
        print("        differ, so report which criterion the quoted layer used.")

    # Statistical tests at best layer
    print("\n" + "=" * 60)
    print(f"STATISTICAL TESTS (Layer {best_layer})")
    print("=" * 60)

    proj_chat = projections[best_layer]["chat_harmful"]
    proj_agent = projections[best_layer].get("agent_harmful",
                 projections[best_layer].get("agent_full_harmful"))

    # Paired t-test
    ttest_result = paired_ttest(proj_chat, proj_agent)
    results["ttest"] = ttest_result
    print(f"  Paired t-test: t={ttest_result.statistic:.4f}, p={ttest_result.p_value:.6f}")
    print(f"  Cohen's d: {ttest_result.effect_size:.4f}")

    # Permutation test
    perm_result = permutation_test(proj_chat, proj_agent)
    results["permutation_test"] = perm_result
    print(f"  Permutation test: p={perm_result.p_value:.6f}")

    # Bootstrap CI for ΔP. Paired resampling over prompts, 10,000 resamples by
    # default; percentile and BCa intervals, plus the explicit zero-crossing
    # verdict that the interval exists to deliver.
    bootstrap = bootstrap_ci(
        proj_chat, proj_agent, n_bootstrap=n_bootstrap, random_seed=seed
    )
    results["bootstrap"] = bootstrap
    print(f"  Bootstrap ΔP ({n_bootstrap} resamples): {bootstrap['mean_delta_p']:.4f} "
          f"[{bootstrap['ci_lower']:.4f}, {bootstrap['ci_upper']:.4f}] percentile")
    print(f"                                  BCa: "
          f"[{bootstrap['ci_lower_bca']:.4f}, {bootstrap['ci_upper_bca']:.4f}]")
    print(f"  CI excludes zero: {bootstrap['excludes_zero']}")

    # ---- The same three tests on the benign arm ----
    # Identical statistics, identical pairing, a different set of prompts. Any
    # asymmetry between this block and the one above is the effect that is
    # specific to harmful requests; anything they share is the format moving
    # everything at once.
    proj_chat_benign = projections[best_layer]["chat_benign"]
    proj_agent_benign = projections[best_layer][agent_benign_key]

    ttest_benign = paired_ttest(proj_chat_benign, proj_agent_benign)
    perm_benign = permutation_test(proj_chat_benign, proj_agent_benign)
    bootstrap_benign = bootstrap_ci(
        proj_chat_benign, proj_agent_benign,
        n_bootstrap=n_bootstrap, random_seed=seed,
    )
    results["ttest_benign"] = ttest_benign
    results["permutation_test_benign"] = perm_benign
    results["bootstrap_benign"] = bootstrap_benign

    print(f"\n  BENIGN CONTROL (same tests, benign prompts)")
    print(f"    Paired t-test: t={ttest_benign.statistic:.4f}, "
          f"p={ttest_benign.p_value:.6f}")
    print(f"    Permutation test: p={perm_benign.p_value:.6f}")
    print(f"    Bootstrap DeltaP_benign: {bootstrap_benign['mean_delta_p']:.4f} "
          f"[{bootstrap_benign['ci_lower']:.4f}, "
          f"{bootstrap_benign['ci_upper']:.4f}]")
    print(f"    CI excludes zero: {bootstrap_benign['excludes_zero']}")

    # Difference-in-differences: the contrast the two arms exist to support.
    specificity = specificity_from_projections(
        chat_harmful=proj_chat,
        agent_harmful=proj_agent,
        chat_benign=proj_chat_benign,
        agent_benign=proj_agent_benign,
        layer_idx=best_layer,
        seed=seed,
        evaluation="in-sample",
    )
    results["specificity"] = specificity
    print()
    print(format_specificity_report(specificity))

    # AUROC by format. In-sample: d_chat was fitted on exactly these prompts.
    auroc = compute_auroc_by_format(projections[best_layer])
    results["auroc"] = auroc
    print(f"  AUROC chat (in-sample): {auroc['auroc_chat']:.4f}")
    print(f"  AUROC agent (in-sample): {auroc['auroc_agent']:.4f}")

    # ---- Held-out evaluation: fit and score on disjoint prompts ----
    auroc_heldout = None
    if use_heldout:
        try:
            results["heldout"] = run_heldout_evaluation(
                activations=activations,
                layers=layers,
                best_layer=best_layer,
                n_folds=n_folds,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            detection = results["heldout"].get("detection")
            if detection:
                auroc_heldout = {
                    "auroc_chat": detection["auroc_chat_heldout"],
                    "auroc_agent": detection["auroc_agent_heldout"],
                    "auroc_gap": detection["auroc_gap_heldout"],
                }
                results["auroc_heldout"] = auroc_heldout
        except Exception as exc:                       # noqa: BLE001
            # The forward passes are the expensive part and they are already
            # done; a failure here must not cost the whole run.
            print(f"  WARNING: held-out evaluation failed ({exc}). "
                  "Only in-sample AUROC is available.")

    # Print full report
    print_full_report(
        ttest_result, perm_result, bootstrap, auroc,
        auroc_heldout=auroc_heldout,
        benign={
            "delta_p": bootstrap_benign["mean_delta_p"],
            "p_value": ttest_benign.p_value,
            "interaction": specificity.interaction,
            "interaction_p": specificity.interaction_p,
            "common_shift_fraction": specificity.common_shift_fraction,
        },
    )

    # Store projections for visualization
    results["projections"] = projections[best_layer]

    return results


def run_heldout_evaluation(
    activations: Dict[str, Dict[int, np.ndarray]],
    layers: List[int],
    best_layer: int,
    n_folds: int = 5,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict:
    """
    Re-measure the format effect with the fitting and evaluation prompts split.

    d_chat is the difference between the chat_harmful and chat_benign
    centroids. Every quantity the pipeline then computes from it — AUROC,
    DeltaP, the standardised separation — is measured on the very prompts that
    defined it. At n = 42 in d = 8192 the direction absorbs a large amount of
    sample-specific noise, and projecting those same prompts back recovers it.
    The result flatters the chat format specifically, because the agent
    condition never entered the estimation, so the chat-vs-agent comparison
    inherits an asymmetry that has nothing to do with format.

    Cross-fitting removes it: the prompt pairs are split into K folds, the
    direction is fitted on K-1 of them, and only the held-out fold is scored.
    Folds are assigned over PAIRS, so a prompt's harmful and benign variants —
    near-paraphrases built from the same seed — are always held out together.
    Every prompt still contributes a score, which is why this is preferred over
    a single train/test split at this n.

    Both arms and both formats are scored on that same held-out footing, so
    the benign control and the difference-in-differences are available here
    too, free of the same circularity.

    Returns:
        Dict with per-layer held-out gaps and AUROCs, the detection comparison
        at `best_layer`, the paired suites for each arm, the held-out
        difference-in-differences, and the out-of-fold projections (kept for
        the figures and for the format-effect stage).
    """
    from src.robust_estimation import (
        crossfit_condition_projections,
        held_out_detection,
        format_heldout_detection_report,
        run_paired_suite,
    )
    from src.format_effect import (
        specificity_from_projections,
        format_specificity_report,
    )
    from sklearn.metrics import roc_auc_score

    print("\n" + "=" * 60)
    print("HELD-OUT EVALUATION OF d_chat (fit and scoring prompts disjoint)")
    print("=" * 60)

    ah_key = "agent_harmful" if "agent_harmful" in activations else "agent_full_harmful"
    ab_key = "agent_benign" if "agent_benign" in activations else "agent_full_benign"
    required = ["chat_harmful", "chat_benign", ah_key, ab_key]
    missing = [k for k in required if k not in activations]
    if missing:
        print(f"  WARNING: missing conditions {missing}. Skipping.")
        return {}

    n_pairs = len(activations["chat_harmful"][layers[0]])
    # K cannot exceed the number of pairs. A --limit-pairs dry run drives
    # n_pairs to 4 while n_folds defaults to 5; clamping keeps the dry run able
    # to exercise this path. At the real n nothing is clamped.
    usable_folds = max(2, min(n_folds, n_pairs))
    if usable_folds != n_folds:
        print(f"  NOTE: K reduced from {n_folds} to {usable_folds} "
              f"({n_pairs} pairs available). Expected on a dry run; these "
              "numbers are not meaningful.")
    n_folds = usable_folds

    def _auroc(harmful, benign):
        y = np.r_[np.ones(len(harmful)), np.zeros(len(benign))]
        return float(roc_auc_score(y, np.r_[harmful, benign]))

    out: Dict = {"n_folds": n_folds, "n_pairs": int(n_pairs), "per_layer": {}}
    projections_by_layer: Dict[int, Dict[str, np.ndarray]] = {}
    by_layer_objects: Dict[int, object] = {}

    for layer in layers:
        # Every variant that is row-aligned by pair travels through the same
        # folds, so the decomposition conditions get held-out scores too.
        conditions = {
            name: acts[layer]
            for name, acts in activations.items()
            if layer in acts and len(acts[layer]) == n_pairs
        }
        ho = crossfit_condition_projections(
            conditions, n_folds=n_folds, seed=seed, layer_idx=layer,
        )
        oof = ho.projections
        projections_by_layer[layer] = oof
        by_layer_objects[layer] = ho

        out["per_layer"][str(layer)] = {
            "delta_p_harmful": float(oof["chat_harmful"].mean()
                                     - oof[ah_key].mean()),
            "delta_p_benign": float(oof["chat_benign"].mean()
                                    - oof[ab_key].mean()),
            "interaction": float((oof["chat_harmful"] - oof[ah_key]).mean()
                                 - (oof["chat_benign"] - oof[ab_key]).mean()),
            "auroc_chat": _auroc(oof["chat_harmful"], oof["chat_benign"]),
            "auroc_agent": _auroc(oof[ah_key], oof[ab_key]),
            "auroc_chat_in_sample": _auroc(
                ho.projections_in_sample["chat_harmful"],
                ho.projections_in_sample["chat_benign"],
            ),
            "auroc_agent_in_sample": _auroc(
                ho.projections_in_sample[ah_key],
                ho.projections_in_sample[ab_key],
            ),
            "inter_fold_cosine_mean": ho.inter_fold_cosine_mean,
        }

    print(f"\n{'layer':>7}{'DP_harm':>10}{'DP_benign':>11}{'interact':>10}"
          f"{'AUROC_ch':>10}{'AUROC_ag':>10}{'AUROC_ch_is':>13}{'cos(d_i,d_j)':>14}")
    print("-" * 85)
    for layer in layers:
        r = out["per_layer"][str(layer)]
        print(f"{layer:>7}{r['delta_p_harmful']:>10.4f}{r['delta_p_benign']:>11.4f}"
              f"{r['interaction']:>10.4f}{r['auroc_chat']:>10.4f}"
              f"{r['auroc_agent']:>10.4f}{r['auroc_chat_in_sample']:>13.4f}"
              f"{r['inter_fold_cosine_mean']:>14.3f}")
    print("  (AUROC_ch_is is the in-sample chat AUROC: the same quantity scored")
    print("   on the prompts that defined the direction, for comparison only.)")

    # ---- Detailed comparison at the reported layer ----
    # Reuse the fold split from the sweep above rather than re-running it: the
    # seed makes the two identical today, and reusing keeps them identical if
    # anyone changes how the seed is derived.
    ho_best = by_layer_objects[best_layer]
    detection = held_out_detection(
        ho_best,
        agent_harmful_key=ah_key,
        agent_benign_key=ab_key,
        n_boot=n_bootstrap,
        seed=seed,
    )
    print()
    print(format_heldout_detection_report(detection))

    oof = ho_best.projections
    paired_harmful = run_paired_suite(
        oof["chat_harmful"], oof[ah_key], seed=seed
    )
    paired_benign = run_paired_suite(
        oof["chat_benign"], oof[ab_key], seed=seed
    )
    specificity = specificity_from_projections(
        chat_harmful=oof["chat_harmful"],
        agent_harmful=oof[ah_key],
        chat_benign=oof["chat_benign"],
        agent_benign=oof[ab_key],
        layer_idx=best_layer,
        seed=seed,
        evaluation="held-out",
    )

    print(f"\n  Held-out DeltaP, harmful arm : {paired_harmful.delta_p:+.4f}  "
          f"[{paired_harmful.ci_lower:+.4f}, {paired_harmful.ci_upper:+.4f}]  "
          f"p={paired_harmful.p_value_permutation:.4f}")
    print(f"  Held-out DeltaP, benign arm  : {paired_benign.delta_p:+.4f}  "
          f"[{paired_benign.ci_lower:+.4f}, {paired_benign.ci_upper:+.4f}]  "
          f"p={paired_benign.p_value_permutation:.4f}")
    print()
    print(format_specificity_report(specificity))

    out["best_layer"] = int(best_layer)
    out["detection"] = {
        "auroc_chat_in_sample": detection.auroc_chat_in_sample,
        "auroc_chat_heldout": detection.auroc_chat_heldout,
        "auroc_agent_in_sample": detection.auroc_agent_in_sample,
        "auroc_agent_heldout": detection.auroc_agent_heldout,
        "optimism_chat": detection.optimism_chat,
        "optimism_agent": detection.optimism_agent,
        "auroc_gap_in_sample": detection.auroc_gap_in_sample,
        "auroc_gap_heldout": detection.auroc_gap_heldout,
        "auroc_chat_ci": list(detection.auroc_chat_ci),
        "auroc_agent_ci": list(detection.auroc_agent_ci),
        "auroc_gap_ci": list(detection.auroc_gap_ci),
        "p_value_gap": detection.p_value_gap,
        "gap_excludes_zero": detection.gap_excludes_zero,
        "inter_fold_cosine_mean": detection.inter_fold_cosine_mean,
    }
    out["paired_harmful"] = paired_harmful
    out["paired_benign"] = paired_benign
    out["specificity"] = specificity
    out["agent_harmful_key"] = ah_key
    out["agent_benign_key"] = ab_key
    # Kept so the figures and the format-effect stage can reuse the same
    # out-of-fold scores instead of re-deriving them from a different seed.
    out["projections"] = {k: v for k, v in oof.items()}
    out["projections_by_layer"] = projections_by_layer

    return out


def run_format_effect_analysis(
    dataset: List[Dict],
    activations: Dict[str, Dict[int, np.ndarray]],
    pair_order: Dict[str, List[str]],
    refusal_directions: Dict[int, np.ndarray],
    layers: List[int],
    best_layer: int,
    output_dir: str = "results",
    seed: int = 42,
    heldout: Optional[Dict] = None,
) -> Dict:
    """
    Specificity and scale of the format effect.

    Answers the two questions DeltaP leaves open: whether the shift is specific
    to harmful requests (benign control) and how much of it survives a change
    of units (standardised separation and AUROC). Also answers objective 4 in
    the terms the objective is stated in — detection capability per
    explicitness stratum, measured by AUROC rather than by DeltaP.
    """
    from src.format_effect import (
        format_effect_specificity,
        format_effect_scale,
        detection_by_stratum,
        format_specificity_report,
        format_scale_report,
        format_stratum_report,
        specificity_from_projections,
        scale_from_projections,
        stratum_detection_from_projections,
    )

    print("\n" + "=" * 60)
    print("FORMAT EFFECT: SPECIFICITY AND SCALE")
    print("=" * 60)

    ah_key = "agent_harmful" if "agent_harmful" in activations else "agent_full_harmful"
    ab_key = "agent_benign" if "agent_benign" in activations else "agent_full_benign"

    required = ["chat_harmful", "chat_benign", ah_key, ab_key]
    missing = [k for k in required if k not in activations]
    if missing:
        print(f"  WARNING: missing conditions {missing}. Skipping.")
        return {}

    out: Dict = {"per_layer": {}}

    # --- Per-layer specificity and scale ---
    for layer in layers:
        d = refusal_directions[layer]
        spec = format_effect_specificity(
            activations["chat_harmful"][layer], activations[ah_key][layer],
            activations["chat_benign"][layer], activations[ab_key][layer],
            direction=d, layer_idx=layer, seed=seed,
        )
        scale = format_effect_scale(
            activations["chat_harmful"][layer], activations[ah_key][layer],
            activations["chat_benign"][layer], activations[ab_key][layer],
            direction=d, layer_idx=layer,
        )
        out["per_layer"][str(layer)] = {
            "delta_p_harmful": spec.delta_p_harmful,
            "delta_p_benign": spec.delta_p_benign,
            "interaction": spec.interaction,
            "interaction_p": spec.interaction_p,
            "interaction_cohens_d": spec.interaction_cohens_d,
            "interaction_ci": [spec.interaction_ci_lower, spec.interaction_ci_upper],
            "common_shift_fraction": spec.common_shift_fraction,
            "is_specific": spec.is_specific,
            "interaction_tested_within_pairs": spec.arms_paired,
            "n_pairs_interaction_positive": spec.n_pairs_interaction_positive,
            "shift_correlation": spec.shift_correlation,
            "interaction_p_two_sample": spec.interaction_p_two_sample,
            "interaction_ci_two_sample": [
                spec.interaction_ci_lower_two_sample,
                spec.interaction_ci_upper_two_sample,
            ],
            "separation_chat": scale.separation_chat,
            "separation_agent": scale.separation_agent,
            "separation_loss_raw": scale.separation_loss_raw,
            "cohens_d_chat": scale.cohens_d_chat,
            "cohens_d_agent": scale.cohens_d_agent,
            "separation_loss_standardised": scale.separation_loss_standardised,
            "scale_contraction": scale.scale_contraction,
            "auroc_chat": scale.auroc_chat,
            "auroc_agent": scale.auroc_agent,
        }

    # --- Summary table across layers ---
    print(f"\n{'layer':>7}{'DP_harm':>10}{'DP_benign':>11}{'interact':>10}"
          f"{'p':>8}{'raw loss':>10}{'std loss':>10}{'AUROC_ch':>10}{'AUROC_ag':>10}")
    print("-" * 86)
    for layer in layers:
        r = out["per_layer"][str(layer)]
        print(f"{layer:>7}{r['delta_p_harmful']:>10.4f}{r['delta_p_benign']:>11.4f}"
              f"{r['interaction']:>10.4f}{r['interaction_p']:>8.4f}"
              f"{r['separation_loss_raw']:>9.1%}{r['separation_loss_standardised']:>10.1%}"
              f"{r['auroc_chat']:>10.4f}{r['auroc_agent']:>10.4f}")

    # --- Detailed reports at the selected layer ---
    d = refusal_directions[best_layer]
    spec = format_effect_specificity(
        activations["chat_harmful"][best_layer], activations[ah_key][best_layer],
        activations["chat_benign"][best_layer], activations[ab_key][best_layer],
        direction=d, layer_idx=best_layer, seed=seed,
    )
    scale = format_effect_scale(
        activations["chat_harmful"][best_layer], activations[ah_key][best_layer],
        activations["chat_benign"][best_layer], activations[ab_key][best_layer],
        direction=d, layer_idx=best_layer,
    )
    print()
    print(format_specificity_report(spec))
    print(format_scale_report(scale))

    # --- Objective 4: detection capability per explicitness stratum ---
    by_pair = {str(e.get("pair_id", e.get("id"))): e for e in dataset}
    strata_h = [by_pair[p].get("subtlety", "unknown") for p in pair_order["chat_harmful"]]
    strata_b = [by_pair[p].get("subtlety", "unknown") for p in pair_order["chat_benign"]]

    strat = detection_by_stratum(
        strata_h, strata_b,
        activations["chat_harmful"][best_layer], activations[ah_key][best_layer],
        activations["chat_benign"][best_layer], activations[ab_key][best_layer],
        direction=d,
    )
    print()
    print(format_stratum_report(strat))

    # ---- The same three analyses on held-out projections ----
    # Everything above is measured with a direction fitted on these prompts.
    # The AUROCs in particular are fit statistics, and the per-stratum ones
    # most of all: each stratum holds a handful of prompts, so the optimism is
    # proportionally largest exactly where the objective needs the number to
    # be trustworthy. Re-run on out-of-fold scores, where each prompt was
    # scored by a direction fitted without it.
    out["held_out"] = {}
    ho_proj = (heldout or {}).get("projections_by_layer") or {}
    if ho_proj:
        ah_ho = (heldout or {}).get("agent_harmful_key", ah_key)
        ab_ho = (heldout or {}).get("agent_benign_key", ab_key)
        per_layer_ho = {}
        for layer in layers:
            oof = ho_proj.get(layer)
            if oof is None:
                continue
            spec_ho = specificity_from_projections(
                chat_harmful=oof["chat_harmful"], agent_harmful=oof[ah_ho],
                chat_benign=oof["chat_benign"], agent_benign=oof[ab_ho],
                layer_idx=layer, seed=seed, evaluation="held-out",
            )
            scale_ho = scale_from_projections(
                chat_harmful=oof["chat_harmful"], agent_harmful=oof[ah_ho],
                chat_benign=oof["chat_benign"], agent_benign=oof[ab_ho],
                layer_idx=layer, evaluation="held-out",
            )
            per_layer_ho[str(layer)] = {
                "delta_p_harmful": spec_ho.delta_p_harmful,
                "delta_p_benign": spec_ho.delta_p_benign,
                "interaction": spec_ho.interaction,
                "interaction_p": spec_ho.interaction_p,
                "interaction_ci": [spec_ho.interaction_ci_lower,
                                   spec_ho.interaction_ci_upper],
                "common_shift_fraction": spec_ho.common_shift_fraction,
                "is_specific": spec_ho.is_specific,
                "interaction_tested_within_pairs": spec_ho.arms_paired,
                "n_pairs_interaction_positive":
                    spec_ho.n_pairs_interaction_positive,
                "separation_loss_raw": scale_ho.separation_loss_raw,
                "separation_loss_standardised": scale_ho.separation_loss_standardised,
                "cohens_d_chat": scale_ho.cohens_d_chat,
                "cohens_d_agent": scale_ho.cohens_d_agent,
                "scale_contraction": scale_ho.scale_contraction,
                "auroc_chat": scale_ho.auroc_chat,
                "auroc_agent": scale_ho.auroc_agent,
            }
        out["held_out"]["per_layer"] = per_layer_ho

        oof_best = ho_proj.get(best_layer)
        if oof_best is not None:
            scale_ho_best = scale_from_projections(
                chat_harmful=oof_best["chat_harmful"], agent_harmful=oof_best[ah_ho],
                chat_benign=oof_best["chat_benign"], agent_benign=oof_best[ab_ho],
                layer_idx=best_layer, evaluation="held-out",
            )
            strat_ho = stratum_detection_from_projections(
                strata_h, strata_b,
                chat_harmful=oof_best["chat_harmful"], agent_harmful=oof_best[ah_ho],
                chat_benign=oof_best["chat_benign"], agent_benign=oof_best[ab_ho],
                evaluation="held-out",
            )
            print()
            print(format_scale_report(scale_ho_best))
            print()
            print(format_stratum_report(strat_ho))
            out["held_out"]["by_stratum"] = [
                {
                    "stratum": s.stratum,
                    "n_harmful": s.n_harmful,
                    "n_benign": s.n_benign,
                    "auroc_chat": s.auroc_chat,
                    "auroc_agent": s.auroc_agent,
                    "auroc_loss": s.auroc_loss,
                    "cohens_d_chat": s.cohens_d_chat,
                    "cohens_d_agent": s.cohens_d_agent,
                    "separation_loss_standardised": s.separation_loss_standardised,
                }
                for s in strat_ho
            ]
    else:
        print("\n  NOTE: no held-out projections available, so every AUROC "
              "above is in-sample.")

    out["best_layer"] = int(best_layer)
    out["by_stratum"] = [
        {
            "stratum": s.stratum,
            "n_harmful": s.n_harmful,
            "n_benign": s.n_benign,
            "auroc_chat": s.auroc_chat,
            "auroc_agent": s.auroc_agent,
            "auroc_loss": s.auroc_loss,
            "cohens_d_chat": s.cohens_d_chat,
            "cohens_d_agent": s.cohens_d_agent,
            "separation_loss_standardised": s.separation_loss_standardised,
        }
        for s in strat
    ]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/format_effect_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved to {output_dir}/format_effect_results.json")

    return out


def run_probing_analysis(
    activations: Dict[str, Dict[int, np.ndarray]],
    layers: List[int],
    refusal_directions: Dict[int, np.ndarray],
    n_folds: int = 5,
    n_repeats: int = 1,
    mlp_n_repeats: int = None,
    n_label_permutations: int = 10,
    use_cv: bool = True,
    seed: int = 42,
) -> Dict:
    """
    Run linear and MLP probing classifiers under K-fold cross-validation.

    `refusal_directions` is passed for the legacy single-split path ONLY. Under
    cross-validation the direction is re-estimated inside each fold from that
    fold's training rows: a direction fitted over the full sample has already
    seen every evaluation fold, so cross-validating the classifier alone would
    leave the AUROC inflated while looking validated.

    The reported AUROC is the out-of-fold mean with its across-fold sd. The
    in-sample value is retained beside it to quantify the inflation, and the
    label-permutation control gives the chance level the mean must beat.
    """
    scheme = (f"{n_folds}-fold" if n_repeats <= 1
              else f"{n_repeats}x{n_folds}-fold repeated")
    print("\n" + "=" * 60)
    print("PROBING CLASSIFIERS"
          + (f"  ({scheme} cross-validated)" if use_cv else "  (single split)"))
    print("=" * 60)

    # Combine chat_harmful + chat_benign for probing
    # (test if harmful/benign is linearly separable)
    probe_activations = {}
    for layer in layers:
        probe_activations[layer] = np.concatenate([
            activations["chat_harmful"][layer],
            activations["chat_benign"][layer],
        ])

    n_harmful = len(activations["chat_harmful"][layers[0]])
    n_benign = len(activations["chat_benign"][layers[0]])
    labels = np.concatenate([np.ones(n_harmful), np.zeros(n_benign)])

    # Row i of each arm is the same prompt pair, so a fold must hold out both
    # rows together. Partitioning the 84 rows freely, as this stage did, let a
    # probe train on a prompt's benign twin and be tested on the harmful one.
    groups = pair_groups(n_harmful, n_benign) if use_cv else None

    if use_cv:
        print(f"  n = {n_harmful + n_benign} samples "
              f"({n_harmful} harmful, {n_benign} benign), K = {n_folds}, "
              f"R = {n_repeats}, {n_label_permutations} permutation repeats")
        print("  Per fold: scaler, probe AND difference-in-means direction are "
              "fitted on\n  the training rows only; the held-out rows enter no "
              "fitted object.")
        print("  Folds hold out whole prompt pairs: "
              + ("yes" if groups is not None else
                 "NO (arms differ in size; folding over rows)"))

    # Run probing suite
    linear_suite, mlp_suite = run_probing_suite(
        activations=probe_activations,
        labels=labels,
        refusal_directions=refusal_directions,
        n_folds=n_folds,
        n_repeats=n_repeats,
        mlp_n_repeats=mlp_n_repeats,
        use_cv=use_cv,
        n_label_permutations=n_label_permutations,
        seed=seed,
        groups=groups,
    )

    # Print summary
    print_probing_summary(linear_suite, mlp_suite)

    # Linear vs MLP on the folds both ran. Two AUROCs that look alike are not
    # thereby equivalent; the interval of the DIFFERENCE is what says how much
    # of a gap the data could still be hiding.
    comparison = compare_linear_vs_mlp(linear_suite, mlp_suite) if use_cv else {}
    if comparison:
        print("\n  MLP minus linear, same folds (Nadeau-Bengio corrected):")
        print(f"    {'layer':>7}{'linear':>9}{'MLP':>9}{'diff':>9}"
              f"{'95% CI':>20}{'p':>8}{'folds':>7}")
        for layer in sorted(comparison):
            c = comparison[layer]
            print(f"    {layer:>7}{c['mean_b']:>9.3f}{c['mean_a']:>9.3f}"
                  f"{c['difference']:>+9.3f}"
                  f"   [{c['ci_lower']:>+6.3f}, {c['ci_upper']:>+6.3f}]"
                  f"{c['p_value']:>8.3f}{c['n_paired_folds']:>7d}")
        print("    An interval that contains zero does not show the probes are")
        print("    equal - only that this many folds cannot tell them apart.")

    # Compare probe directions vs refusal directions. Under CV this returns the
    # per-fold cosine (both vectors fitted on the same training partition).
    cosines = compare_probe_vs_direction(linear_suite, refusal_directions)

    def _field(suite, name):
        return {l: getattr(r, name) for l, r in suite.per_layer.items()}

    out = {
        "linear_suite": linear_suite,
        "mlp_suite": mlp_suite,
        "cosines_by_layer": cosines,
        # Headline series consumed by the figures and results.json. Under CV
        # these are the out-of-fold means, not the in-sample values.
        "linear_aurocs": _field(linear_suite, "auroc"),
        "mlp_aurocs": _field(mlp_suite, "auroc"),
        "probe_cross_validated": use_cv,
        "probe_n_folds": n_folds if use_cv else None,
        "probe_n_repeats": n_repeats if use_cv else None,
        "probe_folds_by_pair": groups is not None,
        "probe_comparison": comparison,
    }

    if use_cv:
        out.update({
            # Nadeau-Bengio corrected interval: the uncertainty to report.
            "linear_auroc_ci_lower": _field(linear_suite, "auroc_ci_lower"),
            "linear_auroc_ci_upper": _field(linear_suite, "auroc_ci_upper"),
            "linear_auroc_se_corrected": _field(linear_suite, "auroc_se_corrected"),
            # Partition instability only — shrinks with R, not a CI.
            "linear_auroc_repeat_std": _field(linear_suite, "auroc_repeat_std"),
            "linear_aurocs_std": _field(linear_suite, "auroc_std"),
            "mlp_aurocs_std": _field(mlp_suite, "auroc_std"),
            "linear_aurocs_folds": _field(linear_suite, "auroc_folds"),
            "mlp_aurocs_folds": _field(mlp_suite, "auroc_folds"),
            "linear_aurocs_in_sample": _field(linear_suite, "auroc_in_sample"),
            "mlp_aurocs_in_sample": _field(mlp_suite, "auroc_in_sample"),
            "linear_aurocs_permuted": _field(linear_suite, "auroc_permuted_mean"),
            "linear_aurocs_permuted_std": _field(linear_suite, "auroc_permuted_std"),
            # Difference-in-means direction, refitted per fold and scored on the
            # held-out rows: the fold-honest counterpart of the direction the
            # projection analysis uses.
            "direction_aurocs_oof": _field(linear_suite, "auroc_direction_oof_mean"),
            "direction_aurocs_oof_std": _field(linear_suite, "auroc_direction_oof_std"),
            "cosines_by_layer_std": _field(
                linear_suite, "cosine_with_refusal_oof_std"
            ),
        })

    return out


def run_behavior_and_causal(
    model,
    tokenizer,
    dataset: List[Dict],
    results: Dict,
    intervention_layers: Optional[List[int]] = None,
    doses: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    n_pairs: Optional[int] = None,
    max_new_tokens: int = 64,
    output_dir: str = "results",
    include_history: bool = False,
    agency_level: Optional[str] = None,
    toolbox: int = 0,
    seed: int = 42,
) -> Dict:
    """
    Behavioural baseline, then the causal test of the refusal direction.

    Stage 1 lets the model answer every prompt in the four conditions with
    nothing altered, and records the projection on the same forward pass. It
    yields the refusal rate per condition and whether the projection tracks
    the behaviour.

    Stage 2 takes the agentic prompts the model complied with, adds back the
    projection the format removed (dose 1 = exactly DeltaP at that layer) and
    generates again - with d_chat, with a random direction of the same norm,
    and with d_chat on the benign prompts. See src/causal_test.py for why each
    of those is there.

    Prompts are built by `format_entry`, the same function the extraction
    stage uses, so the activations analysed and the responses generated belong
    to the same token sequences.
    """
    print("\n" + "=" * 60)
    print("BEHAVIOUR AND CAUSAL TEST")
    print("=" * 60)

    refusal_directions = results["refusal_directions"]

    # ---- Which layers ----
    if not intervention_layers:
        chosen = (results.get("robust", {}).get("layer_selection", {})
                  .get("selected_layer", results["best_layer"]))
        intervention_layers = [int(chosen)]
    missing = [l for l in intervention_layers if l not in refusal_directions]
    if missing:
        raise ValueError(
            f"Intervention layers {missing} were not extracted. "
            f"Available: {sorted(refusal_directions)}"
        )
    print(f"  Intervention layer(s): {intervention_layers}")

    # ---- Prompts, row-aligned by pair id exactly as in extraction ----
    canonical = {
        "chat_harmful": ("chat_harmful",),
        "chat_benign": ("chat_benign",),
        "agent_harmful": ("agent_harmful", "agent_full_harmful"),
        "agent_benign": ("agent_benign", "agent_full_benign"),
    }
    entries: Dict[str, List[Dict]] = {}
    for name, aliases in canonical.items():
        rows = sorted((d for d in dataset if d["variant"] in aliases), key=_pair_id)
        if rows:
            entries[name] = rows
    if "agent_harmful" not in entries:
        raise ValueError(
            "No agent-harmful prompts found for the generation stages. "
            f"Available variants: {sorted({d['variant'] for d in dataset})}"
        )
    pair_ids_all = [_pair_id(e) for e in entries["agent_harmful"]]
    for name, rows in entries.items():
        if [_pair_id(e) for e in rows] != pair_ids_all:
            raise ValueError(f"{name} is not row-aligned with agent_harmful.")

    keep = select_evenly(len(pair_ids_all), n_pairs)
    pair_ids = [pair_ids_all[i] for i in keep]
    prompts = {
        name: [format_entry(tokenizer, rows[i], include_history=include_history,
                            agency_level=agency_level, toolbox=toolbox)
               for i in keep]
        for name, rows in entries.items()
    }
    print(f"  Prompt pairs: {len(pair_ids)} of {len(pair_ids_all)}, "
          f"max_new_tokens={max_new_tokens}")

    intervention = ActivationIntervention(model, tokenizer)
    record_all = {l: refusal_directions[l] for l in intervention_layers}
    primary = intervention_layers[0]
    out_path = f"{output_dir}/behavior_generations.jsonl"

    # ---- Stage 1: behavioural baseline ----
    def generate_baseline(prompt, steer):
        gen = intervention.generate_recorded(
            prompt, max_new_tokens, record=record_all)
        proj = gen["projections"]
        return {
            "text": gen["text"],
            "proj_before": proj.get(primary, {}).get("before"),
            "proj_after": proj.get(primary, {}).get("after"),
            "projections_by_layer": {
                str(l): v.get("before") for l, v in proj.items()},
        }

    print("\n  Stage 1: unsteered generation in the four conditions")
    baseline = run_behavioral_baseline(generate_baseline, prompts, pair_ids)
    for r in baseline["records"]:
        r["layer"] = int(primary)
    print()
    print(format_baseline_report(baseline["summary"]))
    all_records = list(baseline["records"])
    write_generations(out_path, all_records)

    # The link between projection and behaviour, at every intervention layer.
    behavior_by_layer = {str(primary): baseline["summary"]}
    for layer in intervention_layers[1:]:
        swapped = [
            dict(r, projection=(r.get("projections_by_layer") or {}).get(str(layer)))
            for r in baseline["records"]
        ]
        behavior_by_layer[str(layer)] = summarise_baseline(swapped)

    # ---- Stage 2: causal test, one layer at a time ----
    causal: Dict[str, Dict] = {}
    gaps = results["gaps_by_layer"]
    for layer in intervention_layers:
        direction = refusal_directions[layer]
        proj_layer = results["projections_by_layer"][layer]
        chat_level = float(np.mean(proj_layer["chat_harmful"]))
        agent_key = ("agent_harmful" if "agent_harmful" in proj_layer
                     else "agent_full_harmful")
        agent_level = float(np.mean(proj_layer[agent_key]))

        dose_unit = float(gaps[layer])
        unit_kind = "DeltaP harmful (chat - agent)"
        if dose_unit <= 0:
            # No projection was lost at this layer, so there is nothing to
            # "restore". Fall back to the chat harmful/benign separation so
            # the stage still runs, and say so.
            dose_unit = float(chat_level - np.mean(proj_layer["chat_benign"]))
            unit_kind = "chat harmful-benign separation (DeltaP <= 0 here)"
            print(f"\n  NOTE layer {layer}: DeltaP is not positive; dose unit "
                  f"falls back to the {unit_kind}.")

        def generate_steered(prompt, steer, _layer=layer, _dir=direction):
            gen = intervention.generate_recorded(
                prompt, max_new_tokens,
                steer={_layer: steer} if steer is not None else None,
                record={_layer: _dir},
            )
            proj = gen["projections"].get(_layer, {})
            return {"text": gen["text"], "proj_before": proj.get("before"),
                    "proj_after": proj.get("after")}

        print(f"\n  Stage 2: causal test at layer {layer} "
              f"(dose unit = {dose_unit:.4f}, {unit_kind})")
        res = run_causal_test(
            generate_steered, baseline["records"], prompts, pair_ids,
            direction=direction, dose_unit=dose_unit, doses=doses, seed=seed,
        )
        for r in res["records"]:
            r["layer"] = int(layer)
        res["summary"].update({
            "layer": int(layer),
            "dose_unit_kind": unit_kind,
            "chat_level_projection": chat_level,
            "agent_level_projection": agent_level,
        })
        print()
        print(format_causal_report(res["summary"], layer))
        causal[str(layer)] = res["summary"]
        all_records += res["records"]
        write_generations(out_path, all_records)

    out = {
        "intervention_layers": [int(l) for l in intervention_layers],
        "n_pairs": len(pair_ids),
        "max_new_tokens": int(max_new_tokens),
        "behavior": baseline["summary"],
        "behavior_by_layer": behavior_by_layer,
        "causal": causal,
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/causal_results.json", "w", encoding="utf-8") as f:
        json.dump(_jsonify(out), f, indent=2)
    print(f"\n  Every generation, with its label: {out_path}")
    print(f"  Summaries: {output_dir}/causal_results.json")

    # Series the intervention figure reads (dose on the x axis).
    first = causal[str(primary)]
    out["intervention_alphas"] = first["doses"]
    out["intervention_rates"] = first["restore_rates"]
    return out


# Keys whose values are hidden_dim-sized vectors. They belong in a binary
# array file, not in a JSON document a human is expected to read.
_ARRAY_KEYS = {"refusal_directions", "best_w_agent", "pca_direction", "direction"}


def _jsonify(value, depth: int = 0):
    """
    Recursively convert a results tree into JSON-serialisable values.

    The previous version only descended ONE level, so nested arrays — every
    entry of `projections_by_layer`, which is layer -> variant -> ndarray —
    reached `json.dump(default=str)` and were written as the *repr* of the
    array: truncated, whitespace-delimited text that has to be re-parsed by
    hand and silently loses precision and any element past the print limit.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonify(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v, depth + 1) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return {
            k: _jsonify(getattr(value, k), depth + 1)
            for k in value.__dataclass_fields__
            if not isinstance(getattr(value, k), np.ndarray)
        }
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def save_results(results: Dict, output_dir: str):
    """Save all results to disk."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Direction vectors go to .npz at full precision; putting 8192 floats per
    # layer into results.json bloated it to megabytes of text and buried the
    # scalars anyone actually reads.
    arrays = {}
    for key in _ARRAY_KEYS:
        val = results.get(key)
        if isinstance(val, np.ndarray):
            arrays[key] = val
        elif isinstance(val, dict):
            for k, v in val.items():
                if isinstance(v, np.ndarray):
                    arrays[f"{key}__{k}"] = v
    if arrays:
        np.savez_compressed(f"{output_dir}/directions.npz", **arrays)
        print(f"\nDirection vectors saved to {output_dir}/directions.npz")

    json_results = {
        key: _jsonify(value)
        for key, value in results.items()
        if key not in _ARRAY_KEYS
    }

    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"Results saved to {output_dir}/results.json")


def run_factorial_analysis(
    activations: Dict[str, Dict[int, np.ndarray]],
    layers: List[int],
    refusal_directions: Dict[int, np.ndarray],
    output_dir: str = "results",
    n_resamples: int = 5000,
    seed: int = 42,
) -> Dict:
    """
    Split the format effect into system message and tool schemas.

    Needs the two derived conditions of --factorial. For each layer and each
    prompt, with p(.) the projection on d_chat:

        system effect  = p(chat) - p(agent system, no tools)
        tools effect   = p(chat) - p(chat system, with tools)
        total effect   = p(chat) - p(agent)
        non-additive   = total - (system + tools)

    Each is a within-prompt difference and is tested as one. The same four
    contrasts are then taken on the harmful-minus-benign double difference,
    which is the part specific to harmful content: it says whether it is the
    system message or the tools that make harmful and benign representations
    converge.

    The direction is the full-sample d_chat. These are descriptive
    decompositions of an effect whose existence is established elsewhere with
    out-of-fold directions; they are not a new significance claim.
    """
    agent_h = "agent_harmful" if "agent_harmful" in activations else "agent_full_harmful"
    agent_b = "agent_benign" if "agent_benign" in activations else "agent_full_benign"
    needed = ["chat_harmful", "chat_benign", "sysonly_harmful", "sysonly_benign",
              "toolsonly_harmful", "toolsonly_benign", agent_h, agent_b]
    missing = [k for k in needed if k not in activations]
    if missing:
        print(f"\n  Factorial analysis skipped: missing {missing} "
              "(run with --factorial).")
        return {}

    print("\n" + "=" * 60)
    print("FACTORIAL DECOMPOSITION  (system message x tool schemas)")
    print("=" * 60)

    def effect(a: np.ndarray, b: np.ndarray) -> Dict:
        t = run_paired_suite(a, b, n_resamples, n_resamples, "two-sided", seed)
        return {
            "delta": t.delta_p,
            "ci_lower": t.ci_lower,
            "ci_upper": t.ci_upper,
            "p_permutation": max(t.p_value_permutation, t.p_value_floor),
            "cohens_d_paired": t.cohens_d_paired,
            "n_positive": t.n_decreasing,
            "n_pairs": t.n_pairs,
        }

    by_layer: Dict[str, Dict] = {}
    for layer in layers:
        d = refusal_directions[layer]
        P = {k: activations[k][layer] @ d for k in needed}
        ch, cb = P["chat_harmful"], P["chat_benign"]

        shifts_h = {"system": ch - P["sysonly_harmful"],
                    "tools": ch - P["toolsonly_harmful"],
                    "total": ch - P[agent_h]}
        shifts_b = {"system": cb - P["sysonly_benign"],
                    "tools": cb - P["toolsonly_benign"],
                    "total": cb - P[agent_b]}
        harmful = {
            "system": effect(ch, P["sysonly_harmful"]),
            "tools": effect(ch, P["toolsonly_harmful"]),
            "total": effect(ch, P[agent_h]),
        }
        harmful["non_additive"] = effect(
            shifts_h["total"], shifts_h["system"] + shifts_h["tools"])
        interaction = {k: effect(shifts_h[k], shifts_b[k]) for k in shifts_h}
        interaction["non_additive"] = effect(
            shifts_h["total"] - shifts_b["total"],
            (shifts_h["system"] - shifts_b["system"])
            + (shifts_h["tools"] - shifts_b["tools"]))
        by_layer[str(layer)] = {"harmful": harmful, "interaction": interaction}

    for block, title in (("harmful", "Harmful arm: projection lost relative to chat"),
                         ("interaction", "Interaction (harmful - benign)")):
        print(f"\n  {title}")
        print(f"    {'layer':>7}{'system':>10}{'tools':>10}{'both':>10}"
              f"{'non-add.':>10}   non-additivity 95% CI")
        for layer in layers:
            e = by_layer[str(layer)][block]
            na = e["non_additive"]
            print(f"    {layer:>7}{e['system']['delta']:>10.3f}"
                  f"{e['tools']['delta']:>10.3f}{e['total']['delta']:>10.3f}"
                  f"{na['delta']:>10.3f}   "
                  f"[{na['ci_lower']:+.3f}, {na['ci_upper']:+.3f}]")
    print("\n    system : agentic system message alone (no tool schemas)")
    print("    tools  : tool schemas alone (chat system message)")
    print("    both   : the full agentic format")
    print("    non-add.: both - (system + tools); far from zero means the two")
    print("              do not simply add up")

    out = {"by_layer": by_layer}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/factorial_results.json", "w", encoding="utf-8") as f:
        json.dump(_jsonify(out), f, indent=2)
    print(f"\n  Saved to {output_dir}/factorial_results.json")
    return out


def run_decomposition_analysis(
    activations: Dict[str, Dict[int, np.ndarray]],
    layers: List[int],
    main_results: Dict,
) -> Dict:
    """
    Run decomposition analysis: break down the total format effect
    into role, tools, and JSON format components.

    Requires all 4 conditions: chat, role_only, role_plus_tools, agent_full.
    """
    print("\n" + "=" * 60)
    print("DECOMPOSITION ANALYSIS")
    print("=" * 60)

    refusal_directions = main_results["refusal_directions"]
    best_layer = main_results["best_layer"]

    # Project all 4 conditions onto refusal direction at best layer
    conditions_order = ["chat_harmful", "role_only_harmful",
                        "role_plus_tools_harmful", "agent_full_harmful"]

    # Verify all conditions exist
    missing = [c for c in conditions_order if c not in activations]
    if missing:
        print(f"  WARNING: Missing conditions: {missing}")
        print(f"  Available: {list(activations.keys())}")
        return {}

    direction = refusal_directions[best_layer]

    from src.refusal_direction import project_onto_direction

    projections = {}
    for cond in conditions_order:
        projections[cond] = project_onto_direction(
            activations[cond][best_layer], direction
        )

    # Decompose effects
    decomp = decompose_format_effects(projections)

    # Print results
    print(f"\n  Layer: {best_layer}")
    print(f"\n  Mean projections:")
    for cond in conditions_order:
        print(f"    {cond:<30} = {projections[cond].mean():.4f} (±{projections[cond].std():.4f})")

    print(f"\n  Effect decomposition:")
    print(f"    {'Effect':<25} {'ΔP':<10} {'p-value':<12} {'Cohen d':<10} {'Sig?'}")
    print(f"    {'-'*67}")
    for effect_name in ["role_effect", "tools_effect", "json_format_effect", "total_effect"]:
        e = decomp[effect_name]
        sig = "✓" if e["significant"] else "✗"
        print(f"    {effect_name:<25} {e['delta_p']:<10.4f} {e['p_value']:<12.6f} "
              f"{e['cohens_d']:<10.3f} {sig}")

    # Verify additivity
    sum_parts = (decomp["role_effect"]["delta_p"] +
                 decomp["tools_effect"]["delta_p"] +
                 decomp["json_format_effect"]["delta_p"])
    total = decomp["total_effect"]["delta_p"]
    print(f"\n  Additivity check:")
    print(f"    Sum of parts: {sum_parts:.4f}")
    print(f"    Total effect: {total:.4f}")
    print(f"    Residual:     {total - sum_parts:.6f}")

    # Per-layer decomposition
    decomp_by_layer = {}
    for layer in layers:
        if layer not in refusal_directions:
            continue
        d = refusal_directions[layer]
        layer_projs = {}
        for cond in conditions_order:
            if cond in activations and layer in activations[cond]:
                layer_projs[cond] = project_onto_direction(
                    activations[cond][layer], d
                )
        if len(layer_projs) == 4:
            decomp_by_layer[layer] = decompose_format_effects(layer_projs)

    return {
        "best_layer": decomp,
        "by_layer": decomp_by_layer,
        "projections_best_layer": {k: v.tolist() for k, v in projections.items()},
    }


def run_stratified_subtlety_analysis(
    dataset: List[Dict],
    activations: Dict[str, Dict[int, np.ndarray]],
    pair_order: Dict[str, List[str]],
    refusal_direction: np.ndarray,
    best_layer: int,
    output_dir: str = "results",
) -> List:
    """
    Run stratified analysis by prompt subtlety level (explicit/contextual/framed).
    """
    from src.stratified_analysis import (
        run_stratified_analysis,
        print_stratified_summary,
        plot_stratified_results,
    )

    # Stratum labels in the row order of the activation arrays, resolved
    # through pair_id rather than assumed from the dataset file's ordering.
    by_pair = {str(e.get("pair_id", e.get("id"))): e for e in dataset}
    strata = [by_pair[p].get("subtlety", "unknown")
              for p in pair_order["chat_harmful"]]

    results = run_stratified_analysis(
        strata=strata,
        activations=activations,
        refusal_direction=refusal_direction,
        layer=best_layer,
    )

    if results:
        print_stratified_summary(results, best_layer)
        try:
            plot_stratified_results(results, best_layer, output_dir=f"{output_dir}/figures")
        except ImportError:
            print("  (stratified figure skipped: matplotlib is not installed)")

        # Save to JSON
        import json
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        save_data = [
            {
                "subtlety": r.subtlety,
                "n_samples": r.n_samples,
                "delta_p": r.delta_p,
                "cohens_d": r.cohens_d,
                "p_value": r.p_value,
                "ci_lower": r.ci_lower,
                "ci_upper": r.ci_upper,
                "significant": r.significant,
            }
            for r in results
        ]
        with open(f"{output_dir}/stratified_results.json", "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"  Saved: {output_dir}/stratified_results.json")

    return results


def run_agent_direction_analysis(
    activations: Dict[str, Dict[int, np.ndarray]],
    layers: List[int],
    refusal_directions: Dict[int, np.ndarray],
    output_dir: str = "results",
    fallback_layer: int = -1,
    n_folds: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Run the agent-specific direction analysis.

    1. Extract w_agent from linear probe on agent activations
    2. Compare cos(w_agent, d_chat) per layer
    3. Evaluate AUROC of w_agent on agent data
    4. Run safety monitor with calibrated threshold
    5. Re-evaluate 3 and 4 out-of-fold at the selected layer

    Steps 3 and 4 are IN-SAMPLE by construction: w_agent and the monitor
    threshold are fitted on the same activations they are then scored on, which
    at n << d guarantees AUROC = 1.0 and F1 = 1.0 whatever the labels say. Step
    5 adds the cross-validated counterparts so the saved JSON never presents
    the in-sample values on their own.
    """
    from src.agent_direction import (
        extract_agent_directions_all_layers,
        calibrate_threshold,
        evaluate_monitor,
        print_agent_direction_summary,
    )

    print("\n" + "=" * 60)
    print("AGENT-SPECIFIC DIRECTION ANALYSIS")
    print("=" * 60)

    # Resolve agent keys
    agent_harmful_key = "agent_harmful" if "agent_harmful" in activations else "agent_full_harmful"
    agent_benign_key = "agent_benign" if "agent_benign" in activations else "agent_full_benign"

    if agent_harmful_key not in activations or agent_benign_key not in activations:
        print("  WARNING: Agent activations not found. Skipping.")
        return {}

    # Extract w_agent for all layers
    layer_results = extract_agent_directions_all_layers(
        agent_harmful_acts={l: activations[agent_harmful_key][l] for l in layers},
        agent_benign_acts={l: activations[agent_benign_key][l] for l in layers},
        d_chat_directions=refusal_directions,
    )

    # Find best layer (highest AUROC with w_agent).
    #
    # With N samples in 8192 dimensions and N <= d+1, Cover's theorem makes the
    # in-sample AUROC 1.0 at EVERY layer, including on randomly permuted labels
    # (validate_corrections.py demonstrates this). The argmax over a column of
    # identical 1.0s returns whichever layer sorts first — that is how layer 1
    # came to be reported as the best layer for w_agent. When the column is
    # saturated the criterion carries no information and must not be used.
    aurocs = {l: r.auroc_agent_w_agent for l, r in layer_results.items()}
    saturated = sum(a >= 0.999 for a in aurocs.values())

    if saturated > 1:
        best_layer = fallback_layer if fallback_layer in layer_results else max(layer_results)
        print(f"\n  WARNING: in-sample AUROC(w_agent) is saturated at 1.0 in "
              f"{saturated}/{len(aurocs)} layers.")
        print( "           This is Cover's theorem at N << d, not evidence, so the")
        print( "           argmax over it is arbitrary. Falling back to the layer")
        print(f"           selected by the projection analysis: {best_layer}.")
        print( "           Use the out-of-fold AUROC in robust_results.json to")
        print( "           choose a layer on evidence.")
    else:
        best_layer = max(aurocs, key=aurocs.get)

    best = layer_results[best_layer]

    # Safety monitor: calibrate threshold on agent data
    proj_harmful = activations[agent_harmful_key][best_layer] @ best.w_agent
    proj_benign = activations[agent_benign_key][best_layer] @ best.w_agent
    threshold = calibrate_threshold(proj_harmful, proj_benign, metric="f1")

    # Evaluate monitor
    monitor_result = evaluate_monitor(
        harmful_acts=activations[agent_harmful_key][best_layer],
        benign_acts=activations[agent_benign_key][best_layer],
        w_agent=best.w_agent,
        threshold=threshold,
    )

    # Print summary
    print_agent_direction_summary(layer_results, monitor_result)

    # --- Out-of-fold counterparts at the selected layer ---
    # Every number above is in-sample. These are the ones a reviewer can use.
    from src.robust_estimation import (
        cross_validated_agent_direction,
        cross_validated_monitor,
        format_direction_report,
        format_monitor_report,
    )

    cv_block: Dict = {}
    try:
        cv_dir = cross_validated_agent_direction(
            activations[agent_harmful_key][best_layer],
            activations[agent_benign_key][best_layer],
            d_chat=refusal_directions.get(best_layer),
            layer_idx=best_layer, n_folds=n_folds, seed=seed,
        )
        cv_mon = cross_validated_monitor(
            activations[agent_harmful_key][best_layer],
            activations[agent_benign_key][best_layer],
            layer_idx=best_layer, n_folds=n_folds, seed=seed,
        )
        print()
        print(format_direction_report(cv_dir))
        print(format_monitor_report(cv_mon))

        cv_block = {
            "n_folds": cv_dir.n_folds,
            "auroc_w_agent_oof": cv_dir.auroc_oof_mean,
            "auroc_w_agent_oof_sd": cv_dir.auroc_oof_std,
            "auroc_w_agent_oof_folds": cv_dir.auroc_oof_folds,
            "cosine_with_d_chat_oof": cv_dir.cosine_with_d_chat_mean,
            "cosine_with_d_chat_oof_sd": cv_dir.cosine_with_d_chat_std,
            "split_half_reliability": cv_dir.split_half_reliability,
            "attenuation_factor": cv_dir.attenuation_factor,
            "monitor_f1_oof": cv_mon.f1_oof,
            "monitor_precision_oof": cv_mon.precision_oof,
            "monitor_recall_oof": cv_mon.recall_oof,
            "monitor_fpr_oof": cv_mon.fpr_oof,
        }
    except Exception as exc:                       # noqa: BLE001
        print(f"  WARNING: out-of-fold evaluation failed ({exc}). "
              "The saved AUROC and F1 are IN-SAMPLE only.")

    # Save results
    import json
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    save_data = {
        "best_layer": int(best_layer),
        # ---- In-sample: retained only to quantify the inflation ----
        "best_auroc_w_agent": best.auroc_agent_w_agent,
        "best_auroc_d_chat": best.auroc_agent_d_chat,
        "best_cosine": best.cosine_similarity,
        "monitor_threshold": threshold,
        "monitor_precision": monitor_result.precision,
        "monitor_recall": monitor_result.recall,
        "monitor_f1": monitor_result.f1,
        "monitor_fpr": monitor_result.false_positive_rate,
        "in_sample_warning": (
            "best_auroc_w_agent, monitor_* and best_cosine are IN-SAMPLE: the "
            "direction and threshold were fitted on the same activations they "
            "are scored on. At n << d these saturate at 1.0 for any labelling "
            "(Cover's theorem). Report cross_validated.* instead."
        ),
        # ---- Out-of-fold: the numbers to report ----
        "cross_validated": cv_block,
        "per_layer": {
            str(l): {
                "cosine": r.cosine_similarity,
                "auroc_w_agent": r.auroc_agent_w_agent,
                "auroc_d_chat": r.auroc_agent_d_chat,
            }
            for l, r in layer_results.items()
        },
    }
    with open(f"{output_dir}/agent_direction_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved to {output_dir}/agent_direction_results.json")

    # Generate visualizations
    try:
        _plot_agent_direction_figures(
            layer_results=layer_results,
            activations=activations,
            best_layer=best_layer,
            agent_harmful_key=agent_harmful_key,
            agent_benign_key=agent_benign_key,
            output_dir=output_dir,
        )
    except ImportError:
        print("  (agent-direction figures skipped: matplotlib is not installed)")

    return {
        "layer_results": layer_results,
        "best_layer": best_layer,
        "best_w_agent": best.w_agent,
        "monitor_threshold": threshold,
        "monitor_result": monitor_result,
    }


def _plot_agent_direction_figures(
    layer_results: Dict,
    activations: Dict,
    best_layer: int,
    agent_harmful_key: str,
    agent_benign_key: str,
    output_dir: str,
):
    """Generate visualizations for agent direction analysis."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path

    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    layers_sorted = sorted(layer_results.keys())

    # --- Figure 1: Cosine similarity d_chat vs w_agent per layer ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    cosines = [layer_results[l].cosine_similarity for l in layers_sorted]
    auroc_w = [layer_results[l].auroc_agent_w_agent for l in layers_sorted]
    auroc_d = [layer_results[l].auroc_agent_d_chat for l in layers_sorted]

    ax.plot(range(len(layers_sorted)), cosines, 'o-', color='#e74c3c', label='cos(w_agent, d_chat)')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xticks(range(len(layers_sorted)))
    ax.set_xticklabels(layers_sorted, rotation=45, fontsize=8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Direction Alignment: w_agent vs d_chat per Layer')
    ax.legend()
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "cosine_w_agent_vs_d_chat.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fig_dir}/cosine_w_agent_vs_d_chat.png")

    # --- Figure 2: AUROC comparison per layer ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(range(len(layers_sorted)), auroc_w, 'o-', color='#2ecc71', label='AUROC(w_agent) on agent')
    ax.plot(range(len(layers_sorted)), auroc_d, 's--', color='#e74c3c', label='AUROC(d_chat) on agent')
    ax.axhline(0.5, color='gray', linewidth=0.5, linestyle='--', label='Chance (0.5)')
    ax.set_xticks(range(len(layers_sorted)))
    ax.set_xticklabels(layers_sorted, rotation=45, fontsize=8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('AUROC')
    ax.set_title('Agent Harmful/Benign Separability: w_agent vs d_chat')
    ax.legend()
    ax.set_ylim(0.3, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "auroc_w_agent_vs_d_chat.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fig_dir}/auroc_w_agent_vs_d_chat.png")

    # --- Figure 3: Projection distributions using w_agent (4 conditions) ---
    best = layer_results[best_layer]
    w = best.w_agent

    conditions = {}
    if "chat_harmful" in activations and best_layer in activations["chat_harmful"]:
        conditions["Chat Harmful"] = activations["chat_harmful"][best_layer] @ w
    if "chat_benign" in activations and best_layer in activations["chat_benign"]:
        conditions["Chat Benign"] = activations["chat_benign"][best_layer] @ w
    if best_layer in activations.get(agent_harmful_key, {}):
        conditions["Agent Harmful"] = activations[agent_harmful_key][best_layer] @ w
    if best_layer in activations.get(agent_benign_key, {}):
        conditions["Agent Benign"] = activations[agent_benign_key][best_layer] @ w

    if conditions:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        colors = ['#d62728', '#2ca02c', '#ff7f0e', '#1f77b4']
        for i, (label, projs) in enumerate(conditions.items()):
            sns.kdeplot(projs, ax=ax, label=label, color=colors[i % 4], fill=True, alpha=0.3)
            ax.axvline(projs.mean(), color=colors[i % 4], linestyle='--', alpha=0.7)

        ax.set_xlabel('Projection onto w_agent')
        ax.set_ylabel('Density')
        ax.set_title(f'Projection Distributions on Agent Direction (Layer {best_layer})')
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "projection_distributions_w_agent.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fig_dir}/projection_distributions_w_agent.png")


def run_agent_intervention(
    model,
    tokenizer,
    dataset: List[Dict],
    w_agent: np.ndarray,
    best_layer: int,
    n_samples: int = 5,
    output_dir: str = "results",
    agency_level: Optional[str] = None,
    toolbox: int = 0,
) -> Dict:
    """
    Run intervention using w_agent instead of d_chat.

    Applies w_agent at the layer where it was extracted,
    NOT at all layers (since w_agent is layer-specific).

    Returns the per-alpha statistics AND writes them to disk. They used to be
    printed and then discarded — the function returned nothing and the caller
    stored nothing — so this experiment survived only as unstructured text in
    whatever the console happened to capture. It is a causal result (does
    re-adding w_agent restore refusal?), which makes it one of the more
    valuable things the pipeline produces, and it belongs in a file.
    """
    from src.intervention import ActivationIntervention, compute_intervention_success_rate

    print("\n" + "=" * 60)
    print(f"INTERVENTION WITH w_agent (Layer {best_layer})")
    print("=" * 60)

    # Get agent harmful prompts
    agent_entries = [d for d in dataset if d.get("variant") in ("agent_harmful", "agent_full_harmful")]
    agent_entries = agent_entries[:n_samples]

    if not agent_entries:
        print("  No agent harmful entries found. Skipping.")
        return {}

    # Same builder as the extraction stage (see format_entry).
    prompts = [
        format_entry(tokenizer, entry, agency_level=agency_level, toolbox=toolbox)
        for entry in agent_entries
    ]

    # Run intervention with w_agent at its specific layer only
    intervention = ActivationIntervention(model, tokenizer)
    alphas = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]

    print(f"\n  Testing {len(prompts)} prompts with alphas: {alphas}")
    print(f"  Direction: w_agent (layer {best_layer})")
    print()

    per_alpha: Dict[str, Dict] = {}
    for alpha in alphas:
        print(f"\n  --- Alpha = {alpha} ---")
        results = intervention.batch_intervene(
            prompts=prompts,
            refusal_direction=w_agent,
            layer_idx=best_layer,
            alpha=alpha,
            max_new_tokens=100,
            apply_all_layers=False,  # Only at the layer where w_agent was trained
        )

        stats = compute_intervention_success_rate(results)
        per_alpha[str(alpha)] = stats
        print(f"  Success rate: {stats['success_rate']:.1%} "
              f"({stats['refused_after_intervention']}/{stats['complied_before_intervention']} "
              f"restored, {stats['already_refusing']} already refusing)")

    print("\n" + "=" * 60)

    out = {
        "direction": "w_agent",
        "layer": int(best_layer),
        "n_prompts": len(prompts),
        "alphas": alphas,
        "per_alpha": per_alpha,
        "success_rates": [per_alpha[str(a)]["success_rate"] for a in alphas],
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/agent_intervention_results.json", "w") as f:
        json.dump(_jsonify(out), f, indent=2)
    print(f"  Saved to {output_dir}/agent_intervention_results.json")

    return out


def run_robust_analysis(
    activations: Dict[str, Dict[int, np.ndarray]],
    layers: List[int],
    output_dir: str = "results",
    n_folds: int = 5,
    seed: int = 42,
    n_selection_splits: int = 200,
) -> Dict:
    """
    Bias-corrected re-analysis of the primary quantities.

    Runs alongside the original estimators rather than replacing them, so the
    magnitude of each bias is reported instead of quietly disappearing. Three
    corrections are applied:

      - DeltaP is re-estimated by cross-fitting, removing the own-sample
        inflation caused by fitting d_chat on the same activations that are
        then projected onto it.
      - w_agent and the monitor threshold are evaluated out-of-fold, so the
        reported AUROC and F1 are not the separability that Cover's theorem
        guarantees at N << d.
      - The reproducibility of w_agent is measured on DISJOINT split-halves,
        which determines whether the rotation angle can be quoted at all.

    See validate_corrections.py for the controlled checks of these estimators
    against known ground truth.
    """
    from src.robust_estimation import (
        cross_fitted_projection_gap,
        cross_validated_agent_direction,
        cross_validated_monitor,
        format_rotation,
        select_layer_and_test,
        format_gap_report,
        format_direction_report,
        format_monitor_report,
    )

    print("\n" + "=" * 60)
    print("BIAS-CORRECTED RE-ANALYSIS")
    print("=" * 60)

    agent_h_key = "agent_harmful" if "agent_harmful" in activations else "agent_full_harmful"
    agent_b_key = "agent_benign" if "agent_benign" in activations else "agent_full_benign"

    required = ["chat_harmful", "chat_benign", agent_h_key, agent_b_key]
    missing = [k for k in required if k not in activations]
    if missing:
        print(f"  WARNING: missing conditions {missing}. Skipping robust analysis.")
        return {}

    # K cannot exceed the number of prompt pairs, nor the smaller class of the
    # harmful/benign contrast. A --limit-pairs dry run drives both to 4 while
    # n_folds defaults to 5, and the estimators raise rather than silently
    # producing a fold with one member. Clamping here keeps the dry run — whose
    # whole purpose is to exercise this code path cheaply — able to reach the
    # end. At the real n = 42 pairs nothing is clamped.
    n_pairs = len(activations["chat_harmful"][layers[0]])
    n_benign = len(activations[agent_b_key][layers[0]])
    usable_folds = max(2, min(n_folds, n_pairs, n_benign))
    if usable_folds != n_folds:
        print(f"  NOTE: K reduced from {n_folds} to {usable_folds} "
              f"({n_pairs} pairs, {n_benign} benign available).")
        print( "        Expected on a --limit-pairs dry run; these statistics "
               "are not meaningful.")
    n_folds = usable_folds

    out: Dict = {"n_folds": n_folds, "per_layer": {}}
    rotation: Dict[str, Dict] = {}

    # --- Pre-committed layer selection on an independent split ---
    try:
        sel = select_layer_and_test(
            chat_harmful={l: activations["chat_harmful"][l] for l in layers},
            chat_benign={l: activations["chat_benign"][l] for l in layers},
            agent_harmful={l: activations[agent_h_key][l] for l in layers},
            n_folds=n_folds,
            seed=seed,
            agent_benign={l: activations[agent_b_key][l] for l in layers},
            n_splits=n_selection_splits,
        )
        print(f"\n  Split-sample layer selection")
        print(f"    layer chosen on the selection half : {sel['selected_layer']}")
        print(f"    n selection / n inference          : "
              f"{sel['n_selection']} / {sel['n_inference']}")
        print(f"    Bonferroni alpha over {sel['n_layers_tested']} layers    : "
              f"{sel['bonferroni_alpha']:.5f}")
        # Everything below is measured on the INFERENCE half alone. These are
        # the figures to quote for the selected layer: the count of pairs that
        # decrease in particular must not be taken from the full-sample table,
        # which includes the pairs that chose the layer.
        inf = sel["result"].crossfit_stats
        print(f"    split over pairs                   : {sel['split_over_pairs']}")
        print(f"    inference half, DeltaP (cross-fit) : {inf.delta_p:+.4f}  "
              f"95% CI [{inf.ci_lower:+.4f}, {inf.ci_upper:+.4f}]")
        print(f"    inference half, pairs decreasing   : "
              f"{inf.n_decreasing}/{inf.n_pairs}  "
              f"(sign test p={inf.sign_test_p:.2e})")
        print(f"    inference half, permutation p      : "
              f"{max(inf.p_value_permutation, inf.p_value_floor):.2e}  "
              f"(Bonferroni alpha {sel['bonferroni_alpha']:.5f})")
        out["layer_selection"] = {
            "selected_layer": sel["selected_layer"],
            "selection_scores": sel["selection_scores"],
            "bonferroni_alpha": sel["bonferroni_alpha"],
            "split_over_pairs": sel["split_over_pairs"],
            "n_selection": sel["n_selection"],
            "n_inference": sel["n_inference"],
            "delta_p_crossfit": sel["result"].delta_p_crossfit,
            "delta_p_naive": sel["result"].delta_p_naive,
            "inference": {
                "delta_p": inf.delta_p,
                "ci_lower": inf.ci_lower,
                "ci_upper": inf.ci_upper,
                "n_decreasing": inf.n_decreasing,
                "n_pairs": inf.n_pairs,
                "sign_test_p": inf.sign_test_p,
                "p_permutation": inf.p_value_permutation,
                "p_value_floor": inf.p_value_floor,
                "cohens_d_paired": inf.cohens_d_paired,
                "significant_bonferroni": bool(
                    max(inf.p_value_permutation, inf.p_value_floor)
                    < sel["bonferroni_alpha"]),
            },
            "stability": sel["stability"],
        }
        inter = sel["interaction"]
        if inter is not None:
            print(f"    inference half, interaction        : {inter.delta_p:+.4f}  "
                  f"95% CI [{inter.ci_lower:+.4f}, {inter.ci_upper:+.4f}]  "
                  f"({inter.n_decreasing}/{inter.n_pairs} pairs positive)")
            out["layer_selection"]["inference_interaction"] = {
                "interaction": inter.delta_p,
                "ci_lower": inter.ci_lower,
                "ci_upper": inter.ci_upper,
                "n_pairs_positive": inter.n_decreasing,
                "n_pairs": inter.n_pairs,
                "p_permutation": inter.p_value_permutation,
                "p_value_floor": inter.p_value_floor,
                "cohens_d_paired": inter.cohens_d_paired,
            }
        stab = sel["stability"]
        if stab:
            freq = ", ".join(f"L{l} {f:.0%}" for l, f in
                             stab["selected_layer_frequency"].items())
            print(f"    over {stab['n_splits']} random pair splits      : {freq}")
            print(f"      inference pairs decreasing        : mean "
                  f"{stab['fraction_decreasing_mean']:.1%}, 5th pct "
                  f"{stab['fraction_decreasing_p05']:.1%}, worst "
                  f"{stab['fraction_decreasing_min']:.1%}")
    except Exception as exc:                       # noqa: BLE001
        print(f"  WARNING: layer selection failed ({exc}).")

    # --- Per-layer corrected estimates ---
    for layer in layers:
        gap = cross_fitted_projection_gap(
            activations["chat_harmful"][layer],
            activations["chat_benign"][layer],
            activations[agent_h_key][layer],
            layer_idx=layer, n_folds=n_folds, seed=seed,
        )
        direction = cross_validated_agent_direction(
            activations[agent_h_key][layer],
            activations[agent_b_key][layer],
            d_chat=None, layer_idx=layer, n_folds=n_folds, seed=seed,
        )
        monitor = cross_validated_monitor(
            activations[agent_h_key][layer],
            activations[agent_b_key][layer],
            layer_idx=layer, n_folds=n_folds, seed=seed,
        )

        print()
        print(format_gap_report(gap))
        print(format_direction_report(direction))
        print(format_monitor_report(monitor))

        out["per_layer"][str(layer)] = {
            "delta_p_naive": gap.delta_p_naive,
            "delta_p_crossfit": gap.delta_p_crossfit,
            "bias_estimate": gap.bias_estimate,
            "cohens_d_paired_crossfit": gap.crossfit_stats.cohens_d_paired,
            "cohens_d_pooled_crossfit": gap.crossfit_stats.cohens_d_pooled,
            "p_permutation_crossfit": gap.crossfit_stats.p_value_permutation,
            "p_value_floor": gap.crossfit_stats.p_value_floor,
            "ci_lower_crossfit": gap.crossfit_stats.ci_lower,
            "ci_upper_crossfit": gap.crossfit_stats.ci_upper,
            "n_decreasing": gap.crossfit_stats.n_decreasing,
            "n_pairs": gap.crossfit_stats.n_pairs,
            "sign_test_p": gap.crossfit_stats.sign_test_p,
            "auroc_w_agent_in_sample": direction.auroc_in_sample,
            "auroc_w_agent_oof": direction.auroc_oof_mean,
            "auroc_w_agent_oof_sd": direction.auroc_oof_std,
            "split_half_reliability": direction.split_half_reliability,
            "attenuation_factor": direction.attenuation_factor,
            "monitor_f1_in_sample": monitor.f1_in_sample,
            "monitor_f1_oof": monitor.f1_oof,
            "monitor_precision_oof": monitor.precision_oof,
            "monitor_recall_oof": monitor.recall_oof,
            "monitor_fpr_oof": monitor.fpr_oof,
        }

        # The agent direction as a difference of centroids - the same
        # estimator as d_chat - evaluated out-of-fold as a detector, and the
        # rotation between formats measured like for like with each method.
        if len(activations[agent_b_key][layer]) == len(activations[agent_h_key][layer]):
            dm_dir = cross_validated_agent_direction(
                activations[agent_h_key][layer], activations[agent_b_key][layer],
                d_chat=None, layer_idx=layer, n_folds=n_folds, seed=seed,
                method="diff_means",
            )
            dm_mon = cross_validated_monitor(
                activations[agent_h_key][layer], activations[agent_b_key][layer],
                layer_idx=layer, n_folds=n_folds, seed=seed, method="diff_means",
            )
            out["per_layer"][str(layer)].update({
                "auroc_agent_diff_means_oof": dm_dir.auroc_oof_mean,
                "auroc_agent_diff_means_oof_sd": dm_dir.auroc_oof_std,
                "monitor_f1_diff_means_oof": dm_mon.f1_oof,
                "monitor_fpr_diff_means_oof": dm_mon.fpr_oof,
            })
            rotation[str(layer)] = {
                method: format_rotation(
                    activations["chat_harmful"][layer],
                    activations["chat_benign"][layer],
                    activations[agent_h_key][layer],
                    activations[agent_b_key][layer],
                    method=method, n_repeats=reps, seed=seed, layer_idx=layer,
                )
                for method, reps in (("diff_means", 50), ("logistic", 20))
            }

    if rotation:
        out["rotation"] = rotation
        print("\n  ROTATION OF THE HARMFUL/BENIGN DIRECTION, chat -> agent")
        print("  Both directions by the same method, fitted on disjoint halves of")
        print("  the pairs; 'norm.' divides by what the method achieves against")
        print("  itself (1 = no rotation, 0 = a right angle).")
        for method in ("diff_means", "logistic"):
            print(f"\n    {method}")
            print(f"    {'layer':>7}{'full':>8}{'across':>9}{'rel chat':>10}"
                  f"{'rel agent':>11}{'norm.':>8}{'angle':>8}"
                  f"{'AUROC oof':>11}")
            for layer in layers:
                r = rotation.get(str(layer), {}).get(method)
                if not r or "cosine_across_formats" not in r:
                    continue
                per = out["per_layer"][str(layer)]
                auroc = (per["auroc_agent_diff_means_oof"] if method == "diff_means"
                         else per["auroc_w_agent_oof"])
                norm = r["cosine_normalised"]
                print(f"    {layer:>7}{r['cosine_full_sample']:>8.3f}"
                      f"{r['cosine_across_formats']:>9.3f}"
                      f"{r['reliability_chat']:>10.3f}{r['reliability_agent']:>11.3f}"
                      + (f"{norm:>8.3f}{r['angle_deg_normalised']:>7.0f}°"
                         if np.isfinite(norm) else f"{'n/a':>8}{'n/a':>8}")
                      + f"{auroc:>11.3f}")
        print("\n    'n/a': the directions are too unreliable at this sample size")
        print("    for an angle to be stated; report agreement, not degrees.")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/robust_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved to {output_dir}/robust_results.json")

    return out


def _limit_pairs(dataset: List[Dict], limit: Optional[int]) -> List[Dict]:
    """
    Subsample the dataset to the first `limit` prompt pairs.

    Selection is by PAIR_ID, never by row. Taking the first N rows of each
    variant would be equivalent only while the file happens to list them in
    matching order; the moment it does not, chat row i and agent row i become
    different prompts and every paired statistic downstream silently compares
    unrelated prompts. Selecting whole pair_ids makes the guarantee structural.

    Sorted rather than random so two dry runs are comparable.
    """
    if not limit or limit <= 0:
        return dataset

    def _pid(entry: Dict) -> str:
        return str(entry.get("pair_id", entry.get("id", "")))

    all_pairs = sorted({_pid(d) for d in dataset})
    if limit >= len(all_pairs):
        return dataset

    keep = set(all_pairs[:limit])
    limited = [d for d in dataset if _pid(d) in keep]

    print(f"\n  DRY RUN: limited to {len(keep)} of {len(all_pairs)} pairs "
          f"({len(limited)} of {len(dataset)} entries).")
    print( "  Statistics from this run are NOT valid — it exercises the "
           "plumbing only.")
    return limited


def _load_dataset_for_experiment(args) -> List[Dict]:
    """
    Load dataset based on --dataset flag.

    - 'custom': Loads dataset_full.jsonl (paired prompts for activation analysis).
    - 'harmagent': Loads HarmAgent test behaviors (for behavioral validation).
    - 'both': Loads custom as primary, HarmAgent for supplementary validation.

    Returns list of dicts compatible with extract_all_activations().
    """
    from collections import Counter

    if args.dataset in ("custom", "both"):
        # Load custom paired dataset
        custom_entries = load_custom_dataset(getattr(args, "dataset_path", None))
        dataset = []
        for entry in custom_entries:
            dataset.append({
                "id": entry.id,
                "base_prompt": entry.prompt,
                "category": entry.category,
                "subtlety": entry.subtlety,
                "is_harmful": entry.is_harmful,
                "tool": entry.tools[0] if entry.tools else None,
                "format": "agent_full" if entry.format == "agent" else entry.format,
                "variant": entry.variant.replace("agent_harmful", "agent_full_harmful")
                           .replace("agent_benign", "agent_full_benign")
                           if args.decompose else entry.variant,
                "pair_id": entry.pair_id,
                "system_prompt": entry.system_prompt,
                "tools": entry.tools,
                "tool_definitions": entry.tool_definitions,
            })

        variants = Counter(d["variant"] for d in dataset)
        print(f"  Source: {getattr(args, 'dataset_path', None) or 'data/dataset_full.jsonl'} (paired)")
        print(f"  Total entries: {len(dataset)}")
        for v, c in sorted(variants.items()):
            print(f"    {v}: {c}")

    elif args.dataset == "harmagent":
        # Load HarmAgent as primary (behavioral validation mode)
        ha_entries = load_harmagent_dataset(split="test", condition="both")
        dataset = []
        for entry in ha_entries:
            fmt = "agent_full" if entry.condition == "agent" else "chat"
            variant_suffix = "harmful" if entry.is_harmful else "benign"
            variant = f"{fmt}_{variant_suffix}" if fmt == "agent_full" else f"chat_{variant_suffix}"

            dataset.append({
                "id": entry.id,
                "base_prompt": entry.prompt,
                "category": entry.category,
                "is_harmful": entry.is_harmful,
                "tool": None,
                "format": fmt,
                "variant": variant,
                "pair_id": entry.id_original,
            })

        variants = Counter(d["variant"] for d in dataset)
        print(f"  Source: HarmAgent test (behavioral)")
        print(f"  Total entries: {len(dataset)}")
        for v, c in sorted(variants.items()):
            print(f"    {v}: {c}")
        print(f"  NOTE: HarmAgent prompts are NOT paired — activation gap analysis is approximate.")

    return dataset


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("=" * 60)
    print("AGENTIC SAFETY PROBE EXPERIMENT")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Quantize: {'4-bit (forzado)' if args.quantize else 'FP16 (forzado)' if args.no_quantize else 'auto'}")
    print(f"Samples per category: {args.n_samples}")
    print(f"Decompose: {args.decompose}")
    print(f"Factorial: {args.factorial}")
    print(f"Agency level: {args.agency_level or 'dataset default'}")
    print(f"Toolbox     : {args.toolbox or 'task tools only'}")
    print(f"Activations : "
          f"{'loaded from ' + args.load_activations if args.load_activations else 'extracted from the model'}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    start_time = time.time()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model = tokenizer = None
    token_stats: Dict = {}

    if args.load_activations:
        # ---- Analysis only: no model, no GPU ----
        activations, pair_order, meta = load_activations(args.load_activations)
        model_info = meta.get("model_info", {})
        saved_layers = [int(l) for l in meta["layers"]]
        if args.layers:
            wanted = [int(l) for l in args.layers.split(",")]
            layers = [l for l in wanted if l in saved_layers]
            if not layers:
                raise SystemExit(
                    f"None of --layers {wanted} is in the saved activations "
                    f"({saved_layers}).")
        else:
            layers = saved_layers
        token_stats = meta.get("token_stats", {})
        args.model = meta.get("model", args.model)
        print(f"\nLoaded activations of {args.model}: "
              f"{len(activations)} variants, layers {layers}")

        # The corpus supplies what the arrays do not carry (explicitness
        # stratum, category). It must be the one the activations came from.
        if not args.dataset_path and meta.get("dataset_path"):
            args.dataset_path = meta["dataset_path"]
        args.decompose = bool(meta.get("decompose", args.decompose))
        if args.limit_pairs:
            print("  NOTE: --limit-pairs is ignored with --load-activations; "
                  "the saved arrays decide which pairs are analysed.")
        dataset = _load_dataset_for_experiment(args)
        saved_ids = set(next(iter(pair_order.values())))
        dataset = [d for d in dataset if _pair_id(d) in saved_ids]
        found = {_pair_id(d) for d in dataset}
        if found != saved_ids:
            raise SystemExit(
                f"{len(saved_ids - found)} pair ids in the saved activations are "
                "not in the dataset. Pass the --dataset-path the activations "
                "were extracted from.")
    else:
        if _MODEL_STACK_ERROR is not None:
            raise SystemExit(
                "torch / transformers are required to extract activations "
                f"({_MODEL_STACK_ERROR}). To re-analyse a finished run without "
                "them, pass --load-activations <run>/activations.npz.")

        # 1. Load model
        if args.quantize and args.no_quantize:
            raise SystemExit("--quantize y --no-quantize son incompatibles")
        quantize_flag = True if args.quantize else (False if args.no_quantize else None)
        model, tokenizer = load_model_and_tokenizer(
            model_name=args.model,
            quantize=quantize_flag,
        )
        model_info = get_model_info(model)

        # Determine layers to analyze
        n_layers = model_info["num_layers"]
        if args.layers:
            layers = [int(l) for l in args.layers.split(",")]
        else:
            # Sample layers across the network (every 2nd layer for efficiency)
            layers = list(range(0, n_layers, 2))

        print(f"\nAnalyzing layers: {layers}")

        # 2. Load dataset
        print("\n" + "=" * 60)
        print("LOADING DATASET")
        print("=" * 60)

        dataset = _load_dataset_for_experiment(args)
        dataset = _limit_pairs(dataset, args.limit_pairs)

        # 3. Extract activations
        extractor = ActivationExtractor(model, tokenizer)
        activations, pair_order = extract_all_activations(
            extractor=extractor,
            tokenizer=tokenizer,
            dataset=dataset,
            layers=layers,
            include_history=args.include_history,
            agency_level=args.agency_level,
            toolbox=args.toolbox,
            decompose=args.decompose,
            factorial=args.factorial,
        )
        token_stats = getattr(extractor, "token_stats", {})
        # The extractor leaves its hooks registered for reuse. Left in place
        # they would copy every layer's hidden state to host memory on each
        # decoding step of the generation stages.
        extractor.close()

        # Saved BEFORE any analysis runs: a failure in the statistics must
        # never cost a second pass through the model.
        if args.save_activations:
            save_activations(
                f"{args.output_dir}/activations.npz", activations, pair_order,
                meta={
                    "model": args.model,
                    "model_info": model_info,
                    "layers": layers,
                    "dataset_path": args.dataset_path,
                    "decompose": bool(args.decompose),
                    "factorial": bool(args.factorial),
                    "token_stats": token_stats,
                    "seed": args.seed,
                },
            )

    # 4. Main analysis (refusal direction + stats + benign control + held-out)
    results = run_main_analysis(
        activations, layers, n_bootstrap=args.n_bootstrap, seed=args.seed,
        n_folds=args.n_folds, use_heldout=not args.skip_heldout,
    )
    results["model_info"] = model_info
    results["layers_analyzed"] = layers
    results["decompose"] = args.decompose
    results["prompt_tokens"] = token_stats
    results["config"] = {k: v for k, v in vars(args).items()}

    # 4b. Decomposition analysis (if enabled)
    if args.decompose:
        decomp_results = run_decomposition_analysis(activations, layers, results)
        results["decomposition"] = decomp_results

    # 4c. System message x tool schemas (if the derived conditions exist)
    if "sysonly_harmful" in activations:
        results["factorial"] = run_factorial_analysis(
            activations, layers, results["refusal_directions"],
            output_dir=args.output_dir, seed=args.seed,
        )

    # 5. Probing classifiers
    if not args.skip_probes:
        probe_results = run_probing_analysis(
            activations, layers, results["refusal_directions"],
            n_folds=args.n_folds,
            n_repeats=args.n_repeats,
            mlp_n_repeats=args.mlp_n_repeats,
            n_label_permutations=args.n_label_permutations,
            use_cv=not args.no_probe_cv,
            seed=args.seed,
        )
        results.update(probe_results)

    # 5a-bis. Specificity (benign control) and scale-invariant re-expression.
    # Runs before the stratified stage because its verdict conditions how the
    # DeltaP figures produced downstream may be interpreted.
    results["format_effect"] = run_format_effect_analysis(
        dataset=dataset,
        activations=activations,
        pair_order=pair_order,
        refusal_directions=results["refusal_directions"],
        layers=layers,
        best_layer=results["best_layer"],
        output_dir=args.output_dir,
        seed=args.seed,
        heldout=results.get("heldout"),
    )

    # 5b. Stratified analysis by subtlety
    stratified_results = run_stratified_subtlety_analysis(
        dataset=dataset,
        activations=activations,
        pair_order=pair_order,
        refusal_direction=results["refusal_directions"][results["best_layer"]],
        best_layer=results["best_layer"],
        output_dir=args.output_dir,
    )
    results["stratified"] = stratified_results

    # 6. Bias-corrected re-analysis (cross-fitting + out-of-fold evaluation)
    if not args.skip_robust:
        results["robust"] = run_robust_analysis(
            activations=activations,
            layers=layers,
            output_dir=args.output_dir,
            n_folds=args.n_folds,
            seed=args.seed,
            n_selection_splits=args.n_selection_splits,
        )

    # 7. Agent-specific direction analysis
    agent_dir_results = run_agent_direction_analysis(
        activations=activations,
        layers=layers,
        refusal_directions=results["refusal_directions"],
        output_dir=args.output_dir,
        fallback_layer=results["best_layer"],
        n_folds=args.n_folds,
        seed=args.seed,
    )
    results["agent_direction"] = agent_dir_results

    # Checkpoint. Everything above is analysis of activations already in hand;
    # what follows generates text and is the slow, failure-prone part. Writing
    # the results here means a crash during generation loses only generation.
    save_results(results, args.output_dir)

    # 8. Generation stages: behavioural baseline + causal test. They run LAST
    # because they need the layer the analysis selects, and because they are
    # the only stages that can fail after the expensive extraction.
    if model is not None and not args.skip_intervention:
        ilayers = ([int(l) for l in args.intervention_layers.split(",")]
                   if args.intervention_layers else None)
        doses = tuple(float(x) for x in args.intervention_doses.split(","))
        results.update(run_behavior_and_causal(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            results=results,
            intervention_layers=ilayers,
            doses=doses,
            n_pairs=args.intervention_pairs,
            max_new_tokens=args.max_new_tokens,
            output_dir=args.output_dir,
            include_history=args.include_history,
            agency_level=args.agency_level,
            toolbox=args.toolbox,
            seed=args.seed,
        ))

        # 8b. w_agent at its own layer, fixed alpha grid (opt-in)
        if args.agent_intervention and agent_dir_results.get("best_w_agent") is not None:
            results["agent_intervention"] = run_agent_intervention(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                w_agent=agent_dir_results["best_w_agent"],
                best_layer=agent_dir_results["best_layer"],
                n_samples=min(5, args.n_samples),
                output_dir=args.output_dir,
                agency_level=args.agency_level,
                toolbox=args.toolbox,
            )
    elif model is None:
        print("\n  Generation stages skipped: no model loaded "
              "(--load-activations).")

    # 9. Save results. Before the figures, not after: a plotting error at the
    # very end must not be able to cost the numbers.
    save_results(results, args.output_dir)

    # 10. Generate visualizations
    if generate_all_figures is None:
        print("\n  Figures skipped: matplotlib is not installed.")
    else:
        try:
            generate_all_figures(results, output_dir=f"{args.output_dir}/figures")
        except Exception as exc:                       # noqa: BLE001
            print(f"\n  WARNING: figure generation failed ({exc!r}). The results "
                  "are already on disk; the figures can be redrawn from them.")

    # Final summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Total time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Model: {args.model}")
    print(f"Best layer: {results['best_layer']}")
    boot = results["bootstrap"]
    print(f"ΔP: {boot['mean_delta_p']:.4f}  "
          f"95% CI [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}] "
          f"({boot['n_bootstrap']} resamples, excludes zero: {boot['excludes_zero']})")
    print(f"Significant: {results['ttest'].significant}")
    if "specificity" in results:
        spec = results["specificity"]
        print(f"Benign control: DeltaP_benign {spec.delta_p_benign:+.4f}, "
              f"interaction {spec.interaction:+.4f} (p={spec.interaction_p:.4f}, "
              f"specific: {spec.is_specific})")
    if results.get("heldout", {}).get("detection"):
        det = results["heldout"]["detection"]
        print(f"AUROC held-out: chat {det['auroc_chat_heldout']:.3f}, "
              f"agent {det['auroc_agent_heldout']:.3f}  "
              f"(in-sample {det['auroc_chat_in_sample']:.3f} / "
              f"{det['auroc_agent_in_sample']:.3f})")
    if results.get("probe_cross_validated"):
        lin = results["linear_suite"]
        best = lin.per_layer[lin.best_layer]
        perm = best.auroc_permuted_mean
        perm_str = (f", shuffled-label control {perm:.3f}"
                    if perm is not None and np.isfinite(perm) else "")
        scheme = (f"{lin.n_folds}-fold" if (lin.n_repeats or 1) <= 1
                  else f"{lin.n_repeats}x{lin.n_folds}-fold")
        ci_str = ""
        if best.auroc_ci_lower is not None and np.isfinite(best.auroc_ci_lower):
            ci_str = (f"  95% CI [{best.auroc_ci_lower:.3f}, "
                      f"{best.auroc_ci_upper:.3f}]")
        print(f"Probe AUROC ({scheme} CV, layer {lin.best_layer}): "
              f"{best.auroc:.3f} +- {best.auroc_std:.3f}{ci_str}  "
              f"(in-sample {best.auroc_in_sample:.3f}{perm_str})")
    sel = results.get("robust", {}).get("layer_selection", {})
    if sel.get("inference"):
        inf = sel["inference"]
        print(f"Split-sample layer {sel['selected_layer']}: "
              f"{inf['n_decreasing']}/{inf['n_pairs']} inference pairs decrease, "
              f"DeltaP {inf['delta_p']:+.4f} "
              f"[{inf['ci_lower']:+.4f}, {inf['ci_upper']:+.4f}]")
    for layer, c in (results.get("causal") or {}).items():
        doses = ", ".join(
            f"x{d:g}: {r:.0%}" for d, r in zip(c["doses"], c["restore_rates"]))
        ctrl = ", ".join(f"{r:.0%}" for r in c["control_rates"])
        print(f"Causal test, layer {layer}: refusal restored at dose {doses}  "
              f"(random direction: {ctrl})")
    print(f"Results: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
