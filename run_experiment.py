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
import time
import numpy as np
from pathlib import Path
from typing import Dict, List

from src.model_loader import (
    load_model_and_tokenizer,
    format_chat_prompt,
    format_agent_prompt,
    format_agent_prompt_with_history,
    format_role_only_prompt,
    format_role_plus_tools_prompt,
    get_model_info,
    TOOL_DEFINITIONS,
)
from src.activation_extractor import ActivationExtractor
from src.refusal_direction import (
    compute_refusal_direction,
    compute_refusal_direction_pca,
    project_onto_direction,
    compute_gap_by_layer,
)
from src.probes import run_probing_suite, compare_probe_vs_direction, print_probing_summary
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
from src.intervention import (
    ActivationIntervention,
    compute_intervention_success_rate,
)
from src.visualizations import generate_all_figures
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
        "--no-quantize",
        action="store_true",
        help="Disable 4-bit quantization (requires more VRAM)",
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
        help="Number of samples per category",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def extract_all_activations(
    extractor: ActivationExtractor,
    tokenizer,
    dataset: List[Dict],
    layers: List[int],
    include_history: bool = False,
    decompose: bool = False,
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Extract activations for all dataset entries.
    
    Returns:
        Dict with keys like 'chat_harmful', 'agent_harmful', etc.
        In decompose mode also includes 'role_only_harmful', 'role_plus_tools_harmful', etc.
        Each value is a dict mapping layer_idx -> (n_samples, hidden_dim)
    """
    print("\n" + "=" * 60)
    print("EXTRACTING ACTIVATIONS")
    print("=" * 60)

    # Determine variants present in dataset
    variant_names = sorted(set(d["variant"] for d in dataset))
    variants = {v: [d for d in dataset if d["variant"] == v] for v in variant_names}

    all_activations = {}

    for variant_name, entries in variants.items():
        print(f"\n  Processing {variant_name} ({len(entries)} prompts)...")
        prompts = []

        for entry in entries:
            fmt = entry["format"]
            if fmt == "chat":
                prompt = format_chat_prompt(tokenizer, entry["base_prompt"])
            elif fmt == "role_only":
                prompt = format_role_only_prompt(tokenizer, entry["base_prompt"])
            elif fmt == "role_plus_tools":
                prompt = format_role_plus_tools_prompt(tokenizer, entry["base_prompt"])
            elif fmt in ("agent_full", "agent"):
                # Use entry-specific system_prompt and tools if available
                # This avoids injecting ALL 26 tools (which causes truncation)
                sys_prompt = entry.get("system_prompt") or None
                if include_history:
                    prompt = format_agent_prompt_with_history(
                        tokenizer, entry["base_prompt"]
                    )
                else:
                    prompt = format_agent_prompt(
                        tokenizer, entry["base_prompt"],
                        system_message=sys_prompt,
                    )
            else:
                prompt = format_chat_prompt(tokenizer, entry["base_prompt"])
            prompts.append(prompt)

        # Extract activations for all prompts in this variant
        variant_activations = {layer: [] for layer in layers}

        for i, prompt in enumerate(prompts):
            if (i + 1) % 10 == 0:
                print(f"    {i + 1}/{len(prompts)}")

            acts = extractor.extract_single(prompt, layers=layers)
            for layer in layers:
                variant_activations[layer].append(acts[layer])

        # Stack into arrays
        all_activations[variant_name] = {
            layer: np.stack(variant_activations[layer])
            for layer in layers
        }
        print(f"    Shape: {all_activations[variant_name][layers[0]].shape}")

    return all_activations


def run_main_analysis(
    activations: Dict[str, Dict[int, np.ndarray]],
    layers: List[int],
) -> Dict:
    """
    Run the main analysis pipeline:
    - Compute refusal direction
    - Project activations
    - Compute gaps
    - Statistical tests
    """
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
    )
    # Extract gaps from layer results
    gaps = {l: r["gap_analysis"]["delta_p"] for l, r in layer_results.items()}
    results["gaps_by_layer"] = gaps
    results["layer_results"] = layer_results
    print(f"  Gaps computed. Max gap at layer {max(gaps, key=gaps.get)}: {max(gaps.values()):.4f}")

    # Find best layer (largest gap)
    best_layer = max(gaps, key=gaps.get)
    results["best_layer"] = best_layer
    print(f"\n  Best layer for analysis: {best_layer}")

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

    # Bootstrap CI
    bootstrap = bootstrap_ci(proj_chat, proj_agent)
    results["bootstrap"] = bootstrap
    print(f"  Bootstrap ΔP: {bootstrap['mean_delta_p']:.4f} [{bootstrap['ci_lower']:.4f}, {bootstrap['ci_upper']:.4f}]")

    # AUROC by format
    auroc = compute_auroc_by_format(projections[best_layer])
    results["auroc"] = auroc
    print(f"  AUROC chat: {auroc['auroc_chat']:.4f}")
    print(f"  AUROC agent: {auroc['auroc_agent']:.4f}")

    # Print full report
    print_full_report(ttest_result, perm_result, bootstrap, auroc)

    # Store projections for visualization
    results["projections"] = projections[best_layer]

    return results


def run_probing_analysis(
    activations: Dict[str, Dict[int, np.ndarray]],
    layers: List[int],
    refusal_directions: Dict[int, np.ndarray],
) -> Dict:
    """Run linear and MLP probing classifiers."""
    print("\n" + "=" * 60)
    print("PROBING CLASSIFIERS")
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

    # Run probing suite
    linear_suite, mlp_suite = run_probing_suite(
        activations=probe_activations,
        labels=labels,
        refusal_directions=refusal_directions,
    )

    # Print summary
    print_probing_summary(linear_suite, mlp_suite)

    # Compare probe directions vs refusal directions
    cosines = compare_probe_vs_direction(linear_suite, refusal_directions)

    return {
        "linear_suite": linear_suite,
        "mlp_suite": mlp_suite,
        "cosines_by_layer": cosines,
        "linear_aurocs": {l: r.auroc for l, r in linear_suite.per_layer.items()},
        "mlp_aurocs": {l: r.auroc for l, r in mlp_suite.per_layer.items()},
    }


def run_intervention_experiments(
    model,
    tokenizer,
    dataset: List[Dict],
    refusal_directions: Dict[int, np.ndarray],
    best_layer: int,
    n_samples: int = 10,
) -> Dict:
    """Run inference-time intervention experiments."""
    print("\n" + "=" * 60)
    print("INTERVENTION EXPERIMENTS")
    print("=" * 60)

    intervention = ActivationIntervention(model, tokenizer)

    # Select harmful agent prompts
    agent_harmful = [d for d in dataset if d["variant"] == "agent_harmful"][:n_samples]

    # Test different alphas
    alphas = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
    alpha_results = {}

    for alpha in alphas:
        print(f"\n  Testing α={alpha}...")
        prompts = [
            format_agent_prompt(tokenizer, d["base_prompt"])
            for d in agent_harmful
        ]

        results = intervention.batch_intervene(
            prompts=prompts,
            refusal_direction=refusal_directions[best_layer],
            layer_idx=best_layer,
            alpha=alpha,
        )

        metrics = compute_intervention_success_rate(results)
        alpha_results[alpha] = metrics
        print(f"    Success rate: {metrics['success_rate']:.2%}")

    # Find minimum effective alpha
    intervention_rates = [alpha_results[a]["success_rate"] for a in alphas]

    return {
        "alpha_results": alpha_results,
        "intervention_alphas": alphas,
        "intervention_rates": intervention_rates,
    }


def save_results(results: Dict, output_dir: str):
    """Save all results to disk."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save numerical results as JSON
    json_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            json_results[key] = value.tolist()
        elif isinstance(value, dict):
            json_results[key] = {
                str(k): v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in value.items()
            }
        elif hasattr(value, "__dataclass_fields__"):
            json_results[key] = {
                k: getattr(value, k)
                for k in value.__dataclass_fields__
                if not isinstance(getattr(value, k), np.ndarray)
            }
        else:
            try:
                json.dumps(value)
                json_results[key] = value
            except (TypeError, ValueError):
                json_results[key] = str(value)

    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(json_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}/results.json")


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
        custom_entries = load_custom_dataset()
        dataset = []
        for entry in custom_entries:
            dataset.append({
                "id": entry.id,
                "base_prompt": entry.prompt,
                "category": entry.category,
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
        print(f"  Source: dataset_full.jsonl (paired)")
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
    print(f"Quantize: {not args.no_quantize}")
    print(f"Samples per category: {args.n_samples}")
    print(f"Decompose: {args.decompose}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    start_time = time.time()

    # 1. Load model
    quantize_flag = None if not args.no_quantize else False  # None = auto-detect
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

    # 3. Extract activations
    extractor = ActivationExtractor(model, tokenizer)
    activations = extract_all_activations(
        extractor=extractor,
        tokenizer=tokenizer,
        dataset=dataset,
        layers=layers,
        include_history=args.include_history,
        decompose=args.decompose,
    )

    # 4. Main analysis (refusal direction + stats)
    results = run_main_analysis(activations, layers)
    results["model_info"] = model_info
    results["layers_analyzed"] = layers
    results["decompose"] = args.decompose

    # 4b. Decomposition analysis (if enabled)
    if args.decompose:
        decomp_results = run_decomposition_analysis(activations, layers, results)
        results["decomposition"] = decomp_results

    # 5. Probing classifiers
    if not args.skip_probes:
        probe_results = run_probing_analysis(
            activations, layers, results["refusal_directions"]
        )
        results.update(probe_results)

    # 6. Intervention experiments
    if not args.skip_intervention:
        intervention_results = run_intervention_experiments(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            refusal_directions=results["refusal_directions"],
            best_layer=results["best_layer"],
            n_samples=min(10, args.n_samples),
        )
        results.update(intervention_results)

    # 7. Generate visualizations
    generate_all_figures(results, output_dir=f"{args.output_dir}/figures")

    # 8. Save results
    save_results(results, args.output_dir)

    # Final summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Total time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Model: {args.model}")
    print(f"Best layer: {results['best_layer']}")
    print(f"ΔP: {results['bootstrap']['mean_delta_p']:.4f}")
    print(f"Significant: {results['ttest'].significant}")
    print(f"Results: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
