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
    get_model_info,
    TOOL_DEFINITIONS,
)
from src.activation_extractor import ActivationExtractor
from src.refusal_direction import (
    compute_refusal_direction,
    compute_refusal_direction_pca,
    project_activations,
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
)
from src.intervention import (
    ActivationIntervention,
    compute_intervention_success_rate,
)
from src.visualizations import generate_all_figures
from data.build_dataset import build_dataset, HARMFUL_PROMPTS, BENIGN_PROMPTS


def parse_args():
    parser = argparse.ArgumentParser(description="Run the agentic safety probe experiment")
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="HuggingFace model identifier",
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
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Extract activations for all dataset entries.
    
    Returns:
        Dict with keys 'chat_harmful', 'agent_harmful', 'chat_benign', 'agent_benign'
        Each value is a dict mapping layer_idx -> (n_samples, hidden_dim)
    """
    print("\n" + "=" * 60)
    print("EXTRACTING ACTIVATIONS")
    print("=" * 60)

    # Group by variant
    variants = {
        "chat_harmful": [d for d in dataset if d["variant"] == "chat_harmful"],
        "agent_harmful": [d for d in dataset if d["variant"] == "agent_harmful"],
        "chat_benign": [d for d in dataset if d["variant"] == "chat_benign"],
        "agent_benign": [d for d in dataset if d["variant"] == "agent_benign"],
    }

    all_activations = {}

    for variant_name, entries in variants.items():
        print(f"\n  Processing {variant_name} ({len(entries)} prompts)...")
        prompts = []

        for entry in entries:
            if entry["format"] == "chat":
                prompt = format_chat_prompt(tokenizer, entry["base_prompt"])
            elif entry["format"] == "agent":
                if include_history:
                    prompt = format_agent_prompt_with_history(
                        tokenizer, entry["base_prompt"]
                    )
                else:
                    prompt = format_agent_prompt(tokenizer, entry["base_prompt"])
            prompts.append(prompt)

        # Extract activations for all prompts in this variant
        variant_activations = {layer: [] for layer in layers}

        for i, prompt in enumerate(prompts):
            if (i + 1) % 10 == 0:
                print(f"    {i + 1}/{len(prompts)}")

            acts = extractor.extract(prompt, layers=layers)
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
        for variant in ["chat_harmful", "agent_harmful", "chat_benign", "agent_benign"]:
            proj = project_activations(
                activations[variant][layer],
                refusal_directions[layer],
            )
            projections[layer][variant] = proj

    results["projections_by_layer"] = projections

    # Compute gap at each layer
    gaps = compute_gap_by_layer(
        chat_harmful={l: activations["chat_harmful"][l] for l in layers},
        agent_harmful={l: activations["agent_harmful"][l] for l in layers},
        refusal_directions=refusal_directions,
    )
    results["gaps_by_layer"] = gaps
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
    proj_agent = projections[best_layer]["agent_harmful"]

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


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("=" * 60)
    print("AGENTIC SAFETY PROBE EXPERIMENT")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Quantize: {not args.no_quantize}")
    print(f"Samples per category: {args.n_samples}")
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

    # 2. Build dataset
    print("\n" + "=" * 60)
    print("BUILDING DATASET")
    print("=" * 60)
    dataset = build_dataset(
        output_path=f"{args.output_dir}/dataset.jsonl",
        n_harmful_per_category=args.n_samples,
        n_benign_per_category=args.n_samples,
    )

    # 3. Extract activations
    extractor = ActivationExtractor(model, tokenizer)
    activations = extract_all_activations(
        extractor=extractor,
        tokenizer=tokenizer,
        dataset=dataset,
        layers=layers,
        include_history=args.include_history,
    )

    # 4. Main analysis (refusal direction + stats)
    results = run_main_analysis(activations, layers)
    results["model_info"] = model_info
    results["layers_analyzed"] = layers

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
