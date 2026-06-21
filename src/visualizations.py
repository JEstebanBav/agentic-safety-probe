"""
Visualization module.

Generates all figures for the paper and analysis:
- Projection distributions (violin/ridge plots)
- Layer-wise AUROC curves
- PCA/UMAP of activation space
- Intervention dose-response curves
- Cross-tool consistency heatmaps
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# Set style
plt.style.use("seaborn-v0_8-paper")
sns.set_palette("colorblind")

COLORS = {
    "chat_harmful": "#d62728",    # red
    "agent_harmful": "#ff7f0e",   # orange
    "chat_benign": "#2ca02c",     # green
    "agent_benign": "#1f77b4",    # blue
}

LABELS = {
    "chat_harmful": "Chat + Harmful",
    "agent_harmful": "Agent + Harmful",
    "chat_benign": "Chat + Benign",
    "agent_benign": "Agent + Benign",
}


def save_fig(fig, name: str, output_dir: str = "results/figures"):
    """Save figure in multiple formats."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output_dir}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir}/{name}.png/.pdf")


def plot_projection_distributions(
    projections: Dict[str, np.ndarray],
    layer_idx: int,
    title: Optional[str] = None,
    output_dir: str = "results/figures",
):
    """
    Plot violin/box plots comparing projections across conditions.
    
    This is the MAIN FIGURE showing the projection gap.
    
    Args:
        projections: Dict with keys like 'chat_harmful', 'agent_harmful', etc.
        layer_idx: Layer index for labeling.
        title: Optional custom title.
        output_dir: Where to save.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    conditions = ["chat_harmful", "agent_harmful", "chat_benign", "agent_benign"]
    data = []
    labels = []

    for cond in conditions:
        if cond in projections:
            data.append(projections[cond])
            labels.append(LABELS[cond])

    # Violin plot
    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(COLORS[conditions[i]])
        pc.set_alpha(0.7)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Projection onto Refusal Direction", fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f"Projection Distributions (Layer {layer_idx})", fontsize=14)

    # Add gap annotation
    if "chat_harmful" in projections and "agent_harmful" in projections:
        gap = projections["chat_harmful"].mean() - projections["agent_harmful"].mean()
        ax.annotate(
            f"ΔP = {gap:.3f}",
            xy=(1.5, max(projections["chat_harmful"].mean(), projections["agent_harmful"].mean())),
            fontsize=12,
            fontweight="bold",
            ha="center",
            color="black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5),
        )

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    save_fig(fig, f"projection_distributions_layer{layer_idx}", output_dir)


def plot_layer_auroc_curve(
    linear_aurocs: Dict[int, float],
    mlp_aurocs: Dict[int, float],
    output_dir: str = "results/figures",
):
    """
    Plot AUROC as a function of layer depth.
    
    Shows where in the network harmful/benign separation emerges.
    
    Args:
        linear_aurocs: Dict layer_idx -> linear probe AUROC
        mlp_aurocs: Dict layer_idx -> MLP probe AUROC
        output_dir: Where to save.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    layers = sorted(linear_aurocs.keys())
    lin_vals = [linear_aurocs[l] for l in layers]
    mlp_vals = [mlp_aurocs[l] for l in layers]

    ax.plot(layers, lin_vals, "o-", color="#d62728", label="Linear Probe", linewidth=2)
    ax.plot(layers, mlp_vals, "s--", color="#1f77b4", label="MLP Probe", linewidth=2)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Chance")

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("Probing Accuracy by Layer", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(0.4, 1.05)
    ax.grid(alpha=0.3)

    # Highlight best layer
    best_layer = max(linear_aurocs, key=linear_aurocs.get)
    ax.axvline(x=best_layer, color="red", alpha=0.3, linewidth=8)
    ax.annotate(
        f"Best: L{best_layer}",
        xy=(best_layer, linear_aurocs[best_layer]),
        xytext=(best_layer + 2, linear_aurocs[best_layer] - 0.05),
        fontsize=10,
        arrowprops=dict(arrowstyle="->"),
    )

    save_fig(fig, "layer_auroc_curve", output_dir)


def plot_gap_by_layer(
    gaps: Dict[int, float],
    output_dir: str = "results/figures",
):
    """
    Plot the projection gap (ΔP) as a function of layer.
    
    Shows which layers have the largest format-dependent difference.
    
    Args:
        gaps: Dict layer_idx -> ΔP value
        output_dir: Where to save.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    layers = sorted(gaps.keys())
    values = [gaps[l] for l in layers]

    bars = ax.bar(layers, values, color=["#d62728" if v > 0 else "#1f77b4" for v in values], alpha=0.7)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("ΔP (Chat - Agent)", fontsize=12)
    ax.set_title("Projection Gap by Layer", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    # Highlight peak
    peak_layer = max(gaps, key=gaps.get)
    ax.annotate(
        f"Peak: L{peak_layer}\nΔP={gaps[peak_layer]:.3f}",
        xy=(peak_layer, gaps[peak_layer]),
        xytext=(peak_layer + 3, gaps[peak_layer] * 0.8),
        fontsize=10,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->"),
    )

    save_fig(fig, "gap_by_layer", output_dir)


def plot_pca_activations(
    activations: Dict[str, np.ndarray],
    layer_idx: int,
    output_dir: str = "results/figures",
):
    """
    PCA visualization of activation space colored by condition.
    
    Args:
        activations: Dict condition -> (n_samples, hidden_dim)
        layer_idx: Layer index for labeling.
        output_dir: Where to save.
    """
    from sklearn.decomposition import PCA

    # Combine all activations
    all_acts = []
    all_labels = []
    for cond in ["chat_harmful", "agent_harmful", "chat_benign", "agent_benign"]:
        if cond in activations:
            all_acts.append(activations[cond])
            all_labels.extend([cond] * len(activations[cond]))

    X = np.vstack(all_acts)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    start = 0
    for cond in ["chat_harmful", "agent_harmful", "chat_benign", "agent_benign"]:
        if cond in activations:
            n = len(activations[cond])
            ax.scatter(
                X_pca[start:start + n, 0],
                X_pca[start:start + n, 1],
                c=COLORS[cond],
                label=LABELS[cond],
                alpha=0.6,
                s=30,
            )
            start += n

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)", fontsize=12)
    ax.set_title(f"PCA of Activations (Layer {layer_idx})", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    save_fig(fig, f"pca_activations_layer{layer_idx}", output_dir)


def plot_intervention_dose_response(
    alphas: List[float],
    success_rates: List[float],
    output_dir: str = "results/figures",
):
    """
    Plot intervention success rate as a function of α (dose-response).
    
    Shows the minimum α needed to restore refusal.
    
    Args:
        alphas: List of α values tested.
        success_rates: Corresponding success rates.
        output_dir: Where to save.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.plot(alphas, success_rates, "o-", color="#d62728", linewidth=2, markersize=8)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% threshold")

    ax.set_xlabel("α (Intervention Strength)", fontsize=12)
    ax.set_ylabel("Refusal Restoration Rate", fontsize=12)
    ax.set_title("Intervention Dose-Response Curve", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # Find minimum effective α
    for i, rate in enumerate(success_rates):
        if rate >= 0.5:
            ax.axvline(x=alphas[i], color="green", alpha=0.5, linewidth=2)
            ax.annotate(
                f"α_min = {alphas[i]:.1f}",
                xy=(alphas[i], rate),
                xytext=(alphas[i] + 0.5, rate - 0.15),
                fontsize=11,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->"),
            )
            break

    save_fig(fig, "intervention_dose_response", output_dir)


def plot_cross_tool_heatmap(
    gaps_by_tool: Dict[str, float],
    output_dir: str = "results/figures",
):
    """
    Heatmap showing ΔP consistency across tools.
    
    Args:
        gaps_by_tool: Dict tool_name -> ΔP
        output_dir: Where to save.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    tools = list(gaps_by_tool.keys())
    values = [gaps_by_tool[t] for t in tools]

    # Horizontal bar chart (better than heatmap for 1D)
    bars = ax.barh(tools, values, color=["#d62728" if v > 0 else "#1f77b4" for v in values], alpha=0.7)
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_xlabel("ΔP (Projection Gap)", fontsize=12)
    ax.set_title("Projection Gap by Tool", fontsize=14)
    ax.grid(axis="x", alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            val + 0.01 if val > 0 else val - 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left" if val > 0 else "right",
            fontsize=9,
        )

    save_fig(fig, "cross_tool_gap", output_dir)


def plot_cosine_similarity_heatmap(
    cosines_by_layer: Dict[int, float],
    output_dir: str = "results/figures",
):
    """
    Plot cosine similarity between probe direction and refusal direction by layer.
    
    Args:
        cosines_by_layer: Dict layer_idx -> cosine similarity.
        output_dir: Where to save.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    layers = sorted(cosines_by_layer.keys())
    values = [cosines_by_layer[l] for l in layers]

    ax.bar(layers, values, color="#9467bd", alpha=0.7)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axhline(y=1, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Cosine Similarity", fontsize=12)
    ax.set_title("Probe Direction vs. Refusal Direction (Cosine Similarity)", fontsize=14)
    ax.set_ylim(-0.2, 1.1)
    ax.grid(axis="y", alpha=0.3)

    save_fig(fig, "cosine_probe_vs_refusal", output_dir)


def plot_decomposition_waterfall(
    decomposition: Dict,
    layer_idx: int,
    output_dir: str = "results/figures",
):
    """
    Waterfall chart showing how refusal projection drops across conditions.

    Shows: chat → role_only → role_plus_tools → agent_full
    with bars indicating how much each factor contributes to the total drop.

    Args:
        decomposition: Output of decompose_format_effects() containing
                       per-effect delta_p, ci_lower, ci_upper.
        layer_idx: Layer index for labeling.
        output_dir: Where to save.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: Waterfall (absolute projection means) ---
    conditions = ["chat", "role_only", "role_plus_tools", "agent_full"]
    condition_labels = ["Chat\n(baseline)", "Role Only\n(+agent role)",
                        "Role+Tools\n(+tool defs)", "Agent Full\n(+JSON format)"]

    means = [
        decomposition["total_effect"]["mean_a"],  # chat mean
        decomposition["role_effect"]["mean_b"],    # role_only mean
        decomposition["tools_effect"]["mean_b"],   # role_plus_tools mean
        decomposition["json_format_effect"]["mean_b"],  # agent_full mean
    ]

    colors_bar = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]
    bars = ax1.bar(range(4), means, color=colors_bar, alpha=0.8, edgecolor="black", linewidth=0.5)

    # Connect bars with arrows showing the drop
    for i in range(3):
        drop = means[i] - means[i + 1]
        mid_x = i + 0.5
        mid_y = (means[i] + means[i + 1]) / 2
        ax1.annotate(
            f"Δ={drop:.3f}",
            xy=(mid_x, mid_y),
            fontsize=9,
            ha="center",
            color="#c0392b" if drop > 0 else "#2980b9",
            fontweight="bold",
        )
        ax1.plot([i + 0.4, i + 0.6], [means[i], means[i + 1]],
                 color="#7f8c8d", linewidth=1.5, linestyle="--")

    ax1.set_xticks(range(4))
    ax1.set_xticklabels(condition_labels, fontsize=10)
    ax1.set_ylabel("Mean Projection onto Refusal Direction", fontsize=11)
    ax1.set_title(f"Refusal Activation by Condition (Layer {layer_idx})", fontsize=13)
    ax1.grid(axis="y", alpha=0.3)

    # --- Right: Effect size bars ---
    effects = ["role_effect", "tools_effect", "json_format_effect"]
    effect_labels = ["Role\n(chat→role_only)", "Tools\n(role_only→role+tools)",
                     "JSON Format\n(role+tools→agent_full)"]
    deltas = [decomposition[e]["delta_p"] for e in effects]
    ci_lowers = [decomposition[e]["ci_lower"] for e in effects]
    ci_uppers = [decomposition[e]["ci_upper"] for e in effects]
    p_values = [decomposition[e]["p_value"] for e in effects]

    # Error bars
    errors = [[d - cl for d, cl in zip(deltas, ci_lowers)],
              [cu - d for d, cu in zip(deltas, ci_uppers)]]

    bar_colors = ["#f39c12" if p < 0.05 else "#bdc3c7" for p in p_values]
    bars2 = ax2.bar(range(3), deltas, color=bar_colors, alpha=0.8,
                    edgecolor="black", linewidth=0.5, yerr=errors, capsize=5)

    # Add significance stars
    for i, p in enumerate(p_values):
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax2.text(i, deltas[i] + errors[1][i] + 0.01, star,
                 ha="center", fontsize=11, color="#c0392b" if p < 0.05 else "#7f8c8d")

    ax2.set_xticks(range(3))
    ax2.set_xticklabels(effect_labels, fontsize=10)
    ax2.set_ylabel("ΔP (drop in refusal projection)", fontsize=11)
    ax2.set_title("Effect Decomposition\n(orange = p<0.05, gray = ns)", fontsize=13)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.grid(axis="y", alpha=0.3)

    # Add total effect annotation
    total = decomposition["total_effect"]["delta_p"]
    ax2.axhline(total, color="#e74c3c", linewidth=1, linestyle=":", alpha=0.7)
    ax2.text(2.5, total, f"Total ΔP={total:.3f}", fontsize=9,
             color="#e74c3c", ha="right", va="bottom")

    plt.tight_layout()
    save_fig(fig, "decomposition_waterfall", output_dir)


def generate_all_figures(results: Dict, output_dir: str = "results/figures"):
    """
    Generate all figures from experiment results.
    
    Args:
        results: Dict containing all experiment results.
        output_dir: Directory to save figures.
    """
    print("\n" + "=" * 50)
    print("GENERATING FIGURES")
    print("=" * 50)

    # 1. Projection distributions
    if "projections" in results and "best_layer" in results:
        plot_projection_distributions(
            results["projections"],
            results["best_layer"],
            output_dir=output_dir,
        )

    # 2. Layer AUROC curves
    if "linear_aurocs" in results and "mlp_aurocs" in results:
        plot_layer_auroc_curve(
            results["linear_aurocs"],
            results["mlp_aurocs"],
            output_dir=output_dir,
        )

    # 3. Gap by layer
    if "gaps_by_layer" in results:
        plot_gap_by_layer(results["gaps_by_layer"], output_dir=output_dir)

    # 4. PCA
    if "activations" in results and "best_layer" in results:
        plot_pca_activations(
            results["activations"],
            results["best_layer"],
            output_dir=output_dir,
        )

    # 5. Intervention dose-response
    if "intervention_alphas" in results and "intervention_rates" in results:
        plot_intervention_dose_response(
            results["intervention_alphas"],
            results["intervention_rates"],
            output_dir=output_dir,
        )

    # 6. Cross-tool consistency
    if "gaps_by_tool" in results:
        plot_cross_tool_heatmap(results["gaps_by_tool"], output_dir=output_dir)

    # 7. Cosine similarity
    if "cosines_by_layer" in results:
        plot_cosine_similarity_heatmap(
            results["cosines_by_layer"], output_dir=output_dir
        )

    # 8. Decomposition waterfall (if decompose mode was used)
    if "decomposition" in results and results["decomposition"]:
        decomp = results["decomposition"]
        if "best_layer" in decomp and decomp["best_layer"]:
            plot_decomposition_waterfall(
                decomp["best_layer"],
                results.get("best_layer", 0),
                output_dir=output_dir,
            )

    print(f"\nAll figures saved to {output_dir}/")
