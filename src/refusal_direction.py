"""
Refusal direction extraction and analysis module.

Implements the difference-in-means method from Arditi et al. (2024)
to extract the refusal direction from model activations.
Also includes PCA-based validation and permutation testing for
statistical significance of the projection gap.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA


def compute_refusal_direction(
    harmful_activations: np.ndarray,
    benign_activations: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute the refusal direction using difference-in-means.
    
    d_refusal = mean(harmful_activations) - mean(benign_activations)
    
    Args:
        harmful_activations: Array of shape (n_harmful, hidden_dim)
        benign_activations: Array of shape (n_benign, hidden_dim)
        normalize: Whether to L2-normalize the direction.
        
    Returns:
        Refusal direction vector of shape (hidden_dim,)
    """
    mean_harmful = harmful_activations.mean(axis=0)
    mean_benign = benign_activations.mean(axis=0)
    
    direction = mean_harmful - mean_benign
    
    if normalize:
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            raise ValueError(
                "Refusal direction has near-zero norm. "
                "Check that harmful and benign activations are different."
            )
        direction = direction / norm
    
    return direction


def compute_refusal_direction_pca(
    harmful_activations: np.ndarray,
    benign_activations: np.ndarray,
    n_components: int = 5,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute the refusal direction using PCA on the contrast matrix.
    
    Validates that the difference-in-means direction aligns with the
    principal component of variation between harmful and benign activations.
    
    Method:
        1. Center each group by its own mean.
        2. Concatenate and apply PCA to the contrast (harmful - benign centroid).
        3. PC1 of the between-group difference should align with diff-in-means.
    
    Args:
        harmful_activations: Array of shape (n_harmful, hidden_dim)
        benign_activations: Array of shape (n_benign, hidden_dim)
        n_components: Number of principal components to compute.
        
    Returns:
        Tuple of:
          - PC1 direction (hidden_dim,), normalized
          - Validation dict with cosine similarity to diff-in-means,
            explained variance ratios, etc.
    """
    # Difference-in-means direction for comparison
    diff_means_dir = compute_refusal_direction(
        harmful_activations, benign_activations, normalize=True
    )
    
    # Build contrast matrix: shift each sample by global mean
    global_mean = np.concatenate(
        [harmful_activations, benign_activations], axis=0
    ).mean(axis=0)
    
    harmful_centered = harmful_activations - global_mean
    benign_centered = benign_activations - global_mean
    contrast_matrix = np.concatenate(
        [harmful_centered, benign_centered], axis=0
    )
    
    # PCA
    n_components = min(n_components, min(contrast_matrix.shape) - 1)
    pca = PCA(n_components=n_components)
    pca.fit(contrast_matrix)
    
    pc1 = pca.components_[0]
    pc1 = pc1 / np.linalg.norm(pc1)
    
    # Ensure consistent sign (align with diff-in-means)
    cosine_sim = np.dot(pc1, diff_means_dir)
    if cosine_sim < 0:
        pc1 = -pc1
        cosine_sim = -cosine_sim
    
    validation = {
        "cosine_similarity_with_diff_means": float(cosine_sim),
        "explained_variance_ratio_pc1": float(pca.explained_variance_ratio_[0]),
        "explained_variance_ratios": pca.explained_variance_ratio_.tolist(),
        "directions_aligned": bool(cosine_sim > 0.8),
    }
    
    return pc1, validation


def compute_refusal_directions_all_layers(
    harmful_activations: Dict[int, np.ndarray],
    benign_activations: Dict[int, np.ndarray],
) -> Dict[int, np.ndarray]:
    """
    Compute refusal direction for each layer.
    
    Args:
        harmful_activations: Dict mapping layer_idx -> (n_harmful, hidden_dim)
        benign_activations: Dict mapping layer_idx -> (n_benign, hidden_dim)
        
    Returns:
        Dict mapping layer_idx -> normalized direction vector (hidden_dim,)
    """
    directions = {}
    for layer_idx in harmful_activations.keys():
        if layer_idx not in benign_activations:
            continue
        directions[layer_idx] = compute_refusal_direction(
            harmful_activations[layer_idx],
            benign_activations[layer_idx],
        )
    return directions


def project_onto_direction(
    activations: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """
    Project activations onto a direction (dot product).
    
    Args:
        activations: Array of shape (n_samples, hidden_dim) or (hidden_dim,)
        direction: Normalized direction vector of shape (hidden_dim,)
        
    Returns:
        Projections of shape (n_samples,) or scalar
    """
    if activations.ndim == 1:
        return np.dot(activations, direction)
    return activations @ direction


def compute_projection_gap(
    chat_harmful_acts: np.ndarray,
    agent_harmful_acts: np.ndarray,
    direction: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Compute the projection gap between chat and agent formats
    with statistical significance testing.
    
    This is the MAIN METRIC of the project:
    ΔP = mean(projection_chat_harmful) - mean(projection_agent_harmful)
    
    Statistical tests included:
    - Permutation test (non-parametric, no distribution assumptions)
    - Welch's t-test (parametric, for comparison)
    - Cohen's d effect size
    
    Args:
        chat_harmful_acts: Chat format harmful activations (n, hidden_dim)
        agent_harmful_acts: Agent format harmful activations (n, hidden_dim)
        direction: Refusal direction (hidden_dim,)
        n_permutations: Number of permutations for the permutation test.
        seed: Random seed for reproducibility.
        
    Returns:
        Dict with gap statistics and significance measures.
    """
    proj_chat = project_onto_direction(chat_harmful_acts, direction)
    proj_agent = project_onto_direction(agent_harmful_acts, direction)
    
    observed_gap = proj_chat.mean() - proj_agent.mean()
    
    # --- Permutation test ---
    rng = np.random.default_rng(seed)
    combined = np.concatenate([proj_chat, proj_agent])
    n_chat = len(proj_chat)
    
    null_gaps = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(combined)
        null_gaps[i] = perm[:n_chat].mean() - perm[n_chat:].mean()
    
    # Two-sided p-value
    p_value_perm = (np.abs(null_gaps) >= np.abs(observed_gap)).mean()
    
    # --- Welch's t-test (parametric comparison) ---
    t_stat, p_value_ttest = stats.ttest_ind(
        proj_chat, proj_agent, equal_var=False
    )
    
    # --- Effect size: Cohen's d ---
    pooled_std = np.sqrt(
        (proj_chat.std(ddof=1) ** 2 + proj_agent.std(ddof=1) ** 2) / 2
    )
    cohens_d = observed_gap / pooled_std if pooled_std > 1e-10 else 0.0
    
    return {
        "delta_p": float(observed_gap),
        "mean_chat_harmful": float(proj_chat.mean()),
        "std_chat_harmful": float(proj_chat.std()),
        "mean_agent_harmful": float(proj_agent.mean()),
        "std_agent_harmful": float(proj_agent.std()),
        # Statistical significance
        "p_value_permutation": float(p_value_perm),
        "p_value_ttest_welch": float(p_value_ttest),
        "t_statistic": float(t_stat),
        "cohens_d": float(cohens_d),
        "n_permutations": n_permutations,
        "is_significant_005": bool(p_value_perm < 0.05),
        "is_significant_001": bool(p_value_perm < 0.01),
        # Raw projections
        "chat_projections": proj_chat,
        "agent_projections": proj_agent,
    }


def compute_projections_all_conditions(
    activations: Dict[str, np.ndarray],
    direction: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute projections for all experimental conditions.
    
    Args:
        activations: Dict with keys like 'chat_harmful', 'chat_benign',
                     'agent_harmful', 'agent_benign'
        direction: Refusal direction vector.
        
    Returns:
        Dict mapping condition -> projection array
    """
    projections = {}
    for condition, acts in activations.items():
        projections[condition] = project_onto_direction(acts, direction)
    return projections


def find_best_layer(
    harmful_activations: Dict[int, np.ndarray],
    benign_activations: Dict[int, np.ndarray],
    metric: str = "auroc",
) -> Tuple[int, float]:
    """
    Find the layer with best separation between harmful and benign.
    
    Args:
        harmful_activations: Dict mapping layer -> activations
        benign_activations: Dict mapping layer -> activations
        metric: 'auroc' or 'gap' (mean difference)
        
    Returns:
        Tuple of (best_layer_idx, best_score)
    """
    from sklearn.metrics import roc_auc_score
    
    best_layer = -1
    best_score = -1.0
    
    for layer_idx in sorted(harmful_activations.keys()):
        direction = compute_refusal_direction(
            harmful_activations[layer_idx],
            benign_activations[layer_idx],
        )
        
        proj_harmful = project_onto_direction(
            harmful_activations[layer_idx], direction
        )
        proj_benign = project_onto_direction(
            benign_activations[layer_idx], direction
        )
        
        if metric == "auroc":
            labels = np.concatenate([
                np.ones(len(proj_harmful)),
                np.zeros(len(proj_benign)),
            ])
            scores = np.concatenate([proj_harmful, proj_benign])
            score = roc_auc_score(labels, scores)
        elif metric == "gap":
            score = abs(proj_harmful.mean() - proj_benign.mean())
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_layer = layer_idx
    
    return best_layer, best_score


def save_directions(
    directions: Dict[int, np.ndarray],
    path: str,
    metadata: Optional[Dict] = None,
):
    """Save extracted directions to disk."""
    save_dict = {
        "directions": directions,
        "metadata": metadata or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"layer_{k}": v for k, v in directions.items()})
    print(f"Saved directions for {len(directions)} layers to {path}")


def load_directions(path: str) -> Dict[int, np.ndarray]:
    """Load directions from disk."""
    data = np.load(path)
    directions = {}
    for key in data.files:
        layer_idx = int(key.replace("layer_", ""))
        directions[layer_idx] = data[key]
    return directions


def full_refusal_analysis(
    harmful_activations: np.ndarray,
    benign_activations: np.ndarray,
    chat_harmful_acts: np.ndarray,
    agent_harmful_acts: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> Dict:
    """
    Run the complete refusal direction analysis pipeline for a single layer.
    
    Combines:
    1. Difference-in-means direction extraction
    2. PCA validation (checks alignment between methods)
    3. Projection gap with statistical significance
    
    Args:
        harmful_activations: Harmful prompts activations (n, hidden_dim)
            used to define the refusal direction.
        benign_activations: Benign prompts activations (n, hidden_dim)
            used to define the refusal direction.
        chat_harmful_acts: Chat-format harmful activations for gap analysis.
        agent_harmful_acts: Agent-format harmful activations for gap analysis.
        n_permutations: Number of permutations for significance test.
        seed: Random seed.
        
    Returns:
        Dict with:
          - 'direction': the refusal direction vector
          - 'pca_validation': PCA alignment metrics
          - 'gap_analysis': projection gap with p-values and effect size
    """
    # 1. Difference-in-means
    direction = compute_refusal_direction(
        harmful_activations, benign_activations, normalize=True
    )
    
    # 2. PCA validation
    pca_direction, pca_validation = compute_refusal_direction_pca(
        harmful_activations, benign_activations
    )
    
    # 3. Projection gap with significance
    gap_analysis = compute_projection_gap(
        chat_harmful_acts,
        agent_harmful_acts,
        direction,
        n_permutations=n_permutations,
        seed=seed,
    )
    
    return {
        "direction": direction,
        "pca_direction": pca_direction,
        "pca_validation": pca_validation,
        "gap_analysis": gap_analysis,
    }


def compute_gap_by_layer(
    harmful_activations: Dict[int, np.ndarray],
    benign_activations: Dict[int, np.ndarray],
    chat_harmful_acts: Dict[int, np.ndarray],
    agent_harmful_acts: Dict[int, np.ndarray],
    n_permutations: int = 10000,
    seed: int = 42,
) -> Dict[int, Dict]:
    """
    Run full_refusal_analysis for each layer and return results per layer.
    
    Args:
        harmful_activations: Dict layer_idx -> (n, hidden_dim)
        benign_activations: Dict layer_idx -> (n, hidden_dim)
        chat_harmful_acts: Dict layer_idx -> (n, hidden_dim)
        agent_harmful_acts: Dict layer_idx -> (n, hidden_dim)
        n_permutations: Permutations per layer.
        seed: Random seed.
        
    Returns:
        Dict mapping layer_idx -> full_refusal_analysis results
    """
    results = {}
    for layer_idx in sorted(harmful_activations.keys()):
        if layer_idx not in benign_activations:
            continue
        if layer_idx not in chat_harmful_acts:
            continue
        if layer_idx not in agent_harmful_acts:
            continue
        
        results[layer_idx] = full_refusal_analysis(
            harmful_activations[layer_idx],
            benign_activations[layer_idx],
            chat_harmful_acts[layer_idx],
            agent_harmful_acts[layer_idx],
            n_permutations=n_permutations,
            seed=seed,
        )
    
    return results
