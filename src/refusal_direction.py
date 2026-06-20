"""
Refusal direction extraction and analysis module.

Implements the difference-in-means method from Arditi et al. (2024)
to extract the refusal direction from model activations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


def compute_refusal_direction(
    harmful_activations: np.ndarray,
    benign_activations: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute the refusal direction using difference-in-means.
    
    Following Arditi et al. (2024):
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
) -> Dict[str, float]:
    """
    Compute the projection gap between chat and agent formats.
    
    This is the MAIN METRIC of the project:
    ΔP = mean(projection_chat_harmful) - mean(projection_agent_harmful)
    
    Args:
        chat_harmful_acts: Chat format harmful activations (n, hidden_dim)
        agent_harmful_acts: Agent format harmful activations (n, hidden_dim)
        direction: Refusal direction (hidden_dim,)
        
    Returns:
        Dict with gap statistics
    """
    proj_chat = project_onto_direction(chat_harmful_acts, direction)
    proj_agent = project_onto_direction(agent_harmful_acts, direction)
    
    gap = proj_chat.mean() - proj_agent.mean()
    
    return {
        "delta_p": float(gap),
        "mean_chat_harmful": float(proj_chat.mean()),
        "std_chat_harmful": float(proj_chat.std()),
        "mean_agent_harmful": float(proj_agent.mean()),
        "std_agent_harmful": float(proj_agent.std()),
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
