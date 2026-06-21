"""
Agent-Specific Direction Analysis and Safety Monitor.

Key insight: The refusal direction computed from CHAT activations (d_chat)
does not transfer to AGENT activations (AUROC ~0.5). However, a linear probe
trained on agent activations achieves AUROC ~1.0, meaning the information IS
there but in a DIFFERENT direction.

This module:
1. Extracts the agent-specific refusal direction (w_agent) from a trained probe
2. Compares w_agent vs d_chat (cosine similarity per layer)
3. Provides intervention using w_agent instead of d_chat
4. Implements a safety monitor using w_agent for real-time detection
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


@dataclass
class AgentDirectionResult:
    """Result of agent-specific direction extraction."""
    layer_idx: int
    w_agent: np.ndarray         # Normalized agent refusal direction
    d_chat: np.ndarray          # Original chat refusal direction
    cosine_similarity: float    # cos(w_agent, d_chat)
    auroc_agent_w_agent: float  # AUROC on agent data using w_agent
    auroc_agent_d_chat: float   # AUROC on agent data using d_chat (expected ~0.5)


@dataclass
class MonitorResult:
    """Result of safety monitor evaluation."""
    threshold: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    true_positive_rate: float
    n_total: int
    n_harmful: int
    n_benign: int


# ============================================================
# 1. AGENT-SPECIFIC DIRECTION EXTRACTION
# ============================================================

def extract_agent_direction(
    agent_harmful_acts: np.ndarray,
    agent_benign_acts: np.ndarray,
    d_chat: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Extract the agent-specific refusal direction by training a linear probe
    on agent activations and extracting its weight vector.

    Args:
        agent_harmful_acts: Agent harmful activations (n, hidden_dim)
        agent_benign_acts: Agent benign activations (n, hidden_dim)
        d_chat: Optional chat refusal direction for comparison.

    Returns:
        Tuple of (w_agent normalized, metadata dict)
    """
    # Prepare data
    X = np.concatenate([agent_harmful_acts, agent_benign_acts], axis=0)
    y = np.concatenate([
        np.ones(len(agent_harmful_acts)),
        np.zeros(len(agent_benign_acts)),
    ])

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train logistic regression
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
    clf.fit(X_scaled, y)

    # Extract direction from weights (accounting for scaling)
    # The decision boundary in original space: w_original = scaler.inverse_transform_direction(clf.coef_)
    # For direction comparison, we use the raw coef since we'll project scaled data
    w_agent_raw = clf.coef_[0].copy()

    # To get direction in ORIGINAL (unscaled) space:
    # If scaler transforms x -> (x - mean) / std, then the hyperplane in original space
    # has normal vector w_original = w_scaled / std
    w_agent = w_agent_raw / scaler.scale_
    w_agent = w_agent / np.linalg.norm(w_agent)

    # Compute AUROC using w_agent on agent data
    projections_w_agent = X @ w_agent
    auroc_w_agent = float(roc_auc_score(y, projections_w_agent))

    # Compute AUROC using d_chat on agent data (expected ~0.5)
    auroc_d_chat = 0.5
    cosine = 0.0
    if d_chat is not None:
        projections_d_chat = X @ d_chat
        try:
            auroc_d_chat = float(roc_auc_score(y, projections_d_chat))
        except ValueError:
            auroc_d_chat = 0.5
        cosine = float(np.dot(w_agent, d_chat))

    metadata = {
        "auroc_w_agent": auroc_w_agent,
        "auroc_d_chat": auroc_d_chat,
        "cosine_with_d_chat": cosine,
        "probe_accuracy": float(clf.score(X_scaled, y)),
    }

    return w_agent, metadata


def extract_agent_directions_all_layers(
    agent_harmful_acts: Dict[int, np.ndarray],
    agent_benign_acts: Dict[int, np.ndarray],
    d_chat_directions: Dict[int, np.ndarray],
) -> Dict[int, AgentDirectionResult]:
    """
    Extract agent-specific direction for each layer.

    Args:
        agent_harmful_acts: Dict layer -> (n, hidden_dim) agent harmful activations
        agent_benign_acts: Dict layer -> (n, hidden_dim) agent benign activations
        d_chat_directions: Dict layer -> d_chat direction vector

    Returns:
        Dict layer -> AgentDirectionResult
    """
    results = {}

    for layer_idx in sorted(agent_harmful_acts.keys()):
        if layer_idx not in agent_benign_acts:
            continue

        d_chat = d_chat_directions.get(layer_idx)

        w_agent, meta = extract_agent_direction(
            agent_harmful_acts[layer_idx],
            agent_benign_acts[layer_idx],
            d_chat=d_chat,
        )

        results[layer_idx] = AgentDirectionResult(
            layer_idx=layer_idx,
            w_agent=w_agent,
            d_chat=d_chat if d_chat is not None else np.zeros_like(w_agent),
            cosine_similarity=meta["cosine_with_d_chat"],
            auroc_agent_w_agent=meta["auroc_w_agent"],
            auroc_agent_d_chat=meta["auroc_d_chat"],
        )

    return results


# ============================================================
# 2. SAFETY MONITOR
# ============================================================

def calibrate_threshold(
    harmful_projections: np.ndarray,
    benign_projections: np.ndarray,
    metric: str = "f1",
) -> float:
    """
    Calibrate the detection threshold to maximize F1-score.

    Args:
        harmful_projections: Projections of harmful samples onto w_agent.
        benign_projections: Projections of benign samples onto w_agent.
        metric: 'f1' or 'balanced_accuracy' to optimize.

    Returns:
        Optimal threshold value.
    """
    all_projections = np.concatenate([harmful_projections, benign_projections])
    labels = np.concatenate([
        np.ones(len(harmful_projections)),
        np.zeros(len(benign_projections)),
    ])

    # Search over percentiles of the combined distribution
    thresholds = np.percentile(all_projections, np.arange(5, 95, 1))

    best_threshold = float(np.median(all_projections))
    best_score = 0.0

    for t in thresholds:
        predictions = (all_projections > t).astype(int)
        if metric == "f1":
            score = f1_score(labels, predictions, zero_division=0)
        else:
            tp = ((predictions == 1) & (labels == 1)).sum()
            tn = ((predictions == 0) & (labels == 0)).sum()
            score = (tp / labels.sum() + tn / (1 - labels).sum()) / 2

        if score > best_score:
            best_score = score
            best_threshold = float(t)

    return best_threshold


def safety_monitor(
    activations: np.ndarray,
    w_agent: np.ndarray,
    threshold: float,
) -> bool:
    """
    Real-time safety monitor for agentic activations.

    Takes activations from the last token before generation starts,
    projects onto w_agent, and flags if harmful intent is detected.

    Args:
        activations: Activation vector (hidden_dim,) from target layer.
        w_agent: Agent-specific refusal direction (hidden_dim,).
        threshold: Calibrated threshold (from calibrate_threshold).

    Returns:
        True if harmful intent detected (projection > threshold), False otherwise.
    """
    projection = float(np.dot(activations, w_agent))
    return projection > threshold


def evaluate_monitor(
    harmful_acts: np.ndarray,
    benign_acts: np.ndarray,
    w_agent: np.ndarray,
    threshold: float,
) -> MonitorResult:
    """
    Evaluate the safety monitor over a full dataset.

    Args:
        harmful_acts: Harmful activations (n_harmful, hidden_dim)
        benign_acts: Benign activations (n_benign, hidden_dim)
        w_agent: Agent-specific direction.
        threshold: Detection threshold.

    Returns:
        MonitorResult with precision, recall, F1, FPR.
    """
    # Project all samples
    proj_harmful = harmful_acts @ w_agent
    proj_benign = benign_acts @ w_agent

    # Predictions
    pred_harmful = (proj_harmful > threshold).astype(int)
    pred_benign = (proj_benign > threshold).astype(int)

    # Combine
    y_true = np.concatenate([np.ones(len(harmful_acts)), np.zeros(len(benign_acts))])
    y_pred = np.concatenate([pred_harmful, pred_benign])

    # Metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # False positive rate
    fp = pred_benign.sum()
    fpr = fp / len(benign_acts) if len(benign_acts) > 0 else 0.0

    # True positive rate
    tp = pred_harmful.sum()
    tpr = tp / len(harmful_acts) if len(harmful_acts) > 0 else 0.0

    return MonitorResult(
        threshold=threshold,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        false_positive_rate=float(fpr),
        true_positive_rate=float(tpr),
        n_total=len(harmful_acts) + len(benign_acts),
        n_harmful=len(harmful_acts),
        n_benign=len(benign_acts),
    )


# ============================================================
# 3. SUMMARY AND REPORTING
# ============================================================

def print_agent_direction_summary(
    layer_results: Dict[int, AgentDirectionResult],
    monitor_result: Optional[MonitorResult] = None,
):
    """Print comprehensive summary of agent direction analysis."""
    print("\n" + "=" * 70)
    print("AGENT-SPECIFIC DIRECTION ANALYSIS")
    print("=" * 70)

    print(f"\n{'Layer':<8} {'cos(w_agent, d_chat)':<22} {'AUROC(w_agent)':<16} {'AUROC(d_chat)':<16}")
    print("-" * 62)
    for layer_idx in sorted(layer_results.keys()):
        r = layer_results[layer_idx]
        print(f"{layer_idx:<8} {r.cosine_similarity:<22.4f} "
              f"{r.auroc_agent_w_agent:<16.4f} {r.auroc_agent_d_chat:<16.4f}")

    # Best layer
    best_layer = max(layer_results, key=lambda l: layer_results[l].auroc_agent_w_agent)
    best = layer_results[best_layer]
    print(f"\nBest layer: {best_layer}")
    print(f"  w_agent AUROC: {best.auroc_agent_w_agent:.4f}")
    print(f"  d_chat AUROC:  {best.auroc_agent_d_chat:.4f}")
    print(f"  Cosine sim:    {best.cosine_similarity:.4f}")

    if abs(best.cosine_similarity) < 0.3:
        print(f"\n  INTERPRETATION: w_agent and d_chat are NEARLY ORTHOGONAL.")
        print(f"  The model encodes harmfulness in a DIFFERENT direction in agent context.")
        print(f"  This explains why d_chat-based intervention fails in agent mode.")
    elif abs(best.cosine_similarity) > 0.7:
        print(f"\n  INTERPRETATION: w_agent and d_chat are well ALIGNED.")
        print(f"  The issue may be scaling rather than direction.")

    if monitor_result:
        print(f"\n--- SAFETY MONITOR (layer {best_layer}) ---")
        print(f"  Threshold:    {monitor_result.threshold:.4f}")
        print(f"  Precision:    {monitor_result.precision:.4f}")
        print(f"  Recall:       {monitor_result.recall:.4f}")
        print(f"  F1:           {monitor_result.f1:.4f}")
        print(f"  FPR:          {monitor_result.false_positive_rate:.4f}")
        print(f"  TPR:          {monitor_result.true_positive_rate:.4f}")
        print(f"  Samples:      {monitor_result.n_harmful} harmful + {monitor_result.n_benign} benign")

    print("=" * 70)
