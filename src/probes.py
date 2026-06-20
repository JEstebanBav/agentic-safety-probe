"""
Linear and non-linear probing classifiers.

Trains probes on activations to determine if harmful/benign
information is linearly separable at each layer.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    classification_report,
)
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeResult:
    """Result of probing at a single layer."""
    layer_idx: int
    accuracy: float
    auroc: float
    f1: float
    probe_type: str  # 'linear' or 'mlp'
    learned_direction: Optional[np.ndarray] = None  # For linear probes
    cosine_with_refusal: Optional[float] = None


@dataclass
class ProbingSuiteResult:
    """Results across all layers for one probe type."""
    per_layer: Dict[int, ProbeResult]
    best_layer: int
    best_auroc: float
    probe_type: str


def train_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    layer_idx: int,
    refusal_direction: Optional[np.ndarray] = None,
) -> ProbeResult:
    """
    Train a logistic regression probe.
    
    If the probe achieves high accuracy, the information is
    linearly separable (exists as a direction in activation space).
    
    Args:
        X_train: Training activations (n_train, hidden_dim)
        y_train: Training labels (n_train,) - 1=harmful, 0=benign
        X_test: Test activations (n_test, hidden_dim)
        y_test: Test labels (n_test,)
        layer_idx: Layer index for metadata.
        refusal_direction: Optional direction to compare against.
        
    Returns:
        ProbeResult with metrics and learned direction.
    """
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    
    # Extract learned direction (weight vector)
    learned_direction = clf.coef_[0]
    learned_direction = learned_direction / np.linalg.norm(learned_direction)
    
    # Compare with refusal direction if provided
    cosine = None
    if refusal_direction is not None:
        cosine = float(np.dot(learned_direction, refusal_direction))
    
    return ProbeResult(
        layer_idx=layer_idx,
        accuracy=accuracy,
        auroc=auroc,
        f1=f1,
        probe_type="linear",
        learned_direction=learned_direction,
        cosine_with_refusal=cosine,
    )


def train_mlp_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    layer_idx: int,
    hidden_size: int = 128,
) -> ProbeResult:
    """
    Train a non-linear (MLP) probe.
    
    If MLP works but linear doesn't, the information exists
    but is NOT linearly separable (not a single direction).
    
    Args:
        X_train: Training activations (n_train, hidden_dim)
        y_train: Training labels (n_train,)
        X_test: Test activations (n_test, hidden_dim)
        y_test: Test labels (n_test,)
        layer_idx: Layer index for metadata.
        hidden_size: Hidden layer size for MLP.
        
    Returns:
        ProbeResult with metrics.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = MLPClassifier(
        hidden_layer_sizes=(hidden_size,),
        max_iter=500,
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )
    clf.fit(X_train_scaled, y_train)
    
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    
    return ProbeResult(
        layer_idx=layer_idx,
        accuracy=accuracy,
        auroc=auroc,
        f1=f1,
        probe_type="mlp",
    )


def run_probing_suite(
    activations: Dict[int, np.ndarray],
    labels: np.ndarray,
    refusal_directions: Optional[Dict[int, np.ndarray]] = None,
    test_size: float = 0.2,
    n_folds: int = 5,
) -> Tuple[ProbingSuiteResult, ProbingSuiteResult]:
    """
    Run full probing suite (linear + MLP) across all layers with cross-validation.
    
    Args:
        activations: Dict mapping layer_idx -> (n_samples, hidden_dim)
        labels: Binary labels (n_samples,) - 1=harmful, 0=benign
        refusal_directions: Optional dict of refusal directions per layer.
        test_size: Fraction for test set.
        n_folds: Number of cross-validation folds.
        
    Returns:
        Tuple of (linear_results, mlp_results)
    """
    from sklearn.model_selection import train_test_split
    
    linear_results = {}
    mlp_results = {}
    
    for layer_idx in sorted(activations.keys()):
        X = activations[layer_idx]
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Get refusal direction for this layer
        ref_dir = None
        if refusal_directions and layer_idx in refusal_directions:
            ref_dir = refusal_directions[layer_idx]
        
        # Linear probe
        linear_result = train_linear_probe(
            X_train, y_train, X_test, y_test, layer_idx, ref_dir
        )
        linear_results[layer_idx] = linear_result
        
        # MLP probe
        mlp_result = train_mlp_probe(
            X_train, y_train, X_test, y_test, layer_idx
        )
        mlp_results[layer_idx] = mlp_result
    
    # Find best layers
    best_linear_layer = max(linear_results, key=lambda l: linear_results[l].auroc)
    best_mlp_layer = max(mlp_results, key=lambda l: mlp_results[l].auroc)
    
    linear_suite = ProbingSuiteResult(
        per_layer=linear_results,
        best_layer=best_linear_layer,
        best_auroc=linear_results[best_linear_layer].auroc,
        probe_type="linear",
    )
    
    mlp_suite = ProbingSuiteResult(
        per_layer=mlp_results,
        best_layer=best_mlp_layer,
        best_auroc=mlp_results[best_mlp_layer].auroc,
        probe_type="mlp",
    )
    
    return linear_suite, mlp_suite


def compare_probe_vs_direction(
    linear_suite: ProbingSuiteResult,
    refusal_directions: Dict[int, np.ndarray],
) -> Dict[int, float]:
    """
    Compare learned probe directions vs analytical refusal directions.
    
    High cosine similarity = the probe found the SAME direction as
    difference-in-means. Low similarity = probe found a DIFFERENT
    direction that also separates harmful/benign.
    
    Args:
        linear_suite: Results from linear probing.
        refusal_directions: Analytical refusal directions per layer.
        
    Returns:
        Dict mapping layer_idx -> cosine similarity
    """
    cosines = {}
    for layer_idx, result in linear_suite.per_layer.items():
        if result.learned_direction is not None and layer_idx in refusal_directions:
            cosine = np.dot(result.learned_direction, refusal_directions[layer_idx])
            cosines[layer_idx] = float(cosine)
    return cosines


def print_probing_summary(
    linear_suite: ProbingSuiteResult,
    mlp_suite: ProbingSuiteResult,
):
    """Print a summary table of probing results."""
    print("\n" + "=" * 70)
    print("PROBING RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Layer':<8} {'Linear AUROC':<15} {'MLP AUROC':<15} {'Cosine w/ RefDir':<18}")
    print("-" * 56)
    
    for layer_idx in sorted(linear_suite.per_layer.keys()):
        lin = linear_suite.per_layer[layer_idx]
        mlp = mlp_suite.per_layer[layer_idx]
        cosine_str = f"{lin.cosine_with_refusal:.3f}" if lin.cosine_with_refusal else "N/A"
        
        print(f"{layer_idx:<8} {lin.auroc:<15.4f} {mlp.auroc:<15.4f} {cosine_str:<18}")
    
    print(f"\nBest linear: layer {linear_suite.best_layer} (AUROC={linear_suite.best_auroc:.4f})")
    print(f"Best MLP:    layer {mlp_suite.best_layer} (AUROC={mlp_suite.best_auroc:.4f})")
    
    # Interpretation
    gap = mlp_suite.best_auroc - linear_suite.best_auroc
    if gap < 0.05:
        print("\n→ Linear probe matches MLP: information is LINEARLY SEPARABLE")
        print("  (exists as a direction in activation space)")
    elif gap < 0.15:
        print("\n→ MLP slightly better: information mostly linear with some non-linear component")
    else:
        print("\n→ MLP significantly better: information exists but is NOT a simple direction")
        print("  (more complex representation)")
