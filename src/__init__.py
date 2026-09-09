"""
Agentic Safety Probe - Source Package

Investigates how agentic formatting (tool-use context) affects
the activation of safety mechanisms (refusal direction) in LLMs.
"""

from src.model_loader import (
    load_model_and_tokenizer,
    format_chat_prompt,
    format_agent_prompt,
    format_agent_prompt_with_history,
    format_role_only_prompt,
    format_role_plus_tools_prompt,
    TOOL_DEFINITIONS,
)
from src.activation_extractor import ActivationExtractor
from src.refusal_direction import (
    compute_refusal_direction,
    compute_refusal_direction_pca,
    project_onto_direction,
    full_refusal_analysis,
    compute_gap_by_layer,
)
from src.probes import (
    train_linear_probe,
    train_mlp_probe,
    cross_validate_probe,
    run_probing_suite,
    compare_probe_vs_direction,
    print_probing_summary,
    ProbeResult,
    ProbingSuiteResult,
)
from src.robust_estimation import (
    cross_fitted_projection_gap,
    cross_validated_agent_direction,
    cross_validated_monitor,
    crossfit_condition_projections,
    held_out_detection,
    paired_bootstrap_ci,
    paired_permutation_test,
    paired_sign_test,
    run_paired_suite,
    select_layer_and_test,
)
from src.format_effect import (
    format_effect_specificity,
    format_effect_scale,
    detection_by_stratum,
    specificity_from_projections,
    scale_from_projections,
    stratum_detection_from_projections,
)
from src.metrics import (
    paired_ttest,
    permutation_test,
    bootstrap_ci,
    compute_auroc_by_format,
    decompose_format_effects,
)
from src.validation import (
    validate_activations_paired,
    validate_behavior,
    validate_concordance,
)
from src.intervention import ActivationIntervention
from src.visualizations import generate_all_figures

__version__ = "0.1.0"
__author__ = "JEstebanBav"
