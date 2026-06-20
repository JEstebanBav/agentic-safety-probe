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
    TOOL_DEFINITIONS,
)
from src.activation_extractor import ActivationExtractor
from src.refusal_direction import (
    compute_refusal_direction,
    compute_refusal_direction_pca,
    project_activations,
)
from src.probes import LinearProbe, MLPProbe, run_probing_suite
from src.metrics import (
    paired_ttest,
    permutation_test,
    bootstrap_ci,
    compute_auroc_by_format,
)
from src.intervention import ActivationIntervention
from src.visualizations import generate_all_figures

__version__ = "0.1.0"
__author__ = "JEstebanBav"
