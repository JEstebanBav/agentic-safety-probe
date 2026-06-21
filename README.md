# Agentic Safety Probe

**Investigating how agentic formatting (tool-use context) affects the activation of safety mechanisms in Large Language Models.**

This project provides a mechanistic interpretability framework to study whether the "refusal direction" — the internal representation that drives safety refusal — deactivates when models operate in agentic format (with tool definitions and multi-turn history).

---

## Research Question

> Does the presence of agentic scaffolding (tool definitions, system prompts, JSON output format) cause the refusal direction to deactivate in the model's residual stream, making harmful requests more likely to be complied with?

> **And critically**: Which specific component of agentic formatting is responsible? The role framing? The tool definitions? Or the structured output format?

## Core Hypothesis

When a harmful prompt is presented in **agentic format** (with tool definitions), its projection onto the refusal direction is significantly **lower** than the same prompt in standard **chat format** — indicating weakened safety activation.

---

## Architecture

```
agentic-safety-probe/
├── src/
│   ├── __init__.py              # Package exports
│   ├── model_loader.py          # Model loading + prompt formatting (4 conditions)
│   ├── activation_extractor.py  # Hook-based activation extraction from residual stream
│   ├── refusal_direction.py     # Refusal direction: diff-in-means + PCA validation
│   ├── probes.py                # Linear & MLP probing classifiers
│   ├── metrics.py               # Statistical tests + effect decomposition
│   ├── validation.py            # Activation-space & behavioral validation
│   ├── intervention.py          # Inference-time intervention to restore refusal
│   └── visualizations.py        # All figures (waterfall, violin, AUROC, PCA, etc.)
├── data/
│   ├── __init__.py              # Data package exports
│   ├── loader.py                # Unified data loaders (custom + HarmAgent)
│   ├── build_dataset.py         # Custom paired dataset builder
│   ├── dataset_full.jsonl       # 168 entries: same prompts in chat/agent (42 base × 4)
│   ├── *_behaviors_*.json       # HarmAgent benchmark datasets
│   └── chat_*.json              # HarmAgent chat-only subsets
├── tool_definitions.json        # 26 canonical tool schemas
├── run_experiment.py            # Main CLI pipeline
├── run_experiment_runpod.ipynb  # Notebook for cloud execution (RunPod)
├── runpod_setup.sh             # Cloud setup script
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full experiment

```bash
# Standard analysis (chat vs agent)
python run_experiment.py --dataset custom

# With effect decomposition (4 conditions)
python run_experiment.py --dataset custom --decompose

# Quick test run
python run_experiment.py --dataset custom --n-samples 5 --layers 10,14,18,22
```

### 3. Cloud Execution (RunPod)

For GPU access without local hardware:

```bash
# Upload project ZIP to RunPod JupyterLab, then:
cd /workspace
unzip agentic-safety-probe-runpod.zip
pip install -r requirements.txt
# Open run_experiment_runpod.ipynb
```

Recommended GPU: RTX A5000 ($0.27/hr) or RTX 4090 ($0.69/hr) — 24GB VRAM.

---

## Experimental Design

### Dataset: Paired Contrastive Design

The custom dataset (`dataset_full.jsonl`) uses **identical prompts** across conditions, enabling direct paired comparison of activations:

| | **Harmful Prompt** | **Benign Prompt** |
|---|---|---|
| **Chat Format** | chat_harmful | chat_benign |
| **Agent Format** | agent_harmful | agent_benign |

- **42 base prompts** × 4 variants = 168 entries
- **7 harm categories**: fraud, cybercrime, harassment, disinformation, hate, drugs, copyright
- **3 subtlety levels**: explicit, contextual, framed
- **26 realistic tools** (email, search, terminal, crypto, social media, etc.)

### Decomposition Conditions (with `--decompose`)

To isolate which factor causes the effect, 4 intermediate conditions are tested:

```
chat → role_only → role_plus_tools → agent_full
```

| Condition | Role | Tools in context | JSON output |
|-----------|------|-----------------|-------------|
| `chat` | Assistant | ❌ | ❌ |
| `role_only` | Autonomous agent | ❌ | ❌ |
| `role_plus_tools` | Autonomous agent | ✅ | ❌ |
| `agent_full` | Autonomous agent | ✅ | ✅ |

### Pipeline Steps

1. **Load Model** — Llama-3.1-70B or Mistral-7B (auto-detects FP16 vs 4-bit)
2. **Load Dataset** — Paired prompts from `dataset_full.jsonl`
3. **Extract Activations** — Hook into residual stream at each layer
4. **Compute Refusal Direction (d_chat)** — Difference-in-means + PCA validation
5. **Project and Compare** — Compute DeltaP with statistical significance
6. **Stratified Analysis** — Break down DeltaP by subtlety (explicit/contextual/framed)
7. **Probing** — Linear and MLP classifiers; extract agent-specific direction (w_agent)
8. **Agent Direction Analysis** — cos(w_agent, d_chat), AUROC, safety monitor calibration
9. **Intervention** — Add direction back during generation (with d_chat and w_agent)
10. **Decomposition** (optional) — role / tools / format components
11. **Visualize** — All figures

---

## Key Metrics

| Metric | Description |
|--------|-------------|
| **DeltaP** (Projection Gap) | `mean(proj_chat) - mean(proj_agent)` -- main metric |
| **p-value (permutation)** | Non-parametric significance (10,000 permutations) |
| **Cohen's d (global)** | Overall effect size |
| **Cohen's d (stratified)** | Effect size per subtlety level |
| **cos(w_agent, d_chat)** | How much the refusal direction rotates in agent context |
| **AUROC (w_agent)** | Agent-specific harmful/benign separability |
| **Monitor F1** | Safety monitor precision/recall on agent data |
| **Intervention success** | Percentage of prompts where adding direction restores refusal |

---

## Hardware Requirements

| Configuration | VRAM | Cost (RunPod) |
|--------------|------|---------------|
| 4-bit quantized | ~8 GB | RTX 3090 ($0.46/hr) |
| FP16 (recommended) | ~14 GB | RTX A5000 ($0.27/hr) |
| FP16 + intervention | ~18 GB | RTX 4090 ($0.69/hr) |

**Budget estimate**: Full experiment (~35 min) costs ~$0.20 on RTX A5000.

---

## Findings (Llama-3.1-70B-Instruct)

Results from running with `--layers 10,20,30,40,50,60`:

1. **Projection Gap**: DeltaP = 0.99 (significant, p = 0.005) -- refusal direction IS weaker in agent format
2. **Effect Size**: Cohen's d = 0.42 (global, small-medium). Heterogeneous by subtlety -- explicit prompts show larger effect
3. **AUROC**: d_chat achieves 0.96 in both chat and agent (direction transfers with reduced magnitude)
4. **Probing**: Linear probes achieve AUROC 1.0 from layer 20 onwards in agent data
5. **Direction Rotation**: cos(w_agent, d_chat) ~ 0.4 -- the model partially rotates the harmfulness encoding in agent context
6. **Best Layer**: Layer 60 (last layers) show strongest gap

## Interpretation

The refusal direction does not completely deactivate in agent format (AUROC remains high). Instead, the magnitude of its activation decreases. Additionally, the model encodes harmfulness in a partially rotated direction (w_agent) when in agentic context. This means:
- A safety monitor based on d_chat alone will have reduced sensitivity in agent mode
- A monitor using w_agent achieves near-perfect detection
- The rotation is quantifiable and layer-dependent

---

## References

- Arditi, A. et al. (2024). "Refusal in Language Models Is Mediated by a Single Direction"
- Andriushchenko, M. et al. (2024). "AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents"
- Zou, A. et al. (2023). "Representation Engineering"
- Turner, A. et al. (2023). "Activation Addition"

---

## Ethical Note

This research is conducted to understand and improve AI safety mechanisms. The harmful prompts in the dataset are designed solely to test refusal systems and should not be used for any malicious purpose. The goal is to identify vulnerabilities so they can be patched.

---

## License

MIT

---

## Author

**JEstebanBav** — Research on mechanistic interpretability and AI safety in agentic systems.
