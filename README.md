# 🔬 Agentic Safety Probe

**Investigating how agentic formatting (tool-use context) affects the activation of safety mechanisms in Large Language Models.**

This project provides a mechanistic interpretability framework to study whether the "refusal direction" — the internal representation that drives safety refusal — deactivates when models operate in agentic format (with tool definitions and multi-turn history).

---

## 📋 Research Question

> Does the presence of agentic scaffolding (tool definitions, system prompts, JSON output format) cause the refusal direction to deactivate in the model's residual stream, making harmful requests more likely to be complied with?

> **And critically**: Which specific component of agentic formatting is responsible? The role framing? The tool definitions? Or the structured output format?

## 🧠 Core Hypothesis

When a harmful prompt is presented in **agentic format** (with tool definitions), its projection onto the refusal direction is significantly **lower** than the same prompt in standard **chat format** — indicating weakened safety activation.

---

## 🏗️ Architecture

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

## 🚀 Quick Start

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

## 📊 Experimental Design

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

1. **Load Model** — Mistral-7B-Instruct (auto-detects FP16 vs 4-bit based on VRAM)
2. **Load Dataset** — Paired prompts from `dataset_full.jsonl`
3. **Extract Activations** — Hook into residual stream at each layer
4. **Compute Refusal Direction** — Difference-in-means between harmful/benign chat activations
5. **PCA Validation** — Confirm direction aligns with principal component (cosine > 0.8)
6. **Project & Compare** — Compute ΔP with statistical significance:
   - Permutation test (10,000 permutations, non-parametric)
   - Welch's t-test (parametric comparison)
   - Cohen's d (effect size)
   - Bootstrap 95% CI
7. **Decomposition** (optional) — Break total effect into role / tools / format components
8. **Probing** — Linear and MLP classifiers to verify direction quality
9. **Intervention** — Add refusal direction back at inference time
10. **Visualize** — Waterfall decomposition, projections, AUROC curves

---

## 📈 Key Metrics

| Metric | Description |
|--------|-------------|
| **ΔP** (Projection Gap) | `mean(proj_chat) - mean(proj_agent)` — main metric |
| **p-value (permutation)** | Non-parametric significance (no distribution assumptions) |
| **Cohen's d** | Effect size (>0.8 large, >0.5 medium, >0.2 small) |
| **PCA cosine similarity** | Validates diff-in-means direction against PC1 |
| **Decomposition** | `role_effect + tools_effect + json_format_effect = total_effect` |
| **AUROC** | Separability of harmful/benign in each format |

---

## 🖥️ Hardware Requirements

| Configuration | VRAM | Cost (RunPod) |
|--------------|------|---------------|
| 4-bit quantized | ~8 GB | RTX 3090 ($0.46/hr) |
| FP16 (recommended) | ~14 GB | RTX A5000 ($0.27/hr) |
| FP16 + intervention | ~18 GB | RTX 4090 ($0.69/hr) |

**Budget estimate**: Full experiment (~35 min) costs ~$0.20 on RTX A5000.

---

## 🔑 Expected Findings

Based on prior work (Arditi et al. 2024):

1. **Projection Gap (ΔP > 0)**: Agent-format harmful prompts have lower projection onto refusal direction
2. **Statistical Significance**: p < 0.05 (permutation test), Cohen's d > 0.5
3. **Layer Specificity**: Effect concentrates in middle layers (L12-L20)
4. **Decomposition**: Identifies whether role, tools, or JSON format drives the effect
5. **PCA Alignment**: cosine > 0.8 between diff-in-means and PC1

---

## 📚 References

- Arditi, A. et al. (2024). "Refusal in Language Models Is Mediated by a Single Direction"
- Andriushchenko, M. et al. (2024). "AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents"
- Zou, A. et al. (2023). "Representation Engineering"
- Turner, A. et al. (2023). "Activation Addition"

---

## ⚠️ Ethical Note

This research is conducted to **understand and improve** AI safety mechanisms. The harmful prompts in the dataset are designed solely to test refusal systems and should not be used for any malicious purpose. The goal is to identify vulnerabilities so they can be patched.

---

## 📄 License

MIT

---

## 👤 Author

**JEstebanBav** — Research on mechanistic interpretability and AI safety in agentic systems.
