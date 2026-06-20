# 🔬 Agentic Safety Probe

**Investigating how agentic formatting (tool-use context) affects the activation of safety mechanisms in Large Language Models.**

This project provides a mechanistic interpretability framework to study whether the "refusal direction" — the internal representation that drives safety refusal — deactivates when models operate in agentic format (with tool definitions and multi-turn history).

---

## 📋 Research Question

> Does the presence of agentic scaffolding (tool definitions, system prompts, multi-turn history) cause the refusal direction to deactivate in the model's residual stream, making harmful requests more likely to be complied with?

## 🧠 Core Hypothesis

When a harmful prompt is presented in **agentic format** (with tool definitions), its projection onto the refusal direction is significantly **lower** than the same prompt in standard **chat format** — indicating weakened safety activation.

---

## 🏗️ Architecture

```
agentic-safety-probe/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── model_loader.py          # Model loading + prompt formatting (chat vs agent)
│   ├── activation_extractor.py  # Hook-based activation extraction from residual stream
│   ├── refusal_direction.py     # Compute refusal direction (difference-in-means / PCA)
│   ├── probes.py                # Linear & MLP probing classifiers
│   ├── metrics.py               # Statistical tests (t-test, permutation, bootstrap, AUROC)
│   ├── intervention.py          # Inference-time intervention to restore refusal
│   └── visualizations.py        # All figures (violin plots, AUROC curves, PCA, etc.)
├── data/
│   └── build_dataset.py         # Dataset builder (harmful + benign × chat + agent)
├── run_experiment.py            # Main pipeline orchestrator
├── requirements.txt             # Dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full experiment

```bash
python run_experiment.py --model mistralai/Mistral-7B-Instruct-v0.3
```

### 3. With options

```bash
# Use Llama 3.1 without quantization (needs ~16GB VRAM)
python run_experiment.py --model meta-llama/Llama-3.1-8B-Instruct --no-quantize

# Skip slow steps for quick iteration
python run_experiment.py --skip-intervention --skip-probes

# Analyze specific layers
python run_experiment.py --layers 10,12,14,16,18,20

# Include multi-turn agentic history
python run_experiment.py --include-history

# Fewer samples for testing
python run_experiment.py --n-samples 5
```

---

## 📊 Experimental Design

### Conditions (2×2 factorial)

| | **Harmful Prompt** | **Benign Prompt** |
|---|---|---|
| **Chat Format** | chat_harmful (positive control) | chat_benign (negative control) |
| **Agent Format** | agent_harmful (experimental) | agent_benign (negative control) |

### Pipeline Steps

1. **Load Model** — 4-bit quantized for memory efficiency
2. **Build Dataset** — 80 harmful prompts (4 categories × 20) + 80 benign prompts (4 categories × 20) = 320 total entries (× 2 formats)
3. **Extract Activations** — Hook into residual stream at each layer
4. **Compute Refusal Direction** — Difference-in-means between harmful/benign chat activations
5. **Project & Compare** — Project agent activations onto refusal direction; compute ΔP
6. **Statistical Tests** — Paired t-test, permutation test, bootstrap CI
7. **Probing** — Linear and MLP classifiers to verify direction quality
8. **Intervention** — Add refusal direction back at inference time; measure restored refusal
9. **Visualize** — Generate all paper figures

### Harm Categories
- **Fraud** — Phishing, scams, impersonation
- **Cybercrime** — Malware, hacking tools, exploits
- **Harassment** — Threats, doxxing, stalking
- **Disinformation** — Fake news, propaganda, conspiracy theories

### Tools Simulated
- `web_search` — Web information retrieval
- `send_email` — Email composition and sending
- `execute_code` — Code execution in sandbox
- `file_operations` — Filesystem read/write/delete
- `database_query` — SQL query execution
- `post_social_media` — Social media posting

---

## 📈 Key Metrics

| Metric | Description |
|--------|-------------|
| **ΔP** (Projection Gap) | Difference in mean projection between chat and agent format |
| **Cohen's d** | Effect size of the gap |
| **AUROC** | Separability of harmful/benign in each format |
| **Intervention Success Rate** | % of prompts where adding direction restores refusal |
| **Minimum α** | Smallest scaling factor that restores refusal |

---

## 📂 Output Structure

After running the experiment:

```
results/
├── results.json              # All numerical results
├── dataset.jsonl             # Generated dataset
└── figures/
    ├── projection_distributions_layerN.png/.pdf
    ├── layer_auroc_curve.png/.pdf
    ├── gap_by_layer.png/.pdf
    ├── pca_activations_layerN.png/.pdf
    ├── intervention_dose_response.png/.pdf
    ├── cross_tool_gap.png/.pdf
    └── cosine_probe_vs_refusal.png/.pdf
```

---

## 🖥️ Hardware Requirements

| Configuration | VRAM | Notes |
|--------------|------|-------|
| 4-bit quantized (default) | ~8 GB | Works on RTX 3060/4060+ |
| Full precision | ~16 GB | Needs RTX 4090 / A100 |
| With intervention | +2 GB | Generation requires more memory |

---

## 🔑 Key Findings (Expected)

Based on prior work (Arditi et al. 2024, refusal in residual stream):

1. **Projection Gap**: Agent-format harmful prompts have lower projection onto refusal direction than chat-format (~0.1-0.3 lower)
2. **Layer Specificity**: Effect concentrates in middle layers (L12-L20 for 32-layer models)
3. **Tool Independence**: Gap persists across different tool types
4. **Intervention Works**: Adding α × d_refusal at the critical layer restores refusal behavior

---

## 📚 References

- Arditi, A. et al. (2024). "Refusal in Language Models Is Mediated by a Single Direction"
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
