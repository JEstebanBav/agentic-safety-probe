# Methodology

## Experimental Design

**Type:** Within-subjects, repeated measures

**Factor:** Format (chat vs. agent) — within-subjects
- Each prompt appears in BOTH formats
- Eliminates variance due to prompt content

**Measure:** Projection onto refusal direction (continuous)

## Controls

| Control | Purpose |
|---|---|
| Positive control: chat_harmful | Confirm the direction activates (known ground truth) |
| Negative control: chat_benign + agent_benign | Confirm the direction does NOT activate |
| Format control: agent_benign vs chat_benign | Isolate format effect from harmful content effect |

## Two Execution Scenarios

### Scenario 1: Local Model (GPU Available)
- Model: Mistral-7B-Instruct-v0.3 (4-bit quantization)
- Framework: TransformerLens + HuggingFace Transformers
- Full activation extraction capability
- Complete experimental pipeline

### Scenario 2: API Only (No GPU)
- Model: GPT-4o / Claude via API
- Proxy measurements: compliance rate + logprobs
- Behavioral evaluation with multilingual angle

## Analysis Pipeline

### Primary Analysis
```
ΔP = mean(projection_chat_harmful) - mean(projection_agent_harmful)
```

### Secondary Analysis
- AUROC by layer and format
- Linear and non-linear probes per layer

### Exploratory Analysis
- t-SNE/UMAP visualization
- Per-tool and per-harm-category breakdown

## Statistical Tests

1. **Paired t-test**: H₀: projection_chat = projection_agent
2. **Cohen's d**: Effect size (d > 0.8 = large)
3. **Bootstrap CI**: 1000 resamples, 95% confidence interval
4. **Permutation test**: 10,000 randomized label assignments

## Possible Outcomes and Interpretations

### Outcome A: Direction Deactivates (ΔP >> 0)
- The agentic format literally prevents the model from "detecting" harm
- Solution: retrain/strengthen direction for agentic format
- Or: inference-time intervention to restore it

### Outcome B: Direction Active but Ignored (ΔP ≈ 0)
- The model detects harm but generates tool call anyway
- Refusal mechanism and tool-call generation are SEPARATE circuits
- More concerning — requires different intervention strategy
