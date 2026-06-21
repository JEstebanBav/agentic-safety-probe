# Methodology

## Experimental Design

**Type:** Within-subjects, repeated measures (paired)

**Factor:** Format (chat vs. agent) -- within-subjects
- Each prompt appears in BOTH formats with identical text
- Eliminates variance due to prompt content
- Enables paired statistical comparisons

**Measure:** Projection onto refusal direction (continuous scalar per sample per layer)

## Dataset

- **Source:** Custom paired dataset (dataset_full.jsonl)
- **42 base prompts** x 4 variants = 168 entries
- **7 harm categories:** fraud, cybercrime, harassment, disinformation, hate, drugs, copyright
- **3 subtlety levels:** explicit (18), contextual (14), framed (10)
- **26 tools** from tool_definitions.json (only 3-4 relevant tools per prompt)
- **Key property:** Identical user message in chat and agent conditions

## Controls

| Control | Purpose |
|---|---|
| chat_harmful | Confirm refusal direction activates (ground truth) |
| chat_benign | Confirm direction does NOT activate for benign content |
| agent_benign | Isolate format effect from harmful content effect |
| Paired design | Same prompt in both conditions eliminates content confound |

## Model

- **Primary:** Llama-3.1-70B-Instruct (RunPod, FP16)
- **Alternative:** Mistral-7B-Instruct-v0.3 (4-bit, local or cloud)
- Auto-detection of VRAM for quantization decision
- Activation extraction via forward hooks on residual stream

## Analysis Pipeline

### Step 1: Refusal Direction Extraction (d_chat)
```
d_chat = mean(activations_chat_harmful) - mean(activations_chat_benign)
```
Validated with PCA: PC1 must align with d_chat (cosine > 0.8).

### Step 2: Projection and Gap
```
DeltaP = mean(proj_chat_harmful) - mean(proj_agent_harmful)
```

### Step 3: Statistical Significance
- Permutation test (10,000 permutations, non-parametric)
- Welch's t-test (parametric comparison)
- Cohen's d (effect size)
- Bootstrap 95% CI (1,000 resamples)

### Step 4: Stratified Analysis
Break down DeltaP by subtlety level (explicit / contextual / framed) to explain heterogeneous effect sizes.

### Step 5: Agent-Specific Direction (w_agent)
Train LogisticRegression on agent activations, extract weight vector:
- Compare cos(w_agent, d_chat) per layer
- Evaluate AUROC of w_agent on agent data
- Calibrate safety monitor threshold (maximize F1)

### Step 6: Decomposition (optional, with --decompose)
4 intermediate conditions: chat -> role_only -> role_plus_tools -> agent_full
- role_effect = mean(chat) - mean(role_only)
- tools_effect = mean(role_only) - mean(role_plus_tools)
- json_format_effect = mean(role_plus_tools) - mean(agent_full)

### Step 7: Intervention
Add alpha * direction to activations during autoregressive generation:
- First with d_chat (baseline)
- Then with w_agent (agent-specific)
- Applied to multiple layers simultaneously
- Report minimum alpha that restores refusal

## Statistical Tests

| Test | Purpose | Details |
|---|---|---|
| Permutation test | Non-parametric significance | 10,000 permutations, two-sided |
| Welch's t-test | Parametric comparison | Does not assume equal variance |
| Cohen's d | Effect size | >0.8 large, >0.5 medium, >0.2 small |
| Bootstrap CI | Confidence interval | 1,000 resamples, 95% level |
| PCA validation | Direction quality | PC1 cosine with d_chat > 0.8 |

## Possible Outcomes

### Outcome A: DeltaP > 0, significant, moderate-large d
The agentic format weakens the refusal direction activation. The effect is measurable and consistent. Safety monitor using w_agent can detect harmful intent.

### Outcome B: DeltaP ~ 0, AUROC agent ~ 0.5
The refusal direction from chat does NOT transfer to agent context. But if a linear probe achieves high AUROC on agent data, the information exists in a different direction (w_agent != d_chat). This is the "direction rotation" scenario.

### Outcome C: DeltaP ~ 0, AUROC agent high, probe also fails
The model genuinely does not encode harmfulness in agentic context. This would be the most concerning finding -- no simple monitor is possible.
