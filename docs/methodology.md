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
| Benign arm measured with the same statistics | A shift seen on harmful prompts alone cannot be told apart from a rigid displacement of every representation |
| Fitting and scoring prompts disjoint | An AUROC scored on the prompts that defined d_chat is a fit statistic, not a detection rate |

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

### Step 2: Projection and Gap, with the benign control
```
DeltaP_harmful = mean(proj_chat_harmful) - mean(proj_agent_harmful)
DeltaP_benign  = mean(proj_chat_benign)  - mean(proj_agent_benign)
interaction    = DeltaP_harmful - DeltaP_benign
```
The harmful shift alone is compatible with three mechanisms: refusal genuinely
weakens, the agentic context displaces *every* representation along d_chat by a
common offset, or the projection scale contracts. The benign arm is therefore
measured with the identical paired statistics, and the **interaction** is the
part that is specific to harmfulness. `common_shift_fraction` reports how much
of the headline number the benign arm reproduces.

The interaction is computed at every layer, since a shift common to both arms is
a property of the format and should be visible as such across depth. The layer
that maximises DeltaP_harmful and the layer that maximises the interaction need
not coincide; when they differ, the run says so and the reported layer states
which criterion selected it.

### Step 3: Statistical Significance
- Permutation test (10,000 sign-flip permutations, paired)
- Paired t-test, one-sided and sign-aware
- Cohen's d (paired and pooled)
- Bootstrap 95% CI (10,000 paired resamples, percentile + BCa)
- Every one of the above is run on the **benign arm** as well; the
  difference-in-differences carries its own permutation test and bootstrap CI

### Step 3b: Held-out evaluation of d_chat
d_chat is the difference between the chat_harmful and chat_benign centroids, so
asking how well it separates those two groups asks how well a line fits the
points it was fitted to. At n = 42 in d = 8192 the answer is flattering by
construction, and it does not transfer to the agent format, which never entered
the estimation — so the chat-vs-agent comparison inherits an asymmetry that has
nothing to do with format.

The prompt PAIRS are split into K folds; d_k is fitted on K-1 folds and only the
held-out fold is scored. Folds are assigned over pairs, so a prompt's harmful and
benign variants — built from the same seed — are always held out together.
Every prompt still contributes a score, which is why this is preferred over a
single train/test split at this n.

Both AUROCs are reported: in-sample and held-out. Their difference is the
optimism, quoted explicitly rather than corrected away. The held-out AUROCs
carry a paired bootstrap CI over prompt pairs, and so does their chat-minus-agent
difference. The specificity, scale and per-stratum analyses are re-run on the
out-of-fold scores; the per-stratum ones matter most, because each stratum holds
a handful of prompts and the optimism is proportionally largest exactly there.

Disable with `--skip-heldout` (debugging only).

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
| Permutation test | Non-parametric significance | 10,000 sign-flip permutations, paired |
| Paired t-test | Parametric comparison | One-sided, sign-aware |
| Cohen's d | Effect size | >0.8 large, >0.5 medium, >0.2 small |
| Bootstrap CI | Confidence interval | 10,000 paired resamples, percentile + BCa |
| PCA validation | Direction quality | PC1 cosine with d_chat > 0.8 |
| Difference-in-differences | Is the shift specific to harm? | Permutation on within-pair shifts + bootstrap CI on the interaction |
| Cross-fitted AUROC | Is detection real or a fit? | K-fold over prompt PAIRS; optimism = in-sample minus held-out |

## Possible Outcomes

### Outcome A: DeltaP > 0, significant, moderate-large d, interaction > 0
The agentic format weakens the refusal direction activation, and the benign
control confirms the weakening is specific to harmful requests. The effect is
measurable and consistent. Safety monitor using w_agent can detect harmful
intent.

### Outcome A': DeltaP > 0 but the interaction vanishes
Harmful and benign prompts shift by the same amount. This is a change of ORIGIN,
not a weakened refusal representation, and DeltaP must not be reported as a loss
of safety representation. The run states this verdict explicitly rather than
leaving it to be inferred from two numbers.

### Outcome B: DeltaP ~ 0, AUROC agent ~ 0.5
The refusal direction from chat does NOT transfer to agent context. But if a linear probe achieves high AUROC on agent data, the information exists in a different direction (w_agent != d_chat). This is the "direction rotation" scenario.

### Outcome C: DeltaP ~ 0, AUROC agent high, probe also fails
The model genuinely does not encode harmfulness in agentic context. This would be the most concerning finding -- no simple monitor is possible.
