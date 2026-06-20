# Experiment Design

## Experiment 1: Baseline Validation

**Objective:** Confirm we can extract the refusal direction and it works in chat (replicate Arditi et al.)

**Input:** chat_harmful + chat_benign (60% of dataset)

**Output:** d⃗_refusal per layer, AUROC on held-out set

**Success Criterion:** AUROC > 0.85 in at least 3 layers

**If it fails:** Review implementation before continuing

---

## Experiment 2: Main Measurement (Chat vs. Agent)

**Objective:** Does the refusal direction activate in agentic format?

**Input:** agent_harmful + agent_benign (test set)

**Measurement:** projection = a⃗_t · d⃗_refusal per layer

**Comparison:** Paired t-test against chat projections

**This is the MAIN RESULT of the project.**

---

## Experiment 3: Cross-Tool Generalization

**Objective:** Is the effect consistent across tools?

**Analysis:** ΔP grouped by tool_name

**Interpretation:**
- If consistent → "universal monitor" is viable
- If variable → effect depends on the tool

---

## Experiment 4: Layer-wise Probes

**Objective:** At which layers is harmful/benign info separable in agentic format?

**Method:** Linear probe + MLP at each layer

**Output:** Map of "where harm info exists" by format

---

## Experiment 5 (if time allows): Intervention

**Objective:** Can we restore refusal?

**Method:** Add α × d⃗_refusal to agentic activations

**Measure:** Does the model now refuse the tool call?

**Search:** Minimum α that restores refusal

**This converts the finding into a SOLUTION.**

---

## Dataset Construction

### Size
- 80 base prompts × 4 variants = 320 samples

### Harm Categories (from AgentHarm)
- Fraud/Scam (20 prompts)
- Cybercrime (20 prompts)
- Harassment (20 prompts)
- Disinformation (20 prompts)

### Tools (varied for generalization)
- send_email
- web_search
- execute_code
- post_social_media
- file_operations
- database_query

### Prompt Variants per Base Prompt
1. Chat harmful (positive control)
2. Agent harmful (experimental)
3. Chat benign (negative control)
4. Agent benign (negative control)

---

## Metrics

| Metric | What it measures | Target |
|---|---|---|
| Projection Gap (ΔP) | Difference in refusal direction activation | Statistically significant |
| AUROC by format | Separability of harmful/benign | Chat > 0.85, Agent = ? |
| Behavior-activation concordance | Does projection predict behavior? | High correlation |
| Cross-tool variance | Consistency of effect | Low variance |
| Cross-category variance | Consistency across harm types | Low variance |
