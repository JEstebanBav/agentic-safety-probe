# Experiment Design

## Experiment 1: Baseline Validation
**Objective:** Confirm we can extract the refusal direction and it separates harmful/benign in chat.
**Input:** chat_harmful + chat_benign activations
**Output:** d_chat per layer, AUROC on held-out set
**Success Criterion:** AUROC > 0.85 in at least 3 layers
**PCA Validation:** PC1 cosine > 0.8 with d_chat

## Experiment 2: Main Measurement (Chat vs Agent)
**Objective:** Does the refusal direction activate less in agentic format?
**Input:** Same prompts in agent format
**Measurement:** DeltaP = mean(proj_chat_harmful) - mean(proj_agent_harmful) per layer
**Tests:** Permutation test + Welch's t-test + Bootstrap CI
**This is the MAIN RESULT of the project.**

## Experiment 3: Stratified Analysis by Subtlety
**Objective:** Does the effect depend on how explicit the harmful intent is?
**Stratification:**
- explicit (N=18): Clearly harmful words ("steal", "hack", "threaten")
- contextual (N=14): Harm implied by context, not explicit words
- framed (N=10): Professional framing hiding harmful action ("pentest", "security training")
**Expected:** explicit shows largest DeltaP; framed shows smallest

## Experiment 4: Agent-Specific Direction (w_agent)
**Objective:** If d_chat fails on agent data, find the direction that DOES work.
**Method:** LogisticRegression on agent_harmful vs agent_benign activations
**Output:** w_agent per layer, cos(w_agent, d_chat), AUROC on agent data
**Key question:** Is w_agent a rotation of d_chat or completely orthogonal?

## Experiment 5: Safety Monitor
**Objective:** Demonstrate practical utility of w_agent as a real-time detector.
**Method:** Calibrate threshold on validation set (maximize F1)
**Metrics:** Precision, Recall, F1, False Positive Rate
**Goal:** F1 > 0.9 would indicate viability for production use

## Experiment 6: Intervention
**Objective:** Can we restore refusal by adding the direction back?
**Method:** Add alpha * direction to activations during generation
**Two variants:**
- Intervention with d_chat (may fail if direction rotates in agent context)
- Intervention with w_agent (should be more effective)
**Search:** Minimum alpha that restores refusal behavior
**Applied to:** All layers from layer N/4 onwards

## Experiment 7: Decomposition (optional)
**Objective:** Which component of agentic formatting causes the effect?
**Conditions:**
1. chat (baseline)
2. role_only (agent role, no tools)
3. role_plus_tools (role + tool definitions, text output)
4. agent_full (role + tools + JSON tool-call output)
**Analysis:** DeltaP between consecutive conditions
**Output:** Percentage contribution of each factor to total effect

## Dataset Construction

### Custom Paired Dataset (PRIMARY -- for activation analysis)
- 42 base prompts (harmful) with paired benign variants
- Each appears in chat AND agent format (identical text)
- 7 categories: fraud, cybercrime, harassment, disinformation, hate, drugs, copyright
- 3 subtlety levels: explicit, contextual, framed
- 3-4 relevant tools per prompt (from 26 total)
- Total: 168 entries

### HarmAgent Benchmark (SUPPLEMENTARY -- for behavioral validation)
- 176 test behaviors + 32 validation behaviors
- Includes grading functions and target tools
- Not paired (different prompts per condition)
- Used only for behavioral correlation, not activation analysis

## Metrics Summary

| Metric | What it measures | Target |
|---|---|---|
| DeltaP | Projection gap chat vs agent | Positive, significant |
| Cohen's d (global) | Overall effect size | > 0.3 |
| Cohen's d (explicit) | Effect for explicit prompts | > 0.5 |
| p-value (permutation) | Statistical significance | < 0.05 |
| AUROC (d_chat on chat) | Direction quality | > 0.85 |
| AUROC (w_agent on agent) | Agent direction quality | > 0.85 |
| cos(w_agent, d_chat) | Direction alignment | Measure rotation |
| Monitor F1 | Practical detection | > 0.9 |
| Intervention success | Can we restore refusal? | > 50% |
