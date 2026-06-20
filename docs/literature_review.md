# Literature Review

## Core Papers (Our Novel Connection)

### Paper 1: AgentHarm
**Andriushchenko et al. (2024)** — "AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents"

**Key Findings:**
- Models comply with 48-84% of harmful agentic tasks
- No jailbreak needed — just having tools available is sufficient
- Per-tool defenses are insufficient

**What they did NOT explain:**
- WHY does the model comply in agentic format?
- What happens internally?

→ **THIS IS THE GAP WE FILL**

---

### Paper 2: Refusal in Language Models Is Mediated by a Single Direction
**Arditi et al. (2024)**

**Key Findings:**
- Refusal is encoded in ONE direction in the activation space
- Removing it (abliteration) eliminates refusal
- Adding it induces refusal on benign prompts

**What they did NOT test:**
- Does it work the same in agentic format?
- Does the direction activate when tool definitions are present?

→ **THIS IS WHAT WE MEASURE**

---

### Paper 3: Alignment Faking in Large Language Models
**Greenblatt et al. (2024)**

**Relevance:**
- Models can fake alignment during evaluations
- Purely behavioral evaluations are insufficient
- Reinforces the need to look INSIDE the model

→ **JUSTIFIES our mechanistic approach vs. behavioral-only**

---

### Paper 4: Scaling Monosemanticity
**Anthropic (Templeton et al., 2024)**

**Relevance:**
- Interpretable features can be extracted from production models
- Validates that internal representations are auditable
- Features like "deception", "sycophancy" were found

→ **VALIDATES that searching for "harmful intent" as an internal feature is methodologically viable**

---

## Methodological Support Papers

### Paper 5: Inference-Time Intervention
**Li et al. (2024)**
- Intervene on activations during inference
- Basis for our proposed solution
- If the direction deactivates, ITI can restore it

### Paper 6: Representation Engineering
**Zou et al. (2023)**
- General framework for reading/writing concepts in model representations
- Theoretical context for our approach

### Paper 7: Linear Probing
**Alain & Bengio (2017)** — "Understanding Intermediate Layers Using Linear Classifiers"
- Foundation for using linear probes as separability tests
- Justifies our choice of linear vs. MLP probes

---

## Differentiation from Existing Work

| Existing Work | What They Do | How We Differ |
|---|---|---|
| Abliteration repos (OBLITERATUS, etc.) | REMOVE the refusal direction to make "uncensored" models | We STUDY the mechanism in a new context (agentic) |
| SAE-based refusal analysis | Evaluate refusal direction with SAEs | Only in chat. Not in agentic context. |
| Cross-lingual refusal studies | Study if direction works cross-linguistically | They change LANGUAGE. We change FORMAT (chat vs agent). Orthogonal angle. |
| AgentHarm | Behavioral measurement of agent compliance | No mechanistic explanation. We provide the WHY. |
