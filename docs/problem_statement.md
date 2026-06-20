# Problem Statement

## The Problem

LLM-based agentic systems (models with access to tools) execute harmful tasks that the same model would refuse in chat format. Current defenses require per-tool filters that do not scale. No mechanistic explanation exists for why the agentic format reduces the model's internal defenses.

## Research Question

> Does the refusal direction — the internal mechanism that mediates rejection of harmful prompts — activate in the same way when the model processes a prompt in agentic format (tool-calling) as in chat format?

## Hypotheses

**H₁ (Alternative):** The projection of activations onto the refusal direction is significantly lower in agentic format than in chat format for the same harmful prompt (the defense deactivates).

**H₀ (Null):** There is no significant difference in projection between formats (the defense holds and the problem is downstream).

## Variables

### Independent Variable
- **Format**: chat (no tools) vs. agentic (with tool definitions)

### Dependent Variables (Primary)
- Projection of activations onto the refusal direction: `projection = a⃗_t · d⃗_refusal ∈ ℝ`

### Dependent Variables (Secondary)
- Model behavior: refused or complied (binary)
- Logprobs of refusal token vs. tool call token
- Activation norm `‖a⃗_t‖` (control)

### Controlled Variables
- Harmful prompt: IDENTICAL between conditions
- Model: same model, same weights
- Temperature: 0 (deterministic)
- Extraction layer: measured across ALL layers
- Token position: last token before generation

## Industry Context

- 2024-2025: Explosion of LLM agents in production (ChatGPT plugins, Claude MCP, enterprise agents)
- Each agent has 10-50+ tools
- Tool marketplaces where users add their own
- Safety evaluations were done in chat and assumed to transfer to agents
- AgentHarm demonstrated this assumption is FALSE

## Impact

Moving from O(N) per-tool filters to O(1) universal monitor based on activation geometry. Security becomes a property of the model itself, not of each individual skill.
