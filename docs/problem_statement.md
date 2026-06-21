# Problem Statement

## The Problem

LLM-based agentic systems execute harmful tasks that the same model would refuse in chat format. Current defenses require per-tool filters that do not scale. No mechanistic explanation exists for why the agentic format reduces the model's internal defenses.

## Research Question

> Does the refusal direction -- the internal mechanism that mediates rejection of harmful prompts -- activate in the same way when the model processes a prompt in agentic format (tool-calling) as in chat format?

> And critically: which specific component of agentic formatting is responsible for the effect? The role framing? The tool definitions in context? Or the structured output format?

## Hypotheses

**H1 (Alternative):** The projection of activations onto the refusal direction is significantly lower in agentic format than in chat format for the same harmful prompt (the defense deactivates).

**H0 (Null):** There is no significant difference in projection between formats (the defense holds and the problem is downstream).

**H2 (Extended):** The effect varies by prompt subtlety -- explicit prompts show larger deactivation than contextually harmful or professionally framed ones.

## Variables

### Independent Variable
- **Format**: chat (no tools) vs. agentic (with tool definitions and structured output)
- **Subtlety**: explicit, contextual, framed (stratification variable)

### Dependent Variables (Primary)
- Projection of activations onto the refusal direction: `projection = a_t . d_refusal`
- Projection gap: `DeltaP = mean(proj_chat) - mean(proj_agent)`

### Dependent Variables (Secondary)
- Agent-specific direction alignment: `cos(w_agent, d_chat)`
- Safety monitor F1-score using w_agent
- Intervention success rate

### Controlled Variables
- Harmful prompt: IDENTICAL text between chat and agent conditions
- Model: same model, same weights
- Temperature: 0 (deterministic)
- Token position: last token before generation (decision point)

## Key Finding

With Llama-3.1-70B-Instruct, the refusal direction computed from chat (d_chat) gives AUROC ~0.96 in both chat and agent conditions, but the projection magnitude is significantly lower in agent format (DeltaP = 0.99, p = 0.005). Furthermore, a linear probe trained on agent activations finds a DIFFERENT direction (w_agent) with cosine similarity ~0.4 to d_chat -- indicating that the model re-encodes harmfulness in a rotated direction when in agentic context.

## Industry Context

- 2024-2025: Explosion of LLM agents in production (ChatGPT plugins, Claude MCP, enterprise agents)
- Each agent has 10-50+ tools
- Safety evaluations done in chat mode assumed to transfer to agents
- AgentHarm (Andriushchenko et al., 2024) demonstrated this assumption is FALSE
- No mechanistic explanation existed before this work

## Impact

Moving from O(N) per-tool filters to O(1) universal monitor based on activation geometry. The safety monitor uses a single dot product against w_agent to detect harmful intent in agentic context regardless of which tools are available.
