# Decision Log

## Why Mistral-7B and not another model?
- Fits in 20GB GPU in 4-bit quantization
- Has native tool calling support
- Sufficiently studied to have baselines
- Alternative: Llama-3.1-8B if Mistral doesn't support tool format adequately

## Why difference-in-means and not PCA or SAEs?
- It's Arditi et al.'s method → enables direct comparison
- Simpler, more interpretable
- PCA as iteration 2 if difference-in-means fails
- SAEs require separate training (not enough time in hackathon)

## Why the last instruction token?
- It's where Arditi et al. found the strongest signal
- It's the point where the model "decides" what to generate
- Other positions tested as iteration 3

## Why 80 base prompts?
- Arditi et al. used ~100 and it was sufficient
- 80 × 4 variants = 320 samples
- Statistical power for paired t-test with medium effect (d=0.5) and α=0.01 requires n≈50 pairs → we have 80

## Why not more harm categories?
- 4 categories × 20 prompts allows detecting per-category variation
- More categories with fewer prompts per category reduces statistical power within each
- Prioritize depth over breadth for hackathon

## Why not fine-tuning as a solution?
- Fine-tuning is expensive and requires curated data
- Activation intervention is inference-time, zero training cost
- If you fine-tune for one set of tools, it doesn't generalize to new tools
- Activation intervention DOES generalize

## Why not just per-tool filters?
- AgentHarm already demonstrated they don't work
- O(N) cost that doesn't scale
- Can't filter tools that don't exist yet (MCP, plugins)
- Our approach is O(1) — one dot product regardless of N tools

## Scope Boundaries

### What we DO address (experimentally)
- Semantic harm in agentic context (harmful content is in the prompt itself)

### What we IDENTIFY but don't solve
- Intra-session contextual harm (harm emerges from conversation history)
- Requires late-layer probes that integrate full context
- Noted as future work

### What is OUT OF SCOPE
- Cross-session harm (user splits intent across multiple chats)
- The model does NOT have the information
- Requires architectural solutions (persistent risk memory, user profiling)
- Identified as a hard limit of the interpretability approach
