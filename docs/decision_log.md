# Decision Log

## Why Llama-3.1-70B-Instruct as primary model?
- Large enough to have robust safety training (refusal mechanism present)
- Native tool-calling support in the chat template
- Well-studied model with baselines for comparison
- Results with 70B confirmed: DeltaP = 0.99, p = 0.005
- Alternative tested: Mistral-7B (faster but weaker safety alignment)

## Why RunPod instead of local?
- Corporate firewall blocks SSH; RunPod exposes JupyterLab via HTTPS (port 443)
- RTX A5000 ($0.27/hr) or RTX 4090 ($0.69/hr) -- 24GB VRAM sufficient for 7B in FP16
- For 70B: use multi-GPU pods or quantized version
- Budget: $10 gives ~25+ hours of compute

## Why dataset_full.jsonl and not HarmAgent directly?
- HarmAgent prompts are NOT paired (different text in chat vs agent)
- Without pairing, DeltaP could be due to prompt content differences
- Our custom dataset uses IDENTICAL text in both conditions
- HarmAgent is available as supplementary validation (behavioral)

## Why 42 base prompts (not 80 as originally planned)?
- Statistical power for paired design with d=0.42 at alpha=0.05 requires n~45
- 42 harmful base prompts with n=42 per condition is sufficient
- Adding subtlety stratification (18/14/10) was more valuable than more prompts

## Why difference-in-means and not PCA for d_chat?
- It's Arditi et al.'s method -- enables direct replication and comparison
- PCA is used as VALIDATION (cosine > 0.8 confirms alignment)
- Simpler, more interpretable
- Both give essentially the same direction when data is clean

## Why extract w_agent separately?
- With 70B model: d_chat gives AUROC 0.96 on agent data (it works!)
- But cosine(w_agent, d_chat) ~ 0.4 -- partial rotation
- With smaller models: d_chat may give AUROC ~0.5 on agent data (fails)
- w_agent provides a backup direction that always works in agent context
- The cosine between them IS the finding: quantifies how much the direction rotates

## Why stratify by subtlety?
- Global Cohen's d = 0.42 (small-medium) masks heterogeneous effects
- Explicit prompts likely show d > 0.7 (large effect)
- Framed prompts may show d ~ 0.2 (no significant effect)
- This explains WHY the global effect appears small
- Also has practical implications: framed attacks bypass the mechanism

## Why apply intervention to all layers?
- Single-layer intervention (original design) gave 0% success rate
- The refusal mechanism is distributed across multiple layers
- Applying to layers N/4 through N gives much stronger effect
- With w_agent applied at its specific extraction layer, results improve further

## Why measure the benign arm with the same statistics as the harmful one?
- DeltaP on harmful prompts alone has three explanations, not one: refusal
  weakens, the format displaces every representation by a common offset, or the
  projection scale contracts
- The agentic prompt is longer, carries a tool schema and a different system
  message; any of that can translate the whole residual stream along d_chat
- The corpus already contains the benign arm, so the control costs nothing
- The INTERACTION (harmful shift minus benign shift) is the quantity specific to
  harmfulness; `common_shift_fraction` reports how much of the headline number
  the benign arm reproduces
- Run at every layer, not only the reported one: a common shift is a property of
  the format and should be visible across depth
- The verdict is stated in the conclusion block. A significant DeltaP with a null
  interaction now prints "the format DISPLACES the representation; it does not
  selectively weaken refusal" instead of the old "STRONG EVIDENCE"

## Why split the prompts that build d_chat from those that score AUROC?
- d_chat IS the difference between the chat_harmful and chat_benign centroids, so
  AUROC on those same prompts measures how well a line fits the points it was
  fitted to
- At n=42 in d=8192 the direction absorbs a large amount of sample-specific
  noise, and projecting the same prompts back recovers it. On synthetic data with
  NO real signal the in-sample AUROC is 0.998 while the truth is 0.5
- The inflation is asymmetric: the agent condition never enters the estimation,
  so the chat format is flattered and the chat-vs-agent comparison is biased
  before any format effect is measured
- K-fold cross-fitting rather than a single held-out split: at this n a 50/50
  split would fit the direction on 21 pairs and score 21, and every prompt still
  contributing a score matters more than the simplicity of one split
- Folds are assigned over PAIRS, never over rows. A pair's harmful and benign
  variants are built from the same seed, so holding one out while training on the
  other would leak the contrast the direction encodes
- Both numbers are reported. The optimism (in-sample minus held-out) is the
  quantity that says how much the old figure was worth, and hiding it would
  replace one unexamined number with another

## Why permutation test instead of just t-test?
- No distributional assumptions (projections may not be Gaussian)
- More robust with small sample sizes (N=42)
- Reports empirical p-value from 10,000 permutations
- Welch's t-test included as parametric comparison

## Scope Boundaries

### What we address experimentally
- Semantic harm detection in agentic context (harm is in the prompt)
- Direction rotation between chat and agent contexts
- Practical safety monitor using w_agent

### What we identify but don't solve
- Intra-session contextual harm (harm emerges from conversation history)
- Requires late-layer probes that integrate full context
- Noted as future work

### What is out of scope
- Cross-session harm (user splits intent across multiple conversations)
- The model does NOT have the information in a single forward pass
- Requires architectural solutions (persistent risk memory)
