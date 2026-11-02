## Appendix A. Experimental records and reproducibility

### A.1. Corpus

The file `dataset_v2.jsonl` contains 96 pairs and 384 entries: 62 pairs written for this investigation and 34 derived from AgentHarm. Text identity across formats, and symmetry of the system message, tool list and schemas between harmful and benign arms within each format, were verified in all 96 pairs. The corpus uses 64 distinct tools. Explicitness counts are 42 explicit, 24 contextual and 30 legitimately framed. The construction report and annotation file, including the rationale for every explicitness classification, are included in the supporting evidence.

Complete instructions and model generations are retained in the research repository and are not reproduced here as a task catalogue. AgentHarm-derived entries retain the contamination-check identifier published with that dataset.

### A.2. Run configurations

The four runs share a 96-pair corpus, FP16, seed 42, five pair-wise folds, ten linear-probe repetitions and one MLP-probe repetition, ten shuffled-label controls, 10,000 resamples and permutations, 200 splits for layer-selection stability, and extraction of both intermediate factorial conditions. {{t:tab:config}} gives run-specific differences.

{{table:tab:config}}

### A.3. Evidence and replication

All numerical values in the results and discussion derive from the metric evidence in `evidence/metrics.json`, which collects each model's metrics extracted from saved results, their provenance and the SHA-256 hash of each source file.

Every run retains activations at swept layers, complete configuration and results from each stage. The three runs with behavioural evaluation also retain every generated response with its assigned label. Saved activations permit repetition of any statistical analysis in Section 4 without running the models again; repeating behaviour and causal tests requires model inference.

## Appendix B. Notation and abbreviations

### B.1. Notation

**Cone of influence, C(i).** The sequence tokens capable of changing position i's representation. The causal mask makes this {1, …, i}, so the final position of the rendered stimulus has causal access to the complete stimulus, including delimiters. This is why the study measures at that position.

**Reference layer.** The layer at which the procedure maximises the measured shift and reports principal figures: layer 76 in Llama-3.3-70B, 30 in Mistral-7B, 32 in Qwen3-8B and 62 in Qwen3-32B.

**Jaccard index, J.** Shared terms between harmful and benign arms divided by the total number of distinct terms used by both. It measures lexical pairing symmetry: one for identical vocabulary and zero for no shared terms.

**Projection, p.** A geometric measure of how closely an instruction's representation aligns with the refusal direction. A high value indicates greater alignment with that axis, not that the model recognises danger; class-discrimination capacity is quantified separately with AUROC.

**Projection gap, Δph.** Mean harmful-instruction projection conversationally minus the projection of the same instructions agentically. It measures the decrease in alignment with format change and is this work's central result.

**Interaction or difference-in-differences, Δ²p.** Harmful-arm shift minus benign-arm shift: the harmfulness-specific part of the effect.

**Common fraction, φ.** The proportion of the harmful shift reproduced by the benign arm. A negative value indicates opposite-direction arm shifts.

**AUROC.** Probability that a randomly selected harmful instruction projects higher than a randomly selected benign instruction. One indicates perfect separation and 0.5 chance-equivalent separation.

**Detection gap, ΔAUROC.** Conversational minus agentic AUROC. It is the net difference in correct-ordering probabilities, permits changes in both directions and does not count only lost correct predictions. It differs from the projection gap and answers another question: projection gap concerns response intensity, while detection gap concerns discriminative capacity.

**Scale contraction, κ.** Agentic projection standard deviation divided by conversational projection standard deviation. It quantifies how much of a raw-unit loss corresponds to a unit-scale change.

**wDM and wLR.** Agentic directions estimated by difference of means and logistic regression respectively, reported separately. The first contrasts agentic centroids. Its cosine with the conversational refusal direction measures how far the harmfulness-encoding axis has rotated.

**System and tool effects, δsys and δtools.** Projection shifts separately produced by an agentic system message without tools and tool schemas with a conversational system message. Their difference from the complete-format effect is non-additivity, η.

**Dose, m.** Intervention magnitude in multiples of the shift measured at the intervened layer. Dose m = 1 restores exactly the projection removed by agentic format.

**Tool-call rate.** The fraction of responses invoking one of the offered tools. It indicates that the model exercises the agentic format; it is zero conversationally, where no tools are offered.

### B.2. Abbreviations

- **AI:** artificial intelligence.
- **LLM:** large language model.
- **RLHF:** reinforcement learning from human feedback.
- **AUROC:** area under the receiver operating characteristic curve.
- **BCa:** bias-corrected and accelerated bootstrap confidence interval.
- **CV:** cross-validation.
- **MLP:** multilayer perceptron.
- **GPU:** graphics processing unit.
- **FP16:** 16-bit floating-point precision.
- **FPR:** false-positive rate.
- **OOF:** out of fold; estimation on held-out data not used in fitting.
- **PCA:** principal component analysis.
