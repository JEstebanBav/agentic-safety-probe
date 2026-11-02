## 4. Results and evaluation

The results concern the 96-pair corpus in all four models. Unless stated otherwise, the representation and validation figures use directions estimated out of sample in five pair-wise folds. Interpretation is developed in Section 5.

### 4.1. Results, analysis and interpretation

#### 4.1.1. Runs

All four runs process the corpus's 96 pairs, yielding 384 entries in four conditions. A probe or direction within one format has 192 observations, 96 harmful and 96 benign; pair variants are not independent units. Every run uses FP16, seed 42, the same procedure version and the factorial decomposition in Section 3.3. {{t:tab:corridas}} summarises their scope.

{{table:tab:corridas}}

#### 4.1.2. Baseline and discriminative capacity

The conversational direction is fitted on four of five pair folds and evaluated on the remaining one. {{t:tab:linea_base}} reports AUROC of its held-out scores at the reference layer. It measures discrimination between harmful and benign instructions, not actual refusal.

{{table:tab:linea_base}}

The direction retains agentic discrimination above 0.77 in every model, but loses some in all four: gap intervals exclude zero and the associated permutation test gives p &lt; 0.001 throughout. The loss is 0.08–0.10 in the two larger models and close to 0.15 in the smaller two. Optimism, the in-sample minus held-out estimate, is below 0.007 in chat and 0.021 in agent format, so circularity barely affects this measure. Intervals are conditional on obtained cross-fit scores and do not replace resampling that refits the complete direction.

The conversational logistic probe, evaluated with ten repetitions of five folds, achieves AUROC 0.9875 in Llama, 0.9734 in Mistral, 0.9582 in Qwen3-8B and 0.9694 in Qwen3-32B, against shuffled-label controls between 0.46 and 0.51. Its training AUROC is one throughout and is not treated as generalisation evidence. An MLP probe evaluated on the same folds does not outperform the linear probe in any model: MLP-minus-linear differences are −0.022 [−0.052, 0.007], −0.035 [−0.082, 0.013], 0.001 [−0.061, 0.064] and −0.016 [−0.032, −0.0002], using Nadeau–Bengio-corrected intervals. Harmful-content information exploited by these probes is therefore linearly accessible.

![Figure 1. Layer profiles in the four models. Top: harmful-arm projection shift, benign-arm shift and their interaction, using out-of-sample directions. Bottom: out-of-sample AUROC of the conversational direction in each format. The dotted vertical line marks the reference layer. Projection units are not comparable across models.](figures/fig_layers.png)

Figure 1 shows that the pattern is not limited to the reference layer. In all four models, conversational AUROC increases in early layers and stabilises between one-quarter and one-half of network depth, whereas agentic AUROC remains lower from that region onwards. The harmful shift and interaction increase with depth and reach their maxima in the final swept layers.

#### 4.1.3. Shift and benign control

The central result is expressed through within-pair projection changes. A lower projection does not by itself demonstrate lower refusal because it is geometric; the benign arm rules out an equal format-driven shift of both classes along the axis. {{t:tab:gap}} reports both arm shifts and their interaction.

{{table:tab:gap}}

In every model, the harmful arm loses projection in agentic format: 92, 87, 92 and 93 of the 96 pairs decrease, respectively, with sign-test p &lt; 10<sup>−16</sup>. The interaction is positive, its intervals exclude zero, its within-pair permutation p ≤ 10<sup>−4</sup> at 10,000-permutation resolution, and its paired effect size ranges from dz = 1.64 in Mistral to dz = 2.20 in Llama. Between 93 and 95 of the 96 pairs have positive interactions, so the finding does not depend on a subset of instructions.

The benign arm does not reproduce the shift. The common fraction is −12.1% in Llama, −6.2% in Mistral, −0.7% in Qwen3-8B and −1.4% in Qwen3-32B. The benign change is indistinguishable from zero in Mistral and both Qwens (two-sided p = 0.13, 0.88 and 0.74). In Llama it is distinguishable (p &lt; 0.001) but opposite in sign: benign instructions project slightly more in agentic format, accentuating class convergence rather than explaining it. The format effect is therefore specific to harmful content rather than a common translation of both classes.

#### 4.1.4. Effect scale

Raw class separation combines discriminative capacity and projection scale. {{t:tab:escala}} separates it from standardised separation d and contraction κ.

{{table:tab:escala}}

Raw separation drops by 84%–87% in all four models, but scale contraction dominates this figure: agentic projection dispersion is only 23%–37% of conversational dispersion. Standardised loss is smaller, 42%–62%, but remains; AUROC, which depends only on order, confirms a real loss of discrimination. No raw figure should be read as a percentage loss of safety.

#### 4.1.5. Results by explicitness

{{t:tab:estratos}} reports out-of-sample AUROC by explicitness, with 42 explicit, 24 contextual and 30 legitimately framed pairs.

{{table:tab:estratos}}

![Figure 2. Out-of-sample AUROC by harmful-intent explicitness in each model. Bars are point estimates, not formally tested differences between strata or between models.](figures/fig_strata.png)

Three regularities occur in all four models. First, every stratum loses discrimination in agentic format and none falls to chance: the minimum is 0.67 in Mistral's contextual stratum. Second, the explicit stratum retains the highest agentic AUROC in every model. Third, the contextual stratum shows the largest AUROC loss in all four, tied with legitimate framing in Mistral, and already starts with less conversational separation than the explicit stratum.

The ordering depends on the metric, however. In standardised units, the explicit stratum loses the largest fraction of separation in every model. Both readings are compatible: explicit requests start so widely separated that AUROC is near its ceiling and a substantial d loss barely affects ranking, whereas contextual requests start with less separation and the same contraction brings them closer to class confusion. The evidence supports poorer final agentic discrimination of requests whose intent must be inferred from context, not proportionally greater degradation of their representation.

#### 4.1.6. Factorial decomposition: system message and tools

The main contrast changes the system message and tool-schema presence jointly. Factorial decomposition measures each component's shift and non-additivity separately. {{t:tab:factorial}} reports each condition's harmful–benign interaction at the reference layer.

{{table:tab:factorial}}

![Figure 3. Fraction of the full-format interaction reproduced separately by each component, and their non-additivity, at each model's reference layer.](figures/fig_factorial.png)

Tool schemas presented with the conversational system message reproduce 97%–99% of the full-format effect in Llama and both Qwens, and 68% in Mistral. The agentic system message without tools produces a smaller effect, 33%–75%. The components do not add: non-additivity is negative throughout, meaning each already produces much of the effect and their combination adds little. The tools condition also lengthens the stimulus most: in Llama, a harmful instruction grows from 87 tokens in chat to 579 with tool schemas, versus 115 with the agentic system message. The decomposition does not separate schema content from length, so it identifies tools together with their added context as the dominant component.

#### 4.1.7. Reorientation and monitors by estimator

{{t:tab:rotacion}} compares directions estimated in each format with the same estimator, at the reference layer.

{{table:tab:rotacion}}

Mean-difference directions are highly reproducible within format, with reliabilities 0.92–0.97, but poorly aligned across formats: normalised cosine ranges from 0.19 in Qwen3-32B to 0.45 in Mistral. The harmful–benign contrast occupies a substantially different agentic direction. Logistic regression produces still smaller cosines, but its reliabilities of 0.24–0.33 indicate poor estimator reproducibility in this regime, requiring more caution with its angle. The two measures are not averaged.

![Figure 4. Cosine between conversational and agentic directions across each model's relative depth, using the same estimator in both formats.](figures/fig_rotation.png)

A loss of margin along the conversational direction does not mean agentic activations lack harmfulness information. {{t:tab:monitores}} reports monitors fitted agentically and evaluated on held-out pairs.

{{table:tab:monitores}}

The logistic monitor reaches AUROC 0.93–0.98 and F1 0.82–0.92 in all four models, exceeding the conversational direction applied to agentic inputs. Harmfulness signal remains in agentic activations and can be recovered by a monitor fitted in that format. Existence of this axis does not demonstrate its use by the internal refusal mechanism: it concerns what an external classifier can read. Llama's zero false positives concern 96 benign examples in this corpus, not a population rate.

### 4.2. Evaluation and validation

#### 4.2.1. PCA of the conversational contrast

PCA is fitted to concatenated harmful and benign conversational activations, centred by their global mean. {{t:tab:pca}} reports PC1 alignment with the refusal direction at an intermediate and the reference layer in each model.

{{table:tab:pca}}

![Figure 5. Alignment of PC1 with the refusal direction and proportion of variance explained by PC1 across each model's relative depth.](figures/fig_pca.png)

In early layers, the contrast does not coincide with PC1: it aligns better with components two through five and its PC1 cosine is low. Beyond a model-specific depth, the harmful–benign contrast becomes the sample's dominant variation axis, with cosines above 0.95 through the final evaluated layer: from layer 20 of 80 in Llama, 18 of 32 in Mistral, 16 of 36 in Qwen3-8B and 40 of 64 in Qwen3-32B. PC1 does not explain all variance, accounting for 14%–36% depending on model and layer. This supports a dominant geometric structure where the effect is measured; it does not, by itself, identify a principal component with the cause of refusal.

#### 4.2.2. Circularity and layer selection

Estimating and evaluating the direction on the same pairs inflates the measured shift. At the reference layer, naive versus cross-fit estimates change from 15.49 to 15.20 in Llama, 7.83 to 7.75 in Mistral, 179.37 to 176.47 in Qwen3-8B and 408.63 to 397.39 in Qwen3-32B, a bias of 1.0%–2.8% of the corrected value. Correction reduces the effect without changing its sign.

Choosing the reference layer using the same pairs as inference introduces a second bias. To avoid this, 48 pairs select the layer and estimate the direction, while a separate 48 supply inference. {{t:tab:seleccion}} reports the result.

{{table:tab:seleccion}}

In every model, the selection half chooses the same layer as the full sweep and inference-half shifts retain their magnitude: 44–47 of 48 pairs decrease, with sign-test p values far below Bonferroni thresholds. Inference-half interactions also exclude zero in all models. Across 200 random pair partitions, every model chooses its same layer in 100% of cases. These partitions describe selection stability on the observed corpus; they are not additional independent samples.

### 4.3. Behaviour without intervention

Llama-3.3-70B, Mistral-7B and Qwen3-8B answered the 96 instructions in all four conditions without intervention. {{t:tab:conducta}} reports fractions labelled refusal and fractions invoking a tool.

{{table:tab:conducta}}

![Figure 6. Fractions of responses refusing and invoking a tool in each condition. Refusal error bars are 95% Wilson intervals.](figures/fig_behavior.png)

All three models refuse harmful instructions less often agentically. In Llama, 40 pairs stop refusing and one changes in the opposite direction (exact McNemar p = 3.8 × 10<sup>−11</sup>); in Mistral the counts are 42 versus four (p = 5.1 × 10<sup>−9</sup>); in Qwen3-8B, among 63 pairs with complete responses in both formats, 31 versus zero (p = 9.3 × 10<sup>−10</sup>). Agentic format does not increase benign refusal: it remains low in Llama and Mistral and decreases in Qwen3-8B, whose conversational benign-refusal rate is high (18 pairs versus two, p &lt; 0.001).

The agentic format is exercised as such. No conversational response invokes a tool, whereas 97% of Llama's benign agentic responses, 92% of Mistral's and 58% of Qwen3-8B's do. In Llama, 51 of 53 harmful agentic non-refusals are tool calls: when the model complies, it does so by acting. Labels describe the first generated turn; tools are not executed.

Projection onto the refusal direction predicts behaviour within each format. AUROC distinguishing refused from complied-with harmful instructions is 0.972 in chat and 0.971 in agent format for Llama, 0.884 and 0.976 for Mistral, and 0.934 and 0.875 for Qwen3-8B. The relationship between loss magnitude and behavioural change differs across models. Among chat-refused pairs, Llama's requests that stop refusing agentically lose similar projection to those retaining refusal (16.76 versus 18.13; Mann–Whitney p = 0.48), as do Qwen3-8B's (194.4 versus 213.7; p = 0.25). In Mistral, changed requests lose more (11.30 versus 5.45; p &lt; 0.001), although only five pairs retain refusal.

### 4.4. Causal test

The causal test adds αr̂ to the residual stream during generation for harmful instructions complied with in agentic format, comparing with a matched-norm random direction and benign instructions receiving the same refusal-direction addition. Dose is a multiple of the measured layer-specific shift: dose one restores exactly the projection removed by format. Llama layers 76, 56 and 28 were declared before execution; Mistral and Qwen3-8B use separately selected layers. In all conditions, the induced projection shift matched α, verifying the manipulation. {{t:tab:causal}} reports the results.

{{table:tab:causal}}

![Figure 7. Dose response. Fraction labelled refusal when adding the refusal direction to harmful instructions, the random direction to harmful instructions, and the refusal direction to benign instructions. Bars are 95% Wilson intervals.](figures/fig_intervention.png)

In Llama, the test supports a causal role at layers 76 and 56. At dose one, restoring lost projection restores refusal for 21 and 25 of the 53 harmful instructions previously complied with, versus two of 53 under the random direction, while six and seven of 94 benign instructions are refused, close to the conversational benign-refusal rate. At dose two the effect loses selectivity: layer 56 refuses 51 of 53 harmful but also 91 of 94 benign instructions, and degeneration appears at layer 76, becoming almost universal at dose four. Layer 28 shows no effect at dose one, a weak selective effect at dose two, and 24 of 53 refusals under the random direction at dose four.

Mistral's effect is weaker. At dose one, the control difference is not significant (6/87 versus 1/87, p = 0.063). At dose two, refusal is restored for 22 of 87 harmful instructions versus four under the random direction, without increasing benign refusal, which stays at one of 96. At dose four, degeneration appears and the effect decreases.

In Qwen3-8B at layer 32, the refusal direction is indistinguishable from random control at every dose: dose one restores refusal in three of 39 instructions versus five under random control, and the largest difference at dose four is fifteen versus nine (p = 0.11). Across doses, five to sixteen responses per arm fail to complete reasoning within the generation budget, reducing test power. Restoring projection at that layer is insufficient to recover refusal in this model.
