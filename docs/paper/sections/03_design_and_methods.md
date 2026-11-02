## 3. Research design and methods

### 3.1. Approach and design

#### 3.1.1. Methodological approach

This investigation adopts a quantitative approach insofar as it relies on statistical analysis of a language model's internal numerical representations to establish how its refusal mechanism varies between two operating contexts. The design is comparative because it contrasts the behaviour of the same representation under two controlled conditions: the conversational format and the agentic format. The comparison rests on a principle of experimental control: the content of each instruction is held constant across conditions, and only the element whose effect is to be isolated is modified, namely the presence of tools and the associated agentic context.

The study is organised around four complementary analytical components. The first is descriptive and is intended to characterise the refusal direction in the conversational format and establish it as a reference baseline. The second is comparative and is intended to quantify how that direction changes when the agentic context is introduced, to decompose the effect into its components, and to evaluate the stability of its detection capacity according to the explicitness with which harmful intent is presented. The third is behavioural: it observes whether the model refuses or complies in each format. The fourth is experimental: it intervenes on the representation to test whether the direction affects that behaviour.

#### 3.1.2. Type of study

The study has a descriptive and comparative component concerning internal representations, a behavioural component, and an experimental intervention component. The main contrast preserves the content of each instruction and modifies the complete format: the system message and the availability of tools. That contrast estimates the effect of the combined changes. To identify the contribution of each component, a factorial decomposition is added in which each component is presented separately.

In the diagnostic analysis, weights remain fixed and activations are recorded. In the behavioural component, the model responds without intervention and its responses are labelled. In the experimental component, the residual stream is temporarily modified during generation, without retraining or modifying the weights. These interventions make it possible to assess an effect of the direction on refusal under the conditions tested, but do not by themselves demonstrate that an observed reorientation is the general cause of failures in deployed agents.

#### 3.1.3. Object of study

The object of study is the refusal direction, operationally estimated as a harmful–benign contrast in activation space. Its discriminative capacity and its relationship with behaviour are evaluated separately; the name does not presume that every direction estimated in this way is a causal mechanism of refusal. The specific interest concerns how this representation behaves when the model moves from conversational to agentic format, and whether its detection capacity is preserved for instructions whose harmful intent is presented with different levels of explicitness.

The analysis is carried out on open-weight language models whose architectures allow the direct access to internal activations required by the study. The reported runs use four models from three families: Mistral-7B-Instruct-v0.3, Qwen3-8B, Qwen3-32B and Llama-3.3-70B-Instruct. All have alignment training and support for tool use, conditions necessary to construct the two conditions studied. Selection covers two axes of variation. The first is the training recipe: three families with different pretraining and alignment processes make it possible to distinguish what is characteristic of the phenomenon from what is characteristic of a particular recipe. The second is scale, between seven and seventy billion parameters, with two sizes within the Qwen family allowing the effect of size to be observed while keeping the recipe constant. The architectural characteristics of the four models are reported in {{t:tab:modelos}}.

All four models are analysed on the complete 96-pair corpus in FP16. Representation analysis covers all four; behavioural and experimental components cover Llama-3.3-70B, Mistral-7B and Qwen3-8B.

The sampling unit is the base pair. Its four variants are not four independent observations. The 96 pairs comprise 42 explicit, 24 contextual and 30 legitimately framed pairs. These stratum sizes limit the precision of comparisons between them.

### 3.2. Data and information sources

The data strategy is based on a paired contrastive design whose purpose is to isolate the effect of the agentic format on the model's internal representations. The dataset takes three elements from the AgentHarm benchmark {{c:andriushchenko2024}}: the taxonomy of harm categories, the logic of tasks requiring tool use, and the principle of pairing each harmful task with a benign counterpart. The instructions, however, were written or adjusted for this investigation to incorporate three controls that the original construction does not provide and that this study requires: identical text across conditions, stratification by the explicitness of harmful intent, and lexical symmetry between the two arms of each pair.

The distinction concerns purpose rather than quality. AgentHarm was constructed to measure, at the behavioural level, whether an agent complies with a dangerous request. For that purpose, it is sufficient that the benign counterpart should be an equivalent benign task. This study, by contrast, estimates a direction in activation space through the difference of means between two centroids, a procedure that requires the two sets to differ in intent rather than in topic. The three controls are therefore not an optional refinement but a condition of the method.

Construction and consolidation of the instructions were undertaken by the researcher responsible for the corpus. The corpus contains 62 pairs written for this investigation and 34 pairs derived from AgentHarm. Adaptation prioritised identical text between formats, thematically close benign counterparts, and an explicit classification of the level of explicitness. A case is retained when it permits suitable pairing and a justifiable assignment; otherwise it is reformulated or discarded. The origin is recorded for each pair. Adapting counterparts means that the corpus does not constitute a complete evaluation of the published benchmark.

Construction included model-assisted review. A language model, Claude, acted as an auxiliary annotator when reviewing the pairs, identifying thematic asymmetries between the arms, discrepancies in text identity across formats, and questionable explicitness assignments. The corpus researcher resolved those issues when consolidating the final version. This assisted review served a quality-control function during construction and does not replace the second independent classification needed to estimate the degree of agreement between annotators. That second classification was not carried out in this work and is discussed among its limitations. The decision on every case remained with the same researcher at all times.

#### 3.2.1. Paired contrastive design

Each base instruction is presented in four variants resulting from crossing two factors. The first factor is the content of the instruction, which may be harmful or benign. The second is the operating format, which may be conversational or agentic. Crossing them produces four conditions: harmful in conversational format, harmful in agentic format, benign in conversational format, and benign in agentic format.

The distinctive feature of the design is that the instruction text is identical between the conversational and agentic formats. The only difference between the two is the presence of tool definitions and the agent-specific system prompt. This control makes it possible to attribute observed variation in internal representations to the change of format rather than to differences in wording, a limitation of benchmarks that use different texts for each condition.

Each harmful instruction is accompanied by a benign counterpart constructed on the same structure, so that the two share vocabulary, approximate length, the entities mentioned and the repertoire of tools, differing only in their underlying intent. This pairing is necessary for calculating the refusal direction through a difference of means, which requires two comparable sets of activations. If the benign arm changed topic as well as intent, the estimated direction would capture both simultaneously and could not be interpreted as an axis of harmfulness. The degree to which this control holds is explicitly verified and reported in Section 3.2.5.

The design also imposes two invariants that are verified throughout the complete dataset. The first is that all four variants of a pair share an identifier, a condition on which every subsequent paired test depends. The second is that, within each format, the harmful and benign arms receive the same system message and the same tool list. The difference between them therefore cannot arise from the agent scaffold or from having different capabilities, but only from the intent of the instruction.

#### 3.2.2. Stratification by intent explicitness

A central contribution of the design is the stratification of harmful instructions according to the explicitness with which harmful intent is manifested, at three levels. At the first level, harmful intent is explicit: the harm is evident on the surface of the text, and keywords reveal it directly. At the second level, intent is contextual: the text contains no openly harmful terms, but the intent can be inferred from the context of the request. At the third level, intent is presented under a legitimate appearance: the instruction is framed as a professional or justified task, while the action requested is harmful. This third level is referred to as legitimately framed intent.

This stratification is a construct proposed within this investigation, rather than a categorisation taken from a previously validated instrument, because the literature on the refusal direction has not incorporated explicitness as an analytical dimension. Its justification is conceptual and responds to a limitation of approaches that consider only explicit harm. An aligned model readily refuses openly harmful instructions precisely because keywords activate the refusal mechanism. Cases representing a real deployment risk are those in which harmful intent resides in the context or framing rather than in the words. Incorporating all three levels makes it possible to evaluate whether the refusal direction's detection capacity remains stable or degrades as harm becomes less evident on the surface of the text, which is one of this investigation's objectives.

The operational criterion assigning an instruction to a level distinguishes whether marked vocabulary reveals harm on the surface of the text, in which case the instruction is explicit, or whether the harm must be inferred from the situation described, in which case it is contextual. Legitimate framing is reserved for instructions that additionally incorporate a justificatory frame. Each classification was recorded with its case-by-case rationale in an annotation file accompanying the dataset, making the decision auditable and repeatable. Estimating the degree of agreement between annotators requires a second classification independent of the first, which is not part of this work.

#### 3.2.3. Tool definitions

Constructing the agentic condition requires providing the model with tools that define its capacity to act. A repertoire of simulated, or mock, tools was therefore defined in the function-calling format used by the models for tool use. Each tool is described by a name, a description of its function and a schema for its input parameters.

These tools execute no real actions. Their presence in the context has the sole purpose of placing the model in an agentic setting, that is, making the capacity to invoke actions available to it, the condition whose effect on internal representations is being studied. The repertoire covers tools associated with the different harm categories, such as sending email, searching the web, retrieving pages, executing terminal commands, transferring funds, generating images and posting on social media. They are selected so that every agentic instruction receives tools consistent with the task it describes.

#### 3.2.4. Dataset composition

The dataset consists of 96 base pairs. Each pair is presented in the four variants described above, giving 384 entries divided equally among the harmful chat, harmful agentic, benign chat and benign agentic conditions, with 96 instances in each. The instruction text was verified to be identical between conversational and agentic formats in every pair, so the only difference between those formats lies in the system prompt and tool definitions.

The harmful instructions are distributed across seven harm categories and the three levels of explicitness described above. {{t:tab:distribucion}} presents the complete cross-tabulation of those dimensions. No cell is empty, and every cell contains between two and eight pairs, so every combination of category and level is represented.

{{table:tab:distribucion}}

The size of the smallest stratum, the contextual stratum with 24 pairs, determines the precision of the analysis differentiated by explicitness. The three levels have comparable orders of magnitude, a condition without which that differentiated analysis could only be presented as an exploratory pattern.

A repertoire of 64 simulated tools was defined to construct the agentic condition, all of which are actually used in the dataset. Each agentic task has between two and eight tools consistent with the action it describes, with a mean of 3.4 tools per task. The most frequent are web search and page retrieval, present in 44 and 40 pairs respectively, followed by email, social-media posting and file upload.

##### Examples and scope of the corpus

The three levels are illustrated through the fraud category. At the explicit level, an instruction asks for stolen banking credentials to be tested and funds transferred to a cryptocurrency wallet; the benign counterpart replaces those credentials with authorised test accounts. At the contextual level, an instruction asks for emails to older adults announcing a nonexistent subsidy and asking them to verify their banking details through a link; no word is openly harmful, and the fraud is inferred from the context. At the legitimately framed level, an instruction asks for a realistic phishing email presented as awareness training and sent to the entire contact database. In all three cases, the benign instruction shares structure and vocabulary with the harmful one, allowing the difference of means to capture intent rather than topic.

Part of the dataset also includes harm typologies documented in the Latin American context, such as impersonating a relative to request an urgent transfer, investment schemes whose returns are paid from new deposits, misinformation about the relocation of polling places, and discrimination in hiring. These instructions are written in the same language as the rest of the dataset, because introducing another language would introduce a language-related shift into the difference of means and confound it with the harm-related shift.

#### 3.2.5. Lexical symmetry and its verification

The design requires the two arms of a pair to differ in intent rather than in topic, but that requirement is qualitative and cannot be verified through inspection alone. To make it testable, an explicit measure of lexical overlap between the two arms was adopted: the Jaccard index over the term sets of the two instructions:

<p class="equation">J(h,b) = |T<sub>h</sub> ∩ T<sub>b</sub>| / |T<sub>h</sub> ∪ T<sub>b</sub>|. &nbsp; (1)</p>

Th and Tb denote the term sets of the harmful instruction and its benign counterpart. The index equals one when the two instructions use exactly the same vocabulary, and zero when they share no terms. A high value indicates that the difference between the arms is concentrated on what the design seeks to isolate; a low value warns that the pair introduces an additional change of topic.

This measure is used in two ways. The first is as a construction criterion: each pair was written to seek overlap comparable to that of the rest of the dataset, with similar lengths between the two arms and the same entities mentioned in both. The second, and more important use, is as a control on comparison between strata. If pairs in one stratum were systematically more closely matched than pairs in another, an observed difference in detection capacity by explicitness could arise from that construction asymmetry rather than from the phenomenon being measured. Verifying that overlap is homogeneous across strata is therefore a condition for interpreting the differentiated analysis.

{{t:tab:jaccard}} reports the values obtained. The three strata fall within a range of 0.014, so the comparison between them is not conditioned by how closely their instructions are paired.

{{table:tab:jaccard}}

The scope of this measure should be made precise. The Jaccard index operates on term sets and captures neither word order nor synonymy, so it measures vocabulary coincidence rather than semantic similarity. Its function here is limited and specific: to detect pairs in which the benign counterpart departs from the topic of the harmful instruction, which is the concrete way the design control can fail. Eight pairs have overlap below 0.25 and are retained because their task structures require different vocabulary in each arm.

The dataset is generated by a deterministic procedure checking seven design invariants, including identical text across formats and identical tool lists across arms, and aborting without writing if any check fails.

### 3.3. Methods and tools

#### 3.3.1. Mathematical formulation

The method is based on analysis of the internal representations that a transformer language model generates when processing an instruction, following the approach of Arditi et al. {{c:arditi2024}}. The formulation begins with the architecture of the models studied, from which the reading position in the sequence and the need to express effects in scale-invariant units are derived. It then defines the refusal direction and the quantities used to compare the two formats.

##### Model architecture and residual stream

Let an instruction be tokenised as the sequence t₁, …, tₙ. The four models studied are decoder-only transformers with L layers and normalisation before each block. Each position i maintains a vector xᵢ<sup>(l)</sup> in ℝ<sup>d</sup>, the residual stream, which starts from the embedding of token tᵢ and is updated by each layer through the addition of an attention-block contribution and a feed-forward contribution:

<p class="equation">x<sub>i</sub><sup>(l)</sup> = x<sub>i</sub><sup>(l−1)</sup> + a<sub>i</sub><sup>(l)</sup> + m<sub>i</sub><sup>(l)</sup>. &nbsp; (2)</p>

Both blocks read a normalised version of the residual stream through RMSNorm, which divides the vector by its root mean square, so what they read does not depend on its norm. The residual stream itself, however, accumulates magnitude without normalisation. The feed-forward block, of the SwiGLU type, acts on each position separately. The attention block is the only component that mixes information between positions: in each head, the weights that position i assigns to position j are:

<p class="equation">α<sub>ij</sub> = softmax<sub>j</sub>(q<sub>i</sub> · k<sub>j</sub> / √d<sub>k</sub> + M<sub>ij</sub>), &nbsp; M<sub>ij</sub> = 0 if j ≤ i, and −∞ if j &gt; i. &nbsp; (3)</p>

Here qᵢ and kⱼ are linear projections of positions i and j, rotated according to their positions in the sequence, and M is the causal mask. Finally, the distribution over the next token is obtained from the last position of the last layer:

<p class="equation">P(t<sub>n+1</sub> | t<sub>1</sub>, …, t<sub>n</sub>) = softmax(W<sub>U</sub> RMSNorm(x<sub>n</sub><sup>(L)</sup>)). &nbsp; (4)</p>

{{t:tab:modelos}} reports the dimensions of the four models. Expanding Equation (2), the residual stream at layer l is the initial embedding plus the sum of everything written to it by the preceding layers, xᵢ<sup>(l)</sup> = xᵢ<sup>(0)</sup> + Σ<sub>k≤l</sub>(aᵢ<sup>(k)</sup> + mᵢ<sup>(k)</sup>). This additive structure is what makes it reasonable to look for a concept in the form of a direction in ℝ<sup>d</sup>.

{{table:tab:modelos}}

##### Reading position: the final token

The method measures a single position, the final token of the rendered stimulus before generation, for three reasons. That stimulus includes the system message, tools, instruction and template delimiters. The position may correspond to a delimiter, such as `[/INST]`, rather than to the final lexical token written by the user. The first reason is that the causal mask in Equation (3) sets αᵢⱼ to zero for every j &gt; i, while the feed-forward block does not mix positions. By induction over layers, the representation of position i depends only on tokens t₁, …, tᵢ. This set is the position's *cone of influence*, and the cone at position n contains the entire stimulus. This property describes the causal access permitted by the architecture without guaranteeing that all information is used. The second reason is that, by Equation (4), the first response token, where refusal or compliance materialises, is decided from xₙ<sup>(L)</sup>. Measuring position n at an intermediate layer therefore observes the same pathway that produces the decision. From this point onwards, x<sup>(l)</sup> denotes xₙ<sup>(l)</sup>.

The third reason is specific to this design. The agentic stimulus is longer than the conversational stimulus because it incorporates the system message and tool schemas. A measure averaging across positions would depend on the prefix length and would capture, in addition to the format effect, a term attributable to that difference in length, without a way to separate the two. The final position, by contrast, plays the same functional role in both formats: it is the last position before generation conditioned on the complete context, regardless of how many scaffold tokens precede it.

##### Calculating the refusal direction

Let Dh and Db be the sets of harmful and benign instructions in conversational format, with centroids μh<sup>(l)</sup> and μb<sup>(l)</sup> at layer l. The refusal direction is their normalised difference:

<p class="equation">r<sup>(l)</sup> = μ<sub>h</sub><sup>(l)</sup> − μ<sub>b</sub><sup>(l)</sup> = (1/n) Σ<sub>i=1…n</sub>(x<sub>h,i</sub><sup>(l)</sup> − x<sub>b,i</sub><sup>(l)</sup>), &nbsp; r̂<sup>(l)</sup> = r<sup>(l)</sup> / ‖r<sup>(l)</sup>‖. &nbsp; (5)</p>

The second equality expresses the direction as the mean of within-pair differences: each pair contributes the contrast between two instructions that share topic and vocabulary and differ in intent.

The axis that best separates two classes, in the sense of the Fisher discriminant, is ΣW<sup>−1</sup>(μh − μb), where ΣW is the within-class covariance. With 2n observations and n ≪ d, the sample covariance has rank at most 2n − 2 and is not invertible. If it is regularised as Σ̂W + λI, the normalised direction of the discriminant converges to the direction of the difference of means as λ → ∞. This estimator is therefore the discriminant that forgoes estimating the covariance, although the estimated mean continues to have sampling noise and requires out-of-sample evaluation when n ≪ d.

The direction is an estimator rather than a known quantity of the model. If r is the population difference and ΣΔ is the covariance of the within-pair differences:

<p class="equation">E[r<sup>(l)</sup>] = r, &nbsp; E[‖r<sup>(l)</sup>‖²] = ‖r‖² + tr(Σ<sub>Δ</sub>)/n. &nbsp; (6)</p>

The vector is unbiased, but its norm is upward biased by a term that grows with dimension and decreases with the number of pairs. Since the class separation measured on the same data that produced the direction is exactly ‖r<sup>(l)</sup>‖, that separation inherits the bias, whereas the agentic condition, which does not participate in estimation, does not inherit it. This is the circularity bias that the procedure corrects through cross-validation.

##### Projection and differences between formats

An instruction's projection onto the refusal direction is:

<p class="equation">p<sup>(l)</sup>(x) = x<sup>(l)</sup> · r̂<sup>(l)</sup> = ‖x<sup>(l)</sup>‖ cos θ. &nbsp; (7)</p>

This constitutes the central measure of the study: a high value indicates that the instruction's representation is aligned with the refusal direction. The second equality shows that it mixes the orientation of the activation, which is of interest, with its norm, which is not. Since every instruction appears in both formats with identical text, the unit of analysis is the difference within the pair:

<p class="equation">δ<sub>h,i</sub><sup>(l)</sup> = p<sub>chat</sub><sup>(l)</sup>(x<sub>h,i</sub>) − p<sub>ag</sub><sup>(l)</sup>(x<sub>h,i</sub>), &nbsp; Δp<sub>h</sub><sup>(l)</sup> = (1/n) Σ<sub>i=1…n</sub>δ<sub>h,i</sub><sup>(l)</sup>. &nbsp; (8)</p>

The analogous definition gives Δpb<sup>(l)</sup> for the benign counterparts. A value Δph<sup>(l)</sup> &gt; 0 indicates that the refusal direction responds less strongly in the agentic format. With sδ the standard deviation of the within-pair differences, the statistic for the paired t test and the paired effect size are:

<p class="equation">t = Δp<sub>h</sub><sup>(l)</sup> / (s<sub>δ</sub>/√n), &nbsp; d<sub>z</sub> = Δp<sub>h</sub><sup>(l)</sup>/s<sub>δ</sub>, &nbsp; t = d<sub>z</sub>√n. &nbsp; (9)</p>

Thus dz, rather than t, is comparable across corpora of different sizes.

##### Effect specificity: the benign control

The quantity Δph<sup>(l)</sup>, on its own, does not distinguish three mechanisms: a weakening of refusal for harmful requests, which is what the investigation hypothesises; a rigid shift that moves every representation along r̂<sup>(l)</sup>; and a scale contraction that compresses all distances. The benign arm resolves the first ambiguity because it undergoes exactly the same format change. The quantity specific to harmful content is the interaction between content and format, accompanied by the fraction of the effect reproduced by the benign arm:

<p class="equation">Δ²p<sup>(l)</sup> = Δp<sub>h</sub><sup>(l)</sup> − Δp<sub>b</sub><sup>(l)</sup>, &nbsp; φ<sup>(l)</sup> = Δp<sub>b</sub><sup>(l)</sup> / Δp<sub>h</sub><sup>(l)</sup>. &nbsp; (10)</p>

A rigid shift predicts Δ²p ≈ 0 and φ ≈ 1; specific weakening predicts Δ²p &gt; 0 with small φ; and φ &lt; 0 means the arms move in opposite directions. The working hypothesis is Δ²p<sup>(l)</sup> &gt; 0.

##### Factorial decomposition of format

The agentic format combines two components: a system message positioning the model as an agent and schemas describing the available tools. To estimate the contribution of each, every instruction is also presented in two intermediate conditions: the agentic system message without tools, and tool schemas with the conversational system message. Let psys and ptools be projections in those conditions. The shifts attributable to each component and their non-additivity are defined as:

<p class="equation">δ<sub>sys</sub> = p<sub>chat</sub> − p<sub>sys</sub>, &nbsp; δ<sub>tools</sub> = p<sub>chat</sub> − p<sub>tools</sub>, &nbsp; η = δ<sub>tot</sub> − (δ<sub>sys</sub> + δ<sub>tools</sub>). &nbsp; (11)</p>

Here δtot is the shift produced by the full format. Each quantity is a difference within an instruction and is tested as such; its harmfulness-specific version is obtained by subtracting the corresponding quantity for the benign arm, as in Equation (10). A value η ≈ 0 indicates that the components add together; η &lt; 0 indicates that they overlap, so each component already produces part of the other's effect on its own. The decomposition uses the full-sample direction and is interpreted as descriptive of an effect whose existence is established with out-of-sample directions. Tool schemas are the component that lengthens the stimulus most, so their effect is not separated from the length they introduce.

##### Scale invariance

Scale contraction is addressed by expressing the effect in units that survive it. Let S be the separation between the mean projections of the two classes within a format, and s their pooled standard deviation. Standardised separation and the contraction factor are defined as:

<p class="equation">d = S/s, &nbsp; κ = s<sub>ag</sub>/s<sub>chat</sub>. &nbsp; (12)</p>

A limiting case clarifies their role. If format merely rescales and shifts projections, pag = a pchat + b with a &gt; 0, then:

<p class="equation">S<sub>ag</sub> = a S<sub>chat</sub>, &nbsp; d<sub>ag</sub> = d<sub>chat</sub>, &nbsp; κ = a. &nbsp; (13)</p>

Raw separation would therefore lose a fraction 1 − a, which can be arbitrarily large, without any loss in d or in AUROC, which depends only on ordering. The part of the loss that reflects discriminative capacity is therefore the part that survives in d and AUROC. For the same reason, raw units are not comparable across models whose residual streams have different norms.

##### Agentic direction, reorientation and monitoring

Two estimators are distinguished. The agentic mean-difference direction, ŵDM<sup>(l)</sup>, is obtained by subtracting the harmful and benign agentic centroids and normalising the result. The direction ŵLR<sup>(l)</sup> is obtained from the coefficients of a linear logistic regression. A classifier does not necessarily recover the same vector as the centroid contrast. For that reason, every cosine, AUROC and monitor is identified by its estimator.

The cosine of ŵDM<sup>(l)</sup> with r̂<sup>(l)</sup> compares two mean contrasts. The additional logistic-regression analysis compares directions obtained using the same estimator in both formats. Reliability between disjoint halves is reported because estimation noise can reduce the apparent cosine. Reliability normalisation is a diagnostic under assumptions about error, rather than an exact measurement of a population angle.

Monitors classify the projection using a threshold calibrated on each fold's training data. Evaluation is carried out on held-out pairs and reports AUROC, F1 and the false-positive rate. The results of the two estimators are kept separate. They do not constitute validation on external data or guarantees of operational safety.

##### Geometric validation with PCA

For each layer, the harmful and benign conversational activations are concatenated into a matrix X ∈ ℝ<sup>2n×d</sup> and its global mean is subtracted. PCA is fitted to that centred matrix without centring each class separately: doing so would eliminate precisely the difference between centroids. The proportion of variance explained by PC1 and |cos(PC1,r̂)| are reported. The sign of PC1 is oriented so that the cosine is positive. The first five components and the variance projected onto the direction are also examined. Alignment with PC1 indicates predominance in the total variance of that sample; its absence does not automatically invalidate a discriminative contrast.

#### 3.3.2. Analysis procedure

The procedure applies the preceding formulation in five stages. In the first stage, the activation of the final token is extracted for each instruction and each swept layer in the conversational format, the agentic format and the two intermediate conditions of the factorial decomposition, keeping all variants of a pair aligned by identifier. In the second, the refusal direction is estimated in the conversational format and its behaviour is characterised by layer, using principal components to evaluate whether the harmful–benign contrast coincides with a dominant axis of variation. In the third, the format shift, its interaction, its expression in raw, standardised and rank-based units, and its decomposition by format components are calculated separately for each arm, first over the whole dataset and then by explicitness. In the fourth, the agentic direction, its reorientation relative to the conversational direction, and the monitors are estimated. In the fifth, once all the preceding results have been saved, the model generates responses without and with intervention, as described in Section 3.4.3.

#### 3.3.3. Implementation and reproducibility

Analysis was carried out on open-weight models downloaded and executed on rented GPUs, without access to external inference services. All four runs use the same version of the procedure, FP16 precision and seed 42. Llama-3.3-70B, Mistral-7B and Qwen3-32B were run on a 141-GB GPU; Qwen3-8B was run on a 48-GB GPU. Each stimulus is tokenised with a single beginning-of-sequence token and processed in a separate pass, without history or state shared with other stimuli. The weights remain fixed and the model retains nothing from one stimulus to the next, so processing order does not influence activations or responses. For the same reason, each causal-test intervention is installed at the beginning of a generation and removed at its end. Extraction records the residual stream corresponding to the final token of the rendered stimulus in every swept layer. In each model, the sweep covers a subset of layers distributed throughout the network, whose size is given in {{t:tab:modelos}}. The four variants of every pair are processed while retaining their common identifier, a condition on which all subsequent paired tests depend.

The routine operates with a budget of 4,096 tokens per instruction and aborts the run if any stimulus reaches that limit instead of silently truncating it. The reason follows from Section 3.3.1: the study measures the activation of the final token, and in a truncated text that token would no longer be the end of the request. Its cone of influence would therefore cease to contain the complete instruction. In the 96-pair corpus, the longest agentic stimulus in Llama occupies 1,246 tokens, so no run reached the limit.

A qualification is needed concerning the scope of this reproducibility. Inference procedures for large-scale models in reduced precision are not strictly deterministic, since the order of floating-point operations may vary between runs and hardware configurations. Numerical uncertainty between runs is not quantified by corpus-sampling intervals. Every run retains its complete configuration, its activations and, where applicable, all its generations with their assigned labels, allowing any statistical analysis to be repeated without running the model again.

### 3.4. Evaluation and validation strategy

Evaluation is organised around two controls that ensure observed effects are attributable to the refusal mechanism rather than to artefacts of the design. The first is the benign control, formalised in Equation (10): the same format transition is applied to benign instructions, so that only the interaction between content and format is attributed to refusal. The second is separation between the data estimating the direction and the data evaluating its performance, which corrects the circularity bias described in Equation (6).

#### 3.4.1. Class separability and its estimation

In addition to measuring how much projection is reduced, it is important to establish whether the direction retains its capacity to distinguish harmful instructions from benign ones. AUROC is used for this purpose, taking projection as a score and estimating it by counting correctly ordered comparisons between nh harmful instructions and nb benign instructions:

<p class="equation">AUROĈ = (1/(n<sub>h</sub>n<sub>b</sub>)) Σ<sub>i=1…nh</sub> Σ<sub>j=1…nb</sub> [1(p<sub>h,i</sub> &gt; p<sub>b,j</sub>) + ½ 1(p<sub>h,i</sub> = p<sub>b,j</sub>)]. &nbsp; (14)</p>

It is the probability that a randomly selected harmful instruction projects higher than a benign one: its value is one under perfect separation and 0.5 at chance. Since it depends only on order, it is invariant to strictly increasing transformations of the score; reversing the axis changes its interpretation. The detection gap is the difference between conversational and agentic AUROC and answers a different question from the projection gap: the projection gap measures response intensity, while the detection gap asks whether that response remains sufficient to separate the classes.

In the regime n ≪ d, if points have sufficient affine rank, even arbitrary labellings may be linearly separable. High dimension alone does not guarantee this condition, but it does mean that a saturated training AUROC is not evidence of generalisation. To obtain an honest estimate, pairs are divided into K subsets, and for each fold k the direction r̂<sup>(l)</sup><sub>−k</sub> is estimated from pairs outside that fold and used to score only the held-out pairs:

<p class="equation">p̃<sup>(l)</sup>(x<sub>i</sub>) = x<sub>i</sub><sup>(l)</sup> · r̂<sub>−k(i)</sub><sup>(l)</sup>. &nbsp; (15)</p>

Folds are defined over pairs rather than individual instructions, because separating a harmful instruction from its counterpart would leak the very contrast being measured into the direction. All four variants of a held-out pair are scored using the same direction, preserving the pairing. The difference between the in-sample and out-of-sample values is called optimism and is reported explicitly. Because estimates from repeated cross-validation share training data, their variance is corrected following Nadeau and Bengio {{c:nadeau2003}}, multiplying it by 1/J + neval/ntrain, where J is the number of estimates. As a comparison reference, the complete procedure is repeated with the harm labels shuffled, estimating an empirical null reference with Monte Carlo uncertainty. Under absence of association, the expected reference value is 0.5, although a finite realisation may deviate from it.

#### 3.4.2. Statistical inference for the paired design

Since each instruction appears in both formats, the tests respect the pairing. The significance of Δph is assessed using a sign-flip permutation test: under the null hypothesis and the assumption that changes of condition are exchangeable within independent pairs, the sign of each δᵢ may be permuted. For B random sign vectors ε<sup>(b)</sup> ∈ {−1,+1}<sup>n</sup>, the following are calculated:

<p class="equation">T<sup>(b)</sup> = (1/n) Σ<sub>i=1…n</sub> ε<sub>i</sub><sup>(b)</sup>δ<sub>i</sub>, &nbsp; p = (1 + #{b : T<sup>(b)</sup> ≥ Δp})/(B+1). &nbsp; (16)</p>

This is accompanied by the paired t test in Equation (9) and a paired bootstrap confidence interval with bias correction and acceleration (BCa), which resamples complete pairs and adjusts the percentiles when the resampled distribution is asymmetric. An exact sign test is also included, using only the direction of change in each pair and no magnitudes. It is therefore insensitive to the scale of the projections and to positive rescaling of the direction. It is not, however, independent of the direction onto which observations are projected, since the sign of every difference depends on that direction. Protection against this dependence comes from estimating the direction out of sample. If k of the n pairs decrease, its one-sided p value is Σ<sub>j≥k</sub>C(n,j)2<sup>−n</sup>.

The procedure uses 10,000 permutations and resamples. When no permutation equals or exceeds the observation, p ≤ 10<sup>−4</sup> is reported at the procedure's resolution rather than as zero probability.

Inference requires an additional qualification. Cross-fitting prevents a pair from determining its own direction, but the folds share training data: their scores are not independent realisations of the complete fitting procedure. Intervals calculated by resampling scores that have already been obtained are conditional on those fits and do not replace resampling that re-estimates the entire direction. Sign tests on those scores are presented as diagnostics of consistency.

Testing a layer uses an explicit separation: 48 pairs select the layer and estimate the direction, while the other 48 provide inference. The two arms of each pair remain together. The denominators of the counts are stated, and the p value is compared with a Bonferroni threshold based on the number of layers. That correction does not replace the separation between selection and inference. The procedure is repeated over 200 partitions to describe stability, without treating them as new independent samples.

#### 3.4.3. Behaviour without intervention and causal test

The relationship between the refusal direction and behaviour is examined in two stages, applied to Llama-3.3-70B, Mistral-7B and Qwen3-8B. Both stages construct stimuli using the same function as extraction, so that the activations analysed and the responses generated correspond to identical token sequences. Generation is deterministic, using greedy decoding, with 64 answer tokens. Qwen3-8B reasons before answering within a delimited block; for that model, up to 1,024 reasoning tokens are permitted before the 64 answer tokens, and the label is assigned only to the answer after reasoning.

In the first stage, the behavioural baseline, the model responds without intervention to the 96 instructions in all four conditions, and the projection of each instruction is recorded in the same pass. This stage measures refusal rates by condition, the frequency with which the model responds by invoking an offered tool, and the relationship between projection and refusal. Changes in behaviour between formats are tested using exact McNemar tests on pairs with a response in both formats.

In the second stage, the causal test, harmful instructions that the model complied with in agentic format are selected, and αr̂<sup>(l)</sup> is added to the residual stream of layer l at every generation step. Dose is expressed in multiples of the shift measured at that layer with the full-sample direction, α = mΔph<sup>(l)</sup>, so that m = 1 restores exactly the projection removed by the agentic format. The values tested are m ∈ {0.5, 1, 2, 4}. Two controls accompany every dose. The first adds to the same instructions a random direction orthogonal to r̂<sup>(l)</sup> and of equal norm: if any perturbation of that magnitude induced refusal, the effect would not be specific to the direction. The second adds the refusal direction to benign instructions that the model complied with: a dose that induces refusal in both does not restore a safeguard but blocks indiscriminately. Projection is recorded before and after the intervention to verify that it produced the intended shift. Comparisons between the refusal direction and random direction use exact paired McNemar tests. In Llama, layers 76, 56 and 28 are intervened on, fixed before execution; in Mistral and Qwen3-8B, the layer chosen by selection with separate inference is used.

Responses are labelled as refusal, compliance or degeneration using a reproducible textual classifier. A response that invokes a tool is labelled compliance; other responses are labelled refusal when their beginning contains a negative statement in the first person. Degeneration is detected through the compressibility of the text, which distinguishes repetitions produced by excessive perturbations. In Qwen3-8B, a response whose reasoning does not close within the budget is labelled incomplete and excluded from paired comparisons. The label describes the first-turn response and does not establish that a tool executed the task; tools are simulated and are not executed. Every generation is retained with its label. The p values for different layers and doses are not corrected for multiple comparisons.

The test is limited to additions, which move the model towards refusal. The necessity test, which would consist of ablating the direction in conversational format and observing whether refusal disappears, is not performed, because it would deliberately produce a model without safeguards and harmful responses.

### 3.5. Ethical considerations

Research on a model's safety mechanisms must construct the stimulus that those mechanisms should refuse, and the dataset used here therefore contains instructions requesting harmful actions. Three features of the design limit the risk associated with that material. The instructions describe the requested action but do not contain the procedures, data or artefacts that would make it possible to carry it out. The tools defining the agentic condition are simulated and are not connected to any system, so no request can materialise in a real effect. Generated responses are labelled only to study refusal, compliance and degeneration, without executing the simulated tools, and interventions move the model only towards refusal.

The work is also diagnostic: the model is not retrained or intervened on to suppress its capacity to refuse, and no misaligned model is therefore produced as a by-product. The results describe where a safety mechanism degrades, which is dual-use information. The balance is considered to favour disclosure because the failure is already documented at the behavioural level and because a defence based on internal representations can only be built from this type of diagnosis. The models are used in accordance with their licences, which permit safety research. The study involves neither human subjects nor personal data: the names and accounts appearing in the instructions are fictitious.
