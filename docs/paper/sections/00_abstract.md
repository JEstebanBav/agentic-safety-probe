# Transformation of the Refusal Direction in Agentic Language Models

**Juan Esteban Espitia Leiva and Helen Stefany Penagos Galvis**  
Research AI Safety

## Abstract

This work studies how the internal representation of harmful instructions changes when a language model moves from a conversational format to an agentic one, with an agent system message and simulated tools. A contrastive corpus of 96 harmful–benign pairs was built, with identical instruction text across formats and three levels of explicitness of the harmful intent, and Llama-3.3-70B, Mistral-7B, Qwen3-8B and Qwen3-32B were analysed in FP16 precision.

The refusal direction is estimated as a difference in means on the residual stream at the final token of the stimulus and evaluated on held-out pairs. In all four models it retains discriminative power in the agentic format, but loses part of it: its AUROC drops from 0.97 to 0.87 in Llama, from 0.92 to 0.77 in Mistral and Qwen3-8B, and from 0.95 to 0.87 in Qwen3-32B. The loss is specific to harmful content, persists in standardised units, survives correction for circularity and layer selection with separate inference, and reflects a reorientation of the direction, whose angle across formats ranges from 63° to 79°. Tool schemas account for most of the effect. Instructions whose harmful intent is contextual are the worst discriminated in the agentic format.

Behaviourally, refusal of harmful instructions drops in all three models evaluated, from 85% to 45% in Llama, and the models respond by calling tools. Restoring the lost projection brings refusal back in 40 to 47% of cases in Llama, against 4% with a random control; the effect is weaker in Mistral and absent in Qwen3-8B. A monitor fitted in the agentic format recovers the signal with AUROC between 0.93 and 0.98.

**Keywords:** mechanistic interpretability; refusal direction; agentic format; language agents; cross-validation; AI safety.
