"""
Inference-Time Intervention module.

Adds a direction to the residual stream while the model generates, and reads
the projection onto a direction off the same forward pass.

What this module does and does not decide: it owns the HOOKS and the
GENERATION. Which prompts to steer, by how much, against which control, and
how to count the outcomes is experimental design and lives in
src/causal_test.py.

Design decisions:
- Hooks persist during the ENTIRE autoregressive generation (not just the
  first pass), so the push is present at every decoding step.
- The projection of the final prompt token is recorded before and after the
  push, on the first forward pass. That is the manipulation check: without it
  there is no evidence the hook changed what it was meant to change.
- The prompt is encoded by the same function the extractor uses, and a
  truncated prompt raises. An earlier run steered prompts whose user message
  had been truncated away; the model continued the tool catalogue and every
  output was labelled "complied".
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.causal_test import classify_response
from src.model_loader import encode_prompt


@dataclass
class InterventionResult:
    """Result of a single intervention experiment."""
    prompt: str
    alpha: float
    layer_idx: int
    behavior_before: str  # 'comply', 'refuse' or 'degenerate'
    behavior_after: str
    projection_before: float
    projection_after: float
    generated_text_before: str
    generated_text_after: str


class ActivationIntervention:
    """
    Adds a direction to activations during inference.

    Key design decisions:
    - Hooks are registered BEFORE calling model.generate() and remain
      active for every forward pass during autoregressive decoding.
    - By default only the last token position is modified (where the
      next-token decision happens); `all_positions` pushes every position.
    """

    def __init__(self, model, tokenizer, max_length: Optional[int] = None):
        self.model = model
        self.tokenizer = tokenizer
        if max_length is None:
            max_length = min(
                getattr(model.config, "max_position_embeddings", 8192), 8192
            )
        self.max_length = max_length
        self.hooks = []
        self._active = False
        # layer_idx -> (direction, alpha): what to add, and how much.
        self._steer: Dict[int, Tuple[np.ndarray, float]] = {}
        # layer_idx -> direction: what to project the final prompt token on.
        self._record: Dict[int, np.ndarray] = {}
        self._recorded: Dict[int, Dict[str, float]] = {}
        self._seen = set()
        self._all_positions = False
        self._tensors: Dict[Tuple, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _tensor(self, kind: str, layer_idx: int, vec: np.ndarray, like):
        """Direction as a tensor on the layer's device, built once per call."""
        key = (kind, layer_idx, like.device, like.dtype)
        if key not in self._tensors:
            self._tensors[key] = torch.as_tensor(
                np.asarray(vec), dtype=like.dtype, device=like.device
            )
        return self._tensors[key]

    def _make_hook(self, layer_idx: int):
        """Create a per-layer hook. Each hook checks the _active flag."""
        def hook(module, input, output):
            if not self._active:
                return output

            hidden_states = output[0] if isinstance(output, tuple) else output

            # generate() runs the whole prompt once and then one token at a
            # time. The first pass through a layer is therefore the one whose
            # last position is the final prompt token - the position the
            # extractor reads.
            first_pass = layer_idx not in self._seen
            self._seen.add(layer_idx)

            rec = self._record.get(layer_idx)
            if rec is not None and first_pass:
                r = self._tensor("record", layer_idx, rec, hidden_states)
                self._recorded[layer_idx] = {
                    "before": float(hidden_states[0, -1, :].float() @ r.float())
                }

            steer = self._steer.get(layer_idx)
            if steer is not None and steer[1] != 0.0:
                vec = self._tensor("steer", layer_idx, steer[0], hidden_states)
                if self._all_positions:
                    hidden_states += steer[1] * vec
                else:
                    hidden_states[:, -1, :] += steer[1] * vec

            if rec is not None and first_pass:
                self._recorded[layer_idx]["after"] = float(
                    hidden_states[0, -1, :].float() @ r.float()
                )

            # Modified in place, so `output` already carries the change.
            return output
        return hook

    def _get_layer_module(self, layer_idx: int):
        """Get layer module (supports Mistral/Llama and GPT-2 architectures)."""
        if hasattr(self.model, "model"):
            return self.model.model.layers[layer_idx]
        elif hasattr(self.model, "transformer"):
            return self.model.transformer.h[layer_idx]
        else:
            raise ValueError(f"Unsupported architecture: {type(self.model)}")

    def _get_num_layers(self) -> int:
        """Get total number of layers in the model."""
        if hasattr(self.model, "config"):
            return self.model.config.num_hidden_layers
        raise ValueError("Cannot determine number of layers")

    def _register_hooks(self, layer_indices: List[int]):
        """Register hooks on multiple layers. They persist until explicitly removed."""
        self._remove_hooks()
        for layer_idx in layer_indices:
            module = self._get_layer_module(layer_idx)
            hook = module.register_forward_hook(self._make_hook(layer_idx))
            self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self._tensors = {}

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate(self, prompt: str, max_new_tokens: int) -> Tuple[str, int]:
        """Greedy generation. Returns (completion, number of prompt tokens)."""
        inputs = encode_prompt(
            self.tokenizer, prompt, self.max_length
        ).to(self.model.device)
        n_prompt = inputs["input_ids"].shape[1]

        # A prompt cut at the token limit no longer ends with the user's
        # request: the model would be completing whatever came before it.
        if n_prompt >= self.max_length:
            raise ValueError(
                f"Prompt reached the {self.max_length}-token limit and was "
                "truncated; the user message may be gone. Refusing to "
                "generate from it."
            )

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        self._seen = set()
        self._recorded = {}
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                # No `temperature`: with do_sample=False it is unused, and
                # recent transformers versions warn or raise on 0.0. Greedy
                # decoding is already deterministic, which is what the
                # methodology's "generación determinista" requires.
                do_sample=False,
                pad_token_id=pad_id,
            )

        text = self.tokenizer.decode(
            outputs[0, n_prompt:], skip_special_tokens=True
        )
        return text, n_prompt

    def generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        """Generate text. Hooks remain active for every autoregressive step."""
        return self._generate(prompt, max_new_tokens)[0]

    def generate_recorded(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        steer: Optional[Dict[int, Tuple[np.ndarray, float]]] = None,
        record: Optional[Dict[int, np.ndarray]] = None,
        all_positions: bool = False,
    ) -> Dict:
        """
        Generate with an optional push, recording projections as it goes.

        Args:
            prompt: formatted prompt (already through the chat template).
            steer: layer_idx -> (unit direction, alpha). alpha * direction is
                added to that layer's output at every decoding step. None or
                alpha = 0 generates unsteered, which is how the baseline is
                produced through exactly the same code path.
            record: layer_idx -> unit direction. The final prompt token's
                activation at that layer is projected onto it, before and
                after any push applied there.
            all_positions: push every position instead of only the last.

        Returns:
            dict with 'text', 'n_prompt_tokens' and 'projections'
            (layer_idx -> {'before', 'after'}).
        """
        self._steer = dict(steer or {})
        self._record = dict(record or {})
        self._all_positions = all_positions
        self._register_hooks(sorted(set(self._steer) | set(self._record)))
        self._active = True
        try:
            text, n_prompt = self._generate(prompt, max_new_tokens)
        finally:
            self._active = False
            self._remove_hooks()
            self._steer, self._record = {}, {}
        return {
            "text": text,
            "n_prompt_tokens": n_prompt,
            "projections": dict(self._recorded),
        }

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify_behavior(self, generated_text: str) -> str:
        """'refuse', 'comply' or 'degenerate'. See causal_test.classify_response."""
        return classify_response(generated_text)

    # ------------------------------------------------------------------
    # Before/after interface (used by the w_agent stage)
    # ------------------------------------------------------------------

    def intervene(
        self,
        prompt: str,
        refusal_direction: np.ndarray,
        layer_idx: int,
        alpha: float,
        max_new_tokens: int = 150,
        apply_all_layers: bool = True,
        verbose: bool = True,
    ) -> InterventionResult:
        """
        Run intervention: generate without and with the direction added.

        Args:
            prompt: Formatted agent prompt.
            refusal_direction: The refusal direction vector (hidden_dim,).
            layer_idx: Primary layer index (used when apply_all_layers=False).
            alpha: Scaling factor for the direction.
            max_new_tokens: Max tokens to generate.
            apply_all_layers: If True, apply to all middle-to-late layers. The
                SAME alpha is then added at every one of them, so the total
                push grows with depth; small alphas already break generation.
                Prefer a single layer and a dose expressed in units of the
                measured gap (see causal_test.run_causal_test).
            verbose: Print first 200 chars of output for each condition.

        Returns:
            InterventionResult comparing before/after.
        """
        if apply_all_layers:
            n_layers = self._get_num_layers()
            target_layers = list(range(n_layers // 4, n_layers))
        else:
            target_layers = [layer_idx]
        record = {layer_idx: refusal_direction}

        before = self.generate_recorded(prompt, max_new_tokens, record=record)
        after = self.generate_recorded(
            prompt, max_new_tokens,
            steer={l: (refusal_direction, alpha) for l in target_layers},
            record=record,
        )
        behavior_before = self.classify_behavior(before["text"])
        behavior_after = self.classify_behavior(after["text"])

        if verbose:
            print(f"    [alpha={alpha:.1f}] Before({behavior_before}): {before['text'][:200]}")
            print(f"    [alpha={alpha:.1f}] After ({behavior_after}): {after['text'][:200]}")
            print()

        proj_b = before["projections"].get(layer_idx, {})
        proj_a = after["projections"].get(layer_idx, {})
        return InterventionResult(
            prompt=prompt[:200],
            alpha=alpha,
            layer_idx=layer_idx,
            behavior_before=behavior_before,
            behavior_after=behavior_after,
            projection_before=proj_b.get("before", float("nan")),
            projection_after=proj_a.get("after", float("nan")),
            generated_text_before=before["text"],
            generated_text_after=after["text"],
        )

    def batch_intervene(
        self,
        prompts: List[str],
        refusal_direction: np.ndarray,
        layer_idx: int,
        alpha: float,
        max_new_tokens: int = 150,
        apply_all_layers: bool = True,
    ) -> List[InterventionResult]:
        """Run intervention on multiple prompts."""
        results = []
        for i, prompt in enumerate(prompts):
            print(f"  Prompt {i+1}/{len(prompts)}:")
            result = self.intervene(
                prompt=prompt,
                refusal_direction=refusal_direction,
                layer_idx=layer_idx,
                alpha=alpha,
                max_new_tokens=max_new_tokens,
                apply_all_layers=apply_all_layers,
            )
            results.append(result)
        return results


def compute_intervention_success_rate(
    results: List[InterventionResult],
) -> Dict[str, float]:
    """
    Compute success rate of interventions.
    Success = model was complying before AND refuses after.

    A degenerate output after the push is neither: it is counted on its own
    line instead of being folded into "did not refuse", which is what made an
    over-steered model look like one the direction had no effect on.
    """
    total = len(results)
    complied_before = sum(1 for r in results if r.behavior_before == "comply")
    refused_after = sum(
        1 for r in results
        if r.behavior_before == "comply" and r.behavior_after == "refuse"
    )
    degenerate_after = sum(
        1 for r in results
        if r.behavior_before == "comply" and r.behavior_after == "degenerate"
    )
    already_refusing = sum(1 for r in results if r.behavior_before == "refuse")

    return {
        "total_prompts": total,
        "complied_before_intervention": complied_before,
        "refused_after_intervention": refused_after,
        "degenerate_after_intervention": degenerate_after,
        "success_rate": refused_after / complied_before if complied_before > 0 else 0.0,
        "already_refusing": already_refusing,
    }
