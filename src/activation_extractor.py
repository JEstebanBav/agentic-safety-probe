"""
Activation extractor module.

Extracts hidden state activations from specified layers
at specified token positions during model inference.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from src.model_loader import encode_prompt


class ActivationExtractor:
    """
    Extracts activations from model hidden layers using forward hooks.
    
    Usage:
        extractor = ActivationExtractor(model, tokenizer)
        activations = extractor.extract_activations(
            prompts=["Hello world"],
            layers=[12, 16, 20, 24],
            position=-1,  # last token
        )
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_length: int = 4096,
        strict_truncation: bool = True,
    ):
        """
        Args:
            model: HuggingFace model with accessible hidden layers.
            tokenizer: Corresponding tokenizer.
            max_length: Token budget per prompt.
            strict_truncation: Raise instead of silently recording the
                activation of a truncated prompt, whose final token is not the
                end of the instruction.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.strict_truncation = strict_truncation
        self.hooks = []
        self.activations = {}
        self._hooked_layers: Optional[Tuple[int, ...]] = None
        # When set, hooks keep only this token position and copy a single
        # hidden_dim vector to host memory instead of the whole sequence.
        self._capture_position: Optional[int] = None
        # Token ids of the most recent prompt, kept so the caller can report
        # prompt lengths and show which token the activation was read from.
        self.last_input_ids: List[int] = []

    def _get_layer_module(self, layer_idx: int):
        """Get the residual stream output of a specific layer."""
        # Support common architectures
        if hasattr(self.model, "model"):  # Mistral, Llama
            return self.model.model.layers[layer_idx]
        elif hasattr(self.model, "transformer"):  # GPT-2 style
            return self.model.transformer.h[layer_idx]
        else:
            raise ValueError(
                f"Unsupported model architecture: {type(self.model)}"
            )

    def _hook_fn(self, layer_idx: int):
        """Create a hook function that captures activations."""
        def hook(module, input, output):
            # output is typically (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Slice on-device before the host copy. Moving the full
            # (1, seq_len, hidden_dim) tensor across the PCIe bus to keep one
            # row of it dominated extraction time: at 8192 dims and a
            # thousand-token agentic prompt that is ~16 MB per layer per
            # prompt, of which 8 KB is used.
            if self._capture_position is not None:
                hidden_states = hidden_states[:, self._capture_position, :]

            self.activations[layer_idx] = hidden_states.detach().to(
                "cpu", non_blocking=True
            )
        return hook

    def _register_hooks(self, layers: List[int]):
        """
        Register forward hooks on the given layers, reusing them when possible.

        Hooks used to be torn down and rebuilt for every prompt. The set of
        layers is fixed for a whole run, so the churn bought nothing.
        """
        key = tuple(layers)
        if self._hooked_layers == key and self.hooks:
            self.activations = {}
            return

        self._remove_hooks()
        for layer_idx in layers:
            module = self._get_layer_module(layer_idx)
            hook = module.register_forward_hook(self._hook_fn(layer_idx))
            self.hooks.append(hook)
        self._hooked_layers = key

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
        self._hooked_layers = None

    def close(self):
        """Release hooks. Safe to call repeatedly."""
        self._remove_hooks()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def extract_single(
        self,
        prompt: str,
        layers: List[int],
        position: int = -1,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract activations for a single prompt.

        Args:
            prompt: Formatted prompt string.
            layers: List of layer indices to extract from.
            position: Token position to extract (-1 = last token).

        Returns:
            Dict mapping layer_idx -> activation tensor of shape (hidden_dim,)
        """
        # Tokenize first: the capture position must be known before the forward
        # pass so the hooks can slice on-device.
        inputs = encode_prompt(
            self.tokenizer, prompt, self.max_length
        ).to(self.model.device)

        seq_len = inputs["input_ids"].shape[1]
        self.last_input_ids = inputs["input_ids"][0].tolist()
        pos = seq_len + position if position < 0 else min(position, seq_len - 1)

        # A truncated prompt no longer ends where the instruction ends, so the
        # "last token" would be an arbitrary interior token and the activation
        # would not mean what the analysis assumes. Fail rather than record it.
        if self.strict_truncation and seq_len >= self.max_length:
            raise ValueError(
                f"Prompt hit the {self.max_length}-token limit and was truncated; "
                "the final-token activation is no longer the end of the "
                "instruction. Raise max_length or shorten the prompt."
            )

        self._capture_position = pos
        self._register_hooks(layers)

        with torch.inference_mode():
            self.model(**inputs)

        # Hooks already reduced each layer to (1, hidden_dim).
        result = {
            layer_idx: self.activations[layer_idx][0].float()
            for layer_idx in layers
        }

        self._capture_position = None
        self.activations = {}
        return result

    def extract_batch(
        self,
        prompts: List[str],
        layers: List[int],
        position: int = -1,
        batch_size: int = 4,
        show_progress: bool = True,
    ) -> Dict[int, np.ndarray]:
        """
        Extract activations for multiple prompts.

        Args:
            prompts: List of formatted prompt strings.
            layers: List of layer indices to extract from.
            position: Token position to extract (-1 = last token).
            batch_size: Number of prompts to process at once.
            show_progress: Whether to show progress bar.

        Returns:
            Dict mapping layer_idx -> numpy array of shape (n_prompts, hidden_dim)
        """
        all_activations = {l: [] for l in layers}

        iterator = range(0, len(prompts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting activations")

        for i in iterator:
            batch = prompts[i:i + batch_size]

            # Process one at a time to handle variable lengths
            for prompt in batch:
                acts = self.extract_single(prompt, layers, position)
                for layer_idx in layers:
                    all_activations[layer_idx].append(acts[layer_idx].numpy())

        # Stack into arrays
        result = {}
        for layer_idx in layers:
            result[layer_idx] = np.stack(all_activations[layer_idx], axis=0)

        return result

    def extract_all_layers(
        self,
        prompts: List[str],
        position: int = -1,
        batch_size: int = 4,
        show_progress: bool = True,
    ) -> Dict[int, np.ndarray]:
        """
        Extract activations from ALL layers.

        Args:
            prompts: List of formatted prompt strings.
            position: Token position to extract.
            batch_size: Batch size for processing.
            show_progress: Whether to show progress bar.

        Returns:
            Dict mapping layer_idx -> numpy array of shape (n_prompts, hidden_dim)
        """
        n_layers = self.model.config.num_hidden_layers
        all_layers = list(range(n_layers))
        return self.extract_batch(
            prompts, all_layers, position, batch_size, show_progress
        )

    def extract_multi_position(
        self,
        prompt: str,
        layers: List[int],
        positions: List[int],
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        """
        Extract activations at multiple positions for a single prompt.
        Useful for analyzing how the signal evolves across token positions.

        Args:
            prompt: Formatted prompt string.
            layers: List of layer indices.
            positions: List of token positions.

        Returns:
            Dict mapping (layer_idx, position) -> activation tensor
        """
        # Full-sequence capture: this method needs every requested position.
        self._capture_position = None
        self._register_hooks(layers)

        inputs = encode_prompt(
            self.tokenizer, prompt, self.max_length
        ).to(self.model.device)

        with torch.inference_mode():
            self.model(**inputs)

        seq_len = inputs["input_ids"].shape[1]
        result = {}

        for layer_idx in layers:
            for pos in positions:
                actual_pos = pos if pos >= 0 else seq_len + pos
                actual_pos = min(actual_pos, seq_len - 1)
                act = self.activations[layer_idx][0, actual_pos, :]
                result[(layer_idx, pos)] = act.float()

        self.activations = {}
        return result
