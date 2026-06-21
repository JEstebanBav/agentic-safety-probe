"""
Activation extractor module.

Extracts hidden state activations from specified layers
at specified token positions during model inference.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm


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

    def __init__(self, model, tokenizer):
        """
        Args:
            model: HuggingFace model with accessible hidden layers.
            tokenizer: Corresponding tokenizer.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.hooks = []
        self.activations = {}

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
            self.activations[layer_idx] = hidden_states.detach().cpu()
        return hook

    def _register_hooks(self, layers: List[int]):
        """Register forward hooks on specified layers."""
        self._remove_hooks()
        for layer_idx in layers:
            module = self._get_layer_module(layer_idx)
            hook = module.register_forward_hook(self._hook_fn(layer_idx))
            self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}

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
        self._register_hooks(layers)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=4096,
        ).to(self.model.device)

        # Forward pass
        with torch.no_grad():
            self.model(**inputs)

        # Extract activations at specified position
        result = {}
        seq_len = inputs["input_ids"].shape[1]

        # Handle negative indexing
        if position < 0:
            pos = seq_len + position
        else:
            pos = min(position, seq_len - 1)

        for layer_idx in layers:
            # activations[layer_idx] shape: (1, seq_len, hidden_dim)
            act = self.activations[layer_idx][0, pos, :]
            result[layer_idx] = act.float()  # Convert to float32 for analysis

        self._remove_hooks()
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
        self._register_hooks(layers)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        with torch.no_grad():
            self.model(**inputs)

        seq_len = inputs["input_ids"].shape[1]
        result = {}

        for layer_idx in layers:
            for pos in positions:
                actual_pos = pos if pos >= 0 else seq_len + pos
                actual_pos = min(actual_pos, seq_len - 1)
                act = self.activations[layer_idx][0, actual_pos, :]
                result[(layer_idx, pos)] = act.float()

        self._remove_hooks()
        return result
