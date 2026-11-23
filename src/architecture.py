"""
Where the decoder layers of a model live, whatever its architecture.

Text-only models (Llama, Mistral, Qwen3, gpt-oss) keep them at
`model.model.layers`. Multimodal checkpoints (Gemma 4, Qwen3.5/3.8, Muse
Glimmer, GLM-5.3) wrap the language model next to a vision tower, at
`model.model.language_model.layers` or similar. The extractor and the
intervention hooks both need the language model's layers, and reading the
wrong list - or a vision encoder's - would measure something else entirely,
so the lookup is defined once, here, and fails loudly when it is ambiguous.
"""

from typing import List, Optional

import torch.nn as nn

# Tried in order. The first that exists and has the expected length wins.
_KNOWN_PATHS = (
    "model.layers",                    # Llama, Mistral, Qwen3, gpt-oss
    "model.language_model.layers",     # Gemma 3/4, Qwen3.5, most VLMs (transformers v5)
    "language_model.model.layers",     # older VLM layouts
    "model.language_model.model.layers",
    "language_model.layers",
    "model.text_model.layers",
    "transformer.h",                   # GPT-2 style
)

# Submodules that hold layer lists which are NOT the language model.
_NOT_TEXT = ("vision", "visual", "image", "audio", "perception", "encoder",
             "projector", "mtp", "draft")


def text_config(config):
    """The language-model part of a (possibly multimodal) config."""
    getter = getattr(config, "get_text_config", None)
    if callable(getter):
        try:
            return getter()
        except TypeError:
            pass
    return getattr(config, "text_config", None) or config


def num_text_layers(model) -> int:
    return int(text_config(model.config).num_hidden_layers)


def _resolve(obj, path: str):
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def get_decoder_layers(model) -> nn.ModuleList:
    """
    The decoder layers of the language model, as a ModuleList.

    Known paths are tried first; failing those, the module tree is searched
    for the single ModuleList whose length is the text config's
    num_hidden_layers outside any vision/audio/encoder submodule. Zero or
    several candidates raise, rather than guessing.
    """
    n = num_text_layers(model)
    for path in _KNOWN_PATHS:
        layers = _resolve(model, path)
        if isinstance(layers, nn.ModuleList) and len(layers) == n:
            return layers

    candidates: List[str] = []
    for name, module in model.named_modules():
        if (isinstance(module, nn.ModuleList) and len(module) == n
                and not any(tag in name.lower() for tag in _NOT_TEXT)):
            candidates.append(name)
    if len(candidates) == 1:
        return _resolve(model, candidates[0])
    raise ValueError(
        f"Cannot locate the {n} decoder layers of {type(model).__name__}: "
        f"{len(candidates)} candidate ModuleLists {candidates[:5]}. Add its "
        "path to src/architecture.py:_KNOWN_PATHS.")


def decoder_layers_path(model) -> Optional[str]:
    """Dotted path of the decoder layers, for logs and the run manifest."""
    target = get_decoder_layers(model)
    for name, module in model.named_modules():
        if module is target:
            return name
    return None
