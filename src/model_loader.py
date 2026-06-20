"""
Model loader module.

Handles loading the LLM with hooks for activation extraction.
Supports both full precision and quantized (4-bit) loading.
"""

import torch
from typing import Optional, Dict, List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def get_quantization_config() -> BitsAndBytesConfig:
    """4-bit quantization config for 20GB GPU."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def load_model_and_tokenizer(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    quantize: bool = True,
    device_map: str = "auto",
) -> tuple:
    """
    Load model and tokenizer for activation extraction.

    Args:
        model_name: HuggingFace model identifier.
        quantize: Whether to use 4-bit quantization (recommended for 20GB GPU).
        device_map: Device placement strategy.

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    print(f"Quantization: {'4-bit' if quantize else 'full precision'}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "device_map": device_map,
        "torch_dtype": torch.float16,
    }

    if quantize:
        load_kwargs["quantization_config"] = get_quantization_config()

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    # Disable gradients for inference
    for param in model.parameters():
        param.requires_grad = False

    print(f"Model loaded. Layers: {model.config.num_hidden_layers}")
    print(f"Hidden size: {model.config.hidden_size}")

    return model, tokenizer


def format_chat_prompt(
    tokenizer,
    user_message: str,
    system_message: Optional[str] = None,
) -> str:
    """
    Format a prompt in CHAT format (no tools).

    Args:
        tokenizer: The tokenizer with chat template.
        user_message: The user's message.
        system_message: Optional system prompt.

    Returns:
        Formatted prompt string.
    """
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def format_agent_prompt(
    tokenizer,
    user_message: str,
    tools: List[Dict],
    system_message: Optional[str] = None,
) -> str:
    """
    Format a prompt in AGENTIC format (with tool definitions).

    Args:
        tokenizer: The tokenizer with chat template.
        user_message: The user's message.
        tools: List of tool definitions in OpenAI-style format.
        system_message: Optional system prompt.

    Returns:
        Formatted prompt string with tool definitions.
    """
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})

    # Try native tool support first
    try:
        return tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback: inject tool definitions in system prompt
        tool_desc = "You have access to the following tools:\n\n"
        for tool in tools:
            func = tool.get("function", tool)
            tool_desc += f"- {func['name']}: {func.get('description', '')}\n"
            tool_desc += f"  Parameters: {func.get('parameters', {})}\n\n"

        tool_desc += (
            "To use a tool, respond with a JSON object: "
            '{\"tool\": \"tool_name\", \"arguments\": {...}}\n'
        )

        system_with_tools = tool_desc
        if system_message:
            system_with_tools = system_message + "\n\n" + tool_desc

        messages_with_tools = [
            {"role": "system", "content": system_with_tools},
            {"role": "user", "content": user_message},
        ]

        return tokenizer.apply_chat_template(
            messages_with_tools,
            tokenize=False,
            add_generation_prompt=True,
        )


# Available tools for experiments
TOOL_DEFINITIONS = {
    "send_email": {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to a specified recipient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body content"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    "web_search": {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
    "execute_code": {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Execute Python code in a sandboxed environment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                },
                "required": ["code"],
            },
        },
    },
    "post_social_media": {
        "type": "function",
        "function": {
            "name": "post_social_media",
            "description": "Post content to a social media platform.",
            "parameters": {
                "type": "object",
                "properties": {
                    "platform": {"type": "string", "description": "Target platform"},
                    "content": {"type": "string", "description": "Post content"},
                },
                "required": ["platform", "content"],
            },
        },
    },
    "file_operations": {
        "type": "function",
        "function": {
            "name": "file_operations",
            "description": "Read, write, or delete files on the filesystem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["read", "write", "delete"]},
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["action", "path"],
            },
        },
    },
    "database_query": {
        "type": "function",
        "function": {
            "name": "database_query",
            "description": "Execute a SQL query against the database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query to execute"},
                    "database": {"type": "string", "description": "Target database"},
                },
                "required": ["query", "database"],
            },
        },
    },
}
