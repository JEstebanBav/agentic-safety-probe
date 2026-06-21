"""
Unified data loader for all dataset sources.

Provides a consistent interface for loading:
1. Custom paired dataset (dataset_full.jsonl)
2. HarmAgent benchmark datasets (test/validation × harmful/benign × agent/chat)
3. Tool definitions (tool_definitions.json)

Key design: Each loader returns typed dataclass entries with a common interface
for the activation extraction pipeline, while preserving source-specific metadata.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Literal

DATA_DIR = Path(__file__).parent


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class DatasetEntry:
    """
    Entry from the custom paired dataset (dataset_full.jsonl).

    This dataset has IDENTICAL prompts in chat and agent conditions,
    enabling direct paired comparison of activations.
    """
    id: str
    base_id: str
    prompt: str
    category: str
    subtlety: str
    is_harmful: bool
    format: str                     # "chat" | "agent"
    variant: str                    # "chat_harmful" | "agent_harmful" | ...
    tools: List[str]                # tool names (empty for chat)
    tool_definitions: List[Dict]    # full schemas (empty for chat)
    system_prompt: str
    pair_id: str                    # links the 4 variants of same base prompt


@dataclass
class HarmAgentEntry:
    """
    Entry from HarmAgent benchmark datasets.

    These have richer evaluation metadata (grading functions, target tools)
    but prompts are NOT identical between chat and agent conditions.
    Use for behavioral validation, not paired activation comparison.
    """
    id: str
    id_original: str
    name: str
    category: str
    prompt: str
    is_harmful: bool
    split: str                              # "test" | "validation"
    condition: str                          # "agent" | "chat"
    # Agent-specific fields (absent in chat-only files)
    detailed_prompt: Optional[bool] = None
    hint_included: Optional[bool] = None
    target_functions: List[str] = field(default_factory=list)
    grading_function: Optional[str] = None


# ============================================================
# LOADERS
# ============================================================

def load_tool_definitions(path: Optional[str] = None) -> Dict[str, Dict]:
    """
    Load canonical tool definitions.

    Args:
        path: Path to tool_definitions.json. Default: project root.

    Returns:
        Dict mapping tool_name -> tool schema.
    """
    if path is None:
        # Check project root first, then data dir
        root_path = DATA_DIR.parent / "tool_definitions.json"
        data_path = DATA_DIR / "tool_definitions.json"
        path = str(root_path if root_path.exists() else data_path)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_custom_dataset(
    path: Optional[str] = None,
    variants: Optional[List[str]] = None,
) -> List[DatasetEntry]:
    """
    Load the custom paired dataset (dataset_full.jsonl).

    This is the PRIMARY dataset for activation space analysis because
    it uses identical prompts in chat and agent conditions.

    Args:
        path: Path to dataset_full.jsonl. Default: data/dataset_full.jsonl.
        variants: Filter to specific variants, e.g. ["chat_harmful", "agent_harmful"].
                  None = load all.

    Returns:
        List of DatasetEntry objects.
    """
    if path is None:
        path = str(DATA_DIR / "dataset_full.jsonl")

    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)

            if variants and raw.get("variant") not in variants:
                continue

            entries.append(DatasetEntry(
                id=raw["id"],
                base_id=raw["base_id"],
                prompt=raw["prompt"],
                category=raw["category"],
                subtlety=raw.get("subtlety", "unknown"),
                is_harmful=raw["is_harmful"],
                format=raw["format"],
                variant=raw["variant"],
                tools=raw.get("tools", []),
                tool_definitions=raw.get("tool_definitions", []),
                system_prompt=raw.get("system_prompt", ""),
                pair_id=raw.get("pair_id", raw["base_id"]),
            ))

    return entries


def load_harmagent_dataset(
    split: Literal["test", "validation"] = "test",
    condition: Literal["agent", "chat", "both"] = "both",
    harmfulness: Literal["harmful", "benign", "both"] = "both",
) -> List[HarmAgentEntry]:
    """
    Load HarmAgent benchmark datasets.

    These are used for BEHAVIORAL VALIDATION — checking if model actually
    refuses/complies — rather than for paired activation comparison.

    Args:
        split: "test" or "validation".
        condition: "agent" (tool-use), "chat" (text-only), or "both".
        harmfulness: "harmful", "benign", or "both".

    Returns:
        List of HarmAgentEntry objects.
    """
    entries = []

    # Determine which files to load
    files_to_load = []

    if condition in ("agent", "both"):
        if harmfulness in ("harmful", "both"):
            fname = f"harmful_behaviors_{split}_public.json" if split == "test" else f"harmful_behaviors_{split}.json"
            files_to_load.append((fname, True, "agent"))
        if harmfulness in ("benign", "both"):
            fname = f"benign_behaviors_{split}_public.json" if split == "test" else f"benign_behaviors_{split}.json"
            files_to_load.append((fname, False, "agent"))

    if condition in ("chat", "both"):
        fname = f"chat_public_{split}.json" if split == "test" else f"chat_{split}.json"
        fpath = DATA_DIR / fname
        if fpath.exists():
            files_to_load.append((fname, None, "chat"))  # harmfulness mixed in chat

    for fname, is_harmful_flag, cond in files_to_load:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            continue

        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        behaviors = data.get("behaviors", [])

        for b in behaviors:
            # For chat files, infer harmfulness from name or default
            if is_harmful_flag is None:
                is_harmful = "benign" not in b.get("name", "").lower()
            else:
                is_harmful = is_harmful_flag

            entries.append(HarmAgentEntry(
                id=b["id"],
                id_original=b.get("id_original", b["id"]),
                name=b.get("name", ""),
                category=b.get("category", "unknown"),
                prompt=b["prompt"],
                is_harmful=is_harmful,
                split=split,
                condition=cond,
                detailed_prompt=b.get("detailed_prompt"),
                hint_included=b.get("hint_included"),
                target_functions=b.get("target_functions", []),
                grading_function=b.get("grading_function"),
            ))

    return entries


# ============================================================
# DATASET SUMMARY UTILITIES
# ============================================================

def summarize_dataset(entries: List) -> Dict:
    """Print and return summary statistics for any dataset."""
    from collections import Counter

    if not entries:
        return {"count": 0}

    # Works for both DatasetEntry and HarmAgentEntry
    categories = Counter(e.category for e in entries)
    is_harmful_count = sum(1 for e in entries if e.is_harmful)

    summary = {
        "count": len(entries),
        "harmful": is_harmful_count,
        "benign": len(entries) - is_harmful_count,
        "categories": dict(categories),
    }

    if hasattr(entries[0], "variant"):
        summary["variants"] = dict(Counter(e.variant for e in entries))
    if hasattr(entries[0], "condition"):
        summary["conditions"] = dict(Counter(e.condition for e in entries))
    if hasattr(entries[0], "split"):
        summary["splits"] = dict(Counter(e.split for e in entries))

    return summary
