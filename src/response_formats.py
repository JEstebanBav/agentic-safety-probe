"""
Reasoning/answer formats other than <think>...</think>.

The behaviour labels are read off the ANSWER, never the reasoning before it
(src/causal_test.py). Qwen3, Qwen3.8 and GLM mark the reasoning with
<think>...</think>, which the classifier understands directly. Two families
use structured messages instead, made of special tokens that a plain decode
throws away:

- harmony (gpt-oss): <|channel|>analysis<|message|>...<|end|>
  <|start|>assistant<|channel|>final<|message|>...<|return|>, and tool calls as
  ...commentary to=functions.NAME ...<|message|>{json}<|call|>.
- recipient (Muse Glimmer): <|start|>assistant to=self<|message|>...<|eom|>
  for reasoning, to=user for the answer, to=TOOL for a call (<|eot|>/<|eom|>).

For these the generated ids are decoded WITH special tokens, parsed into
messages, and rewritten in the canonical form the classifier already reads:

    <think>{reasoning}</think>{answer}     reasoning closed, answer started
    <think>{reasoning}                     still reasoning (budget ran out)
    {answer}                               no reasoning

A tool call in the answer becomes <tool_call>{"name":..., "arguments":...}
</tool_call> (harmony) or keeps its <atem:function_calls> markup (recipient),
both of which has_tool_call() recognises. Models without these formats are
untouched: their text is produced exactly as before.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

HARMONY = "harmony"
RECIPIENT = "recipient"


@dataclass
class Parsed:
    reasoning: List[str] = field(default_factory=list)
    answer: List[str] = field(default_factory=list)
    answer_started: bool = False
    answer_closed: bool = False

    def canonical(self) -> str:
        answer = "".join(self.answer)
        if not self.reasoning:
            return answer
        text = THINK_OPEN + "\n".join(self.reasoning)
        return text + THINK_CLOSE + answer if self.answer_started else text


def detect_format(tokenizer) -> Optional[str]:
    """HARMONY, RECIPIENT or None, from the chat template and vocabulary."""
    template = getattr(tokenizer, "chat_template", None) or ""
    if not isinstance(template, str):
        return None
    vocab = tokenizer.get_vocab()
    if "<|message|>" not in vocab or "<|start|>" not in vocab:
        return None
    if "<|channel|>" in vocab and "<|channel|>" in template:
        return HARMONY
    if "to=self" in template or "assistant to=" in template:
        return RECIPIENT
    return None


def _messages(raw: str, terminators):
    """(header, content, terminator) for each message, oldest first.

    terminator is the token that closed the message, or "" while it is still
    being generated.
    """
    out = []
    for chunk in ("<|start|>assistant" + raw).split("<|start|>"):
        if not chunk.strip():
            continue
        header, sep, body = chunk.partition("<|message|>")
        if not sep:
            out.append((header, None, ""))
            continue
        cut = [(body.find(t), t) for t in terminators if t in body]
        end, term = min(cut) if cut else (-1, "")
        content = body[:end] if end >= 0 else body
        out.append((header, content, term))
    return out


def _tool_call(name: str, args: str) -> str:
    try:
        payload = json.loads(args)
    except ValueError:
        payload = args.strip()
    return "<tool_call>" + json.dumps({"name": name, "arguments": payload},
                                      ensure_ascii=False) + "</tool_call>"


def parse_harmony(raw: str) -> Parsed:
    # <|end|> closes a message but not the turn (analysis, or a preamble
    # before a tool call); <|return|> and <|call|> end the turn.
    p = Parsed()
    for header, content, term in _messages(
            raw, ("<|end|>", "<|return|>", "<|call|>", "<|endoftext|>")):
        if content is None:
            continue
        chan = re.search(r"<\|channel\|>\s*([A-Za-z_]+)", header)
        chan = chan.group(1) if chan else ""
        to = re.search(r"to=([\w.\-]+)", header)
        to = to.group(1) if to else ""
        if to.startswith("functions.") or (to and to not in ("user", "assistant", "all")):
            p.answer.append(_tool_call(to.split("functions.", 1)[-1], content))
        elif chan == "analysis":
            p.reasoning.append(content.strip())
            continue
        else:
            p.answer.append(content)
        p.answer_started = True
        p.answer_closed = term in ("<|return|>", "<|call|>", "<|endoftext|>")
    return p


def parse_recipient(raw: str) -> Parsed:
    # <|eot|> ends the turn. A tool call ends with <|eom|> (the model then
    # waits for the tool), so it also ends what the model will say now.
    p = Parsed()
    for header, content, term in _messages(
            raw, ("<|eot|>", "<|eom|>", "<|end_of_text|>")):
        if content is None:
            continue
        to = re.search(r"to=([\w.\-]+)", header)
        to = to.group(1) if to else "user"
        if to == "self":
            p.reasoning.append(content.strip())
            continue
        p.answer.append(content)
        p.answer_started = True
        p.answer_closed = (term in ("<|eot|>", "<|end_of_text|>")
                           or (term == "<|eom|>" and to != "user"))
    return p


PARSERS: dict = {HARMONY: parse_harmony, RECIPIENT: parse_recipient}


def parser_for(fmt: Optional[str]) -> Optional[Callable[[str], Parsed]]:
    return PARSERS.get(fmt) if fmt else None
