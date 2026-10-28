"""The answer prompt, shared by every system. Only the context block differs between systems.

The fixed instructions come first and the context and question last, so
consecutive calls share a prompt prefix. The model answers in one line,
``ANSWER: <short answer> [episode ids]``; the scorer reads that line.
"""

from __future__ import annotations

import re

from .synth import TODAY
from .text import is_idk as _is_idk

IDK = "I don't know"

RULES = f"""Rules:
- Use only the context. If the context does not mention the person or building asked about, or does not give the value asked for, reply exactly: ANSWER: {IDK}
- A person with the same first name but a different surname is a different person.
- Values change over time. For "now", use the latest value. For a past date, use the value that held on that date.
- For a list, give every name, separated by commas. For a count, give the number.
- Reply with one line: ANSWER: <short answer> followed by the ids of the messages or notes that support it in brackets, like [a-017]."""


def answer_messages(broker_name: str, context: str, question: str) -> list[dict]:
    system = (f"You are the assistant of {broker_name}, a real-estate broker in the city of Velmora. "
              f"Today is {TODAY}. You answer the broker's questions about the broker's own clients and buildings.")
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"{RULES}\n\nContext:\n{context}\n\nQuestion: {question}"}]


_ANSWER_RE = re.compile(r"answer\s*:\s*(.*)", re.IGNORECASE)
_CITE_RE = re.compile(r"\b([abc]-\d{3})\b")


def answer_line(text: str) -> str:
    """The text after 'ANSWER:' on its line, or the first non-empty line if the model forgot the label."""
    for line in text.splitlines():
        m = _ANSWER_RE.search(line)
        if m:
            return m.group(1).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[0] if lines else ""


def citations(text: str) -> list[str]:
    return sorted(set(_CITE_RE.findall(text)))


_IDK_EXTRA = re.compile(r"\b(?:unknown|not (?:mentioned|stated|provided|specified|available|given)|"
                        r"no (?:information|data|record)|cannot (?:be )?(?:determined|answer))\b", re.IGNORECASE)


def is_idk(line: str) -> bool:
    return _is_idk(line) or bool(_IDK_EXTRA.search(line))
