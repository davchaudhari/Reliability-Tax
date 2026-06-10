"""Answer normalization used by voting / reranking strategies.

To take a majority vote you need a canonical key per rollout so that semantically-identical
answers collide. This is benchmark-dependent. We provide a default that:
  * extracts the text after a 'FINAL:' marker if present (our convention), else uses the whole
    text, and
  * for tool-use answers, extracts and sorts function-call signatures so call order doesn't
    split the vote.

Real adapters can pass a custom `answer_key` to the strategy if they need stricter
normalization (e.g. BFCL's canonical call form).
"""
from __future__ import annotations

import re

_CALL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(([^()]*)\)")
_FINAL_RE = re.compile(r"FINAL:\s*(.+)", re.IGNORECASE | re.DOTALL)


def default_answer_key(text: str) -> str:
    """Canonical, vote-stable key for a rollout's answer."""
    if not text:
        return ""
    calls = _CALL_RE.findall(text)
    if calls:
        # Normalize each call to "name(sorted args)" and sort calls for order-invariance.
        norm = []
        for name, args in calls:
            arg_parts = [a.strip() for a in args.split(",") if a.strip()]
            arg_parts.sort()
            norm.append(f"{name}({','.join(arg_parts)})")
        norm.sort()
        return " ; ".join(norm)
    m = _FINAL_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()
