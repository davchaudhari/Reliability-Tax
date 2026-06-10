"""Prompted-action protocol for multi-turn tool-use (tau-bench).

Design choice (documented as a caveat): instead of vLLM native tool-calling, the agent emits a
single JSON action per turn which we parse into a tau-bench `Action`. This keeps the whole loop on
the already-instrumented text `VLLMClient`, avoids depending on vLLM's tool-parser config, and lets
the reliability strategies operate on text (what they're built for). It is NOT tau-bench's official
tool-calling protocol, so results are for matched cross-strategy comparison, not leaderboard claims.

Action JSON contract (one object, one line):
  {"tool": "<tool_name>", "arguments": { ... }}        -> call a domain tool
  {"tool": "respond", "content": "<message to user>"}  -> reply to the user simulator
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

# tau_bench Action lives in the installed package; imported lazily so the $0 mock pipeline that
# doesn't need tau-bench still imports this module fine.
RESPOND = "respond"


def build_system_prompt(wiki: str, tools_info: list[dict]) -> str:
    """System prompt = domain policy (wiki) + tool catalog + the JSON action contract."""
    lines = [
        wiki.strip(),
        "",
        "# How to act",
        "At each step, choose exactly ONE action and output it as a SINGLE JSON object on one line,",
        "with no surrounding prose. Two forms:",
        '  {"tool": "<tool_name>", "arguments": {<args>}}   to call a tool',
        '  {"tool": "respond", "content": "<message>"}      to reply to the user',
        "",
        "# Available tools",
        "Each tool is given with its FULL JSON parameter schema. Match argument names, types, and",
        "nested structure EXACTLY — e.g. an array-of-objects parameter must be a list of objects",
        "with the specified keys, not a list of strings.",
    ]
    for t in tools_info:
        fn = t.get("function", t)
        name = fn.get("name", "?")
        desc = (fn.get("description", "") or "").strip().replace("\n", " ")
        params = fn.get("parameters", {}) or {}
        # Include the full parameter schema (compact) so the model can produce correctly-typed and
        # correctly-nested arguments — the same information native tool-calling would provide.
        schema = json.dumps(params, separators=(",", ":"))
        lines.append(f"- {name}: {desc}\n    parameters: {schema}")
    lines += [
        "",
        "Use 'respond' to ask the user for missing info or to give the final answer. Follow the",
        "policy above exactly. Output ONLY the JSON action.",
    ]
    return "\n".join(lines)


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_action(text: str) -> tuple[str, dict[str, Any]]:
    """Parse model text into (tool_name, kwargs). Robust to code fences and surrounding prose.

    Falls back to a 'respond' action carrying the raw text if no valid JSON action is found, so a
    malformed turn degrades to 'talk to the user' rather than crashing the episode (self_correct can
    then fix it on a later turn).
    """
    if not text:
        return RESPOND, {"content": ""}
    candidate = text.strip()
    # Strip ```json ... ``` fences if present.
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        candidate = candidate[candidate.find("{") :] if "{" in candidate else candidate
    m = _JSON_OBJ_RE.search(candidate)
    if not m:
        return RESPOND, {"content": text.strip()}
    try:
        obj = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return RESPOND, {"content": text.strip()}
    if not isinstance(obj, dict):
        return RESPOND, {"content": text.strip()}

    tool = obj.get("tool") or obj.get("name")
    if tool == RESPOND or tool is None:
        content = obj.get("content", obj.get("arguments", {}).get("content", "") if isinstance(obj.get("arguments"), dict) else "")
        return RESPOND, {"content": content or text.strip()}
    args = obj.get("arguments", obj.get("kwargs", {}))
    if not isinstance(args, dict):
        args = {}
    return str(tool), args


def action_key(tool: str, kwargs: dict[str, Any]) -> str:
    """Canonical, vote-stable key for an action (for self_consistency majority vote)."""
    if tool == RESPOND:
        # All 'respond' actions collapse to one bucket — voting is about WHICH tool to call,
        # not the exact wording of a reply.
        return "respond"
    try:
        return tool + "|" + json.dumps(kwargs, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return tool + "|" + str(sorted(kwargs.items()))


def action_to_assistant_text(tool: str, kwargs: dict[str, Any]) -> str:
    """Render the chosen action back into the conversation as the assistant's message."""
    if tool == RESPOND:
        return kwargs.get("content", "")
    return json.dumps({"tool": tool, "arguments": kwargs}, default=str)
