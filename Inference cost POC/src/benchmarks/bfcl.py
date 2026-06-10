"""Berkeley Function-Calling Leaderboard (BFCL) adapter.

Scope & honesty caveat (read this before trusting BFCL numbers from here)
-------------------------------------------------------------------------
BFCL's *multi-turn* categories are scored by EXECUTING model-produced and ground-truth call
strings against stateful backend Python classes and comparing instance state (see
`eval_checker/multi_turn_eval/multi_turn_checker.py` upstream). Faithfully reproducing that
requires importing BFCL's backend classes and `eval()`-ing calls — out of scope for this
budget-constrained study and a security/footgun risk to vendor.

What this adapter does instead, and says plainly:
  * Loads BFCL task entries from local data files (JSONL with a `.json` extension) if present.
  * Exposes them through the common `Task` interface (question turns -> messages; function docs
    -> tools).
  * Provides an AST-style structural check for SINGLE-TURN / single-call categories, comparing
    the model's emitted call (name + args) against the `possible_answer` set. This is the part
    BFCL itself does with AST checking and is cheap + judge-free.
  * For multi-turn categories it marks `requires_state_checker=True` in the payload and the
    `check()` here returns a conservative structural verdict, NOT the official state verdict.
    Use this for relative strategy comparison and pipeline wiring; do NOT report it as the
    official BFCL multi-turn accuracy. This limitation is surfaced in the README caveats.

If the data files are absent (e.g. during the $0 mock run), `tasks()` returns [] and the harness
simply uses a different benchmark. Nothing here downloads data implicitly.

Data layout expected (point `data_dir` at a checked-out BFCL `bfcl_eval/data/`):
  data_dir/BFCL_v4_<category>.json                      # questions
  data_dir/possible_answer/BFCL_v4_<category>.json      # ground truth
"""
from __future__ import annotations

import ast
import json
import os
import re
from typing import Any, Optional

from .base import Benchmark, Task, Verdict

# Single-turn categories that are genuinely AST-checkable here (subset; extend as needed).
# NOTE: BFCL v4 splits the simple category by language (simple_python / simple_java /
# simple_javascript). We default to simple_python for the smoke test (no JS/Java runtime needed).
AST_CHECKABLE = {
    "simple",
    "simple_python",
    "simple_java",
    "simple_javascript",
    "parallel",
    "multiple",
    "parallel_multiple",
}
MULTI_TURN = {
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
}


def _load_jsonl(path: str) -> list[dict]:
    """BFCL files are JSONL despite the .json extension."""
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class BFCLBenchmark:
    name = "bfcl"

    def __init__(
        self,
        data_dir: Optional[str] = None,
        categories: Optional[list[str]] = None,
        limit: Optional[int] = None,
    ) -> None:
        self.data_dir = data_dir
        self.categories = categories or ["simple"]
        self.limit = limit

    # ---- loading ----
    def tasks(self) -> list[Task]:
        if not self.data_dir or not os.path.isdir(self.data_dir):
            # No local data -> empty. Keeps the $0 pipeline independent of BFCL downloads.
            return []
        out: list[Task] = []
        for cat in self.categories:
            qpath = os.path.join(self.data_dir, f"BFCL_v4_{cat}.json")
            apath = os.path.join(self.data_dir, "possible_answer", f"BFCL_v4_{cat}.json")
            if not os.path.exists(qpath):
                continue
            questions = _load_jsonl(qpath)
            answers = {r["id"]: r for r in _load_jsonl(apath)} if os.path.exists(apath) else {}
            for row in questions:
                out.extend(self._to_tasks(cat, row, answers.get(row["id"], {})))
        if self.limit is not None:
            out = out[: self.limit]
        return out

    def _to_tasks(self, category: str, row: dict, answer: dict) -> list[Task]:
        # `question` is a list-of-lists (turns x messages). For AST-checkable single-turn
        # categories there is one turn; flatten it to a message list.
        question = row.get("question", [])
        raw_messages: list[dict[str, str]] = []
        if question and isinstance(question[0], list):
            for turn in question:
                raw_messages.extend(turn)
        else:
            raw_messages = list(question)
        tools = row.get("function", []) or row.get("functions", [])

        # BFCL "prompt mode": models that aren't called via the native tool API need the function
        # definitions IN the prompt, plus an explicit output format the AST checker can parse.
        # We prepend a system message carrying the schemas + format instruction, then merge any
        # BFCL-provided system message. This is what makes a plain chat model emit `fn(arg=val)`.
        messages = self._build_tool_prompt(tools, raw_messages)

        payload = {
            "category": category,
            "ground_truth": answer.get("ground_truth", answer.get("possible_answer")),
            "involved_classes": row.get("involved_classes", []),
            "requires_state_checker": category in MULTI_TURN,
        }
        return [
            Task(
                task_id=str(row["id"]),
                messages=messages,
                tools=tools,
                payload=payload,
                category=category,
            )
        ]

    @staticmethod
    def _build_tool_prompt(
        tools: list[dict], raw_messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Prepend a system prompt with the function schemas + a parseable output format.

        The output format ('respond ONLY with the call(s) as func(arg=value)') matches what the
        AST checker in this module parses. Any BFCL-supplied system message is appended after, so
        we don't clobber benchmark instructions.
        """
        lines = [
            "You are a function-calling assistant. You are given a set of functions.",
            "Select the function(s) needed to answer the user's request and call them.",
            "",
            "Available functions:",
        ]
        for fn in tools:
            name = fn.get("name", "<unknown>")
            desc = fn.get("description", "").strip()
            params = (fn.get("parameters", {}) or {}).get("properties", {}) or {}
            required = set((fn.get("parameters", {}) or {}).get("required", []) or [])
            param_parts = []
            for pname, pinfo in params.items():
                ptype = pinfo.get("type", "any")
                req = "required" if pname in required else "optional"
                param_parts.append(f"{pname}: {ptype} ({req})")
            sig = ", ".join(param_parts) if param_parts else "no parameters"
            lines.append(f"- {name}({sig}) — {desc}")
        lines += [
            "",
            "Respond ONLY with the function call(s) needed, in Python syntax, using the ACTUAL",
            "function name(s) from the list above and KEYWORD arguments. For example, if a listed",
            "function were get_area(width, height), you would respond: get_area(width=10, height=5)",
            "Use multiple calls separated by spaces if needed. Do not add any explanation.",
        ]
        system_prompt = "\n".join(lines)

        merged: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        merged.extend(raw_messages)
        return merged

    # ---- checking ----
    def check(
        self, task: Task, final_answer: str, *, transcript: Optional[list] = None
    ) -> Verdict:
        cat = task.payload.get("category", task.category)
        gt = task.payload.get("ground_truth")
        if cat in AST_CHECKABLE:
            ok = _ast_check(final_answer, gt, task.tools)
            return Verdict(success=ok, detail=f"ast:{cat}")
        if cat in MULTI_TURN:
            # Conservative structural check only — NOT the official state checker.
            ok = _loose_call_overlap(final_answer, gt)
            return Verdict(
                success=ok,
                detail=f"structural-only:{cat} (NOT official multi-turn state verdict)",
            )
        # Unknown category: be honest and fail closed.
        return Verdict(success=False, detail=f"unsupported-category:{cat}")


# ---------------------------------------------------------------------------
# Lightweight AST checking: parse a "fn(arg=val, ...)" call and compare name + kwargs
# against any allowed answer. BFCL allows multiple acceptable answers; we accept on a match
# to any. This mirrors the spirit of BFCL AST checking for single-turn categories.
# ---------------------------------------------------------------------------
_CALL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_\.]*\s*\([^()]*\)")


def _extract_calls(text: str) -> list[str]:
    return _CALL_RE.findall(text or "")


def _parse_call(call_str: str) -> Optional[tuple[str, dict[str, Any], list[Any]]]:
    """Parse 'fn(a, b, kw=val)' -> (full_dotted_name, kwargs, positional_args).

    Returns positional args separately so the checker can map them to parameter names using the
    function schema (BFCL ground truth is keyed by name, but models often emit positional calls).
    """
    try:
        node = ast.parse(call_str.strip(), mode="eval").body
        if not isinstance(node, ast.Call):
            return None
        name = _func_name(node.func)
        kwargs: dict[str, Any] = {}
        for kw in node.keywords:
            if kw.arg is None:
                continue
            kwargs[kw.arg] = _literal(kw.value)
        positional = [_literal(a) for a in node.args]
        return name, kwargs, positional
    except (SyntaxError, ValueError):
        return None


def _func_name(node: ast.AST) -> str:
    """Full dotted name: math.factorial stays 'math.factorial' (BFCL GT uses qualified names)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _func_name(node.value)
        return f"{prefix}.{node.attr}" if prefix != "<unknown>" else node.attr
    return "<unknown>"


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return "<non-literal>"


def _normalize_gt(gt: Any) -> list[dict[str, Any]]:
    """Ground truth in BFCL is a list (per-turn) of call strings, or nested. Flatten to parsed
    calls. Each parsed call is {name, args}. Robust to a few shapes."""
    parsed: list[dict[str, Any]] = []

    def _walk(x: Any) -> None:
        if isinstance(x, str):
            p = _parse_call(x)
            if p:
                parsed.append({"name": p[0], "args": p[1]})
        elif isinstance(x, dict):
            # Shape: {fn_name: {arg: [allowed,...]}}
            for name, args in x.items():
                parsed.append({"name": name, "args": args})
        elif isinstance(x, (list, tuple)):
            for item in x:
                _walk(item)

    _walk(gt)
    return parsed


def _param_order(tools: list[dict]) -> dict[str, list[str]]:
    """Map each function name -> ordered parameter names (from its schema), so positional model
    calls can be matched against BFCL's name-keyed ground truth."""
    order: dict[str, list[str]] = {}
    for fn in tools or []:
        name = fn.get("name")
        if not name:
            continue
        props = (fn.get("parameters", {}) or {}).get("properties", {}) or {}
        order[name] = list(props.keys())
    return order


def _names_match(model_name: str, gt_name: str) -> bool:
    """Exact, or suffix match (model emitted `factorial` while GT is `math.factorial`, or vice
    versa). BFCL ground truth uses fully-qualified names; models vary."""
    if model_name == gt_name:
        return True
    return model_name.split(".")[-1] == gt_name.split(".")[-1]


def _ast_check(final_answer: str, gt: Any, tools: Optional[list[dict]] = None) -> bool:
    if not gt:
        return False
    model_calls = [_parse_call(c) for c in _extract_calls(final_answer)]
    model_calls = [c for c in model_calls if c]
    if not model_calls:
        return False
    gt_calls = _normalize_gt(gt)
    param_order = _param_order(tools or [])
    # Accept if every ground-truth call name appears among model calls with compatible args.
    for gtc in gt_calls:
        if not any(_call_matches(mc, gtc, param_order) for mc in model_calls):
            return False
    return True


def _call_matches(
    model_call: tuple[str, dict, list], gtc: dict, param_order: dict[str, list[str]]
) -> bool:
    name, kwargs, positional = model_call
    if not _names_match(name, gtc["name"]):
        return False

    # Map positional args to parameter names using the schema order (keyed by the GT name, with
    # a fallback to the model's own name). Keyword args take precedence over positional.
    order = param_order.get(gtc["name"]) or param_order.get(name) or []
    args = dict(kwargs)
    for i, val in enumerate(positional):
        if i < len(order):
            args.setdefault(order[i], val)
        # Positional beyond known params can't be named -> ignore (will fail any required check).

    gt_args = gtc.get("args", {})
    for k, allowed in gt_args.items():
        allowed_list = allowed if isinstance(allowed, list) else [allowed]
        # BFCL marks an arg OPTIONAL by including "" (empty string) among its allowed values.
        # If the model omits such an arg, that's a valid match (the default was acceptable).
        optional = "" in allowed_list
        if k not in args:
            if optional:
                continue
            return False
        # Accept membership or stringified equality against the allowed values.
        if args[k] not in allowed_list and str(args[k]) not in [str(a) for a in allowed_list]:
            return False
    return True


def _loose_call_overlap(final_answer: str, gt: Any) -> bool:
    """Very weak structural overlap for multi-turn: does the model mention any GT function name?
    Explicitly NOT the official verdict; only keeps the pipeline runnable on multi-turn data."""
    gt_calls = _normalize_gt(gt)
    if not gt_calls:
        return False
    names = {c["name"] for c in gt_calls}
    return any(n in (final_answer or "") for n in names)
