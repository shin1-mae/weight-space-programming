"""Typed tool definitions shared by the LLM agent and the scripted reference agents.

Each tool has a Pydantic input model (→ JSON Schema for the Claude API ``tools`` parameter) and
a handler. :class:`ToolRegistry` validates inputs, catches errors (returned to the model as
``is_error`` tool results instead of crashing the loop), caps result size, and keeps a
transcript of every call for the experiment logs.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wsp.harness.editor import EditRejected, WeightEditor
from wsp.harness.inspector import BudgetExceeded, WeightInspector
from wsp.harness.sandbox import PythonSandbox


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass
class Tool:
    name: str
    description: str
    input_model: type[ToolInput]
    handler: Callable[[Any], Any]

    def schema(self) -> dict[str, Any]:
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        return {"name": self.name, "description": self.description, "input_schema": schema}


@dataclass
class ToolCall:
    name: str
    input: dict[str, Any]
    ok: bool
    output: str
    seconds: float


@dataclass
class ToolResult:
    content: str
    is_error: bool


def _dumps(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, default=lambda o: repr(o))


@dataclass
class ToolRegistry:
    max_result_chars: int = 12_000
    tools: dict[str, Tool] = field(default_factory=dict)
    transcript: list[ToolCall] = field(default_factory=list)
    final_answer: Any = None
    finished: bool = False
    terminal_tool: str | None = None

    def add(self, name: str, description: str, input_model: type[ToolInput], handler: Callable[[Any], Any]) -> None:
        if name in self.tools:
            raise ValueError(f"duplicate tool {name}")
        self.tools[name] = Tool(name, description, input_model, handler)

    def schemas(self) -> list[dict[str, Any]]:
        """Tool definitions in the shape the Claude Messages API expects."""
        return [t.schema() for t in self.tools.values()]

    def call(self, name: str, raw_input: dict[str, Any] | None) -> ToolResult:
        t0 = time.perf_counter()
        raw_input = raw_input or {}
        tool = self.tools.get(name)
        try:
            if tool is None:
                raise KeyError(f"unknown tool {name!r}; available: {sorted(self.tools)}")
            args = tool.input_model.model_validate(raw_input)
            out = tool.handler(args)
            text, is_error = _dumps(out), False
        except (ValidationError, KeyError, ValueError, IndexError, TypeError, EditRejected, BudgetExceeded) as e:
            text, is_error = _dumps({"error": type(e).__name__, "message": str(e)[:2000]}), True
        if len(text) > self.max_result_chars:
            text = text[: self.max_result_chars] + f"... [truncated {len(text) - self.max_result_chars} chars]"
        self.transcript.append(ToolCall(name, raw_input, not is_error, text[:2000], time.perf_counter() - t0))
        return ToolResult(text, is_error)

    def call_json(self, name: str, raw_input: dict[str, Any] | None = None) -> Any:
        """Convenience for scripted agents: call and decode, raising on tool errors."""
        res = self.call(name, raw_input)
        if res.is_error:
            raise RuntimeError(f"tool {name} failed: {res.content}")
        try:
            return json.loads(res.content)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"tool {name} returned a truncated result; request less data") from e

    def transcript_dicts(self) -> list[dict[str, Any]]:
        return [
            {"tool": c.name, "input": c.input, "ok": c.ok, "output": c.output[:500], "seconds": round(c.seconds, 3)}
            for c in self.transcript
        ]


# ----------------------------------------------------------------------------------------------
# Generic weight tools
# ----------------------------------------------------------------------------------------------


class NoInput(ToolInput):
    pass


class ParamName(ToolInput):
    name: str = Field(description="Parameter name as returned by list_parameters, e.g. 'net.0.weight'.")


class SvdInput(ParamName):
    k: int = Field(8, ge=1, le=64, description="Number of leading singular values to return.")


class ReadSliceInput(ParamName):
    index: str | None = Field(
        None, description="numpy-style index such as '3, :' or '0:8, 2'. Omit for the whole tensor."
    )
    max_elements: int = Field(512, ge=1, le=4096)


class TopKInput(ParamName):
    k: int = Field(10, ge=1, le=100)


class RunPythonInput(ToolInput):
    code: str = Field(
        description=(
            "Python source run in an isolated sandbox. Pre-defined: `weights` (dict name -> read-only "
            "numpy array, float64), `meta` (model card dict), `np`. Set `RESULT` to a JSON-serialisable "
            "value to return it; put proposed full-tensor replacements in `EDITS[name] = array` (they are "
            "NOT applied automatically — see apply_sandbox_edits). Allowed imports: numpy, math, json, "
            "itertools, functools, collections, statistics, re, heapq."
        )
    )


class EditInput(ToolInput):
    op: Literal["set", "add", "scale"]
    name: str = Field(description="Parameter name.")
    index: str | None = Field(None, description="numpy-style index; omit for the whole tensor.")
    value: float | list[Any] = Field(description="Value for set/add (scalar or nested list), factor for scale.")
    reason: str = Field(description="Why this edit fixes the problem (kept in the audit log).")


class ApplySandboxEditsInput(ToolInput):
    reason: str = Field(description="Why these sandbox-proposed tensors should replace the current ones.")


class RollbackInput(ToolInput):
    checkpoint: int | None = Field(None, description="Checkpoint id; omit for the latest checkpoint.")


class CheckpointInput(ToolInput):
    label: str = ""


def add_weight_tools(
    reg: ToolRegistry,
    inspector: WeightInspector,
    editor: WeightEditor | None = None,
    sandbox_factory: Callable[[], PythonSandbox] | None = None,
) -> None:
    """Register the generic Read (and, if an editor is given, Fix/Write) tools."""
    reg.add(
        "list_parameters",
        "List every parameter tensor of the target model with its shape.",
        NoInput,
        lambda _: inspector.list_parameters(),
    )
    reg.add(
        "tensor_stats",
        "Summary statistics of one parameter tensor (mean/std/min/max/norms, row/column norms).",
        ParamName,
        lambda a: inspector.tensor_stats(a.name),
    )
    reg.add(
        "svd",
        "Leading singular values, stable rank and effective rank of a weight matrix.",
        SvdInput,
        lambda a: inspector.svd(a.name, a.k),
    )
    reg.add(
        "read_slice",
        "Return the raw values of a slice of a parameter tensor.",
        ReadSliceInput,
        lambda a: inspector.read_slice(a.name, a.index, a.max_elements),
    )
    reg.add(
        "top_k",
        "Largest-magnitude entries of a parameter tensor with their indices.",
        TopKInput,
        lambda a: inspector.top_k(a.name, a.k),
    )

    last_sandbox: dict[str, Any] = {}

    if sandbox_factory is not None:

        def run_python(a: RunPythonInput) -> dict[str, Any]:
            res = sandbox_factory().run(a.code)
            last_sandbox["edits"] = res.edits
            return res.summary()

        reg.add(
            "run_python",
            "Run analysis code over the raw weights in an isolated Python sandbox (numpy available). "
            "Use it to re-implement the forward pass, compute statistics, or propose new tensors.",
            RunPythonInput,
            run_python,
        )

    if editor is None:
        return

    def apply_edit(a: EditInput) -> dict[str, Any]:
        if a.op == "scale":
            if not isinstance(a.value, (int, float)):
                raise ValueError("scale needs a scalar factor")
            rec = editor.scale(a.name, a.index, float(a.value), a.reason)
        elif a.op == "set":
            rec = editor.set(a.name, a.index, a.value, a.reason)
        else:
            rec = editor.add(a.name, a.index, a.value, a.reason)
        return {"applied": rec.to_dict(), "changed_fraction_total": editor.changed_fraction()}

    reg.add(
        "apply_edit",
        "Directly modify weights: set/add a value to, or scale, a slice of a parameter tensor. Every "
        "edit is logged; use checkpoint/rollback to undo.",
        EditInput,
        apply_edit,
    )

    def apply_sandbox_edits(a: ApplySandboxEditsInput) -> dict[str, Any]:
        edits = last_sandbox.get("edits") or {}
        if not edits:
            raise ValueError("the last run_python call proposed no EDITS")
        recs = [editor.set(name, None, value, a.reason).to_dict() for name, value in edits.items()]
        last_sandbox["edits"] = {}
        return {"applied": recs, "changed_fraction_total": editor.changed_fraction()}

    if sandbox_factory is not None:
        reg.add(
            "apply_sandbox_edits",
            "Apply the full-tensor replacements proposed in EDITS by the most recent run_python call.",
            ApplySandboxEditsInput,
            apply_sandbox_edits,
        )
    reg.add(
        "checkpoint",
        "Snapshot the current weights; returns a checkpoint id for rollback.",
        CheckpointInput,
        lambda a: {"checkpoint": editor.checkpoint(a.label)},
    )
    reg.add(
        "rollback",
        "Restore the weights of a checkpoint (default: latest), discarding later edits.",
        RollbackInput,
        lambda a: editor.rollback(a.checkpoint),
    )
    reg.add(
        "edit_summary",
        "Per-parameter summary of all changes relative to the original weights.",
        NoInput,
        lambda _: {
            "diff": editor.diff(),
            "changed_fraction": editor.changed_fraction(),
            "n_edits": len(editor.history),
        },
    )


def add_finish_tool(
    reg: ToolRegistry,
    input_model: type[ToolInput],
    description: str,
    name: str = "finish",
) -> None:
    """Terminal tool: stores the (validated) final answer and ends the agent loop."""

    def handler(a: ToolInput) -> dict[str, Any]:
        reg.final_answer = a.model_dump()
        reg.finished = True
        return {"status": "accepted"}

    reg.add(name, description, input_model, handler)
    reg.terminal_tool = name
