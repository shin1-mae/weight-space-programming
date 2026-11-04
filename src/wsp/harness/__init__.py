"""Harness that lets an agent Read / Fix / Write model weights safely.

* :mod:`wsp.harness.inspector` – weight statistics, SVD, slices, metered activation probes
* :mod:`wsp.harness.editor` – transactional edits with rollback, audit log and hot-fix patches
* :mod:`wsp.harness.sandbox` – out-of-process Python sandbox over raw weight arrays
* :mod:`wsp.harness.tools` – Pydantic-typed tool registry shared by LLM and scripted agents
"""

from wsp.harness.editor import EditRecord, EditRejected, WeightEditor, apply_patch, state_hash
from wsp.harness.inspector import BudgetExceeded, ForwardBudget, WeightInspector, parse_index
from wsp.harness.sandbox import PythonSandbox, SandboxResult, read_snippet, weights_from_module
from wsp.harness.tools import ToolRegistry, add_finish_tool, add_weight_tools

__all__ = [
    "BudgetExceeded",
    "EditRecord",
    "EditRejected",
    "ForwardBudget",
    "PythonSandbox",
    "SandboxResult",
    "ToolRegistry",
    "WeightEditor",
    "WeightInspector",
    "add_finish_tool",
    "add_weight_tools",
    "apply_patch",
    "parse_index",
    "read_snippet",
    "state_hash",
    "weights_from_module",
]
