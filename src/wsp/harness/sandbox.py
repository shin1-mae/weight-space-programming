"""Out-of-process Python sandbox in which an agent analyses (and proposes edits to) raw weights.

The agent's code never touches the live model. It runs in a separate interpreter
(``python -I``) that receives *copies* of the weights as read-only numpy arrays, and can only
*propose* new tensors through the ``EDITS`` dict. The parent process then applies proposals via
:class:`~wsp.harness.editor.WeightEditor`, i.e. inside a validated, auditable transaction.

Defence in depth (not a hard security boundary — use the ``sandbox`` Docker service with
``network_mode: none`` for untrusted models or prompts):

* AST allow-list of imports (numpy, math, ...) and a deny-list of dangerous builtins / dunder
  attribute access;
* restricted ``__builtins__`` with an import hook enforcing the allow-list at runtime;
* resource limits (CPU seconds, address space, file size) and a wall-clock timeout;
* a minimal environment (no API keys or proxy settings are inherited).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_ALLOWED_MODULES = (
    "numpy",
    "math",
    "json",
    "itertools",
    "functools",
    "collections",
    "statistics",
    "re",
    "heapq",
    "typing",
    "dataclasses",
)


@dataclass
class SandboxResult:
    ok: bool
    stdout: str
    error: str | None
    result: Any
    edits: dict[str, np.ndarray] = field(default_factory=dict)
    seconds: float = 0.0

    def summary(self, max_stdout: int = 4000) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "stdout": self.stdout[-max_stdout:],
            "error": self.error,
            "result": self.result,
            "proposed_edits": {k: list(v.shape) for k, v in self.edits.items()},
            "seconds": round(self.seconds, 3),
        }


class PythonSandbox:
    def __init__(
        self,
        weights: dict[str, np.ndarray],
        meta: dict[str, Any] | None = None,
        timeout_s: float = 60.0,
        memory_mb: int = 4096,
        allowed_modules: tuple[str, ...] = DEFAULT_ALLOWED_MODULES,
        threads: int | None = None,
    ):
        self.weights = {k: np.asarray(v) for k, v in weights.items()}
        self.meta = meta or {}
        self.timeout_s = timeout_s
        self.memory_mb = memory_mb
        self.allowed_modules = allowed_modules
        self.threads = threads or min(4, os.cpu_count() or 1)
        self.n_runs = 0

    def run(self, code: str) -> SandboxResult:
        self.n_runs += 1
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="wsp-sandbox-") as tmp:
            job = Path(tmp)
            np.savez(job / "weights.npz", **self.weights)
            (job / "meta.json").write_text(json.dumps(self.meta))
            (job / "code.py").write_text(code)
            (job / "config.json").write_text(
                json.dumps(
                    {
                        "timeout_s": self.timeout_s,
                        "memory_mb": self.memory_mb,
                        "allowed_modules": list(self.allowed_modules),
                    }
                )
            )
            threads = str(self.threads)
            env = {
                "PATH": "/usr/bin:/bin",
                "HOME": tmp,
                "OPENBLAS_NUM_THREADS": threads,
                "OMP_NUM_THREADS": threads,
                "MKL_NUM_THREADS": threads,
            }
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", "-m", "wsp.harness._sandbox_worker", str(job)],
                    env=env,
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s + 10,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return SandboxResult(
                    False, "", f"timeout after {self.timeout_s}s", None, seconds=time.perf_counter() - t0
                )
            out_path = job / "out.json"
            if not out_path.exists():
                err = (proc.stderr or "").strip()[-2000:] or f"sandbox exited with code {proc.returncode}"
                if proc.returncode < 0:
                    err += " (killed by a signal — most likely the CPU-time or memory limit)"
                return SandboxResult(False, proc.stdout[-4000:], err, None, seconds=time.perf_counter() - t0)
            out = json.loads(out_path.read_text())
            edits: dict[str, np.ndarray] = {}
            if (job / "edits.npz").exists():
                with np.load(job / "edits.npz") as z:
                    edits = {k: z[k] for k in z.files}
            return SandboxResult(
                ok=out["ok"],
                stdout=out["stdout"],
                error=out["error"],
                result=out["result"],
                edits=edits,
                seconds=time.perf_counter() - t0,
            )


def weights_from_module(model) -> dict[str, np.ndarray]:
    """Copy a torch module's parameters to float64 numpy arrays (sandbox input format)."""
    return {k: v.detach().cpu().double().numpy() for k, v in model.state_dict().items()}


def read_snippet(name: str) -> str:
    """Source of a packaged analysis snippet in ``wsp/agents/snippets``."""
    path = Path(__file__).resolve().parent.parent / "agents" / "snippets" / f"{name}.py"
    return path.read_text()


__all__ = ["DEFAULT_ALLOWED_MODULES", "PythonSandbox", "SandboxResult", "read_snippet", "weights_from_module"]
