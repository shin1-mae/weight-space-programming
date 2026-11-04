"""Sandbox worker: ``python -I -m wsp.harness._sandbox_worker <job_dir>``.

Reads ``weights.npz``, ``meta.json``, ``code.py`` and ``config.json`` from the job directory,
validates and executes the code with restricted builtins, and writes ``out.json`` (+
``edits.npz`` when the code filled ``EDITS``).
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import io
import json
import sys
import traceback
from pathlib import Path
from typing import Any

FORBIDDEN_NAMES = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "input",
    "breakpoint",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "memoryview",
    "help",
    "exit",
    "quit",
}

SAFE_BUILTINS = [
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "hasattr",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "object",
    "pow",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
    "complex",
    "bytes",
    "chr",
    "ord",
    "hash",
    "id",
    "callable",
    "property",
    "staticmethod",
    "classmethod",
    "super",
    "type",
    "Exception",
    "ValueError",
    "TypeError",
    "KeyError",
    "IndexError",
    "RuntimeError",
    "ZeroDivisionError",
    "ArithmeticError",
    "AssertionError",
    "StopIteration",
    "NotImplementedError",
    "AttributeError",
    "True",
    "False",
    "None",
    "NotImplemented",
    "Ellipsis",
]


class CodeRejected(Exception):
    pass


def validate(code: str, allowed: set[str]) -> ast.Module:
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in allowed:
                    raise CodeRejected(f"import of {alias.name!r} is not allowed (allowed: {sorted(allowed)})")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in allowed:
                raise CodeRejected(f"import from {node.module!r} is not allowed (allowed: {sorted(allowed)})")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise CodeRejected(f"use of {node.id!r} is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            raise CodeRejected(f"dunder attribute access ({node.attr!r}) is not allowed")
    return tree


def to_jsonable(x: Any, depth: int = 0) -> Any:
    import numpy as np

    if depth > 8:
        return repr(x)[:200]
    if isinstance(x, dict):
        return {str(k): to_jsonable(v, depth + 1) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v, depth + 1) for v in x]
    if isinstance(x, np.ndarray):
        if x.size > 4096:
            return {"ndarray_shape": list(x.shape), "note": "too large to return; summarise it instead"}
        return to_jsonable(x.tolist(), depth + 1)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    return repr(x)[:500]


def set_limits(cfg: dict[str, Any]) -> None:
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX
        return
    cpu = int(cfg.get("timeout_s", 60)) + 1
    mem = int(cfg.get("memory_mb", 4096)) * 1024 * 1024
    for res, val in (
        (resource.RLIMIT_CPU, cpu),
        (resource.RLIMIT_AS, mem),
        (resource.RLIMIT_FSIZE, 256 * 1024 * 1024),
    ):
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(res, (val, val))


def main(job_dir: str) -> int:
    job = Path(job_dir)
    cfg = json.loads((job / "config.json").read_text())
    allowed = set(cfg.get("allowed_modules", []))
    code = (job / "code.py").read_text()
    meta = json.loads((job / "meta.json").read_text())

    import numpy as np  # loaded before limits are applied

    with np.load(job / "weights.npz") as z:
        weights = {k: z[k] for k in z.files}
    for v in weights.values():
        v.setflags(write=False)  # analysis code must copy before modifying

    set_limits(cfg)

    real_import = builtins.__import__

    def guarded_import(name, globals_=None, locals_=None, fromlist=(), level=0):
        if level or name.split(".")[0] not in allowed:
            raise ImportError(f"import of {name!r} is not allowed in the sandbox")
        return real_import(name, globals_, locals_, fromlist, level)

    safe = {n: getattr(builtins, n) for n in SAFE_BUILTINS if hasattr(builtins, n)}
    safe["__import__"] = guarded_import
    safe["__build_class__"] = builtins.__build_class__
    env: dict[str, Any] = {
        "__builtins__": safe,
        "__name__": "__sandbox__",
        "weights": weights,
        "meta": meta,
        "EDITS": {},
        "RESULT": None,
        "np": np,
    }

    buf = io.StringIO()
    ok, err = True, None
    try:
        tree = validate(code, allowed)
        with contextlib.redirect_stdout(buf):
            exec(compile(tree, "<agent>", "exec"), env)
    except CodeRejected as e:
        ok, err = False, f"CodeRejected: {e}"
    except BaseException:  # report everything, including MemoryError / SystemExit
        ok, err = False, traceback.format_exc(limit=6)[-3000:]

    edits_raw = env.get("EDITS") or {}
    edits: dict[str, np.ndarray] = {}
    if isinstance(edits_raw, dict):
        for k, v in edits_raw.items():
            try:
                edits[str(k)] = np.asarray(v, dtype=np.float64)
            except Exception:
                ok, err = False, f"EDITS[{k!r}] is not array-like"
    if edits:
        np.savez(job / "edits.npz", **edits)
    out = {"ok": ok, "stdout": buf.getvalue()[-20000:], "error": err, "result": to_jsonable(env.get("RESULT"))}
    (job / "out.json").write_text(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
