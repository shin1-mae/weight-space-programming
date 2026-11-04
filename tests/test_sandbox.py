"""The out-of-process sandbox: results, proposed edits and guard rails."""

import subprocess

import numpy as np
import pytest

from wsp.harness.sandbox import PythonSandbox

W = {"layer.0.weight": np.arange(6, dtype=float).reshape(2, 3), "layer.0.bias": np.zeros(2)}


@pytest.fixture(scope="module")
def sb():
    return PythonSandbox(W, {"note": "test"}, timeout_s=20)


def test_result_stdout_and_edits(sb):
    r = sb.run(
        "import numpy as np\n"
        "print('hello', meta['note'])\n"
        "RESULT = {'sum': float(weights['layer.0.weight'].sum())}\n"
        "EDITS['layer.0.bias'] = weights['layer.0.bias'] + 1\n"
    )
    assert r.ok, r.error
    assert r.result == {"sum": 15.0}
    assert "hello test" in r.stdout
    np.testing.assert_array_equal(r.edits["layer.0.bias"], np.ones(2))


@pytest.mark.parametrize(
    "code, fragment",
    [
        ("import os", "not allowed"),
        ("import subprocess", "not allowed"),
        ("open('/etc/passwd')", "not allowed"),
        ("x = (1).__class__", "dunder"),
        ("eval('1+1')", "not allowed"),
        ("weights['layer.0.bias'][0] = 5", "read-only"),
        ("raise ValueError('boom')", "boom"),
    ],
)
def test_rejections(sb, code, fragment):
    r = sb.run(code)
    assert not r.ok
    assert fragment in r.error


def test_timeout():
    r = PythonSandbox(W, timeout_s=2).run("while True:\n    pass\n")
    assert not r.ok


def test_minimal_environment(monkeypatch):
    """Secrets in the parent environment are not passed to the sandbox process."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-secret")
    seen = {}
    real_run = subprocess.run

    def spy(*args, **kwargs):
        seen["env"] = kwargs.get("env")
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy)
    PythonSandbox(W, timeout_s=20).run("RESULT = 1")
    assert "ANTHROPIC_API_KEY" not in seen["env"]
    assert set(seen["env"]) <= {"PATH", "HOME", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"}
