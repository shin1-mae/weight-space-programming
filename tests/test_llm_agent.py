"""The Claude tool-use loop, exercised with a fake client (no network, no API key)."""

from types import SimpleNamespace

from wsp.agents.base import Task
from wsp.agents.llm import FALLBACK_BETA, ClaudeAgent
from wsp.harness.tools import ToolInput, ToolRegistry, add_finish_tool


class Echo(ToolInput):
    x: int


class Final(ToolInput):
    answer: str


def registry():
    reg = ToolRegistry()
    reg.add("echo", "echo x", Echo, lambda a: {"x": a.x})
    add_finish_tool(reg, Final, "finish", name="finish")
    return reg


def msg(content, stop="tool_use"):
    usage = SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=3, cache_creation_input_tokens=0)
    return SimpleNamespace(content=content, stop_reason=stop, usage=usage)


def tool_use(id_, name, inp):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=inp)


def text(t):
    return SimpleNamespace(type="text", text=t)


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        # snapshot the message list as it was at call time
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def fake_client(responses):
    messages = FakeMessages(responses)
    return SimpleNamespace(messages=messages, beta=SimpleNamespace(messages=messages)), messages


def test_loop_executes_tools_and_stops_at_finish():
    client, fm = fake_client(
        [
            msg([text("checking"), tool_use("t1", "echo", {"x": 1}), tool_use("t2", "echo", {"x": "bad"})]),
            msg([tool_use("t3", "finish", {"answer": "ok"})]),
        ]
    )
    reg = registry()
    res = ClaudeAgent(client=client, max_turns=5).run(Task("t", "sys", "do it"), reg)
    assert res.finished and res.final_answer == {"answer": "ok"} and res.stop_reason == "finished"
    assert res.usage["input_tokens"] == 20 and res.usage["cache_read_input_tokens"] == 6
    # second request carries both tool results (one error) in a single user message, ids matched
    second = fm.calls[1]["messages"]
    results = second[-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert results[0]["is_error"] is False and results[1]["is_error"] is True
    first = fm.calls[0]
    assert first["model"] == "claude-opus-5-5"
    assert first["cache_control"] == {"type": "ephemeral"}
    assert first["output_config"] == {"effort": "high"}
    assert first["betas"] == [FALLBACK_BETA] and first["fallbacks"] == "default"
    assert [t["name"] for t in first["tools"]] == ["echo", "finish"]


def test_reminder_when_model_stops_without_finishing():
    client, fm = fake_client([msg([text("done?")], stop="end_turn"), msg([tool_use("t1", "finish", {"answer": "y"})])])
    res = ClaudeAgent(client=client, use_fallbacks=False).run(Task("t", "sys", "go"), registry())
    assert res.finished
    assert "finish" in fm.calls[1]["messages"][-1]["content"]
    assert "betas" not in fm.calls[0]


def test_refusal_and_truncated_tool_input():
    client, _ = fake_client([msg([], stop="refusal")])
    res = ClaudeAgent(client=client).run(Task("t", "sys", "go"), registry())
    assert res.stop_reason == "refusal" and not res.finished

    client, fm = fake_client(
        [msg([tool_use("t1", "echo", {"x": 1})], stop="max_tokens"), msg([tool_use("t2", "finish", {"answer": "z"})])]
    )
    reg = registry()
    ClaudeAgent(client=client).run(Task("t", "sys", "go"), reg)
    assert fm.calls[1]["messages"][-1]["content"][0]["is_error"] is True
    assert [c.name for c in reg.transcript] == ["finish"]  # the truncated call was not executed
