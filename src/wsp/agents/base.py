"""Common agent interface.

An *agent* receives a task (system prompt + user message) and a :class:`ToolRegistry`, and works
until it calls the registry's terminal tool. Two implementations share this interface:

* :class:`wsp.agents.llm.ClaudeAgent` — Claude decides which tools to call (needs an API key);
* the scripted reference agents in :mod:`wsp.agents.scripted` — deterministic policies that call
  the *same* tools, used for offline/CI runs and as a reproducible baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from wsp.harness.tools import ToolRegistry


@dataclass
class Task:
    name: str
    system: str
    user: str
    context: dict[str, Any] = field(default_factory=dict)  # extra info for scripted agents only


@dataclass
class AgentResult:
    agent: str
    final_answer: Any
    finished: bool
    n_tool_calls: int
    transcript: list[dict[str, Any]]
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "finished": self.finished,
            "n_tool_calls": self.n_tool_calls,
            "usage": self.usage,
            "stop_reason": self.stop_reason,
            "notes": self.notes,
            "final_answer": self.final_answer,
            "transcript": self.transcript,
        }


class Agent(Protocol):
    name: str

    def run(self, task: Task, tools: ToolRegistry) -> AgentResult: ...


def result_from_registry(agent: str, tools: ToolRegistry, stop_reason: str, **kw: Any) -> AgentResult:
    return AgentResult(
        agent=agent,
        final_answer=tools.final_answer,
        finished=tools.finished,
        n_tool_calls=len(tools.transcript),
        transcript=tools.transcript_dicts(),
        stop_reason=stop_reason,
        **kw,
    )
