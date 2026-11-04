"""Claude as the weight-space programmer: a tool-use loop over the harness tools.

Uses the official Anthropic Python SDK (``pip install 'weight-space-programming[llm]'``).
Credentials are resolved by the SDK (``ANTHROPIC_API_KEY``, ``ANTHROPIC_AUTH_TOKEN`` or an
``ant auth login`` profile).

Why a hand-written loop rather than the SDK's beta tool runner: the loop must stop exactly when
the terminal tool (``submit_*`` / ``finish``) is called, re-prompt when the model ends a turn
without calling it, and route every call through :class:`ToolRegistry` (validation, forward-pass
budget, audit transcript) — all of which the experiments rely on.

Request settings
----------------
* model ``claude-opus-5-5`` (override with ``--model`` / ``WSP_CLAUDE_MODEL``);
* thinking is always on for this model; depth is controlled with ``output_config.effort``;
* top-level ``cache_control`` (automatic prompt caching) — the system prompt and tool list are
  frozen for the whole run and the message list is append-only, so every turn re-uses the cached
  prefix;
* server-side refusal fallback (``fallbacks="default"``, beta ``server-side-fallback-2026-07-01``)
  is enabled by default; disable with ``use_fallbacks=False`` (e.g. on Bedrock/Vertex/Foundry,
  where it is not available).
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any

from wsp.agents.base import AgentResult, Task, result_from_registry
from wsp.harness.tools import ToolRegistry

DEFAULT_MODEL = os.environ.get("WSP_CLAUDE_MODEL", "claude-opus-5-5")
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeAgent:
    name = "claude"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = "high",
        max_turns: int = 40,
        max_tokens: int = 16_000,
        use_fallbacks: bool = True,
        client: Any = None,
    ):
        self.model = model
        self.effort = effort
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.use_fallbacks = use_fallbacks
        self.client = client

    def _get_client(self) -> Any:
        if self.client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover - depends on the optional extra
                raise RuntimeError("The Claude agent needs: pip install 'weight-space-programming[llm]'") from e
            self.client = anthropic.Anthropic(max_retries=4)
        return self.client

    def _create(self, **kwargs: Any) -> Any:
        client = self._get_client()
        if self.use_fallbacks:
            return client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
        return client.messages.create(**kwargs)

    @staticmethod
    def _add_usage(total: Counter, usage: Any) -> None:
        for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            total[key] += int(getattr(usage, key, 0) or 0)

    def run(self, task: Task, tools: ToolRegistry) -> AgentResult:
        finish = tools.terminal_tool or "finish"
        messages: list[dict[str, Any]] = [{"role": "user", "content": task.user}]
        usage: Counter = Counter()
        notes: list[str] = []
        reminders = 0
        stop = "max_turns"
        schemas = tools.schemas()  # frozen for the whole run (prompt-cache prefix stability)

        for _turn in range(self.max_turns):
            resp = self._create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=task.system,
                tools=schemas,
                messages=messages,
                output_config={"effort": self.effort},
                cache_control={"type": "ephemeral"},
            )
            self._add_usage(usage, resp.usage)
            if resp.stop_reason == "refusal":
                details = getattr(resp, "stop_details", None)
                notes.append(f"refusal: {getattr(details, 'category', None)}")
                stop = "refusal"
                break

            # Append the full content (thinking + text + tool_use blocks) unchanged.
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason == "pause_turn":
                continue

            tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
            if not tool_uses:
                if tools.finished:
                    stop = "finished"
                    break
                if reminders < 2:
                    reminders += 1
                    messages.append(
                        {"role": "user", "content": f"Continue the task. When you are done, call the `{finish}` tool."}
                    )
                    continue
                stop = str(resp.stop_reason)
                break

            results = []
            for block in tool_uses:
                if resp.stop_reason == "max_tokens":
                    # The tool input may be cut off — do not execute it.
                    content = json.dumps({"error": "output limit reached; tool input may be truncated — retry"})
                    results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": content, "is_error": True}
                    )
                    continue
                res = tools.call(block.name, block.input)
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": res.content, "is_error": res.is_error}
                )
            messages.append({"role": "user", "content": results})
            if tools.finished:
                stop = "finished"
                break

        return result_from_registry(f"claude:{self.model}", tools, stop, usage=dict(usage), notes=notes)
