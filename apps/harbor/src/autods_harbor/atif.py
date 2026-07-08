"""Convert an AutoDS raw trace into a Harbor ATIF trajectory (host-side).

Runs in the harbor process (where ``harbor`` is importable), never inside the
task container. The entrypoint writes a light-weight ``autods_trace.json`` in
the agent log dir; this module maps it onto the ATIF v1.7 pydantic models and
writes the canonical ``trajectory.json`` Harbor discovers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harbor.models.trajectories.agent import Agent
from harbor.models.trajectories.final_metrics import FinalMetrics
from harbor.models.trajectories.metrics import Metrics
from harbor.models.trajectories.observation import Observation
from harbor.models.trajectories.observation_result import ObservationResult
from harbor.models.trajectories.step import Step
from harbor.models.trajectories.tool_call import ToolCall
from harbor.models.trajectories.trajectory import Trajectory

# Keep any single observation payload from bloating trajectory.json; the hub
# still renders the head/tail which is what a reviewer reads.
_MAX_OBSERVATION_CHARS = 200_000


@dataclass
class TrajectoryTotals:
    """Aggregate usage backfilled onto ``AgentContext`` after conversion."""

    n_input_tokens: int | None = None
    n_cache_tokens: int | None = None
    n_output_tokens: int | None = None
    cost_usd: float | None = None


def _truncate(text: str, limit: int = _MAX_OBSERVATION_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n\n[... {omitted} chars omitted ...]\n\n{tail}"


def _coerce_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"input": raw}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {}


def _build_metrics(raw: dict[str, Any] | None) -> Metrics | None:
    if not raw:
        return None
    prompt = raw.get("prompt_tokens")
    completion = raw.get("completion_tokens")
    cached = raw.get("cached_tokens")
    cost = raw.get("cost_usd")
    if all(v is None for v in (prompt, completion, cached, cost)):
        return None
    return Metrics(
        prompt_tokens=prompt,
        completion_tokens=completion,
        cached_tokens=cached or None,
        cost_usd=cost,
    )


def _build_step(step_id: int, raw: dict[str, Any]) -> Step:
    source = raw.get("source", "agent")
    message = raw.get("message")
    if not isinstance(message, str):
        message = "" if message is None else str(message)

    tool_calls: list[ToolCall] = []
    tool_call_ids: set[str] = set()
    for tc in raw.get("tool_calls") or []:
        tc_id = tc.get("id") or tc.get("tool_call_id")
        name = tc.get("name") or tc.get("function_name")
        if not tc_id or not name:
            continue
        tool_calls.append(
            ToolCall(
                tool_call_id=str(tc_id),
                function_name=str(name),
                arguments=_coerce_arguments(tc.get("arguments")),
            )
        )
        tool_call_ids.add(str(tc_id))

    results: list[ObservationResult] = []
    for obs in raw.get("observations") or []:
        source_call_id = obs.get("source_call_id")
        # ATIF requires source_call_id to reference a tool_call in THIS step.
        if source_call_id is not None and str(source_call_id) not in tool_call_ids:
            source_call_id = None
        content = obs.get("content")
        if isinstance(content, str):
            content = _truncate(content)
        results.append(
            ObservationResult(
                source_call_id=str(source_call_id) if source_call_id is not None else None,
                content=content,
            )
        )
    observation = Observation(results=results) if results else None

    if source == "agent":
        return Step(
            step_id=step_id,
            source="agent",
            timestamp=raw.get("timestamp"),
            model_name=raw.get("model_name"),
            message=message,
            reasoning_content=raw.get("reasoning") or None,
            tool_calls=tool_calls or None,
            observation=observation,
            metrics=_build_metrics(raw.get("metrics")),
        )
    # user / system steps: agent-only fields must be absent.
    return Step(
        step_id=step_id,
        source=source,
        timestamp=raw.get("timestamp"),
        message=message,
        observation=observation,
    )


def build_trajectory(raw: dict[str, Any]) -> Trajectory:
    """Build an ATIF ``Trajectory`` from an AutoDS raw trace dict."""
    raw_steps = raw.get("steps") or []
    steps: list[Step] = [_build_step(i + 1, s) for i, s in enumerate(raw_steps)]
    if not steps:
        # ATIF requires >=1 step; synthesize a minimal user step.
        steps = [Step(step_id=1, source="user", message=raw.get("instruction", ""))]

    fm = raw.get("final_metrics") or {}
    final_metrics = FinalMetrics(
        total_prompt_tokens=fm.get("total_prompt_tokens"),
        total_completion_tokens=fm.get("total_completion_tokens"),
        total_cached_tokens=fm.get("total_cached_tokens"),
        total_cost_usd=fm.get("total_cost_usd"),
        total_steps=len(steps),
    )

    return Trajectory(
        schema_version="ATIF-v1.7",
        session_id=raw.get("session_id"),
        agent=Agent(
            name=raw.get("agent_name", "autods"),
            version=str(raw.get("agent_version") or "unknown"),
            model_name=raw.get("model_name"),
            tool_definitions=raw.get("tool_definitions"),
        ),
        steps=steps,
        final_metrics=final_metrics,
    )


def convert_trace_file(raw_trace_path: Path, trajectory_out: Path) -> TrajectoryTotals:
    """Read a raw AutoDS trace, write ATIF ``trajectory.json``, return totals.

    Never raises on malformed input beyond surfacing a ValueError from the ATIF
    models; callers treat failures as best-effort (trace is diagnostic).
    """
    raw = json.loads(Path(raw_trace_path).read_text(encoding="utf-8"))
    trajectory = build_trajectory(raw)
    trajectory_out.parent.mkdir(parents=True, exist_ok=True)
    trajectory_out.write_text(
        json.dumps(trajectory.to_json_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fm = trajectory.final_metrics
    input_tokens = fm.total_prompt_tokens if fm else None
    cache_tokens = fm.total_cached_tokens if fm else None
    return TrajectoryTotals(
        n_input_tokens=input_tokens,
        n_cache_tokens=cache_tokens or None,
        n_output_tokens=fm.total_completion_tokens if fm else None,
        cost_usd=fm.total_cost_usd if fm else None,
    )
