"""`autods-harbor` — run the AutoDS pipeline inside a Harbor task container.

Executes ``build_pipeline(workspace).astream({"task": instruction})``, records the
run with AutoDS's own ``_TranscriptRecorder`` (the exact streaming logic the
AutoDS server uses), and writes a light-weight raw trace + token usage to the
agent log dir. The host-side ``AutoDSAgent.populate_context_post_run`` converts
that raw trace into a Harbor ATIF ``trajectory.json``.

Runs in the container where ``autods`` + langchain are installed; it must not
import ``harbor``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from autods_harbor.constants import RAW_TRACE_FILENAME


class _MemoryService:
    """Minimal in-memory stand-in for AutoDS's SessionService.

    ``_TranscriptRecorder`` only needs message upsert/append (+ a no-op status
    setter) during streaming; we collect the resulting TranscriptMessages in
    insertion order to serialize afterwards.
    """

    def __init__(self) -> None:
        self._by_key: dict[str, Any] = {}
        self._order: list[str] = []

    def _put(self, key: str, message: Any) -> None:
        if key not in self._by_key:
            self._order.append(key)
        self._by_key[key] = message

    def upsert_transcript_message(self, _session_id: str, message: Any) -> None:
        key = getattr(message, "message_id", None) or f"auto-{len(self._order)}"
        self._put(str(key), message)

    def append_transcript_message(self, _session_id: str, message: Any) -> None:
        self._put(f"append-{len(self._order)}", message)

    def set_status(self, _session_id: str, _status: Any) -> None:
        return None

    def messages(self) -> list[Any]:
        return [self._by_key[key] for key in self._order]


def _text(value: Any) -> str:
    return (value or "") if isinstance(value, str) else ("" if value is None else str(value))


def _transcript_to_steps(messages: list[Any], instruction: str) -> list[dict[str, Any]]:
    """Map collected TranscriptMessages onto raw ATIF-friendly step dicts.

    Each tool call becomes its own agent step carrying the call + its observation
    (source_call_id == tool_call_id), which is exactly what ATIF requires and
    sidesteps having to link a tool result back to a specific assistant message.
    """
    steps: list[dict[str, Any]] = [{"source": "user", "message": instruction}]
    for message in messages:
        role = getattr(message, "role", None)
        content = _text(getattr(message, "content", "")).strip()
        tool_call_id = getattr(message, "tool_call_id", None)
        tool_name = getattr(message, "tool_name", None)

        if role == "user":
            continue  # initial task already captured as step 1
        if role == "assistant":
            if content:
                steps.append({"source": "agent", "message": content})
            continue
        if role == "tool" and tool_call_id and tool_name:
            tool_args = getattr(message, "tool_args", None)
            tool_result = getattr(message, "tool_result", None)
            status = getattr(message, "tool_status", None) or "completed"
            step: dict[str, Any] = {
                "source": "agent",
                "message": f"→ {tool_name}",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "name": tool_name,
                        "arguments": tool_args if isinstance(tool_args, (dict, str)) else {},
                    }
                ],
                "observations": [],
            }
            observation_content = tool_result if tool_result is not None else content
            if observation_content:
                step["observations"].append(
                    {
                        "source_call_id": tool_call_id,
                        "content": _text(observation_content),
                        "status": "error" if status == "error" else "completed",
                    }
                )
            steps.append(step)
            continue
        if content:
            steps.append({"source": "agent", "message": content})
    return steps


def _autods_version() -> str:
    try:
        import importlib.metadata as metadata

        return metadata.version("autods")
    except Exception:  # noqa: BLE001
        return "unknown"


def _recursion_limit(default: int = 800) -> int:
    """LangGraph super-step budget (AUTODS_RECURSION_LIMIT overrides)."""
    raw = os.getenv("AUTODS_RECURSION_LIMIT")
    try:
        return max(1, int(raw)) if raw else default
    except ValueError:
        return default


async def _run(instruction: str, workspace: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Imported here so `--help` stays cheap and import errors surface at run time.
    from autods.autods import _TranscriptRecorder
    from autods.pipeline import build_pipeline
    from autods_harbor.usage import UsageCallback

    service = _MemoryService()
    session = SimpleNamespace(id="autods-harbor", principal_id="autods-harbor")
    recorder = _TranscriptRecorder(service, session, instruction, None)
    usage = UsageCallback()

    config: dict[str, Any] = {
        # LangGraph aborts with GraphRecursionError once this many super-steps run.
        # Long multi-agent runs on full-size data need headroom; override with
        # AUTODS_RECURSION_LIMIT.
        "recursion_limit": _recursion_limit(),
        "configurable": {"thread_id": session.id},
        "callbacks": [usage],
    }
    pipeline = build_pipeline(str(workspace))
    try:
        async for chunk in pipeline.astream({"task": instruction}, config=config, stream_mode=["messages"]):
            if isinstance(chunk, tuple) and len(chunk) == 2 and isinstance(chunk[0], str):
                mode, data = chunk
            else:
                mode, data = "messages", chunk
            await recorder.handle_stream_chunk(mode, data)
    finally:
        recorder.finalize_assistant()

    steps = _transcript_to_steps(service.messages(), instruction)
    return steps, usage.final_metrics()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="autods-harbor", description=__doc__)
    parser.add_argument("--instruction-file", required=True, help="Path to the task instruction")
    parser.add_argument("--workspace", default=".", help="Working directory with staged task data")
    parser.add_argument("--trace-out", default=None, help="Where to write the raw AutoDS trace JSON")
    args = parser.parse_args(argv)

    instruction = Path(args.instruction_file).read_text(encoding="utf-8")
    workspace = Path(args.workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    os.chdir(workspace)
    trace_out = Path(args.trace_out) if args.trace_out else workspace / RAW_TRACE_FILENAME

    exit_code = 0
    steps: list[dict[str, Any]] = [{"source": "user", "message": instruction}]
    final_metrics: dict[str, Any] = {}
    error: str | None = None
    try:
        steps, final_metrics = asyncio.run(_run(instruction, workspace))
    except Exception as exc:  # noqa: BLE001 - record failure, still emit trace
        error = f"{type(exc).__name__}: {exc}"
        print(f"[autods-harbor] run failed: {error}", file=sys.stderr)
        steps.append({"source": "agent", "message": f"Run failed: {error}"})
        exit_code = 1

    raw_trace = {
        "session_id": "autods-harbor",
        "agent_name": "autods",
        "agent_version": _autods_version(),
        "model_name": os.getenv("AUTODS_MODEL"),
        "instruction": instruction,
        "steps": steps,
        "final_metrics": final_metrics,
    }
    if error:
        raw_trace["error"] = error
    try:
        trace_out.parent.mkdir(parents=True, exist_ok=True)
        trace_out.write_text(json.dumps(raw_trace, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[autods-harbor] wrote trace with {len(steps)} steps to {trace_out}")
    except Exception as exc:  # noqa: BLE001
        print(f"[autods-harbor] failed to write trace: {exc}", file=sys.stderr)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
