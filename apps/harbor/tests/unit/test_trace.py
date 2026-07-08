"""Unit tests for the container-side trace + usage logic (no harbor needed)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from autods_harbor.entrypoint import _transcript_to_steps
from autods_harbor.usage import UsageCallback


def _msg(**kwargs):
    base = {
        "role": None,
        "content": "",
        "tool_call_id": None,
        "tool_name": None,
        "tool_args": None,
        "tool_result": None,
        "tool_status": None,
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_transcript_to_steps_orders_and_links_tools():
    messages = [
        _msg(role="assistant", content="Analyzing the dataset."),
        _msg(
            role="tool",
            tool_call_id="call_1",
            tool_name="run_python",
            tool_args={"code": "print(1)"},
            tool_result="1\n",
            tool_status="completed",
        ),
    ]
    steps = _transcript_to_steps(messages, "Solve titanic")

    assert steps[0] == {"source": "user", "message": "Solve titanic"}
    assert steps[1] == {"source": "agent", "message": "Analyzing the dataset."}
    tool_step = steps[2]
    assert tool_step["source"] == "agent"
    assert tool_step["tool_calls"][0]["id"] == "call_1"
    assert tool_step["tool_calls"][0]["name"] == "run_python"
    # Observation is linked to the tool call in the SAME step (ATIF requirement).
    assert tool_step["observations"][0]["source_call_id"] == "call_1"
    assert tool_step["observations"][0]["content"] == "1\n"


def test_transcript_to_steps_marks_tool_errors():
    messages = [
        _msg(
            role="tool",
            tool_call_id="c9",
            tool_name="run_shell",
            tool_result="boom",
            tool_status="error",
        )
    ]
    steps = _transcript_to_steps(messages, "task")
    assert steps[1]["observations"][0]["status"] == "error"


def test_transcript_to_steps_skips_duplicate_user_prompt():
    messages = [_msg(role="user", content="task")]
    steps = _transcript_to_steps(messages, "task")
    assert steps == [{"source": "user", "message": "task"}]


def test_usage_callback_accumulates_usage_metadata():
    cb = UsageCallback()
    message = SimpleNamespace(
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 20,
            "input_token_details": {"cache_read": 40},
        }
    )
    response = SimpleNamespace(generations=[[SimpleNamespace(message=message)]], llm_output=None)
    cb.on_llm_end(response)
    cb.on_llm_end(response)

    fm = cb.final_metrics()
    assert fm["total_prompt_tokens"] == 200
    assert fm["total_completion_tokens"] == 40
    assert fm["total_cached_tokens"] == 80


def test_usage_callback_cost_from_price_env(monkeypatch):
    monkeypatch.setenv("AUTODS_PRICE_INPUT_PER_1M", "1.0")
    monkeypatch.setenv("AUTODS_PRICE_OUTPUT_PER_1M", "2.0")
    cb = UsageCallback()
    cb._add(1_000_000, 500_000, 0)
    # 1.0 * 1 + 2.0 * 0.5 = 2.0
    assert cb.cost_usd() == pytest.approx(2.0)


def test_atif_conversion_roundtrip():
    pytest.importorskip("harbor")
    from harbor.models.trajectories.trajectory import Trajectory

    from autods_harbor.atif import build_trajectory

    raw = {
        "agent_name": "autods",
        "agent_version": "0.1.0",
        "model_name": "gpt-5",
        "instruction": "Solve titanic",
        "steps": [
            {"source": "user", "message": "Solve titanic"},
            {
                "source": "agent",
                "message": "run_python",
                "tool_calls": [{"id": "c1", "name": "run_python", "arguments": {"code": "1"}}],
                "observations": [{"source_call_id": "c1", "content": "1"}],
            },
        ],
        "final_metrics": {"total_prompt_tokens": 10, "total_completion_tokens": 2},
    }
    traj = build_trajectory(raw)
    assert traj.schema_version == "ATIF-v1.7"
    assert traj.final_metrics.total_steps == 2
    Trajectory.model_validate(traj.to_json_dict())
