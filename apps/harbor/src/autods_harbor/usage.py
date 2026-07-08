"""LangChain callback that accumulates LLM token usage across the AutoDS run.

Runs inside the task container (langchain is available; ``harbor`` is not). It is
attached to the pipeline ``config["callbacks"]`` — the same hook AutoDS already
uses for its langfuse handler — so it observes every LLM call across all
sub-agents (analyst/researcher/manager/coder/presenter/debugger).
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


def _price_per_1m(env_name: str) -> float | None:
    raw = os.getenv(env_name)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


class UsageCallback(BaseCallbackHandler):
    """Sum input/output/cached tokens (and optional cost) over all LLM calls."""

    def __init__(self) -> None:
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_tokens = 0
        self.call_count = 0

    def _add(self, input_tokens: int, output_tokens: int, cached_tokens: int) -> None:
        self.total_input_tokens += int(input_tokens or 0)
        self.total_output_tokens += int(output_tokens or 0)
        self.total_cached_tokens += int(cached_tokens or 0)
        self.call_count += 1

    def _from_usage_metadata(self, usage: dict[str, Any]) -> bool:
        if not usage:
            return False
        details = usage.get("input_token_details") or {}
        cached = details.get("cache_read") or details.get("cache_creation") or 0
        self._add(
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
            cached,
        )
        return True

    def _from_llm_output(self, llm_output: dict[str, Any] | None) -> bool:
        if not llm_output:
            return False
        usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        if not usage:
            return False
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens") or 0
        self._add(
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            cached,
        )
        return True

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        # Prefer per-generation usage_metadata (populated by langchain-openai even
        # when streaming); fall back to the aggregate llm_output block.
        counted = False
        try:
            for generations in getattr(response, "generations", []) or []:
                for generation in generations:
                    message = getattr(generation, "message", None)
                    usage = getattr(message, "usage_metadata", None) if message else None
                    if usage:
                        counted = self._from_usage_metadata(dict(usage)) or counted
        except Exception:  # noqa: BLE001 - usage accounting must never break a run
            pass
        if not counted:
            try:
                self._from_llm_output(getattr(response, "llm_output", None))
            except Exception:  # noqa: BLE001
                pass

    def cost_usd(self) -> float | None:
        in_price = _price_per_1m("AUTODS_PRICE_INPUT_PER_1M")
        out_price = _price_per_1m("AUTODS_PRICE_OUTPUT_PER_1M")
        if in_price is None and out_price is None:
            return None
        cost = 0.0
        cost += (self.total_input_tokens / 1_000_000) * (in_price or 0.0)
        cost += (self.total_output_tokens / 1_000_000) * (out_price or 0.0)
        return round(cost, 6)

    def final_metrics(self) -> dict[str, Any]:
        return {
            "total_prompt_tokens": self.total_input_tokens or None,
            "total_completion_tokens": self.total_output_tokens or None,
            "total_cached_tokens": self.total_cached_tokens or None,
            "total_cost_usd": self.cost_usd(),
        }
