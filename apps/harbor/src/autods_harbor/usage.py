"""LangChain callback that accumulates LLM token usage across the AutoDS run.

Runs inside the task container (langchain is available; ``harbor`` is not). It is
attached to the pipeline ``config["callbacks"]`` — the same hook AutoDS already
uses for its langfuse handler — so it observes every LLM call across all
sub-agents (analyst/researcher/manager/coder/presenter/debugger).

Cost is computed from ``pricing.compute_cost`` (auto-derived model pricing). Note
that cost is also recomputed host-side in ``atif`` from the token counts, so it
populates even when this container image predates the pricing logic.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from autods_harbor.pricing import compute_cost


class UsageCallback(BaseCallbackHandler):
    """Sum input/output/cached tokens over all LLM calls and derive cost."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_tokens = 0
        self.call_count = 0
        self._model = model or os.getenv("AUTODS_MODEL")
        self._base_url = base_url or os.getenv("AUTODS_BASE_URL")
        self._api_key = api_key or os.getenv("AUTODS_API_KEY")

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
        self._add(usage.get("input_tokens", 0), usage.get("output_tokens", 0), cached)
        return True

    def _from_llm_output(self, llm_output: dict[str, Any] | None) -> bool:
        if not llm_output:
            return False
        usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        if not usage:
            return False
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens") or 0
        self._add(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), cached)
        return True

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
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
        return compute_cost(
            self._model,
            self._base_url,
            self._api_key,
            self.total_input_tokens,
            self.total_output_tokens,
            self.total_cached_tokens,
        )

    def final_metrics(self) -> dict[str, Any]:
        return {
            "total_prompt_tokens": self.total_input_tokens or None,
            "total_completion_tokens": self.total_output_tokens or None,
            "total_cached_tokens": self.total_cached_tokens or None,
            "total_cost_usd": self.cost_usd(),
        }
