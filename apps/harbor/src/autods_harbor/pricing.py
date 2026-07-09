"""Model pricing + cost computation (stdlib-only).

Imported by both the container-side ``usage`` callback and the host-side
``atif`` converter, so it must not import ``harbor``, ``autods`` or langchain.

Cost is derived automatically from the model's live pricing (OpenRouter
``/models``); explicit ``AUTODS_PRICE_INPUT_PER_1M`` / ``AUTODS_PRICE_OUTPUT_PER_1M``
env vars override it for other OpenAI-compatible gateways.
"""

from __future__ import annotations

import json
import os
import urllib.request

# model id -> (prompt, completion, cache_read) $/token; cached per process run.
_PRICE_CACHE: dict[str, tuple[float | None, float | None, float | None]] = {}


def _price_per_1m(env_name: str) -> float | None:
    raw = os.getenv(env_name)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _fetch_openrouter_prices(
    model: str, base_url: str, api_key: str | None
) -> tuple[float | None, float | None, float | None]:
    if not base_url or not model or "openrouter" not in base_url:
        return (None, None, None)
    try:
        req = urllib.request.Request(
            base_url.rstrip("/") + "/models",
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            models = json.load(resp).get("data", [])
    except Exception:  # noqa: BLE001 - pricing is best-effort
        return (None, None, None)

    # Prefer an exact id, then a "<provider>/<model>" suffix match, avoiding the
    # ":free" variant when a bare/paid name was requested.
    best = None
    for entry in models:
        mid = entry.get("id", "")
        if mid == model:
            best = entry
            break
        if mid.endswith("/" + model) and ":free" not in mid:
            best = best or entry
    if best is None:
        return (None, None, None)
    pricing = best.get("pricing", {})
    try:
        prompt = float(pricing.get("prompt", 0) or 0)
        completion = float(pricing.get("completion", 0) or 0)
        cache = float(pricing.get("input_cache_read", pricing.get("prompt", 0)) or 0)
    except (TypeError, ValueError):
        return (None, None, None)
    return (prompt, completion, cache)


def resolve_prices(
    model: str | None, base_url: str | None, api_key: str | None
) -> tuple[float | None, float | None, float | None]:
    """Return ($/token) prompt, completion, cache_read.

    Priority: explicit AUTODS_PRICE_* env (per-1M) → OpenRouter live pricing.
    """
    env_in = _price_per_1m("AUTODS_PRICE_INPUT_PER_1M")
    env_out = _price_per_1m("AUTODS_PRICE_OUTPUT_PER_1M")
    if env_in is not None or env_out is not None:
        prompt = (env_in or 0.0) / 1_000_000
        return (prompt, (env_out or 0.0) / 1_000_000, prompt)

    key = model or ""
    if key not in _PRICE_CACHE:
        _PRICE_CACHE[key] = _fetch_openrouter_prices(model or "", base_url or "", api_key)
    return _PRICE_CACHE[key]


def compute_cost(
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> float | None:
    """Total $ cost, or None when no price is available."""
    prompt, completion, cache = resolve_prices(model, base_url, api_key)
    if prompt is None and completion is None:
        return None
    prompt = prompt or 0.0
    completion = completion or 0.0
    cache = cache if cache is not None else prompt
    non_cached = max(int(prompt_tokens or 0) - int(cached_tokens or 0), 0)
    cost = non_cached * prompt + int(cached_tokens or 0) * cache + int(completion_tokens or 0) * completion
    return round(cost, 6)
