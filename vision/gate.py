"""
Execution gate that compares model direction with the latest Vision timeframe signals.
"""

from __future__ import annotations

import time
from typing import Any

from vision.parser import normalize_timeframes


ALLOWED_SIGNALS_BY_DIRECTION = {
    1: {"buy", "wait", "reversal", "continuation"},
    -1: {"sell", "wait", "reversal", "continuation"},
}

OPPOSITE_SIGNAL_BY_DIRECTION = {
    1: "sell",
    -1: "buy",
}


def evaluate_vision_gate(
    *,
    direction: int,
    configured_timeframes: list[str],
    vision_signals: dict[str, dict[str, Any]] | None,
    connection_state: str | None,
    last_message_ts: float | None,
    max_age_seconds: float,
    require_fresh_signal: bool,
) -> dict[str, Any]:
    normalized_timeframes = normalize_timeframes(configured_timeframes)
    if not normalized_timeframes:
        return {"allowed": True, "reason": "Vision gate disabled - no timeframes configured", "matched": {}}

    if str(connection_state or "").lower() == "disabled":
        return {"allowed": True, "reason": "Vision gate disabled via config", "matched": {}}

    if not vision_signals:
        allowed = not require_fresh_signal
        reason = "No Vision signals available"
        return {"allowed": allowed, "reason": reason, "matched": {}, "state": connection_state}

    now = time.time()
    age_seconds = None if not last_message_ts else max(0.0, now - last_message_ts)
    if require_fresh_signal and (not last_message_ts or age_seconds is None or age_seconds > max_age_seconds):
        age_text = "unavailable" if age_seconds is None else f"{age_seconds:.1f}s"
        return {
            "allowed": False,
            "reason": f"Vision signal stale (age={age_text}, max={max_age_seconds:.1f}s)",
            "matched": {},
            "state": connection_state,
        }

    allowed_signals = ALLOWED_SIGNALS_BY_DIRECTION.get(direction, set())
    opposite_signal = OPPOSITE_SIGNAL_BY_DIRECTION.get(direction)
    missing = []
    blocked = []
    matched = {}

    for timeframe in normalized_timeframes:
        entry = (vision_signals or {}).get(timeframe)
        if not entry:
            missing.append(timeframe)
            continue

        signal = str(entry.get("signal") or "").strip().lower()
        matched[timeframe] = signal

        if signal == opposite_signal:
            blocked.append(f"{timeframe}={signal}")
            continue

        if signal not in allowed_signals:
            blocked.append(f"{timeframe}={signal or 'unknown'}")

    if blocked:
        side = "BUY" if direction == 1 else "SELL"
        return {
            "allowed": False,
            "reason": f"Vision blocked {side} due to opposite/incompatible signals: {', '.join(blocked)}",
            "matched": matched,
            "state": connection_state,
        }

    if missing and require_fresh_signal:
        return {
            "allowed": False,
            "reason": f"Vision missing configured timeframe signals: {', '.join(missing)}",
            "matched": matched,
            "state": connection_state,
        }

    return {
        "allowed": True,
        "reason": f"Vision matched: {', '.join(f'{tf}={sig}' for tf, sig in matched.items())}",
        "matched": matched,
        "state": connection_state,
        "age_seconds": age_seconds,
    }
