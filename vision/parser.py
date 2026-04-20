"""
Helpers for normalizing Vision websocket payloads into execution-friendly state.
"""

from __future__ import annotations

from typing import Any


SUPPORTED_SIGNALS = {"buy", "sell", "wait", "reversal", "continuation"}


def normalize_timeframes(values: list[str] | tuple[str, ...] | None) -> list[str]:
    normalized = []
    for value in values or []:
        text = str(value).strip().upper()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _normalize_signal(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    return text if text in SUPPORTED_SIGNALS else None


def _extract_source_entry(container: dict[str, Any], timeframe: str) -> dict[str, Any] | None:
    return container.get(timeframe.lower()) or container.get(timeframe.upper())


def parse_signal_snapshot(
    payload: dict[str, Any],
    timeframes: list[str],
    htf_timeframes: list[str] | None = None,
) -> dict[str, Any]:
    normalized_timeframes = normalize_timeframes(timeframes)
    normalized_htf = set(normalize_timeframes(htf_timeframes))

    vision = payload.get("vision") or {}
    decision = payload.get("decision") or {}
    decision_ltf = decision.get("ltf_signals") or {}
    opencv = vision.get("opencv") or {}
    claude = vision.get("claude") or {}

    signals: dict[str, dict[str, Any]] = {}

    for timeframe in normalized_timeframes:
        ltf_entry = _extract_source_entry(decision_ltf, timeframe)
        if ltf_entry:
            signal = _normalize_signal(ltf_entry.get("signal"))
            if signal:
                signals[timeframe] = {
                    "signal": signal,
                    "confidence": float(ltf_entry.get("confidence") or 0.0),
                    "source": "decision.ltf_signals",
                    "reason": ltf_entry.get("reason"),
                }
                continue

        if timeframe in normalized_htf:
            signal = _normalize_signal(decision.get("htf_signal"))
            if signal:
                signals[timeframe] = {
                    "signal": signal,
                    "confidence": None,
                    "source": "decision.htf_signal",
                    "reason": decision.get("reason"),
                }
                continue

        for source_name, source_payload in (("vision.claude", claude), ("vision.opencv", opencv)):
            source_entry = _extract_source_entry(source_payload, timeframe)
            if not source_entry:
                continue

            signal = _normalize_signal(source_entry.get("signal"))
            if not signal:
                continue

            signals[timeframe] = {
                "signal": signal,
                "confidence": float(source_entry.get("confidence") or 0.0),
                "source": source_name,
                "reason": source_entry.get("reasoning"),
            }
            break

    return {
        "schema": payload.get("schema"),
        "service": payload.get("service"),
        "type": payload.get("type"),
        "event_id": payload.get("event_id"),
        "cycle_id": payload.get("cycle_id"),
        "symbol": payload.get("symbol"),
        "published_at": payload.get("published_at"),
        "source_state_ts": payload.get("source_state_ts"),
        "decision_action": decision.get("action"),
        "decision_reason": decision.get("reason"),
        "signals": signals,
    }
