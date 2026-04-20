"""
Consumes the Vision websocket feed and keeps the latest timeframe signals in shared state.
"""

from __future__ import annotations

import json
import threading
import time

import config as cfg
from utils.state import log, stop_event, update_state
from vision.parser import normalize_timeframes, parse_signal_snapshot

try:
    import websocket

    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False


class VisionSignalWorker(threading.Thread):
    def __init__(self):
        super().__init__(name="VisionSignalWorker", daemon=True)
        self.required_for_runtime = False
        self._timeframes = normalize_timeframes(getattr(cfg, "VISION_TIMEFRAMES", []))
        self._htf_timeframes = [cfg.MT5_HTF] if str(cfg.MT5_HTF).upper() in self._timeframes else []
        self._last_logged_signals: dict[str, str] = {}
        self._last_logged_action = None

    def _update_connection_state(self, state: str, **extra):
        payload = {"vision_connection_state": state}
        payload.update(extra)
        update_state(**payload)

    def _log_transitions(self, snapshot: dict):
        current_signals = snapshot.get("signals") or {}
        for timeframe in self._timeframes:
            current_signal = (current_signals.get(timeframe) or {}).get("signal")
            previous_signal = self._last_logged_signals.get(timeframe)
            if current_signal != previous_signal:
                source = (current_signals.get(timeframe) or {}).get("source")
                log.info(
                    f"[VisionWorker] Signal transition | timeframe={timeframe} | "
                    f"{previous_signal or 'none'} -> {current_signal or 'none'} | source={source or 'n/a'}"
                )
                if current_signal:
                    self._last_logged_signals[timeframe] = current_signal
                elif timeframe in self._last_logged_signals:
                    del self._last_logged_signals[timeframe]

        action = snapshot.get("decision_action")
        if action != self._last_logged_action:
            log.info(f"[VisionWorker] Decision action transition | {self._last_logged_action or 'none'} -> {action or 'none'}")
            self._last_logged_action = action

    @staticmethod
    def _format_signal_snapshot(signals: dict[str, dict] | None) -> str:
        if not signals:
            return "none"

        parts = []
        for timeframe in sorted(signals):
            entry = signals.get(timeframe) or {}
            signal = entry.get("signal") or "unknown"
            source = entry.get("source") or "n/a"
            confidence = entry.get("confidence")
            if confidence is None:
                parts.append(f"{timeframe}={signal}@{source}")
            else:
                parts.append(f"{timeframe}={signal}({float(confidence):.2f})@{source}")
        return ", ".join(parts)

    def _log_snapshot_summary(self, snapshot: dict):
        signals = snapshot.get("signals") or {}
        action = snapshot.get("decision_action") or "none"
        reason = snapshot.get("decision_reason") or "n/a"
        summary = (
            f"[VisionWorker] LIVE | action={action} | "
            f"signals={self._format_signal_snapshot(signals)} | "
            f"reason={reason}"
        )

        if signals or action != "none":
            log.info(summary)
        else:
            log.debug(summary)

    def run(self):
        if not getattr(cfg, "VISION_ENABLED", True):
            self._update_connection_state("disabled")
            log.info("[VisionWorker] Disabled via config.")
            return

        if not WEBSOCKET_AVAILABLE:
            self._update_connection_state("unavailable")
            log.error("[VisionWorker] websocket-client is not installed. Add it to the environment before running.")
            return

        if not self._timeframes:
            self._update_connection_state("disabled")
            log.warning("[VisionWorker] No Vision timeframes configured. Worker will not start.")
            return

        self._update_connection_state("starting")
        log.info(
            f"[VisionWorker] Starting | url={cfg.VISION_WS_URL} | "
            f"timeframes={','.join(self._timeframes)}"
        )

        while not stop_event.is_set():
            ws = None
            try:
                ws = websocket.create_connection(
                    cfg.VISION_WS_URL,
                    timeout=cfg.VISION_SOCKET_TIMEOUT,
                    enable_multithread=True,
                )
                ws.settimeout(cfg.VISION_SOCKET_TIMEOUT)
                self._update_connection_state("connected")
                log.info("[VisionWorker] Connected.")

                while not stop_event.is_set():
                    try:
                        raw_message = ws.recv()
                    except websocket.WebSocketTimeoutException:
                        continue

                    if raw_message is None:
                        raise RuntimeError("Vision websocket returned an empty frame")

                    payload = json.loads(raw_message)
                    snapshot = parse_signal_snapshot(payload, self._timeframes, self._htf_timeframes)
                    received_at_ts = time.time()

                    self._update_connection_state(
                        "connected",
                        vision_last_event_id=snapshot.get("event_id"),
                        vision_last_message_ts=received_at_ts,
                        vision_last_published_at=snapshot.get("published_at"),
                        vision_cycle_id=snapshot.get("cycle_id"),
                        vision_symbol=snapshot.get("symbol"),
                        vision_decision_action=snapshot.get("decision_action"),
                        vision_decision_reason=snapshot.get("decision_reason"),
                        vision_signals=snapshot.get("signals") or {},
                    )

                    if snapshot.get("signals"):
                        self._update_connection_state(
                            "connected",
                            vision_last_nonempty_message_ts=received_at_ts,
                            vision_last_nonempty_published_at=snapshot.get("published_at"),
                            vision_last_nonempty_signals=snapshot.get("signals") or {},
                        )

                    self._log_snapshot_summary(snapshot)
                    self._log_transitions(snapshot)
                    log.debug(
                        f"[VisionWorker] Snapshot received | event_id={snapshot.get('event_id')} | "
                        f"signals={snapshot.get('signals')}"
                    )

            except Exception as exc:
                self._update_connection_state("reconnecting")
                log.warning(f"[VisionWorker] Connection loop error: {exc}")
            finally:
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass

            if not stop_event.is_set():
                stop_event.wait(cfg.VISION_RECONNECT_SECONDS)

        self._update_connection_state("stopped")
        log.info("[VisionWorker] Stopped.")
