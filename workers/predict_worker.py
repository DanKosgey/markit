"""
workers/predict_worker.py — Model Prediction Worker.

Loads the fine-tuned XGBoost model from disk, consumes feature rows
from feature_q, generates class probabilities, and pushes trade signals
onto signal_q when confidence exceeds PROB_THRESHOLD.
"""

import time
import threading
import joblib

import numpy as np
import xgboost as xgb

import config as cfg
from utils.state import (
    log, stop_event, feature_q, signal_q,
    get_state, set_state, trading_halted,
)


class PredictWorker(threading.Thread):
    """
    Runs inference on each incoming feature row.
    Signal schema pushed to signal_q:
        {
            "direction":   1 | -1,          # 1=Bull/Long, -1=Bear/Short
            "p_bull":      float,
            "p_bear":      float,
            "p_neut":      float,
            "atr":         float,            # current ATR for SL/TP sizing
            "close":       float,            # entry reference price
            "datetime":    pd.Timestamp,
        }
    """

    def __init__(self):
        super().__init__(name="PredictWorker", daemon=True)
        self.model = None

    # ── internal ──────────────────────────────────────────────────────────────

    def _load_model(self):
        if not cfg.MODEL_PATH.exists():
            log.error(
                f"[PredictWorker] Model not found at {cfg.MODEL_PATH}. "
                "Run the notebook to train and save the model."
            )
            return False

        try:
            self.model = joblib.load(str(cfg.MODEL_PATH))
            log.info(f"[PredictWorker] ✓ Model loaded from {cfg.MODEL_PATH}")
            return True
        except Exception as e:
            log.error(f"[PredictWorker] Failed to load model: {e}")
            return False

    def _predict(self, feat_row) -> tuple:
        """Returns (p_bear, p_neut, p_bull) as floats."""
        proba = self.model.predict_proba(feat_row)
        return float(proba[0, 0]), float(proba[0, 1]), float(proba[0, 2])

    def _is_within_session(self, dt) -> bool:
        """Check if the bar falls within an active trading session (UTC hour)."""
        if cfg.SKIP_WEEKENDS and dt.weekday() >= 5:
            return False
        for start_h, end_h in cfg.TRADING_SESSIONS:
            if start_h <= dt.hour < end_h:
                return True
        return False

    # ── public ────────────────────────────────────────────────────────────────

    def run(self):
        log.info("[PredictWorker] Starting…")

        if not self._load_model():
            log.warning("[PredictWorker] Running in DEMO mode — random predictions.")

        while not stop_event.is_set():
            # ── Respect trading halt ───────────────────────────────────────────
            if trading_halted.is_set():
                log.debug("[PredictWorker] Trading halted — skipping prediction.")
                stop_event.wait(timeout=10)
                continue

            try:
                payload = feature_q.get(timeout=cfg.QUEUE_TIMEOUT)
            except Exception:
                continue

            try:
                feat_row = payload["features"]
                meta     = payload["meta"]
                dt       = meta["datetime"].iloc[0]
                close    = float(meta["close"].iloc[0])
                atr      = float(meta["atr"].iloc[0])
                rsi      = float(meta["rsi"].iloc[0])

                # ── Session gate ───────────────────────────────────────────────
                if not self._is_within_session(dt):
                    log.debug(f"[PredictWorker] Outside session at {dt.hour}h — skip.")
                    continue

                # ── Inference ─────────────────────────────────────────────────
                if self.model is not None:
                    p_bear, p_neut, p_bull = self._predict(feat_row)
                else:
                    # Demo: random probabilities
                    rng             = np.random.default_rng(int(time.time()))
                    raw             = rng.dirichlet([1, 2, 1])
                    p_bear, p_neut, p_bull = raw

                # ── Inference Display ─────────────────────────────────────────
                classes = ["BEAR", "NEUT", "BULL"]
                probs   = [p_bear, p_neut, p_bull]
                lead_idx = np.argmax(probs)
                lead_cls = classes[lead_idx]
                conf     = probs[lead_idx]

                log.info(
                    f"[PredictWorker] ❯ LIVE | Bear:{p_bear:.1%}  Neut:{p_neut:.1%}  Bull:{p_bull:.1%}  "
                    f"| Lead: {lead_cls} ({conf:.1%})"
                )

                # ── Signal threshold check ────────────────────────────────────
                direction = 0
                # Only execute if the lead class is Bull or Bear, and conf >= PROB_THRESHOLD
                if lead_cls == "BULL" and p_bull >= cfg.PROB_THRESHOLD:
                    direction = 1
                elif lead_cls == "BEAR" and p_bear >= cfg.PROB_THRESHOLD:
                    direction = -1

                if direction == 0:
                    log.debug("[PredictWorker] No signal — confidence below threshold.")
                    continue

                # ── Cooldown check ────────────────────────────────────────────
                last_sig = get_state("last_signal_ts")
                if (time.time() - last_sig) < cfg.SIGNAL_COOLDOWN:
                    log.debug("[PredictWorker] Signal cooldown active — skipping.")
                    continue

                set_state("last_signal_ts", time.time())

                signal = {
                    "direction": direction,
                    "p_bull":    p_bull,
                    "p_bear":    p_bear,
                    "p_neut":    p_neut,
                    "atr":       atr,
                    "close":     close,
                    "rsi":       rsi,
                    "datetime":  dt,
                }

                dir_str = "🟢 BULL" if direction == 1 else "🔴 BEAR"
                conf    = p_bull if direction == 1 else p_bear
                log.info(
                    f"[PredictWorker] ◆ SIGNAL  {dir_str}  "
                    f"conf={conf:.3f}  price={close:.5f}  ATR={atr:.5f}"
                )

                try:
                    signal_q.put_nowait(signal)
                except Exception:
                    log.warning("[PredictWorker] signal_q full — dropping signal.")

            except Exception as exc:
                log.exception(f"[PredictWorker] Prediction error: {exc}")

        log.info("[PredictWorker] Stopped.")
