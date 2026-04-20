"""
workers/predict_worker.py - Model Prediction Worker.

Consumes aligned feature rows, runs model inference, and pushes trade
signals onto signal_q.
"""

import threading
import time
from pathlib import Path

import joblib
import numpy as np

import config as cfg
from utils.state import feature_q, get_state, log, set_state, signal_q, stop_event, trading_halted


class PredictWorker(threading.Thread):
    """
    Runs inference on each incoming feature row.
    """

    def __init__(self):
        super().__init__(name="PredictWorker", daemon=True)
        self.model = None

    @staticmethod
    def _resolve_model_path() -> Path | None:
        configured_path = Path(cfg.MODEL_PATH)
        if configured_path.exists():
            return configured_path

        candidate_dirs = [Path(cfg.MODELS_DIR), Path(cfg.MODELS_DIR) / "backup"]
        candidates = []
        for directory in candidate_dirs:
            if not directory.exists():
                continue
            candidates.extend(directory.glob("xgboost_base_model_*.pkl"))

        if not candidates:
            return None

        fallback_path = max(candidates, key=lambda path: path.stat().st_mtime)
        log.warning(
            f"[PredictWorker] Configured model missing at {configured_path}. "
            f"Falling back to latest available model: {fallback_path}"
        )
        return fallback_path

    def _load_model(self) -> bool:
        model_path = self._resolve_model_path()
        if model_path is None:
            log.error(
                f"[PredictWorker] Model not found at {cfg.MODEL_PATH}. "
                "Run the notebook to train and save the model."
            )
            return False

        try:
            self.model = joblib.load(str(model_path))
            log.info(f"[PredictWorker] Model loaded from {model_path}")
            return True
        except Exception as exc:
            log.error(f"[PredictWorker] Failed to load model: {exc}")
            return False

    def _predict(self, feat_row) -> tuple[float, float, float]:
        proba = self.model.predict_proba(feat_row)
        return float(proba[0, 0]), float(proba[0, 1]), float(proba[0, 2])

    @staticmethod
    def _signal_mode() -> str:
        mode = str(getattr(cfg, "SIGNAL_MODE", "threshold")).strip().lower()
        return mode if mode in {"threshold", "lead_class"} else "threshold"

    def _is_within_session(self, dt) -> bool:
        if cfg.SKIP_WEEKENDS and dt.weekday() >= 5:
            return False
        for start_h, end_h in cfg.TRADING_SESSIONS:
            if start_h <= dt.hour < end_h:
                return True
        return False

    def _select_direction(self, p_bear: float, p_neut: float, p_bull: float):
        classes = ["BEAR", "NEUT", "BULL"]
        probs = [p_bear, p_neut, p_bull]
        lead_idx = int(np.argmax(probs))
        lead_cls = classes[lead_idx]
        conf = probs[lead_idx]
        mode = self._signal_mode()

        if lead_cls == "NEUT":
            return 0, lead_cls, conf, "lead class is NEUT"

        if mode == "lead_class":
            direction = 1 if lead_cls == "BULL" else -1
            return direction, lead_cls, conf, ""

        if lead_cls == "BULL" and p_bull >= cfg.PROB_THRESHOLD:
            return 1, lead_cls, conf, ""
        if lead_cls == "BEAR" and p_bear >= cfg.PROB_THRESHOLD:
            return -1, lead_cls, conf, ""

        return 0, lead_cls, conf, f"lead confidence below threshold {cfg.PROB_THRESHOLD:.2f}"

    def run(self):
        log.info("[PredictWorker] Starting.")

        if not self._load_model():
            log.warning("[PredictWorker] Running in DEMO mode - random predictions.")

        while not stop_event.is_set():
            if trading_halted.is_set():
                stop_event.wait(timeout=10)
                continue

            try:
                payload = feature_q.get(timeout=cfg.QUEUE_TIMEOUT)
            except Exception:
                continue

            try:
                feat_row = payload["features"]
                meta = payload["meta"]
                dt = meta["datetime"].iloc[0]
                close = float(meta["close"].iloc[0])
                atr = float(meta["atr"].iloc[0])
                rsi = float(meta["rsi"].iloc[0])

                if not self._is_within_session(dt):
                    log.info(f"[PredictWorker] SKIP | outside session at hour={dt.hour}")
                    continue

                if self.model is not None:
                    p_bear, p_neut, p_bull = self._predict(feat_row)
                else:
                    rng = np.random.default_rng(int(time.time()))
                    p_bear, p_neut, p_bull = rng.dirichlet([1, 2, 1])

                direction, lead_cls, conf, reason = self._select_direction(p_bear, p_neut, p_bull)

                log.info(
                    f"[PredictWorker] LIVE | bar={dt} | "
                    f"Bear:{p_bear:.1%} Neut:{p_neut:.1%} Bull:{p_bull:.1%} | "
                    f"Lead={lead_cls} ({conf:.1%})"
                )

                if direction == 0:
                    log.info(f"[PredictWorker] NO SIGNAL | bar={dt} | reason={reason}")
                    continue

                last_sig = get_state("last_signal_ts")
                elapsed = time.time() - last_sig
                if elapsed < cfg.SIGNAL_COOLDOWN:
                    remaining = max(0.0, cfg.SIGNAL_COOLDOWN - elapsed)
                    log.info(f"[PredictWorker] SIGNAL SKIPPED | cooldown active for {remaining:.1f}s")
                    continue

                set_state("last_signal_ts", time.time())

                signal = {
                    "direction": direction,
                    "p_bull": p_bull,
                    "p_bear": p_bear,
                    "p_neut": p_neut,
                    "lead_class": lead_cls,
                    "lead_confidence": conf,
                    "atr": atr,
                    "close": close,
                    "rsi": rsi,
                    "datetime": dt,
                }

                direction_str = "BULL" if direction == 1 else "BEAR"
                confidence = p_bull if direction == 1 else p_bear
                log.info(
                    f"[PredictWorker] SIGNAL | direction={direction_str} | "
                    f"conf={confidence:.3f} | price={close:.5f} | ATR={atr:.5f} | "
                    f"mode={self._signal_mode()}"
                )

                try:
                    signal_q.put_nowait(signal)
                except Exception:
                    log.warning("[PredictWorker] signal_q full - dropping signal.")
            except Exception as exc:
                log.exception(f"[PredictWorker] Prediction error: {exc}")

        log.info("[PredictWorker] Stopped.")
