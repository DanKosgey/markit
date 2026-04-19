"""
app.py — Market Structure XGBoost Live Trading Bot
====================================================

Run with:
    python app.py

Prerequisites:
  1. MetaTrader5 installed and logged in (or runs in DEMO mode automatically)
  2. Model exported from the notebook:
       model_ft.save_model("model_ft.json")
       import json; json.dump(list(X_train.columns), open("feature_columns.json","w"))
  3. Both files must be in the same directory as app.py
  4. pip install MetaTrader5 xgboost pandas numpy scipy

Architecture (parallel worker threads):
──────────────────────────────────────────────────────────────────────
  ┌──────────────┐     raw_bar_q     ┌───────────────┐
  │  DataWorker  │ ──────────────►  │ FeatureWorker │
  │  (MT5 / Demo)│                  │  (pipeline)   │
  └──────────────┘                  └───────┬───────┘
                                            │ feature_q
                                            ▼
                                    ┌───────────────┐     signal_q    ┌─────────────────┐
                                    │ PredictWorker │ ──────────────► │ ExecutorWorker  │
                                    │  (XGBoost)    │                 │ (MT5 orders)    │
                                    └───────────────┘                 └─────────────────┘
                                                                               │
                                    ┌───────────────┐                          │
                                    │ MonitorWorker │◄─────────────────────────┘
                                    │  (dashboard)  │
                                    └───────────────┘
──────────────────────────────────────────────────────────────────────
"""

import sys
import signal
import time

from utils.state import log, stop_event

from workers.data_worker    import DataWorker
from workers.feature_worker import FeatureWorker
from workers.predict_worker import PredictWorker
from workers.executor_worker import ExecutorWorker
from workers.monitor_worker  import MonitorWorker


# ─────────────────────────────────────────────────────────────────────────────
#  BANNER
# ─────────────────────────────────────────────────────────────────────────────

BANNER = r"""
╔══════════════════════════════════════════════════════════════════╗
║        MARKET STRUCTURE XGBOOST — LIVE TRADING SYSTEM           ║
║        XAUUSD  |  1-Min  |  BOS / CHoCH / OB / Swings           ║
╚══════════════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────────────────────────────────────
#  GRACEFUL SHUTDOWN
# ─────────────────────────────────────────────────────────────────────────────

def _handle_signal(signum, frame):
    log.info(f"Signal {signum} received — initiating graceful shutdown…")
    stop_event.set()


signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(BANNER)
    log.info("Initialising workers…")

    workers = [
        DataWorker(),
        FeatureWorker(),
        PredictWorker(),
        ExecutorWorker(),
        MonitorWorker(),
    ]

    # Start all workers
    for w in workers:
        w.start()
        log.info(f"  ▶  {w.name} started (thread id={w.ident})")

    log.info("All workers running. Press Ctrl+C to stop.\n")

    # ── Keep main thread alive, checking worker health ─────────────────────────
    try:
        while not stop_event.is_set():
            dead = [w.name for w in workers if not w.is_alive()]
            if dead:
                log.error(f"Workers unexpectedly stopped: {dead}. Shutting down.")
                stop_event.set()
                break
            time.sleep(5)

    except KeyboardInterrupt:
        log.info("KeyboardInterrupt — stopping.")
        stop_event.set()

    # ── Wait for all workers to finish ────────────────────────────────────────
    for w in workers:
        w.join(timeout=10)
        if w.is_alive():
            log.warning(f"{w.name} did not stop cleanly.")

    log.info("Bot shutdown complete. Goodbye.")
    sys.exit(0)


if __name__ == "__main__":
    main()
