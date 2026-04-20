"""
app.py - Market Structure XGBoost Live Trading Bot
==================================================

Run with:
    python app.py
"""

import signal
import sys
import time

import config as cfg
from utils.state import log, stop_event
from workers.data_worker import DataWorker
from workers.executor_worker import ExecutorWorker
from workers.feature_worker import FeatureWorker
from workers.monitor_worker import MonitorWorker
from workers.predict_worker import PredictWorker
from workers.vision_worker import VisionSignalWorker


def _handle_signal(signum, frame):
    log.info(f"Signal {signum} received - initiating graceful shutdown.")
    stop_event.set()


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def _render_banner() -> str:
    return (
        "\n"
        "=================================================================\n"
        " MARKET STRUCTURE XGBOOST - LIVE TRADING SYSTEM\n"
        f" Symbol: {cfg.MT5_SYMBOL}\n"
        f" Timeframe: {cfg.MT5_TIMEFRAME}\n"
        f" Vision TFs: {', '.join(getattr(cfg, 'VISION_TIMEFRAMES', []))}\n"
        "=================================================================\n"
    )


def main():
    print(_render_banner())
    log.info("Initialising workers.")

    workers = [
        DataWorker(),
        FeatureWorker(),
        PredictWorker(),
        VisionSignalWorker(),
        ExecutorWorker(),
        MonitorWorker(),
    ]

    for worker in workers:
        worker.start()
        log.info(f"  -> {worker.name} started (thread id={worker.ident})")

    log.info("All workers running. Press Ctrl+C to stop.\n")
    optional_stopped = set()

    try:
        while not stop_event.is_set():
            dead_workers = [worker for worker in workers if not worker.is_alive()]
            dead_required = [worker.name for worker in dead_workers if getattr(worker, "required_for_runtime", True)]
            dead_optional = [
                worker.name
                for worker in dead_workers
                if not getattr(worker, "required_for_runtime", True) and worker.name not in optional_stopped
            ]

            if dead_optional:
                optional_stopped.update(dead_optional)
                log.warning(f"Optional workers stopped: {dead_optional}. Core runtime will continue.")

            if dead_required:
                log.error(f"Workers unexpectedly stopped: {dead_required}. Shutting down.")
                stop_event.set()
                break
            time.sleep(5)
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt - stopping.")
        stop_event.set()

    for worker in workers:
        worker.join(timeout=10)
        if worker.is_alive():
            log.warning(f"{worker.name} did not stop cleanly.")

    log.info("Bot shutdown complete. Goodbye.")
    sys.exit(0)


if __name__ == "__main__":
    main()
