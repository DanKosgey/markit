"""
utils/state.py — Shared inter-worker communication objects.

All workers import from here so they share the same queue/event instances.
"""

import queue
import threading
import logging
import sys
from datetime import datetime
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(name: str = "BOT") -> logging.Logger:
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"bot_{datetime.utcnow().strftime('%Y%m%d')}.log"

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s [%(threadName)-20s] %(message)s",
        datefmt="%H:%M:%S"
    )

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # File handler (DEBUG level)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


log = setup_logger()


# ─────────────────────────────────────────────────────────────────────────────
#  INTER-WORKER QUEUES
#  DataWorker → raw_bar_q → FeatureWorker → feature_q → PredictWorker
#  PredictWorker → signal_q → ExecutorWorker
# ─────────────────────────────────────────────────────────────────────────────

raw_bar_q   = queue.Queue(maxsize=5)   # Raw OHLCV DataFrames from MT5
feature_q   = queue.Queue(maxsize=5)   # Engineered feature rows (1 row per bar)
signal_q    = queue.Queue(maxsize=10)  # Trade signals from predictor


# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL FLAGS
# ─────────────────────────────────────────────────────────────────────────────

stop_event        = threading.Event()   # Set to True → all workers shut down gracefully
trading_halted    = threading.Event()   # Set when daily-loss / risk limit hit

# ─────────────────────────────────────────────────────────────────────────────
#  SHARED RUNTIME STATE  (protected by a lock)
# ─────────────────────────────────────────────────────────────────────────────

_state_lock = threading.Lock()

_state = {
    "daily_pnl_usd":      0.0,
    "open_positions":     0,
    "last_signal_ts":     0.0,    # epoch seconds of last signal fired
    "last_bar_ts":        None,   # datetime of last processed bar
    "account_balance":    0.0,
    "peak_balance":       0.0,
    "total_trades":       0,
    "total_wins":         0,
    "total_losses":       0,
}


def get_state(key: str):
    with _state_lock:
        return _state[key]


def set_state(key: str, value):
    with _state_lock:
        _state[key] = value


def update_state(**kwargs):
    with _state_lock:
        _state.update(kwargs)


def get_full_state() -> dict:
    with _state_lock:
        return dict(_state)
