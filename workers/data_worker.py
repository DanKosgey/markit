"""
workers/data_worker.py — Fetches live OHLCV bars from MetaTrader 5.

Runs in its own thread, polling for new 1-min bars at DATA_POLL_SECONDS
intervals. When a new completed bar is detected, it pushes the full
historical window (needed for feature warm-up) onto raw_bar_q.
"""

import time
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from utils.state import log, stop_event, raw_bar_q, set_state
import config as cfg

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    log.warning("MetaTrader5 package not found — DataWorker will run in DEMO mode.")

# ─── MT5 timeframe map ────────────────────────────────────────────────────────
_TF_MAP = {
    "M1":  1,
    "M5":  5,
    "M15": 15,
    "M30": 30,
    "H1":  16385,
    "H4":  16388,
    "D1":  16408,
}


def _mt5_tf(tf_str: str) -> int:
    if not MT5_AVAILABLE:
        return 1
    tf_map = {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
    }
    return tf_map.get(tf_str, mt5.TIMEFRAME_M1)


def connect_mt5() -> bool:
    """Initialise and check MT5 connection. Returns True on success."""
    if not MT5_AVAILABLE:
        log.warning("[DataWorker] MT5 not available — using DEMO data.")
        return False

    if not mt5.initialize():
        log.error(f"[DataWorker] mt5.initialize() failed: {mt5.last_error()}")
        return False

    info = mt5.account_info()
    if info is None:
        log.error(f"[DataWorker] Not logged in to MT5: {mt5.last_error()}")
        return False

    log.info(
        f"[DataWorker] ✓ MT5 connected | Account: {info.login} | "
        f"Server: {info.server} | Balance: {info.balance:.2f}"
    )
    set_state("account_balance", info.balance)
    set_state("peak_balance", info.balance)
    return True


def fetch_bars_mt5(symbol: str, tf_str: str, n_bars: int) -> pd.DataFrame:
    """Fetch the last n_bars OHLCV from MT5 and return as a clean DataFrame."""
    tf  = _mt5_tf(tf_str)
    
    # Ensure symbol is visible in Market Watch
    if not mt5.symbol_select(symbol, True):
        log.warning(f"[DataWorker] Failed to select symbol {symbol}. Check the name in MT5.")
        return pd.DataFrame()

    raw = mt5.copy_rates_from_pos(symbol, tf, 0, n_bars)
    if raw is None or len(raw) == 0:
        log.warning(f"[DataWorker] No data for {symbol} {tf_str}: {mt5.last_error()}")
        return pd.DataFrame()

    df = pd.DataFrame(raw)
    df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(None)
    df = df.rename(columns={
        "open":       "open",
        "high":       "high",
        "low":        "low",
        "close":      "close",
        "tick_volume": "volume",
    })[["datetime", "open", "high", "low", "close", "volume"]]
    df = df.dropna().reset_index(drop=True)
    return df


def _demo_bar(last_close: float, bar_idx: int) -> pd.Series:
    """Generate a synthetic OHLCV bar for demo/testing purposes."""
    rng   = np.random.default_rng(bar_idx)
    open_ = last_close
    chg   = rng.normal(0, 0.5)
    close = round(open_ + chg, 2)
    high  = round(max(open_, close) + abs(rng.normal(0, 0.3)), 2)
    low   = round(min(open_, close) - abs(rng.normal(0, 0.3)), 2)
    vol   = int(rng.integers(100, 2000))
    return pd.Series({
        "datetime": pd.Timestamp.utcnow().replace(tzinfo=None),
        "open": open_, "high": high, "low": low, "close": close, "volume": vol,
    })


class DataWorker(threading.Thread):
    """
    Continuously polls MT5 for new 1-min bars (primary) and 15-min bars (HTF).
    When a new completed primary bar arrives, pushes both DataFrames onto raw_bar_q.
    """

    def __init__(self):
        super().__init__(name="DataWorker", daemon=True)
        self._connected    = False
        self._last_bar_ts  = None          # datetime of last pushed bar
        self._last_poll_ts = 0             # last time we pushed data (unix)
        self._demo_df      = None          # synthetic history for demo mode
        self._demo_bar_idx = 0

    # ── internal ──────────────────────────────────────────────────────────────

    def _init_demo_history(self):
        """Create a synthetic historical OHLCV DataFrame for demo mode."""
        n    = cfg.MT5_BARS
        base = 2000.0
        rows = []
        ts   = pd.Timestamp("2025-01-01 08:00:00")
        for i in range(n):
            rng   = np.random.default_rng(i)
            open_ = base
            chg   = rng.normal(0, 0.5)
            close = round(open_ + chg, 2)
            high  = round(max(open_, close) + abs(rng.normal(0, 0.3)), 2)
            low   = round(min(open_, close) - abs(rng.normal(0, 0.3)), 2)
            vol   = int(rng.integers(100, 2000))
            rows.append({"datetime": ts, "open": open_, "high": high,
                         "low": low, "close": close, "volume": vol})
            ts   = ts + pd.Timedelta(minutes=1)
            base = close

        self._demo_df      = pd.DataFrame(rows)
        self._demo_df["datetime"] = pd.to_datetime(self._demo_df["datetime"])
        self._demo_bar_idx = n
        log.info(f"[DataWorker] Demo history initialised: {n} synthetic bars")

    def _fetch_primary(self) -> pd.DataFrame:
        if self._connected:
            return fetch_bars_mt5(cfg.MT5_SYMBOL, cfg.MT5_TIMEFRAME, cfg.MT5_BARS)
        else:
            return self._demo_df.copy() if self._demo_df is not None else pd.DataFrame()

    def _fetch_htf(self) -> pd.DataFrame:
        if self._connected:
            return fetch_bars_mt5(cfg.MT5_SYMBOL, cfg.MT5_HTF, cfg.MT5_HTF_BARS)
        else:
            # Return a minimal HTF frame derived from demo primary
            if self._demo_df is None:
                return pd.DataFrame()
            htf = self._demo_df.copy()
            htf["datetime"] = htf["datetime"].dt.floor("15min")
            htf = (
                htf.groupby("datetime")
                .agg(open=("open","first"), high=("high","max"),
                     low=("low","min"), close=("close","last"),
                     volume=("volume","sum"))
                .reset_index()
            )
            return htf

    def _push_new_bar(self):
        """
        In demo mode: append a new synthetic bar to demo_df and push.
        In live mode: re-fetch from MT5 and push if the last bar is new.
        """
        if not self._connected:
            # Demo: append one synthetic bar
            last_close = self._demo_df["close"].iloc[-1]
            new_bar    = _demo_bar(last_close, self._demo_bar_idx)
            self._demo_df = pd.concat(
                [self._demo_df, new_bar.to_frame().T], ignore_index=True
            )
            self._demo_bar_idx += 1
            # Trim to keep size manageable
            if len(self._demo_df) > cfg.MT5_BARS + 200:
                self._demo_df = self._demo_df.iloc[-cfg.MT5_BARS:].reset_index(drop=True)

        df_1m = self._fetch_primary()
        df_htf = self._fetch_htf()

        if df_1m.empty:
            log.warning("[DataWorker] Empty primary frame — skipping push.")
            return

        # -- Stale Data Detection --
        current_ts = df_1m["datetime"].iloc[-1]
        
        if hasattr(self, "_stale_count"):
            if current_ts == self._last_bar_ts:
                self._stale_count += 1
            else:
                self._stale_count = 0
        else:
            self._stale_count = 0

        self._last_poll_ts = time.time()
        set_state("last_bar_ts", current_ts)

        _TF_SECS = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600, "H4": 14400, "D1": 86400}
        max_stale_secs = _TF_SECS.get(cfg.MT5_TIMEFRAME, 60)
        max_stale_loops = max_stale_secs / getattr(cfg, "DATA_POLL_SECONDS", 1.0)

        if self._stale_count > max_stale_loops + 5:
            log.warning(
                f"[DataWorker] STALE DATA | Last bar ts {current_ts} hasn't changed for {int(self._stale_count * getattr(cfg, 'DATA_POLL_SECONDS', 1.0))}s+ | "
                f"Check MT5 connection or market hours (TF={cfg.MT5_TIMEFRAME})."
            )

        # ====== STRICT BAR-CLOSE ENFORCEMENT ======
        if self._last_bar_ts is not None and current_ts == self._last_bar_ts:
            return  # The current bar is still forming. Wait for exact candle-close tick.

        self._last_bar_ts  = current_ts
        self._last_poll_ts = time.time()
        set_state("last_bar_ts", self._last_bar_ts)

        try:
            # We explicitly drop .iloc[-1] because it is still open/forming!
            # The downstream workers will now predict exclusively on the fully closed .iloc[-2] bar.
            df_1m_closed = df_1m.iloc[:-1].copy() if len(df_1m) > 1 else df_1m
            df_htf_closed = df_htf.iloc[:-1].copy() if len(df_htf) > 1 else df_htf
            
            raw_bar_q.put_nowait({"df_1m": df_1m_closed, "df_htf": df_htf_closed})
            log.info(f"[DataWorker] Pushed fully closed bar data — queue_size={raw_bar_q.qsize()}")
        except Exception:
            log.warning("[DataWorker] raw_bar_q full — dropping bar.")

    # ── public ────────────────────────────────────────────────────────────────

    def run(self):
        log.info("[DataWorker] Starting…")
        self._connected = connect_mt5()

        if not self._connected:
            self._init_demo_history()

        while not stop_event.is_set():
            try:
                self._push_new_bar()
            except Exception as exc:
                log.exception(f"[DataWorker] Unexpected error: {exc}")

            # Refresh account state every cycle if live
            if self._connected:
                try:
                    info = mt5.account_info()
                    if info:
                        set_state("account_balance", info.balance)
                        if info.balance > (set_state("peak_balance", info.balance) or 0):
                            set_state("peak_balance", info.balance)
                except Exception:
                    pass

            stop_event.wait(timeout=cfg.DATA_POLL_SECONDS)

        log.info("[DataWorker] Stopped.")
        if self._connected and MT5_AVAILABLE:
            mt5.shutdown()
