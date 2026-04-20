"""
workers/data_worker.py - Fetches live OHLCV bars from MetaTrader 5.

Runs in its own thread, polling for new bars. When a new completed bar is
detected, it pushes the historical window needed for feature warm-up onto
raw_bar_q.
"""

import re
import threading
import time

import numpy as np
import pandas as pd

import config as cfg
from utils.state import get_state, log, raw_bar_q, set_state, stop_event

try:
    import MetaTrader5 as mt5

    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    log.warning("MetaTrader5 package not found - DataWorker will run in DEMO mode.")


def _mt5_tf(tf_str: str) -> int:
    if not MT5_AVAILABLE:
        return 1

    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    return tf_map.get(tf_str, mt5.TIMEFRAME_M1)


def _normalise_symbol_name(symbol: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", symbol.lower())


def _resolve_mt5_symbol(requested_symbol: str) -> str | None:
    if not MT5_AVAILABLE:
        return None

    if mt5.symbol_info(requested_symbol) is not None:
        return requested_symbol

    symbols = mt5.symbols_get()
    if not symbols:
        return None

    names = [symbol.name for symbol in symbols]
    by_lower = {name.lower(): name for name in names}
    by_norm = {_normalise_symbol_name(name): name for name in names}

    if requested_symbol.lower() in by_lower:
        return by_lower[requested_symbol.lower()]

    normalized_requested = _normalise_symbol_name(requested_symbol)
    if normalized_requested in by_norm:
        return by_norm[normalized_requested]

    suffix_trimmed = re.sub(r"([._-]\d+)+$", "", requested_symbol).strip()
    if suffix_trimmed.lower() in by_lower:
        return by_lower[suffix_trimmed.lower()]

    normalized_trimmed = _normalise_symbol_name(suffix_trimmed)
    if normalized_trimmed in by_norm:
        return by_norm[normalized_trimmed]

    return None


def connect_mt5() -> bool:
    if not MT5_AVAILABLE:
        log.warning("[DataWorker] MT5 not available - using DEMO data.")
        return False

    if not mt5.initialize():
        log.error(f"[DataWorker] mt5.initialize() failed: {mt5.last_error()}")
        return False

    info = mt5.account_info()
    if info is None:
        log.error(f"[DataWorker] Not logged in to MT5: {mt5.last_error()}")
        return False

    resolved_symbol = _resolve_mt5_symbol(cfg.MT5_SYMBOL)
    if resolved_symbol is None:
        log.error(
            f"[DataWorker] Symbol {cfg.MT5_SYMBOL} was not found in MT5. "
            "Check the broker symbol name in config.py."
        )
        return False

    if resolved_symbol != cfg.MT5_SYMBOL:
        log.warning(f"[DataWorker] Symbol alias resolved: {cfg.MT5_SYMBOL} -> {resolved_symbol}")
        cfg.MT5_SYMBOL = resolved_symbol

    if not mt5.symbol_select(cfg.MT5_SYMBOL, True):
        log.error(f"[DataWorker] Failed to select symbol {cfg.MT5_SYMBOL}: {mt5.last_error()}")
        return False

    log.info(
        f"[DataWorker] MT5 connected | Account: {info.login} | "
        f"Server: {info.server} | Balance: {info.balance:.2f}"
    )
    set_state("account_balance", info.balance)
    set_state("peak_balance", info.balance)
    return True


def fetch_bars_mt5(symbol: str, tf_str: str, n_bars: int) -> pd.DataFrame:
    tf = _mt5_tf(tf_str)

    if not mt5.symbol_select(symbol, True):
        log.warning(f"[DataWorker] Failed to select symbol {symbol}: {mt5.last_error()}")
        return pd.DataFrame()

    raw = mt5.copy_rates_from_pos(symbol, tf, 0, n_bars)
    if raw is None or len(raw) == 0:
        log.warning(f"[DataWorker] No data for {symbol} {tf_str}: {mt5.last_error()}")
        return pd.DataFrame()

    df = pd.DataFrame(raw)
    df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(None)
    df = df.rename(
        columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "tick_volume": "volume",
        }
    )[["datetime", "open", "high", "low", "close", "volume"]]
    return df.dropna().reset_index(drop=True)


def _demo_bar(last_close: float, bar_idx: int) -> pd.Series:
    rng = np.random.default_rng(bar_idx)
    open_ = last_close
    chg = rng.normal(0, 0.5)
    close = round(open_ + chg, 2)
    high = round(max(open_, close) + abs(rng.normal(0, 0.3)), 2)
    low = round(min(open_, close) - abs(rng.normal(0, 0.3)), 2)
    vol = int(rng.integers(100, 2000))
    return pd.Series(
        {
            "datetime": pd.Timestamp.utcnow().replace(tzinfo=None),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
        }
    )


class DataWorker(threading.Thread):
    def __init__(self):
        super().__init__(name="DataWorker", daemon=True)
        self._connected = False
        self._last_bar_ts = None
        self._demo_df = None
        self._demo_bar_idx = 0
        self._stale_count = 0

    def _init_demo_history(self):
        n_bars = cfg.MT5_BARS
        base = 2000.0
        rows = []
        ts = pd.Timestamp("2025-01-01 08:00:00")

        for bar_idx in range(n_bars):
            rng = np.random.default_rng(bar_idx)
            open_ = base
            chg = rng.normal(0, 0.5)
            close = round(open_ + chg, 2)
            high = round(max(open_, close) + abs(rng.normal(0, 0.3)), 2)
            low = round(min(open_, close) - abs(rng.normal(0, 0.3)), 2)
            vol = int(rng.integers(100, 2000))
            rows.append(
                {
                    "datetime": ts,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": vol,
                }
            )
            ts += pd.Timedelta(minutes=1)
            base = close

        self._demo_df = pd.DataFrame(rows)
        self._demo_df["datetime"] = pd.to_datetime(self._demo_df["datetime"])
        self._demo_bar_idx = n_bars
        log.info(f"[DataWorker] Demo history initialised: {n_bars} synthetic bars")

    def _fetch_primary(self) -> pd.DataFrame:
        if self._connected:
            return fetch_bars_mt5(cfg.MT5_SYMBOL, cfg.MT5_TIMEFRAME, cfg.MT5_BARS)
        return self._demo_df.copy() if self._demo_df is not None else pd.DataFrame()

    def _fetch_htf(self) -> pd.DataFrame:
        if self._connected:
            return fetch_bars_mt5(cfg.MT5_SYMBOL, cfg.MT5_HTF, cfg.MT5_HTF_BARS)

        if self._demo_df is None:
            return pd.DataFrame()

        htf = self._demo_df.copy()
        htf["datetime"] = htf["datetime"].dt.floor("15min")
        return (
            htf.groupby("datetime")
            .agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
            )
            .reset_index()
        )

    def _push_new_bar(self):
        if not self._connected:
            last_close = self._demo_df["close"].iloc[-1]
            new_bar = _demo_bar(last_close, self._demo_bar_idx)
            self._demo_df = pd.concat([self._demo_df, new_bar.to_frame().T], ignore_index=True)
            self._demo_bar_idx += 1
            if len(self._demo_df) > cfg.MT5_BARS + 200:
                self._demo_df = self._demo_df.iloc[-cfg.MT5_BARS :].reset_index(drop=True)

        df_1m = self._fetch_primary()
        df_htf = self._fetch_htf()

        if df_1m.empty:
            log.warning("[DataWorker] Empty primary frame - skipping push.")
            return

        current_ts = df_1m["datetime"].iloc[-1]
        if current_ts == self._last_bar_ts:
            self._stale_count += 1
        else:
            self._stale_count = 0

        tf_seconds = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600, "H4": 14400, "D1": 86400}
        max_stale_secs = tf_seconds.get(cfg.MT5_TIMEFRAME, 60)
        max_stale_loops = max_stale_secs / max(getattr(cfg, "DATA_POLL_SECONDS", 1.0), 1.0)
        if self._stale_count > max_stale_loops + 5:
            stale_secs = int(self._stale_count * getattr(cfg, "DATA_POLL_SECONDS", 1.0))
            log.warning(
                f"[DataWorker] STALE DATA | last bar ts {current_ts} unchanged for {stale_secs}s+ | "
                f"Check MT5 connection or market hours (TF={cfg.MT5_TIMEFRAME})."
            )

        if self._last_bar_ts is not None and current_ts == self._last_bar_ts:
            return

        self._last_bar_ts = current_ts

        df_1m_closed = df_1m.iloc[:-1].copy() if len(df_1m) > 1 else df_1m
        df_htf_closed = df_htf.iloc[:-1].copy() if len(df_htf) > 1 else df_htf
        if df_1m_closed.empty:
            log.warning("[DataWorker] Closed-bar frame is empty - skipping push.")
            return

        closed_ts = df_1m_closed["datetime"].iloc[-1]
        set_state("last_bar_ts", closed_ts)

        try:
            raw_bar_q.put_nowait({"df_1m": df_1m_closed, "df_htf": df_htf_closed})
            log.info(
                f"[DataWorker] Pushed closed bar data | closed_ts={closed_ts} | "
                f"queue_size={raw_bar_q.qsize()}"
            )
        except Exception:
            log.warning("[DataWorker] raw_bar_q full - dropping bar.")

    def run(self):
        log.info("[DataWorker] Starting.")
        self._connected = connect_mt5()
        set_state("mt5_live", self._connected)
        set_state("runtime_symbol", cfg.MT5_SYMBOL)

        if not self._connected:
            self._init_demo_history()

        while not stop_event.is_set():
            try:
                self._push_new_bar()
            except Exception as exc:
                log.exception(f"[DataWorker] Unexpected error: {exc}")

            if self._connected:
                try:
                    info = mt5.account_info()
                    if info:
                        set_state("account_balance", info.balance)
                        if info.balance > get_state("peak_balance"):
                            set_state("peak_balance", info.balance)
                except Exception:
                    pass

            stop_event.wait(timeout=cfg.DATA_POLL_SECONDS)

        log.info("[DataWorker] Stopped.")
        if self._connected and MT5_AVAILABLE:
            mt5.shutdown()
