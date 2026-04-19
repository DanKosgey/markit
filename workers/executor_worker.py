"""
workers/executor_worker.py — Trade Execution Worker.

Consumes trade signals from signal_q, enforces risk rules, then
places market orders on MT5 with SL = entry ± (SL_MULT × ATR)
and TP = entry ± (TP_MULT × ATR).

All stop/target levels are calculated from the live ATR value
embedded in the signal, matching the notebook's backtest engine.
"""

import time
import threading
from datetime import datetime, date

import numpy as np

import config as cfg
from utils.state import (
    log, stop_event, signal_q, trading_halted,
    get_state, set_state, update_state,
)

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
#  MT5 HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_symbol_info():
    if not MT5_AVAILABLE:
        return None
    info = mt5.symbol_info(cfg.MT5_SYMBOL)
    if info is None:
        log.error(f"[Executor] Symbol {cfg.MT5_SYMBOL} not found: {mt5.last_error()}")
    return info


def _round_price(price: float, tick_size: float) -> float:
    if tick_size <= 0:
        return price
    return round(round(price / tick_size) * tick_size, 10)


def _place_order(direction: int, atr: float) -> dict | None:
    """
    Places a market order with SL and TP calculated from ATR multipliers.

    SL = entry ∓ (SL_MULT × ATR)
    TP = entry ± (TP_MULT × ATR)

    Returns the order result dict or None on failure.
    """
    if not MT5_AVAILABLE:
        # ── DEMO mode: simulate fill ───────────────────────────────────────────
        tick = _get_demo_tick()
        entry  = tick["ask"] if direction == 1 else tick["bid"]
        sl     = entry - direction * cfg.SL_MULT * atr
        tp     = entry + direction * cfg.TP_MULT * atr
        log.info(
            f"[Executor] [DEMO] {'BUY' if direction==1 else 'SELL'}  "
            f"entry={entry:.5f}  SL={sl:.5f}  TP={tp:.5f}  "
            f"(SL_MULT={cfg.SL_MULT}×ATR  TP_MULT={cfg.TP_MULT}×ATR)"
        )
        return {
            "demo":    True,
            "direction": direction,
            "entry":   entry,
            "sl":      sl,
            "tp":      tp,
            "ticket":  int(time.time()),
        }

    # ── Live MT5 order ─────────────────────────────────────────────────────────
    sym_info = _get_symbol_info()
    if sym_info is None:
        return None

    tick = mt5.symbol_info_tick(cfg.MT5_SYMBOL)
    if tick is None:
        log.error(f"[Executor] Cannot get tick for {cfg.MT5_SYMBOL}: {mt5.last_error()}")
        return None

    ts = sym_info.trade_tick_size
    entry  = tick.ask if direction == 1 else tick.bid
    sl     = _round_price(entry - direction * cfg.SL_MULT * atr, ts)
    tp     = _round_price(entry + direction * cfg.TP_MULT * atr, ts)

    order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL

    # Determine correct filling mode supported by the broker for this symbol.
    # The MetaTrader5 Python module often lacks the SYMBOL_FILLING constants.
    # We use their MQL5 integer bitmask equivalents: 1 (FOK) and 2 (IOC).
    filling_mode = mt5.ORDER_FILLING_FOK
    if sym_info.filling_mode & 1:
        filling_mode = mt5.ORDER_FILLING_FOK
    elif sym_info.filling_mode & 2:
        filling_mode = mt5.ORDER_FILLING_IOC
    else:
        filling_mode = mt5.ORDER_FILLING_RETURN

    request = {
        "action":     mt5.TRADE_ACTION_DEAL,
        "symbol":     cfg.MT5_SYMBOL,
        "volume":     cfg.LOT_SIZE,
        "type":       order_type,
        "price":      entry,
        "sl":         sl,
        "tp":         tp,
        "deviation":  cfg.SLIPPAGE_POINTS,
        "magic":      cfg.MAGIC,
        "comment":    cfg.ORDER_COMMENT,
        "type_time":  mt5.ORDER_TIME_GTC,
        "type_filling": filling_mode,
    }

    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        err = result.comment if result else mt5.last_error()
        log.error(f"[Executor] Order FAILED: retcode={getattr(result,'retcode','?')}  {err}")
        return None

    log.info(
        f"[Executor] ✓ Order placed  ticket={result.order}  "
        f"{'BUY' if direction==1 else 'SELL'}  "
        f"entry={entry:.5f}  SL={sl:.5f}  TP={tp:.5f}  "
        f"(SL_MULT={cfg.SL_MULT}×ATR={atr:.5f}  TP_MULT={cfg.TP_MULT}×ATR)"
    )

    return {
        "demo":      False,
        "direction": direction,
        "entry":     entry,
        "sl":        sl,
        "tp":        tp,
        "ticket":    result.order,
    }


def _get_open_positions() -> list:
    """Return list of open positions belonging to our magic number."""
    if not MT5_AVAILABLE:
        return []
    positions = mt5.positions_get(symbol=cfg.MT5_SYMBOL)
    if positions is None:
        return []
    return [p for p in positions if p.magic == cfg.MAGIC]


def _get_demo_tick() -> dict:
    return {"bid": 2000.0 + np.random.uniform(-0.5, 0.5),
            "ask": 2000.1 + np.random.uniform(-0.5, 0.5)}


# ─────────────────────────────────────────────────────────────────────────────
#  EXECUTOR WORKER
# ─────────────────────────────────────────────────────────────────────────────

class ExecutorWorker(threading.Thread):
    """
    Consumes signals → validates risk rules → places MT5 orders.
    """

    def __init__(self):
        super().__init__(name="ExecutorWorker", daemon=True)
        self._daily_reset_date = date.today()
        self._demo_positions   = []   # list of demo trade dicts

    # ── risk checks ───────────────────────────────────────────────────────────

    def _daily_reset_if_needed(self):
        today = date.today()
        if today != self._daily_reset_date:
            self._daily_reset_date = today
            set_state("daily_pnl_usd", 0.0)
            log.info("[Executor] Daily PnL counter reset.")

    def _check_risk(self, signal: dict) -> bool:
        """
        Returns True if we're allowed to trade, False if risk limit is hit.
        Checks:
          1. Daily loss limit
          2. Max drawdown from peak balance
          3. Max concurrent positions
        """
        # 1. Daily loss
        daily_pnl = get_state("daily_pnl_usd")
        if daily_pnl <= -cfg.MAX_DAILY_LOSS_USD:
            if not trading_halted.is_set():
                trading_halted.set()
                log.warning(
                    f"[Executor] ⛔ Daily loss limit hit "
                    f"(${daily_pnl:.2f} ≤ -${cfg.MAX_DAILY_LOSS_USD}). "
                    "Trading halted for today."
                )
            return False

        # 2. Drawdown
        balance = get_state("account_balance")
        peak    = get_state("peak_balance")
        if peak > 0 and (peak - balance) / peak >= cfg.MAX_DRAWDOWN_PCT:
            if not trading_halted.is_set():
                trading_halted.set()
                log.warning(
                    f"[Executor] ⛔ Max drawdown {cfg.MAX_DRAWDOWN_PCT*100:.0f}% hit. "
                    "Trading halted."
                )
            return False

        # 3. Open positions
        if MT5_AVAILABLE:
            n_open = len(_get_open_positions())
        else:
            n_open = len(self._demo_positions)

        set_state("open_positions", n_open)
        if n_open >= cfg.MAX_OPEN_TRADES:
            log.debug(f"[Executor] Max open trades ({cfg.MAX_OPEN_TRADES}) reached — skip.")
            return False

        return True

    # ── demo position tracking ────────────────────────────────────────────────

    def _settle_demo_positions(self, current_price: float):
        """Check if any demo positions have hit SL or TP."""
        remaining = []
        for pos in self._demo_positions:
            d      = pos["direction"]
            entry  = pos["entry"]
            tp     = pos["tp"]
            sl     = pos["sl"]

            hit_tp = (d == 1 and current_price >= tp) or (d == -1 and current_price <= tp)
            hit_sl = (d == 1 and current_price <= sl) or (d == -1 and current_price >= sl)

            if hit_tp or hit_sl:
                pnl_pts  = d * ((tp if hit_tp else sl) - entry)
                atr      = pos.get("atr", 1.0)
                pnl_atr  = pnl_pts / atr
                outcome  = "TP ✅" if hit_tp else "SL ❌"
                log.info(
                    f"[Executor] [DEMO] Position closed  {outcome}  "
                    f"ticket={pos['ticket']}  pnl={pnl_pts:+.5f}  ({pnl_atr:+.2f} ATR)"
                )
                # Approximate USD PnL (1 XAU lot ≈ 100 oz, ~$0.1/point for micro)
                pnl_usd = pnl_pts * cfg.LOT_SIZE * 100
                _update_trade_stats(pnl_usd, outcome == "TP ✅")
            else:
                remaining.append(pos)

        self._demo_positions = remaining

    # ── public ────────────────────────────────────────────────────────────────

    def run(self):
        log.info("[Executor] Starting…")

        while not stop_event.is_set():
            self._daily_reset_if_needed()

            # ── In demo mode, settle any open positions ────────────────────────
            if not MT5_AVAILABLE and self._demo_positions:
                tick = _get_demo_tick()
                mid  = (tick["bid"] + tick["ask"]) / 2
                self._settle_demo_positions(mid)

            # ── Consume signal ────────────────────────────────────────────────
            try:
                signal = signal_q.get(timeout=cfg.QUEUE_TIMEOUT)
            except Exception:
                continue

            if trading_halted.is_set():
                log.debug("[Executor] Trading halted — ignoring signal.")
                continue

            if not self._check_risk(signal):
                continue

            direction = signal["direction"]
            atr       = signal["atr"]
            close     = signal["close"]
            conf      = signal["p_bull"] if direction == 1 else signal["p_bear"]

            log.info(
                f"[Executor] Executing signal  "
                f"{'LONG' if direction==1 else 'SHORT'}  "
                f"conf={conf:.3f}  price≈{close:.5f}  ATR={atr:.5f}  "
                f"SL={cfg.SL_MULT}×ATR={cfg.SL_MULT*atr:.5f}  "
                f"TP={cfg.TP_MULT}×ATR={cfg.TP_MULT*atr:.5f}"
            )

            result = _place_order(direction, atr)

            if result:
                set_state("open_positions", get_state("open_positions") + 1)
                update_state(total_trades=get_state("total_trades") + 1)

                if not MT5_AVAILABLE:
                    result["atr"] = atr
                    self._demo_positions.append(result)

        log.info("[Executor] Stopped.")


def _update_trade_stats(pnl_usd: float, is_win: bool):
    """Update shared trade stats after a position closes."""
    daily  = get_state("daily_pnl_usd") + pnl_usd
    wins   = get_state("total_wins")   + (1 if is_win else 0)
    losses = get_state("total_losses") + (0 if is_win else 1)
    update_state(
        daily_pnl_usd=daily,
        total_wins=wins,
        total_losses=losses,
        open_positions=max(0, get_state("open_positions") - 1),
    )
    log.info(
        f"[Executor] Trade settled  pnl_usd={pnl_usd:+.2f}  "
        f"daily_pnl={daily:+.2f}  W/L={wins}/{losses}"
    )