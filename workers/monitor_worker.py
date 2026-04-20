"""
workers/monitor_worker.py - Position Monitor and Dashboard Worker.
"""

import threading
import time
from datetime import datetime, timedelta

import config as cfg
from utils.state import get_full_state, get_state, log, stop_event, trading_halted

try:
    import MetaTrader5 as mt5

    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

_SEPARATOR = "-" * 65


def _format_vision_signals(signals: dict | None) -> str:
    if not signals:
        return "none"
    parts = []
    for timeframe in sorted(signals):
        entry = signals.get(timeframe) or {}
        signal = entry.get("signal") or "unknown"
        source = entry.get("source") or "n/a"
        parts.append(f"{timeframe}={signal}@{source}")
    return ", ".join(parts)


def _format_dashboard(state: dict) -> str:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    status = "HALTED" if trading_halted.is_set() else "ACTIVE"
    last_bar = str(state.get("last_bar_ts", "n/a"))[:19]
    symbol = state.get("runtime_symbol") or cfg.MT5_SYMBOL
    total = state["total_trades"]
    wins = state["total_wins"]
    losses = state["total_losses"]
    win_rate = f"{wins / total * 100:.1f}%" if total else "n/a"
    vision_state = state.get("vision_connection_state") or "unknown"
    vision_action = state.get("vision_decision_action") or "none"
    vision_age_seconds = None
    if state.get("vision_last_message_ts"):
        vision_age_seconds = max(0.0, time.time() - float(state["vision_last_message_ts"]))
    vision_age_text = "n/a" if vision_age_seconds is None else f"{vision_age_seconds:.1f}s"

    return (
        f"\n{_SEPARATOR}\n"
        f"  MARKET STRUCTURE XGBOOST BOT  [{now}]\n"
        f"{_SEPARATOR}\n"
        f"  Status         : {status}\n"
        f"  Symbol         : {symbol}  ({cfg.MT5_TIMEFRAME})\n"
        f"  Last Closed Bar: {last_bar}\n"
        f"  Open Positions : {state['open_positions']}\n"
        f"  Account Bal    : ${state['account_balance']:.2f}\n"
        f"  Daily P&L      : ${state['daily_pnl_usd']:+.2f}  (limit -${cfg.MAX_DAILY_LOSS_USD:.0f})\n"
        f"  Total Trades   : {total}  Wins: {wins}  Losses: {losses}  WR: {win_rate}\n"
        f"  Signal Mode    : {getattr(cfg, 'SIGNAL_MODE', 'threshold')}  "
        f"Threshold: {cfg.PROB_THRESHOLD}\n"
        f"  Vision State   : {vision_state}  Action: {vision_action}  Age: {vision_age_text}\n"
        f"  Vision Current : {_format_vision_signals(state.get('vision_signals'))}\n"
        f"  Vision Last OK : {_format_vision_signals(state.get('vision_last_nonempty_signals'))}\n"
        f"  SLxATR         : {cfg.SL_MULT}  TPxATR: {cfg.TP_MULT}\n"
        f"{_SEPARATOR}"
    )


class MonitorWorker(threading.Thread):
    def __init__(self):
        super().__init__(name="MonitorWorker", daemon=True)
        self._known_tickets: set = set()

    def _modify_sl(self, ticket: int, symbol: str, new_sl: float, tp: float):
        sym_info = mt5.symbol_info(symbol)
        if sym_info and sym_info.trade_tick_size > 0:
            tick_size = sym_info.trade_tick_size
            new_sl = round(round(new_sl / tick_size) * tick_size, 10)

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": symbol,
            "sl": new_sl,
            "tp": tp,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            log.info(f"[MonitorWorker] Trailing stop updated for #{ticket}: SL={new_sl:.5f}")
            return

        error = result.comment if result else mt5.last_error()
        retcode = result.retcode if result else -1
        log.warning(f"[MonitorWorker] Trailing stop failed for #{ticket} | retcode={retcode} | error={error}")

    def _sync_mt5_positions(self):
        if not (MT5_AVAILABLE and get_state("mt5_live")):
            return

        mt5.symbol_select(cfg.MT5_SYMBOL, True)
        positions = mt5.positions_get(symbol=cfg.MT5_SYMBOL)
        if positions is None:
            return

        bot_positions = {pos.ticket: pos for pos in positions if pos.magic == cfg.MAGIC}
        current_tickets = set(bot_positions.keys())

        if getattr(cfg, "USE_TRAILING_STOP", False):
            for ticket, pos in bot_positions.items():
                tick = mt5.symbol_info_tick(pos.symbol)
                sym_info = mt5.symbol_info(pos.symbol)
                if tick is None or sym_info is None:
                    continue

                min_dist = sym_info.trade_stops_level * sym_info.point
                trail_dist = getattr(cfg, "TRAIL_DISTANCE", 1.0)
                safe_dist = max(trail_dist, min_dist)

                if pos.type == mt5.POSITION_TYPE_BUY:
                    current_price = tick.bid
                    profit_points = current_price - pos.price_open
                    if profit_points >= getattr(cfg, "TRAIL_ACTIVATION", 2.0):
                        new_sl = current_price - safe_dist
                        if pos.sl == 0.0 or new_sl > pos.sl:
                            if new_sl < current_price:
                                self._modify_sl(ticket, pos.symbol, new_sl, pos.tp)
                elif pos.type == mt5.POSITION_TYPE_SELL:
                    current_price = tick.ask
                    profit_points = pos.price_open - current_price
                    if profit_points >= getattr(cfg, "TRAIL_ACTIVATION", 2.0):
                        new_sl = current_price + safe_dist
                        if pos.sl == 0.0 or new_sl < pos.sl:
                            if new_sl > current_price:
                                self._modify_sl(ticket, pos.symbol, new_sl, pos.tp)

        closed_tickets = self._known_tickets - current_tickets
        for ticket in closed_tickets:
            t_from = datetime.utcnow() - timedelta(hours=24)
            t_to = datetime.utcnow()
            history = mt5.history_deals_get(t_from, t_to)
            if not history:
                continue

            for deal in history:
                if deal.position_id == ticket and deal.magic == cfg.MAGIC:
                    pnl_usd = deal.profit
                    is_win = pnl_usd > 0
                    from workers.executor_worker import _update_trade_stats

                    _update_trade_stats(pnl_usd, is_win)
                    break

        self._known_tickets = current_tickets

        from utils.state import set_state, update_state

        set_state("open_positions", len(bot_positions))
        info = mt5.account_info()
        if info:
            peak = get_state("peak_balance")
            update_state(account_balance=info.balance, peak_balance=max(info.balance, peak))

    def run(self):
        log.info("[MonitorWorker] Starting.")
        loop_count = 0

        while not stop_event.is_set():
            try:
                self._sync_mt5_positions()
                if loop_count % (cfg.MONITOR_POLL_SECONDS * 6) == 0:
                    print(_format_dashboard(get_full_state()), flush=True)
                loop_count += 1
            except Exception as exc:
                log.exception(f"[MonitorWorker] Error: {exc}")

            stop_event.wait(timeout=1.0)

        log.info("[MonitorWorker] Stopped.")
