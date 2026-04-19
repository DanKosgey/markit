"""
workers/monitor_worker.py — Position Monitor & Dashboard Worker.

Periodically queries MT5 for open positions, checks for fills,
updates shared state, and prints a live performance dashboard to stdout.
"""

import time
import threading
from datetime import datetime

import config as cfg
from utils.state import log, stop_event, get_full_state, trading_halted

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

_SEPARATOR = "─" * 65


def _format_dashboard(state: dict) -> str:
    now   = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    halt  = "⛔ HALTED" if trading_halted.is_set() else "✅ ACTIVE"
    last  = str(state.get("last_bar_ts", "—"))[:19]
    total = state["total_trades"]
    wins  = state["total_wins"]
    losses= state["total_losses"]
    wr    = f"{wins/(total or 1)*100:.1f}%" if total else "—"

    return (
        f"\n{_SEPARATOR}\n"
        f"  MARKET STRUCTURE XGBOOST BOT  [{now}]\n"
        f"{_SEPARATOR}\n"
        f"  Status         : {halt}\n"
        f"  Symbol         : {cfg.MT5_SYMBOL}  ({cfg.MT5_TIMEFRAME})\n"
        f"  Last Bar       : {last}\n"
        f"  Open Positions : {state['open_positions']}\n"
        f"  Account Bal    : ${state['account_balance']:.2f}\n"
        f"  Daily P&L      : ${state['daily_pnl_usd']:+.2f}  "
        f"(limit -${cfg.MAX_DAILY_LOSS_USD:.0f})\n"
        f"  Total Trades   : {total}  Wins: {wins}  Losses: {losses}  WR: {wr}\n"
        f"  Prob Threshold : {cfg.PROB_THRESHOLD}  "
        f"SL×ATR: {cfg.SL_MULT}  TP×ATR: {cfg.TP_MULT}\n"
        f"{_SEPARATOR}"
    )


class MonitorWorker(threading.Thread):
    """
    Periodically:
      • Syncs open position count from MT5
      • Checks for newly closed positions and books P&L
      • Prints a dashboard summary to the console
    """

    def __init__(self):
        super().__init__(name="MonitorWorker", daemon=True)
        self._known_tickets: set = set()   # tickets we've seen open

    # ── internal ──────────────────────────────────────────────────────────────

    def _modify_sl(self, ticket: int, symbol: str, new_sl: float, tp: float):
        sym_info = mt5.symbol_info(symbol)
        if sym_info and sym_info.trade_tick_size > 0:
            ts = sym_info.trade_tick_size
            new_sl = round(round(new_sl / ts) * ts, 10)

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": symbol,
            "sl": new_sl,
            "tp": tp,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            log.info(f"[MonitorWorker] Trailing Stop locked profit for #{ticket}: SL = {new_sl:.5f}")
        else:
            err = result.comment if result else mt5.last_error()
            retcode = result.retcode if result else -1
            log.warning(f"[MonitorWorker] Trailing SL failed #{ticket} (Retcode: {retcode}, SL={new_sl:.5f}): {err}")

    def _sync_mt5_positions(self):
        if not MT5_AVAILABLE:
            return

        positions = mt5.positions_get(symbol=cfg.MT5_SYMBOL)
        if positions is None:
            return

        bot_positions = {p.ticket: p for p in positions if p.magic == cfg.MAGIC}
        current_tickets = set(bot_positions.keys())

        # ── Trailing Stop Logic ───────────────────────────────────────────────
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
                        # Only move SL up
                        if pos.sl == 0.0 or new_sl > pos.sl:
                            if new_sl < current_price:
                                self._modify_sl(ticket, pos.symbol, new_sl, pos.tp)
                
                elif pos.type == mt5.POSITION_TYPE_SELL:
                    current_price = tick.ask
                    profit_points = pos.price_open - current_price
                    if profit_points >= getattr(cfg, "TRAIL_ACTIVATION", 2.0):
                        new_sl = current_price + safe_dist
                        # Only move SL down
                        if pos.sl == 0.0 or new_sl < pos.sl:
                            if new_sl > current_price:
                                self._modify_sl(ticket, pos.symbol, new_sl, pos.tp)

        # Detect newly closed positions
        closed = self._known_tickets - current_tickets
        for ticket in closed:
            # Look in history for the deal
            from datetime import timedelta
            t_from = datetime.utcnow() - timedelta(hours=24)
            t_to   = datetime.utcnow()
            history = mt5.history_deals_get(t_from, t_to)
            if history:
                for deal in history:
                    if deal.position_id == ticket and deal.magic == cfg.MAGIC:
                        pnl_usd = deal.profit
                        is_win  = pnl_usd > 0
                        from workers.executor_worker import _update_trade_stats
                        _update_trade_stats(pnl_usd, is_win)
                        break

        self._known_tickets = current_tickets

        from utils.state import set_state
        set_state("open_positions", len(bot_positions))

        # Refresh account balance
        info = mt5.account_info()
        if info:
            from utils.state import update_state, get_state
            bal  = info.balance
            peak = get_state("peak_balance")
            update_state(
                account_balance=bal,
                peak_balance=max(bal, peak),
            )

    # ── public ────────────────────────────────────────────────────────────────

    def run(self):
        log.info("[MonitorWorker] Starting…")
        loop_count = 0

        while not stop_event.is_set():
            try:
                self._sync_mt5_positions()
                
                if loop_count % (cfg.MONITOR_POLL_SECONDS * 6) == 0:
                    dashboard = _format_dashboard(get_full_state())
                    print(dashboard, flush=True)
                
                loop_count += 1
            except Exception as exc:
                log.exception(f"[MonitorWorker] Error: {exc}")

            stop_event.wait(timeout=1.0)  # fast poll for trailing SL

        log.info("[MonitorWorker] Stopped.")
