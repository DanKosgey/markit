"""
workers/executor_worker.py - Trade Execution Worker.

Consumes trade signals, enforces risk rules, and places market orders on MT5.
"""

import time
import threading
from datetime import date

import numpy as np

import config as cfg
from utils.state import get_state, log, set_state, signal_q, stop_event, trading_halted, update_state
from vision.gate import evaluate_vision_gate
from vision.parser import normalize_timeframes

try:
    import MetaTrader5 as mt5

    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False


def _is_live_mt5() -> bool:
    return MT5_AVAILABLE and bool(get_state("mt5_live"))


def _decimal_places(step: float) -> int:
    text = f"{step:.8f}".rstrip("0")
    return len(text.split(".")[1]) if "." in text else 0


def _round_price(price: float, tick_size: float) -> float:
    if tick_size <= 0:
        return price
    return round(round(price / tick_size) * tick_size, 10)


def _normalise_volume(requested_volume: float, sym_info) -> float:
    step = sym_info.volume_step or 0.01
    min_vol = sym_info.volume_min or step
    max_vol = sym_info.volume_max or requested_volume

    volume = min(max(requested_volume, min_vol), max_vol)
    steps = round((volume - min_vol) / step)
    normalized = min_vol + (steps * step)
    return round(normalized, _decimal_places(step))


def _enforce_stop_distance(direction: int, entry: float, sl: float, tp: float, sym_info) -> tuple[float, float]:
    tick_size = sym_info.trade_tick_size or sym_info.point or 0.0
    min_stop = max(sym_info.trade_stops_level * sym_info.point, tick_size or sym_info.point or 0.0)
    if min_stop <= 0:
        return sl, tp

    if direction == 1:
        sl = min(sl, entry - min_stop)
        tp = max(tp, entry + min_stop)
    else:
        sl = max(sl, entry + min_stop)
        tp = min(tp, entry - min_stop)

    return sl, tp


def _get_symbol_info():
    if not _is_live_mt5():
        return None

    mt5.symbol_select(cfg.MT5_SYMBOL, True)
    info = mt5.symbol_info(cfg.MT5_SYMBOL)
    if info is None:
        log.error(f"[Executor] Symbol {cfg.MT5_SYMBOL} not found: {mt5.last_error()}")
    return info


def _get_demo_tick() -> dict:
    return {
        "bid": 2000.0 + np.random.uniform(-0.5, 0.5),
        "ask": 2000.1 + np.random.uniform(-0.5, 0.5),
    }


def _format_signal_map(signals: dict | None) -> str:
    if not signals:
        return "none"
    parts = []
    for timeframe in sorted(signals):
        entry = signals.get(timeframe) or {}
        signal = entry.get("signal") or "unknown"
        source = entry.get("source") or "n/a"
        confidence = entry.get("confidence")
        if confidence is None:
            parts.append(f"{timeframe}={signal}@{source}")
        else:
            parts.append(f"{timeframe}={signal}({float(confidence):.2f})@{source}")
    return ", ".join(parts)


def _place_order(direction: int, atr: float) -> dict | None:
    if not _is_live_mt5():
        tick = _get_demo_tick()
        entry = tick["ask"] if direction == 1 else tick["bid"]
        sl = entry - direction * cfg.SL_MULT * atr
        tp = entry + direction * cfg.TP_MULT * atr
        log.info(
            f"[Executor] [DEMO] {'BUY' if direction == 1 else 'SELL'} | "
            f"entry={entry:.5f} SL={sl:.5f} TP={tp:.5f} | "
            f"SLxATR={cfg.SL_MULT} TPxATR={cfg.TP_MULT}"
        )
        return {
            "demo": True,
            "direction": direction,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "ticket": int(time.time()),
        }

    sym_info = _get_symbol_info()
    if sym_info is None:
        return None

    tick = mt5.symbol_info_tick(cfg.MT5_SYMBOL)
    if tick is None:
        log.error(f"[Executor] Cannot get tick for {cfg.MT5_SYMBOL}: {mt5.last_error()}")
        return None

    tick_size = sym_info.trade_tick_size or sym_info.point or 0.0
    entry = tick.ask if direction == 1 else tick.bid
    sl = entry - direction * cfg.SL_MULT * atr
    tp = entry + direction * cfg.TP_MULT * atr
    sl, tp = _enforce_stop_distance(direction, entry, sl, tp, sym_info)
    sl = _round_price(sl, tick_size)
    tp = _round_price(tp, tick_size)

    volume = _normalise_volume(cfg.LOT_SIZE, sym_info)
    if volume != cfg.LOT_SIZE:
        log.info(f"[Executor] Volume adjusted from {cfg.LOT_SIZE} to broker-safe {volume}")

    order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
    filling_mode = mt5.ORDER_FILLING_FOK
    if sym_info.filling_mode & 1:
        filling_mode = mt5.ORDER_FILLING_FOK
    elif sym_info.filling_mode & 2:
        filling_mode = mt5.ORDER_FILLING_IOC
    else:
        filling_mode = mt5.ORDER_FILLING_RETURN

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": cfg.MT5_SYMBOL,
        "volume": volume,
        "type": order_type,
        "price": entry,
        "sl": sl,
        "tp": tp,
        "deviation": cfg.SLIPPAGE_POINTS,
        "magic": cfg.MAGIC,
        "comment": cfg.ORDER_COMMENT,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_mode,
    }

    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        err = result.comment if result else mt5.last_error()
        retcode = getattr(result, "retcode", "?")
        log.error(f"[Executor] Order FAILED | retcode={retcode} | error={err}")
        return None

    log.info(
        f"[Executor] Order placed | ticket={result.order} | "
        f"{'BUY' if direction == 1 else 'SELL'} | "
        f"entry={entry:.5f} SL={sl:.5f} TP={tp:.5f}"
    )
    return {
        "demo": False,
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "ticket": result.order,
    }


def _get_open_positions() -> list:
    if not _is_live_mt5():
        return []

    mt5.symbol_select(cfg.MT5_SYMBOL, True)
    positions = mt5.positions_get(symbol=cfg.MT5_SYMBOL)
    if positions is None:
        return []
    return [pos for pos in positions if pos.magic == cfg.MAGIC]


class ExecutorWorker(threading.Thread):
    def __init__(self):
        super().__init__(name="ExecutorWorker", daemon=True)
        self._daily_reset_date = date.today()
        self._demo_positions = []
        self._vision_timeframes = normalize_timeframes(getattr(cfg, "VISION_TIMEFRAMES", []))

    def _daily_reset_if_needed(self):
        today = date.today()
        if today != self._daily_reset_date:
            self._daily_reset_date = today
            set_state("daily_pnl_usd", 0.0)
            log.info("[Executor] Daily PnL counter reset.")

    def _check_risk(self) -> tuple[bool, str]:
        daily_pnl = get_state("daily_pnl_usd")
        if daily_pnl <= -cfg.MAX_DAILY_LOSS_USD:
            if not trading_halted.is_set():
                trading_halted.set()
                log.warning(
                    f"[Executor] Daily loss limit hit | {daily_pnl:.2f} <= -{cfg.MAX_DAILY_LOSS_USD:.2f}"
                )
            return False, f"daily loss limit hit ({daily_pnl:.2f})"

        balance = get_state("account_balance")
        peak = get_state("peak_balance")
        if peak > 0 and (peak - balance) / peak >= cfg.MAX_DRAWDOWN_PCT:
            if not trading_halted.is_set():
                trading_halted.set()
                log.warning(f"[Executor] Max drawdown {cfg.MAX_DRAWDOWN_PCT * 100:.0f}% hit.")
            return False, f"max drawdown hit ({cfg.MAX_DRAWDOWN_PCT * 100:.0f}%)"

        n_open = len(_get_open_positions()) if _is_live_mt5() else len(self._demo_positions)
        set_state("open_positions", n_open)
        if n_open >= cfg.MAX_OPEN_TRADES:
            return False, f"max open trades reached ({n_open}/{cfg.MAX_OPEN_TRADES})"

        return True, ""

    def _settle_demo_positions(self, current_price: float):
        remaining = []
        for pos in self._demo_positions:
            direction = pos["direction"]
            entry = pos["entry"]
            tp = pos["tp"]
            sl = pos["sl"]

            hit_tp = (direction == 1 and current_price >= tp) or (direction == -1 and current_price <= tp)
            hit_sl = (direction == 1 and current_price <= sl) or (direction == -1 and current_price >= sl)

            if not (hit_tp or hit_sl):
                remaining.append(pos)
                continue

            exit_price = tp if hit_tp else sl
            pnl_pts = direction * (exit_price - entry)
            atr = pos.get("atr", 1.0)
            pnl_atr = pnl_pts / atr if atr else 0.0
            outcome = "TP HIT" if hit_tp else "SL HIT"
            log.info(
                f"[Executor] [DEMO] Position closed | outcome={outcome} | "
                f"ticket={pos['ticket']} | pnl={pnl_pts:+.5f} ({pnl_atr:+.2f} ATR)"
            )
            pnl_usd = pnl_pts * cfg.LOT_SIZE * 100
            _update_trade_stats(pnl_usd, hit_tp)

        self._demo_positions = remaining

    def run(self):
        log.info("[Executor] Starting.")

        while not stop_event.is_set():
            self._daily_reset_if_needed()

            if not _is_live_mt5() and self._demo_positions:
                tick = _get_demo_tick()
                mid = (tick["bid"] + tick["ask"]) / 2
                self._settle_demo_positions(mid)

            try:
                signal = signal_q.get(timeout=cfg.QUEUE_TIMEOUT)
            except Exception:
                continue

            direction = signal["direction"]
            atr = signal["atr"]
            close = signal["close"]
            conf = signal["p_bull"] if direction == 1 else signal["p_bear"]
            model_summary = (
                f"Bear={signal['p_bear']:.1%} Neut={signal['p_neut']:.1%} Bull={signal['p_bull']:.1%} | "
                f"lead={signal.get('lead_class', 'n/a')} ({signal.get('lead_confidence', 0.0):.1%})"
            )
            current_vision_signals = get_state("vision_signals")
            last_nonempty_vision_signals = get_state("vision_last_nonempty_signals")
            vision_state = get_state("vision_connection_state")
            vision_action = get_state("vision_decision_action") or "none"
            vision_reason = get_state("vision_decision_reason") or "n/a"
            vision_context = (
                f"state={vision_state} | action={vision_action} | current={_format_signal_map(current_vision_signals)} | "
                f"last_nonempty={_format_signal_map(last_nonempty_vision_signals)} | "
                f"reason={vision_reason}"
            )

            if trading_halted.is_set():
                log.info(
                    f"[Executor] DECISION=SKIP | side={'LONG' if direction == 1 else 'SHORT'} | "
                    f"model={model_summary} | vision={vision_context} | risk=trading halted"
                )
                continue

            risk_allowed, risk_reason = self._check_risk()
            if not risk_allowed:
                log.info(
                    f"[Executor] DECISION=SKIP | side={'LONG' if direction == 1 else 'SHORT'} | "
                    f"model={model_summary} | vision={vision_context} | risk={risk_reason}"
                )
                continue

            vision_gate = evaluate_vision_gate(
                direction=direction,
                configured_timeframes=self._vision_timeframes,
                vision_signals=current_vision_signals,
                last_nonempty_signals=last_nonempty_vision_signals,
                connection_state=vision_state,
                last_message_ts=get_state("vision_last_message_ts"),
                last_nonempty_message_ts=get_state("vision_last_nonempty_message_ts"),
                max_age_seconds=float(getattr(cfg, "VISION_MAX_SIGNAL_AGE_SECONDS", 0)),
                require_fresh_signal=bool(getattr(cfg, "VISION_REQUIRE_FRESH_SIGNAL", True)),
            )
            vision_context = (
                f"state={vision_state} | action={vision_action} | current={_format_signal_map(current_vision_signals)} | "
                f"last_nonempty={_format_signal_map(last_nonempty_vision_signals)} | "
                f"source={vision_gate.get('signal_source', 'n/a')} | reason={vision_reason}"
            )
            if not vision_gate["allowed"]:
                log.info(
                    f"[Executor] DECISION=SKIP | side={'LONG' if direction == 1 else 'SHORT'} | "
                    f"model={model_summary} | vision={vision_context} | gate={vision_gate['reason']}"
                )
                continue

            log.info(
                f"[Executor] DECISION=EXECUTE | side={'LONG' if direction == 1 else 'SHORT'} | "
                f"model={model_summary} | conf={conf:.3f} | price~={close:.5f} | ATR={atr:.5f} | "
                f"vision={vision_context} | gate={vision_gate['reason']} | "
                f"SLxATR={cfg.SL_MULT:.2f} TPxATR={cfg.TP_MULT:.2f}"
            )

            result = _place_order(direction, atr)
            if not result:
                continue

            set_state("open_positions", get_state("open_positions") + 1)
            update_state(total_trades=get_state("total_trades") + 1)

            if not _is_live_mt5():
                result["atr"] = atr
                self._demo_positions.append(result)

        log.info("[Executor] Stopped.")


def _update_trade_stats(pnl_usd: float, is_win: bool):
    daily = get_state("daily_pnl_usd") + pnl_usd
    wins = get_state("total_wins") + (1 if is_win else 0)
    losses = get_state("total_losses") + (0 if is_win else 1)
    update_state(
        daily_pnl_usd=daily,
        total_wins=wins,
        total_losses=losses,
        open_positions=max(0, get_state("open_positions") - 1),
    )
    log.info(f"[Executor] Trade settled | pnl_usd={pnl_usd:+.2f} | daily_pnl={daily:+.2f} | W/L={wins}/{losses}")
