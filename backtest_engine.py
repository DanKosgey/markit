
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import json
from datetime import datetime
from pathlib import Path

import config as cfg
from engine.features import build_feature_matrix
from workers.data_worker import fetch_bars_mt5, connect_mt5

# ─────────────────────────────────────────────────────────────────────────────
#  1. IMPROVED SERIAL BACKTESTER
# ─────────────────────────────────────────────────────────────────────────────

def run_robust_backtest(df, y_prob, threshold=0.52, tp_mult=2.0, sl_mult=1.0, lot_size=1.0):
    """
    Simulates trades with Next-Bar Entry logic.
    Prediction at Bar i -> Entry at Open of Bar i+1.
    """
    # y_prob is (N, 3) -> [Bear, Neut, Bull]
    n_bars = len(df)
    trades = []
    
    in_trade = False
    pos = None # {type, entry_price, sl, tp, entry_idx}
    
    for i in range(n_bars - 1): # -1 because we enter on i+1
        if not in_trade:
            # Signal detection on bar i
            p_bear, p_neut, p_bull = y_prob[i]
            
            classes = ["BEAR", "NEUT", "BULL"]
            probs = [p_bear, p_neut, p_bull]
            lead_idx = np.argmax(probs)
            lead_cls = classes[lead_idx]

            direction = 0
            if lead_cls == "BULL" and p_bull >= threshold:
                direction = 1
            elif lead_cls == "BEAR" and p_bear >= threshold:
                direction = -1
                
            if direction != 0:
                # ENTER on OPEN of bar i+1
                entry_idx = i + 1
                entry_price = df.loc[entry_idx, 'open']
                atr = df.loc[entry_idx, 'atr']
                
                if np.isnan(atr) or atr <= 0:
                    continue # Cannot size without ATR
                    
                in_trade = True
                if direction == 1:
                    pos = {
                        'type': 'BULL',
                        'entry_price': entry_price,
                        'sl': entry_price - (atr * sl_mult),
                        'tp': entry_price + (atr * tp_mult),
                        'entry_time': df.loc[entry_idx, 'datetime'],
                        'entry_idx': entry_idx
                    }
                else:
                    pos = {
                        'type': 'BEAR',
                        'entry_price': entry_price,
                        'sl': entry_price + (atr * sl_mult),
                        'tp': entry_price - (atr * tp_mult),
                        'entry_time': df.loc[entry_idx, 'datetime'],
                        'entry_idx': entry_idx
                    }
        else:
            # CHECK EXIT on bar i+1
            curr_idx = i + 1
            row = df.loc[curr_idx]
            exit_triggered = False
            exit_price = 0
            exit_reason = ""
            
            if pos['type'] == 'BULL':
                if row['low'] <= pos['sl']:
                    exit_triggered = True
                    exit_price = pos['sl']
                    exit_reason = "SL"
                elif row['high'] >= pos['tp']:
                    exit_triggered = True
                    exit_price = pos['tp']
                    exit_reason = "TP"
                    
                if not exit_triggered and getattr(cfg, "USE_TRAILING_STOP", False):
                    profit = row['close'] - pos['entry_price']
                    if profit >= getattr(cfg, "TRAIL_ACTIVATION", 2.0):
                        new_sl = row['close'] - getattr(cfg, "TRAIL_DISTANCE", 1.0)
                        if new_sl > pos['sl']:
                            pos['sl'] = new_sl
                            
            else: # BEAR
                # Correct Bear Logic
                if row['high'] >= pos['sl']:
                    exit_triggered = True
                    exit_price = pos['sl']
                    exit_reason = "SL"
                elif row['low'] <= pos['tp']:
                    exit_triggered = True
                    exit_price = pos['tp']
                    exit_reason = "TP"
                
                if not exit_triggered and getattr(cfg, "USE_TRAILING_STOP", False):
                    profit = pos['entry_price'] - row['close']
                    if profit >= getattr(cfg, "TRAIL_ACTIVATION", 2.0):
                        new_sl = row['close'] + getattr(cfg, "TRAIL_DISTANCE", 1.0)
                        if new_sl < pos['sl']:
                            pos['sl'] = new_sl
                    
            if exit_triggered:
                pnl = (exit_price - pos['entry_price']) if pos['type'] == 'BULL' else (pos['entry_price'] - exit_price)
                # Simple point-based pnl. For Volatility Index, points = dollars * lot_size approx
                trades.append({
                    'entry_time': pos['entry_time'],
                    'exit_time': row['datetime'],
                    'type': pos['type'],
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'pnl': pnl * lot_size,
                    'reason': exit_reason
                })
                in_trade = False
                pos = None
                
    return pd.DataFrame(trades)

# ─────────────────────────────────────────────────────────────────────────────
#  2. MAIN ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  STRATEGY BACKTESTER — VOLATILITY INDEX EDITION")
    print("="*60)
    
    if not connect_mt5():
        return

    # -- 1. Data Fetching --
    num_bars = 10000 # ~6.2 years on M5 timeframe!
    print(f"[*] Fetching {num_bars} bars (attempting up to 6+ years) for {cfg.MT5_SYMBOL}...")
    df_1m = fetch_bars_mt5(cfg.MT5_SYMBOL, cfg.MT5_TIMEFRAME, num_bars)
    df_htf = fetch_bars_mt5(cfg.MT5_SYMBOL, cfg.MT5_HTF, (num_bars // 2) + 1000)
    
    if df_1m.empty:
        print("[!] Failed to fetch data. Check symbol name and market hours.")
        return

    # -- 2. Feature Engineering --
    print("[*] Building feature matrix...")
    full_df = build_feature_matrix(df_1m, df_htf, cfg)
    
    # Mirror FeatureWorker Alignment
    with open(cfg.FEAT_COLS_PATH, 'r') as f:
        feature_cols = json.load(f)
        
    X = full_df.copy()
    # Filling NaNs like FeatureWorker does
    for col in X.columns:
        if col in cfg.CATEGORICAL_COLS:
            X[col] = X[col].fillna(-1).astype(int).astype('category')
        elif "bars_ago" in col:
            X[col] = X[col].fillna(9999)
        else:
            X[col] = X[col].fillna(0)
            
    X_feat = X[feature_cols]
    
    # -- 3. Prediction --
    print(f"[*] Loading model: {cfg.MODEL_PATH.name}")
    model = joblib.load(cfg.MODEL_PATH)
    
    print(f"[*] Generating predictions for {len(X_feat)} bars...")
    y_prob = model.predict_proba(X_feat)
    
    # -- 4. Execution --
    print(f"[*] Running serial backtest (Threshold={cfg.PROB_THRESHOLD})...")
    results = run_robust_backtest(
        full_df.reset_index(drop=True), 
        y_prob, 
        threshold=cfg.PROB_THRESHOLD,
        tp_mult=cfg.TP_MULT,
        sl_mult=cfg.SL_MULT,
        lot_size=cfg.LOT_SIZE
    )
    
    # -- 5. Report --
    if results.empty:
        print("\n[!] NO TRADES TAKEN. Try lowering PROB_THRESHOLD in config.py.")
    else:
        print("\n" + "─"*30 + " PERFORMANCE REPORT " + "─"*30)
        total_pnl = results['pnl'].sum()
        win_rate = (results['pnl'] > 0).mean() * 100
        pf = abs(results[results['pnl'] > 0]['pnl'].sum() / results[results['pnl'] < 0]['pnl'].sum()) if any(results['pnl'] < 0) else float('inf')
        
        print(f"  Total Trades    : {len(results)}")
        print(f"  Win Rate        : {win_rate:.1f}%")
        print(f"  Profit Factor   : {pf:.2f}")
        print(f"  Total Net PnL   : ${total_pnl:.2f}")
        print(f"  Avg Trade PnL   : ${results['pnl'].mean():.2f}")
        print(f"  Max Win         : ${results['pnl'].max():.2f}")
        print(f"  Max Loss        : ${results['pnl'].min():.2f}")
        
        results['cum_pnl'] = results['pnl'].cumsum()
        
        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(results['cum_pnl'], color='#3fb950', linewidth=2, label='Equity Curve')
        plt.fill_between(range(len(results)), results['cum_pnl'], 0, color='#3fb950', alpha=0.1)
        plt.title(f'Backtest: {cfg.MT5_SYMBOL} (Trades={len(results)}, WR={win_rate:.1f}%)')
        plt.xlabel('Trade Number')
        plt.ylabel('Profit / Loss ($)')
        plt.grid(True, alpha=0.2)
        plt.legend()
        
        out_plot = Path("backtest_analysis.png")
        plt.savefig(out_plot)
        print(f"\n[*] Equity curve saved to {out_plot}")
        
        # Save CSV
        out_csv = Path("backtest_trades.csv")
        results.to_csv(out_csv, index=False)
        print(f"[*] Detailed trades saved to {out_csv}")
        
    mt5.shutdown()
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
