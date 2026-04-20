import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
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
   
    open_positions = []
    max_trades = getattr(cfg, "MAX_OPEN_TRADES", 1)
   
    for i in range(n_bars - 1): # -1 because we enter on i+1
        curr_idx = i + 1
        row = df.loc[curr_idx]
       
        # ── 1. EVALUATE EXITS ── (allows intra-bar SL/TP hits on the entry candle)
        remaining_positions = []
        for pos in open_positions:
            exit_triggered = False
            exit_price = 0
            exit_reason = ""
           
            # TRACK MFE/MAE EXTREMES FOR THIS BAR
            if pos['type'] == 'BULL':
                if row['high'] > pos.get('mfe_price', pos['entry_price']): pos['mfe_price'] = row['high']
                if row['low'] < pos.get('mae_price', pos['entry_price']): pos['mae_price'] = row['low']
               
                # Pessimistic real-world check: check SL before TP
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
                if row['low'] < pos.get('mfe_price', pos['entry_price']): pos['mfe_price'] = row['low']
                if row['high'] > pos.get('mae_price', pos['entry_price']): pos['mae_price'] = row['high']
               
                # Pessimistic real-world check: check SL before TP
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
                # Cap MFE/MAE realistically against the exit boundary
                if pos['type'] == 'BULL':
                    mfe = max(0, pos['mfe_price'] - pos['entry_price'])
                    mae = max(0, pos['entry_price'] - pos['mae_price'])
                    if exit_reason == "SL": mae = max(0, pos['entry_price'] - exit_price)
                    pnl = (exit_price - pos['entry_price']) * lot_size
                else:
                    mfe = max(0, pos['entry_price'] - pos['mfe_price'])
                    mae = max(0, pos['mae_price'] - pos['entry_price'])
                    if exit_reason == "SL": mae = max(0, exit_price - pos['entry_price'])
                    pnl = (pos['entry_price'] - exit_price) * lot_size
               
                trades.append({
                    'entry_time': pos['entry_time'],
                    'exit_time': row['datetime'],
                    'type': pos['type'],
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'reason': exit_reason,
                    'mfe': mfe,
                    'mae': mae,
                    'p_bear': pos.get('p_bear', np.nan),
                    'p_neut': pos.get('p_neut', np.nan),
                    'p_bull': pos.get('p_bull', np.nan),
                    'lead_prob': pos.get('lead_prob', np.nan),
                })
            else:
                remaining_positions.append(pos)
               
        open_positions = remaining_positions
       
        # ── 2. EVALUATE ENTRIES ──
        n_longs = sum(1 for p in open_positions if p['type'] == 'BULL')
        n_shorts = sum(1 for p in open_positions if p['type'] == 'BEAR')
       
        can_go_long = False
        can_go_short = False
       
        if max_trades == 1:
            # NO hedging: maximum 1 trade active globally.
            if (n_longs + n_shorts) == 0:
                can_go_long = True
                can_go_short = True
        else:
            # HEDGING: allowed max_trades PER direction
            if n_longs < max_trades:
                can_go_long = True
            if n_shorts < max_trades:
                can_go_short = True
               
        if can_go_long or can_go_short:
            # Signal detection on bar i
            p_bear, p_neut, p_bull = y_prob[i]
           
            classes = ["BEAR", "NEUT", "BULL"]
            probs = [p_bear, p_neut, p_bull]
            lead_idx = np.argmax(probs)
            lead_cls = classes[lead_idx]

            direction = 0
            if lead_cls == "BULL" and p_bull >= threshold and can_go_long:
                direction = 1
            elif lead_cls == "BEAR" and p_bear >= threshold and can_go_short:
                direction = -1
               
            if direction != 0:
                # ENTER on OPEN of bar i+1
                entry_idx = i + 1
                entry_price = df.loc[entry_idx, 'open']
                atr = df.loc[entry_idx, 'atr']
               
                if not np.isnan(atr) and atr > 0:
                    lead_prob = p_bull if direction == 1 else p_bear
                    if direction == 1:
                        open_positions.append({
                            'type': 'BULL',
                            'entry_price': entry_price,
                            'sl': entry_price - (atr * sl_mult),
                            'tp': entry_price + (atr * tp_mult),
                            'entry_time': df.loc[entry_idx, 'datetime'],
                            'entry_idx': entry_idx,
                            'mfe_price': entry_price,
                            'mae_price': entry_price,
                            'p_bear': p_bear, 'p_neut': p_neut, 'p_bull': p_bull,
                            'lead_prob': lead_prob,
                        })
                    else:
                        open_positions.append({
                            'type': 'BEAR',
                            'entry_price': entry_price,
                            'sl': entry_price + (atr * sl_mult),
                            'tp': entry_price - (atr * tp_mult),
                            'entry_time': df.loc[entry_idx, 'datetime'],
                            'entry_idx': entry_idx,
                            'mfe_price': entry_price,
                            'mae_price': entry_price,
                            'p_bear': p_bear, 'p_neut': p_neut, 'p_bull': p_bull,
                            'lead_prob': lead_prob,
                        })

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
    num_bars = 100 # ~6.2 years on M5 timeframe!
    print(f"[*] Fetching {num_bars} bars (attempting up to 6+ years) for {cfg.MT5_SYMBOL}...")
    df_1m = fetch_bars_mt5(cfg.MT5_SYMBOL, cfg.MT5_TIMEFRAME, num_bars)
    df_htf = fetch_bars_mt5(cfg.MT5_SYMBOL, cfg.MT5_HTF, (num_bars // 2) + 1000)
   
    if df_1m.empty:
        print("[!] Failed to fetch data. Check symbol name and market hours.")
        return

    first_date = df_1m.iloc[0]['datetime']
    last_date = df_1m.iloc[-1]['datetime']
    hist_duration = last_date - first_date
    print(f"\n[*] Historical Data Span:")
    print(f"    Start Date : {first_date}")
    print(f"    End Date   : {last_date}")
    print(f"    Duration   : {hist_duration}\n")

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
       
        longs = results[results['type'] == 'BULL']
        shorts = results[results['type'] == 'BEAR']
       
        print(f"  Total Trades    : {len(results)}")
        print(f"  Win Rate        : {win_rate:.1f}%")
        print(f"  Profit Factor   : {pf:.2f}")
        print(f"  Total Net PnL   : ${total_pnl:.2f}")
        print(f"  Avg Trade PnL   : ${results['pnl'].mean():.2f}")
        print(f"  Max Win         : ${results['pnl'].max():.2f}")
        print(f"  Max Loss        : ${results['pnl'].min():.2f}")
       
        print("\n" + "─"*30 + " DIRECTIONAL BREAKDOWN " + "─"*30)
        if not longs.empty:
            l_wr = (longs['pnl'] > 0).mean() * 100
            print(f"  [BUYS]  Count: {len(longs):<3} | Win Rate: {l_wr:>5.1f}% | Net PnL: ${longs['pnl'].sum():>7.2f} | Avg PnL: ${longs['pnl'].mean():>6.2f}")
        else:
            print("  [BUYS]  No Trades taken.")
           
        if not shorts.empty:
            s_wr = (shorts['pnl'] > 0).mean() * 100
            print(f"  [SELLS] Count: {len(shorts):<3} | Win Rate: {s_wr:>5.1f}% | Net PnL: ${shorts['pnl'].sum():>7.2f} | Avg PnL: ${shorts['pnl'].mean():>6.2f}")
        else:
            print("  [SELLS] No Trades taken.")
           
        print("\n" + "─"*30 + " EXCURSION METRICS " + "─"*30)
        avg_mfe = results['mfe'].mean()
        avg_mae = results['mae'].mean()
        print(f"  Avg MFE per trade : {avg_mfe:.4f} pts")
        print(f"  Avg MAE per trade  : {avg_mae:.4f} pts")
        if avg_mae > 0:
             print(f"  Global MFE/MAE Ratio: {(avg_mfe / avg_mae):.2f}")
       
        results['cum_pnl'] = results['pnl'].cumsum()
       
        # ── PLOT 1: Equity Curve ──────────────────────────────────────────────
        plt.style.use('dark_background')
        fig1, ax1 = plt.subplots(figsize=(14, 5))
        ax1.plot(results['cum_pnl'], color='#3fb950', linewidth=2, label='Equity Curve')
        ax1.fill_between(range(len(results)), results['cum_pnl'], 0,
                         where=results['cum_pnl'] >= 0, color='#3fb950', alpha=0.15)
        ax1.fill_between(range(len(results)), results['cum_pnl'], 0,
                         where=results['cum_pnl'] < 0, color='#f85149', alpha=0.15)
        ax1.set_title(f'Equity Curve — {cfg.MT5_SYMBOL}  (Trades={len(results)}, WR={win_rate:.1f}%)',
                      fontsize=13, pad=10)
        ax1.set_xlabel('Trade Number', fontsize=10)
        ax1.set_ylabel('Cumulative PnL ($)', fontsize=10)
        ax1.grid(True, alpha=0.15)
        ax1.legend(fontsize=10)
        fig1.tight_layout()
        out_plot = Path("backtest_analysis.png")
        fig1.savefig(out_plot, dpi=150, bbox_inches='tight')
        print(f"\n[*] Equity curve saved to {out_plot}")
        plt.close(fig1)

        # ── PLOT 2: Probability Distributions ────────────────────────────────
        _plot_prob_distributions(y_prob, results)
       
        # ── PLOT 3: Probability Confidence of Signals ─────────────────────────
        _plot_signal_confidence(results)
       
        # Save CSV
        out_csv = Path("backtest_trades.csv")
        results.to_csv(out_csv, index=False)
        print(f"[*] Detailed trades saved to {out_csv}")
       
    mt5.shutdown()
    print("="*60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  3. PROBABILITY DISTRIBUTION PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def _plot_prob_distributions(y_prob: np.ndarray, results: pd.DataFrame):
    """
    Three-panel figure:
      Panel A – KDE of all-bar class probabilities (Bear / Neut / Bull)
      Panel B – Violin plot of lead_prob split by Win vs Loss
      Panel C – Scatter: lead_prob vs PnL, coloured by direction
    """
    plt.style.use('dark_background')
    COLORS = {'bear': '#f85149', 'neut': '#e3b341', 'bull': '#3fb950'}

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle('Model Probability Distribution — Backtest Analysis',
                 fontsize=15, fontweight='bold', y=1.01)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── Panel A: KDE of all bars ──────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0])
    x_grid = np.linspace(0, 1, 400)
    for col_idx, (label, color) in enumerate(
            zip(['Bear', 'Neutral', 'Bull'], [COLORS['bear'], COLORS['neut'], COLORS['bull']])
    ):
        data = y_prob[:, col_idx]
        try:
            kde = gaussian_kde(data, bw_method='scott')
            ax_a.plot(x_grid, kde(x_grid), color=color, linewidth=2, label=label)
            ax_a.fill_between(x_grid, kde(x_grid), alpha=0.12, color=color)
        except Exception:
            pass
    ax_a.set_title('All-Bar Class Probabilities (KDE)', fontsize=11)
    ax_a.set_xlabel('Probability', fontsize=9)
    ax_a.set_ylabel('Density', fontsize=9)
    ax_a.set_xlim(0, 1)
    ax_a.legend(fontsize=9)
    ax_a.grid(True, alpha=0.15)
    _add_stats_text(ax_a, y_prob)

    # ── Panel B: Violin – Win vs Loss lead probability ─────────────────────
    ax_b = fig.add_subplot(gs[1])
    if 'lead_prob' in results.columns and results['lead_prob'].notna().any():
        wins  = results.loc[results['pnl'] > 0, 'lead_prob'].dropna()
        losses = results.loc[results['pnl'] <= 0, 'lead_prob'].dropna()
        groups = []
        labels = []
        if len(wins):   groups.append(wins.values);   labels.append(f'Win\n(n={len(wins)})')
        if len(losses): groups.append(losses.values); labels.append(f'Loss\n(n={len(losses)})')
        parts = ax_b.violinplot(groups, positions=range(len(groups)),
                                showmedians=True, showextrema=True)
        vcolors = ['#3fb950', '#f85149']
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(vcolors[i % 2])
            pc.set_alpha(0.45)
        parts['cmedians'].set_colors(['white'] * len(groups))
        parts['cmaxes'].set_colors(['gray'] * len(groups))
        parts['cmins'].set_colors(['gray'] * len(groups))
        parts['cbars'].set_colors(['gray'] * len(groups))
        ax_b.set_xticks(range(len(labels)))
        ax_b.set_xticklabels(labels, fontsize=9)
        ax_b.set_title('Signal Confidence: Win vs Loss', fontsize=11)
        ax_b.set_ylabel('Lead-Class Probability', fontsize=9)
        ax_b.set_ylim(0, 1)
        ax_b.axhline(np.median(results['lead_prob'].dropna()),
                     color='#58a6ff', linestyle='--', linewidth=1, alpha=0.6,
                     label=f'Overall Median: {np.median(results["lead_prob"].dropna()):.3f}')
        ax_b.legend(fontsize=8)
        ax_b.grid(True, alpha=0.15)
    else:
        ax_b.text(0.5, 0.5, 'lead_prob not captured\n(no signal trades)',
                  ha='center', va='center', transform=ax_b.transAxes, color='gray')

    # ── Panel C: Scatter lead_prob vs PnL ─────────────────────────────────
    ax_c = fig.add_subplot(gs[2])
    if 'lead_prob' in results.columns and results['lead_prob'].notna().any():
        for ttype, color, marker in [('BULL', COLORS['bull'], '^'), ('BEAR', COLORS['bear'], 'v')]:
            sub = results[results['type'] == ttype].dropna(subset=['lead_prob'])
            if not sub.empty:
                ax_c.scatter(sub['lead_prob'], sub['pnl'],
                             c=color, marker=marker, alpha=0.55, s=30, label=ttype)
        ax_c.axhline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
        ax_c.set_title('Signal Confidence vs Trade PnL', fontsize=11)
        ax_c.set_xlabel('Lead-Class Probability', fontsize=9)
        ax_c.set_ylabel('PnL ($)', fontsize=9)
        ax_c.legend(fontsize=9)
        ax_c.grid(True, alpha=0.15)
    else:
        ax_c.text(0.5, 0.5, 'No signal data', ha='center', va='center',
                  transform=ax_c.transAxes, color='gray')

    fig.tight_layout()
    out = Path('backtest_prob_distributions.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'[*] Probability distribution chart saved to {out}')
    plt.close(fig)


def _plot_signal_confidence(results: pd.DataFrame):
    """
    Three-panel figure:
      Panel A – Histogram of p_bear / p_neut / p_bull for signal bars only
      Panel B – Stacked bar of class probs at each trade (chronological)
      Panel C – CDF of lead probability (Win vs Loss)
    """
    prob_cols = ['p_bear', 'p_neut', 'p_bull']
    if not all(c in results.columns for c in prob_cols):
        return
    df = results.dropna(subset=prob_cols).reset_index(drop=True)
    if df.empty:
        return

    plt.style.use('dark_background')
    COLORS = {'bear': '#f85149', 'neut': '#e3b341', 'bull': '#3fb950'}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Signal-Bar Probability Deep-Dive', fontsize=15, fontweight='bold', y=1.01)

    # ── Panel A: Histogram of signal probs ──────────────────────────────────
    ax = axes[0]
    bins = np.linspace(0, 1, 30)
    for col, label, color in zip(prob_cols, ['Bear', 'Neutral', 'Bull'],
                                 [COLORS['bear'], COLORS['neut'], COLORS['bull']]):
        ax.hist(df[col], bins=bins, alpha=0.5, color=color, label=label, density=True)
    ax.set_title('Class Prob Distribution at Signal Bars', fontsize=11)
    ax.set_xlabel('Probability', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    # ── Panel B: Stacked bar per trade ──────────────────────────────────────
    ax = axes[1]
    x = np.arange(len(df))
    ax.bar(x, df['p_bear'], color=COLORS['bear'], label='Bear', width=1.0)
    ax.bar(x, df['p_neut'], bottom=df['p_bear'], color=COLORS['neut'], label='Neutral', width=1.0)
    ax.bar(x, df['p_bull'], bottom=df['p_bear'] + df['p_neut'], color=COLORS['bull'], label='Bull', width=1.0)
    ax.axhline(1.0, color='white', linewidth=0.5, alpha=0.3)
    ax.set_title('Stacked Class Probs per Trade (chronological)', fontsize=11)
    ax.set_xlabel('Trade Index', fontsize=9)
    ax.set_ylabel('Probability (stacked)', fontsize=9)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.1)

    # ── Panel C: CDF of lead prob – Win vs Loss ────────────────────────────
    ax = axes[2]
    if 'lead_prob' in df.columns:
        for outcome, color, label in [
            (df['pnl'] > 0,  COLORS['bull'], 'Win'),
            (df['pnl'] <= 0, COLORS['bear'], 'Loss'),
        ]:
            sub = df.loc[outcome, 'lead_prob'].sort_values()
            if not sub.empty:
                cdf = np.arange(1, len(sub) + 1) / len(sub)
                ax.plot(sub.values, cdf, color=color, linewidth=2, label=f'{label} (n={len(sub)})')
        ax.set_title('CDF of Lead-Class Probability', fontsize=11)
        ax.set_xlabel('Lead-Class Probability', fontsize=9)
        ax.set_ylabel('Cumulative Fraction', fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.15)

    fig.tight_layout()
    out = Path('backtest_signal_confidence.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'[*] Signal confidence chart saved to {out}')
    plt.close(fig)


def _add_stats_text(ax, y_prob: np.ndarray):
    """Overlay mean ± std annotation on the KDE axis."""
    labels = ['Bear', 'Neut', 'Bull']
    colors = ['#f85149', '#e3b341', '#3fb950']
    lines = []
    for i, (lbl, col) in enumerate(zip(labels, colors)):
        m = y_prob[:, i].mean()
        s = y_prob[:, i].std()
        lines.append(f'{lbl}: μ={m:.3f} σ={s:.3f}')
    text = '\n'.join(lines)
    ax.text(0.97, 0.97, text, transform=ax.transAxes, fontsize=7.5,
            va='top', ha='right', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#161b22', alpha=0.7))


if __name__ == "__main__":
    main()
