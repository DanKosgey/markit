"""
engine/features.py — Full feature pipeline mirroring the notebook exactly.

Given a raw OHLCV DataFrame (from MT5 or any source), this module produces
the same feature matrix the XGBoost model was trained on.
"""

import itertools
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — CORE INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def calc_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def calc_rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast  = series.ewm(span=fast, adjust=False).mean()
    ema_slow  = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line  = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, sig_line, macd_line - sig_line


def calc_bollinger(series: pd.Series, n: int = 20, k: float = 2.0):
    mid   = series.rolling(n).mean()
    std   = series.rolling(n).std()
    upper = mid + k * std
    lower = mid - k * std
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    bw    = (upper - lower) / mid.replace(0, np.nan)
    return upper, mid, lower, pct_b, bw


def calc_vwap(df: pd.DataFrame) -> pd.Series:
    """Session VWAP — resets each calendar day."""
    df = df.copy()
    df["_tp"]   = (df["high"] + df["low"] + df["close"]) / 3
    df["_date"] = df["datetime"].dt.date
    df["_tpv"]  = df["_tp"] * df["volume"]
    vwap = (
        df.groupby("_date")["_tpv"].cumsum()
        / df.groupby("_date")["volume"].cumsum()
    )
    return vwap


def session_label(h: int) -> int:
    if 0  <= h < 7:  return 0   # Asian
    if 7  <= h < 12: return 1   # London
    if 12 <= h < 17: return 2   # New York
    return 3                    # Off-hours


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4.3 — MULTI-MA RELATIONAL FEATURE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def build_ma_features(
    df: pd.DataFrame,
    ma_periods: list = None,
    lookback: int = 30,
    atr_col: str = "atr",
) -> pd.DataFrame:
    if ma_periods is None:
        ma_periods = [3, 5, 8, 13, 21, 34, 55]

    feat_df = pd.DataFrame(index=df.index)
    atr     = df[atr_col].replace(0, np.nan).ffill()
    close   = df["close"]

    # Layer 1: MA stack + distance from price
    ma_cols = []
    for p in ma_periods:
        col = f"ema_{p}"
        feat_df[col] = close.ewm(span=p, adjust=False).mean()
        feat_df[f"ma_dist_price_{p}"] = (feat_df[col] - close) / atr
        ma_cols.append(col)

    # Layer 2: Pairwise ATR-normalised distance
    for i, j in itertools.combinations(ma_periods, 2):
        pfx = f"ma_{i}_{j}"
        ei, ej = feat_df[f"ema_{i}"], feat_df[f"ema_{j}"]
        feat_df[f"{pfx}_dist"]      = (ei - ej) / atr
        feat_df[f"{pfx}_absdist"]   = feat_df[f"{pfx}_dist"].abs()
        feat_df[f"{pfx}_dist_roc"]  = feat_df[f"{pfx}_dist"].diff()
        dist_prev = feat_df[f"{pfx}_absdist"].shift(lookback)
        feat_df[f"{pfx}_expanding"] = np.where(
            feat_df[f"{pfx}_absdist"] > dist_prev, 1, -1
        )

    # Layer 3: Ordinal rank
    ema_vals = feat_df[ma_cols].values
    ranks    = np.zeros_like(ema_vals, dtype=int)
    for t in range(len(feat_df)):
        ranks[t] = np.argsort(np.argsort(-ema_vals[t])) + 1
    for idx, p in enumerate(ma_periods):
        feat_df[f"ma_rank_{p}"]     = ranks[:, idx]
        feat_df[f"ma_rank_chg_{p}"] = feat_df[f"ma_rank_{p}"].diff()

    feat_df["stack_perfect_bull"] = (feat_df[ma_cols].diff(axis=1).iloc[:, 1:] < 0).all(axis=1).astype(int)
    feat_df["stack_perfect_bear"] = (feat_df[ma_cols].diff(axis=1).iloc[:, 1:] > 0).all(axis=1).astype(int)

    # Layer 4: Crossover memory
    for i, j in itertools.combinations(ma_periods, 2):
        ei, ej = feat_df[f"ema_{i}"], feat_df[f"ema_{j}"]
        up_cross = (ei > ej) & (ei.shift(1) <= ej.shift(1))
        dn_cross = (ei < ej) & (ei.shift(1) >= ej.shift(1))
        feat_df[f"cross_{i}_{j}"] = np.where(up_cross, 1, np.where(dn_cross, -1, 0))

        feat_df[f"bars_since_up_{i}_{j}"] = up_cross.cumsum()
        feat_df[f"bars_since_up_{i}_{j}"] = feat_df.groupby(f"bars_since_up_{i}_{j}").cumcount().clip(upper=lookback)
        feat_df[f"bars_since_dn_{i}_{j}"] = dn_cross.cumsum()
        feat_df[f"bars_since_dn_{i}_{j}"] = feat_df.groupby(f"bars_since_dn_{i}_{j}").cumcount().clip(upper=lookback)

    # Layer 5: Higher-order aggregates
    dist_cols = [c for c in feat_df.columns if "_absdist" in c]
    feat_df["ma_fan_angle"] = feat_df[dist_cols].mean(axis=1)
    feat_df["ma_fan_roc"]   = feat_df["ma_fan_angle"].diff()
    for p in ma_periods:
        feat_df[f"ma_vel_{p}"] = feat_df[f"ema_{p}"].diff() / atr
        feat_df[f"ma_acc_{p}"] = feat_df[f"ma_vel_{p}"].diff()
    feat_df["avg_ma_velocity"]     = feat_df[[f"ma_vel_{p}" for p in ma_periods]].mean(axis=1)
    feat_df["velocity_dispersion"] = feat_df[[f"ma_vel_{p}" for p in ma_periods]].std(axis=1)

    return feat_df


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6 — MARKET STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

def fractal_pivots(df: pd.DataFrame, left: int, right: int, min_atr_mult: float = 0.3):
    highs = df["high"].values
    lows  = df["low"].values
    atr   = df["atr"].values
    n     = len(df)
    is_ph = np.zeros(n, dtype=bool)
    is_pl = np.zeros(n, dtype=bool)

    for i in range(left, n - right):
        sz = highs[i] - lows[i]
        if highs[i] == highs[i - left : i + right + 1].max() and sz >= atr[i] * min_atr_mult:
            is_ph[i] = True
        if lows[i]  == lows[i  - left : i + right + 1].min() and sz >= atr[i] * min_atr_mult:
            is_pl[i] = True

    return pd.Series(is_ph, index=df.index), pd.Series(is_pl, index=df.index)


def compute_swing_strength_reversal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["swing_strength"] = np.nan
    df["reversal_prob"]  = np.nan
    last_sl_price = None
    last_sh_price = None

    def _strength(row, sz):
        atr = row["atr"] if row["atr"] > 0 else 1
        s1  = min(sz / (atr * 3.0), 1.0) * 100
        s2  = min(row["vol_ratio"] / 2.5, 1.0) * 100
        s3  = min(abs(row["rsi"] - 50) / 30.0, 1.0) * 100
        s4  = min(row["vol_regime"] / 1.5, 1.0) * 100
        return round(s1 * 0.40 + s2 * 0.25 + s3 * 0.20 + s4 * 0.15, 1)

    def _prob_high(row, sz, prior_sl=None):
        atr = row["atr"] if row["atr"] > 0 else 1
        p   = 0.50
        rsi = row["rsi"]
        p  += 0.20 if rsi > 70 else (0.10 if rsi > 60 else (-0.10 if rsi < 50 else 0))
        p  += 0.10 if row["vol_ratio"] > 1.5 else (-0.05 if row["vol_ratio"] < 0.8 else 0.05)
        p  += 0.10 if sz > atr * 2 else (0.05 if sz > atr else 0)
        if prior_sl is not None:
            p += 0.05 if abs(row["close"] - prior_sl) < atr * 0.5 else 0
        return round(min(max(p, 0.10), 0.95) * 100, 1)

    def _prob_low(row, sz, prior_sh=None):
        atr = row["atr"] if row["atr"] > 0 else 1
        p   = 0.50
        rsi = row["rsi"]
        p  += 0.20 if rsi < 30 else (0.10 if rsi < 40 else (-0.10 if rsi > 50 else 0))
        p  += 0.10 if row["vol_ratio"] > 1.5 else (-0.05 if row["vol_ratio"] < 0.8 else 0.05)
        p  += 0.10 if sz > atr * 2 else (0.05 if sz > atr else 0)
        if prior_sh is not None:
            p += 0.05 if abs(row["close"] - prior_sh) < atr * 0.5 else 0
        return round(min(max(p, 0.10), 0.95) * 100, 1)

    for idx, row in df.iterrows():
        sz = row["high"] - row["low"]
        if row["is_swing_high"]:
            df.at[idx, "swing_strength"] = _strength(row, sz)
            df.at[idx, "reversal_prob"]  = _prob_high(row, sz, last_sl_price)
            last_sh_price = row["close"]
        if row["is_swing_low"]:
            df.at[idx, "swing_strength"] = _strength(row, sz)
            df.at[idx, "reversal_prob"]  = _prob_low(row, sz, last_sh_price)
            last_sl_price = row["close"]

    return df


def compute_bos_choch(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["is_bos_bull", "is_bos_bear", "is_choch_bull", "is_choch_bear"]:
        df[col] = False

    last_sh = prev_sh = last_sl = prev_sl = None

    for i in range(1, len(df)):
        row  = df.iloc[i]
        prev = df.iloc[i - 1]

        if df.iloc[i]["is_swing_high"]:
            prev_sh, last_sh = last_sh, row["close"]
        if df.iloc[i]["is_swing_low"]:
            prev_sl, last_sl = last_sl, row["close"]

        bos_bull   = last_sh is not None and prev["close"] <= last_sh < row["close"]
        bos_bear   = last_sl is not None and prev["close"] >= last_sl > row["close"]
        choch_bull = bos_bull and prev_sl is not None and last_sl is not None and last_sl < prev_sl
        choch_bear = bos_bear and prev_sh is not None and last_sh is not None and last_sh > prev_sh

        df.at[df.index[i], "is_bos_bull"]   = bos_bull   and not choch_bull
        df.at[df.index[i], "is_bos_bear"]   = bos_bear   and not choch_bear
        df.at[df.index[i], "is_choch_bull"] = choch_bull
        df.at[df.index[i], "is_choch_bear"] = choch_bear

    trend = False
    df["is_bullish_trend"] = False
    for i in range(len(df)):
        if df.iloc[i]["is_bos_bull"] or df.iloc[i]["is_choch_bull"]:
            trend = True
        elif df.iloc[i]["is_bos_bear"] or df.iloc[i]["is_choch_bear"]:
            trend = False
        df.at[df.index[i], "is_bullish_trend"] = trend

    return df


def detect_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["is_ob_bull", "is_ob_bear"]:
        df[col] = False
    for col in ["ob_bull_top", "ob_bull_bot", "ob_bear_top", "ob_bear_bot"]:
        df[col] = np.nan
    df["ob_bull_filled"] = False
    df["ob_bear_filled"] = False

    for i in range(1, len(df)):
        if df.iloc[i]["is_bos_bull"] or df.iloc[i]["is_choch_bull"]:
            for lb in range(i - 1, 0, -1):
                if df.iloc[lb]["close"] < df.iloc[lb]["open"]:
                    df.at[df.index[lb], "is_ob_bull"]  = True
                    df.at[df.index[lb], "ob_bull_top"] = df.iloc[lb]["high"]
                    df.at[df.index[lb], "ob_bull_bot"] = df.iloc[lb]["low"]
                    break
        if df.iloc[i]["is_bos_bear"] or df.iloc[i]["is_choch_bear"]:
            for lb in range(i - 1, 0, -1):
                if df.iloc[lb]["close"] > df.iloc[lb]["open"]:
                    df.at[df.index[lb], "is_ob_bear"]  = True
                    df.at[df.index[lb], "ob_bear_top"] = df.iloc[lb]["high"]
                    df.at[df.index[lb], "ob_bear_bot"] = df.iloc[lb]["low"]
                    break

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 8 — LOOKBACK FEATURE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_lookback_features(
    df: pd.DataFrame,
    event_col: str,
    prefix: str,
    n: int,
    extra_cols: list = None,
) -> pd.DataFrame:
    events = []
    for i, row in df.iterrows():
        if row[event_col]:
            ev = {"pos": df.index.get_loc(i), "close": row["close"]}
            if extra_cols:
                for c in extra_cols:
                    ev[c] = row[c]
            events.append(ev)

    rows = []
    for i, (idx, row) in enumerate(df.iterrows()):
        cur_atr  = row["atr"] if (not np.isnan(row["atr"]) and row["atr"] > 0) else 1.0
        past_evs = [e for e in events if e["pos"] < i][::-1]
        feat     = {}
        for k in range(1, n + 1):
            pfx = f"{prefix}{k}"
            if k - 1 < len(past_evs):
                ev = past_evs[k - 1]
                feat[f"{pfx}_bars_ago"] = i - ev["pos"]
                feat[f"{pfx}_raw_dist"] = round(row["close"] - ev["close"], 5)
                feat[f"{pfx}_atr_dist"] = round((row["close"] - ev["close"]) / cur_atr, 4)
                if extra_cols:
                    for c in extra_cols:
                        feat[f"{pfx}_{c}"] = ev[c]
            else:
                feat[f"{pfx}_bars_ago"] = np.nan
                feat[f"{pfx}_raw_dist"] = np.nan
                feat[f"{pfx}_atr_dist"] = np.nan
                if extra_cols:
                    for c in extra_cols:
                        feat[f"{pfx}_{c}"] = np.nan
        rows.append(feat)

    return pd.DataFrame(rows, index=df.index)


def build_ob_lookback_features(
    df: pd.DataFrame,
    event_col: str,
    prefix: str,
    n: int,
    top_col: str,
    bot_col: str,
) -> pd.DataFrame:
    events = []
    for i, row in df.iterrows():
        if row[event_col]:
            events.append({
                "pos":   df.index.get_loc(i),
                "close": row["close"],
                "top":   row[top_col],
                "bot":   row[bot_col],
            })

    is_bull = "bull" in prefix
    rows = []
    for i, (idx, row) in enumerate(df.iterrows()):
        cur_atr  = row["atr"] if (not np.isnan(row["atr"]) and row["atr"] > 0) else 1.0
        past_evs = [e for e in events if e["pos"] < i][::-1]
        feat     = {}
        for k in range(1, n + 1):
            pfx = f"{prefix}{k}"
            if k - 1 < len(past_evs):
                ev      = past_evs[k - 1]
                window  = df.iloc[ev["pos"] + 1: i + 1]
                filled  = False
                if not window.empty and ev["top"] is not None and not np.isnan(ev["top"]):
                    filled = (window["low"].min() <= ev["top"]) if is_bull else (window["high"].max() >= ev["bot"])
                feat[f"{pfx}_bars_ago"]  = i - ev["pos"]
                feat[f"{pfx}_raw_dist"]  = round(row["close"] - ev["close"], 5)
                feat[f"{pfx}_atr_dist"]  = round((row["close"] - ev["close"]) / cur_atr, 4)
                feat[f"{pfx}_is_filled"] = int(filled)
            else:
                feat[f"{pfx}_bars_ago"]  = np.nan
                feat[f"{pfx}_raw_dist"]  = np.nan
                feat[f"{pfx}_atr_dist"]  = np.nan
                feat[f"{pfx}_is_filled"] = 0
        rows.append(feat)

    return pd.DataFrame(rows, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
#  FULL PIPELINE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    df_1m: pd.DataFrame,
    df_htf: pd.DataFrame = None,
    cfg=None,
) -> pd.DataFrame:
    """
    Given raw 1-min OHLCV (and optional HTF frame), produce the full
    feature matrix as trained in the notebook.

    Parameters
    ----------
    df_1m   : columns [datetime, open, high, low, close, volume]
    df_htf  : optional 15-min frame for HTF EMA bias
    cfg     : config module (uses defaults if None)

    Returns
    -------
    dataset : pd.DataFrame  — one row per bar, all features, no target
    """
    if cfg is None:
        import config as cfg

    df = df_1m.copy().reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # ── HTF merge ─────────────────────────────────────────────────────────────
    if df_htf is not None and not df_htf.empty:
        htf = df_htf.copy()
        htf["htf_ema50"]  = htf["close"].ewm(span=50, adjust=False).mean()
        htf["htf_ema200"] = htf["close"].ewm(span=200, adjust=False).mean()
        htf["htf_bullish"] = (htf["htf_ema50"] > htf["htf_ema200"]).astype(int).shift(1)
        htf = htf[["datetime", "htf_ema50", "htf_ema200", "htf_bullish"]].sort_values("datetime")
        df  = pd.merge_asof(df.sort_values("datetime"), htf, on="datetime", direction="backward")
    else:
        df["htf_ema50"] = df["htf_ema200"] = df["htf_bullish"] = np.nan

    # ── Section 4.2: Base indicators ─────────────────────────────────────────
    df["atr"]       = calc_atr(df, cfg.ATR_LEN)
    df["atr_sma20"] = df["atr"].rolling(20).mean()
    df["vol_regime"] = df["atr"] / df["atr_sma20"].replace(0, np.nan)

    for p in [7, 14, 21]:
        df[f"rsi_{p}"] = calc_rsi(df["close"], p)
    df["rsi"] = df[f"rsi_{cfg.RSI_LEN}"]

    for span in [9, 20, 50, 200]:
        df[f"ema_{span}"] = df["close"].ewm(span=span, adjust=False).mean()
    df["ema_20_50_cross"]  = (df["ema_20"]  > df["ema_50"]).astype(int)
    df["ema_50_200_cross"] = (df["ema_50"] > df["ema_200"]).astype(int)

    df["macd"], df["macd_signal"], df["macd_hist"] = calc_macd(df["close"])
    df["macd_cross"] = (df["macd"] > df["macd_signal"]).astype(int)

    df["bb_upper"], df["bb_mid"], df["bb_lower"], df["bb_pct_b"], df["bb_bw"] = \
        calc_bollinger(df["close"])

    df["vwap"]      = calc_vwap(df)
    df["vwap_dist"] = (df["close"] - df["vwap"]) / df["atr"].replace(0, np.nan)

    df["vol_sma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma20"].replace(0, np.nan)
    df["vol_delta"] = df["volume"].diff()

    df["hour"]       = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["session"]    = df["hour"].apply(session_label)
    df["is_active_session"] = df["session"].isin([1, 2]).astype(int)

    # ── Section 4.3: Multi-MA relational features ─────────────────────────────
    ma_block = build_ma_features(df, ma_periods=cfg.MA_PERIODS)
    df = pd.concat([df, ma_block], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]

    # ── Section 5: Derivatives ────────────────────────────────────────────────
    for col in ["open", "high", "low", "close"]:
        df[f"{col}_v1"] = df[col].diff()
        df[f"{col}_v2"] = df[f"{col}_v1"].diff()
    df["hl_range"]     = df["high"]  - df["low"]
    df["co_spread"]    = df["close"] - df["open"]
    df["hl_range_v1"]  = df["hl_range"].diff()
    df["hl_range_v2"]  = df["hl_range_v1"].diff()
    df["co_spread_v1"] = df["co_spread"].diff()
    df["co_spread_v2"] = df["co_spread_v1"].diff()
    df["body_ratio"]   = df["co_spread"].abs() / df["hl_range"].replace(0, np.nan)
    df["upper_wick"]   = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"]   = df[["open", "close"]].min(axis=1) - df["low"]
    df["wick_ratio"]   = (df["upper_wick"] + df["lower_wick"]) / df["hl_range"].replace(0, np.nan)

    # ── Section 6: Market structure ───────────────────────────────────────────
    df["is_swing_high"], df["is_swing_low"] = fractal_pivots(
        df, cfg.LEFT_BARS, cfg.RIGHT_BARS, cfg.MIN_ATR_MULT
    )
    df = compute_swing_strength_reversal(df)
    df = compute_bos_choch(df)
    df = detect_order_blocks(df)

    # ── Section 7: Context enrichment ─────────────────────────────────────────
    df["range_high"] = df["high"].rolling(cfg.LOOKBACK_RANGE).max()
    df["range_low"]  = df["low"].rolling(cfg.LOOKBACK_RANGE).min()
    df["range_pct"]  = (df["close"] - df["range_low"]) / (
        (df["range_high"] - df["range_low"]).replace(0, np.nan)
    )
    df["in_premium"]     = (df["range_pct"] > 0.618).astype(int)
    df["in_discount"]    = (df["range_pct"] < 0.382).astype(int)
    df["in_equilibrium"] = ((df["range_pct"] >= 0.382) & (df["range_pct"] <= 0.618)).astype(int)

    df["last_sh_price"] = np.nan
    df["last_sl_price"] = np.nan
    _lsh = _lsl = np.nan
    for idx, row in df.iterrows():
        if row["is_swing_high"]: _lsh = row["high"]
        if row["is_swing_low"]:  _lsl = row["low"]
        df.at[idx, "last_sh_price"] = _lsh
        df.at[idx, "last_sl_price"] = _lsl

    fib_range = df["last_sh_price"] - df["last_sl_price"]
    for fib in cfg.FIB_LEVELS:
        fib_px = df["last_sl_price"] + fib * fib_range
        df[f"fib_{int(fib*1000)}_dist"] = (df["close"] - fib_px) / df["atr"].replace(0, np.nan)

    df["bull_confluence"] = (
        df["is_bullish_trend"].astype(int)
        + df["htf_bullish"].fillna(0).astype(int)
        + df["macd_cross"].astype(int)
        + df["ema_20_50_cross"].astype(int)
        + df["in_discount"].astype(int)
        + (df["rsi"] < 40).astype(int)
        + (df["vwap_dist"] < 0).astype(int)
    )
    df["bear_confluence"] = (
        (~df["is_bullish_trend"]).astype(int)
        + (1 - df["htf_bullish"].fillna(0).astype(int))
        + (1 - df["macd_cross"].astype(int))
        + (1 - df["ema_20_50_cross"].astype(int))
        + df["in_premium"].astype(int)
        + (df["rsi"] > 60).astype(int)
        + (df["vwap_dist"] > 0).astype(int)
    )

    # ── Section 8: Lookback blocks ────────────────────────────────────────────
    feat_sh        = build_lookback_features(df, "is_swing_high", "sh_", cfg.N, ["swing_strength", "reversal_prob"])
    feat_sl        = build_lookback_features(df, "is_swing_low",  "sl_", cfg.N, ["swing_strength", "reversal_prob"])
    feat_bos_bull  = build_lookback_features(df, "is_bos_bull",   "bos_bull_",  cfg.N)
    feat_bos_bear  = build_lookback_features(df, "is_bos_bear",   "bos_bear_",  cfg.N)
    feat_choch_bull= build_lookback_features(df, "is_choch_bull", "choch_bull_", cfg.N)
    feat_choch_bear= build_lookback_features(df, "is_choch_bear", "choch_bear_", cfg.N)
    feat_ob_bull   = build_ob_lookback_features(df, "is_ob_bull", "ob_bull_", cfg.N, "ob_bull_top", "ob_bull_bot")
    feat_ob_bear   = build_ob_lookback_features(df, "is_ob_bear", "ob_bear_", cfg.N, "ob_bear_top", "ob_bear_bot")

    dataset = pd.concat([
        df,
        feat_sh, feat_sl,
        feat_bos_bull, feat_bos_bear,
        feat_choch_bull, feat_choch_bear,
        feat_ob_bull, feat_ob_bear,
    ], axis=1)

    dataset = dataset.loc[:, ~dataset.columns.duplicated()]
    return dataset
