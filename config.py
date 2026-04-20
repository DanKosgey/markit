"""
config.py — Central configuration for the Market Structure XGBoost Trading Bot.
Edit ONLY this file to tune all runtime behaviour.
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "xgboost_base_model_20260419_180530.pkl"  # Saved XGBoost model (joblib pickle)
FEAT_COLS_PATH = BASE_DIR / "feature_columns.json"   # saved column order

# ─────────────────────────────────────────────────────────────────────────────
#  MT5 CONNECTION
# ─────────────────────────────────────────────────────────────────────────────
MT5_SYMBOL   = "Volatility 100 (1s) Index" # Symbol for Deriv Synthetic Index
MT5_TIMEFRAME = "M1"             # Primary timeframe (M1 = 1-minute)
MT5_HTF       = "M30"            # Higher timeframe for EMA bias
MT5_BARS      = 1500              # Bars to fetch for feature warm-up
MT5_HTF_BARS  = 1500              # HTF bars

# VISION WEBSOCKET
VISION_ENABLED = True
VISION_WS_URL = "wss://api.ruthwestlimited.com/vision_deriv/ws/signals"
VISION_TIMEFRAME = "M1"
VISION_TIMEFRAMES = [VISION_TIMEFRAME]  # Add more timeframes here, e.g. ["M1", "M5"]
VISION_RECONNECT_SECONDS = 5
VISION_SOCKET_TIMEOUT = 5
VISION_MAX_SIGNAL_AGE_SECONDS = 120
VISION_REQUIRE_FRESH_SIGNAL = True

# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING  (must match notebook exactly)
# ─────────────────────────────────────────────────────────────────────────────
N            = 3        # Lookback depth per structure event
LEFT_BARS    = 5        # Fractal left wing
RIGHT_BARS   = 2        # Fractal right wing (confirmation lag)
ATR_LEN      = 14
RSI_LEN      = 14
MIN_ATR_MULT = 0.3
LOOKBACK_RANGE = 100    # Rolling range for premium/discount
FIB_LEVELS   = [0.236, 0.382, 0.5, 0.618, 0.786]
MA_PERIODS   = [3, 5, 8, 13, 21, 34, 55]

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL / SIGNAL
# ─────────────────────────────────────────────────────────────────────────────
PROB_THRESHOLD  = 0.4   # Used only when SIGNAL_MODE="threshold"
SIGNAL_MODE     = "lead_class"  # "lead_class" or "threshold"
SIGNAL_COOLDOWN = 60    # Seconds to wait after a trade before signalling again

# ─────────────────────────────────────────────────────────────────────────────
#  TRADE EXECUTION  — SL/TP use ATR × multiplier
# ─────────────────────────────────────────────────────────────────────────────
TP_MULT        = 1.40     # Take-profit = ATR × TP_MULT
SL_MULT        = 0.75     # Stop-loss   = ATR × SL_MULT
LOT_SIZE       = 4.0    # Fixed lot size (adjust for your account)
MAGIC          = 20250418 # Magic number to identify bot orders
ORDER_COMMENT  = "XAUBOT_ML"
MAX_OPEN_TRADES = 1      # Maximum concurrent positions
SLIPPAGE_POINTS = 10     # Max allowed slippage in points

# ─────────────────────────────────────────────────────────────────────────────
#  TRAILING STOP (Risk Manager)
# ─────────────────────────────────────────────────────────────────────────────
USE_TRAILING_STOP = False
TRAIL_ACTIVATION = 1    # Activate trailing SL when profit reaches this many points
TRAIL_DISTANCE   = 0.5    # Keep SL this many points behind the current price

# ─────────────────────────────────────────────────────────────────────────────
#  RISK MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────
MAX_DAILY_LOSS_USD  = 2000.0   # Hard stop: stop trading after this daily loss
MAX_DRAWDOWN_PCT    = 0.20   # Max allowed drawdown as fraction of balance
TRADING_SESSIONS    = [(0, 24)]  # Widened to 24h for testing
SKIP_WEEKENDS       = False      # Disabled for testing

# ─────────────────────────────────────────────────────────────────────────────
#  WORKER TIMING
# ─────────────────────────────────────────────────────────────────────────────
DATA_POLL_SECONDS    = 10    # How often DataWorker polls MT5 for new bar
MONITOR_POLL_SECONDS = 5     # How often MonitorWorker checks open positions
QUEUE_TIMEOUT        = 30    # Seconds before a queue.get() times out

# ─────────────────────────────────────────────────────────────────────────────
#  CATEGORICAL COLUMNS  (must match training)
# ─────────────────────────────────────────────────────────────────────────────
CATEGORICAL_COLS = [
    'is_swing_high', 'is_swing_low', 'is_bos_bull', 'is_bos_bear',
    'is_choch_bull', 'is_choch_bear', 'is_ob_bull', 'is_ob_bear',
    'is_active_session', 'in_premium', 'in_discount', 'in_equilibrium',
    'is_bullish_trend', 'macd_cross', 'ema_20_50_cross', 'ema_50_200_cross',
    'htf_bullish', 'session', 'day_of_week',
]

NON_FEATURES = [
    'datetime', 'smart_target', 'log_ret',
    'ob_bull_top', 'ob_bull_bot', 'ob_bear_top', 'ob_bear_bot',
    'ob_bull_filled', 'ob_bear_filled',
    'range_high', 'range_low',
    'mfe_mae_ratio',
    '_date', '_tp', '_tpv', 'hour',
]
