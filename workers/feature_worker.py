"""
workers/feature_worker.py — Feature Engineering Worker.

Consumes raw OHLCV DataFrames from raw_bar_q, runs the complete
feature pipeline (mirroring the notebook), and pushes the final
feature row (last bar only) onto feature_q.
"""

import json
import threading

import numpy as np
import pandas as pd

import config as cfg
from engine.features import build_feature_matrix
from utils.state import log, stop_event, raw_bar_q, feature_q


class FeatureWorker(threading.Thread):
    """
    Pulls from raw_bar_q → builds full feature matrix → pushes last row to feature_q.
    """

    def __init__(self):
        super().__init__(name="FeatureWorker", daemon=True)
        self._feature_cols = self._load_feature_cols()

    # ── internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _load_feature_cols() -> list:
        """Load the saved column order used during training."""
        if cfg.FEAT_COLS_PATH.exists():
            with open(cfg.FEAT_COLS_PATH) as f:
                cols = json.load(f)
            log.info(f"[FeatureWorker] Loaded {len(cols)} feature columns from {cfg.FEAT_COLS_PATH}")
            return cols
        log.warning(
            "[FeatureWorker] feature_columns.json not found. "
            "Column alignment will be inferred from data — ensure this matches training."
        )
        return []

    def _prepare_row(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and align the last row of the feature matrix so it matches
        the column order and dtypes used during model training.
        """
        non_feats = set(cfg.NON_FEATURES)
        
        # -- 1. Ensure all expected columns are present (Alignment) --
        if self._feature_cols:
            # Use training column order
            final_cols = self._feature_cols
            available = set(dataset.columns)
            
            # Create a row with all expected columns, filling missing with NaN
            row = pd.DataFrame(index=[dataset.index[-1]], columns=final_cols)
            for col in final_cols:
                if col in available:
                    row[col] = dataset[col].iloc[-1]
                else:
                    # Column completely missing from current feature build
                    row[col] = np.nan
        else:
            # Fallback to current columns if no json found
            cols = [c for c in dataset.columns if c not in non_feats]
            row = dataset[cols].tail(1).copy()
            final_cols = cols

        # -- 2. Smarter Null Handling --
        # We don't want to fill 'bars_ago' with 0 (which means 'now')
        # We'll use -1 for categoricals and 0 for others, but handle NaNs carefully
        for col in row.columns:
            if col in cfg.CATEGORICAL_COLS:
                # Categorical: Fill NaN with -1
                row[col] = row[col].fillna(-1).astype(int).astype("category")
            else:
                # Numeric: Check if it's a distance feature
                if "bars_ago" in col:
                    row[col] = row[col].fillna(9999) # Large distance for missing
                else:
                    row[col] = row[col].fillna(0)
                
                # Ensure float32 for XGBoost
                if row[col].dtype == bool:
                    row[col] = row[col].astype(int)
                row[col] = pd.to_numeric(row[col], errors="coerce").astype("float32")

        # -- 3. Metadata for logging --
        meta = dataset[["datetime", "close", "high", "low", "atr", "rsi"]].tail(1).copy()
        return row, meta

    # ── public ────────────────────────────────────────────────────────────────

    def run(self):
        log.info("[FeatureWorker] Starting…")

        while not stop_event.is_set():
            try:
                payload = raw_bar_q.get(timeout=cfg.QUEUE_TIMEOUT)
            except Exception:
                continue  # timeout or empty — loop back

            try:
                df_1m  = payload["df_1m"]
                df_htf = payload.get("df_htf")

                log.debug(
                    f"[FeatureWorker] Processing {len(df_1m)} bars "
                    f"(ts={df_1m['datetime'].iloc[-1]})"
                )

                dataset = build_feature_matrix(df_1m, df_htf, cfg)
                
                # -- Data Quality Check --
                nan_cols = dataset.columns[dataset.isna().all()].tolist()
                constant_cols = [c for c in dataset.columns if dataset[c].nunique() <= 1]
                
                nan_counts = dataset.isna().sum().sum()
                total_cells = dataset.size
                nan_pct = (nan_counts / total_cells) * 100 if total_cells > 0 else 0
                
                sh_count = dataset['is_swing_high'].sum()
                sl_count = dataset['is_swing_low'].sum()
                last_price = dataset['close'].iloc[-1]
                last_atr = dataset['atr'].iloc[-1]
                
                if nan_cols:
                    log.warning(f"[FeatureWorker] DATA QUALITY | {len(nan_cols)} columns are 100% NaN: {nan_cols[:5]}...")
                
                log.info(
                    f"[FeatureWorker] Artifact Check | NaNs: {nan_pct:.1f}% | "
                    f"Cst: {len(constant_cols)} | Swings: H={sh_count}/L={sl_count} | "
                    f"Price: {last_price:.2f} | ATR: {last_atr:.4f}"
                )

                feat_row, meta_row = self._prepare_row(dataset)

                try:
                    feature_q.put_nowait({"features": feat_row, "meta": meta_row})
                    log.info(f"[FeatureWorker] ✓ Ready | {feat_row.shape[1]} features aligned")
                except Exception:
                    log.warning("[FeatureWorker] feature_q full — dropping row.")

            except Exception as exc:
                log.exception(f"[FeatureWorker] Error during feature build: {exc}")

        log.info("[FeatureWorker] Stopped.")
