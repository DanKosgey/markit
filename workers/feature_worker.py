"""
workers/feature_worker.py - Feature Engineering Worker.

Consumes raw OHLCV DataFrames from raw_bar_q, runs the full feature
pipeline, and pushes the final feature row onto feature_q.
"""

import json
import threading

import numpy as np
import pandas as pd

import config as cfg
from engine.features import build_feature_matrix
from utils.state import feature_q, log, raw_bar_q, stop_event


class FeatureWorker(threading.Thread):
    """
    Pulls from raw_bar_q, builds the full feature matrix, and pushes the
    latest aligned row to feature_q.
    """

    def __init__(self):
        super().__init__(name="FeatureWorker", daemon=True)
        self._feature_cols = self._load_feature_cols()

    @staticmethod
    def _load_feature_cols() -> list:
        if cfg.FEAT_COLS_PATH.exists():
            with open(cfg.FEAT_COLS_PATH) as feature_file:
                cols = json.load(feature_file)
            log.info(f"[FeatureWorker] Loaded {len(cols)} feature columns from {cfg.FEAT_COLS_PATH}")
            return cols

        log.warning(
            "[FeatureWorker] feature_columns.json not found. "
            "Column alignment will be inferred from data - ensure this matches training."
        )
        return []

    def _prepare_row(self, dataset: pd.DataFrame):
        non_features = set(cfg.NON_FEATURES)

        if self._feature_cols:
            final_cols = self._feature_cols
            available = set(dataset.columns)
            row = pd.DataFrame(index=[dataset.index[-1]], columns=final_cols)
            for col in final_cols:
                row[col] = dataset[col].iloc[-1] if col in available else np.nan
        else:
            final_cols = [col for col in dataset.columns if col not in non_features]
            row = dataset[final_cols].tail(1).copy()

        for col in row.columns:
            if col in cfg.CATEGORICAL_COLS:
                row[col] = row[col].fillna(-1).astype(int).astype("category")
                continue

            if "bars_ago" in col:
                row[col] = row[col].fillna(9999)
            else:
                row[col] = row[col].fillna(0)

            if row[col].dtype == bool:
                row[col] = row[col].astype(int)
            row[col] = pd.to_numeric(row[col], errors="coerce").astype("float32")

        meta = dataset[["datetime", "close", "high", "low", "atr", "rsi"]].tail(1).copy()
        return row, meta

    def run(self):
        log.info("[FeatureWorker] Starting.")

        while not stop_event.is_set():
            try:
                payload = raw_bar_q.get(timeout=cfg.QUEUE_TIMEOUT)
            except Exception:
                continue

            try:
                df_1m = payload["df_1m"]
                df_htf = payload.get("df_htf")

                dataset = build_feature_matrix(df_1m, df_htf, cfg)

                nan_cols = dataset.columns[dataset.isna().all()].tolist()
                constant_cols = [col for col in dataset.columns if dataset[col].nunique() <= 1]
                nan_counts = dataset.isna().sum().sum()
                total_cells = dataset.size
                nan_pct = (nan_counts / total_cells) * 100 if total_cells > 0 else 0

                sh_count = dataset["is_swing_high"].sum()
                sl_count = dataset["is_swing_low"].sum()
                last_price = dataset["close"].iloc[-1]
                last_atr = dataset["atr"].iloc[-1]

                if nan_cols:
                    log.warning(
                        f"[FeatureWorker] DATA QUALITY | {len(nan_cols)} columns are 100% NaN: {nan_cols[:5]}..."
                    )

                log.info(
                    f"[FeatureWorker] Artifact Check | NaNs: {nan_pct:.1f}% | "
                    f"Cst: {len(constant_cols)} | Swings: H={sh_count}/L={sl_count} | "
                    f"Price: {last_price:.2f} | ATR: {last_atr:.4f}"
                )

                feat_row, meta_row = self._prepare_row(dataset)

                try:
                    feature_q.put_nowait({"features": feat_row, "meta": meta_row})
                    log.info(f"[FeatureWorker] Ready | {feat_row.shape[1]} features aligned")
                except Exception:
                    log.warning("[FeatureWorker] feature_q full - dropping row.")
            except Exception as exc:
                log.exception(f"[FeatureWorker] Error during feature build: {exc}")

        log.info("[FeatureWorker] Stopped.")
