"""
save_model.py — Run this ONCE from inside the notebook environment
               (or after executing the notebook) to export the trained
               model and feature column list to disk.

Usage (at the bottom of the notebook or in a separate cell):
    %run save_model.py

Or manually in a notebook cell:
    exec(open("save_model.py").read())
"""

import json
from pathlib import Path

# ── Assumes these variables exist in the current kernel ──────────────────────
#   model_ft    : xgb.XGBClassifier  (fine-tuned model)
#   X_train     : pd.DataFrame        (training features, correct column order)
# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR = Path(".")

# 1. Save model
model_path = OUT_DIR / "model_ft.json"
model_ft.save_model(str(model_path))
print(f"✓  Model saved  →  {model_path}")

# 2. Save feature column list (preserves training order)
cols_path = OUT_DIR / "feature_columns.json"
feature_columns = list(X_train.columns)
with open(cols_path, "w") as f:
    json.dump(feature_columns, f, indent=2)
print(f"✓  Feature columns saved  →  {cols_path}  ({len(feature_columns)} columns)")

print("\nBoth files should be placed in the same directory as app.py.")
