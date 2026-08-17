"""
Train a Random Forest to predict GEDI-derived biomass (agbd, Mg/ha) from
GEDI relative-height (rh) bands plus Sentinel-2 features, and validate
with spatial block cross-validation.

This is the "trained model" approach (see CLAUDE.md: ~±6% on site total at
CEPT, but needs field plots at every new site — here GEDI shots stand in
for field plots, per the GEDI-as-training-data strategy in
gee/gedi_s2_training_extract.js). This script only trains and validates;
it does not apply the model anywhere.

HEIGHT ADDED: the first version of this model (optical-only) validated
against CEPT's field census with a decent site-total match (+14%) but a
NEGATIVE per-cell Spearman correlation (-0.28) — optical features alone
can't distinguish tall-and-green (trees) from flat-and-green (lawns/
crops). gee/gedi_s2_training_extract.js now joins GEDI L2A relative-
height bands (rh50, rh75, rh90, rh98) onto each training shot; rh98
alone correlates with agbd at Spearman +0.97 region-wide (vs. +0.43 for
NDVI), so height is expected to dominate the feature importances below.

RF_PARAMS is now capped (n_estimators=200, max_depth=20): the earlier
uncapped 500-tree model produced a 2.66 GB .joblib, far past what git can
hold. Capping trades a small amount of accuracy for a file small enough
to version-control.

CLAUDE.md rules followed:
  - Rule 1 (raw kg/Mg, never log): agbd is used as-is, in raw Mg/ha. No
    log transform anywhere in this script. Predicting the median (what a
    log-target model does) under-estimates a right-skewed SUM by ~48% —
    the predicted/actual SUM ratio reported below is exactly the check
    that would catch that mistake.
  - Rule 2 (spatial block CV, not random k-fold): shots from adjacent
    pixels share Sentinel-2 pixels and nearby canopy, so a random split
    leaks and reports falsely high accuracy. Shots are grouped into
    ~0.25 deg blocks and whole blocks are held out together, repeated
    with shifted block-grid origins for an honest uncertainty band
    (not just one arbitrary grid alignment).
  - Rule 5 (site totals, not per-cell accuracy, are what's reliable):
    the predicted-sum / actual-sum ratio per held-out fold is reported
    alongside R²/RMSE because that ratio is the number this project
    actually trusts, not per-shot accuracy.

No-leakage notes:
  - agbd_se (GEDI's own uncertainty estimate for agbd) is NOT a feature.
    It's derived from agbd itself and wouldn't exist at inference time on
    a new site without GEDI coverage — including it would leak target
    information through a back door.
  - lat/lon are NOT features, only used to build spatial CV blocks. A
    model that uses raw coordinates can memorize location-specific
    biomass instead of learning the spectral relationship, which is
    exactly what spatial block CV is meant to catch.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =============================================================================
# CONFIG
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DATA_PATH = REPO_ROOT / "data" / "processed" / "gedi_s2_training_WITHHEIGHT.csv"
MODEL_OUTPUT_PATH = REPO_ROOT / "outputs" / "biomass_rf_model.joblib"
METRICS_OUTPUT_PATH = REPO_ROOT / "outputs" / "biomass_rf_spatial_cv_metrics.json"

TARGET_COLUMN = "agbd"  # raw Mg/ha -- CLAUDE.md rule 1, never log-transformed
FEATURE_COLUMNS = [
    "rh50", "rh75", "rh90", "rh98",
    "ndvi_postmonsoon", "evi_postmonsoon", "ndvi_dryseason", "evi_dryseason",
    "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12",
]
# Excluded from features: agbd_se (leaks target-derived info, unavailable
# without GEDI at inference time), lat/lon (used only to build spatial CV
# blocks below, not as model inputs -- see no-leakage notes above).

# Spatial block cross-validation (CLAUDE.md rule 2). Unchanged from the
# optical-only run, so the two are directly comparable.
BLOCK_SIZE_DEG = 0.25
N_SPATIAL_FOLDS = 5
N_SHIFT_ORIGINS = 4  # repeats with shifted block-grid origin, for an honest uncertainty band
RANDOM_SEED = 42

# n_estimators/max_depth capped -- the earlier uncapped 500-tree model
# produced a 2.66 GB .joblib, far past GitHub's 100 MB push limit. See
# module docstring.
RF_PARAMS = dict(
    n_estimators=200,
    max_depth=20,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)


# =============================================================================
# Spatial blocking
# =============================================================================

def assign_blocks(lat, lon, lat_min, lon_min, offset_deg):
    """Bin (lat, lon) into BLOCK_SIZE_DEG blocks, with the grid origin
    shifted by offset_deg in both directions -- shifting the origin means
    the SAME shots fall into different block groupings across shift
    iterations, so no single arbitrary grid alignment can flatter (or
    unfairly penalize) the reported metrics.
    """
    block_row = np.floor((lat - lat_min + offset_deg) / BLOCK_SIZE_DEG).astype(int)
    block_col = np.floor((lon - lon_min + offset_deg) / BLOCK_SIZE_DEG).astype(int)
    return block_row.astype(str) + "_" + block_col.astype(str)


def spatial_cv_folds(block_ids, n_folds, rng):
    """Assign each unique block to one of n_folds groups (shuffled), and
    yield (train_mask, test_mask) boolean arrays -- whole blocks held out
    together, never split across train/test.
    """
    unique_blocks = block_ids.unique()
    rng.shuffle(unique_blocks)
    block_to_fold = {b: i % n_folds for i, b in enumerate(unique_blocks)}
    fold_of_row = block_ids.map(block_to_fold).to_numpy()
    for fold in range(n_folds):
        test_mask = fold_of_row == fold
        yield ~test_mask, test_mask


# =============================================================================
# Run spatial block CV, repeated with shifted origins
# =============================================================================

def run_spatial_cv(df):
    lat_min, lon_min = df["lat"].min(), df["lon"].min()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    fold_results = []
    for shift_i in range(N_SHIFT_ORIGINS):
        offset_deg = (shift_i / N_SHIFT_ORIGINS) * BLOCK_SIZE_DEG
        block_ids = assign_blocks(df["lat"].to_numpy(), df["lon"].to_numpy(), lat_min, lon_min, offset_deg)
        block_ids = pd.Series(block_ids, index=df.index)

        rng = np.random.default_rng(RANDOM_SEED + shift_i)
        for fold_i, (train_mask, test_mask) in enumerate(spatial_cv_folds(block_ids, N_SPATIAL_FOLDS, rng)):
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            model = RandomForestRegressor(**RF_PARAMS)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred) ** 0.5
            sum_ratio = y_pred.sum() / y_test.sum()

            fold_results.append({
                "shift_origin": shift_i,
                "fold": fold_i,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "n_blocks_test": int(block_ids[test_mask].nunique()),
                "r2": r2,
                "rmse": rmse,
                "pred_sum_over_actual_sum": sum_ratio,
            })

    return pd.DataFrame(fold_results)


def summarize_cv(cv_results):
    summary = {}
    for metric in ["r2", "rmse", "pred_sum_over_actual_sum"]:
        summary[metric] = {
            "mean": float(cv_results[metric].mean()),
            "std": float(cv_results[metric].std()),
            "min": float(cv_results[metric].min()),
            "max": float(cv_results[metric].max()),
        }
    return summary


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"Loading training data from {TRAINING_DATA_PATH}")
    df = pd.read_csv(TRAINING_DATA_PATH)
    print(f"Rows: {len(df)}, features: {len(FEATURE_COLUMNS)}, target: {TARGET_COLUMN} (raw Mg/ha)")

    print(f"\nRunning spatial block CV: {N_SPATIAL_FOLDS} folds x {N_SHIFT_ORIGINS} shifted "
          f"origins = {N_SPATIAL_FOLDS * N_SHIFT_ORIGINS} fold-runs, {BLOCK_SIZE_DEG} deg blocks")
    cv_results = run_spatial_cv(df)

    print("\n--- Per fold-run results ---")
    print(cv_results.to_string(index=False))

    summary = summarize_cv(cv_results)
    print("\n--- Spatial CV summary (mean +/- std across all fold-runs) ---")
    for metric, stats in summary.items():
        print(f"{metric}: {stats['mean']:.4f} +/- {stats['std']:.4f} "
              f"(min {stats['min']:.4f}, max {stats['max']:.4f})")

    print("\nNote: pred_sum_over_actual_sum is the ratio this project actually trusts "
          "(CLAUDE.md rule 5) -- 1.0 means the model's predicted total matches the "
          "held-out actual total exactly; R²/RMSE are per-shot diagnostics on top of that.")

    # -------------------------------------------------------------------
    # Final model: trained on ALL data (CV above is for validation only,
    # these fold models are discarded).
    # -------------------------------------------------------------------
    print("\nTraining final model on the full training set...")
    final_model = RandomForestRegressor(**RF_PARAMS)
    final_model.fit(df[FEATURE_COLUMNS], df[TARGET_COLUMN])

    importances = pd.Series(final_model.feature_importances_, index=FEATURE_COLUMNS)
    importances = importances.sort_values(ascending=False)
    print("\n--- Feature importances (final model, trained on all data) ---")
    print(importances.to_string())

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "feature_columns": FEATURE_COLUMNS}, MODEL_OUTPUT_PATH)
    model_size_mb = MODEL_OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\nSaved trained model to {MODEL_OUTPUT_PATH} ({model_size_mb:.1f} MB)")
    if model_size_mb > 100:
        print(f"WARNING: {model_size_mb:.1f} MB is still over GitHub's 100 MB push limit -- "
              f"stays gitignored (outputs/*.joblib), not committed.")

    metrics_out = {
        "config": {
            "block_size_deg": BLOCK_SIZE_DEG,
            "n_spatial_folds": N_SPATIAL_FOLDS,
            "n_shift_origins": N_SHIFT_ORIGINS,
            "random_seed": RANDOM_SEED,
            "rf_params": RF_PARAMS,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
        },
        "spatial_cv_summary": summary,
        "spatial_cv_fold_results": cv_results.to_dict(orient="records"),
        "feature_importances": importances.to_dict(),
    }
    with open(METRICS_OUTPUT_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Saved spatial CV metrics + feature importances to {METRICS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
