"""
Fit a regional GEDI height -> biomass calibration curve.

STRATEGIC PIVOT: the previous model used GEDI's own rh98 as a PREDICTION
feature, which is exactly backwards -- GEDI doesn't cover prediction sites,
that's the whole problem this project solves. New architecture:
  1. GEDI calibrates a regional height -> biomass curve ONCE, here, from
     GEDI shots that already carry both quantities.
  2. At any real site, canopy STRUCTURE (height) comes from a published
     canopy-height map (mature sites) or a drone flight (young sites) --
     never from GEDI. That external height gets looked up against this
     curve to estimate biomass. GEDI never touches a prediction site.

This script is height -> biomass ONLY. No Sentinel-2 features: the whole
point of this pivot is that canopy structure (height) is what carries the
biomass signal (rh98 alone explained 93% of feature importance in the
earlier combined optical+height model -- optical was doing cleanup on a
residual, not carrying the signal), and structure will come from a height
product, not from optical imagery, at prediction time.

Two candidate curves, compared:
  A. Power law: agbd = a * rh98^b (standard allometric form, rh98 only --
     matches what an external CHM/drone product can actually supply: a
     single top-of-canopy height per pixel, not a full vertical profile).
  B. Random Forest on all four rh bands (rh50/rh75/rh90/rh98) -- a ceiling
     check on how much the extra profile bands would help IF an external
     source could supply them. Most published CHMs and drone
     photogrammetry only give top height, so B may not be deployable in
     practice even if it scores better here -- that's a judgment call for
     a later step, not decided in this script.

CLAUDE.md rules followed:
  - Rule 1 (raw kg/Mg, never log): the power law is fit by NONLINEAR least
    squares directly on raw (rh98, agbd) pairs (scipy.optimize.curve_fit),
    NOT by log-log linearization. Log-log OLS effectively fits the median/
    geometric mean of agbd given height, which under-predicts the raw
    arithmetic-mean SUM for a right-skewed target -- precisely the ~48%
    under-estimation failure this rule exists to prevent. Fitting directly
    in raw space avoids that bias entirely.
  - Rule 2 (spatial block CV): both curves are validated with the same
    0.25 deg block scheme as src/train_biomass_model.py (imported from
    there, not reimplemented) -- honest accuracy for "how well does
    height alone predict biomass", which is what both prediction tiers
    rely on. The height-lookup uncertainty band (see block_bootstrap_bands
    below) resamples whole spatial BLOCKS, not individual shots, for the
    same reason plain k-fold CV is rejected: neighboring shots aren't
    independent, so resampling points would understate the true
    uncertainty.
  - Rule 5 (site totals over per-cell accuracy): predicted-sum/actual-sum
    ratio reported per fold alongside R^2/RMSE, same as the biomass model.

No-leakage notes: agbd_se and lat/lon excluded from features, same
reasoning as src/train_biomass_model.py (agbd_se is derived from the
target; lat/lon are for spatial CV blocking only, not model inputs).
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_biomass_model import (  # noqa: E402
    assign_blocks, spatial_cv_folds, summarize_cv,
    BLOCK_SIZE_DEG, N_SPATIAL_FOLDS, N_SHIFT_ORIGINS, RANDOM_SEED,
)

# =============================================================================
# CONFIG
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DATA_PATH = REPO_ROOT / "data" / "processed" / "gedi_s2_training_WITHHEIGHT.csv"
POWER_LAW_OUTPUT_PATH = REPO_ROOT / "outputs" / "height_biomass_power_law.json"
RF_MODEL_OUTPUT_PATH = REPO_ROOT / "outputs" / "height_biomass_rf_model.joblib"
METRICS_OUTPUT_PATH = REPO_ROOT / "outputs" / "height_biomass_curve_metrics.json"

TARGET_COLUMN = "agbd"  # raw Mg/ha -- CLAUDE.md rule 1, never log-transformed
HEIGHT_COLUMN = "rh98"  # power-law input -- top-of-canopy, what external CHM/drone products supply
RF_FEATURE_COLUMNS = ["rh50", "rh75", "rh90", "rh98"]

# Same spatial CV scheme as src/train_biomass_model.py (imported above),
# so results are directly comparable.

RF_PARAMS = dict(n_estimators=200, max_depth=20, random_state=RANDOM_SEED, n_jobs=-1)

LOOKUP_HEIGHTS_M = [10, 20, 30]
N_BOOTSTRAP = 200  # block-bootstrap resamples for the power-law prediction band


# =============================================================================
# Power law: agbd = a * rh98^b, fit by nonlinear least squares in RAW space
# =============================================================================

def power_law(x, a, b):
    return a * np.power(x, b)


def fit_power_law(height, biomass):
    # bounds keep a, b positive -- biomass should rise monotonically with
    # height, not fall or go negative. method='trf' supports bounds
    # (curve_fit's default 'lm' does not).
    popt, _ = curve_fit(
        power_law, height, biomass,
        p0=[1.0, 1.0], bounds=(0, [np.inf, np.inf]),
        method="trf", maxfev=10000,
    )
    return popt[0], popt[1]  # a, b


# =============================================================================
# Spatial block CV for both candidate curves
# =============================================================================

def run_power_law_cv(df):
    height_all = df[HEIGHT_COLUMN].to_numpy()
    biomass_all = df[TARGET_COLUMN].to_numpy()
    lat_min, lon_min = df["lat"].min(), df["lon"].min()

    fold_results = []
    for shift_i in range(N_SHIFT_ORIGINS):
        offset_deg = (shift_i / N_SHIFT_ORIGINS) * BLOCK_SIZE_DEG
        block_ids = assign_blocks(df["lat"].to_numpy(), df["lon"].to_numpy(), lat_min, lon_min, offset_deg)
        block_ids = pd.Series(block_ids, index=df.index)

        rng = np.random.default_rng(RANDOM_SEED + shift_i)
        for fold_i, (train_mask, test_mask) in enumerate(spatial_cv_folds(block_ids, N_SPATIAL_FOLDS, rng)):
            h_train, h_test = height_all[train_mask], height_all[test_mask]
            y_train, y_test = biomass_all[train_mask], biomass_all[test_mask]

            a, b = fit_power_law(h_train, y_train)
            y_pred = power_law(h_test, a, b)

            fold_results.append({
                "shift_origin": shift_i,
                "fold": fold_i,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "a": a,
                "b": b,
                "r2": r2_score(y_test, y_pred),
                "rmse": mean_squared_error(y_test, y_pred) ** 0.5,
                "pred_sum_over_actual_sum": y_pred.sum() / y_test.sum(),
            })

    return pd.DataFrame(fold_results)


def run_rf_cv(df):
    X_all = df[RF_FEATURE_COLUMNS]
    y_all = df[TARGET_COLUMN]
    lat_min, lon_min = df["lat"].min(), df["lon"].min()

    fold_results = []
    for shift_i in range(N_SHIFT_ORIGINS):
        offset_deg = (shift_i / N_SHIFT_ORIGINS) * BLOCK_SIZE_DEG
        block_ids = assign_blocks(df["lat"].to_numpy(), df["lon"].to_numpy(), lat_min, lon_min, offset_deg)
        block_ids = pd.Series(block_ids, index=df.index)

        rng = np.random.default_rng(RANDOM_SEED + shift_i)
        for fold_i, (train_mask, test_mask) in enumerate(spatial_cv_folds(block_ids, N_SPATIAL_FOLDS, rng)):
            X_train, X_test = X_all[train_mask], X_all[test_mask]
            y_train, y_test = y_all[train_mask], y_all[test_mask]

            model = RandomForestRegressor(**RF_PARAMS)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            fold_results.append({
                "shift_origin": shift_i,
                "fold": fold_i,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "r2": r2_score(y_test, y_pred),
                "rmse": mean_squared_error(y_test, y_pred) ** 0.5,
                "pred_sum_over_actual_sum": y_pred.sum() / y_test.sum(),
            })

    return pd.DataFrame(fold_results)


# =============================================================================
# Height -> biomass lookup table with a block-bootstrap uncertainty band
# =============================================================================

def block_bootstrap_bands(df, heights_m, n_bootstrap):
    """Refit the power law on n_bootstrap resamples of whole spatial BLOCKS
    (not individual shots -- see module docstring rule 2 note) and report
    the spread of predicted biomass at each height in heights_m.
    """
    lat_min, lon_min = df["lat"].min(), df["lon"].min()
    block_ids = assign_blocks(df["lat"].to_numpy(), df["lon"].to_numpy(), lat_min, lon_min, 0.0)
    height = df[HEIGHT_COLUMN].to_numpy()
    biomass = df[TARGET_COLUMN].to_numpy()

    block_to_indices = {}
    for idx, b in enumerate(block_ids):
        block_to_indices.setdefault(b, []).append(idx)
    block_to_indices = {b: np.array(idxs) for b, idxs in block_to_indices.items()}
    unique_blocks = np.array(list(block_to_indices.keys()))

    rng = np.random.default_rng(RANDOM_SEED)
    height_grid = np.array(heights_m, dtype=float)
    predicted = np.empty((n_bootstrap, len(height_grid)))

    for i in range(n_bootstrap):
        sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        idx = np.concatenate([block_to_indices[b] for b in sampled_blocks])
        a, b = fit_power_law(height[idx], biomass[idx])
        predicted[i, :] = power_law(height_grid, a, b)

    bands = {}
    for j, h in enumerate(heights_m):
        vals = predicted[:, j]
        bands[h] = {
            "median_Mg_ha": float(np.median(vals)),
            "mean_Mg_ha": float(np.mean(vals)),
            "p2_5_Mg_ha": float(np.percentile(vals, 2.5)),
            "p97_5_Mg_ha": float(np.percentile(vals, 97.5)),
        }
    return bands


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"Loading training data from {TRAINING_DATA_PATH}")
    df = pd.read_csv(TRAINING_DATA_PATH)
    print(f"Rows: {len(df)}, height-only target: {TARGET_COLUMN} (raw Mg/ha) ~ {HEIGHT_COLUMN} (m)")

    # -------------------------------------------------------------------
    # A. Power law spatial CV
    # -------------------------------------------------------------------
    print(f"\n=== A. Power law (agbd = a * {HEIGHT_COLUMN}^b) ===")
    print(f"Running spatial block CV: {N_SPATIAL_FOLDS} folds x {N_SHIFT_ORIGINS} shifted "
          f"origins = {N_SPATIAL_FOLDS * N_SHIFT_ORIGINS} fold-runs, {BLOCK_SIZE_DEG} deg blocks")
    power_law_cv = run_power_law_cv(df)
    print(power_law_cv.to_string(index=False))

    power_law_summary = summarize_cv(power_law_cv)
    print("\n--- Power law spatial CV summary (mean +/- std) ---")
    for metric, stats in power_law_summary.items():
        print(f"{metric}: {stats['mean']:.4f} +/- {stats['std']:.4f} "
              f"(min {stats['min']:.4f}, max {stats['max']:.4f})")
    print(f"a (per fold): {power_law_cv['a'].mean():.6f} +/- {power_law_cv['a'].std():.6f}")
    print(f"b (per fold): {power_law_cv['b'].mean():.6f} +/- {power_law_cv['b'].std():.6f}")

    # -------------------------------------------------------------------
    # B. Random Forest (all 4 rh bands) spatial CV
    # -------------------------------------------------------------------
    print(f"\n=== B. Random Forest ({', '.join(RF_FEATURE_COLUMNS)}) ===")
    print(f"Running spatial block CV: {N_SPATIAL_FOLDS} folds x {N_SHIFT_ORIGINS} shifted "
          f"origins = {N_SPATIAL_FOLDS * N_SHIFT_ORIGINS} fold-runs, {BLOCK_SIZE_DEG} deg blocks")
    rf_cv = run_rf_cv(df)
    print(rf_cv.to_string(index=False))

    rf_summary = summarize_cv(rf_cv)
    print("\n--- Random Forest spatial CV summary (mean +/- std) ---")
    for metric, stats in rf_summary.items():
        print(f"{metric}: {stats['mean']:.4f} +/- {stats['std']:.4f} "
              f"(min {stats['min']:.4f}, max {stats['max']:.4f})")

    # -------------------------------------------------------------------
    # Final fits on ALL data (CV above is for honest validation only;
    # these are the deployed artifacts).
    # -------------------------------------------------------------------
    print("\n=== Final fits on full data ===")
    a_final, b_final = fit_power_law(df[HEIGHT_COLUMN].to_numpy(), df[TARGET_COLUMN].to_numpy())
    print(f"Power law: agbd = {a_final:.6f} * {HEIGHT_COLUMN}^{b_final:.6f}")

    rf_final = RandomForestRegressor(**RF_PARAMS)
    rf_final.fit(df[RF_FEATURE_COLUMNS], df[TARGET_COLUMN])
    rf_importances = pd.Series(rf_final.feature_importances_, index=RF_FEATURE_COLUMNS).sort_values(ascending=False)
    print("\nRandom Forest feature importances (final model, all data):")
    print(rf_importances.to_string())

    # -------------------------------------------------------------------
    # Height -> biomass lookup table with block-bootstrap band
    # -------------------------------------------------------------------
    print(f"\n=== Height -> biomass lookup (power law, {N_BOOTSTRAP}x block bootstrap 95% band) ===")
    bands = block_bootstrap_bands(df, LOOKUP_HEIGHTS_M, N_BOOTSTRAP)
    for h in LOOKUP_HEIGHTS_M:
        point_estimate = power_law(h, a_final, b_final)
        band = bands[h]
        print(f"{HEIGHT_COLUMN} = {h:>2} m -> {point_estimate:7.2f} Mg/ha "
              f"(bootstrap median {band['median_Mg_ha']:.2f}, "
              f"95% band [{band['p2_5_Mg_ha']:.2f}, {band['p97_5_Mg_ha']:.2f}])")

    # -------------------------------------------------------------------
    # Save artifacts
    # -------------------------------------------------------------------
    POWER_LAW_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    power_law_out = {
        "form": "agbd = a * rh98^b",
        "fit_method": "nonlinear least squares in raw space (scipy.optimize.curve_fit, method=trf) "
                       "-- NOT log-log linearization, see module docstring CLAUDE.md rule 1 note",
        "target_column": TARGET_COLUMN,
        "height_column": HEIGHT_COLUMN,
        "a": a_final,
        "b": b_final,
        "spatial_cv_summary": power_law_summary,
        "lookup_table": {
            str(h): {
                "point_estimate_Mg_ha": float(power_law(h, a_final, b_final)),
                **bands[h],
            }
            for h in LOOKUP_HEIGHTS_M
        },
        "n_bootstrap": N_BOOTSTRAP,
    }
    with open(POWER_LAW_OUTPUT_PATH, "w") as f:
        json.dump(power_law_out, f, indent=2)
    print(f"\nSaved power-law curve to {POWER_LAW_OUTPUT_PATH}")

    joblib.dump({"model": rf_final, "feature_columns": RF_FEATURE_COLUMNS}, RF_MODEL_OUTPUT_PATH)
    rf_size_mb = RF_MODEL_OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Saved Random Forest model to {RF_MODEL_OUTPUT_PATH} ({rf_size_mb:.1f} MB)")
    if rf_size_mb > 100:
        print(f"WARNING: {rf_size_mb:.1f} MB is over GitHub's 100 MB push limit -- "
              f"stays gitignored (outputs/*.joblib), not committed.")

    metrics_out = {
        "config": {
            "block_size_deg": BLOCK_SIZE_DEG,
            "n_spatial_folds": N_SPATIAL_FOLDS,
            "n_shift_origins": N_SHIFT_ORIGINS,
            "random_seed": RANDOM_SEED,
            "rf_params": RF_PARAMS,
            "target_column": TARGET_COLUMN,
        },
        "power_law": {
            "height_column": HEIGHT_COLUMN,
            "final_a": a_final,
            "final_b": b_final,
            "spatial_cv_summary": power_law_summary,
            "spatial_cv_fold_results": power_law_cv.to_dict(orient="records"),
        },
        "random_forest": {
            "feature_columns": RF_FEATURE_COLUMNS,
            "spatial_cv_summary": rf_summary,
            "spatial_cv_fold_results": rf_cv.to_dict(orient="records"),
            "feature_importances": rf_importances.to_dict(),
        },
    }
    with open(METRICS_OUTPUT_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Saved combined curve-comparison metrics to {METRICS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
