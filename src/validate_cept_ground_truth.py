"""
Validate the trained biomass model against CEPT University's field census.

CEPT campus (Ahmedabad) has a full field census that no satellite ever
touched: 3,261 individually measured trees, 978 Mg AGB over 537 grid cells
(~21.5 ha, ~46 Mg/ha) -- see data/raw/grid_cells_for_GEE.csv. Applying the
model trained in src/train_biomass_model.py to CEPT's Sentinel-2 features
(extracted separately by gee/cept_feature_extraction.js, using IDENTICAL
feature logic to training) and comparing the predicted total against the
known 978 Mg is the honest end-to-end test of the whole GEDI -> Sentinel-2
method (CLAUDE.md "Immediate next task").

This script only predicts and reports. No retraining, no tuning here --
if the comparison looks bad, that's a finding to act on deliberately in a
later step, not something to quietly fix by tweaking this script.

CLAUDE.md rules followed:
  - Rule 1 (raw kg/Mg, never log): the model predicts raw Mg/ha directly
    (it was trained on raw agbd -- see train_biomass_model.py); no log or
    other transform is applied anywhere here.
  - Rule 5 (site totals are what's reliable, not per-cell values): the
    predicted-vs-actual SITE TOTAL is reported as the headline number.
    Per-cell Spearman correlation is reported too, but only as a
    secondary "does it track the spatial pattern" diagnostic -- expect it
    to be much weaker than the total, consistent with the spatial-CV
    result (R^2 ~0.22) already found in training.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from scipy.stats import spearmanr

# =============================================================================
# CONFIG
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "outputs" / "biomass_rf_model.joblib"
CEPT_FEATURES_PATH = REPO_ROOT / "data" / "raw" / "cept_s2_features_validation.csv"
CEPT_GROUND_TRUTH_PATH = REPO_ROOT / "data" / "raw" / "grid_cells_for_GEE.csv"
OUTPUT_PATH = REPO_ROOT / "outputs" / "cept_validation_per_cell.csv"

CELL_AREA_HA = 0.04  # each CEPT grid cell is 0.04 ha


def load_model():
    if not MODEL_PATH.exists():
        print(f"{MODEL_PATH} not found -- regenerating via src/train_biomass_model.py "
              f"(this re-runs spatial CV + final fit; took ~42 min last time).")
        sys.path.insert(0, str(REPO_ROOT / "src"))
        import train_biomass_model
        train_biomass_model.main()
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["feature_columns"]


def load_and_join():
    features = pd.read_csv(CEPT_FEATURES_PATH)
    truth = pd.read_csv(CEPT_GROUND_TRUTH_PATH)

    print(f"CEPT features: {len(features)} cells. CEPT ground truth: {len(truth)} cells.")

    merged = features.merge(
        truth[["cell_id", "n_trees", "AGB_kg", "AGB_Mg_ha"]],
        on="cell_id", how="inner",
    )
    print(f"Joined on cell_id: {len(merged)} cells.")

    missing_from_truth = set(features["cell_id"]) - set(truth["cell_id"])
    missing_from_features = set(truth["cell_id"]) - set(features["cell_id"])
    if missing_from_truth:
        print(f"WARNING: {len(missing_from_truth)} cell(s) in features but not in "
              f"ground truth: {sorted(missing_from_truth)}")
    if missing_from_features:
        print(f"WARNING: {len(missing_from_features)} cell(s) in ground truth but not "
              f"in features: {sorted(missing_from_features)}")

    return merged


def main():
    model, feature_columns = load_model()
    print(f"Loaded model (n_estimators={model.n_estimators}), "
          f"{len(feature_columns)} features.")

    df = load_and_join()

    # Predict raw Mg/ha directly -- CLAUDE.md rule 1, no transform.
    df["predicted_Mg_ha"] = model.predict(df[feature_columns])
    df["predicted_AGB_kg"] = df["predicted_Mg_ha"] * CELL_AREA_HA * 1000

    predicted_total_Mg = df["predicted_AGB_kg"].sum() / 1000
    actual_total_Mg = df["AGB_kg"].sum() / 1000
    pct_diff = (predicted_total_Mg - actual_total_Mg) / actual_total_Mg * 100

    predicted_mean_Mg_ha = df["predicted_Mg_ha"].mean()
    actual_mean_Mg_ha = df["AGB_Mg_ha"].mean()

    spearman_corr, spearman_p = spearmanr(df["predicted_Mg_ha"], df["AGB_Mg_ha"])

    print("\n" + "=" * 60)
    print("THE NUMBER: predicted site total vs. field ground truth")
    print("=" * 60)
    print(f"Predicted total:  {predicted_total_Mg:.2f} Mg")
    print(f"Field truth:      {actual_total_Mg:.2f} Mg (expected 978 Mg)")
    print(f"Difference:       {pct_diff:+.2f}%")

    print("\n--- Mean Mg/ha ---")
    print(f"Predicted mean:   {predicted_mean_Mg_ha:.2f} Mg/ha")
    print(f"Field mean:       {actual_mean_Mg_ha:.2f} Mg/ha (expected ~46 Mg/ha)")

    print("\n--- Per-cell spatial pattern (secondary diagnostic, not the headline"
          " number -- CLAUDE.md rule 5: totals are reliable, per-cell values are not) ---")
    print(f"Spearman correlation (predicted vs. field, per cell): "
          f"rho={spearman_corr:.4f}, p={spearman_p:.4g}")

    df["diff_kg"] = df["predicted_AGB_kg"] - df["AGB_kg"]
    output_columns = ["cell_id", "n_trees", "AGB_kg", "AGB_Mg_ha",
                       "predicted_AGB_kg", "predicted_Mg_ha", "diff_kg"]
    df[output_columns].to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved per-cell comparison to {OUTPUT_PATH} ({len(df)} rows).")


if __name__ == "__main__":
    main()
