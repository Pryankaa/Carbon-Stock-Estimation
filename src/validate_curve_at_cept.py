"""
Validate the height -> biomass curve against CEPT's field census, using a
published canopy-height map at the CEPT cells -- NOT GEDI.

This is the decisive test of the new architecture (see CLAUDE.md and chat
history): the power-law curve (agbd = a * rh98^b, fit in
src/fit_height_biomass_curve.py from GEDI shots) is only useful if it
still works when the height input comes from what a real prediction site
actually has -- a published canopy-height map (Tier 1) or a drone flight
(Tier 2) -- not from GEDI, which doesn't cover prediction sites at all.
CEPT's field census (3,261 hand-measured trees, 978 Mg AGB over 537 cells,
~21.5 ha, ~46 Mg/ha) never touched a satellite, so comparing the curve's
prediction (fed by an external CHM) against that 978 Mg is an honest,
independent test of the whole no-GEDI-at-the-site pipeline.

Height source: ETH Global Canopy Height 2020 (Lang et al. 2022), sampled
at the 537 CEPT cells by gee/cept_canopy_height_extraction.js -- see that
script's header for the important caveat: this CHM and GEDI's rh98 measure
"canopy top height" via different sensors/methodology, and are not
guaranteed to be numerically interchangeable. That assumption is exactly
what this script tests, not something already proven.

This script only predicts and reports. No retraining, no re-fitting the
curve here -- if the comparison looks bad, that's a finding to act on
deliberately in a later step, not something to quietly fix here.

CLAUDE.md rules followed:
  - Rule 1 (raw kg/Mg, never log): the power-law curve predicts raw Mg/ha
    directly (it was fit by nonlinear least squares in raw space, not
    log-log linearization -- see fit_height_biomass_curve.py); no
    transform is applied anywhere here.
  - Rule 5 (site totals are what's reliable, not per-cell values): the
    predicted-vs-actual SITE TOTAL is the headline number. Per-cell
    Spearman correlation is reported too, specifically because it's the
    metric that was NEGATIVE (-0.28) for the earlier optical-only model --
    seeing whether height fixes that is the whole point of this script,
    even though CLAUDE.md rule 5 says per-cell accuracy isn't what's
    ultimately trusted for reporting.
"""

import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
POWER_LAW_PATH = REPO_ROOT / "outputs" / "height_biomass_power_law.json"
CEPT_HEIGHT_PATH = REPO_ROOT / "data" / "raw" / "cept_canopy_height_validation.csv"
CEPT_GROUND_TRUTH_PATH = REPO_ROOT / "data" / "raw" / "grid_cells_for_GEE.csv"
OUTPUT_PATH = REPO_ROOT / "outputs" / "cept_curve_validation_per_cell.csv"

CELL_AREA_HA = 0.04  # each CEPT grid cell is 0.04 ha
HEIGHT_COLUMN = "canopy_height_m"  # from gee/cept_canopy_height_extraction.js


def load_curve():
    if not POWER_LAW_PATH.exists():
        sys.exit(f"{POWER_LAW_PATH} not found -- run src/fit_height_biomass_curve.py first.")
    with open(POWER_LAW_PATH) as f:
        curve = json.load(f)
    return curve["a"], curve["b"]


def load_and_join():
    if not CEPT_HEIGHT_PATH.exists():
        sys.exit(
            f"{CEPT_HEIGHT_PATH} not found.\n\n"
            "This needs a GEE export first: run gee/cept_canopy_height_extraction.js "
            "in the Earth Engine Code Editor (samples ETH Global Canopy Height 2020 at "
            "the 537 CEPT cells), then place the exported CSV at that path before "
            "re-running this script."
        )
    heights = pd.read_csv(CEPT_HEIGHT_PATH)
    truth = pd.read_csv(CEPT_GROUND_TRUTH_PATH)

    print(f"CEPT heights: {len(heights)} cells. CEPT ground truth: {len(truth)} cells.")

    merged = heights.merge(
        truth[["cell_id", "n_trees", "AGB_kg", "AGB_Mg_ha"]],
        on="cell_id", how="inner",
    )
    print(f"Joined on cell_id: {len(merged)} cells.")

    missing_from_truth = set(heights["cell_id"]) - set(truth["cell_id"])
    missing_from_heights = set(truth["cell_id"]) - set(heights["cell_id"])
    if missing_from_truth:
        print(f"WARNING: {len(missing_from_truth)} cell(s) in heights but not in "
              f"ground truth: {sorted(missing_from_truth)}")
    if missing_from_heights:
        print(f"WARNING: {len(missing_from_heights)} cell(s) in ground truth but not "
              f"in heights: {sorted(missing_from_heights)}")

    missing_height = merged[HEIGHT_COLUMN].isna().sum()
    if missing_height:
        print(f"WARNING: {missing_height} cell(s) have a null {HEIGHT_COLUMN} "
              f"(CHM nodata at that pixel) -- dropping them from the comparison.")
        merged = merged.dropna(subset=[HEIGHT_COLUMN])

    return merged


def main():
    a, b = load_curve()
    print(f"Loaded curve: agbd = {a:.4f} * {HEIGHT_COLUMN}^{b:.4f}")

    df = load_and_join()

    # Predict raw Mg/ha directly -- CLAUDE.md rule 1, no transform.
    df["predicted_Mg_ha"] = a * df[HEIGHT_COLUMN].clip(lower=0) ** b
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

    print("\n--- Per-cell spatial pattern (this is what was NEGATIVE, -0.28, for the"
          " earlier optical-only model -- see if height fixes it) ---")
    print(f"Spearman correlation (predicted vs. field, per cell): "
          f"rho={spearman_corr:.4f}, p={spearman_p:.4g}")

    df["diff_kg"] = df["predicted_AGB_kg"] - df["AGB_kg"]
    output_columns = ["cell_id", "n_trees", "AGB_kg", "AGB_Mg_ha", HEIGHT_COLUMN,
                       "predicted_AGB_kg", "predicted_Mg_ha", "diff_kg"]
    df[output_columns].to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved per-cell comparison to {OUTPUT_PATH} ({len(df)} rows).")


if __name__ == "__main__":
    main()
