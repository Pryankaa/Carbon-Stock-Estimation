"""
Apply the GEDI-calibrated height -> biomass curve to a real client
plantation site (Botanical), using canopy height from a published CHM --
not GEDI, not CEPT.

REFRAME (see CLAUDE.md and chat history): CEPT is a manicured urban
campus of heritage exotic species with abnormally thick trunks -- an
outlier relative to the regional vegetation (dryland/cropland/plantation)
the curve was actually trained on. Region-wide the curve tracks biomass
at Spearman 0.94 with roughly +10% bias, and at a fixed height biomass
only varies ~2x (not the ~20x CEPT implied). Client sites are plantations,
much closer to regional vegetation than to CEPT, so this is a more
representative test of the curve than CEPT was.

NO FIELD TRUTH EXISTS for this site, so this script cannot check
ACCURACY -- only whether the curve produces a SANE number (a real,
plausible biomass value) on an actual client site, unlike CEPT's outlier
behavior. Treat the output as a behavior check, not a validated estimate.

CLAUDE.md rules followed:
  - Rule 1 (raw kg/Mg, never log): the curve predicts raw Mg/ha directly,
    no transform anywhere in this script.
  - Rule 5 (site totals, not per-point values, are what's reliable): the
    headline number is the site total (site-mean Mg/ha x area), reported
    with an honest +/-30-40% uncertainty band since there's no field
    truth here to calibrate a tighter one. Per-point predictions are
    shown only as a sanity-check distribution, not as reported numbers.
"""

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
POWER_LAW_PATH = REPO_ROOT / "outputs" / "height_biomass_power_law.json"

# -----------------------------------------------------------------------
# Site config -- Botanical is the primary test: the largest/oldest of the
# four plantation boundaries (Ngo, Amshet, Botanical, Jagadiya). Planted
# 2018; ETH canopy height is a 2020 snapshot, so it captures only ~2
# years of growth -- height (and therefore predicted biomass) is likely
# thin relative to the plantation's actual current state. This tests
# whether the curve behaves sanely on a real site, NOT a definitive
# current biomass number (see the caveat printed at runtime).
# -----------------------------------------------------------------------
SITE_NAME = "botanical"
SITE_AREA_HA = 1.820407571906957  # data/boundaries/Botanical_boundary.geojson Area_Ha
SITE_HEIGHT_CSV = REPO_ROOT / "data" / "raw" / "botanical_canopy_height_validation.csv"
OUTPUT_PATH = REPO_ROOT / "outputs" / f"{SITE_NAME}_curve_validation_per_point.csv"

HEIGHT_COLUMN = "canopy_height_m"
UNCERTAINTY_BAND_PCTS = [30, 40]  # honest range given no field truth to calibrate a tighter one


def load_curve():
    if not POWER_LAW_PATH.exists():
        sys.exit(f"{POWER_LAW_PATH} not found -- run src/fit_height_biomass_curve.py first.")
    with open(POWER_LAW_PATH) as f:
        curve = json.load(f)
    return curve["a"], curve["b"]


def load_heights():
    if not SITE_HEIGHT_CSV.exists():
        sys.exit(
            f"{SITE_HEIGHT_CSV} not found.\n\n"
            f"This needs a GEE export first: run "
            f"gee/plantation_canopy_height_extraction.js in the Earth Engine "
            f"Code Editor (samples ETH Global Canopy Height 2020 over a grid "
            f"of points within the {SITE_NAME.title()} boundary), then place "
            f"the exported CSV at that path before re-running this script."
        )
    df = pd.read_csv(SITE_HEIGHT_CSV)
    n_missing = df[HEIGHT_COLUMN].isna().sum()
    if n_missing:
        print(f"WARNING: {n_missing} point(s) have a null {HEIGHT_COLUMN} "
              f"(CHM nodata at that pixel) -- dropping them.")
        df = df.dropna(subset=[HEIGHT_COLUMN])
    return df


def main():
    a, b = load_curve()
    print(f"Loaded curve: agbd = {a:.4f} * {HEIGHT_COLUMN}^{b:.4f}")
    print(f"\nSite: {SITE_NAME} ({SITE_AREA_HA} ha)")
    print("CAVEAT: ETH canopy height is a 2020 snapshot; this site was planted "
          "2018, so the height layer captures only ~2 years of growth. This "
          "tests curve BEHAVIOR on a real site, not a definitive current "
          "biomass number -- treat the result as a rough, likely-low sanity "
          "check, not a final estimate.")

    df = load_heights()
    print(f"\nPoints sampled: {len(df)}")
    print(df[HEIGHT_COLUMN].describe().to_string())

    df["predicted_Mg_ha"] = a * df[HEIGHT_COLUMN].clip(lower=0) ** b

    site_mean_Mg_ha = df["predicted_Mg_ha"].mean()
    site_total_Mg = site_mean_Mg_ha * SITE_AREA_HA

    print("\n" + "=" * 60)
    print(f"THE NUMBER: {SITE_NAME} predicted site total")
    print("=" * 60)
    print(f"Site-mean predicted biomass: {site_mean_Mg_ha:.2f} Mg/ha")
    print(f"Site area:                   {SITE_AREA_HA} ha")
    print(f"Site total:                  {site_total_Mg:.2f} Mg")

    print("\n--- Honest uncertainty band (no field truth here to calibrate a "
          "tighter one) ---")
    for pct in UNCERTAINTY_BAND_PCTS:
        low = site_total_Mg * (1 - pct / 100)
        high = site_total_Mg * (1 + pct / 100)
        print(f"+/-{pct}%: [{low:.2f}, {high:.2f}] Mg")

    print("\n--- Per-point prediction distribution (sanity check only -- "
          "CLAUDE.md rule 5: site totals are reliable, per-point values "
          "are not) ---")
    print(df["predicted_Mg_ha"].describe().to_string())

    output_columns = ["point_id", "lat", "lon", HEIGHT_COLUMN, "predicted_Mg_ha"]
    df[output_columns].to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved per-point predictions to {OUTPUT_PATH} ({len(df)} rows).")


if __name__ == "__main__":
    main()
