"""
Combine GEDI L4A x Sentinel-2 training tiles, de-duplicate, and stratify.

The GEE extractor (gee/gedi_s2_training_extract.js) exports one CSV per
tile of the region, with every quality-filtered GEDI shot in that tile and
no biomass filtering — a single .sample()/.sampleRegions() call over the
whole region exceeds GEE's per-operation size limit, so stratification
could not happen there. Per CLAUDE.md (hard-won rule 8), stratified
sampling of GEDI training shots lives here instead, applied exactly once
across the combined dataset. Capping per tile in GEE would have
over-sampled each tile's low-biomass majority and under-represented rare
high-biomass shots relative to the combined region.

This script only combines, cleans, and stratifies. No modelling here.

CLAUDE.md rules followed:
  - Rule 1 (raw kg/Mg, never log): agbd is left in its raw Mg/ha units
    throughout — no log transform anywhere in this script.
  - Rule 2 (spatial block CV, not random k-fold): not relevant yet (no
    train/test split happens in this script), but the combined output
    still needs a spatial block split downstream, not a random one, when
    a model is eventually trained on it.
  - Rule 8 (stratification lives in Python, not GEE): this script IS that
    step.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "gedi_s2_training_combined_stratified.csv"

# Must match gee/gedi_s2_training_extract.js: EXPORT_FILE_PREFIX (the
# non-TEST_MODE value) + '_tile_' + i + '_' + j, and TILE_GRID_SIZE.
TILE_FILE_PATTERN = "gedi_l4a_s2_training_gujarat_maharashtra_tile_*.csv"
TILE_GRID_SIZE = 8  # 8x8 = 64 tiles expected
TILE_NAME_RE = re.compile(r"_tile_(\d+)_(\d+)\.csv$")

# Stratified sampling (moved here from GEE — see module docstring).
BIOMASS_THRESHOLD_MG_HA = 20.0  # high/low split; region's 90th percentile is ~39 Mg/ha, so this keeps essentially all real tree signal
LOW_BIOMASS_SAMPLE_SIZE = 30000  # target row count for the random low-biomass subsample; all high-biomass shots are kept, uncapped
RANDOM_SEED = 42  # fixed seed so the low-biomass subsample is reproducible across runs

# Six-bin distribution used for reporting only (not sampling) — same bins
# used throughout this project's GEE scripts.
DISTRIBUTION_BAND_EDGES = [0, 5, 20, 40, 70, 120, np.inf]
DISTRIBUTION_LABELS = ["0-5", "5-20", "20-40", "40-70", "70-120", "120+"]


# =============================================================================
# 1. Combine tile CSVs, verifying the full grid is present
# =============================================================================

def find_tile_files():
    return sorted(RAW_DIR.glob(TILE_FILE_PATTERN))


def parse_tile_index(path):
    match = TILE_NAME_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not parse tile row/col from filename: {path.name}")
    return int(match.group(1)), int(match.group(2))


def verify_tile_grid(files):
    found = {parse_tile_index(f) for f in files}
    expected = {(r, c) for r in range(TILE_GRID_SIZE) for c in range(TILE_GRID_SIZE)}
    missing = sorted(expected - found)
    unexpected = sorted(found - expected)
    return missing, unexpected


def combine_tiles(files):
    frames = []
    for f in files:
        r, c = parse_tile_index(f)
        df = pd.read_csv(f)
        df["tile_id"] = f"{r}_{c}"  # provenance only — not a model feature, drop before training
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# 2. De-duplicate (tiles share edges)
# =============================================================================

def deduplicate(df):
    return df.drop_duplicates(subset=["lat", "lon", "agbd"])


# =============================================================================
# 3. Stratified sampling — keep every high-biomass shot, subsample the rest
# =============================================================================

def stratified_sample(df):
    low_mask = df["agbd"] < BIOMASS_THRESHOLD_MG_HA
    high = df.loc[~low_mask]
    low = df.loc[low_mask]

    if len(low) > LOW_BIOMASS_SAMPLE_SIZE:
        low_sampled = low.sample(n=LOW_BIOMASS_SAMPLE_SIZE, random_state=RANDOM_SEED)
    else:
        low_sampled = low

    combined = pd.concat([high, low_sampled], ignore_index=True)
    return combined, len(low), len(low_sampled), len(high)


# =============================================================================
# 4. Reporting helpers
# =============================================================================

def biomass_distribution(df):
    binned = pd.cut(df["agbd"], bins=DISTRIBUTION_BAND_EDGES, labels=DISTRIBUTION_LABELS, right=False)
    return binned.value_counts().reindex(DISTRIBUTION_LABELS).fillna(0).astype(int)


def main():
    print(f"Looking for tile CSVs in {RAW_DIR} matching '{TILE_FILE_PATTERN}'")
    files = find_tile_files()
    print(f"Found {len(files)} tile file(s).")

    if not files:
        print("No tile CSVs found — nothing to combine. "
              "Add the tile exports to data/raw/ and re-run.")
        return

    missing, unexpected = verify_tile_grid(files)
    if missing:
        print(f"WARNING: {len(missing)} tile(s) missing from the "
              f"{TILE_GRID_SIZE}x{TILE_GRID_SIZE} grid: {missing}")
    else:
        print(f"All {TILE_GRID_SIZE * TILE_GRID_SIZE} expected tiles are present.")
    if unexpected:
        print(f"WARNING: {len(unexpected)} tile filename(s) fall outside the "
              f"expected {TILE_GRID_SIZE}x{TILE_GRID_SIZE} grid: {unexpected}")

    combined = combine_tiles(files)
    total_combined = len(combined)
    print(f"\nTotal rows combined across all tiles: {total_combined}")

    deduped = deduplicate(combined)
    total_deduped = len(deduped)
    print(f"Rows after de-duplication (same lat/lon/agbd): {total_deduped} "
          f"(dropped {total_combined - total_deduped})")

    stratified, low_total, low_kept, high_total = stratified_sample(deduped)
    print(f"\nLow-biomass shots (< {BIOMASS_THRESHOLD_MG_HA} Mg/ha): "
          f"{low_total} -> subsampled to {low_kept}")
    print(f"High-biomass shots (>= {BIOMASS_THRESHOLD_MG_HA} Mg/ha): "
          f"kept all {high_total} (no cap)")
    total_stratified = len(stratified)
    print(f"Total rows after stratified sampling: {total_stratified}")

    print("\n--- Row counts: before -> after ---")
    print(f"Combined (raw, all tiles): {total_combined}")
    print(f"After de-duplication:      {total_deduped}")
    print(f"After stratified sampling: {total_stratified}")

    print("\n--- Biomass distribution BEFORE stratification (post-dedup) ---")
    dist_before = biomass_distribution(deduped)
    print(dist_before.to_string())

    print("\n--- Biomass distribution AFTER stratification ---")
    dist_after = biomass_distribution(stratified)
    print(dist_after.to_string())
    print(f"\n120+ Mg/ha (dense-canopy ceiling) shot count: {dist_after['120+']}")

    print("\n--- Missing-value counts per column (post-stratification) ---")
    missing_counts = stratified.isna().sum()
    print(missing_counts.to_string())

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stratified.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved combined, de-duplicated, stratified training set to "
          f"{OUTPUT_PATH} ({total_stratified} rows).")


if __name__ == "__main__":
    main()
