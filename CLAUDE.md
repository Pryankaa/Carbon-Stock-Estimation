# Carbon Stock Estimation

Satellite-based above-ground biomass (AGB) and carbon stock estimation — estimating how
much carbon is stored in the trees of a site without visiting it and measuring every tree
by hand. The ground truth for building the method is a full field census; the goal is a
satellite pipeline that reproduces it, so future sites need little or no fieldwork.

Owner: AccionLAND Private Limited (geospatial intelligence startup, Ahmedabad).

## The goal, stated precisely

Estimate carbon STOCK (carbon standing at one point in time), not sequestration (carbon
accrued per year). These are different quantities. Sequestration needs two or more time
points and is out of scope for now. Any output, variable name, or report wording must say
"stock", never "sequestration".

## Current status

A working method exists, validated at the CEPT University campus, Ahmedabad against a
field census of 3,261 individually measured trees (978 Mg AGB over 21.5 ha, ~46 Mg/ha).
Three approaches were built and validated:

1. **Trained model** (Random Forest on satellite features) — ~±6% on site total, but needs
   field plots at every new site.
2. **Zero-calibration allometry** (published coefficients × canopy cover) — ~±35%, no
   fieldwork, but depends on a canopy height model.
3. **GEDI-calibrated height→biomass power law** (`agbd = 0.5326 × rh98^1.8307`, spatial-CV
   R² 0.92) — GEDI calibrates this curve ONCE, regionally; it is never used at a
   prediction site. At any real site, height instead comes from a published canopy height
   map (Tier 1, mature sites) or a drone flight (Tier 2, young sites), and the curve
   converts that height to biomass. This is the current deployable engine. Validated
   regionally (Spearman 0.94, +10% bias) and sanity-checked on a real plantation site
   (Botanical, ±30–40%); under-predicts by 73% at CEPT specifically because CEPT's dense
   heritage exotic species are an outlier relative to the regional training vegetation —
   not a general method failure. Full write-up: `docs/METHOD.md`.

## HARD-WON RULES — do not relearn these

These were discovered empirically and cost real effort. Treat them as constraints, not
suggestions.

1. **Target must be RAW kg, never log(kg), for any site TOTAL.** A log-target model
   predicts the median; summing medians under-estimates a right-skewed total by ~48%. Use
   raw for totals. Log models are allowed ONLY for relative hotspot maps, never for a
   reported total.

2. **Validate with SPATIAL block cross-validation, never plain random k-fold.** Adjacent
   cells share trees and pixels; random CV leaks and reports falsely high accuracy. Group
   cells into ~100 m blocks and hold out whole blocks. Repeat with shifted origins for an
   honest uncertainty band.

3. **The Meta/WRI 1 m canopy height model is from ~2016 imagery.** It is blind to anything
   planted after that. Do NOT use it for young/recent plantations. Use GEDI (2019–2024) or
   a current height product instead.

4. **Published allometry must be scaled by CANOPY COVER.** Applied raw to a site that is
   part buildings/roads/bare ground it over-estimates massively (+135% at CEPT). Multiply
   by the vegetated fraction (canopy cover). This single correction took CEPT to −9%.

5. **Report SITE TOTALS, not per-cell values.** Per-cell rank correlation is moderate
   (~0.5) and near zero within high-biomass cells, which hold most of the carbon. Totals
   are reliable because errors cancel; individual cells are not. Never sell per-tree or
   per-cell numbers.

6. **Sentinel-1 SAR and optical-band GLCM texture were tested and REJECTED** at CEPT (no
   signal, degraded the model). GLCM on the canopy height model was KEPT (real signal).
   Re-test rather than assume at a new site, but do not add SAR back by default.

7. **Carbon accounting constants:** carbon fraction 0.47, CO2 ratio 44/12, root:shoot 0.26
   (IPCC 2006 defaults). These are defaults, not site-measured — their uncertainty is on
   top of the model uncertainty.

8. **Stratified sampling of GEDI training shots lives in the Python pipeline, not the GEE
   script.** A single `.sample()` over a full multi-degree region (~3.6M shots) exceeds
   GEE's per-operation size limit ("Image.sample: Computed value is too large") regardless
   of any downstream capping, so the GEE extractor tiles the region and exports every
   quality-filtered shot per tile, unstratified. Capping per tile in GEE would over-sample
   each tile's low-biomass majority and under-represent rare high-biomass shots relative to
   the combined dataset — stratify once, in Python, after combining the tile CSVs. Do not
   re-add biomass thresholding/capping to the GEE script.

9. **Sentinel-2 optical is dropped from the biomass model.** Tested against height-based
   biomass at a fixed canopy height and found no usable signal (Spearman rho ~0.05) —
   optical reflectance saturates over closed canopy and cannot separate a tall dense stand
   from a tall sparse one. The deployable engine is the two-coefficient GEDI-calibrated
   power law (`agbd = a × rh98^b`, see rule 3 above and `docs/METHOD.md`), driven by height
   alone. Do not re-add Sentinel-2 bands/indices as biomass predictors without new evidence
   of signal at a new site.

## Immediate next task

Validate GEDI L4A against the CEPT field census before trusting it anywhere else. If GEDI
reports ~46 Mg/ha at CEPT, the method is trustworthy for sites without ground truth. Then
apply it to plantation sites, reporting GEDI shot count per site (small sites may have too
few shots to be reliable).

## Tech stack

Google Earth Engine (JavaScript) for satellite feature/biomass extraction; Python (pandas,
scikit-learn, numpy, scipy) for modelling and validation. Data lives in CSV exports from
GEE.

## Repo layout

- `gee/` — GEE JavaScript extraction scripts
- `data/raw/`, `data/processed/`, `data/boundaries/` — unmodified exports, cleaned tables,
  site boundary shapefiles
- `src/` — Python pipeline code
- `notebooks/` — exploratory only
- `validation/` — spatial block CV and per-site validation reports
- `outputs/` — generated site totals, shot counts, model artifacts
- `docs/` — method write-ups, decision log, site metadata
- `configs/` — per-site parameters (date ranges, thresholds), referencing `data/boundaries/`
