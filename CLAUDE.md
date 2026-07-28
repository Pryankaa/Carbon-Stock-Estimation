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
3. **GEDI L4A direct biomass** — reads biomass straight from spaceborne LiDAR, no
   allometry. This is the current focus because it works on sites planted after 2016.

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
