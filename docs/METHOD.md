# Method: satellite-based carbon stock estimation

This document describes the validated method for estimating above-ground biomass (AGB)
and carbon **stock** — carbon standing at one point in time, not sequestration (carbon
accrued per year) — from satellite data, without a field visit. It is written for a
client or colleague deciding whether to trust the numbers this pipeline produces.

## 1. Architecture

The method has one job: turn a canopy height measurement into a biomass estimate, without
ever needing GEDI (spaceborne LiDAR) to cover the actual site being reported on.

```
   GEDI (regional shots, 2019-2024)
          |
          |  calibrate ONCE
          v
   height -> biomass power law
   agbd = 0.5326 x rh98^1.8307
          |
          |  apply at ANY site, using height from:
          v
   +-----------------------+-----------------------+
   | Tier 1: mature sites  | Tier 2: young sites    |
   | published canopy      | drone photogrammetry   |
   | height map            | (one flight)           |
   +-----------------------+-----------------------+
          |
          v
   site biomass total (Mg) -> carbon stock (Mg C) -> CO2e (t)
```

**GEDI never touches a prediction site.** It is used exactly once, regionally, to fit the
curve below. At any real client site — the site this report is about — height comes from
an external source, and the curve converts that height to biomass. This is deliberate: an
earlier version of this method used GEDI height as a per-site prediction feature, which is
undeployable, since GEDI's spaceborne LiDAR footprints do not reliably cover arbitrary
client sites.

### 1.1 The calibrated curve

Fit by nonlinear least squares directly in raw (not log-log) space, per the raw-target
rule in `CLAUDE.md` — a log-log fit would bias toward the median and under-predict the
site total:

```
agbd (Mg/ha) = 0.5326 x rh98 (m) ^ 1.8307
```

- Fit on ~60,000 quality-filtered GEDI L4A biomass shots (`agbd`) joined to GEDI L2A
  relative height (`rh98`), pooled across the regional training area.
- Validated with spatial block cross-validation (0.25° blocks, 5 folds x 4 shifted
  origins = 20 fold-runs — see `CLAUDE.md` rule 2 on why plain random k-fold is
  disqualified here): **R² = 0.922 ± 0.026**, predicted/actual sum ratio
  **1.02 ± 0.03** (near-unbiased on totals).
- A four-band Random Forest (`rh50, rh75, rh90, rh98`) was fit against the same target as
  a check: R² = 0.917 ± 0.030, essentially tied, with `rh98` alone carrying 94% of the
  RF's feature importance. The extra three height bands buy negligible accuracy.
- **The power law, not the Random Forest, is the deployable engine.** An external height
  source (published CHM or drone photogrammetry) gives a single top-of-canopy height per
  point, not a full relative-height profile — so the two-coefficient power law is what a
  real site can actually supply as input. It is also simpler to audit, version, and
  re-derive than a serialized forest.
- Coefficients and the full CV summary are versioned at
  `outputs/height_biomass_power_law.json`, produced by `src/fit_height_biomass_curve.py`.

### 1.2 Why Sentinel-2 optical was dropped

Sentinel-2 spectral indices (NDVI, EVI, raw bands) were tested as biomass predictors
alongside height and do not add usable signal: at a fixed canopy height, the correlation
between optical features and biomass is approximately **rho ~ 0.05** — noise, not signal.
Optical reflectance saturates over closed canopy and cannot distinguish a tall, dense
stand from a tall, sparse one the way structural height can. Sentinel-2 is therefore
**excluded from the biomass model**. (This is a separate finding from `CLAUDE.md` rule 6,
which rejected SAR and GLCM texture on the optical bands specifically — both point the
same direction: structure, not spectral signature, is what predicts biomass here.)

Optical data may still be useful for canopy-cover masking or vegetation delineation, but
it is not part of the biomass value chain.

### 1.3 Deployment tiers

| Tier | Site type | Height source | Fieldwork |
|---|---|---|---|
| **Tier 1** | Mature / established canopy | Published canopy height map (e.g. a global or regional CHM) | None |
| **Tier 2** | Young plantations, or sites where the published map predates planting | Drone photogrammetry (structure-from-motion canopy height model) | One flight; repeatable on the same flight plan for future monitoring |

Tier selection is a per-site judgment call: if the published height map's imagery date is
close to or after planting, Tier 1 is appropriate (see `CLAUDE.md` rule 3 — do not use a
stale height map on anything planted after the map's imagery date). If planting postdates
the map, the site needs Tier 2.

### 1.4 Carbon accounting

Biomass Mg -> carbon stock is a fixed, standard conversion (IPCC 2006 defaults, per
`CLAUDE.md` rule 7 — not site-measured, so their own uncertainty stacks on top of the
model's):

```
carbon (Mg C)      = AGB (Mg) x 0.47                       (carbon fraction)
total carbon (Mg C) = carbon (Mg C) x (1 + 0.26)            (+ root:shoot below-ground)
CO2e (t)            = total carbon (Mg C) x 44/12
```

## 2. Validation results

Three tests, at three levels of representativeness, stated without smoothing over the
weak one:

| Test | Site type | Result | Verdict |
|---|---|---|---|
| Regional natural vegetation | Dryland / cropland / plantation mosaic, GEDI-covered | Spearman rho = 0.94, ~+10% bias | **Strong** — the curve tracks biomass well across the vegetation the training data was actually drawn from. |
| Botanical plantation | Real client site, planted 2018, 1.82 ha, height from ETH 2020 CHM | 13.2 Mg AGB (~29 tCO2e), ±30-40% | **Plausible for a young plantation, but undercounts.** 35% of sample points read zero height because the CHM predates two years of growth that occurred between planting and canopy maturation — see limitation 2.1 below. Behaves sanely; not a validated accuracy number, since no field census exists for this site. |
| CEPT campus | Field census, 3,261 trees, 978 Mg AGB, 21.5 ha | Predicted 73% below field truth | **Outlier, not a method failure.** CEPT is a manicured urban campus of dense, heritage exotic species with abnormally thick trunks for their height — a very different height-to-biomass relationship than the regional vegetation the curve was calibrated on. See limitation 2.2. |

### 2.1 Young-plantation undercounting (stale height map)

A published CHM reflects canopy height as of its own imagery date, not today. If a site
was planted after that date, the map understates (or, as at Botanical, in places entirely
misses) canopy that has grown since. This is not a curve defect — it is a height-input
problem, and it is exactly what Tier 2 (drone height) exists to fix: a drone flight
captures the site's *current* canopy, whatever its age.

### 2.2 Unusual/dense species need local calibration

The regional power law assumes vegetation broadly similar to what GEDI sampled across the
training region. A site with an atypical height-to-biomass relationship — dense plantings,
unusual species composition, multi-stemmed or heavily pruned trees, or (as at CEPT)
mature heritage trees with unusually thick trunks for their canopy height — will not be
well served by the regional curve. Flag such sites for local calibration (a small field
plot to refit or bias-correct the curve locally) rather than trusting the regional
coefficients directly.

## 3. Honest product definition

This method produces a **screening-grade carbon stock estimate**:

- **Accuracy: roughly ±30-40%** on a site total, for regionally-typical vegetation with an
  appropriately current height source. This is wider than the trained-model approach's
  ~±6% (which needs field plots at every site) and comparable to the zero-calibration
  allometry's ~±35% — but with no fieldwork required, and with better structural grounding
  than optical-only allometry.
- **No fieldwork required** for Tier 1 sites with a current published height map.
- **Young plantations need a current height layer** — a drone flight (Tier 2), not a
  possibly-stale published map — to avoid the undercounting shown at Botanical.
- **Unusual or unusually dense species need local calibration** before the regional curve
  can be trusted, per the CEPT result.
- **This is NOT an exact-tonnes number.** A site total from this method is a defensible
  screening estimate, appropriate for prioritization, portfolio-level reporting, or a
  first-pass carbon stock figure. It is explicitly not a substitute for a field-verified
  measurement where regulatory or transactional precision is required — that still needs a
  site visit, and this document says so rather than overselling the satellite number.
- **Site totals, not per-point or per-cell values, are what's reliable** (`CLAUDE.md` rule
  5). Individual grid points or cells within a site should not be reported or sold as
  precise numbers; only the aggregated site total carries the stated uncertainty band.

## 4. Where this lives in the repo

- `src/fit_height_biomass_curve.py` — fits and validates the power law (and the RF
  comparison), produces `outputs/height_biomass_power_law.json`.
- `gee/cept_canopy_height_extraction.js`, `src/validate_curve_at_cept.py` — CEPT
  validation (Section 2, row 3).
- `gee/plantation_canopy_height_extraction.js`, `src/validate_curve_at_plantation.py` —
  Botanical plantation validation (Section 2, row 2).
- `CLAUDE.md` — hard-won operating rules this method must not violate (raw targets,
  spatial CV, stale-CHM caveat, canopy-cover scaling for the older allometry approach,
  site totals, carbon accounting constants).
