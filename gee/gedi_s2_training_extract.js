/**
 * GEDI L4A x Sentinel-2 training-data extractor
 * ===============================================
 *
 * WHY GEDI IS USED AS TRAINING DATA, NOT A PER-SITE ESTIMATOR
 * -------------------------------------------------------------
 * GEDI L4A gives direct, allometry-free aboveground biomass density (agbd) at
 * individual spaceborne-lidar footprints (~25 m circles), with no need to
 * assume an allometric equation. But GEDI shots sit on sparse orbit tracks —
 * within a track, footprints are ~60 m apart; between tracks, several km
 * apart. A hectare-scale site will contain only a handful of shots, if any,
 * so GEDI alone CANNOT produce a wall-to-wall estimate for a single site
 * (see CLAUDE.md: GEDI is a regional product). Sentinel-2 has no direct
 * biomass signal of its own, but it covers every pixel of every site.
 *
 * So the strategy is:
 *   1. Pool quality-filtered GEDI shots across a whole REGION (not one site)
 *      as (sparse, but plentiful in aggregate) biomass TRAINING LABELS.
 *   2. At each shot, sample Sentinel-2 (+ a height predictor) as the
 *      PREDICTORS a model can see everywhere.
 *   3. A later step trains a model on this table and applies it wall-to-wall
 *      wherever Sentinel-2 exists, including hectare sites with zero GEDI
 *      coverage of their own.
 *
 * NOTE: step 2 originally matched each shot to Sentinel-2 imagery from its
 * own year (see CLAUDE.md's time-alignment rule). That per-year join broke
 * silently (an ee.Number/plain-JS-number equality mismatch) and produced
 * an export with correct headers but zero rows. This version replaces it
 * with ONE multi-year composite (all of SEASON_YEARS pooled together, see
 * section 2) sampled at every shot regardless of its date — trading away
 * per-shot time-alignment for a working end-to-end export. Reintroduce
 * per-year/per-shot matching once the rest of the pipeline is validated.
 *
 * THIS SCRIPT ONLY BUILDS AND EXPORTS THE TRAINING TABLE.
 * No model, no prediction, no per-site inference happens here.
 *
 * The region (Gujarat + Maharashtra) turns out to be heavily biomass-
 * imbalanced: median shot biomass ~4 Mg/ha, 90th percentile ~39 Mg/ha, since
 * most of the region is dryland/cropland/urban rather than forest. A naive
 * random sample of the ~3.6M quality shots would be almost entirely
 * near-zero biomass and would starve a model of the high-biomass (tree)
 * signal it needs. So this script applies STRATIFIED sampling on both
 * ends: the low-biomass majority is randomly subsampled down to a
 * manageable size, AND the high-biomass tier is split into bands (most
 * high shots cluster at 20-40 Mg/ha, very few above 120) with its own
 * per-band cap, so neither the near-zero majority nor the common
 * mid-high band can crowd out the rare dense-canopy shots (see CONFIG).
 *
 * GEDI coverage note: this script reads GEDI04_A_002_MONTHLY (see CONFIG —
 * the footprint-level GEDI04_A_002 asset is a table/index folder, not an
 * ImageCollection, and errors in GEE). That monthly raster mosaic covers
 * March 2019 - March 2023 only, so all training labels here are 2019-2023,
 * not 2024.
 *
 * Output: one CSV row per sampled quality GEDI shot, columns = agbd,
 * agbd_se, lat, lon (derived from each sampled feature's own geometry —
 * lat_lowestmode/lon_lowestmode are footprint-TABLE columns, not bands on
 * this gridded MONTHLY raster, and selecting them made .sample() return
 * empty; see CONFIG and section 1), and Sentinel-2 post-monsoon/dry-season
 * NDVI + EVI plus reflectance bands B2-B12 (B10 excluded: it is an
 * L1C-only cirrus band, not present in the L2A surface-reflectance
 * product) — all from ONE multi-year composite, the same for every shot
 * regardless of its date (see NOTE above and section 2). No shot_date
 * column (dropped along with per-shot time-matching) and no
 * sensitivity/landsat_treecover/pft_class for now — commented out in
 * section 1 pending verification against the 'GEDI monthly bands' print.
 *
 * External canopy-height sampling is DISABLED for this first run — see the
 * TODO in the CONFIG section below.
 *
 * CLAUDE.md rules this script follows:
 *   - Rule 3: does NOT use the Meta/WRI ~2016 canopy height model (blind to
 *     anything planted after 2016). The external ~2020 replacement is
 *     disabled too for the same reason (see TODO below). GEDI's own
 *     sensitivity/landsat_treecover/pft_class covariates were meant to
 *     stand in instead, but are temporarily commented out in section 1
 *     pending verification against the 'GEDI monthly bands' print — no
 *     height/vegetation signal is included in this test-mode run.
 *   - Rule 6: does NOT add Sentinel-1 SAR or optical GLCM texture back in.
 *   - GEDI is explicitly treated as regional training data here, never as a
 *     per-site estimator (see header above and the immediate-next-task note
 *     in CLAUDE.md about validating GEDI before trusting it elsewhere).
 */

// -----------------------------------------------------------------------
// PERFORMANCE NOTE: this script is intentionally quiet in the Console. The
// full region has ~3.6M quality GEDI shots — evaluating that collection
// for a live print (a .size(), a full reduce, a per-band or distribution
// scan) is enough on its own to make the Code Editor unresponsive.
// Everything below stays lazy; the heavy work only actually runs once,
// inside the Export.table.toDrive task. To inspect the biomass
// distribution and shot counts, do it AFTER exporting — load the CSV in
// Python (Claude Code can do this) rather than adding print()s here. The
// one exception is the capped TEST_MODE row-count check right before the
// export (section 4) — see TEST_MODE above.
// -----------------------------------------------------------------------

// =============================================================================
// 0. CONFIG — placeholders to replace with real values before running
// =============================================================================

// TEST_MODE: fail-fast check on a small, known-vegetated area before
// burning a long run on the full region. When true, REGION below is
// overridden with a 0.3 deg box in the Western Ghats (real forest, plenty
// of GEDI shots expected) and the ONE interactive print near the export
// at the bottom is enabled.
//
// Workflow: run with TEST_MODE = true, confirm "TEST row count" in the
// Console is > 0, THEN set TEST_MODE = false and start the real export
// task. Leave TEST_MODE = false for the real regional export — the test
// print is gated on it and won't fire (or hang the UI) once it's off.
var TEST_MODE = true;

// Placeholder region: Gujarat/Maharashtra, ~20-24 N, 72-76 E.
// Replace with the real regional bounding box before running.
var REGION = ee.Geometry.Rectangle([72, 20, 76, 24]);

if (TEST_MODE) {
  // Small box in the Western Ghats — real forest, should return a
  // nonzero row count quickly without evaluating the full ~3.6M-shot
  // region.
  REGION = ee.Geometry.Rectangle([73.0, 20.0, 73.3, 20.3]);
}

// TODO(height strategy): external canopy-height sampling is disabled. The
// candidate asset ID below is unverified (unconfirmed in this environment),
// and more importantly it is a fixed ~2020 snapshot — the same "blind to
// anything planted after the snapshot year" problem that ruled out the 2016
// Meta/WRI CHM (CLAUDE.md rule 3). GEDI's own sensitivity/landsat_treecover/
// pft_class covariates were meant to stand in instead (contemporaneous with
// each shot), but are currently commented out of the .select() in section 1
// pending verification against the 'GEDI monthly bands' print below — so
// this test-mode run carries no height/vegetation signal at all yet.
// Revisit once we know whether these covariates carry enough signal on
// their own, or whether a genuinely current external height product is
// needed.
// var CANOPY_HEIGHT_ASSET_ID =
//   'projects/sat-io/open-datasets/ETH_GlobalCanopyHeight_2020_10m_v1';

// GEDI04_A_002 (footprint-level) is a table/index folder in this GEE asset,
// not an ImageCollection — loading it directly throws "found IndexedFolder".
// Use the monthly raster mosaic instead; it's the same asset the 3.6M-shot
// checkpoint used. Coverage is March 2019 - March 2023 only (see header).
var GEDI_COLLECTION_ID = 'LARSE/GEDI/GEDI04_A_002_MONTHLY';
var S2_COLLECTION_ID = 'COPERNICUS/S2_SR_HARMONIZED';

// One-time verification print: this asset's real band list. The earlier
// 3.6M-shot checkpoint only ever selected 'agbd', so lat_lowestmode /
// lon_lowestmode / shot_date_millis being invalid band names on this
// gridded MONTHLY raster (they're footprint-TABLE columns, not raster
// bands) went unnoticed until .select() on them made .sample() return
// empty. This only inspects a single Image's metadata, not the ~3.6M
// shot collection, so it's cheap — keep it for now while sensitivity/
// landsat_treecover/pft_class below are still unverified against it.
print('GEDI monthly bands:', ee.ImageCollection(GEDI_COLLECTION_ID).first().bandNames());

// Years folded into the single multi-year post-monsoon/dry-season
// composite below (section 2) — hardcoded to match GEDI04_A_002_MONTHLY's
// actual coverage (March 2019 - March 2023) instead of computing it from
// the shot collection's min/max date, which would require a .getInfo()
// round trip over the full ~3.6M shot collection just to find the range.
var SEASON_YEARS = [2019, 2020, 2021, 2022, 2023];

// Cloud Score+ replaces QA60 for cloud masking (see section 2 below for
// why). CS_PLUS_CLEAR_THRESHOLD follows Google's own guidance for the
// cs_cdf band: >= 0.6 is a reasonable default "clear" cutoff.
var CS_PLUS_COLLECTION_ID = 'GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED';
var CS_PLUS_BAND = 'cs_cdf';
var CS_PLUS_CLEAR_THRESHOLD = 0.6;

// Stratified sampling to fix the region's biomass imbalance (median ~4
// Mg/ha, 90th percentile ~39 Mg/ha — see header). Shots below
// BIOMASS_THRESHOLD_MG_HA are randomly subsampled down to
// LOW_BIOMASS_SAMPLE_SIZE; shots at or above it are split into bands and
// each band is capped at its own target, so the training set is balanced
// in both directions instead of swamped by near-zero OR dominated by the
// most common high-biomass band.
var BIOMASS_THRESHOLD_MG_HA = 20; // high/low split; ~90th percentile is 39 Mg/ha, so this keeps essentially all real tree signal
var LOW_BIOMASS_SAMPLE_SIZE = 30000; // target row count for the random low-biomass subsample
var HIGH_BIOMASS_SAMPLE_SIZE = 30000; // total target across all high-biomass bands combined, matching the low tier

// Most high-biomass shots cluster at 20-40 Mg/ha with very few above 120;
// a flat cap on the whole high tier would be dominated by 20-40 Mg/ha
// shots and lose the dense-canopy upper range. Splitting into bands and
// giving each an equal share of HIGH_BIOMASS_SAMPLE_SIZE keeps the rare,
// most valuable dense-canopy shots from being crowded out — a band with
// fewer real shots than its target just keeps everything it has.
// First edge is tied to BIOMASS_THRESHOLD_MG_HA (not re-hardcoded) so the
// low tier and the high tier's first band can never drift apart into an
// overlap or a gap.
var HIGH_BIOMASS_BAND_EDGES = [BIOMASS_THRESHOLD_MG_HA, 40, 70, 120, Infinity]; // Mg/ha; edit to change bands
var HIGH_BAND_TARGET = Math.floor(HIGH_BIOMASS_SAMPLE_SIZE / (HIGH_BIOMASS_BAND_EDGES.length - 1));

var RANDOM_SEED = 42; // fixed seed so the subsamples are reproducible across runs

// Surface-reflectance bands to export. B10 is intentionally excluded: it is
// an L1C-only cirrus-detection band and does not exist in the L2A (SR)
// product this script uses.
var S2_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'];

var EXPORT_FOLDER = 'carbon_stock_estimation';
var EXPORT_FILE_PREFIX = 'gedi_l4a_s2_training_gujarat_maharashtra';

// These two are also cheap: REGION is a small client-side geometry, and
// the rest are plain JS config values — neither touches the GEDI shot
// collection or triggers any server computation. (The GEDI band-names
// print above is the third; the TEST_MODE row-count print near the export
// is the fourth and only conditional one — see PERFORMANCE NOTE.)
print('Region:', REGION);
print('Config — biomass threshold (Mg/ha):', BIOMASS_THRESHOLD_MG_HA,
  '| low-tier sample size:', LOW_BIOMASS_SAMPLE_SIZE,
  '| high-tier sample size:', HIGH_BIOMASS_SAMPLE_SIZE,
  '| high-tier bands (Mg/ha):', HIGH_BIOMASS_BAND_EDGES,
  '| season years:', SEASON_YEARS);

// =============================================================================
// 1. GEDI L4A footprint-level shots, quality-filtered
// =============================================================================

// Keep only quality shots (spec: l4_quality_flag == 1, degrade_flag == 0).
// No date band anymore — lat_lowestmode/lon_lowestmode/shot_date_millis
// were all invalid on this gridded MONTHLY raster (see the band-names
// print above and the CONFIG comment), and shot_date isn't needed for
// anything now that per-shot time-matching is gone (see section 2).
function qualityMask(image) {
  var quality = image.select('l4_quality_flag').eq(1)
    .and(image.select('degrade_flag').eq(0));
  return image.updateMask(quality);
}

var gediRaw = ee.ImageCollection(GEDI_COLLECTION_ID)
  .filterBounds(REGION)
  .map(qualityMask);

// sensitivity/landsat_treecover/pft_class are commented out until
// verified against the 'GEDI monthly bands' print above — re-enable
// (and add back to exportColumns in section 4) once confirmed present.
var gediMosaic = gediRaw
  .select([
    'agbd', 'agbd_se'
    // , 'sensitivity', 'landsat_treecover', 'pft_class'
  ])
  .mosaic();

// GEDI footprints are rasterized at ~25 m in this asset with all in-between
// pixels masked out, so sampling on the native grid and dropping masked
// pixels recovers exactly the quality shot list (one feature per footprint).
var gediShots = gediMosaic.sample({
  region: REGION,
  scale: 25,
  geometries: true,
  tileScale: 16
});

// lat/lon come from each sampled feature's own geometry (populated because
// geometries: true above), not from nonexistent lat_lowestmode/
// lon_lowestmode bands.
gediShots = gediShots.map(function(f) {
  var c = f.geometry().coordinates();
  return f.set({
    lon: c.get(0),
    lat: c.get(1)
  });
});

// No shot-count/date-range checkpoint here anymore: .size() and any
// reduce() over gediShots (~3.6M features) forces a full server
// computation for the Console and is what was hanging the Code Editor.
// Check shot counts and date coverage from the exported CSV instead (see
// PERFORMANCE NOTE at the top of this file).

// -----------------------------------------------------------------------
// Stratified sampling — fixes the region's biomass imbalance (see header
// and CONFIG): subsample the low-biomass majority, and cap each
// high-biomass band separately so dense-canopy shots aren't crowded out
// by the far more common 20-40 Mg/ha band.
//
// lowOnlyShots (agbd < BIOMASS_THRESHOLD_MG_HA) and highOnlyShots
// (agbd >= BIOMASS_THRESHOLD_MG_HA) partition the quality-filtered shots
// exactly, with no gap and no overlap at the threshold. The low subsample
// and every high band are filtered from these two disjoint collections
// (never from the unsplit gediShots), so no single shot can ever be drawn
// into both tiers.
// -----------------------------------------------------------------------
var lowOnlyShots = gediShots.filter(ee.Filter.lt('agbd', BIOMASS_THRESHOLD_MG_HA));
var highOnlyShots = gediShots.filter(ee.Filter.gte('agbd', BIOMASS_THRESHOLD_MG_HA));

var lowBiomassShots = lowOnlyShots
  .randomColumn('random', RANDOM_SEED)
  .sort('random')
  .limit(LOW_BIOMASS_SAMPLE_SIZE);

var highBiomassBands = [];
for (var i = 0; i < HIGH_BIOMASS_BAND_EDGES.length - 1; i++) {
  var bandLo = HIGH_BIOMASS_BAND_EDGES[i];
  var bandHi = HIGH_BIOMASS_BAND_EDGES[i + 1];
  var bandFilter = isFinite(bandHi)
    ? ee.Filter.and(ee.Filter.gte('agbd', bandLo), ee.Filter.lt('agbd', bandHi))
    : ee.Filter.gte('agbd', bandLo);
  var bandShots = highOnlyShots
    .filter(bandFilter)
    .randomColumn('random', RANDOM_SEED + i) // vary seed per band
    .sort('random')
    .limit(HIGH_BAND_TARGET);
  // No per-band print here: .size() on these filtered collections still
  // forces a server computation per band, which is exactly what was
  // hanging the Code Editor. Inspect the sampled distribution from the
  // exported CSV instead (see PERFORMANCE NOTE at the top of this file).
  highBiomassBands.push(bandShots);
}
var highBiomassShots = ee.FeatureCollection(highBiomassBands).flatten();

gediShots = highBiomassShots.merge(lowBiomassShots);

// No count/distribution prints here either — same reason. The whole point
// of this restructuring is that gediShots stays an unevaluated, lazy
// FeatureCollection all the way through the Sentinel-2 matching below; it
// only actually gets computed once, inside the export task.

// =============================================================================
// 2. Sentinel-2: cloud masking, indices, and ONE multi-year composite
// =============================================================================

// PER-YEAR TIME-ALIGNMENT REMOVED. The previous version matched each shot
// to its own year's composite via ee.Filter.eq('season_year', y), but
// season_year was a server-side ee.Number on the shots and a plain JS
// integer on the composites — the equality never matched, every year's
// join came back empty, and the export produced zero rows with correct
// headers. Rather than debug that join further, this version drops
// per-shot time-matching entirely: ALL quality shots are sampled against
// a single composite built from Oct-Dec and Jan-Mar imagery pooled across
// every year in SEASON_YEARS (2019-2023), regardless of each shot's own
// date. This trades away the time-alignment CLAUDE.md originally asked
// for in favor of a working end-to-end export; reintroduce per-year (or
// per-shot) matching later once the rest of the pipeline — sampling,
// training, validation — is proven out on real data.

// QA60-based masking was dropped: in Sentinel-2's newer processing baseline
// (roughly post-2022), QA60 is often all-zero, so clouds pass straight
// through unmasked. GEDI shots here run through March 2023 (see header),
// squarely inside that newer-baseline window, so this would have silently
// contaminated a large share of the training features with unmasked cloud
// pixels. Cloud Score+ gives a per-pixel ML-based clear-sky probability
// (cs_cdf) instead, joined to each S2 scene by system:index via
// linkCollection. (If this join ever proves awkward, the documented
// fallback is the SCL band, excluding classes 3/8/9/10/11 — cloud shadow,
// cloud medium/high probability, thin cirrus, and snow/ice.)
function maskS2Clouds(image) {
  var clearMask = image.select(CS_PLUS_BAND).gte(CS_PLUS_CLEAR_THRESHOLD);
  return image.updateMask(clearMask)
    .select(S2_BANDS)
    .multiply(0.0001) // scale digital numbers to reflectance fraction
    .copyProperties(image, ['system:time_start']);
}

function addIndices(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var evi = image.expression(
    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
      NIR: image.select('B8'),
      RED: image.select('B4'),
      BLUE: image.select('B2')
    }).rename('EVI');
  return image.addBands([ndvi, evi]);
}

// Builds a single ee.Filter that's the OR of one date range per year in
// `years`, e.g. for Oct 1 - Dec 31 across 2019-2023 that's five ranges
// OR'd together — matching any scene that falls in that month-window in
// ANY of those years, not just one.
function multiYearDateFilter(years, startMonth, startDay, endMonth, endDay) {
  var yearFilters = years.map(function(y) {
    return ee.Filter.date(
      ee.Date.fromYMD(y, startMonth, startDay),
      ee.Date.fromYMD(y, endMonth, endDay).advance(1, 'day') // filterDate's end is exclusive
    );
  });
  return ee.Filter.or.apply(null, yearFilters);
}

var POST_MONSOON_FILTER = multiYearDateFilter(SEASON_YEARS, 10, 1, 12, 31);
var DRY_SEASON_FILTER = multiYearDateFilter(SEASON_YEARS, 1, 1, 3, 31);

// No scene-count print here: a live .size() on the filtered S2 collection
// is exactly the kind of interactive cost this script avoids everywhere
// else (see PERFORMANCE NOTE at the top). If either season effectively has
// no clear scenes across all five years pooled together, that would show
// up as near-empty NDVI/EVI/reflectance columns in the exported CSV.
function seasonalMedian(region, dateFilter) {
  var s2 = ee.ImageCollection(S2_COLLECTION_ID)
    .filterBounds(region)
    .filter(dateFilter);
  var csPlus = ee.ImageCollection(CS_PLUS_COLLECTION_ID);
  var s2WithCs = s2.linkCollection(csPlus, [CS_PLUS_BAND]);

  return s2WithCs.map(maskS2Clouds).median();
}

// var canopyHeight = ee.Image(CANOPY_HEIGHT_ASSET_ID).select(0).rename('canopy_height');
// ^ disabled — see TODO in the CONFIG section above.

var postMonsoon = addIndices(seasonalMedian(REGION, POST_MONSOON_FILTER));
var dry = addIndices(seasonalMedian(REGION, DRY_SEASON_FILTER));

var postMonsoonIndices = postMonsoon.select(['NDVI', 'EVI'], ['ndvi_postmonsoon', 'evi_postmonsoon']);
var dryIndices = dry.select(['NDVI', 'EVI'], ['ndvi_dryseason', 'evi_dryseason']);
// Reflectance bands are taken from the dry-season composite only (clearer
// atmosphere, fewer monsoon-residual clouds) to avoid exporting the same
// eleven bands twice under two names.
var dryReflectance = dry.select(S2_BANDS);

// canopyHeight would be .addBands() here too if re-enabled — see TODO in
// the CONFIG section above.
var seasonalComposite = postMonsoonIndices
  .addBands(dryIndices)
  .addBands(dryReflectance);

// =============================================================================
// 3. Sample the single multi-year composite at every stratified GEDI shot
// =============================================================================

var training = seasonalComposite.sampleRegions({
  collection: gediShots,
  scale: 10,
  geometries: true,
  tileScale: 16
});

// =============================================================================
// 4. Export training CSV to Drive
// =============================================================================

// The ONE interactive evaluation this script allows, and only in
// TEST_MODE: capped with .limit(500) first, so it's cheap regardless of
// how many rows the small test region would otherwise produce, and it
// never runs against the full region. This is a deliberate exception to
// the PERFORMANCE NOTE at the top of the file — everywhere else, no
// .size()/.getInfo() on GEDI-derived collections. For the real regional
// export, set TEST_MODE = false above; this block then does nothing (no
// print, no evaluation) and can be left in place or deleted.
if (TEST_MODE) {
  print('TEST row count:', training.limit(500).size());
}

// sensitivity/landsat_treecover/pft_class dropped from selectors along with
// the .select() in section 1 above — add back together once verified.
var exportColumns = ['agbd', 'agbd_se', 'lat', 'lon',
    // 'sensitivity', 'landsat_treecover', 'pft_class',
    'ndvi_postmonsoon', 'evi_postmonsoon', 'ndvi_dryseason', 'evi_dryseason']
  .concat(S2_BANDS);

Export.table.toDrive({
  collection: training,
  description: 'gedi_l4a_s2_training_data',
  folder: EXPORT_FOLDER,
  fileNamePrefix: EXPORT_FILE_PREFIX,
  fileFormat: 'CSV',
  selectors: exportColumns
});
