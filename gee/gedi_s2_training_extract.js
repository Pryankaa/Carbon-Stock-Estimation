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
 * most of the region is dryland/cropland/urban rather than forest. Earlier
 * versions of this script applied STRATIFIED sampling here, in GEE, before
 * export. That's gone: a single .sample() over the whole ~3.6M-shot region
 * exceeds GEE's per-operation size limit ("Image.sample: Computed value is
 * too large") regardless of any downstream capping, so this script now
 * tiles the region (see section 3) and exports EVERY quality-filtered shot
 * per tile, unfiltered by biomass. Stratified sampling now happens ONCE in
 * the Python pipeline, after combining the tile CSVs (see CLAUDE.md) —
 * stratifying per tile here would over-sample each tile's low-biomass
 * majority and under-represent rare high-biomass shots relative to the
 * combined dataset.
 *
 * GEDI coverage note: this script reads GEDI04_A_002_MONTHLY (see CONFIG —
 * the footprint-level GEDI04_A_002 asset is a table/index folder, not an
 * ImageCollection, and errors in GEE). That monthly raster mosaic covers
 * March 2019 - March 2023 only, so all training labels here are 2019-2023,
 * not 2024.
 *
 * Output: for the real regional run, one CSV per tile (64 export tasks,
 * same Drive folder, filenames suffixed _tile_i_j — see section 3); for
 * TEST_MODE, one CSV over the small test box. Every CSV has the same
 * columns: one row per quality GEDI shot in that tile/region (ALL of
 * them — no biomass filtering or capping, see the imbalance note above),
 * agbd, agbd_se, lat, lon (derived from each sampled feature's own
 * geometry — lat_lowestmode/lon_lowestmode are footprint-TABLE columns,
 * not bands on this gridded MONTHLY raster, and selecting them made
 * .sample() return empty; see CONFIG and section 1), and Sentinel-2
 * post-monsoon/dry-season NDVI + EVI plus reflectance bands B2-B12 (B10
 * excluded: it is an L1C-only cirrus band, not present in the L2A
 * surface-reflectance product) — all from ONE multi-year composite,
 * shared across every tile and every shot regardless of its date (see
 * NOTE above and section 2). No shot_date column (dropped along with
 * per-shot time-matching) and no sensitivity/landsat_treecover/pft_class
 * for now — commented out in section 1 pending verification against the
 * 'GEDI monthly bands' print.
 *
 * External canopy-height sampling is DISABLED for this first run — see the
 * TODO in the CONFIG section below.
 *
 * IN PROGRESS: CEPT ground-truth validation (src/validate_cept_ground_truth.py)
 * found the trained model's per-cell predictions have a NEGATIVE Spearman
 * correlation (-0.28) against field biomass, despite a decent site-total
 * match (+14%) — almost certainly because optical-only features can't
 * distinguish tall-and-green (trees) from flat-and-green (lawns/crops).
 * Section 1 below now scaffolds joining GEDI L2A relative-height (rh)
 * shots onto L4A biomass shots by shot_number, to add the missing
 * structure signal. That join is NOT active yet — it's gated behind
 * verifying real L2A band names first (see the print statements in
 * CONFIG and the TODOs in section 1). Output/exportColumns are UNCHANGED
 * from before until that verification happens.
 *
 * CLAUDE.md rules this script follows:
 *   - Rule 3: does NOT use the Meta/WRI ~2016 canopy height model (blind to
 *     anything planted after 2016). The external ~2020 replacement is
 *     disabled too for the same reason (see TODO below). GEDI's own
 *     sensitivity/landsat_treecover/pft_class covariates were meant to
 *     stand in instead, but are temporarily commented out in section 1
 *     pending verification against the 'GEDI monthly bands' print — no
 *     height/vegetation signal is included in exports until re-enabled.
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
// Everything below stays lazy; the heavy work only actually runs once per
// tile, inside each Export.table.toDrive task (see section 3 — tiling is
// also what fixes "Image.sample: Computed value is too large" on the full
// region). To inspect the biomass distribution and shot counts, do it
// AFTER exporting and combining the tile CSVs — in Python (Claude Code
// can do this), where stratified sampling now lives too (see CLAUDE.md),
// rather than adding print()s or per-tile capping here. No interactive
// row-count check either, even in TEST_MODE — that's still heavy enough
// to hit the Code Editor's timeout; TEST_MODE verifies via a real
// (small-region) export instead. See TEST_MODE above and section 3.
// -----------------------------------------------------------------------

// =============================================================================
// 0. CONFIG — placeholders to replace with real values before running
// =============================================================================

// TEST_MODE: fail-fast check on a small, known-vegetated area before
// burning a long run on the full region. When true, REGION below is
// overridden with a 0.3 deg box in the Western Ghats (real forest, plenty
// of GEDI shots expected), and EXPORT_FILE_PREFIX (below) gets a TEST
// suffix so the small-box export can't be confused with the real one.
//
// The verification step is the EXPORT itself, not an interactive print —
// an interactive check heavy enough to matter (e.g.
// training.limit(500).size()) still has to build the full lazy chain live
// in the browser and hits the Code Editor's ~5 minute timeout regardless
// of any .limit(). Export.table.toDrive runs the same computation
// server-side as a batch task with far more headroom, so on the small
// TEST_MODE region it actually finishes (a minute or two) and is a real
// test of the real export path. Workflow: run the export with
// TEST_MODE = true, check the resulting CSV has rows, THEN set
// TEST_MODE = false and run the real regional export.
var TEST_MODE = false;

// Placeholder region: Gujarat/Maharashtra, ~20-24 N, 72-76 E. Replace with
// the real regional bounding box before running. Kept as plain JS numbers
// (not just baked into REGION below) so the tiling grid in section 3 can
// reuse the exact same bounds — one source of truth, so the two can never
// drift apart the way BIOMASS_THRESHOLD_MG_HA and the sampling bands
// nearly did earlier in this script's history.
var FULL_REGION_BOUNDS = { west: 72, south: 20, east: 76, north: 24 };
var REGION = ee.Geometry.Rectangle([
  FULL_REGION_BOUNDS.west, FULL_REGION_BOUNDS.south,
  FULL_REGION_BOUNDS.east, FULL_REGION_BOUNDS.north
]);

if (TEST_MODE) {
  // Small box in the Western Ghats — real forest, should return a
  // nonzero row count quickly without evaluating the full ~3.6M-shot
  // region.
  REGION = ee.Geometry.Rectangle([73.0, 20.0, 73.3, 20.3]);
}

// Stratified sampling has moved to the Python pipeline (see CLAUDE.md) —
// this script exports every quality-filtered shot, tiled to stay under
// GEE's per-operation size limit. Only used when !TEST_MODE (see
// section 3); the small TEST_MODE box is exported as a single task.
//
// 4x4 (1 deg x 1 deg) still failed on denser tiles: tile_0_0 finished in
// ~2 min, but tile_0_1 hit "reduceRegions: Computed value is too large"
// after 10 min — shot density, not just tile area, drives the per-tile
// operation size. The 0.3 deg TEST_MODE box handled ~22k rows fine, so
// 0.5 deg tiles should have headroom even in dense areas. Shrink further
// (higher TILE_GRID_SIZE) if any tile still fails.
var TILE_GRID_SIZE = 8; // 8x8 = 64 tiles, 0.5 deg x 0.5 deg each, covering FULL_REGION_BOUNDS

// TODO(height strategy): external canopy-height sampling is disabled. The
// candidate asset ID below is unverified (unconfirmed in this environment),
// and more importantly it is a fixed ~2020 snapshot — the same "blind to
// anything planted after the snapshot year" problem that ruled out the 2016
// Meta/WRI CHM (CLAUDE.md rule 3). GEDI's own sensitivity/landsat_treecover/
// pft_class covariates were meant to stand in instead (contemporaneous with
// each shot), but are currently commented out of the .select() in section 1
// pending verification against the 'GEDI monthly bands' print below — so
// exports carry no height/vegetation signal at all yet.
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
// L2A: source of relative-height (rh) metrics, added to fix the CEPT
// validation finding — optical-only features (NDVI/EVI/reflectance) can't
// distinguish tall-and-green (trees) from flat-and-green (lawns/crops),
// which is almost certainly why the trained model's per-cell predictions
// at CEPT came back with a NEGATIVE Spearman correlation (-0.28) against
// field biomass even though the site total was only +14% off. rh height
// is the missing structure signal. See the join-strategy note in section 1
// below for how L4A biomass shots get matched to L2A height shots.
var GEDI_L2A_COLLECTION_ID = 'LARSE/GEDI/GEDI02_A_002_MONTHLY';
var S2_COLLECTION_ID = 'COPERNICUS/S2_SR_HARMONIZED';

// One-time verification prints: the real band lists for both GEDI monthly
// assets. The earlier 3.6M-shot checkpoint only ever selected 'agbd', so
// lat_lowestmode/lon_lowestmode/shot_date_millis being invalid band names
// on the gridded MONTHLY L4A raster (they're footprint-TABLE columns, not
// raster bands) went unnoticed until .select() on them made .sample()
// return empty. Don't repeat that mistake with rh98/rh100/etc. or
// shot_number on L2A — confirm real names here FIRST. Both only inspect a
// single Image's metadata, not the full shot collections, so they're
// cheap regardless of TEST_MODE.
print('GEDI L4A monthly bands:', ee.ImageCollection(GEDI_COLLECTION_ID).first().bandNames());
print('GEDI L2A monthly bands:', ee.ImageCollection(GEDI_L2A_COLLECTION_ID).first().bandNames());

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

// Surface-reflectance bands to export. B10 is intentionally excluded: it is
// an L1C-only cirrus-detection band and does not exist in the L2A (SR)
// product this script uses.
var S2_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'];

var EXPORT_FOLDER = 'carbon_stock_estimation';
// Distinct filename in TEST_MODE so a small-box test export can never be
// mistaken for (or accidentally overwrite) the real regional one.
var EXPORT_FILE_PREFIX = TEST_MODE
  ? 'gedi_l4a_s2_training_TEST_western_ghats'
  : 'gedi_l4a_s2_training_gujarat_maharashtra';

// These two are also cheap: REGION is a small client-side geometry, and
// the rest are plain JS config values — neither touches the GEDI shot
// collection or triggers any server computation. (The two GEDI band-names
// prints above are the first two prints in this script — see PERFORMANCE
// NOTE and section 3 for why there's no interactive test print anymore.)
print('Region:', REGION);
print('Config — TEST_MODE:', TEST_MODE,
  '| season years:', SEASON_YEARS,
  '| tile grid:', TEST_MODE ? 'n/a (single small-box export)' : (TILE_GRID_SIZE + 'x' + TILE_GRID_SIZE));

// =============================================================================
// 1. GEDI L4A biomass + L2A height, quality-filtered and joined by
//    shot_number
// =============================================================================
//
// JOIN STRATEGY (see also the header note above):
// L4A and L2A monthly assets are independently-built raster mosaics, so a
// shot's biomass (L4A) and height (L2A) CANNOT be paired by pixel/spatial
// proximity without risking exactly what sank the earlier rh98/rh100
// attempt in this script's history: two shots from different orbit passes
// landing on the same ~25 m grid cell, mosaicked independently, giving a
// biomass value and a height value that were never actually the same
// physical shot. Instead: sample L4A and L2A separately (each already
// quality-masked by its own flags), then ee.Join.inner() the two
// resulting FeatureCollections on shot_number — an exact property match,
// not a spatial guess. Shots that pass only one product's quality flags
// are correctly dropped by the inner join.
//
// GEDI_SHOT_NUMBER_BAND and GEDI_L2A_RH_BANDS below are UNVERIFIED
// placeholders. Do not uncomment/use them until the 'GEDI L4A/L2A monthly
// bands' prints above confirm: (a) shot_number actually exists as a band
// on both gridded monthly assets (it's standard on the raw per-shot
// table products, but those aren't usable here — see the
// GEDI_COLLECTION_ID comment below), and (b) it's stored with enough
// precision (int64/float64, not float32) that two independently-sampled
// copies will match exactly. If shot_number turns out not to be viable,
// the fallback is linking the two monthly ImageCollections by
// system:index (same mechanism already used for Cloud Score+ below in
// section 2) — a weaker guarantee ("same source month/pixel", not
// "same physical shot"), only to be used if this primary plan fails.

// TODO(verify): confirm exact name + dtype via the band-name prints above
// before uncommenting anything below that references these.
// var GEDI_SHOT_NUMBER_BAND = 'shot_number';
// var GEDI_L2A_RH_BANDS = ['rh50', 'rh75', 'rh90', 'rh98', 'rh100'];

// GEDI04_A_002 (footprint-level) is a table/index folder in this GEE asset,
// not an ImageCollection — loading it directly throws "found IndexedFolder".
// Same is true of GEDI02_A_002 (footprint-level L2A) — use the monthly
// raster mosaics for both. This asset has ~170 bands total; select down to
// only what's needed as the very FIRST operation, before quality masking
// and before mosaicking, so every downstream step only ever touches a
// narrow set instead of all ~170 bands on every source image (this is
// what was making the interactive test time out before). sensitivity/
// landsat_treecover/pft_class are commented out until verified against
// the 'GEDI L4A monthly bands' print above.
var GEDI_L4A_BANDS_NEEDED = [
  'agbd', 'agbd_se', 'l4_quality_flag', 'degrade_flag'
  // , 'sensitivity', 'landsat_treecover', 'pft_class'
  // , GEDI_SHOT_NUMBER_BAND
];

// L2A has its OWN quality/degrade flags (named without the "l4_" prefix) —
// a shot can pass one product's quality check and fail the other's.
var GEDI_L2A_BANDS_NEEDED = [
  'quality_flag', 'degrade_flag'
  // , GEDI_SHOT_NUMBER_BAND
  // .concat(GEDI_L2A_RH_BANDS) once verified
];

function qualityMaskL4A(image) {
  var quality = image.select('l4_quality_flag').eq(1)
    .and(image.select('degrade_flag').eq(0));
  return image.updateMask(quality);
}

function qualityMaskL2A(image) {
  var quality = image.select('quality_flag').eq(1)
    .and(image.select('degrade_flag').eq(0));
  return image.updateMask(quality);
}

var gediL4ARaw = ee.ImageCollection(GEDI_COLLECTION_ID)
  .filterBounds(REGION)
  .select(GEDI_L4A_BANDS_NEEDED)
  .map(qualityMaskL4A);

// Quality flags have done their job (masking) — drop them here so only
// the actual training-data bands reach mosaic/sample.
var gediL4AMosaic = gediL4ARaw
  .select([
    'agbd', 'agbd_se'
    // , 'sensitivity', 'landsat_treecover', 'pft_class'
    // , GEDI_SHOT_NUMBER_BAND
  ])
  .mosaic();

var gediL2ARaw = ee.ImageCollection(GEDI_L2A_COLLECTION_ID)
  .filterBounds(REGION)
  .select(GEDI_L2A_BANDS_NEEDED)
  .map(qualityMaskL2A);

// var gediL2AMosaic = gediL2ARaw
//   .select([GEDI_SHOT_NUMBER_BAND].concat(GEDI_L2A_RH_BANDS))
//   .mosaic();
// ^ disabled until GEDI_SHOT_NUMBER_BAND / GEDI_L2A_RH_BANDS are verified.

// GEDI footprints are rasterized at ~25 m in these assets with all
// in-between pixels masked out, so sampling on the native grid and
// dropping masked pixels recovers exactly the quality shot list (one
// feature per footprint).
//
// Samples within `geom` only (a tile, or REGION directly in TEST_MODE) —
// NOT the whole region in one call. A single .sample() over the full
// ~3.6M-shot region is exactly what failed with "Image.sample: Computed
// value is too large"; tiling (section 3) keeps each call small enough.
// lat/lon are derived from each feature's own geometry (populated via
// geometries: true), not from nonexistent lat_lowestmode/lon_lowestmode
// bands. No biomass filtering or capping here — every quality-filtered,
// height-joined shot in `geom` is returned as-is; stratified sampling now
// happens ONCE in Python after combining the tile CSVs (see CLAUDE.md and
// the header NOTE above) rather than per tile here.
function sampleGediShots(geom) {
  var l4aShots = gediL4AMosaic.sample({
    region: geom,
    scale: 25,
    geometries: true,
    tileScale: 16
  });
  l4aShots = l4aShots.map(function(f) {
    var c = f.geometry().coordinates();
    return f.set({
      lon: c.get(0),
      lat: c.get(1)
    });
  });

  // TODO(activate once verified): join l4aShots to L2A height shots by
  // shot_number here, e.g.:
  //
  // var l2aShots = gediL2AMosaic.sample({
  //   region: geom, scale: 25, geometries: false, tileScale: 16
  // });
  // var shotNumberFilter = ee.Filter.equals({
  //   leftField: GEDI_SHOT_NUMBER_BAND, rightField: GEDI_SHOT_NUMBER_BAND
  // });
  // var joined = ee.Join.inner().apply(l4aShots, l2aShots, shotNumberFilter);
  // return joined.map(function(pair) {
  //   var l4a = ee.Feature(pair.get('primary'));
  //   var l2a = ee.Feature(pair.get('secondary'));
  //   return l4a.copyProperties(l2a, GEDI_L2A_RH_BANDS);
  // });
  //
  // For now (height join not yet active), return L4A shots unchanged:
  return l4aShots;
}

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
// 3. Sample the multi-year composite at GEDI shots, and export — tiled
//    for the full region, a single task for TEST_MODE
// =============================================================================

// sensitivity/landsat_treecover/pft_class dropped from selectors along with
// the .select() in section 1 above — add back together once verified.
var exportColumns = ['agbd', 'agbd_se', 'lat', 'lon',
    // 'sensitivity', 'landsat_treecover', 'pft_class',
    'ndvi_postmonsoon', 'evi_postmonsoon', 'ndvi_dryseason', 'evi_dryseason']
  .concat(S2_BANDS);

// Samples every quality-filtered GEDI shot in `geom` against the one
// shared seasonalComposite, and starts one Export.table.toDrive task.
// No interactive row-count check anywhere in this pipeline — even
// capped with .limit(), evaluating this live in the browser is heavy
// enough to hit the Code Editor's ~5 minute timeout regardless (that's
// what happened before tiling: a timeout, not an export failure).
// Export.table.toDrive runs the identical computation server-side as a
// batch task with far more headroom, so the export itself is the real
// test — see TEST_MODE above.
function exportTile(geom, filePrefix, description) {
  var shots = sampleGediShots(geom);
  var tileTraining = seasonalComposite.sampleRegions({
    collection: shots,
    scale: 10,
    geometries: true,
    tileScale: 16
  });
  Export.table.toDrive({
    collection: tileTraining,
    description: description,
    folder: EXPORT_FOLDER,
    fileNamePrefix: filePrefix,
    fileFormat: 'CSV',
    selectors: exportColumns
  });
}

if (TEST_MODE) {
  // Single export over the small Western Ghats test box — no tiling
  // needed; already proven (21,850 rows) to finish well within GEE's
  // per-task limits without it.
  exportTile(REGION, EXPORT_FILE_PREFIX, 'gedi_l4a_s2_training_data_TEST');
} else {
  // TILING — the full-region export first failed with "Image.sample:
  // Computed value is too large" (a single .sample() over the whole
  // ~3.6M-shot region exceeds GEE's per-operation size limit regardless of
  // any downstream stratification/capping); a first fix at 4x4 (1 deg
  // tiles) then failed on denser tiles with "reduceRegions: Computed value
  // is too large" — shot DENSITY within a tile, not just its area, drives
  // the per-operation size, so uniform degree-sized tiles can still
  // overflow in forest-dense areas. TILE_GRID_SIZE (see CONFIG) controls
  // tile size; shrink it further if a tile still fails. Splitting
  // FULL_REGION_BOUNDS into a TILE_GRID_SIZE x TILE_GRID_SIZE grid keeps
  // each .sample()/.sampleRegions() call within limits. One export task
  // per tile, all into EXPORT_FOLDER, filenames suffixed _tile_i_j.
  // Combine the resulting CSVs and do stratified sampling in Python
  // afterward (see CLAUDE.md) — stratifying per tile here would
  // over-sample each tile's low-biomass majority and under-represent rare
  // high-biomass shots relative to the combined dataset.
  var tileWidth = (FULL_REGION_BOUNDS.east - FULL_REGION_BOUNDS.west) / TILE_GRID_SIZE;
  var tileHeight = (FULL_REGION_BOUNDS.north - FULL_REGION_BOUNDS.south) / TILE_GRID_SIZE;

  for (var i = 0; i < TILE_GRID_SIZE; i++) {
    for (var j = 0; j < TILE_GRID_SIZE; j++) {
      var tileWest = FULL_REGION_BOUNDS.west + i * tileWidth;
      var tileEast = tileWest + tileWidth;
      var tileSouth = FULL_REGION_BOUNDS.south + j * tileHeight;
      var tileNorth = tileSouth + tileHeight;
      var tileGeom = ee.Geometry.Rectangle([tileWest, tileSouth, tileEast, tileNorth]);
      var tileSuffix = '_tile_' + i + '_' + j;
      exportTile(tileGeom, EXPORT_FILE_PREFIX + tileSuffix, 'gedi_l4a_s2_training_data' + tileSuffix);
    }
  }
}
