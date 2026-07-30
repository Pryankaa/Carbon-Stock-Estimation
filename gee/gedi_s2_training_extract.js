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
 * Output: one CSV row per sampled quality GEDI shot, columns = agbd,
 * agbd_se, GEDI shot date, lat, lon, GEDI's own covariates (sensitivity,
 * landsat_treecover, pft_class — already in L4A, standing in for an
 * external height layer for now), Sentinel-2 post-monsoon/dry-season
 * NDVI + EVI, and post-monsoon/dry-season-matched reflectance bands B2-B12
 * (B10 excluded: it is an L1C-only cirrus band, not present in the L2A
 * surface-reflectance product).
 *
 * External canopy-height sampling is DISABLED for this first run — see the
 * TODO in the CONFIG section below.
 *
 * CLAUDE.md rules this script follows:
 *   - Rule 3: does NOT use the Meta/WRI ~2016 canopy height model (blind to
 *     anything planted after 2016). The external ~2020 replacement is
 *     disabled too for the same reason (see TODO below); GEDI's own
 *     sensitivity/landsat_treecover/pft_class covariates are used as the
 *     current height/vegetation signal instead.
 *   - Rule 6: does NOT add Sentinel-1 SAR or optical GLCM texture back in.
 *   - GEDI is explicitly treated as regional training data here, never as a
 *     per-site estimator (see header above and the immediate-next-task note
 *     in CLAUDE.md about validating GEDI before trusting it elsewhere).
 */

// =============================================================================
// 0. CONFIG — placeholders to replace with real values before running
// =============================================================================

// Placeholder region: Gujarat/Maharashtra, ~20-24 N, 72-76 E.
// Replace with the real regional bounding box before running.
var REGION = ee.Geometry.Rectangle([72, 20, 76, 24]);

// TODO(height strategy): external canopy-height sampling is disabled. The
// candidate asset ID below is unverified (unconfirmed in this environment),
// and more importantly it is a fixed ~2020 snapshot — the same "blind to
// anything planted after the snapshot year" problem that ruled out the 2016
// Meta/WRI CHM (CLAUDE.md rule 3). GEDI's own sensitivity/landsat_treecover/
// pft_class covariates (selected directly from L4A below) stand in for now,
// since they're contemporaneous with each shot. Revisit once we know
// whether these covariates carry enough height/structure signal on their
// own, or whether a genuinely current external height product is needed.
// var CANOPY_HEIGHT_ASSET_ID =
//   'projects/sat-io/open-datasets/ETH_GlobalCanopyHeight_2020_10m_v1';

var GEDI_COLLECTION_ID = 'LARSE/GEDI/GEDI04_A_002';
var S2_COLLECTION_ID = 'COPERNICUS/S2_SR_HARMONIZED';

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

// =============================================================================
// 1. GEDI L4A footprint-level shots, quality-filtered
// =============================================================================

// Keep only quality shots (spec: l4_quality_flag == 1, degrade_flag == 0),
// and stamp each pixel with its source granule's acquisition date so that,
// after mosaicking many orbits together, every surviving pixel still knows
// exactly when it was collected.
function qualityMaskAndDate(image) {
  var quality = image.select('l4_quality_flag').eq(1)
    .and(image.select('degrade_flag').eq(0));
  var dateBand = ee.Image.constant(image.date().millis())
    .rename('shot_date_millis')
    .toDouble();
  return image.updateMask(quality).addBands(dateBand);
}

var gediRaw = ee.ImageCollection(GEDI_COLLECTION_ID)
  .filterBounds(REGION)
  .map(qualityMaskAndDate);

// GEDI covariates already present in L4A itself (sensitivity,
// landsat_treecover, pft_class) stand in for an external height layer for
// now — no separate L2A join needed.
var gediMosaic = gediRaw
  .select(['agbd', 'agbd_se', 'lat_lowestmode', 'lon_lowestmode', 'shot_date_millis',
    'sensitivity', 'landsat_treecover', 'pft_class'])
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

// Tag each shot with the "season year" its matched Sentinel-2 window will
// use: shots from Jul-Dec belong to that calendar year's post-monsoon/dry
// pair; shots from Jan-Jun belong to the PREVIOUS year's pair (whose dry
// season runs into Jan-Mar of the shot's own year). Either way the matched
// window stays within about 6 months of the shot, per spec.
gediShots = gediShots.map(function(f) {
  var d = ee.Date(f.get('shot_date_millis'));
  var month = d.get('month');
  var year = d.get('year');
  var seasonYear = ee.Algorithms.If(month.gte(7), year, year.subtract(1));
  return f.set({
    shot_date: d.format('YYYY-MM-dd'),
    lat: f.get('lat_lowestmode'),
    lon: f.get('lon_lowestmode'),
    season_year: seasonYear
  });
});

// -----------------------------------------------------------------------
// CHECKPOINT — inspect these two prints before proceeding to the (much
// more expensive) Sentinel-2 matching and export below.
// -----------------------------------------------------------------------
print('Total quality GEDI L4A shots in region:', gediShots.size());

var shotDatesMillis = gediShots.aggregate_array('shot_date_millis');
var minShotDate = ee.Date(shotDatesMillis.reduce(ee.Reducer.min()));
var maxShotDate = ee.Date(shotDatesMillis.reduce(ee.Reducer.max()));
print('GEDI shot date range:',
  minShotDate.format('YYYY-MM-dd'), 'to', maxShotDate.format('YYYY-MM-dd'));

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
  print('High-biomass band ' + bandLo + '-' + (isFinite(bandHi) ? bandHi : 'inf') +
    ' Mg/ha sampled to (of target ' + HIGH_BAND_TARGET + '):', bandShots.size());
  highBiomassBands.push(bandShots);
}
var highBiomassShots = ee.FeatureCollection(highBiomassBands).flatten();

gediShots = highBiomassShots.merge(lowBiomassShots);

print('Low-biomass shots subsampled to:', lowBiomassShots.size());
print('High-biomass shots kept (all bands):', highBiomassShots.size());
print('Total shots after stratified sampling:', gediShots.size());

// -----------------------------------------------------------------------
// Final training-set biomass distribution — a shape check before the
// (much more expensive) Sentinel-2 matching and export below.
// -----------------------------------------------------------------------
var DISTRIBUTION_BAND_EDGES = [0, 5, 20, 40, 70, 120, Infinity]; // Mg/ha
for (var d = 0; d < DISTRIBUTION_BAND_EDGES.length - 1; d++) {
  var distLo = DISTRIBUTION_BAND_EDGES[d];
  var distHi = DISTRIBUTION_BAND_EDGES[d + 1];
  var distFilter = isFinite(distHi)
    ? ee.Filter.and(ee.Filter.gte('agbd', distLo), ee.Filter.lt('agbd', distHi))
    : ee.Filter.gte('agbd', distLo);
  var distLabel = distLo + '-' + (isFinite(distHi) ? distHi : '+') + ' Mg/ha';
  print('Final training shots, ' + distLabel + ':', gediShots.filter(distFilter).size());
}

// =============================================================================
// 2. Sentinel-2: cloud masking, indices, and time-matched seasonal composites
// =============================================================================

// QA60-based masking was dropped: in Sentinel-2's newer processing baseline
// (roughly post-2022), QA60 is often all-zero, so clouds pass straight
// through unmasked. Since GEDI shots run to 2024, that would have silently
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

// label identifies the seasonal window in the Console (e.g. "2021
// post-monsoon"). Prints the scene count BEFORE compositing/masking, so an
// empty window is visible immediately rather than silently producing a
// blank (fully masked) composite that would drop that year's shots from
// the export with no warning.
function seasonalMedian(region, start, end, label) {
  var s2 = ee.ImageCollection(S2_COLLECTION_ID)
    .filterBounds(region)
    .filterDate(start, end);
  var csPlus = ee.ImageCollection(CS_PLUS_COLLECTION_ID);
  var s2WithCs = s2.linkCollection(csPlus, [CS_PLUS_BAND]);

  print('Sentinel-2 scenes, ' + label + ':', s2WithCs.size());

  return s2WithCs.map(maskS2Clouds).median();
}

// var canopyHeight = ee.Image(CANOPY_HEIGHT_ASSET_ID).select(0).rename('canopy_height');
// ^ disabled — see TODO in the CONFIG section above.

// One post-monsoon (Oct-Dec) + dry-season (Jan-Mar of the following year)
// composite pair per calendar year present in the GEDI data, rather than one
// composite per individual shot — far cheaper, and every shot still lands
// within ~6 months of its matched window (see season_year tagging above).
// y is a plain JS number (see the client-side years loop below), not an
// ee.Number — needed so the per-season scene-count prints in
// seasonalMedian actually fire once per year instead of once for the whole
// (server-side-mapped) computation graph.
function seasonalComposites(y) {
  var postMonsoonStart = ee.Date.fromYMD(y, 10, 1);
  var postMonsoonEnd = ee.Date.fromYMD(y, 12, 31);
  var dryStart = ee.Date.fromYMD(y + 1, 1, 1);
  var dryEnd = ee.Date.fromYMD(y + 1, 3, 31);

  var postMonsoon = addIndices(seasonalMedian(
    REGION, postMonsoonStart, postMonsoonEnd, y + ' post-monsoon (Oct-Dec)'));
  var dry = addIndices(seasonalMedian(
    REGION, dryStart, dryEnd, y + ' dry-season (Jan-Mar ' + (y + 1) + ')'));

  var postMonsoonIndices = postMonsoon.select(['NDVI', 'EVI'], ['ndvi_postmonsoon', 'evi_postmonsoon']);
  var dryIndices = dry.select(['NDVI', 'EVI'], ['ndvi_dryseason', 'evi_dryseason']);
  // Reflectance bands are taken from the dry-season composite only (clearer
  // atmosphere, fewer monsoon-residual clouds) to avoid exporting the same
  // eleven bands twice under two names.
  var dryReflectance = dry.select(S2_BANDS);

  return postMonsoonIndices
    .addBands(dryIndices)
    .addBands(dryReflectance)
    // .addBands(canopyHeight) // disabled — see TODO in the CONFIG section above.
    .set('season_year', y);
}

// Same Jul-Dec / Jan-Jun season-year rule as the per-shot tagging above,
// applied to the min/max shot dates to get the full range of season years
// actually present in the data.
var startYear = ee.Algorithms.If(
  minShotDate.get('month').gte(7), minShotDate.get('year'), minShotDate.get('year').subtract(1));
var endYear = ee.Algorithms.If(
  maxShotDate.get('month').gte(7), maxShotDate.get('year'), maxShotDate.get('year').subtract(1));

// Pulled down as plain JS numbers (one blocking round trip) so the loop
// below runs client-side — required for the per-season scene-count prints
// in seasonalMedian to fire once per year/season rather than once for the
// whole server-side computation graph (an ee.List.map() callback can't
// produce per-iteration Console output).
var seasonYearRange = ee.List([startYear, endYear]).getInfo();
var years = [];
for (var yr = seasonYearRange[0]; yr <= seasonYearRange[1]; yr++) {
  years.push(yr);
}

print('Season years covered:', years);

var yearlyComposites = ee.ImageCollection(years.map(seasonalComposites));

// =============================================================================
// 3. Sample Sentinel-2 at each GEDI shot, matched by year
// =============================================================================

var trainingByYear = years.map(function(y) {
  var shotsThisYear = gediShots.filter(ee.Filter.eq('season_year', y));
  var composite = ee.Image(yearlyComposites.filter(ee.Filter.eq('season_year', y)).first());
  return composite.sampleRegions({
    collection: shotsThisYear,
    scale: 10,
    geometries: true,
    tileScale: 16
  });
});

var training = ee.FeatureCollection(trainingByYear).flatten();

// =============================================================================
// 4. Export training CSV to Drive
// =============================================================================

var exportColumns = ['agbd', 'agbd_se', 'shot_date', 'lat', 'lon',
    'sensitivity', 'landsat_treecover', 'pft_class',
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
