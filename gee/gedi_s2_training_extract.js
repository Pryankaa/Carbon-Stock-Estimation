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
 * signal it needs. So this script applies STRATIFIED sampling: every shot
 * at or above a biomass threshold is kept, and the low-biomass majority is
 * randomly subsampled down to a comparable, manageable size (see CONFIG).
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

// Stratified sampling to fix the region's biomass imbalance (median ~4
// Mg/ha, 90th percentile ~39 Mg/ha — see header). Every shot at or above
// this threshold is kept in full; everything below it is randomly
// subsampled down to LOW_BIOMASS_SAMPLE_SIZE rows so near-zero shots don't
// swamp the training set.
var BIOMASS_THRESHOLD_MG_HA = 20; // high/low split; ~90th percentile is 39 Mg/ha, so this keeps essentially all real tree signal
var LOW_BIOMASS_SAMPLE_SIZE = 30000; // target row count for the random low-biomass subsample
var RANDOM_SEED = 42; // fixed seed so the low-biomass subsample is reproducible across runs

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
// and CONFIG). Keep every high-biomass shot; randomly subsample the rest.
// -----------------------------------------------------------------------
var highBiomassShots = gediShots.filter(ee.Filter.gte('agbd', BIOMASS_THRESHOLD_MG_HA));

var lowBiomassShots = gediShots
  .filter(ee.Filter.lt('agbd', BIOMASS_THRESHOLD_MG_HA))
  .randomColumn('random', RANDOM_SEED)
  .sort('random')
  .limit(LOW_BIOMASS_SAMPLE_SIZE);

gediShots = highBiomassShots.merge(lowBiomassShots);

print('High-biomass shots kept (>= ' + BIOMASS_THRESHOLD_MG_HA + ' Mg/ha):', highBiomassShots.size());
print('Low-biomass shots subsampled to:', lowBiomassShots.size());
print('Total shots after stratified sampling:', gediShots.size());

// =============================================================================
// 2. Sentinel-2: cloud masking, indices, and time-matched seasonal composites
// =============================================================================

function maskS2Clouds(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
    .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask)
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

function seasonalMedian(region, start, end) {
  return ee.ImageCollection(S2_COLLECTION_ID)
    .filterBounds(region)
    .filterDate(start, end)
    .map(maskS2Clouds)
    .median();
}

// var canopyHeight = ee.Image(CANOPY_HEIGHT_ASSET_ID).select(0).rename('canopy_height');
// ^ disabled — see TODO in the CONFIG section above.

// One post-monsoon (Oct-Dec) + dry-season (Jan-Mar of the following year)
// composite pair per calendar year present in the GEDI data, rather than one
// composite per individual shot — far cheaper, and every shot still lands
// within ~6 months of its matched window (see season_year tagging above).
function seasonalComposites(y) {
  y = ee.Number(y);
  var postMonsoonStart = ee.Date.fromYMD(y, 10, 1);
  var postMonsoonEnd = ee.Date.fromYMD(y, 12, 31);
  var dryStart = ee.Date.fromYMD(y.add(1), 1, 1);
  var dryEnd = ee.Date.fromYMD(y.add(1), 3, 31);

  var postMonsoon = addIndices(seasonalMedian(REGION, postMonsoonStart, postMonsoonEnd));
  var dry = addIndices(seasonalMedian(REGION, dryStart, dryEnd));

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
var years = ee.List.sequence(startYear, endYear);

print('Season years covered:', years);

var yearlyComposites = ee.ImageCollection(years.map(seasonalComposites));

// =============================================================================
// 3. Sample Sentinel-2 at each GEDI shot, matched by year
// =============================================================================

var trainingByYear = years.map(function(y) {
  y = ee.Number(y);
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
