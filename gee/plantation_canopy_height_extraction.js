/**
 * Botanical plantation canopy-height extraction -- Tier 1 deployment
 * validation on a real client site (not GEDI, not CEPT)
 * ======================================================================
 *
 * WHY THIS SCRIPT EXISTS
 * -----------------------
 * CEPT (see gee/cept_canopy_height_extraction.js and
 * src/validate_curve_at_cept.py) tested the GEDI-calibrated height ->
 * biomass curve (agbd = 0.5326 * rh98^1.8307, spatial-CV R^2 0.92) via a
 * published CHM, but CEPT is a manicured urban campus of heritage exotic
 * species with abnormally thick trunks -- an outlier relative to the
 * regional vegetation (dryland/cropland/plantation) the curve was
 * actually trained on. Client sites are plantations, much closer to
 * regional vegetation than CEPT, so this is a more representative test.
 *
 * This script gets canopy height for the Botanical boundary
 * (data/boundaries/Botanical_boundary.geojson): planted 2018, ~1.82 ha,
 * the oldest/largest of the four plantation sites (Ngo, Amshet,
 * Botanical, Jagadiya). Once the exported CSV is in data/raw/,
 * src/validate_curve_at_plantation.py applies the curve and reports a
 * sanity-checked site total -- there is no field census at this site, so
 * this is a behavior check, not an accuracy validation.
 *
 * HEIGHT SOURCE: ETH Global Canopy Height 2020 (Lang et al. 2022),
 * 'users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1', 10 m resolution.
 * This script prints the asset's real band names before selecting
 * anything (verify-before-select, same discipline as every GEDI
 * band-name check earlier in this project).
 *
 * IMPORTANT CAVEAT (flag, not fix), same one carried from the CEPT
 * script: the power-law curve was calibrated against GEDI's rh98; this
 * CHM measures top-of-canopy height via a different sensor/methodology.
 * Both approximate "canopy top height in meters" but are not guaranteed
 * to be numerically interchangeable.
 *
 * SECOND CAVEAT, specific to this site: the CHM is a 2020 snapshot: the
 * plantation was planted in 2018, so it captures only ~2 years of growth.
 * Predicted biomass from this height will likely read low relative to
 * the site's actual current (2026) state -- this is a known limitation,
 * not something this script can fix.
 *
 * THIS SCRIPT DOES ONLY HEIGHT EXTRACTION: it builds a grid of sample
 * points inside the Botanical boundary at ~10 m spacing (i.e. the CHM's
 * native resolution -- one point per pixel via .sample()), samples canopy
 * height at each, and exports point_id, lat, lon, canopy_height_m. No
 * curve application, no biomass prediction happens here -- that is
 * src/validate_curve_at_plantation.py, once this CSV exists.
 *
 * No tiling / TEST_MODE here: Botanical is ~1.82 ha, far below GEE's
 * per-operation size limit that required tiling for the multi-degree
 * GEDI training region.
 *
 * Boundary coordinates below are copied verbatim (9 decimal places) from
 * data/boundaries/Botanical_boundary.geojson's single Polygon ring.
 */

// =============================================================================
// 0. CONFIG
// =============================================================================

var CHM_ASSET_ID = 'users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1';
var EXPORT_FOLDER = 'carbon_stock_estimation';
var EXPORT_FILE_PREFIX = 'botanical_canopy_height_validation';
var SAMPLE_SCALE_M = 10; // ETH CHM native resolution -- one grid point per pixel

// One-time verification print: confirm the real band name(s) before
// selecting anything below. Cheap -- inspects one Image's metadata only.
print('ETH canopy height bands:', ee.Image(CHM_ASSET_ID).bandNames());

// =============================================================================
// 1. Botanical boundary polygon -- copied from
//    data/boundaries/Botanical_boundary.geojson (Site_Name: "Botanical",
//    Area_Ha: 1.820407571906957, Baseline_Y: 2018)
// =============================================================================

var BOTANICAL_BOUNDARY = ee.Geometry.Polygon([[
    [73.474027679, 18.102214063],
    [73.473826153, 18.102498710],
    [73.473695735, 18.102584816],
    [73.473648092, 18.102622981],
    [73.473574237, 18.102753941],
    [73.473552826, 18.102798178],
    [73.473711697, 18.102866466],
    [73.473872487, 18.102915497],
    [73.474194081, 18.103011884],
    [73.474600803, 18.103193568],
    [73.474729199, 18.103239829],
    [73.474831727, 18.103136744],
    [73.474872897, 18.103036526],
    [73.474848741, 18.103018321],
    [73.474819115, 18.102974098],
    [73.474854378, 18.102847860],
    [73.474921413, 18.102593678],
    [73.474990207, 18.102338673],
    [73.475047025, 18.102151441],
    [73.475000237, 18.102090744],
    [73.474950119, 18.102009912],
    [73.475013012, 18.101829431],
    [73.475009509, 18.101626639],
    [73.474989439, 18.101515882],
    [73.474923870, 18.101398062],
    [73.474872081, 18.101393146],
    [73.474866091, 18.101392578],
    [73.474776859, 18.101376790],
    [73.474605057, 18.101385485],
    [73.474474670, 18.101468240],
    [73.474378630, 18.101631702],
    [73.474353606, 18.101688479],
    [73.474328437, 18.101762012],
    [73.474285906, 18.101816975],
    [73.474187165, 18.101988794],
    [73.474068052, 18.102157037],
    [73.474027679, 18.102214063]
]]);

// =============================================================================
// 2. Sample canopy height on a grid inside the boundary, and export
// =============================================================================

// select(0) rather than a specific band name -- defensive against the ETH
// asset's exact band naming (verify against the 'ETH canopy height bands'
// print above; adjust here if the height band isn't the first one, e.g.
// if band 0 turns out to be an uncertainty/SD band instead).
var canopyHeight = ee.Image(CHM_ASSET_ID).select(0).rename('canopy_height_m');

// .sample() over a region (no pre-defined points) generates one sample
// point per pixel at the given scale -- this is the grid, at ~10 m
// spacing, matching the CHM's native resolution. geometries: true keeps
// each point's coordinates so lat/lon can be read back out below.
var samplePoints = canopyHeight.sample({
  region: BOTANICAL_BOUNDARY,
  scale: SAMPLE_SCALE_M,
  geometries: true,
  tileScale: 16
});

var botanicalHeights = samplePoints.map(function(f) {
  var coords = f.geometry().coordinates();
  return f.set({
    point_id: f.id(),
    lon: coords.get(0),
    lat: coords.get(1)
  });
});

// Cheap: a single .size() call on a ~1.82 ha grid (a few hundred points
// at most), not the large multi-degree collections that required
// avoiding .size()/.getInfo() elsewhere in this project.
print('Sample points:', botanicalHeights.size());

Export.table.toDrive({
  collection: botanicalHeights,
  description: 'botanical_canopy_height_validation',
  folder: EXPORT_FOLDER,
  fileNamePrefix: EXPORT_FILE_PREFIX,
  fileFormat: 'CSV',
  selectors: ['point_id', 'lat', 'lon', 'canopy_height_m']
});
