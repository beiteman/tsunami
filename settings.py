# ---------------------
# DATA RELATED
# ---------------------
NORM_ALPHA_E = 1.0/3000 # multiplier of e for normalization
NORM_ALPHA_Z = 1.0*3000 # Wave height (at any time)
NORM_ALPHA_DZ = 1.0*3000 # height changes (depend on timestamp)
NORM_APLHA_ZMAX = 1.0 # maximum wave height
NORM_ALPHA_SDC = 1.0/3000 # Signed Distance to Coast
SAMPLE_PER_SCENARIO = 256 # number of samples selected among the timesteps (can be redundant)
SAMPLE_POINT_PERCENT = 5.0 # number of points among available sensor area as sampling
SAMPLE_POINT_MIN = 100 # if the sample less than this, skip the sample

# Deep-Ocean Sensors (DART Buoys): 50 km to 200 km (30–120 miles) offshore.
SAMPLE_MIN_OFFSHORE_DIST=50 # sensor placement minimum (in meter) from offshore
SAMPLE_MAX_OFFSHORE_DIST=5_000_000 # sensor placement maximum (in meter) from offshore