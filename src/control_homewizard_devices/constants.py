DELTA_T_SEC = 60
DELTA_T = DELTA_T_SEC / 3600  # Convert seconds to hours

MAX_GRID_DRAW = 1e6  # Maximum grid draw in watts
# Penalty for not reaching the energy stored for needed devices
ENERGY_STORED_PENALTY = 1e6
EPSILON = 1e-3  # Small tolerance for numerical stability

AGGREGATE_BATTERY = "aggregate_battery"
