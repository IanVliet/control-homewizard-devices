from enum import Enum
from pulp import (
    LpBinary,
    LpInteger,
    LpContinuous,
)

DELTA_T_SEC = 60
DELTA_T = DELTA_T_SEC / 3600  # Convert seconds to hours

MAX_GRID_DRAW = 1e6  # Maximum grid draw in watts
# Penalty for not reaching the energy stored for needed devices
ENERGY_STORED_PENALTY = 1e6
EPSILON = 1e-3  # Small tolerance for numerical stability


class VariableCategory(Enum):
    """
    Enum for the different variable categories used in scheduling.
    """

    BINARY = LpBinary
    INTEGER = LpInteger
    CONTINUOUS = LpContinuous
