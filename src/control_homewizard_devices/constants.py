from enum import Enum
from pulp import (
    LpBinary,
    LpInteger,
    LpContinuous,
)

DELTA_T_SEC = 60
DELTA_T = DELTA_T_SEC / 3600  # Convert seconds to hours


class VariableCategory(Enum):
    """
    Enum for the different variable categories used in scheduling.
    """

    BINARY = LpBinary
    INTEGER = LpInteger
    CONTINUOUS = LpContinuous
