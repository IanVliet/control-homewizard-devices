from enum import Enum


DELTA_T_SEC = 60
DELTA_T = DELTA_T_SEC / 3600  # Convert seconds to hours

MAX_GRID_DRAW = 1e6  # Maximum grid draw in watts
# Penalty for not reaching the energy stored for needed devices
ENERGY_STORED_PENALTY = 1e6
EPSILON = 1e-3  # Small tolerance for numerical stability

SCIP_BINARY = "B"
SCIP_INTEGER = "I"
SCIP_CONTINUOUS = "C"


class VariableCategory(Enum):
    """
    Enum for the different variable categories used in scheduling.
    """

    BINARY = "binary"
    INTEGER = "integer"
    CONTINUOUS = "continuous"

    def to_pulp(self):
        from pulp import (
            LpBinary,
            LpInteger,
            LpContinuous,
        )

        return {
            VariableCategory.BINARY: LpBinary,
            VariableCategory.INTEGER: LpInteger,
            VariableCategory.CONTINUOUS: LpContinuous,
        }[self]

    def to_scip(self):
        return {
            VariableCategory.BINARY: SCIP_BINARY,
            VariableCategory.INTEGER: SCIP_INTEGER,
            VariableCategory.CONTINUOUS: SCIP_CONTINUOUS,
        }[self]
