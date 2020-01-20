from enum import Enum
from typing import TypeVar

from nexinfosys.common.helper import FloatExp

Node = TypeVar('Node')  # Generic node type
Weight = FloatExp  # Type alias
Value = FloatExp  # Type alias


class EdgeType(Enum):
    """ Type of edge of a ComputationGraph """
    DIRECT = 0
    REVERSE = 1
