from enum import Enum
from typing import TypeVar

Node = TypeVar('Node')  # Generic node type
Weight = float  # Type alias
Value = float  # Type alias


class EdgeType(Enum):
    """ Type of edge of a ComputationGraph """
    DIRECT = 0
    REVERSE = 1
