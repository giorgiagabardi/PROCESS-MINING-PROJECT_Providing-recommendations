from dataclasses import dataclass
from typing import Union

@dataclass
class BooleanCondition:
    # Represents a boolean condition on a feature/activity

    feature: str
    value: bool
    
    def __repr__(self):
        return f"{self.feature} = {self.value}"
    
    def __hash__(self):
        return hash((self.feature, self.value))
    
    def __eq__(self, other):
        if isinstance(other, BooleanCondition):
            return self.feature == other.feature and self.value == other.value
        return False


@dataclass
class ThresholdCondition:
    # Represents a threshold condition (used for prefix_length)
    feature: str
    op: str
    threshold: float
    
    def __repr__(self):
        return f"{self.feature} {self.op} {self.threshold}"
    
    def __hash__(self):
        return hash((self.feature, self.op, self.threshold))
    
    def __eq__(self, other):
        if isinstance(other, ThresholdCondition):
            return (self.feature == other.feature and 
                   self.op == other.op and 
                   self.threshold == other.threshold)
        return False

# Type alias for a generic condition
Condition = Union[BooleanCondition, ThresholdCondition]

# Type alias for a path in the decision tree
Path = list[Condition]
