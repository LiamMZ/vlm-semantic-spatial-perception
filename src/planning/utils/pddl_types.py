"""
PDDL Type Definitions

Data structures for PDDL domain and problem components.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class PDDLRequirements(Enum):
    """PDDL requirement flags."""
    STRIPS = ":strips"
    TYPING = ":typing"
    NEGATIVE_PRECONDITIONS = ":negative-preconditions"
    DISJUNCTIVE_PRECONDITIONS = ":disjunctive-preconditions"
    EQUALITY = ":equality"
    EXISTENTIAL_PRECONDITIONS = ":existential-preconditions"
    UNIVERSAL_PRECONDITIONS = ":universal-preconditions"
    CONDITIONAL_EFFECTS = ":conditional-effects"
    ACTION_COSTS = ":action-costs"


@dataclass
class ObjectType:
    """PDDL object type definition."""
    name: str
    parent: Optional[str] = None  # For type hierarchy

    def to_pddl(self) -> str:
        """Convert to PDDL format."""
        if self.parent:
            return f"{self.name} - {self.parent}"
        return self.name


@dataclass
class Predicate:
    """PDDL predicate definition."""
    name: str
    parameters: List[Tuple[str, str]]  # [(param_name, type), ...]
    description: Optional[str] = None

    def to_pddl(self) -> str:
        """Convert to PDDL format (STRIPS): (predicate_name ?p1 ?p2)"""
        # Use STRIPS format without type annotations
        params = " ".join([f"?{name}" for name, typ in self.parameters])
        return f"({self.name} {params})"

    def __hash__(self):
        return hash((self.name, tuple(self.parameters)))

    def __eq__(self, other):
        if not isinstance(other, Predicate):
            return False
        return self.name == other.name and self.parameters == other.parameters


@dataclass
class Action:
    """PDDL action schema."""
    name: str
    parameters: List[Tuple[str, str]]  # [(param_name, type), ...]
    precondition: str  # PDDL expression
    effect: str  # PDDL expression
    description: Optional[str] = None
    is_llm_generated: bool = False

    def to_pddl(self) -> str:
        """Convert to PDDL action definition (STRIPS format)."""
        # Use STRIPS format without type annotations
        params = " ".join([f"?{name}" for name, typ in self.parameters])

        return f"""  (:action {self.name}
    :parameters ({params})
    :precondition {self.precondition}
    :effect {self.effect}
  )"""


@dataclass
class ObjectInstance:
    """Object instance in the problem."""
    name: str
    object_type: str

    def to_pddl(self) -> str:
        """Convert to PDDL format (STRIPS): just the name"""
        # Use STRIPS format without type annotations
        return f"{self.name}"


@dataclass
class Literal:
    """PDDL literal (predicate instance or negated predicate)."""
    predicate: str
    arguments: List[str]
    negated: bool = False

    def to_pddl(self) -> str:
        """Convert to PDDL format."""
        pred_str = f"({self.predicate} {' '.join(self.arguments)})"
        return f"(not {pred_str})" if self.negated else pred_str

    def __hash__(self):
        return hash((self.predicate, tuple(self.arguments), self.negated))

    def __eq__(self, other):
        if not isinstance(other, Literal):
            return False
        return (self.predicate == other.predicate and
                self.arguments == other.arguments and
                self.negated == other.negated)
