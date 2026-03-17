"""
Task Analysis Types

Data structures for task analysis and monitoring.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class TaskAnalysis:
    """Result of LLM task analysis."""

    # Task understanding
    goal_predicates: List[str]  # Goal state predicates
    preconditions: List[str]  # Required preconditions

    # Relevant objects
    goal_objects: List[str]  # Primary objects for task

    # Scene-specific predicates
    initial_predicates: List[str]  # Current state predicates
    global_predicates: List[str]  # Global/robot state predicates (not object-related)
    relevant_predicates: List[str]  # Task-relevant predicate types
    # Action definitions
    required_actions: List[Dict]  # PDDL action schemas


# ---------------------------------------------------------------------------
# Layered Domain Generation Artifacts (L1–L5)
# ---------------------------------------------------------------------------

@dataclass
class L1GoalArtifact:
    """Output of L1: Goal Specification layer."""
    goal_predicates: List[str]    # grounded PDDL literals: ["(on red_block_1 blue_block_1)"]
    goal_objects: List[str]       # observed object IDs referenced in goals
    global_predicates: List[str]  # robot-state predicates: ["hand-empty"]
    generation_attempts: int = 1
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class L2PredicateArtifact:
    """Output of L2: Predicate Vocabulary layer."""
    predicate_signatures: List[str]  # ["(on ?obj ?surface)", "(holding ?obj)"]
    sensed_predicates: List[str]     # predicates tagged as sensed/external
    checked_variants: List[str]      # auto-generated checked-X variants
    generation_attempts: int = 1
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class L3ActionArtifact:
    """Output of L3: Action Schema layer."""
    actions: List[Dict]          # same schema as TaskAnalysis.required_actions
    sensing_actions: List[Dict]  # auto-generated sensing/check actions
    generation_attempts: int = 1
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class L4GroundingArtifact:
    """Output of L4: Grounding pre-check (algorithmic)."""
    object_bindings: Dict[str, str]  # action-param-type -> observed object_id example
    warnings: List[str]              # non-blocking feasibility warnings


@dataclass
class L5InitialStateArtifact:
    """Output of L5: Initial State Construction (algorithmic)."""
    true_literals: List[Tuple[str, List[str]]]   # [(pred_name, [arg1, arg2])]
    false_literals: List[Tuple[str, List[str]]]  # always-false (checked-X predicates)


@dataclass
class LayeredDomainArtifact:
    """Complete output of the layered domain generation pipeline."""
    l1: L1GoalArtifact
    l2: L2PredicateArtifact
    l3: L3ActionArtifact
    l4: Optional[L4GroundingArtifact]
    l5: Optional[L5InitialStateArtifact]
    task_description: str
    scene_objects: List[Dict]

    def to_task_analysis(self) -> "TaskAnalysis":
        """Bridge method: convert to legacy TaskAnalysis for backward compatibility."""
        all_actions = self.l3.actions + self.l3.sensing_actions
        initial_predicates: List[str] = []
        if self.l5:
            for pred_name, args in self.l5.true_literals:
                if args:
                    initial_predicates.append(f"({pred_name} {' '.join(args)})")
                else:
                    initial_predicates.append(f"({pred_name})")
        return TaskAnalysis(
            goal_predicates=self.l1.goal_predicates,
            preconditions=[],
            goal_objects=self.l1.goal_objects,
            initial_predicates=initial_predicates,
            global_predicates=self.l1.global_predicates,
            relevant_predicates=self.l2.predicate_signatures,
            required_actions=all_actions,
        )


class TaskState(Enum):
    """Possible states for task execution."""
    EXPLORE = "explore"  # Need more environmental information
    PLAN_AND_EXECUTE = "plan_and_execute"  # Ready to generate and execute plan
    REFINE_DOMAIN = "refine_domain"  # Domain incomplete, needs refinement
    GOAL_UNREACHABLE = "goal_unreachable"  # Goal cannot be achieved with current knowledge
    COMPLETE = "complete"  # Task successfully completed


@dataclass
class TaskStateDecision:
    """
    Decision about current task state and recommended action.
    """
    state: TaskState
    confidence: float  # 0.0 to 1.0
    reasoning: str
    blockers: List[str]  # What's preventing progress
    recommendations: List[str]  # Suggested actions
    metrics: Dict[str, any]  # Supporting data
