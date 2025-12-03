"""
Task Analysis Types

Data structures for task analysis and monitoring.
"""

from typing import Dict, List
from dataclasses import dataclass
from enum import Enum


@dataclass
class TaskAnalysis:
    """Result of LLM task analysis."""

    # Task understanding
    action_sequence: List[str]  # High-level action steps
    goal_predicates: List[str]  # Goal state predicates
    preconditions: List[str]  # Required preconditions

    # Relevant objects
    goal_objects: List[str]  # Primary objects for task
    tool_objects: List[str]  # Tools/intermediate objects
    obstacle_objects: List[str]  # Objects to avoid

    # Scene-specific predicates
    initial_predicates: List[str]  # Current state predicates
    global_predicates: List[str]  # Global/robot state predicates (not object-related)
    relevant_predicates: List[str]  # Task-relevant predicate types
    relevant_types: List[str]  # Task-relevant PDDL types
    # Action definitions
    required_actions: List[Dict]  # PDDL action schemas

    # Metadata
    complexity: str  # 'simple', 'medium', 'complex'
    estimated_steps: int


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
