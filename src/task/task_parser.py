"""
Task parser for extracting structured information from natural language tasks.
"""

import re
from typing import Dict, List, Optional, Set

import numpy as np


class TaskParser:
    """
    Parses natural language task descriptions into structured task representations.
    """

    def __init__(self):
        """Initialize task parser."""
        # Common action verbs in robotic tasks
        self.action_verbs = {
            "pick", "place", "grasp", "grab", "move", "push", "pull",
            "open", "close", "pour", "lift", "put", "bring", "fetch",
            "arrange", "stack", "unstack", "insert", "remove", "clean"
        }

        # Spatial prepositions
        self.spatial_preps = {
            "on", "in", "near", "above", "below", "next to", "beside",
            "behind", "in front of", "to the left of", "to the right of",
            "inside", "outside", "between", "around"
        }

        # Common object attributes
        self.color_words = {
            "red", "blue", "green", "yellow", "black", "white",
            "orange", "purple", "pink", "brown", "gray", "grey"
        }

        self.size_words = {
            "small", "large", "big", "tiny", "huge", "medium"
        }

    def parse(self, task_description: str) -> Dict:
        """
        Parse task description into structured format.

        Args:
            task_description: Natural language task description

        Returns:
            Dict containing parsed task information
        """
        task_description = task_description.lower().strip()

        # Extract components
        action = self._extract_action(task_description)
        objects = self._extract_objects(task_description)
        goal_objects = self._extract_goal_objects(task_description, action)
        tool_objects = self._extract_tool_objects(task_description)
        spatial_constraints = self._extract_spatial_constraints(task_description)
        attributes = self._extract_attributes(task_description)

        # Generate goal predicates
        goal_predicates = self._generate_goal_predicates(
            action, goal_objects, spatial_constraints
        )

        return {
            "description": task_description,
            "action": action,
            "objects": objects,
            "goal_objects": goal_objects,
            "tool_objects": tool_objects,
            "spatial_constraints": spatial_constraints,
            "attributes": attributes,
            "goal_predicates": goal_predicates,
        }

    def _extract_action(self, text: str) -> Optional[str]:
        """Extract primary action verb from task."""
        words = text.split()
        for word in words:
            if word in self.action_verbs:
                return word
        return None

    def _extract_objects(self, text: str) -> List[str]:
        """Extract all mentioned objects."""
        # Simple noun phrase extraction
        # This is a simplified version - could use NLP for better extraction
        objects = []

        # Common object patterns
        patterns = [
            r'\b(?:the|a|an)\s+(\w+\s+)?(\w+)\b',
            r'\b(\w+\s+)?(?:cup|bottle|box|book|pen|phone|laptop|plate|bowl)\b'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                obj = ' '.join(filter(None, match)).strip()
                if obj and obj not in self.action_verbs and obj not in self.spatial_preps:
                    objects.append(obj)

        return list(set(objects))

    def _extract_goal_objects(self, text: str, action: Optional[str]) -> List[str]:
        """Extract objects that are the goal of the action."""
        if not action:
            return []

        # Find objects immediately after the action verb
        goal_objects = []
        action_idx = text.find(action)

        if action_idx != -1:
            after_action = text[action_idx + len(action):].strip()
            # Extract first noun phrase after action
            match = re.match(r'(?:the|a|an)?\s*(\w+(?:\s+\w+)?)', after_action)
            if match:
                goal_objects.append(match.group(1).strip())

        return goal_objects

    def _extract_tool_objects(self, text: str) -> List[str]:
        """Extract tools that might be needed."""
        # Objects mentioned with "with" or "using"
        tool_objects = []

        patterns = [
            r'with (?:the|a|an)?\s*(\w+)',
            r'using (?:the|a|an)?\s*(\w+)'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            tool_objects.extend(matches)

        return tool_objects

    def _extract_spatial_constraints(self, text: str) -> List[Dict]:
        """Extract spatial constraints from task."""
        constraints = []

        for prep in self.spatial_preps:
            if prep in text:
                # Find what comes after the preposition
                idx = text.find(prep)
                after = text[idx + len(prep):].strip()
                match = re.match(r'(?:the|a|an)?\s*(\w+(?:\s+\w+)?)', after)
                if match:
                    reference_object = match.group(1).strip()
                    constraints.append({
                        "relation": prep.replace(" ", "_"),
                        "reference": reference_object
                    })

        return constraints

    def _extract_attributes(self, text: str) -> Dict[str, List[str]]:
        """Extract object attributes (color, size, etc.)."""
        attributes = {
            "colors": [],
            "sizes": []
        }

        # Extract colors
        for color in self.color_words:
            if color in text:
                attributes["colors"].append(color)

        # Extract sizes
        for size in self.size_words:
            if size in text:
                attributes["sizes"].append(size)

        return attributes

    def _generate_goal_predicates(
        self,
        action: Optional[str],
        goal_objects: List[str],
        spatial_constraints: List[Dict]
    ) -> List[str]:
        """Generate PDDL-like goal predicates from parsed task."""
        predicates = []

        if not action or not goal_objects:
            return predicates

        goal_obj = goal_objects[0] if goal_objects else "object"

        # Action-specific goal predicates
        if action in ["pick", "grasp", "grab"]:
            predicates.append(f"holding({goal_obj})")

        elif action in ["place", "put"]:
            if spatial_constraints:
                ref = spatial_constraints[0]["reference"]
                predicates.append(f"on({goal_obj}, {ref})")
            else:
                predicates.append(f"placed({goal_obj})")

        elif action in ["open"]:
            predicates.append(f"open({goal_obj})")

        elif action in ["close"]:
            predicates.append(f"closed({goal_obj})")

        elif action in ["move", "bring"]:
            if spatial_constraints:
                ref = spatial_constraints[0]["reference"]
                relation = spatial_constraints[0]["relation"]
                predicates.append(f"{relation}({goal_obj}, {ref})")

        # Add spatial constraint predicates
        for constraint in spatial_constraints:
            relation = constraint["relation"]
            reference = constraint["reference"]
            if goal_obj != reference:
                predicates.append(f"{relation}({goal_obj}, {reference})")

        return predicates

    def extract_keywords(self, text: str) -> Set[str]:
        """Extract important keywords for focused perception."""
        keywords = set()

        # Add action verbs
        for verb in self.action_verbs:
            if verb in text:
                keywords.add(verb)

        # Add colors
        for color in self.color_words:
            if color in text:
                keywords.add(color)

        # Add size words
        for size in self.size_words:
            if size in text:
                keywords.add(size)

        return keywords
