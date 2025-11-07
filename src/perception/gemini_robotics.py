"""
Gemini Robotics-ER 1.5 integration for robotic perception and reasoning.

Based on: https://ai.google.dev/gemini-api/docs/robotics-overview

This module provides a client for interacting with Gemini's robotics-specific
model for:
- Object detection with 2D coordinates
- Spatial reasoning and scene understanding
- Trajectory planning
- Task decomposition
- Video tracking
"""

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import io

import numpy as np
from PIL import Image

# Try to import the new Google GenAI SDK first (for robotics model)
try:
    from google import genai as genai_new
    from google.genai import types
    USE_NEW_SDK = True
except ImportError:
    USE_NEW_SDK = False
    import google.generativeai as genai_old


@dataclass
class ObjectDetectionResult:
    """Result from object detection query."""

    objects: List[Dict[str, Any]]  # List of detected objects with positions
    raw_response: str  # Full LLM response
    processing_time: float  # Time taken (seconds)
    model_used: str

    def get_object_by_label(self, label: str) -> Optional[Dict]:
        """Find first object matching label."""
        for obj in self.objects:
            if obj.get("label", "").lower() == label.lower():
                return obj
        return None

    def get_objects_by_type(self, object_type: str) -> List[Dict]:
        """Find all objects of a given type."""
        return [obj for obj in self.objects if object_type.lower() in obj.get("label", "").lower()]


@dataclass
class SpatialReasoningResult:
    """Result from spatial reasoning query."""

    reasoning: str  # Natural language reasoning
    relationships: List[Dict[str, Any]]  # Spatial relationships
    recommendations: List[str]  # Action recommendations
    raw_response: str
    processing_time: float
    model_used: str


@dataclass
class TrajectoryResult:
    """Result from trajectory planning query."""

    waypoints: List[Dict[str, Any]]  # Sequential waypoints with coordinates
    description: str  # Path description
    estimated_distance: Optional[float]  # Total path length
    raw_response: str
    processing_time: float
    model_used: str


@dataclass
class TaskDecompositionResult:
    """Result from task decomposition query."""

    subtasks: List[Dict[str, Any]]  # Ordered subtask list
    reasoning: str  # Task analysis
    dependencies: List[Tuple[int, int]]  # Subtask dependencies (from, to)
    estimated_complexity: str  # "simple", "medium", "complex"
    raw_response: str
    processing_time: float
    model_used: str


@dataclass
class InteractionPointResult:
    """Result from interaction point detection query."""

    object_label: str  # Object being interacted with
    interaction_point: List[int]  # [y, x] in 0-1000 scale
    interaction_type: str  # "grasp", "push", "pull", "open", etc.
    confidence: float  # Confidence score
    reasoning: str  # Why this point was chosen
    alternative_points: List[Dict[str, Any]]  # Other possible interaction points
    raw_response: str
    processing_time: float
    model_used: str


class GeminiRoboticsClient:
    """
    Client for Gemini Robotics-ER 1.5 model.

    Provides high-level interface for robotics-specific vision-language tasks
    including object detection, spatial reasoning, trajectory planning, and
    task decomposition.

    Example:
        >>> client = GeminiRoboticsClient(api_key="your_key")
        >>> result = client.detect_objects(image, "Find all cups and bowls")
        >>> for obj in result.objects:
        ...     print(f"{obj['label']} at {obj['position']}")
    """

    # Model configurations
    ROBOTICS_MODEL = "gemini-robotics-er-1.5-preview"  # Robotics-ER model (new SDK)
    FLASH_MODEL = "gemini-1.5-flash"  # Fast general model (old SDK)
    PRO_MODEL = "gemini-1.5-pro"  # High quality model (old SDK)

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "auto",  # "auto" selects best available
        default_temperature: float = 0.5,  # Higher for robotics model
        enable_thinking: bool = True,
        thinking_budget: int = 0  # 0 for fast robotics tasks
    ):
        """
        Initialize Gemini Robotics client.

        Args:
            api_key: Google AI API key (None to use GOOGLE_API_KEY env var)
            model_name: Model to use:
                - "auto": Automatically use robotics model if available
                - "gemini-robotics-er-1.5-preview": Robotics-specific model
                - "gemini-1.5-flash": Fast general model
                - "gemini-1.5-pro": High quality general model
            default_temperature: Default temperature for generation (0.0-1.0)
            enable_thinking: Enable extended thinking mode
            thinking_budget: Token budget for thinking (0=disabled for speed)
        """
        self.api_key = api_key
        self.use_new_sdk = USE_NEW_SDK

        # Auto-select best model
        if model_name == "auto":
            if USE_NEW_SDK:
                self.model_name = self.ROBOTICS_MODEL
                print(f"ℹ Using Gemini Robotics-ER model: {self.model_name}")
            else:
                self.model_name = self.FLASH_MODEL
                print(f"ℹ New GenAI SDK not available. Using: {self.model_name}")
                print(f"  Install with: pip install google-genai")
        else:
            self.model_name = model_name

        # Initialize appropriate SDK
        if USE_NEW_SDK and "robotics" in self.model_name:
            # Use new GenAI SDK for robotics model
            self.client = genai_new.Client(api_key=api_key)
            self.model = None  # Not used with new SDK
            print(f"✓ Initialized with new GenAI SDK (robotics optimized)")
        else:
            # Use old SDK for general models
            if api_key:
                genai_old.configure(api_key=api_key)
            self.client = None
            self.model = genai_old.GenerativeModel(self.model_name)
            print(f"✓ Initialized with legacy SDK")

        self.default_temperature = default_temperature
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget

        # Response cache for identical queries
        self._cache: Dict[str, Any] = {}
        self._cache_enabled = True

    def detect_objects(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        query: str = "Detect all objects in the image with their positions",
        temperature: Optional[float] = None,
        return_json: bool = True
    ) -> ObjectDetectionResult:
        """
        Detect objects in image with 2D coordinates.

        Returns normalized [y, x] coordinates (0-1000 scale) for each object.

        Args:
            image: Input image (numpy array, PIL Image, or file path)
            query: Detection query/prompt
            temperature: Generation temperature (None = use default)
            return_json: Request structured JSON output

        Returns:
            ObjectDetectionResult with detected objects and positions

        Example:
            >>> result = client.detect_objects(
            ...     image,
            ...     "Find all graspable objects like cups, bottles, and tools"
            ... )
            >>> result.objects
            [
                {
                    "label": "red cup",
                    "position": [450, 320],  # [y, x] in 0-1000 scale
                    "confidence": 0.95,
                    "properties": {"color": "red", "graspable": true}
                }
            ]
        """
        start_time = time.time()

        # Convert image to PIL if needed
        pil_image = self._prepare_image(image)

        # Build detection prompt
        prompt = self._build_detection_prompt(query, return_json)

        # Check cache
        cache_key = f"detect_{hash(prompt)}_{hash(pil_image.tobytes())}"
        if self._cache_enabled and cache_key in self._cache:
            cached = self._cache[cache_key]
            cached.processing_time = time.time() - start_time  # Update time
            return cached

        # Generate content using appropriate SDK
        try:
            if self.use_new_sdk and self.client:
                # Use new GenAI SDK (robotics model)
                response_text = self._generate_with_new_sdk(pil_image, prompt, temperature)
                processing_time = time.time() - start_time

                # Parse response
                if return_json:
                    parsed_data = self._parse_json_response(response_text)
                    objects = parsed_data.get("objects", []) if isinstance(parsed_data, dict) else []
                else:
                    objects = self._parse_detection_response(response_text)

                raw_response = response_text

            else:
                # Use old SDK (general models)
                config = self._build_generation_config(temperature, return_json)

                response = self.model.generate_content(
                    [prompt, pil_image],
                    generation_config=config
                )

                processing_time = time.time() - start_time

                # Parse response
                if return_json:
                    parsed_data = self._parse_json_response(response.text)
                    objects = parsed_data.get("objects", []) if isinstance(parsed_data, dict) else []
                else:
                    objects = self._parse_detection_response(response.text)

                raw_response = response.text

            result = ObjectDetectionResult(
                objects=objects,
                raw_response=raw_response,
                processing_time=processing_time,
                model_used=self.model_name
            )

            # Cache result
            if self._cache_enabled:
                self._cache[cache_key] = result

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            import traceback
            error_details = f"{e}\n{traceback.format_exc()}"
            print(f"   ⚠ Object detection failed: {e}")
            print(f"   → Check your API key and model availability")
            return ObjectDetectionResult(
                objects=[],
                raw_response=error_details,
                processing_time=processing_time,
                model_used=self.model_name
            )

    def analyze_spatial_relationships(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        query: str = "Analyze spatial relationships between objects",
        focus_objects: Optional[List[str]] = None,
        temperature: Optional[float] = None
    ) -> SpatialReasoningResult:
        """
        Analyze spatial relationships and scene understanding.

        Args:
            image: Input image
            query: Spatial reasoning query
            focus_objects: Specific objects to focus on (None = all objects)
            temperature: Generation temperature

        Returns:
            SpatialReasoningResult with relationships and reasoning

        Example:
            >>> result = client.analyze_spatial_relationships(
            ...     image,
            ...     "Which object is closest to the robot gripper?",
            ...     focus_objects=["cup", "bottle", "bowl"]
            ... )
        """
        start_time = time.time()

        pil_image = self._prepare_image(image)

        # Build spatial reasoning prompt
        prompt = self._build_spatial_prompt(query, focus_objects)

        try:
            # Generate content using appropriate SDK
            if self.use_new_sdk and self.client:
                # Use new GenAI SDK (robotics model)
                response_text = self._generate_with_new_sdk(pil_image, prompt, temperature)
                processing_time = time.time() - start_time
                data = self._parse_json_response(response_text)
                raw_response = response_text

            else:
                # Use old SDK (general models)
                config = self._build_generation_config(temperature, return_json=True)

                response = self.model.generate_content(
                    [prompt, pil_image],
                    generation_config=config
                )

                processing_time = time.time() - start_time
                data = self._parse_json_response(response.text)
                raw_response = response.text

            return SpatialReasoningResult(
                reasoning=data.get("reasoning", ""),
                relationships=data.get("relationships", []),
                recommendations=data.get("recommendations", []),
                raw_response=raw_response,
                processing_time=processing_time,
                model_used=self.model_name
            )

        except Exception as e:
            processing_time = time.time() - start_time
            print(f"Spatial reasoning failed: {e}")
            return SpatialReasoningResult(
                reasoning=str(e),
                relationships=[],
                recommendations=[],
                raw_response=str(e),
                processing_time=processing_time,
                model_used=self.model_name
            )

    def plan_trajectory(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        start_point: Optional[Tuple[int, int]] = None,
        end_point: Optional[Tuple[int, int]] = None,
        query: str = "Plan a safe trajectory path",
        temperature: Optional[float] = None
    ) -> TrajectoryResult:
        """
        Plan trajectory path for robot movement.

        Args:
            image: Input image showing environment
            start_point: Starting position [y, x] (None = current position)
            end_point: Target position [y, x] (None = infer from query)
            query: Trajectory planning query
            temperature: Generation temperature

        Returns:
            TrajectoryResult with waypoints and path description

        Example:
            >>> result = client.plan_trajectory(
            ...     image,
            ...     start_point=[500, 300],
            ...     end_point=[700, 600],
            ...     query="Plan path avoiding obstacles to reach the cup"
            ... )
        """
        start_time = time.time()

        pil_image = self._prepare_image(image)

        # Build trajectory prompt
        prompt = self._build_trajectory_prompt(query, start_point, end_point)

        try:
            # Generate content using appropriate SDK
            if self.use_new_sdk and self.client:
                # Use new GenAI SDK (robotics model)
                response_text = self._generate_with_new_sdk(pil_image, prompt, temperature)
                processing_time = time.time() - start_time
                data = self._parse_json_response(response_text)
                raw_response = response_text

            else:
                # Use old SDK (general models)
                config = self._build_generation_config(temperature, return_json=True)

                response = self.model.generate_content(
                    [prompt, pil_image],
                    generation_config=config
                )

                processing_time = time.time() - start_time
                data = self._parse_json_response(response.text)
                raw_response = response.text

            return TrajectoryResult(
                waypoints=data.get("waypoints", []),
                description=data.get("description", ""),
                estimated_distance=data.get("estimated_distance"),
                raw_response=raw_response,
                processing_time=processing_time,
                model_used=self.model_name
            )

        except Exception as e:
            processing_time = time.time() - start_time
            print(f"Trajectory planning failed: {e}")
            return TrajectoryResult(
                waypoints=[],
                description=str(e),
                estimated_distance=None,
                raw_response=str(e),
                processing_time=processing_time,
                model_used=self.model_name
            )

    def decompose_task(
        self,
        task_description: str,
        image: Optional[Union[np.ndarray, Image.Image, str, Path]] = None,
        available_actions: Optional[List[str]] = None,
        temperature: Optional[float] = None
    ) -> TaskDecompositionResult:
        """
        Decompose high-level task into executable subtasks.

        Args:
            task_description: Natural language task
            image: Optional scene image for context
            available_actions: List of available robot actions
            temperature: Generation temperature

        Returns:
            TaskDecompositionResult with subtasks and dependencies

        Example:
            >>> result = client.decompose_task(
            ...     "Put the apple in the bowl",
            ...     image=scene_image,
            ...     available_actions=["navigate", "grasp", "place", "open"]
            ... )
        """
        start_time = time.time()

        # Build decomposition prompt
        prompt = self._build_decomposition_prompt(task_description, available_actions)

        try:
            # Prepare image if provided
            pil_image = self._prepare_image(image) if image is not None else None

            # Generate content using appropriate SDK
            if self.use_new_sdk and self.client and pil_image is not None:
                # Use new GenAI SDK with image (robotics model)
                response_text = self._generate_with_new_sdk(pil_image, prompt, temperature)
                processing_time = time.time() - start_time
                data = self._parse_json_response(response_text)
                raw_response = response_text

            elif self.use_new_sdk and self.client and pil_image is None:
                # Use new GenAI SDK without image (text only)
                temp = temperature if temperature is not None else self.default_temperature
                config = types.GenerateContentConfig(
                    temperature=temp,
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=self.thinking_budget
                    ) if self.enable_thinking else None
                )

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config
                )

                processing_time = time.time() - start_time
                data = self._parse_json_response(response.text)
                raw_response = response.text

            else:
                # Use old SDK (general models)
                config = self._build_generation_config(temperature, return_json=True)

                # Include image if provided
                content = [prompt]
                if pil_image is not None:
                    content.append(pil_image)

                response = self.model.generate_content(
                    content,
                    generation_config=config
                )

                processing_time = time.time() - start_time
                data = self._parse_json_response(response.text)
                raw_response = response.text

            return TaskDecompositionResult(
                subtasks=data.get("subtasks", []),
                reasoning=data.get("reasoning", ""),
                dependencies=data.get("dependencies", []),
                estimated_complexity=data.get("complexity", "medium"),
                raw_response=raw_response,
                processing_time=processing_time,
                model_used=self.model_name
            )

        except Exception as e:
            processing_time = time.time() - start_time
            print(f"Task decomposition failed: {e}")
            return TaskDecompositionResult(
                subtasks=[],
                reasoning=str(e),
                dependencies=[],
                estimated_complexity="unknown",
                raw_response=str(e),
                processing_time=processing_time,
                model_used=self.model_name
            )

    def detect_interaction_point(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        object_label: str,
        action: Optional[str] = None,
        task_context: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> InteractionPointResult:
        """
        Detect optimal interaction point on an object for manipulation.

        This method identifies where a robot should interact with an object
        (grasp point, push point, etc.) based on the object's visual appearance
        and optionally conditioned on a specific action or task.

        Args:
            image: Input image showing the object
            object_label: Label/name of the object to interact with
            action: Optional action type ("grasp", "push", "pull", "open", etc.)
            task_context: Optional task description for context
            temperature: Generation temperature

        Returns:
            InteractionPointResult with interaction point and reasoning

        Example:
            >>> # Find grasp point on a cup
            >>> result = client.detect_interaction_point(
            ...     image,
            ...     object_label="red cup",
            ...     action="grasp",
            ...     task_context="Pick up the cup to pour water"
            ... )
            >>> print(f"Grasp at {result.interaction_point}")
            >>> print(f"Reasoning: {result.reasoning}")
        """
        start_time = time.time()

        pil_image = self._prepare_image(image)

        # Build interaction point prompt
        prompt = self._build_interaction_point_prompt(
            object_label,
            action,
            task_context
        )

        try:
            # Generate content using appropriate SDK
            if self.use_new_sdk and self.client:
                # Use new GenAI SDK (robotics model)
                response_text = self._generate_with_new_sdk(pil_image, prompt, temperature)
                processing_time = time.time() - start_time
                data = self._parse_json_response(response_text)
                raw_response = response_text

            else:
                # Use old SDK (general models)
                config = self._build_generation_config(temperature, return_json=True)

                response = self.model.generate_content(
                    [prompt, pil_image],
                    generation_config=config
                )

                processing_time = time.time() - start_time
                data = self._parse_json_response(response.text)
                raw_response = response.text

            return InteractionPointResult(
                object_label=object_label,
                interaction_point=data.get("interaction_point", [500, 500]),
                interaction_type=data.get("interaction_type", action or "grasp"),
                confidence=data.get("confidence", 0.5),
                reasoning=data.get("reasoning", ""),
                alternative_points=data.get("alternative_points", []),
                raw_response=raw_response,
                processing_time=processing_time,
                model_used=self.model_name
            )

        except Exception as e:
            processing_time = time.time() - start_time
            print(f"Interaction point detection failed: {e}")
            return InteractionPointResult(
                object_label=object_label,
                interaction_point=[500, 500],
                interaction_type=action or "unknown",
                confidence=0.0,
                reasoning=str(e),
                alternative_points=[],
                raw_response=str(e),
                processing_time=processing_time,
                model_used=self.model_name
            )

    def track_objects_in_video(
        self,
        video_frames: List[Union[np.ndarray, Image.Image]],
        query: str = "Track object movement across frames",
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Track objects across video frames.

        Args:
            video_frames: List of video frames
            query: Tracking query
            temperature: Generation temperature

        Returns:
            Dictionary with tracking results
        """
        # Convert frames to PIL
        pil_frames = [self._prepare_image(frame) for frame in video_frames]

        prompt = f"{query}\n\nProvide tracking information in JSON format with object IDs and positions per frame."

        try:
            # Generate content using appropriate SDK
            if self.use_new_sdk and self.client:
                # Use new GenAI SDK (robotics model)
                # Convert all frames to bytes
                img_byte_arr_list = []
                for pil_frame in pil_frames:
                    img_byte_arr = io.BytesIO()
                    pil_frame.save(img_byte_arr, format='PNG')
                    img_byte_arr_list.append(img_byte_arr.getvalue())

                # Create content parts (all frames + prompt)
                content_parts = []
                for img_bytes in img_byte_arr_list:
                    content_parts.append(
                        types.Part.from_bytes(
                            data=img_bytes,
                            mime_type='image/png',
                        )
                    )
                content_parts.append(prompt)

                # Build config
                temp = temperature if temperature is not None else self.default_temperature
                config = types.GenerateContentConfig(
                    temperature=temp,
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=self.thinking_budget
                    ) if self.enable_thinking else None
                )

                # Generate
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=content_parts,
                    config=config
                )

                return self._parse_json_response(response.text)

            else:
                # Use old SDK (general models)
                config = self._build_generation_config(temperature, return_json=True)

                response = self.model.generate_content(
                    [prompt] + pil_frames,
                    generation_config=config
                )

                return self._parse_json_response(response.text)

        except Exception as e:
            print(f"Video tracking failed: {e}")
            return {"error": str(e)}

    # ========================================================================
    # Utility methods
    # ========================================================================

    def _prepare_image(self, image: Union[np.ndarray, Image.Image, str, Path]) -> Image.Image:
        """Convert image to PIL Image format."""
        if isinstance(image, (str, Path)):
            return Image.open(image)
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image)
        elif isinstance(image, Image.Image):
            return image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def _generate_with_new_sdk(
        self,
        pil_image: Image.Image,
        prompt: str,
        temperature: Optional[float]
    ) -> str:
        """Generate content using new GenAI SDK (for robotics model)."""
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Create content parts
        content_parts = [
            types.Part.from_bytes(
                data=img_byte_arr,
                mime_type='image/png',
            ),
            prompt
        ]

        # Build config
        temp = temperature if temperature is not None else self.default_temperature
        config = types.GenerateContentConfig(
            temperature=temp,
            thinking_config=types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            ) if self.enable_thinking else None
        )

        # Generate
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=content_parts,
            config=config
        )

        return response.text

    def _build_generation_config(
        self,
        temperature: Optional[float],
        return_json: bool = False
    ):
        """Build generation configuration for old SDK."""
        config = {
            "temperature": temperature if temperature is not None else self.default_temperature,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }

        if return_json:
            config["response_mime_type"] = "application/json"

        return genai_old.GenerationConfig(**config)

    def _build_detection_prompt(self, query: str, return_json: bool) -> str:
        """Build object detection prompt."""
        base = f"""You are a robotic vision system. {query}

For each object, provide:
- label: Object name/type
- position: [y, x] coordinates in 0-1000 normalized scale
- confidence: Detection confidence (0.0-1.0)
- properties: Dict of relevant properties (color, size, affordances, etc.)"""

        if return_json:
            base += """

Return in JSON format:
{
  "objects": [
    {
      "label": "object_name",
      "position": [y, x],
      "confidence": 0.95,
      "properties": {"color": "red", "graspable": true}
    }
  ]
}"""

        return base

    def _build_spatial_prompt(self, query: str, focus_objects: Optional[List[str]]) -> str:
        """Build spatial reasoning prompt."""
        prompt = f"""You are a robotic spatial reasoning system. {query}"""

        if focus_objects:
            prompt += f"\n\nFocus on these objects: {', '.join(focus_objects)}"

        prompt += """

Provide in JSON format:
{
  "reasoning": "Natural language explanation of spatial layout",
  "relationships": [
    {"object1": "cup", "relation": "left-of", "object2": "bottle", "distance": "near"}
  ],
  "recommendations": ["Action recommendations based on spatial analysis"]
}"""

        return prompt

    def _build_trajectory_prompt(
        self,
        query: str,
        start: Optional[Tuple[int, int]],
        end: Optional[Tuple[int, int]]
    ) -> str:
        """Build trajectory planning prompt."""
        prompt = f"""You are a robotic trajectory planner. {query}"""

        if start:
            prompt += f"\n\nStart position: {start} (y, x in 0-1000 scale)"
        if end:
            prompt += f"\nEnd position: {end} (y, x in 0-1000 scale)"

        prompt += """

Provide in JSON format:
{
  "waypoints": [
    {"id": 0, "position": [y, x], "label": "start"},
    {"id": 1, "position": [y, x], "label": "waypoint_1"},
    {"id": 2, "position": [y, x], "label": "goal"}
  ],
  "description": "Path description with reasoning",
  "estimated_distance": 123.5
}"""

        return prompt

    def _build_decomposition_prompt(
        self,
        task: str,
        actions: Optional[List[str]]
    ) -> str:
        """Build task decomposition prompt."""
        prompt = f"""You are a robotic task planner. Decompose this task: {task}"""

        if actions:
            prompt += f"\n\nAvailable actions: {', '.join(actions)}"

        prompt += """

Provide in JSON format:
{
  "subtasks": [
    {"id": 0, "action": "navigate", "target": "cup", "description": "Move to cup location"},
    {"id": 1, "action": "grasp", "target": "cup", "description": "Pick up the cup"}
  ],
  "reasoning": "Explanation of decomposition strategy",
  "dependencies": [[0, 1], [1, 2]],
  "complexity": "simple|medium|complex"
}"""

        return prompt

    def _build_interaction_point_prompt(
        self,
        object_label: str,
        action: Optional[str],
        task_context: Optional[str]
    ) -> str:
        """Build interaction point detection prompt."""
        prompt = f"""You are a robotic manipulation system. Analyze the image and identify the optimal interaction point on the {object_label}."""

        if action:
            prompt += f"\n\nAction: {action}"
            prompt += f"\nIdentify the best point to {action} this object based on its visual features."

        if task_context:
            prompt += f"\n\nTask context: {task_context}"
            prompt += "\nConsider this task when choosing the interaction point."

        prompt += f"""

Analyze the {object_label} visually and determine:
1. The optimal interaction point based on the object's shape, handles, graspable features
2. Why this point is best for the specified action
3. Alternative interaction points if applicable

Provide the interaction point as [y, x] coordinates in 0-1000 normalized scale, where:
- [0, 0] is top-left corner
- [1000, 1000] is bottom-right corner
- [500, 500] is center

Consider:
- For grasping: handles, edges, stable grip points
- For pushing: flat surfaces, center of mass
- For pulling: handles, edges facing the robot
- For opening: handles, knobs, pull points

Return in JSON format:
{{
  "interaction_point": [y, x],
  "interaction_type": "grasp|push|pull|open|etc",
  "confidence": 0.0-1.0,
  "reasoning": "Detailed explanation of why this point was chosen",
  "alternative_points": [
    {{"point": [y, x], "reason": "Alternative option explanation"}}
  ]
}}"""

        return prompt

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response from LLM."""
        try:
            # Handle markdown code blocks
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end]
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end]

            return json.loads(response_text.strip())
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Response: {response_text[:500]}")
            return {}

    def _parse_detection_response(self, response_text: str) -> List[Dict]:
        """Parse natural language detection response."""
        # Fallback parser for non-JSON responses
        objects = []
        # Simple parsing - extend as needed
        return objects

    def clear_cache(self):
        """Clear response cache."""
        self._cache.clear()

    def set_cache_enabled(self, enabled: bool):
        """Enable or disable response caching."""
        self._cache_enabled = enabled
