"""
Object tracking system using Gemini Robotics VLM.

This module provides object detection, affordance analysis, and interaction
point detection for robotic manipulation tasks.
"""

import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import io
import concurrent.futures

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

# Import coordinate conversion utilities
from .utils.coordinates import (
    normalized_to_pixel,
    pixel_to_normalized,
    compute_3d_position,
    project_3d_to_2d,
)

# Import object registry
from .object_registry import DetectedObject, InteractionPoint, DetectedObjectRegistry


class ObjectTracker:
    """
    Object tracking system using Gemini Robotics VLM.

    This class detects objects in camera frames, analyzes their affordances,
    and identifies interaction points for robot manipulation. It maintains
    a registry of detected objects.

    Example:
        >>> tracker = ObjectTracker(api_key="your_key")
        >>> tracker.detect_objects(color_frame, depth_frame, camera_intrinsics)
        >>> for obj in tracker.get_all_objects():
        ...     print(f"{obj.object_id}: {obj.affordances}")
        ...     for affordance, point in obj.interaction_points.items():
        ...         print(f"  {affordance} at {point.position_3d}")
    """

    # Model configurations
    ROBOTICS_MODEL = "gemini-robotics-er-1.5-preview"
    FLASH_MODEL = "gemini-1.5-flash"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "auto",
        default_temperature: float = 0.5,
        thinking_budget: int = 0,
        max_parallel_requests: int = 5
    ):
        """
        Initialize object tracker.

        Args:
            api_key: Google AI API key
            model_name: "auto" to use robotics model if available
            default_temperature: Generation temperature (0.0-1.0)
            thinking_budget: Token budget for extended thinking (0=disabled)
            max_parallel_requests: Max parallel VLM requests for object details
        """
        self.api_key = api_key
        self.use_new_sdk = USE_NEW_SDK
        self.max_parallel_requests = max_parallel_requests

        # Auto-select best model
        if model_name == "auto":
            if USE_NEW_SDK:
                self.model_name = self.ROBOTICS_MODEL
                print(f"ℹ ObjectTracker using: {self.model_name}")
            else:
                self.model_name = self.FLASH_MODEL
                print(f"ℹ New GenAI SDK not available. Using: {self.model_name}")
                print(f"  Install with: pip install google-genai")
        else:
            self.model_name = model_name

        # Initialize appropriate SDK
        if USE_NEW_SDK and "robotics" in self.model_name:
            self.client = genai_new.Client(api_key=api_key)
            self.model = None
            print(f"✓ ObjectTracker initialized with new GenAI SDK")
        else:
            if api_key:
                genai_old.configure(api_key=api_key)
            self.client = None
            self.model = genai_old.GenerativeModel(self.model_name)
            print(f"✓ ObjectTracker initialized with legacy SDK")

        self.default_temperature = default_temperature
        self.thinking_budget = thinking_budget

        # Object registry with thread safety
        self.registry = DetectedObjectRegistry()

        # Cache for current frame
        self._current_color_frame: Optional[Image.Image] = None
        self._current_depth_frame: Optional[np.ndarray] = None
        self._current_intrinsics: Optional[Any] = None

        # Performance optimization: image encoding cache
        self._encoded_image_cache: Optional[bytes] = None
        self._cache_image_id: Optional[int] = None

    def detect_objects(
        self,
        color_frame: Union[np.ndarray, Image.Image],
        depth_frame: Optional[np.ndarray] = None,
        camera_intrinsics: Optional[Any] = None,
        temperature: Optional[float] = None
    ) -> List[DetectedObject]:
        """
        Detect all objects in scene with affordances and interaction points.

        This is the main entry point. It:
        1. Detects all object names in the scene
        2. For each object in parallel:
           - Analyzes affordances
           - Detects interaction points for each affordance
        3. Updates object registry

        Args:
            color_frame: RGB image from camera
            depth_frame: Optional depth image (same size as color)
            camera_intrinsics: Optional camera intrinsics for 3D projection
            temperature: Generation temperature

        Returns:
            List of detected objects with full information
        """
        print("🔍 Detecting objects in scene...")
        start_time = time.time()

        # Prepare image
        pil_image = self._prepare_image(color_frame)

        # Store current frame for parallel processing
        self._current_color_frame = pil_image
        self._current_depth_frame = depth_frame
        self._current_intrinsics = camera_intrinsics

        # Performance optimization: Encode image once for all requests
        encode_start = time.time()
        self._cache_image_id = id(pil_image)
        if self.use_new_sdk and self.client:
            # Pre-encode image for new SDK (reused in parallel requests)
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            self._encoded_image_cache = img_byte_arr.getvalue()
        else:
            # Old SDK passes PIL image directly (no encoding needed)
            self._encoded_image_cache = None
        encode_time = time.time() - encode_start

        # Step 1 & 2: Stream object names and immediately dispatch workers
        names_start = time.time()
        print(f"   → Detecting objects (streaming with {self.max_parallel_requests} workers)...")

        detected_objects = []
        future_to_name = {}
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel_requests)

        def on_object_detected(object_name: str):
            """Callback when object name is detected - immediately start analysis worker"""
            print(f"      • Found: {object_name} → dispatching worker...")
            future = executor.submit(
                self._analyze_single_object,
                object_name,
                pil_image,
                depth_frame,
                camera_intrinsics,
                temperature
            )
            future_to_name[future] = object_name

        try:
            # Use streaming detection with immediate worker dispatch
            self._detect_object_names_streaming(pil_image, temperature, on_object_detected)

            if not future_to_name:
                print("   ⚠ No objects detected")
                self._encoded_image_cache = None  # Clear cache
                executor.shutdown(wait=False)
                return []

            names_time = time.time() - names_start
            print(f"   ✓ Detection phase complete in {names_time:.1f}s ({len(future_to_name)} objects)")
            print(f"   → Workers analyzing objects in parallel...")

            # Collect results as workers complete
            parallel_start = time.time()
            for future in concurrent.futures.as_completed(future_to_name):
                object_name = future_to_name[future]
                try:
                    detected_obj = future.result()
                    if detected_obj:
                        detected_objects.append(detected_obj)
                        # Update registry immediately (thread-safe)
                        self.registry.add_object(detected_obj)
                        print(f"      ✓ {object_name}: {len(detected_obj.affordances)} affordances, {len(detected_obj.interaction_points)} points")
                except Exception as e:
                    print(f"      ⚠ Failed to analyze {object_name}: {e}")

            parallel_time = time.time() - parallel_start

        finally:
            executor.shutdown(wait=True)

        total_time = time.time() - start_time

        # Clear encoding cache
        self._encoded_image_cache = None
        self._cache_image_id = None

        # Performance summary
        print(f"   ✓ Detection complete in {total_time:.1f}s")
        # `future_to_name` holds the mapping of dispatched workers to object names
        # so use it to report how many objects reused the encoded image cache.
        cached_count = len(future_to_name) if 'future_to_name' in locals() else 0
        print(f"      • Image encoding: {encode_time*1000:.0f}ms (cached for {cached_count} objects)")
        print(f"      • Object names: {names_time:.1f}s")
        print(f"      • Parallel analysis: {parallel_time:.1f}s ({len(detected_objects)} objects)")
        if len(detected_objects) > 0:
            avg_time = parallel_time / len(detected_objects)
            print(f"      • Average per object: {avg_time:.1f}s (effective with parallelism)")

        return detected_objects

    def _detect_object_names_streaming(
        self,
        image: Image.Image,
        temperature: Optional[float],
        callback
    ):
        """
        Detect names of all objects with streaming support.
        Calls callback for each object name as it's detected.

        Args:
            image: PIL Image
            temperature: Generation temperature
            callback: Function called with (object_name) for each detected object
        """
        prompt = """You are a robotic vision system. Identify all distinct objects in this image.

For each object, provide a descriptive name that includes identifying features (color, type, etc.).

Return object names ONE PER LINE in this format:
OBJECT: red cup
OBJECT: blue bottle
OBJECT: white bowl
OBJECT: wooden table
END

Important:
- Include color or distinguishing features in names
- Be specific (e.g., "red cup" not just "cup")
- Each object on a new line with "OBJECT: " prefix
- End with "END" on its own line
- Focus on manipulatable objects and surfaces"""

        try:
            # Use streaming API if available
            for chunk in self._generate_content_streaming(image, prompt, temperature):
                # Parse streaming chunks for object names
                lines = chunk.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('OBJECT:'):
                        object_name = line[7:].strip()
                        if object_name:
                            callback(object_name)
                    elif line == 'END':
                        break

        except Exception as e:
            print(f"   ⚠ Streaming object detection failed, falling back to batch: {e}")
            # Fallback: use non-streaming batch detection
            prompt = """You are a robotic vision system. Identify all distinct objects in this image.

For each object, provide a descriptive name that includes identifying features (color, type, etc.).

Return ONLY a JSON list of object names:
{
  "objects": ["red cup", "blue bottle", "white bowl", "wooden table"]
}

Important:
- Include color or distinguishing features in names
- Be specific (e.g., "red cup" not just "cup")
- List all visible objects
- Focus on manipulatable objects and surfaces"""

            try:
                response_text = self._generate_content(image, prompt, temperature)
                data = self._parse_json_response(response_text)
                object_names = data.get("objects", [])
                for name in object_names:
                    callback(name)
            except Exception as fallback_error:
                print(f"   ⚠ Batch fallback also failed: {fallback_error}")

    def _analyze_single_object(
        self,
        object_name: str,
        image: Image.Image,
        depth_frame: Optional[np.ndarray],
        camera_intrinsics: Optional[Any],
        temperature: Optional[float]
    ) -> Optional[DetectedObject]:
        """
        Analyze a single object: affordances, properties, and interaction points.

        Args:
            object_name: Name of object to analyze
            image: RGB image
            depth_frame: Optional depth image
            camera_intrinsics: Optional camera intrinsics
            temperature: Generation temperature

        Returns:
            DetectedObject with full information
        """
        # Build comprehensive prompt
        prompt = f"""You are a robotic manipulation system. Analyze the {object_name} in this image.

Provide detailed information:

1. Object position: Center point [y, x] in 0-1000 normalized coordinates
2. Affordances: What robot actions are possible with this object?
   - Common affordances: graspable, pourable, containable, pushable, pullable, openable, closable, supportable, stackable
3. Interaction points: For EACH affordance, identify the optimal interaction point
4. Properties: Color, size, material, state, etc.
5. Bounding box: [y1, x1, y2, x2] in 0-1000 normalized coordinates

Return in JSON format:
{{
  "object_type": "cup",
  "position": [450, 320],
  "bounding_box": [400, 280, 500, 360],
  "affordances": ["graspable", "pourable", "containable"],
  "interaction_points": {{
    "graspable": {{
      "position": [450, 340],
      "confidence": 0.95,
      "reasoning": "Handle on the right side provides stable grasp point"
    }},
    "pourable": {{
      "position": [450, 320],
      "confidence": 0.90,
      "reasoning": "Top rim center for tilting and pouring"
    }}
  }},
  "properties": {{
    "color": "red",
    "material": "ceramic",
    "size": "medium",
    "state": "upright"
  }},
  "confidence": 0.95
}}

Analyze the {object_name} based on its visual appearance."""

        try:
            response_text = self._generate_content(image, prompt, temperature)
            data = self._parse_json_response(response_text)

            if not data:
                return None

            # Handle case where VLM returns a list instead of dict
            if isinstance(data, list):
                print(f"      ⚠ Unexpected list response for {object_name}, skipping")
                return None

            # Extract affordances
            affordances = set(data.get("affordances", []))

            # Extract interaction points
            interaction_points = {}
            interaction_points_data = data.get("interaction_points", {})

            for affordance, point_data in interaction_points_data.items():
                pos_2d = point_data.get("position", [500, 500])

                # Compute 3D position if depth available. Prefer explicit depth_frame
                # passed to the worker; otherwise fall back to the cached frame on
                # the tracker instance. We do NOT include depth in prompts to the LLM.
                pos_3d = None
                use_depth = depth_frame if depth_frame is not None else self._current_depth_frame
                use_intrinsics = camera_intrinsics if camera_intrinsics is not None else self._current_intrinsics
                if use_depth is not None and use_intrinsics is not None:
                    pos_3d = compute_3d_position(
                        pos_2d,
                        use_depth,
                        use_intrinsics
                    )

                interaction_points[affordance] = InteractionPoint(
                    position_2d=pos_2d,
                    position_3d=pos_3d,
                    confidence=point_data.get("confidence", 0.5),
                    reasoning=point_data.get("reasoning", ""),
                    alternative_points=point_data.get("alternative_points", [])
                )

            # Extract center position
            pos_2d = data.get("position", [500, 500])
            pos_3d = None
            use_depth = depth_frame if depth_frame is not None else self._current_depth_frame
            use_intrinsics = camera_intrinsics if camera_intrinsics is not None else self._current_intrinsics
            if use_depth is not None and use_intrinsics is not None:
                pos_3d = compute_3d_position(pos_2d, use_depth, use_intrinsics)

            # Create object ID
            object_type = data.get("object_type", object_name.split()[0])
            object_id = self._generate_object_id(object_name, object_type)

            # Create DetectedObject
            detected_obj = DetectedObject(
                object_type=object_type,
                object_id=object_id,
                affordances=affordances,
                interaction_points=interaction_points,
                position_2d=pos_2d,
                position_3d=pos_3d,
                bounding_box_2d=data.get("bounding_box"),
                properties=data.get("properties", {}),
                confidence=data.get("confidence", 0.5)
            )

            return detected_obj

        except Exception as e:
            print(f"      ⚠ Failed to analyze {object_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_object(self, object_id: str) -> Optional[DetectedObject]:
        """
        Get object by ID from registry (thread-safe).

        Returns:
            DetectedObject or None if not found
        """
        return self.registry.get_object(object_id)

    def get_all_objects(self) -> List[DetectedObject]:
        """
        Get all objects from registry (thread-safe).

        Returns:
            List of all detected objects (snapshot)
        """
        return self.registry.get_all_objects()

    def get_objects_by_type(self, object_type: str) -> List[DetectedObject]:
        """
        Get all objects of a specific type (thread-safe).

        Args:
            object_type: Type to filter by (e.g., "cup", "bottle")

        Returns:
            List of objects matching the type
        """
        return self.registry.get_objects_by_type(object_type)

    def get_objects_with_affordance(self, affordance: str) -> List[DetectedObject]:
        """
        Get all objects that have a specific affordance (thread-safe).

        Args:
            affordance: Affordance to filter by (e.g., "graspable", "pourable")

        Returns:
            List of objects with the affordance
        """
        return self.registry.get_objects_with_affordance(affordance)

    def clear_registry(self):
        """Clear object registry (thread-safe)."""
        self.registry.clear()

    def save_detections(self, output_path: str, include_timestamp: bool = True):
        """
        Save detected objects to JSON file (thread-safe).

        Args:
            output_path: Path to save JSON file
            include_timestamp: Whether to include timestamp in filename

        Returns:
            Path to saved file
        """
        return self.registry.save_to_json(output_path, include_timestamp)

    def load_detections(self, input_path: str):
        """
        Load detected objects from JSON file (thread-safe).

        Args:
            input_path: Path to JSON file

        Returns:
            List of loaded DetectedObject instances
        """
        return self.registry.load_from_json(input_path)

    def update_interaction_point(
        self,
        object_id: str,
        affordance: str,
        task_context: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> Optional[InteractionPoint]:
        """
        Update interaction point for a specific object and affordance (thread-safe).

        Useful for refining interaction points based on task context.

        Args:
            object_id: Object to update
            affordance: Which affordance to update
            task_context: Optional task description for context
            temperature: Generation temperature

        Returns:
            Updated InteractionPoint or None if failed
        """
        obj = self.get_object(object_id)  # Already thread-safe
        if not obj:
            print(f"Object {object_id} not found in registry")
            return None

        if self._current_color_frame is None:
            print("No current frame cached")
            return None

        # Build prompt for specific interaction point
        prompt = f"""You are a robotic manipulation system. Identify the optimal interaction point for {affordance} action on the {obj.object_id}.

Object type: {obj.object_type}
Action: {affordance}"""

        if task_context:
            prompt += f"\nTask context: {task_context}"

        prompt += f"""

Analyze the {obj.object_id} and determine the best point to {affordance} based on:
- Object shape and features
- Intended action
- Task requirements (if provided)

Return in JSON format:
{{
  "position": [y, x],
  "confidence": 0.0-1.0,
  "reasoning": "Detailed explanation"
}}

Position is [y, x] in 0-1000 normalized coordinates."""

        try:
            response_text = self._generate_content(
                self._current_color_frame,
                prompt,
                temperature
            )
            data = self._parse_json_response(response_text)

            pos_2d = data.get("position", [500, 500])

            # Compute 3D if depth available
            pos_3d = None
            if self._current_depth_frame is not None and self._current_intrinsics is not None:
                pos_3d = compute_3d_position(
                    pos_2d,
                    self._current_depth_frame,
                    self._current_intrinsics
                )

            # Create updated interaction point
            interaction_point = InteractionPoint(
                position_2d=pos_2d,
                position_3d=pos_3d,
                confidence=data.get("confidence", 0.5),
                reasoning=data.get("reasoning", "")
            )

            # Update in registry (thread-safe)
            obj = self.registry.get_object(object_id)
            if obj:
                obj.interaction_points[affordance] = interaction_point
                self.registry.update_object(object_id, obj)

            return interaction_point

        except Exception as e:
            print(f"Failed to update interaction point: {e}")
            return None

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _generate_object_id(self, object_name: str, object_type: str) -> str:
        """Generate unique object ID (thread-safe)."""
        return self.registry.generate_unique_id(object_name, object_type)

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

    def _generate_content(
        self,
        image: Image.Image,
        prompt: str,
        temperature: Optional[float]
    ) -> str:
        """
        Generate content using streaming API and collect all chunks.
        This is a convenience wrapper around _generate_content_streaming.
        """
        chunks = []
        for chunk in self._generate_content_streaming(image, prompt, temperature):
            chunks.append(chunk)
        return ''.join(chunks)

    def _generate_content_streaming(
        self,
        image: Image.Image,
        prompt: str,
        temperature: Optional[float]
    ):
        """
        Generate content with streaming support.
        Yields text chunks as they arrive.
        """
        if self.use_new_sdk and self.client:
            # Use new GenAI SDK with streaming
            # Check if we can use cached encoded image
            if (self._encoded_image_cache is not None and
                self._cache_image_id == id(image)):
                img_bytes = self._encoded_image_cache
            else:
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()

            content_parts = [
                types.Part.from_bytes(data=img_bytes, mime_type='image/png'),
                prompt
            ]

            temp = temperature if temperature is not None else self.default_temperature
            config = types.GenerateContentConfig(
                temperature=temp,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=self.thinking_budget
                ) if self.thinking_budget > 0 else None
            )

            # Stream response
            response_stream = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=content_parts,
                config=config
            )

            for chunk in response_stream:
                if hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text

        else:
            # Old SDK: use stream=True if available
            temp = temperature if temperature is not None else self.default_temperature
            config = genai_old.GenerationConfig(
                temperature=temp,
            )

            response_stream = self.model.generate_content(
                [prompt, image],
                generation_config=config,
                stream=True
            )

            for chunk in response_stream:
                if hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response from LLM."""
        import json

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
            print(f"Failed to parse JSON: {e}")
            print(f"Response: {response_text[:500]}")
            return {}

