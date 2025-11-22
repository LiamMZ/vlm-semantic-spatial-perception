"""
Async object tracking system using Gemini Robotics VLM.

This module provides async object detection, affordance analysis, and interaction
point detection for robotic manipulation tasks using Google Gen AI's native async support.
"""

import time
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import io
import yaml

import numpy as np
from PIL import Image

# Import the new Google GenAI SDK
from google import genai
from google.genai import types

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
    
    # Default prompts configuration path
    DEFAULT_PROMPTS_CONFIG = str(Path(__file__).parent.parent.parent / "config" / "prompts_config.yaml")

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "auto",
        default_temperature: float = 0.5,
        thinking_budget: int = -1,
        max_parallel_requests: int = 5,
        crop_target_size: int = 512,
        enable_affordance_caching: bool = True,
        fast_mode: bool = False,
        pddl_predicates: Optional[List[str]] = None,
        prompts_config_path: Optional[str] = None
    ):
        """
        Initialize object tracker.

        Args:
            api_key: Google AI API key
            model_name: "auto" to use robotics model if available
            default_temperature: Generation temperature (0.0-1.0)
            thinking_budget: Token budget for extended thinking (0=disabled)
            max_parallel_requests: Max parallel VLM requests for object details
            crop_target_size: Resize crops to this size before VLM (0=disable, default 512)
            enable_affordance_caching: Cache affordance results for similar objects
            fast_mode: Skip detailed interaction points, only detect affordances
            pddl_predicates: List of PDDL predicate names to extract from objects (e.g., ["clean", "dirty", "opened"])
            prompts_config_path: Path to prompts config YAML file (defaults to config/prompts_config.yaml)
        """
        self.api_key = api_key
        self.max_parallel_requests = max_parallel_requests
        self.crop_target_size = crop_target_size
        self.enable_affordance_caching = enable_affordance_caching
        self.fast_mode = fast_mode

        # PDDL predicate tracking
        self.pddl_predicates: List[str] = pddl_predicates or []
        
        # Load prompts configuration
        if prompts_config_path is None:
            prompts_config_path = self.DEFAULT_PROMPTS_CONFIG
        
        with open(prompts_config_path, 'r') as f:
            self.prompts = yaml.safe_load(f)
        
        print(f"✓ Loaded prompts from {prompts_config_path}")

        # Auto-select best model
        if model_name == "auto":
            self.model_name = self.ROBOTICS_MODEL
            print(f"ℹ ObjectTracker using: {self.model_name}")
        else:
            self.model_name = model_name

        # Initialize GenAI SDK
        self.client = genai.Client(api_key=api_key)
        print(f"✓ ObjectTracker initialized with GenAI SDK")

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

        # Affordance caching: object_type -> {affordances, properties}
        self._affordance_cache: Dict[str, Dict[str, Any]] = {}

    def set_pddl_predicates(self, predicates: List[str]) -> None:
        """
        Set the list of PDDL predicates to track for objects.

        Args:
            predicates: List of predicate names (e.g., ["clean", "dirty", "opened", "filled"])
        """
        self.pddl_predicates = predicates
        print(f"ℹ PDDL predicates updated: {predicates}")

    def add_pddl_predicate(self, predicate: str) -> None:
        """
        Add a single PDDL predicate to track.

        Args:
            predicate: Predicate name to add
        """
        if predicate not in self.pddl_predicates:
            self.pddl_predicates.append(predicate)
            print(f"ℹ Added PDDL predicate: {predicate}")

    def remove_pddl_predicate(self, predicate: str) -> None:
        """
        Remove a PDDL predicate from tracking.

        Args:
            predicate: Predicate name to remove
        """
        if predicate in self.pddl_predicates:
            self.pddl_predicates.remove(predicate)
            print(f"ℹ Removed PDDL predicate: {predicate}")

    def get_pddl_predicates(self) -> List[str]:
        """Get the current list of tracked PDDL predicates."""
        return self.pddl_predicates.copy()

    async def detect_objects(
        self,
        color_frame: Union[np.ndarray, Image.Image],
        depth_frame: Optional[np.ndarray] = None,
        camera_intrinsics: Optional[Any] = None,
        temperature: Optional[float] = None
    ) -> List[DetectedObject]:
        """
        Detect all objects in scene with affordances and interaction points (async).

        Uses Google Gen AI's native async support for true concurrent VLM requests.

        This is the main entry point. It:
        1. Detects all object names in the scene
        2. For each object concurrently using async/await:
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
        print("🔍 Detecting objects in scene (async)...")
        start_time = time.time()

        # Prepare image
        pil_image = self._prepare_image(color_frame)

        # Store current frame for analysis
        self._current_color_frame = pil_image
        self._current_depth_frame = depth_frame
        self._current_intrinsics = camera_intrinsics

        # Performance optimization: Encode image once for all requests
        encode_start = time.time()
        self._cache_image_id = id(pil_image)
        # Pre-encode image (reused in parallel requests)
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        self._encoded_image_cache = img_byte_arr.getvalue()
        encode_time = time.time() - encode_start

        # Step 1: Stream object names and collect them
        names_start = time.time()
        print(f"   → Detecting objects (async streaming)...")

        object_data_list = []

        async def on_object_detected(object_name: str, bounding_box: Optional[List[int]]):
            """Callback when object detected"""
            bbox_str = f"bbox={bounding_box}" if bounding_box else "no bbox"
            print(f"      • Found: {object_name} ({bbox_str})")
            object_data_list.append((object_name, bounding_box))

        # Use async streaming detection
        await self._detect_object_names_streaming(pil_image, temperature, on_object_detected)

        if not object_data_list:
            print("   ⚠ No objects detected")
            self._encoded_image_cache = None
            return []

        names_time = time.time() - names_start
        print(f"   ✓ Detection phase complete in {names_time:.1f}s ({len(object_data_list)} objects)")
        print(f"   → Analyzing objects concurrently (async)...")

        # Step 2: Analyze all objects concurrently using asyncio.gather
        parallel_start = time.time()

        tasks = [
            self._analyze_single_object(
                object_name,
                pil_image,
                depth_frame,
                camera_intrinsics,
                temperature,
                bounding_box
            )
            for object_name, bounding_box in object_data_list
        ]

        # Run all analysis tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        detected_objects = []
        for i, result in enumerate(results):
            object_name = object_data_list[i][0]
            if isinstance(result, Exception):
                print(f"      ⚠ Failed to analyze {object_name}: {result}")
            elif result is not None:
                detected_objects.append(result)
                self.registry.add_object(result)
                print(f"      ✓ {object_name}: {len(result.affordances)} affordances, {len(result.interaction_points)} points")

        parallel_time = time.time() - parallel_start
        total_time = time.time() - start_time

        # Clear encoding cache
        self._encoded_image_cache = None
        self._cache_image_id = None

        # Performance summary
        print(f"   ✓ Detection complete in {total_time:.1f}s")
        print(f"      • Image encoding: {encode_time*1000:.0f}ms")
        print(f"      • Object names: {names_time:.1f}s")
        print(f"      • Async analysis: {parallel_time:.1f}s ({len(detected_objects)} objects)")
        if len(detected_objects) > 0:
            avg_time = parallel_time / len(detected_objects)
            print(f"      • Average per object: {avg_time:.1f}s (effective with async)")

        return detected_objects

    async def _detect_object_names_streaming(
        self,
        image: Image.Image,
        temperature: Optional[float],
        callback
    ):
        """
        Detect names and bounding boxes with async streaming.

        Args:
            image: PIL Image
            temperature: Generation temperature
            callback: Async function called with (object_name, bounding_box)
        """
        prompt = self.prompts['detection']['streaming']

        try:
            # Use async streaming API
            async for chunk in self._generate_content_streaming(image, prompt, temperature):
                # Parse streaming chunks for object names and bounding boxes
                lines = chunk.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('OBJECT:'):
                        # Parse format: "OBJECT: name | [y1, x1, y2, x2]"
                        content = line[7:].strip()
                        if '|' in content:
                            parts = content.split('|')
                            object_name = parts[0].strip()
                            try:
                                import json
                                bbox_str = parts[1].strip()
                                bounding_box = json.loads(bbox_str)
                                if object_name and isinstance(bounding_box, list) and len(bounding_box) == 4:
                                    await callback(object_name, bounding_box)
                            except:
                                # If bbox parsing fails, use None
                                if object_name:
                                    await callback(object_name, None)
                        else:
                            # No bbox provided, use None
                            if content:
                                await callback(content, None)
                    elif line == 'END':
                        break

        except Exception as e:
            print(f"   ⚠ Streaming object detection failed, falling back to batch: {e}")
            # Fallback: use non-streaming batch detection
            prompt = self.prompts['detection']['batch']

            try:
                response_text = await self._generate_content(image, prompt, temperature)
                data = self._parse_json_response(response_text)
                objects_data = data.get("objects", [])
                for obj_data in objects_data:
                    if isinstance(obj_data, dict):
                        name = obj_data.get("name")
                        bbox = obj_data.get("bbox")
                        if name:
                            await callback(name, bbox)
                    elif isinstance(obj_data, str):
                        # Fallback if old format returned
                        await callback(obj_data, None)
            except Exception as fallback_error:
                print(f"   ⚠ Batch fallback also failed: {fallback_error}")

    async def _analyze_single_object(
        self,
        object_name: str,
        image: Image.Image,
        depth_frame: Optional[np.ndarray],
        camera_intrinsics: Optional[Any],
        temperature: Optional[float],
        bounding_box: Optional[List[int]] = None
    ) -> Optional[DetectedObject]:
        """
        Analyze a single object: affordances, properties, and interaction points.

        If bounding_box is provided, crops the image to that region and back-projects
        interaction points to full image coordinates.

        Args:
            object_name: Name of object to analyze
            image: RGB image (full size)
            depth_frame: Optional depth image (full size)
            camera_intrinsics: Optional camera intrinsics
            temperature: Generation temperature
            bounding_box: Optional bounding box [y1, x1, y2, x2] in 0-1000 normalized coords

        Returns:
            DetectedObject with full information
        """
        # Crop image if bounding box provided
        analysis_image = image
        crop_offset = None

        if bounding_box is not None:
            # Convert normalized bbox to pixel coordinates
            from .utils.coordinates import normalized_to_pixel
            img_height, img_width = image.height, image.width

            y1, x1, y2, x2 = bounding_box
            # Convert each corner to pixels
            py1, px1 = normalized_to_pixel([y1, x1], (img_height, img_width))
            py2, px2 = normalized_to_pixel([y2, x2], (img_height, img_width))

            # Ensure valid crop bounds
            px1, px2 = max(0, min(px1, px2)), min(img_width, max(px1, px2))
            py1, py2 = max(0, min(py1, py2)), min(img_height, max(py1, py2))

            # Crop the image (PIL uses left, upper, right, lower)
            analysis_image = image.crop((px1, py1, px2, py2))
            crop_offset = (py1, px1, py2, px2)  # Store for back-projection

            # Resize crop to target size for faster VLM processing
            if self.crop_target_size > 0:
                analysis_image = analysis_image.resize(
                    (self.crop_target_size, self.crop_target_size),
                    Image.Resampling.LANCZOS
                )

        # Check affordance cache for this object type
        object_type_guess = object_name.split()[-1] if ' ' in object_name else object_name
        cached_data = None
        if self.enable_affordance_caching and object_type_guess in self._affordance_cache:
            cached_data = self._affordance_cache[object_type_guess]

        # Build prompt based on mode and cache
        crop_note = " (This is a cropped region focusing on the object.)" if crop_offset else ""

        # Build PDDL predicate section if predicates are specified
        pddl_section = ""
        pddl_example = ""
        if self.pddl_predicates:
            pddl_list = ", ".join(self.pddl_predicates)
            pddl_section = self.prompts['pddl']['section_template'].format(pddl_list=pddl_list)
            pddl_example = self.prompts['pddl']['example_template']

        if self.fast_mode:
            # Fast mode: only detect affordances and properties, no interaction points
            prompt = self.prompts['analysis']['fast_mode'].format(
                object_name=object_name,
                crop_note=crop_note,
                pddl_section=pddl_section,
                pddl_example=pddl_example
            )
        elif cached_data:
            # Use cached affordances, only detect interaction points
            prompt = self.prompts['analysis']['cached_mode'].format(
                object_name=object_name,
                crop_note=crop_note,
                cached_affordances=cached_data['affordances'],
                object_type=cached_data['object_type']
            )
        else:
            # Full analysis
            prompt = self.prompts['analysis']['full'].format(
                object_name=object_name,
                crop_note=crop_note,
                pddl_section=pddl_section,
                pddl_example=pddl_example
            )

        try:
            response_text = await self._generate_content(analysis_image, prompt, temperature)
            data = self._parse_json_response(response_text)

            if not data:
                return None

            # Handle case where VLM returns a list instead of dict
            if isinstance(data, list):
                print(f"      ⚠ Unexpected list response for {object_name}, skipping")
                return None

            # Extract affordances (use cached if available)
            if cached_data:
                affordances = set(cached_data["affordances"])
            else:
                affordances = set(data.get("affordances", []))

            # Helper function to back-project coordinates from crop to full image
            def backproject_to_full_image(crop_pos_2d: List[int]) -> List[int]:
                """Convert position from cropped image coords to full image coords."""
                if crop_offset is None:
                    return crop_pos_2d  # No cropping, return as-is

                from .utils.coordinates import normalized_to_pixel, pixel_to_normalized
                crop_y, crop_x = crop_pos_2d
                crop_height = crop_offset[2] - crop_offset[0]
                crop_width = crop_offset[3] - crop_offset[1]

                # Convert normalized coords to pixels in crop
                pixel_y, pixel_x = normalized_to_pixel([crop_y, crop_x], (crop_height, crop_width))

                # Add crop offset to get full image pixels
                full_pixel_y = pixel_y + crop_offset[0]
                full_pixel_x = pixel_x + crop_offset[1]

                # Convert back to normalized coords in full image
                img_height, img_width = image.height, image.width
                full_norm_y, full_norm_x = pixel_to_normalized(
                    [full_pixel_y, full_pixel_x],
                    (img_height, img_width)
                )

                return [full_norm_y, full_norm_x]

            # Extract interaction points
            interaction_points = {}
            interaction_points_data = data.get("interaction_points", {})

            for affordance, point_data in interaction_points_data.items():
                pos_2d_crop = point_data.get("position", [500, 500])
                # Back-project to full image coordinates
                pos_2d = backproject_to_full_image(pos_2d_crop)

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
            pos_2d_crop = data.get("position", [500, 500])
            pos_2d = backproject_to_full_image(pos_2d_crop)
            pos_3d = None
            use_depth = depth_frame if depth_frame is not None else self._current_depth_frame
            use_intrinsics = camera_intrinsics if camera_intrinsics is not None else self._current_intrinsics
            if use_depth is not None and use_intrinsics is not None:
                pos_3d = compute_3d_position(pos_2d, use_depth, use_intrinsics)

            # Create object ID
            object_type = data.get("object_type", object_name.split()[0])
            object_id = self._generate_object_id(object_name, object_type)

            # Use bounding box from initial detection, not from detailed analysis
            # (since detailed analysis was done on cropped image)
            final_bbox = bounding_box if bounding_box is not None else data.get("bounding_box")

            # Update affordance cache if enabled and not already cached
            if self.enable_affordance_caching and not cached_data and affordances:
                self._affordance_cache[object_type] = {
                    "object_type": object_type,
                    "affordances": list(affordances),
                    "properties": data.get("properties", {})
                }

            # Extract PDDL state predicates if present
            pddl_state = data.get("pddl_state", {})

            # Create DetectedObject
            detected_obj = DetectedObject(
                object_type=object_type,
                object_id=object_id,
                affordances=affordances,
                interaction_points=interaction_points,
                position_2d=pos_2d,
                position_3d=pos_3d,
                bounding_box_2d=final_bbox,
                properties=data.get("properties", {}),
                pddl_state=pddl_state,
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

    async def update_interaction_point(
        self,
        object_id: str,
        affordance: str,
        task_context: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> Optional[InteractionPoint]:
        """
        Update interaction point for a specific object and affordance (async).

        Useful for refining interaction points based on task context.

        Args:
            object_id: Object to update
            affordance: Which affordance to update
            task_context: Optional task description for context
            temperature: Generation temperature

        Returns:
            Updated InteractionPoint or None if failed
        """
        obj = self.get_object(object_id)
        if not obj:
            print(f"Object {object_id} not found in registry")
            return None

        if self._current_color_frame is None:
            print("No current frame cached")
            return None

        # Build prompt for specific interaction point
        task_context_section = f"\nTask context: {task_context}" if task_context else ""
        
        prompt = self.prompts['interaction']['update'].format(
            affordance=affordance,
            object_id=obj.object_id,
            object_type=obj.object_type,
            task_context_section=task_context_section
        )

        try:
            response_text = await self._generate_content(
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

    async def _generate_content(
        self,
        image: Image.Image,
        prompt: str,
        temperature: Optional[float]
    ) -> str:
        """
        Generate content using async streaming API and collect all chunks.
        This is a convenience wrapper around _generate_content_streaming.
        """
        chunks = []
        async for chunk in self._generate_content_streaming(image, prompt, temperature):
            chunks.append(chunk)
        return ''.join(chunks)

    async def _generate_content_streaming(
        self,
        image: Image.Image,
        prompt: str,
        temperature: Optional[float]
    ):
        """
        Generate content with async streaming support using client.aio.
        Yields text chunks as they arrive.
        """
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
            )
        )

        # Use async streaming via client.aio
        stream = await self.client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=content_parts,
            config=config
        )
        async for chunk in stream:
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

    async def aclose(self):
        """Close async client resources."""
        if self.client:
            await self.client.aio.aclose()

