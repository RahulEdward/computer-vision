"""Computer Genie's native AI model implementation"""

import asyncio
from typing import Optional, Union, List, Type, Any, Dict
from PIL import Image
import numpy as np
import torch
import cv2
from transformers import AutoModel, AutoTokenizer

from computer_genie.models.base import ActModel, GetModel, LocateModel
from computer_genie.types import Point, MessageParam, ResponseSchema, Locator, ActSettings
from computer_genie.vision import OCR, ElementDetector
from computer_genie.utils import retry_with_backoff, setup_logger

logger = setup_logger(__name__)

class GenieVisionModel(ActModel, GetModel, LocateModel):
    """Computer Genie's native vision model with all capabilities"""
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__("genie-vision")
        self.model_path = model_path or "genie-ai/vision-v1"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Components
        self.ocr = OCR()
        self.element_detector = ElementDetector()
        self.model = None
        self.tokenizer = None
        
    async def initialize(self):
        """Load model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Initialized Genie Vision Model on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            # Fallback to CPU implementation
            self.device = "cpu"
    
    async def act(
        self,
        messages: List[MessageParam],
        context: Dict[str, Any],
        settings: Optional[ActSettings] = None
    ) -> None:
        """Execute complex instruction using planning and tool use"""
        
        instruction = messages[-1]["content"] if messages else ""
        tools = context.get("tools", {})
        screenshot_fn = context.get("screenshot_fn")
        max_steps = context.get("max_steps", 50)
        
        logger.info(f"Executing instruction: {instruction}")
        
        # Create execution plan
        plan = await self._create_plan(instruction, screenshot_fn)
        
        # Execute plan steps
        for i, step in enumerate(plan[:max_steps]):
            logger.debug(f"Executing step {i+1}: {step['action']}")
            
            try:
                if step["action"] == "click":
                    await self._execute_click(step["target"], tools["os"], screenshot_fn)
                elif step["action"] == "type":
                    await self._execute_type(step["text"], tools["os"])
                elif step["action"] == "wait":
                    await asyncio.sleep(step.get("duration", 1))
                elif step["action"] == "screenshot":
                    await self._verify_state(step.get("expected"), screenshot_fn)
                    
            except Exception as e:
                logger.error(f"Step {i+1} failed: {e}")
                if settings and settings.stop_on_error:
                    raise
    
    async def _create_plan(self, instruction: str, screenshot_fn) -> List[Dict]:
        """Create execution plan from instruction"""
        try:
            # Take screenshot for context
            screenshot = await screenshot_fn()
            
            # Analyze screen state
            screen_elements = await self.element_detector.detect(screenshot)
            screen_text = await self.ocr.extract_text(screenshot)
            
            # Log OCR status
            if "not available" in screen_text or "error" in screen_text.lower():
                logger.warning(f"OCR issue: {screen_text}")
                screen_text = ""  # Continue without OCR
            
        except Exception as e:
            logger.error(f"Error analyzing screen: {e}")
            screen_elements = []
            screen_text = ""
        
        # Generate plan using model or heuristics
        plan = []
        
        # Simple heuristic-based planning (replace with model inference)
        if "search" in instruction.lower():
            plan.append({"action": "click", "target": "search field"})
            plan.append({"action": "type", "text": self._extract_search_term(instruction)})
            plan.append({"action": "click", "target": "search button"})
        elif "login" in instruction.lower():
            plan.append({"action": "click", "target": "username field"})
            plan.append({"action": "type", "text": "username"})
            plan.append({"action": "click", "target": "password field"})
            plan.append({"action": "type", "text": "password"})
            plan.append({"action": "click", "target": "login button"})
        elif "describe" in instruction.lower() or "see" in instruction.lower():
            # For describe/see instructions, just take a screenshot and provide basic info
            plan.append({"action": "screenshot", "expected": "screen captured"})
        else:
            # Default fallback - just take a screenshot
            plan.append({"action": "screenshot", "expected": "basic screen analysis"})
        
        return plan
    
    async def _execute_click(self, target: str, os_controller, screenshot_fn):
        """Execute click action"""
        screenshot = await screenshot_fn()
        point = await self.locate(target, screenshot)
        if point:
            await os_controller.click(point.x, point.y)
    
    async def _execute_type(self, text: str, os_controller):
        """Execute type action"""
        await os_controller.type_text(text)
    
    async def _verify_state(self, expected: str, screenshot_fn):
        """Verify screen state"""
        try:
            screenshot = await screenshot_fn()
            
            if "screen captured" in expected:
                # For screenshot actions, provide basic screen info
                logger.info(f"Screenshot captured: {screenshot.size if screenshot else 'Failed'}")
                return True
            elif "basic screen analysis" in expected:
                # Provide basic analysis without OCR
                logger.info("Performing basic screen analysis...")
                logger.info(f"Screen size: {screenshot.size if screenshot else 'Unknown'}")
                return True
            else:
                # Try to verify using the get method
                actual = await self.get(f"Is {expected} visible?", screenshot)
                return actual.lower() in ["yes", "true"]
        except Exception as e:
            logger.error(f"Error verifying state: {e}")
            return False
    
    def _extract_search_term(self, instruction: str) -> str:
        """Extract search term from instruction"""
        # Simple extraction logic
        if "for" in instruction:
            return instruction.split("for")[-1].strip()
        return ""
    
    @retry_with_backoff(max_retries=3)
    async def get(
        self,
        query: str,
        image: Union[Image.Image, np.ndarray],
        response_schema: Optional[Type[ResponseSchema]] = None
    ) -> Union[str, ResponseSchema]:
        """Extract information from image"""
        
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Extract text using OCR
        text_content = await self.ocr.extract_text(image_np)
        
        # Detect elements
        elements = await self.element_detector.detect(image_np)
        
        # Process query with model or heuristics
        if self.model:
            # Use model for complex queries
            result = await self._query_model(query, image_np, text_content, elements)
        else:
            # Fallback to heuristics
            result = await self._query_heuristic(query, text_content, elements)
        
        # Format response according to schema
        if response_schema:
            return self._format_response(result, response_schema)
        
        return result
    
    async def _query_model(self, query: str, image: np.ndarray, text: str, elements: List) -> str:
        """Query model for information extraction"""
        # Implement model inference
        # This is a placeholder - implement actual model inference
        return f"Model response to: {query}"
    
    async def _query_heuristic(self, query: str, text: str, elements: List) -> str:
        """Use heuristics for simple queries"""
        query_lower = query.lower()
        
        # Check for specific patterns
        if "how many" in query_lower:
            if "button" in query_lower:
                buttons = [e for e in elements if e["type"] == "button"]
                return str(len(buttons))
            elif "text" in query_lower or "word" in query_lower:
                words = text.split()
                return str(len(words))
        
        elif "what is" in query_lower or "what's" in query_lower:
            if "title" in query_lower or "heading" in query_lower:
                # Find largest text
                lines = text.split('\n')
                return lines[0] if lines else "No title found"
            elif "url" in query_lower:
                # Look for URL pattern
                import re
                url_pattern = r'https?://[^\s]+'
                urls = re.findall(url_pattern, text)
                return urls[0] if urls else "No URL found"
        
        elif any(word in query_lower for word in ["is there", "can you see", "is visible"]):
            # Boolean queries
            search_terms = query_lower.split()
            for term in search_terms:
                if term in text.lower():
                    return "yes"
            return "no"
        
        return text[:200] if text else "No information found"
    
    def _format_response(self, result: str, response_schema: Type[ResponseSchema]) -> ResponseSchema:
        """Format response according to schema"""
        if response_schema == bool:
            return result.lower() in ["yes", "true", "1"]
        elif response_schema == int:
            try:
                return int(result)
            except:
                return 0
        elif response_schema == float:
            try:
                return float(result)
            except:
                return 0.0
        else:
            # For Pydantic models
            try:
                return response_schema.parse_raw(result)
            except:
                return response_schema()
    
    async def locate(
        self,
        locator: Union[str, Locator],
        image: Union[Image.Image, np.ndarray],
        timeout: float = 10.0
    ) -> Optional[Point]:
        """Locate element in image"""
        
        # Convert to numpy
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Parse locator
        if isinstance(locator, str):
            target_text = locator
            target_type = None
        else:
            target_text = locator.text if hasattr(locator, 'text') else str(locator)
            target_type = locator.type if hasattr(locator, 'type') else None
        
        # Try OCR-based location first
        if target_text:
            text_locations = await self.ocr.find_text(image_np, target_text)
            if text_locations:
                return Point(x=text_locations[0][0], y=text_locations[0][1])
        
        # Try element detection
        elements = await self.element_detector.detect(image_np)
        
        for element in elements:
            # Match by text
            if target_text and element.get("text", "").lower() == target_text.lower():
                bbox = element["bbox"]
                center_x = (bbox[0] + bbox[2]) // 2
                center_y = (bbox[1] + bbox[3]) // 2
                return Point(x=center_x, y=center_y)
            
            # Match by type
            if target_type and element.get("type") == target_type:
                bbox = element["bbox"]
                center_x = (bbox[0] + bbox[2]) // 2
                center_y = (bbox[1] + bbox[3]) // 2
                return Point(x=center_x, y=center_y)
        
        return None
    
    async def locate_all(
        self,
        locator: Union[str, Locator],
        image: Union[Image.Image, np.ndarray]
    ) -> List[Point]:
        """Locate all matching elements"""
        
        points = []
        
        # Convert to numpy
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Parse locator
        if isinstance(locator, str):
            target_text = locator
        else:
            target_text = str(locator)
        
        # Find all text occurrences
        text_locations = await self.ocr.find_text(image_np, target_text)
        for loc in text_locations:
            points.append(Point(x=loc[0], y=loc[1]))
        
        # Find all matching elements
        elements = await self.element_detector.detect(image_np)
        for element in elements:
            if element.get("text", "").lower() == target_text.lower():
                bbox = element["bbox"]
                center_x = (bbox[0] + bbox[2]) // 2
                center_y = (bbox[1] + bbox[3]) // 2
                points.append(Point(x=center_x, y=center_y))
        
        return points
    
    async def cleanup(self):
        """Cleanup model resources"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        torch.cuda.empty_cache()