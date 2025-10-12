"""Vision components for Computer Genie"""

from PIL import Image
import numpy as np
from typing import List, Tuple, Dict, Any

class Screenshot:
    """Screenshot capture utility"""
    
    def __init__(self, display: int = 1):
        self.display = display
    
    async def capture(self) -> Image.Image:
        """Capture screenshot"""
        import mss
        with mss.mss() as sct:
            monitor = sct.monitors[self.display]
            screenshot = sct.grab(monitor)
            return Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

class OCR:
    """Optical Character Recognition"""
    
    def __init__(self):
        pass
    
    async def extract_text(self, image: np.ndarray) -> str:
        """Extract text from image"""
        try:
            import pytesseract
            return pytesseract.image_to_string(image)
        except ImportError:
            return "OCR not available - pytesseract not installed"
        except Exception as e:
            # Handle Tesseract not found error gracefully
            if "tesseract is not installed" in str(e).lower():
                return "OCR not available - Tesseract not installed. Please install Tesseract OCR."
            return f"OCR error: {str(e)}"
    
    async def find_text(self, image: np.ndarray, text: str) -> List[Tuple[int, int]]:
        """Find text locations in image"""
        # Placeholder implementation
        return []

class ElementDetector:
    """UI element detection"""
    
    def __init__(self):
        pass
    
    async def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect UI elements in image"""
        # Placeholder implementation
        return []

__all__ = ['Screenshot', 'OCR', 'ElementDetector']