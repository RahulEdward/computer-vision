"""Computer Genie - AI-powered computer automation framework""" 

from computer_genie.__version__ import __version__ 
from computer_genie.core.agent import VisionAgent, AndroidAgent, WebAgent 
from computer_genie.models import ModelRegistry 
from computer_genie.locators import Text, Element, Image, XPath 
from computer_genie.types import Point, Rectangle, ScreenInfo 
from computer_genie.exceptions import GenieException 

__all__ = [ 
    "__version__", 
    "VisionAgent", 
    "AndroidAgent", 
    "WebAgent", 
    "ModelRegistry", 
    "Text", 
    "Element", 
    "Image", 
    "XPath", 
    "Point", 
    "Rectangle", 
    "ScreenInfo", 
    "GenieException", 
]