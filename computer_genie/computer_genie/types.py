"""Custom types for Computer Genie"""

from typing import NamedTuple, Union, Dict, Any, List, Optional
from pydantic import BaseModel

class Point(NamedTuple):
    x: int
    y: int

class Rectangle(NamedTuple):
    x: int
    y: int
    width: int
    height: int

class ScreenInfo(BaseModel):
    width: int
    height: int
    depth: int = 24

class MessageParam(BaseModel):
    role: str
    content: str

class ActSettings(BaseModel):
    max_steps: int = 50
    verbose: bool = False

class ResponseSchema(BaseModel):
    data: Dict[str, Any]

class Locator(BaseModel):
    type: str
    value: str