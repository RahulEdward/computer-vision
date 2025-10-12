"""
Intent Classifier

Classifies user intents and extracts entities from natural language
to understand what automation actions the user wants to perform.
"""

import re
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class Intent(Enum):
    """Possible user intents for automation."""
    CLICK = "click"
    TYPE = "type"
    WAIT = "wait"
    NAVIGATE = "navigate"
    SCROLL = "scroll"
    DRAG_DROP = "drag_drop"
    COPY = "copy"
    PASTE = "paste"
    OPEN = "open"
    CLOSE = "close"
    SAVE = "save"
    DELETE = "delete"
    SEARCH = "search"
    SELECT = "select"
    UPLOAD = "upload"
    DOWNLOAD = "download"
    CONDITION = "condition"
    LOOP = "loop"
    VARIABLE = "variable"
    FUNCTION = "function"
    UNKNOWN = "unknown"


class EntityType(Enum):
    """Types of entities that can be extracted."""
    ELEMENT = "element"
    TEXT = "text"
    URL = "url"
    FILE_PATH = "file_path"
    NUMBER = "number"
    TIME = "time"
    VARIABLE = "variable"
    CONDITION = "condition"


@dataclass
class Entity:
    """Represents an extracted entity."""
    entity_type: EntityType
    value: str
    confidence: float
    start_pos: int = 0
    end_pos: int = 0


@dataclass
class ClassificationResult:
    """Result of intent classification."""
    intent: Intent
    confidence: float
    entities: List[Entity]
    raw_text: str


class IntentClassifier:
    """
    Advanced intent classifier for understanding user automation requests
    from natural language descriptions.
    """
    
    def __init__(self):
        """Initialize the intent classifier."""
        self.logger = logging.getLogger(__name__)
        
        # Intent patterns for classification
        self.intent_patterns = {
            Intent.CLICK: [
                r"\b(?:click|tap|press|hit)\b",
                r"\b(?:select|choose)\b.*\b(?:button|link|menu)\b",
                r"\b(?:activate|trigger)\b"
            ],
            Intent.TYPE: [
                r"\b(?:type|enter|input|write)\b",
                r"\b(?:fill|complete)\b.*\b(?:field|form|box)\b",
                r"\b(?:insert|add)\b.*\b(?:text|content)\b"
            ],
            Intent.WAIT: [
                r"\b(?:wait|pause|delay)\b",
                r"\b(?:sleep|hold)\b",
                r"\b(?:until|for)\b.*\b(?:seconds?|minutes?|hours?)\b"
            ],
            Intent.NAVIGATE: [
                r"\b(?:go to|navigate to|visit|open)\b.*\b(?:url|website|page)\b",
                r"\b(?:browse to|load)\b",
                r"\bhttps?://\b"
            ],
            Intent.SCROLL: [
                r"\b(?:scroll|swipe)\b",
                r"\b(?:move|slide)\b.*\b(?:up|down|left|right)\b",
                r"\b(?:page up|page down)\b"
            ],
            Intent.DRAG_DROP: [
                r"\b(?:drag|move)\b.*\b(?:to|into|onto)\b",
                r"\b(?:drop|place)\b",
                r"\b(?:drag and drop|drag & drop)\b"
            ],
            Intent.COPY: [
                r"\b(?:copy|duplicate)\b",
                r"\bctrl\+c\b",
                r"\b(?:copy to clipboard)\b"
            ],
            Intent.PASTE: [
                r"\b(?:paste|insert)\b",
                r"\bctrl\+v\b",
                r"\b(?:paste from clipboard)\b"
            ],
            Intent.OPEN: [
                r"\b(?:open|launch|start|run)\b",
                r"\b(?:execute|begin)\b"
            ],
            Intent.CLOSE: [
                r"\b(?:close|exit|quit|end)\b",
                r"\b(?:shut down|terminate)\b"
            ],
            Intent.SAVE: [
                r"\b(?:save|store|preserve)\b",
                r"\bctrl\+s\b",
                r"\b(?:save as|export)\b"
            ],
            Intent.DELETE: [
                r"\b(?:delete|remove|erase)\b",
                r"\b(?:clear|clean)\b",
                r"\b(?:trash|discard)\b"
            ],
            Intent.SEARCH: [
                r"\b(?:search|find|look for)\b",
                r"\b(?:locate|discover)\b",
                r"\bctrl\+f\b"
            ],
            Intent.SELECT: [
                r"\b(?:select|highlight|mark)\b",
                r"\b(?:choose|pick)\b",
                r"\bctrl\+a\b"
            ],
            Intent.UPLOAD: [
                r"\b(?:upload|attach|send)\b.*\b(?:file|document|image)\b",
                r"\b(?:browse|choose file)\b"
            ],
            Intent.DOWNLOAD: [
                r"\b(?:download|save|get)\b.*\b(?:file|document|image)\b",
                r"\b(?:fetch|retrieve)\b"
            ],
            Intent.CONDITION: [
                r"\b(?:if|when|unless)\b",
                r"\b(?:condition|check)\b",
                r"\b(?:then|else|otherwise)\b"
            ],
            Intent.LOOP: [
                r"\b(?:repeat|loop|iterate)\b",
                r"\b(?:for each|while|until)\b",
                r"\b(?:again|multiple times)\b"
            ],
            Intent.VARIABLE: [
                r"\b(?:set|assign|store)\b.*\b(?:variable|value)\b",
                r"\b(?:remember|save)\b.*\b(?:as|to)\b",
                r"\b(?:variable|var)\b"
            ],
            Intent.FUNCTION: [
                r"\b(?:function|method|procedure)\b",
                r"\b(?:define|create)\b.*\b(?:function|method)\b",
                r"\b(?:call|invoke|execute)\b.*\b(?:function|method)\b"
            ]
        }
        
        # Entity patterns for extraction
        self.entity_patterns = {
            EntityType.ELEMENT: [
                r"(?:button|link|menu|field|box|input|textarea|dropdown|checkbox|radio|tab|window|dialog|popup|modal|panel|sidebar|header|footer|navigation|nav|toolbar|statusbar|progressbar|slider|toggle|switch|icon|image|video|audio|canvas|table|row|column|cell|list|item|card|badge|chip|avatar|tooltip|alert|notification|banner|breadcrumb|pagination|accordion|carousel|gallery|grid|form|label|legend|fieldset|select|option|optgroup|datalist|output|progress|meter|details|summary|mark|time|address|blockquote|cite|code|pre|kbd|samp|var|sub|sup|small|strong|em|b|i|u|s|del|ins|abbr|dfn|q|ruby|rt|rp|bdi|bdo|span|div|section|article|aside|main|header|footer|nav|figure|figcaption|picture|source|track|embed|object|param|iframe|canvas|svg|math|script|noscript|template|slot)",
                r"(?:the|a|an)\s+([^,.\s]+(?:\s+[^,.\s]+)*?)(?:\s+(?:button|link|menu|field|box|input))",
                r"\"([^\"]+)\"",
                r"'([^']+)'"
            ],
            EntityType.TEXT: [
                r"(?:text|content|message|title|label|caption|description|note|comment|value|data|information|info):\s*[\"']([^\"']+)[\"']",
                r"(?:type|enter|input|write)\s+[\"']([^\"']+)[\"']",
                r"[\"']([^\"']+)[\"']"
            ],
            EntityType.URL: [
                r"https?://[^\s]+",
                r"www\.[^\s]+",
                r"[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?"
            ],
            EntityType.FILE_PATH: [
                r"[A-Za-z]:\\[^<>:\"|?*\n\r]+",
                r"/[^<>:\"|?*\n\r]+",
                r"[./~][^<>:\"|?*\n\r]+"
            ],
            EntityType.NUMBER: [
                r"\b\d+(?:\.\d+)?\b",
                r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b"
            ],
            EntityType.TIME: [
                r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b",
                r"\b\d+\s*(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b",
                r"\b(?:today|tomorrow|yesterday|now|later|soon|immediately|instantly)\b"
            ],
            EntityType.VARIABLE: [
                r"\$[a-zA-Z_][a-zA-Z0-9_]*",
                r"\{[a-zA-Z_][a-zA-Z0-9_]*\}",
                r"(?:variable|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)"
            ],
            EntityType.CONDITION: [
                r"(?:if|when|unless)\s+(.+?)(?:\s+then|\s+do|$)",
                r"(?:condition|check)\s+(.+?)(?:\s+then|\s+do|$)"
            ]
        }
        
        # Common element selectors
        self.element_selectors = [
            "id", "class", "name", "tag", "xpath", "css",
            "text", "partial_text", "link_text", "partial_link_text"
        ]
    
    async def classify_intent(self, text: str) -> ClassificationResult:
        """
        Classify the intent of a natural language text.
        
        Args:
            text: Input text to classify
            
        Returns:
            ClassificationResult: Classification result with intent and entities
        """
        self.logger.info(f"Classifying intent for: {text}")
        
        try:
            # Normalize text
            normalized_text = self._normalize_text(text)
            
            # Classify intent
            intent, intent_confidence = await self._classify_intent(normalized_text)
            
            # Extract entities
            entities = await self._extract_entities(normalized_text)
            
            # Create result
            result = ClassificationResult(
                intent=intent,
                confidence=intent_confidence,
                entities=entities,
                raw_text=text
            )
            
            self.logger.info(f"Classified as {intent.value} with confidence {intent_confidence}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to classify intent: {e}")
            return ClassificationResult(
                intent=Intent.UNKNOWN,
                confidence=0.0,
                entities=[],
                raw_text=text
            )
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for better processing."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Expand contractions
        contractions = {
            "don't": "do not",
            "won't": "will not",
            "can't": "cannot",
            "shouldn't": "should not",
            "wouldn't": "would not",
            "couldn't": "could not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "hasn't": "has not",
            "haven't": "have not",
            "hadn't": "had not"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    async def _classify_intent(self, text: str) -> Tuple[Intent, float]:
        """Classify the intent of normalized text."""
        best_intent = Intent.UNKNOWN
        best_confidence = 0.0
        
        # Check each intent pattern
        for intent, patterns in self.intent_patterns.items():
            confidence = 0.0
            matches = 0
            
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    matches += 1
                    confidence += 1.0 / len(patterns)
            
            # Boost confidence based on number of matches
            if matches > 0:
                confidence = min(confidence * (1 + matches * 0.1), 1.0)
            
            # Update best intent if this is better
            if confidence > best_confidence:
                best_intent = intent
                best_confidence = confidence
        
        # Apply intent-specific boosts
        best_confidence = self._apply_intent_boosts(text, best_intent, best_confidence)
        
        return best_intent, best_confidence
    
    def _apply_intent_boosts(self, text: str, intent: Intent, confidence: float) -> float:
        """Apply intent-specific confidence boosts."""
        # Boost for specific keywords
        keyword_boosts = {
            Intent.CLICK: ["click", "button", "link"],
            Intent.TYPE: ["type", "enter", "input", "field"],
            Intent.NAVIGATE: ["navigate", "go to", "url", "website"],
            Intent.WAIT: ["wait", "pause", "delay", "sleep"],
            Intent.SCROLL: ["scroll", "swipe", "page"],
            Intent.CONDITION: ["if", "when", "then", "else"],
            Intent.LOOP: ["repeat", "loop", "for each", "while"]
        }
        
        if intent in keyword_boosts:
            for keyword in keyword_boosts[intent]:
                if keyword in text:
                    confidence = min(confidence + 0.1, 1.0)
        
        return confidence
    
    async def _extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from normalized text."""
        entities = []
        
        # Extract each entity type
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    # Determine entity value
                    if match.groups():
                        value = match.group(1)
                    else:
                        value = match.group(0)
                    
                    # Calculate confidence based on pattern specificity
                    confidence = self._calculate_entity_confidence(entity_type, value, pattern)
                    
                    # Create entity
                    entity = Entity(
                        entity_type=entity_type,
                        value=value.strip(),
                        confidence=confidence,
                        start_pos=match.start(),
                        end_pos=match.end()
                    )
                    
                    entities.append(entity)
        
        # Remove duplicate entities
        entities = self._deduplicate_entities(entities)
        
        # Sort by confidence
        entities.sort(key=lambda e: e.confidence, reverse=True)
        
        return entities
    
    def _calculate_entity_confidence(self, entity_type: EntityType, value: str, pattern: str) -> float:
        """Calculate confidence for an extracted entity."""
        base_confidence = 0.7
        
        # Boost confidence for specific patterns
        if entity_type == EntityType.URL and value.startswith(('http://', 'https://')):
            base_confidence = 0.95
        elif entity_type == EntityType.FILE_PATH and ('\\' in value or '/' in value):
            base_confidence = 0.9
        elif entity_type == EntityType.NUMBER and value.isdigit():
            base_confidence = 0.9
        elif entity_type == EntityType.ELEMENT and len(value) > 2:
            base_confidence = 0.8
        
        # Reduce confidence for very short values
        if len(value) < 2:
            base_confidence *= 0.5
        
        return min(base_confidence, 1.0)
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities."""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            key = (entity.entity_type, entity.value.lower())
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    async def extract_action_parameters(self, result: ClassificationResult) -> Dict[str, Any]:
        """
        Extract action parameters from classification result.
        
        Args:
            result: Classification result
            
        Returns:
            Dict: Action parameters for workflow execution
        """
        parameters = {
            'intent': result.intent.value,
            'confidence': result.confidence,
            'raw_text': result.raw_text
        }
        
        # Extract parameters based on intent
        if result.intent == Intent.CLICK:
            parameters.update(self._extract_click_parameters(result.entities))
        elif result.intent == Intent.TYPE:
            parameters.update(self._extract_type_parameters(result.entities))
        elif result.intent == Intent.NAVIGATE:
            parameters.update(self._extract_navigate_parameters(result.entities))
        elif result.intent == Intent.WAIT:
            parameters.update(self._extract_wait_parameters(result.entities))
        elif result.intent == Intent.SCROLL:
            parameters.update(self._extract_scroll_parameters(result.entities))
        elif result.intent == Intent.CONDITION:
            parameters.update(self._extract_condition_parameters(result.entities))
        elif result.intent == Intent.LOOP:
            parameters.update(self._extract_loop_parameters(result.entities))
        
        return parameters
    
    def _extract_click_parameters(self, entities: List[Entity]) -> Dict[str, Any]:
        """Extract parameters for click actions."""
        params = {}
        
        # Find element to click
        for entity in entities:
            if entity.entity_type == EntityType.ELEMENT:
                params['target'] = entity.value
                params['selector_type'] = 'text'
                break
        
        return params
    
    def _extract_type_parameters(self, entities: List[Entity]) -> Dict[str, Any]:
        """Extract parameters for type actions."""
        params = {}
        
        # Find text to type
        for entity in entities:
            if entity.entity_type == EntityType.TEXT:
                params['text'] = entity.value
                break
        
        # Find target element
        for entity in entities:
            if entity.entity_type == EntityType.ELEMENT:
                params['target'] = entity.value
                params['selector_type'] = 'text'
                break
        
        return params
    
    def _extract_navigate_parameters(self, entities: List[Entity]) -> Dict[str, Any]:
        """Extract parameters for navigate actions."""
        params = {}
        
        # Find URL
        for entity in entities:
            if entity.entity_type == EntityType.URL:
                params['url'] = entity.value
                break
        
        return params
    
    def _extract_wait_parameters(self, entities: List[Entity]) -> Dict[str, Any]:
        """Extract parameters for wait actions."""
        params = {}
        
        # Find time duration
        for entity in entities:
            if entity.entity_type == EntityType.TIME:
                params['duration'] = entity.value
                break
            elif entity.entity_type == EntityType.NUMBER:
                params['duration'] = f"{entity.value} seconds"
                break
        
        return params
    
    def _extract_scroll_parameters(self, entities: List[Entity]) -> Dict[str, Any]:
        """Extract parameters for scroll actions."""
        params = {}
        
        # Determine scroll direction from text
        text = ' '.join([e.value for e in entities])
        if 'up' in text:
            params['direction'] = 'up'
        elif 'down' in text:
            params['direction'] = 'down'
        elif 'left' in text:
            params['direction'] = 'left'
        elif 'right' in text:
            params['direction'] = 'right'
        else:
            params['direction'] = 'down'  # default
        
        return params
    
    def _extract_condition_parameters(self, entities: List[Entity]) -> Dict[str, Any]:
        """Extract parameters for condition actions."""
        params = {}
        
        # Find condition text
        for entity in entities:
            if entity.entity_type == EntityType.CONDITION:
                params['condition'] = entity.value
                break
        
        return params
    
    def _extract_loop_parameters(self, entities: List[Entity]) -> Dict[str, Any]:
        """Extract parameters for loop actions."""
        params = {}
        
        # Find loop count
        for entity in entities:
            if entity.entity_type == EntityType.NUMBER:
                params['count'] = int(entity.value)
                break
        
        return params