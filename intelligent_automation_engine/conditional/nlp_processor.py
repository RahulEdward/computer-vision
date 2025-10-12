"""
Natural Language Processor

Processes natural language text to extract conditional logic patterns,
understand context, and prepare text for condition parsing.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import spacy
from collections import defaultdict, Counter


class LanguagePattern(Enum):
    """Types of language patterns for conditional logic."""
    IF_THEN = "if_then"
    IF_THEN_ELSE = "if_then_else"
    WHEN_THEN = "when_then"
    UNLESS = "unless"
    WHILE = "while"
    UNTIL = "until"
    SWITCH_CASE = "switch_case"
    EITHER_OR = "either_or"
    NEITHER_NOR = "neither_nor"
    BOTH_AND = "both_and"


class IntentType(Enum):
    """Types of user intents in conditional statements."""
    CONDITION = "condition"
    ACTION = "action"
    COMPARISON = "comparison"
    LOGICAL_OPERATION = "logical_operation"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    QUANTITATIVE = "quantitative"
    QUALITATIVE = "qualitative"


@dataclass
class LanguageEntity:
    """Represents an entity extracted from natural language."""
    text: str
    entity_type: str
    start_pos: int
    end_pos: int
    confidence: float = 0.0
    
    # Additional properties
    normalized_value: Optional[Any] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'text': self.text,
            'entity_type': self.entity_type,
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'confidence': self.confidence,
            'normalized_value': self.normalized_value,
            'context': self.context
        }


@dataclass
class LanguageIntent:
    """Represents an intent extracted from natural language."""
    intent_type: IntentType
    text: str
    confidence: float
    
    # Supporting entities
    entities: List[LanguageEntity] = field(default_factory=list)
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    
    def add_entity(self, entity: LanguageEntity):
        """Add an entity to this intent."""
        self.entities.append(entity)
    
    def get_entities_by_type(self, entity_type: str) -> List[LanguageEntity]:
        """Get entities of a specific type."""
        return [e for e in self.entities if e.entity_type == entity_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'intent_type': self.intent_type.value,
            'text': self.text,
            'confidence': self.confidence,
            'entities': [e.to_dict() for e in self.entities],
            'context': self.context
        }


@dataclass
class ProcessedText:
    """Represents processed natural language text."""
    original_text: str
    normalized_text: str
    
    # Detected patterns
    patterns: List[LanguagePattern] = field(default_factory=list)
    
    # Extracted intents
    intents: List[LanguageIntent] = field(default_factory=list)
    
    # Extracted entities
    entities: List[LanguageEntity] = field(default_factory=list)
    
    # Text structure
    sentences: List[str] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)
    
    # Metadata
    language: str = "en"
    confidence: float = 0.0
    processing_time: float = 0.0
    
    def add_pattern(self, pattern: LanguagePattern):
        """Add a detected pattern."""
        if pattern not in self.patterns:
            self.patterns.append(pattern)
    
    def add_intent(self, intent: LanguageIntent):
        """Add an extracted intent."""
        self.intents.append(intent)
    
    def add_entity(self, entity: LanguageEntity):
        """Add an extracted entity."""
        self.entities.append(entity)
    
    def get_intents_by_type(self, intent_type: IntentType) -> List[LanguageIntent]:
        """Get intents of a specific type."""
        return [i for i in self.intents if i.intent_type == intent_type]
    
    def get_entities_by_type(self, entity_type: str) -> List[LanguageEntity]:
        """Get entities of a specific type."""
        return [e for e in self.entities if e.entity_type == entity_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'original_text': self.original_text,
            'normalized_text': self.normalized_text,
            'patterns': [p.value for p in self.patterns],
            'intents': [i.to_dict() for i in self.intents],
            'entities': [e.to_dict() for e in self.entities],
            'sentences': self.sentences,
            'tokens': self.tokens,
            'language': self.language,
            'confidence': self.confidence,
            'processing_time': self.processing_time
        }


class NaturalLanguageProcessor:
    """
    Processes natural language text to extract conditional logic patterns,
    understand context, and prepare text for condition parsing.
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize the NLP processor."""
        self.logger = logging.getLogger(__name__)
        
        # Load spaCy model
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            self.logger.warning(f"Model {model_name} not found, using blank model")
            self.nlp = spacy.blank("en")
        
        # Pattern definitions
        self.conditional_patterns = self._initialize_conditional_patterns()
        self.entity_patterns = self._initialize_entity_patterns()
        self.intent_patterns = self._initialize_intent_patterns()
        
        # Normalization rules
        self.normalization_rules = self._initialize_normalization_rules()
        
        # Configuration
        self.config = {
            'min_confidence': 0.5,
            'max_sentence_length': 1000,
            'enable_entity_linking': True,
            'enable_coreference': False,
            'case_sensitive': False
        }
        
        # Statistics
        self.stats = {
            'texts_processed': 0,
            'patterns_detected': 0,
            'entities_extracted': 0,
            'intents_identified': 0
        }
    
    def _initialize_conditional_patterns(self) -> Dict[LanguagePattern, List[str]]:
        """Initialize conditional language patterns."""
        return {
            LanguagePattern.IF_THEN: [
                r'\bif\s+(.+?)\s+then\s+(.+)',
                r'\bwhen\s+(.+?)\s+then\s+(.+)',
                r'\bshould\s+(.+?)\s+then\s+(.+)',
                r'\bin\s+case\s+(.+?)\s+then\s+(.+)'
            ],
            LanguagePattern.IF_THEN_ELSE: [
                r'\bif\s+(.+?)\s+then\s+(.+?)\s+else\s+(.+)',
                r'\bif\s+(.+?)\s+then\s+(.+?)\s+otherwise\s+(.+)',
                r'\bwhen\s+(.+?)\s+then\s+(.+?)\s+else\s+(.+)',
                r'\bshould\s+(.+?)\s+then\s+(.+?)\s+otherwise\s+(.+)'
            ],
            LanguagePattern.UNLESS: [
                r'\bunless\s+(.+?)\s+then\s+(.+)',
                r'\bunless\s+(.+?)\s+do\s+(.+)',
                r'\bif\s+not\s+(.+?)\s+then\s+(.+)'
            ],
            LanguagePattern.WHILE: [
                r'\bwhile\s+(.+?)\s+do\s+(.+)',
                r'\bwhile\s+(.+?)\s+then\s+(.+)',
                r'\bas\s+long\s+as\s+(.+?)\s+do\s+(.+)'
            ],
            LanguagePattern.UNTIL: [
                r'\buntil\s+(.+?)\s+do\s+(.+)',
                r'\buntil\s+(.+?)\s+then\s+(.+)',
                r'\bkeep\s+(.+?)\s+until\s+(.+)'
            ],
            LanguagePattern.SWITCH_CASE: [
                r'\bswitch\s+(.+?)\s+case\s+(.+)',
                r'\bdepending\s+on\s+(.+?)\s+if\s+(.+)',
                r'\bbased\s+on\s+(.+?)\s+when\s+(.+)'
            ],
            LanguagePattern.EITHER_OR: [
                r'\beither\s+(.+?)\s+or\s+(.+)',
                r'\b(.+?)\s+or\s+(.+?)\s+but\s+not\s+both'
            ],
            LanguagePattern.NEITHER_NOR: [
                r'\bneither\s+(.+?)\s+nor\s+(.+)',
                r'\bnot\s+(.+?)\s+and\s+not\s+(.+)'
            ],
            LanguagePattern.BOTH_AND: [
                r'\bboth\s+(.+?)\s+and\s+(.+)',
                r'\b(.+?)\s+and\s+(.+?)\s+both'
            ]
        }
    
    def _initialize_entity_patterns(self) -> Dict[str, List[str]]:
        """Initialize entity extraction patterns."""
        return {
            'variable': [
                r'\b[a-zA-Z_][a-zA-Z0-9_]*\b',
                r'\$[a-zA-Z_][a-zA-Z0-9_]*',
                r'\{[a-zA-Z_][a-zA-Z0-9_]*\}'
            ],
            'number': [
                r'\b\d+\.?\d*\b',
                r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b'
            ],
            'string': [
                r'"([^"]*)"',
                r"'([^']*)'"
            ],
            'boolean': [
                r'\b(?:true|false|yes|no|on|off)\b'
            ],
            'comparison': [
                r'\b(?:equals?|is|are|greater\s+than|less\s+than|contains?|includes?|matches?)\b'
            ],
            'logical': [
                r'\b(?:and|or|not|but|however|although)\b'
            ],
            'temporal': [
                r'\b(?:before|after|during|while|when|until|since)\b',
                r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?\b',
                r'\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'
            ],
            'quantifier': [
                r'\b(?:all|any|some|none|every|each|many|few|several)\b',
                r'\b(?:more|less|fewer|greater|smaller)\s+than\b'
            ]
        }
    
    def _initialize_intent_patterns(self) -> Dict[IntentType, List[str]]:
        """Initialize intent classification patterns."""
        return {
            IntentType.CONDITION: [
                r'\b(?:if|when|unless|while|until|should|in\s+case)\b',
                r'\b(?:check|verify|test|validate)\b'
            ],
            IntentType.ACTION: [
                r'\b(?:do|execute|run|perform|start|stop|create|delete|update)\b',
                r'\b(?:click|type|select|choose|open|close|save)\b'
            ],
            IntentType.COMPARISON: [
                r'\b(?:equals?|is|are|greater|less|contains?|matches?)\b',
                r'\b(?:same|different|similar|identical)\b'
            ],
            IntentType.LOGICAL_OPERATION: [
                r'\b(?:and|or|not|but|however|although)\b'
            ],
            IntentType.TEMPORAL: [
                r'\b(?:before|after|during|while|when|until|since)\b',
                r'\b(?:time|date|hour|minute|second|day|week|month|year)\b'
            ],
            IntentType.SPATIAL: [
                r'\b(?:above|below|left|right|inside|outside|near|far)\b',
                r'\b(?:position|location|coordinate|place)\b'
            ],
            IntentType.QUANTITATIVE: [
                r'\b(?:count|number|amount|quantity|size|length|width|height)\b',
                r'\b\d+(?:\.\d+)?\s*(?:%|percent|degrees?)\b'
            ],
            IntentType.QUALITATIVE: [
                r'\b(?:color|style|type|kind|category|class)\b',
                r'\b(?:good|bad|better|worse|best|worst)\b'
            ]
        }
    
    def _initialize_normalization_rules(self) -> List[Tuple[str, str]]:
        """Initialize text normalization rules."""
        return [
            # Contractions
            (r"won't", "will not"),
            (r"can't", "cannot"),
            (r"n't", " not"),
            (r"'re", " are"),
            (r"'ve", " have"),
            (r"'ll", " will"),
            (r"'d", " would"),
            (r"'m", " am"),
            
            # Common abbreviations
            (r"\bw/", "with"),
            (r"\bw/o", "without"),
            (r"\betc\.", "etcetera"),
            (r"\be\.g\.", "for example"),
            (r"\bi\.e\.", "that is"),
            
            # Normalize whitespace
            (r"\s+", " "),
            (r"^\s+|\s+$", "")
        ]
    
    def process_text(self, text: str) -> ProcessedText:
        """
        Process natural language text to extract conditional logic patterns.
        
        Args:
            text: Input text to process
            
        Returns:
            ProcessedText: Processed text with extracted patterns and entities
        """
        import time
        start_time = time.time()
        
        try:
            # Normalize text
            normalized_text = self._normalize_text(text)
            
            # Create processed text object
            processed = ProcessedText(
                original_text=text,
                normalized_text=normalized_text
            )
            
            # Process with spaCy
            doc = self.nlp(normalized_text)
            
            # Extract basic structure
            processed.sentences = [sent.text.strip() for sent in doc.sents]
            processed.tokens = [token.text for token in doc if not token.is_space]
            
            # Detect language patterns
            self._detect_patterns(processed)
            
            # Extract entities
            self._extract_entities(processed, doc)
            
            # Identify intents
            self._identify_intents(processed)
            
            # Calculate overall confidence
            processed.confidence = self._calculate_confidence(processed)
            
            # Record processing time
            processed.processing_time = time.time() - start_time
            
            # Update statistics
            self.stats['texts_processed'] += 1
            self.stats['patterns_detected'] += len(processed.patterns)
            self.stats['entities_extracted'] += len(processed.entities)
            self.stats['intents_identified'] += len(processed.intents)
            
            self.logger.debug(f"Processed text with {len(processed.patterns)} patterns, "
                            f"{len(processed.entities)} entities, {len(processed.intents)} intents")
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Error processing text: {e}")
            # Return minimal processed text
            return ProcessedText(
                original_text=text,
                normalized_text=text,
                processing_time=time.time() - start_time
            )
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text using predefined rules."""
        normalized = text
        
        # Apply normalization rules
        for pattern, replacement in self.normalization_rules:
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        # Convert to lowercase if not case sensitive
        if not self.config['case_sensitive']:
            normalized = normalized.lower()
        
        return normalized.strip()
    
    def _detect_patterns(self, processed: ProcessedText):
        """Detect conditional language patterns in text."""
        text = processed.normalized_text
        
        for pattern_type, patterns in self.conditional_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
                
                for match in matches:
                    processed.add_pattern(pattern_type)
                    self.logger.debug(f"Detected pattern {pattern_type.value}: {match.group()}")
    
    def _extract_entities(self, processed: ProcessedText, doc):
        """Extract entities from text."""
        # Extract using spaCy NER
        for ent in doc.ents:
            entity = LanguageEntity(
                text=ent.text,
                entity_type=ent.label_,
                start_pos=ent.start_char,
                end_pos=ent.end_char,
                confidence=0.8  # Default confidence for spaCy entities
            )
            processed.add_entity(entity)
        
        # Extract using custom patterns
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, processed.normalized_text, re.IGNORECASE)
                
                for match in matches:
                    entity = LanguageEntity(
                        text=match.group(),
                        entity_type=entity_type,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.6  # Lower confidence for pattern-based extraction
                    )
                    
                    # Normalize entity value
                    entity.normalized_value = self._normalize_entity_value(entity)
                    
                    processed.add_entity(entity)
    
    def _identify_intents(self, processed: ProcessedText):
        """Identify intents in the text."""
        text = processed.normalized_text
        
        for intent_type, patterns in self.intent_patterns.items():
            confidence_scores = []
            
            for pattern in patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                if matches:
                    confidence_scores.append(len(matches) * 0.2)  # Base confidence per match
            
            if confidence_scores:
                total_confidence = min(sum(confidence_scores), 1.0)
                
                if total_confidence >= self.config['min_confidence']:
                    intent = LanguageIntent(
                        intent_type=intent_type,
                        text=text,
                        confidence=total_confidence
                    )
                    
                    # Add relevant entities to intent
                    relevant_entities = self._get_relevant_entities(processed.entities, intent_type)
                    for entity in relevant_entities:
                        intent.add_entity(entity)
                    
                    processed.add_intent(intent)
    
    def _normalize_entity_value(self, entity: LanguageEntity) -> Any:
        """Normalize entity value based on its type."""
        text = entity.text.strip()
        
        if entity.entity_type == 'number':
            try:
                # Remove commas and convert to number
                clean_text = text.replace(',', '')
                if '.' in clean_text:
                    return float(clean_text)
                else:
                    return int(clean_text)
            except ValueError:
                return text
        
        elif entity.entity_type == 'boolean':
            lower_text = text.lower()
            if lower_text in ['true', 'yes', 'on']:
                return True
            elif lower_text in ['false', 'no', 'off']:
                return False
            return text
        
        elif entity.entity_type == 'string':
            # Remove quotes
            if (text.startswith('"') and text.endswith('"')) or \
               (text.startswith("'") and text.endswith("'")):
                return text[1:-1]
            return text
        
        else:
            return text
    
    def _get_relevant_entities(self, entities: List[LanguageEntity], 
                             intent_type: IntentType) -> List[LanguageEntity]:
        """Get entities relevant to a specific intent type."""
        relevant_types = {
            IntentType.CONDITION: ['variable', 'comparison', 'boolean'],
            IntentType.ACTION: ['variable', 'string'],
            IntentType.COMPARISON: ['variable', 'number', 'string', 'comparison'],
            IntentType.LOGICAL_OPERATION: ['logical'],
            IntentType.TEMPORAL: ['temporal', 'number'],
            IntentType.SPATIAL: ['variable', 'number'],
            IntentType.QUANTITATIVE: ['number', 'quantifier'],
            IntentType.QUALITATIVE: ['string', 'variable']
        }
        
        target_types = relevant_types.get(intent_type, [])
        return [e for e in entities if e.entity_type in target_types]
    
    def _calculate_confidence(self, processed: ProcessedText) -> float:
        """Calculate overall confidence score for processed text."""
        scores = []
        
        # Pattern detection confidence
        if processed.patterns:
            scores.append(min(len(processed.patterns) * 0.3, 1.0))
        
        # Entity extraction confidence
        if processed.entities:
            entity_confidences = [e.confidence for e in processed.entities if e.confidence > 0]
            if entity_confidences:
                scores.append(sum(entity_confidences) / len(entity_confidences))
        
        # Intent identification confidence
        if processed.intents:
            intent_confidences = [i.confidence for i in processed.intents]
            scores.append(sum(intent_confidences) / len(intent_confidences))
        
        # Text structure confidence
        if processed.sentences and processed.tokens:
            structure_score = min(len(processed.sentences) * 0.1 + len(processed.tokens) * 0.01, 1.0)
            scores.append(structure_score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def extract_conditional_structure(self, text: str) -> Dict[str, Any]:
        """
        Extract conditional structure from text for easier parsing.
        
        Args:
            text: Input text
            
        Returns:
            Dict containing structured conditional information
        """
        processed = self.process_text(text)
        
        structure = {
            'original_text': text,
            'normalized_text': processed.normalized_text,
            'detected_patterns': [p.value for p in processed.patterns],
            'conditions': [],
            'actions': [],
            'logical_operators': [],
            'entities': {}
        }
        
        # Group entities by type
        for entity in processed.entities:
            entity_type = entity.entity_type
            if entity_type not in structure['entities']:
                structure['entities'][entity_type] = []
            structure['entities'][entity_type].append({
                'text': entity.text,
                'normalized_value': entity.normalized_value,
                'confidence': entity.confidence
            })
        
        # Extract conditions and actions from intents
        for intent in processed.intents:
            if intent.intent_type == IntentType.CONDITION:
                structure['conditions'].append({
                    'text': intent.text,
                    'confidence': intent.confidence,
                    'entities': [e.to_dict() for e in intent.entities]
                })
            elif intent.intent_type == IntentType.ACTION:
                structure['actions'].append({
                    'text': intent.text,
                    'confidence': intent.confidence,
                    'entities': [e.to_dict() for e in intent.entities]
                })
            elif intent.intent_type == IntentType.LOGICAL_OPERATION:
                structure['logical_operators'].append({
                    'text': intent.text,
                    'confidence': intent.confidence
                })
        
        return structure
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            **self.stats,
            'patterns_supported': len(self.conditional_patterns),
            'entity_types_supported': len(self.entity_patterns),
            'intent_types_supported': len(self.intent_patterns),
            'normalization_rules': len(self.normalization_rules)
        }