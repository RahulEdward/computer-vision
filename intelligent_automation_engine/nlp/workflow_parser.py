"""
Natural Language Workflow Parser

Converts plain English descriptions into executable automation workflows
with high accuracy understanding of user intent and requirements.
"""

import re
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .conditional_logic import ConditionalLogicEngine
from .intent_classifier import IntentClassifier


class ActionType(Enum):
    """Types of automation actions."""
    CLICK = "click"
    TYPE = "type"
    WAIT = "wait"
    NAVIGATE = "navigate"
    SCROLL = "scroll"
    DRAG_DROP = "drag_drop"
    CONDITION = "condition"
    LOOP = "loop"
    PARALLEL = "parallel"
    CUSTOM = "custom"


@dataclass
class WorkflowStep:
    """Represents a single step in an automation workflow."""
    action_type: ActionType
    target: str
    parameters: Dict[str, Any]
    conditions: Optional[List[str]] = None
    description: str = ""
    confidence: float = 1.0


@dataclass
class ParsedWorkflow:
    """Represents a complete parsed workflow."""
    name: str
    description: str
    steps: List[WorkflowStep]
    variables: Dict[str, Any]
    metadata: Dict[str, Any]


class WorkflowParser:
    """
    Advanced natural language parser that converts English descriptions
    into structured automation workflows.
    """
    
    def __init__(self):
        """Initialize the workflow parser."""
        self.logger = logging.getLogger(__name__)
        self.conditional_engine = ConditionalLogicEngine()
        self.intent_classifier = IntentClassifier()
        
        # Action patterns for natural language understanding
        self.action_patterns = {
            ActionType.CLICK: [
                r"click (?:on )?(.+)",
                r"press (?:the )?(.+) button",
                r"select (.+)",
                r"tap (?:on )?(.+)"
            ],
            ActionType.TYPE: [
                r"type (.+) (?:in|into) (.+)",
                r"enter (.+) (?:in|into) (.+)",
                r"input (.+) (?:in|into) (.+)",
                r"write (.+) (?:in|into) (.+)"
            ],
            ActionType.WAIT: [
                r"wait (?:for )?(\d+) seconds?",
                r"pause (?:for )?(\d+) seconds?",
                r"wait (?:until|for) (.+) (?:appears|loads|is visible)"
            ],
            ActionType.NAVIGATE: [
                r"go to (.+)",
                r"navigate to (.+)",
                r"open (.+)",
                r"visit (.+)"
            ],
            ActionType.SCROLL: [
                r"scroll (up|down|left|right)",
                r"scroll to (.+)",
                r"scroll (\d+) pixels (up|down|left|right)"
            ],
            ActionType.DRAG_DROP: [
                r"drag (.+) to (.+)",
                r"move (.+) to (.+)",
                r"drop (.+) (?:on|onto) (.+)"
            ]
        }
        
        # Conditional patterns
        self.conditional_patterns = [
            r"if (.+) then (.+)",
            r"when (.+) do (.+)",
            r"if (.+) then (.+) else (.+)",
            r"if (.+) then (.+) otherwise (.+)"
        ]
        
        # Loop patterns
        self.loop_patterns = [
            r"repeat (.+) (\d+) times",
            r"for each (.+) in (.+) do (.+)",
            r"while (.+) do (.+)",
            r"until (.+) do (.+)"
        ]
        
        # Parallel execution patterns
        self.parallel_patterns = [
            r"simultaneously (.+) and (.+)",
            r"at the same time (.+) and (.+)",
            r"in parallel (.+) and (.+)"
        ]
    
    async def parse_description(self, description: str) -> Dict[str, Any]:
        """
        Parse a natural language description into a structured workflow.
        
        Args:
            description: Plain English description of the workflow
            
        Returns:
            Dict: Structured workflow definition
        """
        self.logger.info(f"Parsing workflow description: {description}")
        
        try:
            # Clean and preprocess the description
            cleaned_description = self._preprocess_text(description)
            
            # Extract workflow metadata
            workflow_name = self._extract_workflow_name(cleaned_description)
            
            # Split into sentences/steps
            sentences = self._split_into_sentences(cleaned_description)
            
            # Parse each sentence into workflow steps
            steps = []
            variables = {}
            
            for i, sentence in enumerate(sentences):
                parsed_steps = await self._parse_sentence(sentence, i)
                steps.extend(parsed_steps)
                
                # Extract variables from the sentence
                sentence_vars = self._extract_variables(sentence)
                variables.update(sentence_vars)
            
            # Optimize and validate the workflow
            optimized_steps = await self._optimize_workflow_steps(steps)
            
            # Create the parsed workflow
            workflow = {
                'name': workflow_name,
                'description': description,
                'steps': [self._step_to_dict(step) for step in optimized_steps],
                'variables': variables,
                'metadata': {
                    'parsed_at': asyncio.get_event_loop().time(),
                    'sentence_count': len(sentences),
                    'step_count': len(optimized_steps),
                    'confidence': self._calculate_overall_confidence(optimized_steps)
                }
            }
            
            self.logger.info(f"Successfully parsed workflow with {len(optimized_steps)} steps")
            return workflow
            
        except Exception as e:
            self.logger.error(f"Failed to parse workflow description: {e}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess the input text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize punctuation
        text = re.sub(r'[.!?]+', '.', text)
        
        # Convert to lowercase for pattern matching
        return text.lower()
    
    def _extract_workflow_name(self, description: str) -> str:
        """Extract a meaningful name for the workflow."""
        # Try to find explicit naming patterns
        name_patterns = [
            r"(?:create|build|make) (?:a |an )?(.+?) (?:workflow|automation|process)",
            r"automate (.+?) (?:process|task|workflow)",
            r"(?:this|the) (.+?) (?:automation|workflow|process)"
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, description)
            if match:
                return match.group(1).title()
        
        # Fallback: use first few words
        words = description.split()[:3]
        return ' '.join(word.title() for word in words) + " Automation"
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into individual sentences/steps."""
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        
        # Also split on common step separators
        all_sentences = []
        for sentence in sentences:
            # Split on "then", "next", "after that", etc.
            sub_sentences = re.split(r'\b(?:then|next|after that|afterwards)\b', sentence)
            all_sentences.extend(sub_sentences)
        
        # Clean and filter empty sentences
        return [s.strip() for s in all_sentences if s.strip()]
    
    async def _parse_sentence(self, sentence: str, step_index: int) -> List[WorkflowStep]:
        """Parse a single sentence into one or more workflow steps."""
        steps = []
        
        # Check for conditional logic
        if await self._is_conditional(sentence):
            conditional_steps = await self._parse_conditional(sentence, step_index)
            steps.extend(conditional_steps)
            return steps
        
        # Check for loops
        if self._is_loop(sentence):
            loop_steps = self._parse_loop(sentence, step_index)
            steps.extend(loop_steps)
            return steps
        
        # Check for parallel execution
        if self._is_parallel(sentence):
            parallel_steps = self._parse_parallel(sentence, step_index)
            steps.extend(parallel_steps)
            return steps
        
        # Parse as regular action
        action_step = self._parse_action(sentence, step_index)
        if action_step:
            steps.append(action_step)
        
        return steps
    
    def _parse_action(self, sentence: str, step_index: int) -> Optional[WorkflowStep]:
        """Parse a sentence as a regular action step."""
        # Try to match against known action patterns
        for action_type, patterns in self.action_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, sentence)
                if match:
                    return self._create_action_step(
                        action_type, match, sentence, step_index
                    )
        
        # If no pattern matches, create a custom action
        return WorkflowStep(
            action_type=ActionType.CUSTOM,
            target="unknown",
            parameters={'raw_text': sentence},
            description=sentence,
            confidence=0.5
        )
    
    def _create_action_step(self, action_type: ActionType, match: re.Match, 
                          sentence: str, step_index: int) -> WorkflowStep:
        """Create a workflow step from a matched pattern."""
        groups = match.groups()
        
        if action_type == ActionType.CLICK:
            return WorkflowStep(
                action_type=action_type,
                target=groups[0],
                parameters={'element': groups[0]},
                description=sentence,
                confidence=0.9
            )
        
        elif action_type == ActionType.TYPE:
            return WorkflowStep(
                action_type=action_type,
                target=groups[1] if len(groups) > 1 else groups[0],
                parameters={
                    'text': groups[0],
                    'field': groups[1] if len(groups) > 1 else 'input'
                },
                description=sentence,
                confidence=0.9
            )
        
        elif action_type == ActionType.WAIT:
            if groups[0].isdigit():
                return WorkflowStep(
                    action_type=action_type,
                    target="time",
                    parameters={'duration': int(groups[0])},
                    description=sentence,
                    confidence=0.95
                )
            else:
                return WorkflowStep(
                    action_type=action_type,
                    target="element",
                    parameters={'element': groups[0]},
                    description=sentence,
                    confidence=0.85
                )
        
        elif action_type == ActionType.NAVIGATE:
            return WorkflowStep(
                action_type=action_type,
                target=groups[0],
                parameters={'url': groups[0]},
                description=sentence,
                confidence=0.9
            )
        
        elif action_type == ActionType.SCROLL:
            if len(groups) == 1:
                return WorkflowStep(
                    action_type=action_type,
                    target="page",
                    parameters={'direction': groups[0]},
                    description=sentence,
                    confidence=0.9
                )
            elif len(groups) == 2:
                return WorkflowStep(
                    action_type=action_type,
                    target="page",
                    parameters={
                        'pixels': int(groups[0]),
                        'direction': groups[1]
                    },
                    description=sentence,
                    confidence=0.9
                )
            else:
                return WorkflowStep(
                    action_type=action_type,
                    target=groups[0],
                    parameters={'element': groups[0]},
                    description=sentence,
                    confidence=0.85
                )
        
        elif action_type == ActionType.DRAG_DROP:
            return WorkflowStep(
                action_type=action_type,
                target=groups[0],
                parameters={
                    'source': groups[0],
                    'destination': groups[1]
                },
                description=sentence,
                confidence=0.85
            )
        
        # Default case
        return WorkflowStep(
            action_type=action_type,
            target=groups[0] if groups else "unknown",
            parameters={'raw_groups': groups},
            description=sentence,
            confidence=0.7
        )
    
    async def _is_conditional(self, sentence: str) -> bool:
        """Check if a sentence contains conditional logic."""
        return await self.conditional_engine.is_conditional(sentence)
    
    async def _parse_conditional(self, sentence: str, step_index: int) -> List[WorkflowStep]:
        """Parse conditional logic from a sentence."""
        return await self.conditional_engine.parse_conditional(sentence, step_index)
    
    def _is_loop(self, sentence: str) -> bool:
        """Check if a sentence contains loop logic."""
        return any(re.search(pattern, sentence) for pattern in self.loop_patterns)
    
    def _parse_loop(self, sentence: str, step_index: int) -> List[WorkflowStep]:
        """Parse loop logic from a sentence."""
        for pattern in self.loop_patterns:
            match = re.search(pattern, sentence)
            if match:
                groups = match.groups()
                
                if "repeat" in pattern:
                    return [WorkflowStep(
                        action_type=ActionType.LOOP,
                        target="repeat",
                        parameters={
                            'action': groups[0],
                            'count': int(groups[1]),
                            'type': 'repeat'
                        },
                        description=sentence,
                        confidence=0.9
                    )]
                
                elif "for each" in pattern:
                    return [WorkflowStep(
                        action_type=ActionType.LOOP,
                        target="foreach",
                        parameters={
                            'item': groups[0],
                            'collection': groups[1],
                            'action': groups[2],
                            'type': 'foreach'
                        },
                        description=sentence,
                        confidence=0.85
                    )]
                
                elif "while" in pattern:
                    return [WorkflowStep(
                        action_type=ActionType.LOOP,
                        target="while",
                        parameters={
                            'condition': groups[0],
                            'action': groups[1],
                            'type': 'while'
                        },
                        description=sentence,
                        confidence=0.8
                    )]
        
        return []
    
    def _is_parallel(self, sentence: str) -> bool:
        """Check if a sentence contains parallel execution logic."""
        return any(re.search(pattern, sentence) for pattern in self.parallel_patterns)
    
    def _parse_parallel(self, sentence: str, step_index: int) -> List[WorkflowStep]:
        """Parse parallel execution logic from a sentence."""
        for pattern in self.parallel_patterns:
            match = re.search(pattern, sentence)
            if match:
                groups = match.groups()
                return [WorkflowStep(
                    action_type=ActionType.PARALLEL,
                    target="parallel_group",
                    parameters={
                        'actions': list(groups),
                        'type': 'parallel'
                    },
                    description=sentence,
                    confidence=0.8
                )]
        
        return []
    
    def _extract_variables(self, sentence: str) -> Dict[str, Any]:
        """Extract variables and their values from a sentence."""
        variables = {}
        
        # Look for variable patterns like "set X to Y" or "X = Y"
        var_patterns = [
            r"set (\w+) to (.+)",
            r"(\w+) (?:is|equals?) (.+)",
            r"(\w+) = (.+)"
        ]
        
        for pattern in var_patterns:
            matches = re.finditer(pattern, sentence)
            for match in matches:
                var_name = match.group(1)
                var_value = match.group(2).strip()
                
                # Try to convert to appropriate type
                if var_value.isdigit():
                    variables[var_name] = int(var_value)
                elif var_value.replace('.', '').isdigit():
                    variables[var_name] = float(var_value)
                elif var_value.lower() in ['true', 'false']:
                    variables[var_name] = var_value.lower() == 'true'
                else:
                    variables[var_name] = var_value
        
        return variables
    
    async def _optimize_workflow_steps(self, steps: List[WorkflowStep]) -> List[WorkflowStep]:
        """Optimize the workflow steps for better performance and accuracy."""
        optimized_steps = []
        
        for i, step in enumerate(steps):
            # Merge consecutive similar actions
            if (i > 0 and 
                step.action_type == steps[i-1].action_type and
                step.action_type in [ActionType.TYPE, ActionType.CLICK]):
                
                # Merge with previous step
                prev_step = optimized_steps[-1]
                if step.action_type == ActionType.TYPE:
                    prev_step.parameters['text'] += ' ' + step.parameters.get('text', '')
                    prev_step.description += ' and ' + step.description
                    continue
            
            # Add wait steps after navigation
            if step.action_type == ActionType.NAVIGATE:
                optimized_steps.append(step)
                # Add implicit wait
                wait_step = WorkflowStep(
                    action_type=ActionType.WAIT,
                    target="page_load",
                    parameters={'duration': 2},
                    description="Wait for page to load",
                    confidence=0.9
                )
                optimized_steps.append(wait_step)
                continue
            
            optimized_steps.append(step)
        
        return optimized_steps
    
    def _calculate_overall_confidence(self, steps: List[WorkflowStep]) -> float:
        """Calculate overall confidence score for the workflow."""
        if not steps:
            return 0.0
        
        total_confidence = sum(step.confidence for step in steps)
        return total_confidence / len(steps)
    
    def _step_to_dict(self, step: WorkflowStep) -> Dict[str, Any]:
        """Convert a WorkflowStep to a dictionary."""
        return {
            'action_type': step.action_type.value,
            'target': step.target,
            'parameters': step.parameters,
            'conditions': step.conditions,
            'description': step.description,
            'confidence': step.confidence
        }