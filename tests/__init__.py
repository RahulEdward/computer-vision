"""
Test Suite for Intelligent Automation Engine

This package contains comprehensive tests for all components of the
intelligent automation engine, including unit tests, integration tests,
and performance tests.
"""

# Import test modules for easy access
from . import test_parallel_execution
from . import test_scheduling
from . import test_integration

# Test suite metadata
__version__ = "1.0.0"
__author__ = "Intelligent Automation Engine Team"

# Available test modules
__all__ = [
    'test_parallel_execution',
    'test_scheduling', 
    'test_integration'
]