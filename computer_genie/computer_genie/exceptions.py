"""Custom exceptions for Computer Genie"""

class GenieException(Exception):
    """Base exception for Computer Genie"""
    pass

class ModelNotFoundError(GenieException):
    """Raised when a model is not found"""
    pass

class ElementNotFoundError(GenieException):
    """Raised when an element cannot be found"""
    pass

class ActionFailedError(GenieException):
    """Raised when an action fails to execute"""
    pass

class ConfigurationError(GenieException):
    """Raised when there's a configuration issue"""
    pass

class AuthenticationError(GenieException):
    """Raised when authentication fails"""
    pass

class TimeoutError(GenieException):
    """Raised when an operation times out"""
    pass

class PlatformError(GenieException):
    """Raised when there's a platform-specific error"""
    pass

__all__ = [
    'GenieException',
    'ModelNotFoundError', 
    'ElementNotFoundError',
    'ActionFailedError',
    'ConfigurationError',
    'AuthenticationError',
    'TimeoutError',
    'PlatformError'
]