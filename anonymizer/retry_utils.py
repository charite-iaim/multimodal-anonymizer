"""
Retry utilities for handling API errors, rate limits, and timeouts.

This module provides robust retry mechanisms to ensure all files are processed
even when encountering transient API errors.
"""

import time
import random
import functools
from typing import Callable, TypeVar, Any, Optional, Type, Tuple, Union
import logging

# Common exceptions from various APIs
try:
    from openai import RateLimitError, APITimeoutError, APIConnectionError, APIError
    OPENAI_EXCEPTIONS = (RateLimitError, APITimeoutError, APIConnectionError, APIError)
except ImportError:
    OPENAI_EXCEPTIONS = ()

try:
    from httpx import TimeoutException, ConnectError, HTTPStatusError
    HTTPX_EXCEPTIONS = (TimeoutException, ConnectError, HTTPStatusError)
except ImportError:
    HTTPX_EXCEPTIONS = ()

# Common retryable exceptions
RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    *OPENAI_EXCEPTIONS,
    *HTTPX_EXCEPTIONS,
)

T = TypeVar('T')

# Configure logging
logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_retries: int = 5,
        initial_delay: float = 1.0,
        max_delay: float = 120.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Tuple[Type[Exception], ...] = RETRYABLE_EXCEPTIONS,
    ):
        """
        Initialize retry configuration.
        
        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            exponential_base: Base for exponential backoff calculation
            jitter: Whether to add random jitter to delays
            retryable_exceptions: Tuple of exception types to retry on
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions


# Default configuration for API calls
DEFAULT_RETRY_CONFIG = RetryConfig(
    max_retries=5,
    initial_delay=2.0,
    max_delay=120.0,
    exponential_base=2.0,
    jitter=True,
)


def calculate_delay(
    attempt: int,
    config: RetryConfig,
    error: Optional[Exception] = None
) -> float:
    """
    Calculate delay before next retry using exponential backoff.
    
    Args:
        attempt: Current attempt number (1-based)
        config: Retry configuration
        error: The exception that triggered the retry
        
    Returns:
        Delay in seconds before next retry
    """
    # Check if the error has a retry-after header (rate limit)
    retry_after = None
    if error is not None:
        # OpenAI API often includes retry_after in rate limit errors
        if hasattr(error, 'response') and error.response is not None:
            retry_after = error.response.headers.get('retry-after')
            if retry_after:
                try:
                    retry_after = float(retry_after)
                except ValueError:
                    retry_after = None
        
        # Some errors have retry_after as an attribute
        if retry_after is None and hasattr(error, 'retry_after'):
            retry_after = error.retry_after
    
    if retry_after is not None and retry_after > 0:
        # Use the API-specified retry time, but cap it
        delay = min(retry_after, config.max_delay)
    else:
        # Exponential backoff: delay = initial_delay * base^(attempt-1)
        delay = config.initial_delay * (config.exponential_base ** (attempt - 1))
        delay = min(delay, config.max_delay)
    
    # Add jitter to prevent thundering herd
    if config.jitter:
        jitter_range = delay * 0.25  # 25% jitter
        delay = delay + random.uniform(-jitter_range, jitter_range)
        delay = max(config.initial_delay, delay)  # Ensure minimum delay
    
    return delay


def is_retryable_error(error: Exception, config: RetryConfig) -> bool:
    """
    Check if an error is retryable.
    
    Args:
        error: The exception to check
        config: Retry configuration
        
    Returns:
        True if the error should trigger a retry
    """
    # Check against configured retryable exceptions
    if isinstance(error, config.retryable_exceptions):
        return True
    
    # Check error message for common retryable patterns
    error_msg = str(error).lower()
    retryable_patterns = [
        'rate limit',
        'rate_limit',
        'ratelimit',
        'too many requests',
        '429',
        'timeout',
        'timed out',
        'connection',
        'temporarily unavailable',
        'service unavailable',
        '503',
        '502',
        'bad gateway',
        'server error',
        '500',
        'overloaded',
        'capacity',
    ]
    
    for pattern in retryable_patterns:
        if pattern in error_msg:
            return True
    
    return False


def retry_with_backoff(
    func: Callable[..., T],
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> T:
    """
    Execute a function with retry logic and exponential backoff.
    
    Args:
        func: Function to execute (should be a no-argument callable, use functools.partial)
        config: Retry configuration (uses default if not provided)
        on_retry: Optional callback called on each retry with (attempt, error, delay)
        
    Returns:
        The result of the function
        
    Raises:
        The last exception if all retries are exhausted
    """
    if config is None:
        config = DEFAULT_RETRY_CONFIG
    
    last_error = None
    
    for attempt in range(1, config.max_retries + 2):  # +2 because first attempt is not a retry
        try:
            return func()
        except Exception as e:
            last_error = e
            
            # Check if we've exhausted retries
            if attempt > config.max_retries:
                logger.warning(f"All {config.max_retries} retries exhausted. Last error: {e}")
                raise
            
            # Check if error is retryable
            if not is_retryable_error(e, config):
                logger.warning(f"Non-retryable error encountered: {e}")
                raise
            
            # Calculate delay
            delay = calculate_delay(attempt, config, e)
            
            # Log the retry
            error_type = type(e).__name__
            logger.info(f"Attempt {attempt}/{config.max_retries + 1} failed with {error_type}: {e}")
            logger.info(f"Retrying in {delay:.1f} seconds...")
            
            # Call the on_retry callback if provided
            if on_retry:
                on_retry(attempt, e, delay)
            
            # Wait before retry
            time.sleep(delay)
    
    # Should never reach here, but just in case
    raise last_error


def with_retry(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
):
    """
    Decorator to add retry logic to a function.
    
    Args:
        config: Retry configuration (uses default if not provided)
        on_retry: Optional callback called on each retry
        
    Returns:
        Decorated function with retry logic
        
    Example:
        @with_retry(config=RetryConfig(max_retries=3))
        def call_api():
            return api.invoke(prompt)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return retry_with_backoff(
                lambda: func(*args, **kwargs),
                config=config,
                on_retry=on_retry,
            )
        return wrapper
    return decorator


class RetryableAPIError(Exception):
    """Custom exception for retryable API errors."""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


def create_retry_callback(
    prefix: str = "",
    print_func: Callable[[str], None] = print,
) -> Callable[[int, Exception, float], None]:
    """
    Create a retry callback that prints retry information.
    
    Args:
        prefix: Prefix for log messages
        print_func: Function to use for printing (default: print)
        
    Returns:
        Callback function for use with retry_with_backoff
    """
    def callback(attempt: int, error: Exception, delay: float) -> None:
        error_type = type(error).__name__
        msg = f"{prefix}Retry {attempt}: {error_type} - waiting {delay:.1f}s"
        print_func(msg)
    
    return callback
