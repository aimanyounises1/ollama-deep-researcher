# src/assistant/utils/ssl_fix.py

import logging
import functools
import ssl
import asyncio
import os
from functools import wraps
from typing import Callable, TypeVar, Any, Optional, cast

logger = logging.getLogger(__name__)

# Type variables for better type hinting
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

def initialize_ssl_fixes() -> None:
    """Initialize global SSL fixes for the application."""
    # Set up environment variables if needed
    if os.environ.get('SKIP_SSL_VERIFY', 'false').lower() == 'true':
        logger.warning("SSL verification disabled globally via SKIP_SSL_VERIFY=true")
        # Not recommended for production, but useful for development/testing
        _disable_ssl_verification()
    
    # Apply HuggingFace-specific SSL fixes
    patch_ssl_for_huggingface()

def _disable_ssl_verification() -> None:
    """
    Disable SSL verification globally (USE WITH CAUTION!)
    Only for testing/development environments.
    """
    try:
        # Create unverified SSL context
        ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore
        logger.warning("SSL verification disabled globally. NOT RECOMMENDED FOR PRODUCTION!")
    except Exception as e:
        logger.error(f"Failed to disable SSL verification: {e}")

def patch_ssl_for_huggingface() -> None:
    """
    Apply patches specifically needed for HuggingFace integrations.
    """
    try:
        # Set environment variables used by HF libraries
        os.environ['CURL_CA_BUNDLE'] = ''
        os.environ['REQUESTS_CA_BUNDLE'] = ''
        
        # For transformers library
        os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = 'true'
        os.environ['TRANSFORMERS_OFFLINE'] = 'true'  # May be needed in some environments
        
        logger.debug("Applied HuggingFace-specific SSL patches")
    except Exception as e:
        logger.error(f"Failed to apply HuggingFace SSL patches: {e}")

def with_ssl_disabled(func: Optional[F] = None) -> F:
    """
    Decorator to temporarily disable SSL verification for a specific function call.
    Much safer than disabling globally, as it restores the original context after.
    
    Usage:
        @with_ssl_disabled
        def my_function():
            # Function code that needs SSL verification disabled
            
        @with_ssl_disabled
        async def my_async_function():
            # Async function code
    """
    if func is None:
        return cast(F, lambda f: with_ssl_disabled(f))
    
    @wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        # Save original context
        original_context = ssl._create_default_https_context
        try:
            # Temporarily disable verification
            ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore
            # Call the async function
            return await func(*args, **kwargs)
        finally:
            # Restore original context
            ssl._create_default_https_context = original_context
    
    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        # Save original context
        original_context = ssl._create_default_https_context
        try:
            # Temporarily disable verification
            ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore
            # Call the function
            return func(*args, **kwargs)
        finally:
            # Restore original context
            ssl._create_default_https_context = original_context
    
    # Return the appropriate wrapper based on function type
    if asyncio.iscoroutinefunction(func):
        return cast(F, async_wrapper)
    return cast(F, sync_wrapper)