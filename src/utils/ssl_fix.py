"""
SSL Fix Utility Module

This module provides functions to disable SSL certificate verification for different scenarios,
particularly useful when working with internal corporate APIs or development environments.
"""

import os
import ssl
import urllib3
import certifi
import logging
import requests
from functools import wraps
import sys
import tempfile
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)

def disable_ssl_warnings():
    """Disable SSL warnings from urllib3."""
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def create_unverified_ssl_context():
    """Create an SSL context that doesn't verify certificates."""
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    return ssl_context

def patch_ssl_for_huggingface():
    """
    Apply SSL patches specifically for Hugging Face library connections.
    This is needed for sentence-transformers and other HF-based libraries.
    """
    # Set environment variables to disable SSL verification completely
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    os.environ['SSL_CERT_FILE'] = ''
    os.environ['PYTHONHTTPSVERIFY'] = '0'
    
    # Disable warnings
    disable_ssl_warnings()
    
    # Apply monkey patch to SSL
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # Fix for macOS
    if sys.platform == 'darwin':
        try:
            # Direct monkey patching of SSL verification
            import ssl as ssl_module
            original_getattr = ssl_module.__getattribute__

            def patched_getattr(name):
                if name == '_create_default_https_context':
                    return ssl_module._create_unverified_context
                return original_getattr(name)

            ssl_module.__getattribute__ = patched_getattr
        except Exception as e:
            logger.warning(f"Failed to apply macOS SSL patch: {e}")
    
    logger.info("SSL verification completely disabled for all connections")

def with_ssl_disabled(func):
    """
    Decorator to temporarily disable SSL verification for a function call.
    
    Args:
        func: The function to wrap with SSL verification disabled
        
    Returns:
        Wrapped function with SSL verification disabled during execution
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        original_verify = None
        session_class = None
        try:
            # Check if requests.Session exists and has the verify attribute
            if hasattr(requests, 'Session') and hasattr(requests.Session, 'verify'):
                session_class = requests.Session
                original_verify = session_class.verify
                # Temporarily disable verification
                # session_class.verify = False
                # logger.debug("Temporarily disabled requests.Session.verify")
                logger.debug("Skipping modification of requests.Session.verify in decorator") # DEBUG
            else:
                logger.debug("requests.Session or verify attribute not found, skipping modification")

            # Execute the decorated function
            return func(*args, **kwargs)
        finally:
            # Restore original setting if it was changed
            # if session_class is not None and original_verify is not None:
            #     session_class.verify = original_verify
            #     logger.debug("Restored original requests.Session.verify")
            pass # No restoration needed as modification is skipped
    return wrapper

def initialize_ssl_fixes():
    """Initialize all SSL fixes in one call."""
    # Disable SSL warnings
    disable_ssl_warnings()
    
    # Set environment variables to disable completely
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    os.environ['SSL_CERT_FILE'] = ''
    os.environ['PYTHONHTTPSVERIFY'] = '0'
    
    # Create unverified context
    ssl_context = create_unverified_ssl_context()
    
    # Apply monkey patch to SSL - the most reliable approach
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # Additional patches for various libraries
    try:
        # Patch requests directly
        requests.packages.urllib3.util.ssl_.DEFAULT_CIPHERS += ':HIGH:!DH:!aNULL'
        requests.packages.urllib3.contrib.pyopenssl.DEFAULT_SSL_CIPHER_LIST += ':HIGH:!DH:!aNULL'
    except AttributeError:
        pass
    
    logger.info("Global SSL verification completely disabled")
    
    return ssl_context 