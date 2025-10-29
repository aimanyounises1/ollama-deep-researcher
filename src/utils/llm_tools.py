"""LLM-related utility functions.
"""
import logging
import ssl

import urllib3

logger = logging.getLogger(__name__)

def patch_ssl_for_huggingface():
    """Patch SSL for HuggingFace compatibility."""
    try:
        # Disable SSL verification warnings
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Create a custom SSL context
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Set the default SSL context
        ssl._create_default_https_context = lambda: ssl_context
        
        logger.info("SSL patched for HuggingFace compatibility")
    except Exception as e:
        logger.error(f"Error patching SSL for HuggingFace: {e}")
        raise 