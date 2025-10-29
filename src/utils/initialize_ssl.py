"""SSL initialization utilities for secure connections.
"""
import logging
import ssl

import urllib3

logger = logging.getLogger(__name__)

def initialize_ssl_fixes():
    """Initialize SSL fixes for secure connections."""
    try:
        # Disable SSL verification warnings
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Create a custom SSL context
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Set the default SSL context
        ssl._create_default_https_context = lambda: ssl_context
        
        logger.info("SSL fixes initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing SSL fixes: {e}")
        raise 