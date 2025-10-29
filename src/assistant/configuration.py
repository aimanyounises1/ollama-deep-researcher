# src/assistant/configuration.py

import logging
import os
from typing import Any, Dict, List, Optional, Union
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)

class Configuration:
    """
    Global configuration singleton class for deep researcher settings.
    Handles both default and runtime configuration parameters.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(Configuration, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize configuration with default values."""
        if self._initialized:
            return
            
        # System settings
        self.debug_mode = os.environ.get('DEBUG', 'false').lower() == 'true'
        self.disable_ssl_verification = os.environ.get('SKIP_SSL_VERIFY', 'false').lower() == 'true'
        
        # LLM Configuration
        self.ollama_base_url = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.local_llm = os.environ.get('LOCAL_LLM', 'qwen3:30b-a3b')
        self.embedding_model = os.environ.get('EMBEDDING_MODEL', 'nomic-embed-text')
        
        # Tool Configuration
        self.enable_jira = os.environ.get('ENABLE_JIRA', 'true').lower() == 'true'
        self.enable_confluence = os.environ.get('ENABLE_CONFLUENCE', 'true').lower() == 'true'
        self.enable_perforce = os.environ.get('ENABLE_PERFORCE', 'true').lower() == 'true'
        self.enable_web = os.environ.get('ENABLE_WEB', 'true').lower() == 'true'
        
        # Tool Endpoints
        self.jira_url = os.environ.get('JIRA_URL', '')
        self.jira_username = os.environ.get('JIRA_USERNAME', '')
        self.jira_token = os.environ.get('JIRA_TOKEN', '')
        
        self.confluence_url = os.environ.get('CONFLUENCE_URL', '')
        self.confluence_username = os.environ.get('CONFLUENCE_USERNAME', '')
        self.confluence_token = os.environ.get('CONFLUENCE_TOKEN', '')
        
        self.perforce_port = os.environ.get('P4PORT', '')
        self.perforce_user = os.environ.get('P4USER', '')
        self.perforce_password = os.environ.get('P4PASSWD', '')
        self.perforce_client = os.environ.get('P4CLIENT', '')
        
        # Processing Settings
        self.max_jira_results = int(os.environ.get('MAX_JIRA_RESULTS', '50'))
        self.max_confluence_results = int(os.environ.get('MAX_CONFLUENCE_RESULTS', '30'))
        self.max_perforce_results = int(os.environ.get('MAX_PERFORCE_RESULTS', '30'))
        self.max_web_results = int(os.environ.get('MAX_WEB_RESULTS', '20'))
        
        # Cache settings
        self.use_cache = os.environ.get('USE_CACHE', 'true').lower() == 'true'
        self.cache_dir = os.environ.get('CACHE_DIR', '.cache')
        self.cache_ttl = int(os.environ.get('CACHE_TTL', '3600'))  # Default 1 hour
        
        # Advanced settings
        self.timeout = int(os.environ.get('DEFAULT_TIMEOUT', '120'))  # Default 2 minutes
        self.chunk_overlap = int(os.environ.get('CHUNK_OVERLAP', '100'))
        self.chunk_size = int(os.environ.get('CHUNK_SIZE', '1000'))
        
        # Mark as initialized
        self._initialized = True
        
        if self.debug_mode:
            logger.info("Configuration initialized with debug mode enabled")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Configuration':
        """
        Create or update Configuration from a dictionary.
        
        Args:
            config_dict: Dictionary of configuration values to set
            
        Returns:
            Configuration instance with updated values
        """
        instance = cls()
        
        for key, value in config_dict.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
                
        return instance
    
    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> 'Configuration':
        """
        Extract configuration from LangChain RunnableConfig if available.
        
        Args:
            config: Optional RunnableConfig object from LangChain
            
        Returns:
            Configuration instance with values from RunnableConfig
        """
        instance = cls()
        
        if config is None:
            return instance
            
        # Extract configurable values from RunnableConfig if available
        if isinstance(config, dict):
            # Check for LangSmith configurable values
            config_dict = config.get('configurable', {})
            
            if isinstance(config_dict, dict):
                for key, value in config_dict.items():
                    if hasattr(instance, key):
                        setattr(instance, key, value)
        
        return instance
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary, excluding sensitive values.
        
        Returns:
            Dict with configuration values
        """
        # Use vars to get all instance variables
        config_dict = vars(self).copy()
        
        # Remove non-configuration fields
        config_dict.pop('_initialized', None)
        
        # Remove sensitive data
        sensitive_fields = [
            'jira_token', 'confluence_token', 
            'perforce_password', 'perforce_user'
        ]
        
        for field in sensitive_fields:
            if field in config_dict:
                config_dict[field] = '***REDACTED***'
                
        return config_dict