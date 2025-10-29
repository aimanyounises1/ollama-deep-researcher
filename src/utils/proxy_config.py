"""Proxy Configuration Utility Module

This module provides utilities for configuring proxy settings for different services
such as SolutionBook and JIRA. It uses ProxyHelper under the hood to determine
the best proxy settings for each service.
"""

import logging
import os

from src.utils.proxy_helper import ProxyHelper

logger = logging.getLogger(__name__)

class ProxyConfigurator:
    """Utility class for configuring proxy settings for different services."""
    
    def __init__(self):
        """Initialize the proxy configurator."""
        self.helper = ProxyHelper()
        # Save original environment for restoration
        self.original_env = {
            'HTTP_PROXY': os.environ.get('HTTP_PROXY'),
            'HTTPS_PROXY': os.environ.get('HTTPS_PROXY'),
            'NO_PROXY': os.environ.get('NO_PROXY'),
        }
    
    def configure_solutionbook(self, querier):
        """Configure SolutionBookQuerier with appropriate proxy settings.
        
        Args:
            querier: An initialized SolutionBookQuerier instance
            
        Returns:
            The configured SolutionBookQuerier instance
        """
        logger.info(f"Configuring proxy for SolutionBook ({querier.domain})")
        
        # Get a session with appropriate proxy settings
        url = f"https://{querier.domain}"
        session = self.helper.get_session_for_url(url)
        
        # Replace the default session with our configured one
        querier.session = session
        
        # Log connection details
        proxy_info = "No proxy" if not session.proxies else f"Proxy: {session.proxies}"
        ssl_info = "SSL verification disabled" if not session.verify else "SSL verification enabled"
        logger.info(f"SolutionBook configured with: {proxy_info}, {ssl_info}")
        
        return querier
    
    def configure_jira(self, jira_project):
        """Configure JiraProject with appropriate proxy settings.
        JIRA already uses environment variables for proxy settings,
        so we just need to apply the appropriate environment.
        
        Args:
            jira_project: An initialized JiraProject instance
            
        Returns:
            The same JiraProject instance (no changes needed)
        """
        logger.info(f"Configuring proxy for JIRA ({jira_project.server})")
        
        # Apply environment variables suitable for CLI tools
        env_vars = self.helper.get_cli_env_vars()
        for key, value in env_vars.items():
            if value is not None:
                os.environ[key] = value
        
        # Log connection details
        http_proxy = os.environ.get('HTTP_PROXY', 'Not set')
        https_proxy = os.environ.get('HTTPS_PROXY', 'Not set')
        ssl_verify = "Disabled" if os.environ.get('PYTHONHTTPSVERIFY') == '0' else "Enabled"
        
        logger.info(f"JIRA environment configured: HTTP_PROXY={http_proxy}, "
                   f"HTTPS_PROXY={https_proxy}, SSL verification: {ssl_verify}")
        
        return jira_project
    
    def reset_environment(self):
        """Reset environment variables to their original values."""
        for key, value in self.original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]
        
        logger.info("Environment variables reset to original values")


def configure_connections():
    """Prepare connection configuration functions for SolutionBook and JIRA.
    This is a factory function that returns functions for configuring each service.
    
    Returns:
        Tuple of (configure_solutionbook_fn, configure_jira_fn)
    """
    configurator = ProxyConfigurator()
    
    def configure_solutionbook(domain, token):
        """Create and configure a SolutionBookQuerier instance.
        
        Args:
            domain: SolutionBook domain
            token: Authentication token
            
        Returns:
            Configured SolutionBookQuerier instance
        """
        from src.assistant.tools.SolutionBookTool import SolutionBookQuerier
        
        # Initialize SolutionBookQuerier
        querier = SolutionBookQuerier(domain=domain, auth_token=token)
        
        # Configure proxy settings
        return configurator.configure_solutionbook(querier)
    
    def configure_jira(server, token):
        """Create and configure a JiraProject instance.
        
        Args:
            server: JIRA server address
            token: Authentication token
            
        Returns:
            Configured JiraProject instance
        """
        from src.assistant.tools.jira_tool import JiraProject
        
        # Set up environment before creating JiraProject
        configurator.configure_jira(None)
        
        # Initialize JiraProject
        jira_project = JiraProject(server=server, token=token)
        
        return jira_project
    
    return configure_solutionbook, configure_jira


# Simple usage example:
"""
# Get configuration functions
configure_solutionbook, configure_jira = configure_connections()

# Initialize SolutionBookQuerier
domain = os.getenv('SOLUTIONBOOK_DOMAIN')
token = os.getenv('SOLUTIONBOOK_ACCESS_TOKEN')
sb_querier = configure_solutionbook(domain, token)

# Initialize JiraProject
server = os.getenv('JIRA_SERVER')
token = os.getenv('JIRA_API_TOKEN')
jira_project = configure_jira(server, token)

# Use the tools...

# Reset environment when done
configurator = ProxyConfigurator()
configurator.reset_environment()
""" 