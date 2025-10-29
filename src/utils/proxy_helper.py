import logging
import os
import socket
import traceback
from functools import wraps
from urllib.parse import urlparse

import certifi
import requests
import urllib3

# Disable SSL warnings globally
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

class ProxyHelper:
    """Helper class to manage proxy and SSL settings for different hosts."""
    
    def __init__(self):
        # Load proxy configuration
        self.proxy_config = {
            'http': os.getenv('HTTP_PROXY'),
            'https': os.getenv('HTTPS_PROXY')
        }
        
        # Load NO_PROXY list
        self.no_proxy_list = os.getenv('NO_PROXY', '').split(',')

        # Override SSL verification setting
        self.verify_ssl = os.getenv('JIRA_VERIFY_SSL', 'False').lower() == 'true'
        
        # Set default SSL cert path for secure connections - prioritize venv path
        self.cert_path = self._find_best_cert_path()
        
        # Known hosts that should bypass proxy (based on our test results and common local services)
        self.direct_hosts = [
            'localhost',
            '127.0.0.1',
            '::1',
            'ollama',
            'ollama.local'
        ]
        
        # External API services that should always use proxy in corporate environments
        self.api_services = [
            'api.anthropic.com',
            'auth.anthropic.com',
            'claude.ai',
            'api.openai.com',
            'huggingface.co',
            'api.github.com',
            'registry.npmjs.org'
        ]
        
        # Add hosts from NO_PROXY that are not wildcard patterns
        for entry in self.no_proxy_list:
            if not entry.startswith('*') and not entry.startswith('.'):
                if entry and ':' in entry:
                    # Handle entries with ports
                    entry = entry.split(':')[0]
                if entry:  # Only add non-empty entries
                    self.direct_hosts.append(entry)
        
        # Remove entries that should always use proxy
        for api_host in self.api_services:
            if api_host in self.direct_hosts:
                self.direct_hosts.remove(api_host)
        
        # Known hosts that need SSL verification disabled based on tests
        # Add your internal hosts here if needed
        self.ssl_disable_hosts = []

        # Add any custom hosts from NO_PROXY that may need SSL disabled
        # You can configure this based on your environment
        
        # Remove duplicates
        self.direct_hosts = list(set(self.direct_hosts))
        self.ssl_disable_hosts = list(set(self.ssl_disable_hosts))
        
        # Host mapping for DNS resolution issues
        # Add your custom host-to-IP mappings here if needed
        self.host_mapping = {}
        
        # Apply host mapping to the socket module
        self._apply_host_mapping()
        
        logger.debug(f"ProxyHelper initialized with {len(self.direct_hosts)} direct hosts and "
                   f"{len(self.ssl_disable_hosts)} SSL-disabled hosts")
        logger.debug(f"Using SSL certificate path: {self.cert_path}")
        logger.debug(f"Internal hosts with SSL disabled: {', '.join(self.ssl_disable_hosts[:5])}...")
    
    def _apply_host_mapping(self):
        """Apply host mapping to socket module for DNS resolution."""
        # Save the original getaddrinfo function
        original_getaddrinfo = socket.getaddrinfo
        
        # Define the patched getaddrinfo function
        def patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
            # Check if host is in our mapping
            if host in self.host_mapping:
                mapped_ip = self.host_mapping[host]
                logger.debug(f"Using mapped IP {mapped_ip} for host {host}")
                return original_getaddrinfo(mapped_ip, port, family, type, proto, flags)
            # Use original function if not in our mapping
            return original_getaddrinfo(host, port, family, type, proto, flags)
        
        # Apply the patch
        socket.getaddrinfo = patched_getaddrinfo
    
    def _find_best_cert_path(self):
        """Find the best certificate path based on the environment."""
        # Priority list of potential certificate paths
        cert_paths = [
            # Environment variable (highest priority)
            os.getenv('SSL_CERT_FILE'),
            
            # Python 3.11 virtual environment path
            os.path.join(os.getcwd(), '.venv311/lib/python3.11/site-packages/certifi/cacert.pem'),
            
            # Default virtual environment path
            os.path.join(os.getcwd(), '.venv/lib/python3.11/site-packages/certifi/cacert.pem'),
            os.path.join(os.getcwd(), '.venv/lib/python3.10/site-packages/certifi/cacert.pem'),
            
            # System Python paths
            '/opt/homebrew/Caskroom/miniforge/base/lib/python3.10/site-packages/certifi/cacert.pem',
            '/opt/homebrew/lib/python3.11/site-packages/certifi/cacert.pem',
            
            # Default certifi path (lowest priority)
            certifi.where()
        ]
        
        # Return the first valid path
        for path in cert_paths:
            if path and os.path.exists(path):
                return path
        
        # Fallback to certifi default
        return certifi.where()
    
    def get_session_for_url(self, url):
        """Get a properly configured session for a URL."""
        hostname = self.get_hostname_from_url(url)
        
        session = requests.Session()
        session.trust_env = False  # Don't use environment proxy settings automatically
        
        # Configure proxy based on hostname
        use_proxy = not self._should_bypass_proxy(hostname)
        
        # Always disable SSL verification as per user's request
        session.verify = False
        
        # Special handling for external API services
        if hostname in self.api_services:
            # Always force proxy for external APIs in corporate environments
            session.proxies = self.proxy_config
            logger.debug(f"Using proxy with SSL verification disabled for external API: {hostname}")
        # Configure proxy for internal services
        elif self._should_disable_ssl(hostname):
            # For internal services, do not use proxy
            session.proxies = {}
            logger.debug(f"Bypassing proxy with SSL verification disabled for internal service: {hostname}")
        else:
            # Standard configuration for other services
            if use_proxy:
                session.proxies = self.proxy_config
                logger.debug(f"Using proxy with SSL verification disabled for {hostname}")
            else:
                session.proxies = {}
                logger.debug(f"Using direct connection with SSL verification disabled for {hostname}")
        
        # Configure with better retry strategy for robustness
        retry_strategy = urllib3.util.retry.Retry(
            total=5,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PUT", "DELETE"]
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def get_hostname_from_url(self, url):
        """Extract hostname from URL."""
        parsed_url = urlparse(url)
        return parsed_url.netloc.split(':')[0]
    
    def _should_bypass_proxy(self, hostname):
        """Determine if a host should bypass proxy based on rules."""
        # External API services should never bypass proxy in corporate environments
        if hostname in self.api_services:
            return False
            
        # Check direct hosts list
        if hostname in self.direct_hosts:
            return True
            
        # Check domain/subdomain patterns in NO_PROXY
        for pattern in self.no_proxy_list:
            # Skip empty entries
            if not pattern:
                continue
                
            # Wildcard subdomain match
            if pattern.startswith('*.') and hostname.endswith(pattern[2:]):
                return True
                
            # Domain suffix match
            if pattern.startswith('.') and hostname.endswith(pattern):
                return True
                
            # IP address with port match
            if ':' in pattern and hostname == pattern.split(':')[0]:
                return True
        
        # Check for localhost variants
        if hostname in ('localhost', '127.0.0.1', '::1'):
            return True
            
        # By default, use proxy
        return False
    
    def _should_disable_ssl(self, hostname):
        """Determine if SSL verification should be disabled for a host."""
        # External API services should always use proper SSL verification
        if hostname in self.api_services:
            return False
            
        # Check if host is in the disable list
        for host in self.ssl_disable_hosts:
            if hostname == host:
                return True
            # Check subdomain
            if hostname.endswith(f".{host}"):
                return True
        
        return False
    
    def should_use_proxy(self, url_or_hostname):
        """Determine if proxy should be used for a URL or hostname."""
        hostname = url_or_hostname
        if '/' in url_or_hostname:
            hostname = self.get_hostname_from_url(url_or_hostname)
        
        return not self._should_bypass_proxy(hostname)
    
    def should_verify_ssl(self, url_or_hostname):
        """Determine if SSL verification should be used for a URL or hostname."""
        # Always return False as per user's request
        return False
    
    def with_appropriate_session(func):
        """Decorator to use appropriate session for a URL."""
        @wraps(func)
        def wrapper(self, url, *args, **kwargs):
            # Create a ProxyHelper instance if not already provided
            helper = kwargs.pop('proxy_helper', ProxyHelper())
            
            # Get a configured session
            session = helper.get_session_for_url(url)
            
            # Call the original function with the session
            return func(self, url, session=session, *args, **kwargs)
        return wrapper
    
    def configure_global_session(self, session):
        """Configure a global session with optimal proxy settings."""
        # Set proxy exceptions
        session.trust_env = False
        
        # Use proxy configuration from environment
        if self.proxy_config.get('https') or self.proxy_config.get('http'):
            session.proxies = self.proxy_config
        
        # Configure retry strategy with more robust settings
        retry_strategy = urllib3.util.retry.Retry(
            total=5,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PUT", "DELETE"]
        )
        
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Always disable SSL verification as per user's request
        session.verify = False
        
        return session
    
    def test_connection(self, url, max_attempts=3):
        """Test connection to a URL and return diagnostics information."""
        hostname = self.get_hostname_from_url(url)
        results = {
            "url": url,
            "hostname": hostname,
            "should_use_proxy": self.should_use_proxy(hostname),
            "should_verify_ssl": self.should_verify_ssl(hostname),
            "ssl_cert_path": self.cert_path,
            "proxy_config": self.proxy_config,
            "success": False,
            "status_code": None,
            "error": None,
            "dns_resolved": False
        }
        
        # Add host mapping information
        if hostname in self.host_mapping:
            results["mapped_ip"] = self.host_mapping[hostname]
        
        # First, test DNS resolution
        try:
            ip = socket.gethostbyname(hostname)
            results["dns_resolved"] = True
            results["ip_address"] = ip
        except socket.gaierror as e:
            results["error"] = f"DNS resolution failed: {str(e)}"
            results["dns_error"] = str(e)
            return results
        
        # Get appropriate session
        session = self.get_session_for_url(url)
        
        # Test connection
        for attempt in range(max_attempts):
            try:
                response = session.head(url, timeout=10)
                results["success"] = True
                results["status_code"] = response.status_code
                results["headers"] = dict(response.headers)
                break
            except requests.exceptions.RequestException as e:
                results["error"] = str(e)
                results["exception_type"] = type(e).__name__
                results["traceback"] = traceback.format_exc()
                # Continue trying next attempt
        
        return results
        
    def get_cli_env_vars(self):
        """Return environment variables dictionary for CLI tools."""
        env_vars = os.environ.copy()
        
        # Ensure proxy settings are correct for CLI tools
        if self.proxy_config.get('http'):
            env_vars['HTTP_PROXY'] = self.proxy_config['http']
            env_vars['http_proxy'] = self.proxy_config['http']
        
        if self.proxy_config.get('https'):
            env_vars['HTTPS_PROXY'] = self.proxy_config['https'] 
            env_vars['https_proxy'] = self.proxy_config['https']
        
        # Set NO_PROXY to include internal domains
        no_proxy_value = 'localhost,127.0.0.1,::1'
        # Add our direct hosts that need to bypass proxy
        for host in self.direct_hosts:
            if host not in ('localhost', '127.0.0.1', '::1'):
                no_proxy_value += f',{host}'
        
        env_vars['NO_PROXY'] = no_proxy_value
        env_vars['no_proxy'] = no_proxy_value
        
        # Disable SSL verification for all tools
        env_vars['CURL_CA_BUNDLE'] = ''
        env_vars['SSL_CERT_FILE'] = ''
        env_vars['NODE_TLS_REJECT_UNAUTHORIZED'] = '0'
        env_vars['NODE_EXTRA_CA_CERTS'] = ''
        env_vars['PYTHONHTTPSVERIFY'] = '0'
        env_vars['REQUESTS_CA_BUNDLE'] = ''
        
        # Add hosts file entries for DNS resolution
        hosts_entries = ""
        for host, ip in self.host_mapping.items():
            hosts_entries += f"\n{ip} {host}"
        
        env_vars['HOSTS_ENTRIES'] = hosts_entries
        
        return env_vars

# Create a singleton instance
proxy_helper = ProxyHelper()

# Function to get a CLI-ready environment with proper proxy settings
def get_cli_environment():
    """Get an environment dictionary with proxy settings for CLI tools."""
    return proxy_helper.get_cli_env_vars()

# Function to create and return a request session for a specific URL
def get_session_for_url(url):
    """Get a properly configured session for the given URL."""
    return proxy_helper.get_session_for_url(url)

# Function to test connectivity to a URL
def test_url_connection(url):
    """Test connection to a URL and return diagnostics."""
    return proxy_helper.test_connection(url) 