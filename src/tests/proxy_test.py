import logging
import os
import socket
import ssl
from urllib.parse import urlparse

import requests
import urllib3
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger("ProxyTester")

# Disable SSL warnings for cleaner output
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class ProxyConnectionTester:
    """Test class to determine which services need proxy and which don't."""
    
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Load proxy configuration
        self.proxy_host = os.getenv('PROXY_HOST')
        self.proxy_port = int(os.getenv('PROXY_PORT', 8080))
        self.http_proxy = os.getenv('HTTP_PROXY')
        self.https_proxy = os.getenv('HTTPS_PROXY')
        self.no_proxy_list = os.getenv('NO_PROXY', '').split(',')
        
        # Configure proxies for requests
        self.proxy_config = {
            'http': self.http_proxy,
            'https': self.https_proxy
        }
        
        # Load service configs from environment
        self.jira_server = os.getenv('JIRA_SERVER')
        self.p4_port = os.getenv('P4PORT')
        self.p4_web_base = os.getenv('P4_WEB_BASE')
        self.solutionbook_domain = os.getenv('SOLUTIONBOOK_DOMAIN')
        self.elk_api_url = os.getenv('ELK_API_URL')
        self.ollama_endpoint = os.getenv('OLLAMA_ENDPOINT')
        
        logger.info("Loaded proxy configuration from environment:")
        logger.info(f"HTTP_PROXY: {self.http_proxy}")
        logger.info(f"HTTPS_PROXY: {self.https_proxy}")
        logger.info(f"NO_PROXY has {len(self.no_proxy_list)} entries")
        
        # Test endpoints
        self.endpoints = self._build_endpoints()
    
    def _build_endpoints(self):
        """Build list of endpoints to test based on environment variables."""
        endpoints = [
            # Local services
            {'name': 'Ollama', 'url': self.ollama_endpoint or 'http://localhost:11434'},
            
            # External services without parameters
            {'name': 'HuggingFace', 'url': 'https://huggingface.co'},
            {'name': 'LangSmith', 'url': 'https://api.smith.langchain.com'},
            {'name': 'Google', 'url': 'https://www.google.com'},
        ]
        
        # Add JIRA if configured
        if self.jira_server:
            endpoints.append({'name': 'JIRA', 'url': self.jira_server})
        
        # Add Perforce if configured
        if self.p4_web_base:
            endpoints.append({'name': 'Perforce Web', 'url': self.p4_web_base})
        
        # Add SolutionBook if configured  
        if self.solutionbook_domain:
            endpoints.append({'name': 'SolutionBook', 'url': f'https://{self.solutionbook_domain}'})
        
        # Add ELK if configured
        if self.elk_api_url:
            endpoints.append({'name': 'ELK', 'url': self.elk_api_url})
        
        return endpoints
    
    def _should_bypass_proxy(self, hostname):
        """Determine if a host should bypass proxy based on NO_PROXY rules."""
        # Check exact matches
        if hostname in self.no_proxy_list:
            return True
            
        # Check domain/subdomain patterns
        for pattern in self.no_proxy_list:
            # Wildcard subdomain match
            if pattern.startswith('*.') and hostname.endswith(pattern[2:]):
                return True
            # Domain suffix match
            if pattern.startswith('.') and hostname.endswith(pattern):
                return True
            # IP address with port
            if ':' in pattern and pattern.split(':')[0] == hostname.split(':')[0]:
                return True
        
        return False
    
    def _test_direct_tcp_connection(self, hostname, port, timeout=5):
        """Test direct TCP connection to a host:port without proxy."""
        try:
            sock = socket.create_connection((hostname, port), timeout=timeout)
            sock.close()
            return True
        except (socket.timeout, OSError):
            return False
    
    def _test_ssl_cert_validity(self, hostname, port=443):
        """Test if a server's SSL certificate is valid."""
        try:
            context = ssl.create_default_context()
            with socket.create_connection((hostname, port)) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    return True, None
        except ssl.SSLCertVerificationError as e:
            return False, f"Certificate verification failed: {str(e)}"
        except Exception as e:
            return False, f"SSL connection error: {str(e)}"
    
    def test_endpoint(self, endpoint):
        """Test connection to an endpoint with various settings."""
        name = endpoint['name']
        url = endpoint['url']
        parsed_url = urlparse(url)
        hostname = parsed_url.netloc.split(':')[0]
        port = parsed_url.port or (443 if parsed_url.scheme == 'https' else 80)
        
        logger.info(f"Testing endpoint: {name} ({url})")
        
        results = {
            'name': name,
            'url': url,
            'hostname': hostname,
            'direct_tcp_works': False,
            'direct_http_works': False,
            'direct_http_no_ssl_works': False,
            'proxy_http_works': False,
            'proxy_http_no_ssl_works': False,
            'ssl_valid': False,
            'ssl_error': None,
            'no_proxy_setting': self._should_bypass_proxy(hostname),
            'best_method': None,
            'config_recommendations': {}
        }
        
        # Test 1: Direct TCP connection
        tcp_result = self._test_direct_tcp_connection(hostname, port)
        results['direct_tcp_works'] = tcp_result
        logger.info(f"Direct TCP connection: {'SUCCESS' if tcp_result else 'FAILED'}")
        
        # Test 2: SSL certificate validity (for https)
        if parsed_url.scheme == 'https':
            ssl_valid, ssl_error = self._test_ssl_cert_validity(hostname)
            results['ssl_valid'] = ssl_valid
            results['ssl_error'] = ssl_error
            logger.info(f"SSL certificate validity: {'VALID' if ssl_valid else 'INVALID'}")
            if ssl_error:
                logger.info(f"SSL error: {ssl_error}")
        
        # Test 3: Direct HTTP connection with SSL verification
        try:
            direct_session = requests.Session()
            direct_session.proxies = {}  # No proxy
            direct_session.verify = True  # Use SSL verification
            direct_response = direct_session.get(url, timeout=10)
            results['direct_http_works'] = direct_response.status_code < 400
            results['direct_http_status'] = direct_response.status_code
            logger.info(f"Direct HTTP with SSL: {'SUCCESS' if results['direct_http_works'] else 'FAILED'} (Status: {direct_response.status_code})")
        except Exception as e:
            results['direct_http_error'] = str(e)
            logger.info(f"Direct HTTP with SSL: FAILED - {str(e)}")
        
        # Test 4: Direct HTTP without SSL verification
        if parsed_url.scheme == 'https':
            try:
                direct_no_ssl_session = requests.Session()
                direct_no_ssl_session.proxies = {}  # No proxy
                direct_no_ssl_session.verify = False  # No SSL verification
                direct_no_ssl_response = direct_no_ssl_session.get(url, timeout=10)
                results['direct_http_no_ssl_works'] = direct_no_ssl_response.status_code < 400
                results['direct_http_no_ssl_status'] = direct_no_ssl_response.status_code
                logger.info(f"Direct HTTP without SSL: {'SUCCESS' if results['direct_http_no_ssl_works'] else 'FAILED'} (Status: {direct_no_ssl_response.status_code})")
            except Exception as e:
                results['direct_http_no_ssl_error'] = str(e)
                logger.info(f"Direct HTTP without SSL: FAILED - {str(e)}")
        
        # Test 5: HTTP connection via proxy with SSL verification
        try:
            proxy_session = requests.Session()
            proxy_session.proxies = self.proxy_config
            proxy_session.verify = True  # Use SSL verification
            proxy_response = proxy_session.get(url, timeout=10)
            results['proxy_http_works'] = proxy_response.status_code < 400
            results['proxy_http_status'] = proxy_response.status_code
            logger.info(f"Proxy HTTP with SSL: {'SUCCESS' if results['proxy_http_works'] else 'FAILED'} (Status: {proxy_response.status_code})")
        except Exception as e:
            results['proxy_http_error'] = str(e)
            logger.info(f"Proxy HTTP with SSL: FAILED - {str(e)}")
        
        # Test 6: HTTP connection via proxy without SSL verification
        if parsed_url.scheme == 'https':
            try:
                proxy_no_ssl_session = requests.Session()
                proxy_no_ssl_session.proxies = self.proxy_config
                proxy_no_ssl_session.verify = False  # No SSL verification
                proxy_no_ssl_response = proxy_no_ssl_session.get(url, timeout=10)
                results['proxy_http_no_ssl_works'] = proxy_no_ssl_response.status_code < 400
                results['proxy_http_no_ssl_status'] = proxy_no_ssl_response.status_code
                logger.info(f"Proxy HTTP without SSL: {'SUCCESS' if results['proxy_http_no_ssl_works'] else 'FAILED'} (Status: {proxy_no_ssl_response.status_code})")
            except Exception as e:
                results['proxy_http_no_ssl_error'] = str(e)
                logger.info(f"Proxy HTTP without SSL: FAILED - {str(e)}")
        
        # Determine best method
        if results['direct_http_works']:
            results['best_method'] = 'direct_with_ssl'
        elif results['direct_http_no_ssl_works']:
            results['best_method'] = 'direct_without_ssl'
        elif results['proxy_http_works']:
            results['best_method'] = 'proxy_with_ssl'
        elif results['proxy_http_no_ssl_works']:
            results['best_method'] = 'proxy_without_ssl'
        else:
            results['best_method'] = 'none'
        
        logger.info(f"Best connection method: {results['best_method']}")
        
        # Generate config recommendations
        results['config_recommendations'] = self._generate_recommendations(results)
        logger.info(f"Recommendations: {results['config_recommendations']}")
        
        logger.info("-" * 50)
        return results
    
    def _generate_recommendations(self, result):
        """Generate configuration recommendations based on test results."""
        recommendations = {}
        
        # Proxy recommendation
        if result['best_method'] in ('direct_with_ssl', 'direct_without_ssl'):
            recommendations['proxy'] = "No proxy needed"
            if not result['no_proxy_setting']:
                recommendations['no_proxy'] = f"Add {result['hostname']} to NO_PROXY"
        elif result['best_method'] in ('proxy_with_ssl', 'proxy_without_ssl'):
            recommendations['proxy'] = "Use proxy"
            if result['no_proxy_setting']:
                recommendations['no_proxy'] = f"Remove {result['hostname']} from NO_PROXY"
        
        # SSL recommendation
        if result['best_method'] in ('direct_without_ssl', 'proxy_without_ssl'):
            recommendations['ssl'] = "Disable SSL verification"
        
        return recommendations
    
    def run_all_tests(self):
        """Run tests for all endpoints."""
        results = []
        
        for endpoint in self.endpoints:
            result = self.test_endpoint(endpoint)
            results.append(result)
        
        return results
    
    def generate_proxy_helper_class(self, results):
        """Generate a proxy helper class based on test results."""
        direct_hosts = []
        ssl_disable_hosts = []
        
        for r in results:
            if r['best_method'] in ('direct_with_ssl', 'direct_without_ssl'):
                direct_hosts.append(r['hostname'])
            if r['best_method'] in ('direct_without_ssl', 'proxy_without_ssl'):
                ssl_disable_hosts.append(r['hostname'])
        
        # Convert lists to string representation for inclusion in the template
        direct_hosts_str = ", ".join([f"'{host}'" for host in direct_hosts])
        ssl_disable_hosts_str = ", ".join([f"'{host}'" for host in ssl_disable_hosts])
        
        code = f"""
import os
import requests
from urllib.parse import urlparse
import logging
import urllib3

# Disable SSL warnings globally
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

class ProxyHelper:
    \"\"\"Helper class to manage proxy and SSL settings for different hosts.\"\"\"
    
    def __init__(self):
        # Load proxy configuration
        self.proxy_config = {{
            'http': os.getenv('HTTP_PROXY'),
            'https': os.getenv('HTTPS_PROXY')
        }}
        
        # Hosts that should bypass proxy
        self.direct_hosts = [{direct_hosts_str}]
        
        # Hosts that need SSL verification disabled
        self.ssl_disable_hosts = [{ssl_disable_hosts_str}]
    
    def get_session_for_url(self, url):
        \"\"\"Get a properly configured session for a URL.\"\"\"
        parsed_url = urlparse(url)
        hostname = parsed_url.netloc.split(':')[0]
        
        session = requests.Session()
        
        # Configure proxy
        if hostname in self.direct_hosts:
            logger.debug(f"Using direct connection for {{hostname}}")
            session.proxies = {{}}
        else:
            logger.debug(f"Using proxy for {{hostname}}")
            session.proxies = self.proxy_config
        
        # Configure SSL verification
        if hostname in self.ssl_disable_hosts:
            logger.debug(f"Disabling SSL verification for {{hostname}}")
            session.verify = False
        else:
            session.verify = True
        
        return session
    
    def get_hostname_from_url(self, url):
        \"\"\"Extract hostname from URL.\"\"\"
        parsed_url = urlparse(url)
        return parsed_url.netloc.split(':')[0]
    
    def should_use_proxy(self, url_or_hostname):
        \"\"\"Determine if proxy should be used for a URL or hostname.\"\"\"
        hostname = url_or_hostname
        if '/' in url_or_hostname:
            hostname = self.get_hostname_from_url(url_or_hostname)
        
        return hostname not in self.direct_hosts
    
    def should_verify_ssl(self, url_or_hostname):
        \"\"\"Determine if SSL verification should be used for a URL or hostname.\"\"\"
        hostname = url_or_hostname
        if '/' in url_or_hostname:
            hostname = self.get_hostname_from_url(url_or_hostname)
        
        return hostname not in self.ssl_disable_hosts
        
    @staticmethod
    def with_appropriate_session(func):
        \"\"\"Decorator to use appropriate session for a URL.\"\"\"
        def wrapper(self, url, *args, **kwargs):
            helper = ProxyHelper()
            session = helper.get_session_for_url(url)
            return func(self, url, session=session, *args, **kwargs)
        return wrapper
"""
        
        return code
    
    def generate_recommendations_summary(self, results):
        """Generate a summary of recommendations."""
        no_proxy_changes = []
        ssl_recommendations = []
        
        for result in results:
            for key, value in result.get('config_recommendations', {}).items():
                if key == 'no_proxy':
                    no_proxy_changes.append(value)
                elif key == 'ssl':
                    ssl_recommendations.append(f"{result['hostname']}: {value}")
        
        recommendations = []
        
        if no_proxy_changes:
            recommendations.append("NO_PROXY changes:")
            recommendations.extend([f"  {change}" for change in no_proxy_changes])
        
        if ssl_recommendations:
            recommendations.append("SSL recommendations:")
            recommendations.extend([f"  {rec}" for rec in ssl_recommendations])
        
        return "\n".join(recommendations)


# Execute tests if run directly
if __name__ == "__main__":
    print("Starting proxy connection tests...")
    tester = ProxyConnectionTester()
    results = tester.run_all_tests()
    
    # Generate and print recommendations
    summary = tester.generate_recommendations_summary(results)
    print("\n===== RECOMMENDATIONS =====\n")
    print(summary)
    
    # Generate helper class code
    helper_code = tester.generate_proxy_helper_class(results)
    print("\n===== PROXY HELPER CLASS =====\n")
    print(helper_code)
    
    print("\nTest complete! Use the generated ProxyHelper class to manage connections in your application.") 