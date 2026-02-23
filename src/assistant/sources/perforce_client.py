import asyncio
import logging
import os
import re
import ssl
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any  # Added Optional and Any

import nest_asyncio
from P4 import P4, P4Exception
from cachetools import TTLCache
from dotenv import load_dotenv

# from sentence_transformers import SentenceTransformer # Keep commented unless needed

# Import SecurityAnalyzer if it exists and is needed
try:
    from src.assistant.analyzers.security_analyzer import SecurityAnalyzer
except ImportError:
    # Define a dummy class if SecurityAnalyzer is not found
    class SecurityAnalyzer:
        @staticmethod
        def is_sensitive_file(filename: str) -> bool: return False

        @staticmethod
        def redact_sensitive_data(content: str) -> str: return content

        @staticmethod
        def analyze_diff(diff_content: str) -> List: return []

        @staticmethod
        def analyze_file_content(file_content: str, filename: str = '') -> List: return []


    logging.warning("SecurityAnalyzer not found, using dummy class.")

# Allow nested event loops (e.g. in Streamlit)
nest_asyncio.apply()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)

# Performance configuration
changelist_cache = TTLCache(maxsize=5000, ttl=1800)  # 30-minute TTL
BATCH_SIZE = int(os.getenv("P4_BATCH_SIZE", "50"))

# MTV pattern matching configuration
MTV_PATTERN = r'\bMTV\d{4,}\b'  # Matches "MTV" followed by 4+ digits
CACHE_SIZE = 1000  # Number of changelists to search through


# Function to disable SSL verification context for development
def with_ssl_disabled(func):
    """Decorator to disable SSL verification when running the function."""

    def wrapper(*args, **kwargs):
        original_context = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context
        try:
            return func(*args, **kwargs)
        finally:
            ssl._create_default_https_context = original_context

    return wrapper


class PerforceHelper:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if PerforceHelper._initialized:
            return

        # Removed load_dotenv() to prevent conflicts with main application loading
        # Environment variables should already be loaded in run_graph_3.py
        self.p4 = P4()
        self.max_changes: int = int(os.getenv("P4_MAX_CHANGES", "1000"))  # Configurable max changes

        # --- START Enhancement: Load Swarm URL ---
        self.p4_swarm_url = os.getenv("P4_SWARM_URL", "").rstrip('/')
        if self.p4_swarm_url:
            logger.info(f"Perforce Swarm URL configured: {self.p4_swarm_url}")
        else:
            logger.warning("P4_SWARM_URL environment variable not set. Swarm links will not be generated.")
        # --- END Enhancement ---

        # Initialize core attributes
        self.depot_path = os.getenv('DEPOT_PATH')
        if not self.depot_path:
            raise ValueError("DEPOT_PATH must be set in .env file")

        # Initialize connection parameters
        self.p4.client = os.getenv('P4CLIENT')
        self.p4.port = os.getenv("P4PORT")
        self.p4.user = os.getenv('P4USER')
        self.p4.password = os.getenv('P4PASSWD')  # Load password here

        # Validate required env vars
        required_env = ["P4USER", "P4PORT", "P4CLIENT", "P4PASSWD"]
        missing_vars = [evar for evar in required_env if not os.getenv(evar)]
        if missing_vars:
            logger.error(f"Missing required Perforce environment variables: {', '.join(missing_vars)}")
            raise ValueError(f"Missing required Perforce environment variables: {', '.join(missing_vars)}")

        # Log connection details safely
        logger.info(f"Perforce connection details:\n"
                    f"Client: {self.p4.client}\n"
                    f"Port: {self.p4.port}\n"
                    f"User: {self.p4.user}\n"
                    f"Depot Path: {self.depot_path}")
        logger.info("P4PASSWD is set (value hidden)")

        # Configure P4 timeout
        timeout_value = int(os.getenv('P4TIMEOUT', '520'))
        os.environ["P4TIMEOUT"] = str(timeout_value)  # Set as environment variable for P4Python
        logger.info(f"Perforce timeout set to {timeout_value} seconds")

        # Add changelist cache
        self.change_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour cache

        # Error logging directory
        self.error_log_path = Path("perforce_logs")
        try:
            self.error_log_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        except Exception as e:
            logger.error(f"Could not create error log directory {self.error_log_path}: {e}")
            self.error_log_path = Path(".")  # Fallback to current directory

        logger.info(f"Error logging directory configured at: {self.error_log_path.absolute()}")

        # MTV caches
        self.mtv_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour cache

        # Thread pool for synchronous P4Python calls
        self._executor = ThreadPoolExecutor(max_workers=int(os.getenv("P4_EXECUTOR_WORKERS", "4")))
        self._executor_shutdown = False  # Flag to track if executor is shut down

        PerforceHelper._initialized = True

    def _log_p4_error(self, e: P4Exception, context: str = ""):
        """
        Enhanced error logging for P4Exception with detailed categorization.
        Based on best practices from P4Python documentation.

        Args:
            e: The P4Exception that occurred
            context: The context where the exception occurred
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            error_file = self.error_log_path / f"p4_error_{timestamp}.log"
            error_message = f"Context: {context}\n"
            error_message += f"Error Type: {type(e).__name__}\n"
            
            # Add severity classification from P4Python
            severity_level = "Unknown"
            if hasattr(e, 'severity'):
                severity_code = e.severity
                if severity_code == 0:
                    severity_level = "Empty (Information only)"
                elif severity_code == 1:
                    severity_level = "Warning"
                elif severity_code == 2:
                    severity_level = "Error"
                elif severity_code == 3:
                    severity_level = "Critical"
                error_message += f"Severity: {severity_level} ({severity_code})\n"
            
            # Add generic information
            if hasattr(e, 'generic'):
                generic_code = e.generic
                generic_text = self._get_generic_error_text(generic_code)
                error_message += f"Generic Code: {generic_code} ({generic_text})\n"
            
            error_message += f"Message: {str(e)}\n"
            
            # Enhanced error details processing
            if hasattr(e, 'errors') and e.errors:
                error_message += "Detailed Errors:\n"
                for i, err in enumerate(e.errors):
                    error_message += f"  [{i+1}] {err}\n"
            
            if hasattr(e, 'warnings') and e.warnings:
                error_message += "\nWarnings:\n"
                for i, warn in enumerate(e.warnings):
                    error_message += f"  [{i+1}] {warn}\n"
            
            # Add command information if available
            if hasattr(e, 'cmdline'):
                error_message += f"\nCommand: {e.cmdline}\n"

            with open(error_file, "w") as f:
                f.write(error_message)
            logger.error(f"Perforce Error ({context}): {self._sanitize_error(e)}. Details logged to {error_file}")
        except Exception as log_err:
            logger.error(f"Failed to write P4 error log: {log_err}")
            # Log original error to main logger as fallback
            logger.error(f"Original P4 Error ({context}): {type(e).__name__} - {str(e)}")

    def _get_generic_error_text(self, generic_code: int) -> str:
        """
        Get descriptive text for generic error codes from P4Python.
        
        Args:
            generic_code: The generic code from P4Exception
            
        Returns:
            Description of the generic error code
        """
        error_texts = {
            1: "Error in command usage",
            2: "Connection error",
            3: "Maximum results exceeded",
            4: "Locked file action failed",
            5: "Command failed",
            6: "Network/system error",
            7: "SSL/TLS error",
            8: "Unicode error",
            9: "Syntax error",
            # Add more codes as documented in P4Python
        }
        return error_texts.get(generic_code, "Unknown error code")

    def connect(self):
        """
        Establish connection to Perforce server with enhanced login handling and error recovery.
        Based on best practices from P4Python documentation.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self.p4.connected():
                logger.debug("Already connected to Perforce.")
                return True  # Already connected

            logger.info("Attempting to connect to Perforce server...")
            self.p4.connect()
            logger.info("Connected. Checking login status...")

            # Check for server info to verify basic connectivity
            try:
                server_info = self.p4.run_info()
                if server_info:
                    logger.debug(f"Server version: {server_info[0].get('serverVersion', 'unknown')}")
            except P4Exception as info_err:
                logger.warning(f"Connected but couldn't get server info: {info_err}")

            # Check if already logged in
            try:
                login_stat = self.p4.run_login("-s")
                # Check if ticket is valid (P4Python might return bytes/str or empty list)
                if login_stat and isinstance(login_stat, list) and len(login_stat) > 0:
                    if isinstance(login_stat[0], (str, bytes)) and len(login_stat[0]) > 10:  # Basic ticket format check
                        logger.info("Already logged in to Perforce.")
                        return True  # Valid ticket exists
            except P4Exception as stat_err:
                # Determine if this is a fatal login error or just indicating we need to login
                if "Your session has expired" in str(stat_err) or "Single sign-on required" in str(stat_err) or "not logged in" in str(stat_err):
                    logger.info("Login needed: Existing ticket expired or not present")
                else:
                    logger.warning(f"Error checking login status: {stat_err}")
            except Exception as e:
                logger.warning(f"Unexpected error checking login status: {e}")

            # If not logged in, attempt login
            logger.info("Not logged in or ticket expired. Attempting login with password...")
            try:
                # Ensure password is set and is up to date from environment variable
                env_passwd = os.getenv('P4PASSWD')
                if not env_passwd:
                    raise ValueError("P4PASSWD not set for login attempt.")

                # Always refresh password from environment
                self.p4.password = env_passwd
                logger.debug("Password updated from environment")

                # Progressive login strategy for different authentication types
                login_successful = False
                
                # Try standard login first
                try:
                    login_result = self.p4.run_login()
                    login_successful = True
                    logger.info(f"Standard login command succeeded")
                except P4Exception as login_err:
                    if ("LDAPauthenticate" in str(login_err) or 
                        "wrong password" in str(login_err).lower() or
                        "LDAP authentication" in str(login_err)):
                        logger.info("LDAP authentication issue detected. Trying alternate login method...")
                        try:
                            # Attempt plaintext login as fallback for LDAP
                            login_result = self.p4.run_login("-p")
                            login_successful = True
                            logger.info(f"Plaintext login command succeeded")
                        except P4Exception as ldap_err:
                            # Try login with SSO mode if available
                            if "Single sign-on" in str(ldap_err):
                                try:
                                    logger.info("Attempting SSO authentication...")
                                    login_result = self.p4.run_login("-f")
                                    login_successful = True
                                    logger.info("SSO login succeeded")
                                except P4Exception as sso_err:
                                    logger.error(f"All login methods failed. Last error: {sso_err}")
                                    self._log_p4_error(sso_err, "Login SSO Attempt")
                                    raise PermissionError(f"Perforce login failed after multiple attempts: {str(sso_err)}")
                                else:
                                    # Re-raise LDAP error if SSO not indicated
                                    self._log_p4_error(ldap_err, "LDAP Login Attempt")
                                    raise PermissionError(f"LDAP authentication failed: {str(ldap_err)}")
                    else:
                        # Re-raise if it's not an auth-related error
                        self._log_p4_error(login_err, "Standard Login Attempt")
                        raise

                # Verify ticket is valid after successful login
                if login_successful:
                    try:
                        verify_ticket = self.p4.run_login("-s")
                        if verify_ticket and isinstance(verify_ticket, list) and len(verify_ticket) > 0:
                            logger.info("Perforce login successful - ticket acquired.")
                            return True
                        else:
                            logger.warning("Login command succeeded but no valid ticket found. Continuing anyway...")
                            return True  # Continue anyway, as login appeared to succeed
                    except P4Exception as ticket_err:
                        logger.warning(f"Error validating ticket after login: {ticket_err}")
                        # Continue if login succeeded but ticket validation failed
                        return True  
                else:
                    raise PermissionError("No login method succeeded")

            except P4Exception as login_error:
                self._log_p4_error(login_error, "Login Attempt")

                # Create a more user-friendly error with clearer diagnosis
                if "LDAPauthenticate" in str(login_error):
                    error_msg = "Perforce LDAP authentication failed. Please check P4PASSWD environment variable."
                elif "wrong password" in str(login_error).lower():
                    error_msg = "Perforce login failed: Wrong password. Please check P4PASSWD environment variable."
                elif "SSL library must be at least" in str(login_error):
                    error_msg = f"Perforce SSL error: {str(login_error)}. You may need to update SSL libraries or set P4_SSL_DISABLE=1."
                elif "TCP connect to" in str(login_error) and "failed" in str(login_error):
                    error_msg = f"Perforce connection failed: Could not reach server. Check P4PORT setting and network."
                else:
                    error_msg = f"Perforce login failed: {self._sanitize_error(login_error)}"

                raise PermissionError(error_msg) from login_error

        except P4Exception as e:
            self._log_p4_error(e, "Connection")
            raise ConnectionError(f"Failed to connect to Perforce: {self._sanitize_error(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error during Perforce connect: {e}", exc_info=True)
            raise ConnectionError(f"Unexpected error connecting to Perforce: {e}") from e

    def disconnect(self):
        """Clean disconnect."""
        try:
            if self.p4.connected():
                self.p4.disconnect()
                logger.info("Disconnected from Perforce.")
        except P4Exception as e:
            self._log_p4_error(e, "Disconnect")
        except Exception as e:
            logger.error(f"Unexpected error disconnecting: {e}")

    def disconnect_all(self):
        """
        Safely clean up connection resources and shutdown thread pool.
        This implementation is designed to completely prevent segmentation faults
        by avoiding any calls that might crash when cleaning up P4 resources.
        """
        # --- Step 1: Check if we need to do anything at all ---
        # Only attempt cleanup if we're initialized
        if not PerforceHelper._initialized:
            logger.debug("Skipping disconnect_all - helper not initialized")
            return

        # --- Step 2: Wrap the entire process in a try-except ---
        try:
            # --- Step 3: P4 Connection Cleanup ---
            if hasattr(self, 'p4') and self.p4 is not None:
                try:
                    # Check if connected without calling P4 methods that might crash
                    connection_state = False
                    try:
                        # Safely get connection state by checking specific attribute
                        connection_state = hasattr(self.p4, 'connected') and self.p4.connected()
                    except Exception as e:
                        logger.warning(f"Could not check P4 connection state safely: {e}")

                    # Only try disconnect if we're fairly sure it's connected
                    if connection_state:
                        try:
                            logger.debug("P4 appears connected, attempting disconnect")
                            self.p4.disconnect()
                            logger.debug("P4 disconnect successful")
                        except Exception as e:
                            logger.warning(f"Error during P4 disconnect: {e}")
                    else:
                        logger.debug("P4 not connected, skipping disconnect")

                    # Final cleanup - clear client info
                    try:
                        if hasattr(self.p4, 'client'):
                            self.p4.client = None
                    except Exception as e:
                        logger.warning(f"Error clearing P4 client: {e}")

                except Exception as e:
                    logger.warning(f"Error during P4 connection cleanup: {e}")

                # Regardless of errors, set p4 to None to prevent reuse
                try:
                    self.p4 = None
                except Exception as e:
                    logger.warning(f"Error nullifying P4 reference: {e}")

            # --- Step 4: Thread Pool Cleanup (only if exists and active) ---
            if hasattr(self, '_executor') and self._executor is not None:
                try:
                    # Create local reference and clear attribute first
                    local_executor = self._executor
                    self._executor = None  # Clear reference in object first

                    # Now try to shut down the executor
                    logger.debug("Shutting down executor thread pool")
                    local_executor.shutdown(wait=False)  # Non-blocking shutdown
                    logger.debug("Executor shutdown initiated")

                except Exception as e:
                    logger.warning(f"Error during thread pool shutdown: {e}")
                    # Ensure we nullify attribute even if shutdown fails
                    self._executor = None
            else:
                logger.debug("No executor to shutdown or already null")

            # --- Step 5: Final Cleanup of Other Resources ---
            # Clear any other potentially problematic resources
            if hasattr(self, '_changes_cache'):
                self._changes_cache = {}

        except Exception as e:
            logger.error(f"Critical error during Perforce cleanup: {e}", exc_info=True)
            # Attempt last-resort cleanup of critical attributes
            try:
                self.p4 = None
                self._executor = None
            except:
                pass  # Last resort failed, nothing more we can do

        logger.debug("Perforce cleanup complete")

    def __del__(self):
        """
        Safe destructor to ensure proper cleanup when the object is garbage collected.
        This helps prevent segmentation faults by ensuring resources are released.
        """
        try:
            # Avoid any potentially problematic calls that might crash
            # Simply null out critical references

            # First, reset any threading resources
            if hasattr(self, '_executor') and self._executor is not None:
                try:
                    logger.debug("Destructor: Shutting down thread pool")
                    self._executor.shutdown(wait=False)
                except Exception as e:
                    logger.warning(f"Destructor: Non-fatal error in thread pool shutdown: {e}")
                finally:
                    self._executor = None

            # Handle P4 connection
            if hasattr(self, 'p4') and self.p4 is not None:
                try:
                    logger.debug("Destructor: Clearing P4 connection")
                    # Don't call any methods, just null out the reference
                    self.p4 = None
                except Exception:
                    # Completely swallow any exceptions
                    pass

            # Clear other references
            if hasattr(self, '_changes_cache'):
                self._changes_cache = {}

            logger.debug("Destructor: PerforceHelper cleanup complete")
        except Exception:
            # Never let the destructor raise exceptions
            pass

    def test_connection(self) -> bool:
        logger.info("Testing Perforce connection...")
        try:
            self.connect()  # Ensure connected
            info = self.p4.run("info")
            logger.info("Perforce connection successful")
            # logger.info(info) # Avoid logging potentially sensitive server info
            return True
        except (P4Exception, ConnectionError, PermissionError, ValueError) as e:
            logger.error("Perforce connection test failed: %s", self._sanitize_error(e))
            return False
        except Exception as e:
            logger.error(f"Unexpected error during connection test: {e}")
            return False

    def _sanitize_error(self, error: Exception) -> str:
        """Sanitize P4Exception messages for logging/display."""
        try:
            msg = str(error).replace('\n', ' ').strip()
            # Optionally remove specific sensitive details if needed
            # msg = re.sub(...)
            return msg
        except:
            return "Error message could not be formatted"

    def _is_sensitive_path(self, path: str) -> bool:
        """Check if a path is sensitive using SecurityAnalyzer."""
        try:
            # Normalize path if needed
            path_str = str(path)
            return SecurityAnalyzer.is_sensitive_file(path_str)
        except Exception as e:
            logger.warning(f"Error checking sensitive path '{path}': {str(e)}")
            return True  # Err on the side of caution

    def _is_sensitive_file(self, file_path):
        """
        Check if a file is sensitive and should not be included in results.

        Args:
            file_path: The depot path to check

        Returns:
            Boolean indicating if the file is sensitive
        """
        sensitive_patterns = [
            '/passwords/',
            '/secrets/',
            '/credentials/',
            'password.txt',
            'secret.key',
            '.pem',
            '.key',
            '.pfx'
        ]

        for pattern in sensitive_patterns:
            if pattern in file_path.lower():
                return True

        return False

    # ---------------- Changelist / File Retrieval ----------------

    def get_change_details(self, cl_number: str) -> Dict:
        """
        Gets the detailed information for a change list including files, descriptions, etc.
        """
        if not cl_number or not cl_number.strip():
            logger.warning("Empty CL number provided to get_change_details")
            return {}

        # Ensure we have a valid connection
        self._ensure_connected()

        try:
            # Run describe command without the -L flag that was causing issues
            command = f"describe -s {cl_number}"
            result = self.run_p4_command(command)

            if not result:
                logger.warning(f"No result for CL {cl_number} from describe command")
                return {}

            # Parse the describe output
            return self._parse_describe_output(result, cl_number)

        except Exception as e:
            logger.error(f"Error getting details for CL {cl_number}: {e}")
            return {}

    def get_changelist_files(self, changelist: str) -> List[Dict]:
        """Get structured file info (path, action, type, rev) for a changelist."""
        details = self.get_change_details(changelist)
        return details.get("files", [])

    def get_file_content(self, depot_path: str, revision: Optional[str] = None) -> Optional[str]:
        """Retrieve file content for a single revision, returning None on error or for binary."""
        logger.debug(f"Getting content for {depot_path}@{revision or 'HEAD'}")
        try:
            self.connect()

            # Check file type first to avoid printing large binaries
            fstat = self.p4.run('fstat', depot_path)
            if fstat and isinstance(fstat, list) and isinstance(fstat[0], dict):
                file_type = fstat[0].get('headType', '')  # Check head type for binary/text
                if 'binary' in file_type:
                    logger.warning(f"Skipping content retrieval for binary file: {depot_path}")
                    return "[Binary File Content]"

            file_spec = f"{depot_path}@{revision}" if revision else depot_path  # Default to head

            # Use -q for quiet mode (suppresses header line)
            result = self.p4.run_print("-q", file_spec)

            if not result or not isinstance(result, list):
                logger.warning(f"No content returned by 'p4 print' for {file_spec}")
                return None

            # P4Python typically returns list: [metadata_dict, content_str_or_bytes]
            # Or just [content_str_or_bytes] with -q
            # Need to handle both cases and potential bytes
            content_part = result[0]  # Should be content with -q
            if isinstance(content_part, dict):  # Handle case where metadata might still be present
                content_part = result[1] if len(result) > 1 else ''

            if isinstance(content_part, bytes):
                try:
                    # Try UTF-8 first, then Latin-1 as fallback
                    return content_part.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        return content_part.decode('latin1')
                    except UnicodeDecodeError:
                        logger.warning(f"Could not decode content for {depot_path}")
                        return "[Undecodable Content]"
            elif isinstance(content_part, str):
                return content_part
            else:
                logger.warning(f"Unexpected content type from 'p4 print' for {file_spec}: {type(content_part)}")
                return None

        except P4Exception as e:
            # Log specific P4 errors, but return None
            # Common error: "no such file(s)" - log as warning
            if "no such file" in str(e).lower():
                logger.warning(f"File not found in Perforce: {depot_path}@{revision or 'HEAD'}")
            else:
                self._log_p4_error(e, f"print {depot_path}@{revision or 'HEAD'}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting content for {depot_path}: {e}", exc_info=True)
            return None

    def get_changelist_diffs(self, cl_number: str) -> List[Dict]:
        """Enhanced method to retrieve detailed code diffs for a changelist with code reasoning."""
        logger.debug(f"Retrieving diffs for CL {cl_number}")
        files_with_diffs = []
        try:
            self.connect()
            # Get basic file info first
            files_info = self.get_changelist_files(cl_number)
            if not files_info:
                return []

            # Track overall change impact
            change_impact = {
                "high_impact_files": [],
                "total_lines_added": 0,
                "total_lines_removed": 0,
                "languages": set(),
                "file_types": set()
            }

            for file_info in files_info:
                depot_file = file_info.get('depotFile')
                action = file_info.get('action', 'unknown')
                revision = file_info.get('revision')
                file_type = file_info.get('type', '')

                if not depot_file or not revision: continue  # Skip if essential info missing

                # Skip diffs for binary files
                if 'binary' in file_type:
                    file_info['diff'] = "[Binary file - diff not shown]"
                    files_with_diffs.append(file_info)
                    continue

                # Detect file language and type for better reasoning
                file_ext = os.path.splitext(depot_file)[1].lower()
                language = self._detect_language(file_ext)
                change_impact["languages"].add(language)
                change_impact["file_types"].add(file_ext if file_ext else "no_extension")

                diff_text = None
                diff_analysis = {}
                lines_added = 0
                lines_removed = 0

                try:
                    if action in ['edit', 'integrate', 'branch', 'move/add']:  # Actions likely to have diffs
                        current_rev_num = int(revision)
                        prev_rev_num = current_rev_num - 1

                        if prev_rev_num > 0:
                            # Use diff2 for comparing specific revisions, -du for unified diff
                            diff_result = self.p4.run("diff2", "-du", f"{depot_file}#{prev_rev_num}",
                                                      f"{depot_file}@{current_rev_num}")
                            if diff_result and isinstance(diff_result, list):
                                # P4Python diff2 output is often complex; extract readable diff
                                diff_lines = []
                                for line in diff_result:
                                    if isinstance(line, dict) and 'data' in line:
                                        diff_lines.append(line['data'])
                                    elif isinstance(line, str):  # Sometimes plain strings
                                        diff_lines.append(line)
                                diff_text = "\n".join(diff_lines)

                                # Count added/removed lines
                                for line in diff_lines:
                                    if line.startswith('+') and not line.startswith('+++'):
                                        lines_added += 1
                                    elif line.startswith('-') and not line.startswith('---'):
                                        lines_removed += 1

                    elif action == 'add':
                        # For added files, the "diff" is the full content
                        content = self.get_file_content(depot_file, revision)
                        if content:
                            diff_text = "\n".join([f"+ {line}" for line in content.splitlines()])
                            lines_added = len(content.splitlines())
                        else:
                            diff_text = "[Added file - content retrieval failed]"

                    elif action == 'delete':
                        # For deleted files, show previous content marked as deleted
                        prev_rev_num = int(revision)  # Revision before delete
                        content = self.get_file_content(depot_file, str(prev_rev_num))
                        if content:
                            diff_text = "\n".join([f"- {line}" for line in content.splitlines()])
                            lines_removed = len(content.splitlines())
                        else:
                            diff_text = "[Deleted file - could not retrieve previous content]"
                    else:
                        diff_text = f"[{action.capitalize()} operation - no standard diff]"

                    # Update the overall change impact
                    change_impact["total_lines_added"] += lines_added
                    change_impact["total_lines_removed"] += lines_removed

                    # Determine file impact level
                    impact_level = "low"
                    if lines_added + lines_removed > 100:
                        impact_level = "high"
                        change_impact["high_impact_files"].append(depot_file)
                    elif lines_added + lines_removed > 20:
                        impact_level = "medium"

                    # Add analysis metadata
                    diff_analysis = {
                        "language": language,
                        "lines_added": lines_added,
                        "lines_removed": lines_removed,
                        "impact_level": impact_level,
                        "file_type": file_ext if file_ext else "no_extension",
                        "action": action
                    }

                except P4Exception as diff_err:
                    # Log specific diff errors, e.g., file not found at previous rev
                    if "no such file" in str(diff_err).lower():
                        logger.warning(
                            f"Could not generate diff for {depot_file}@{revision} (previous revision likely missing): {diff_err}")
                        diff_text = "[Diff unavailable - previous revision missing]"
                    else:
                        self._log_p4_error(diff_err, f"diff CL {cl_number}, file {depot_file}")
                        diff_text = f"[Error retrieving diff: {self._sanitize_error(diff_err)}]"
                except ValueError as e:
                    logger.warning(f"Could not parse revision for diff {depot_file}@{revision}: {e}")
                    diff_text = "[Diff unavailable - invalid revision number]"
                except Exception as e:
                    logger.error(f"Unexpected error getting diff for {depot_file} in CL {cl_number}: {e}",
                                 exc_info=True)
                    diff_text = f"[Unexpected error retrieving diff]"

                file_info['diff'] = diff_text if diff_text is not None else "[Diff not applicable or failed]"
                file_info['diff_analysis'] = diff_analysis
                files_with_diffs.append(file_info)

            # Add overall change analysis as a metadata field
            for file_info in files_with_diffs:
                file_info['change_impact'] = {
                    "total_files": len(files_with_diffs),
                    "total_lines_added": change_impact["total_lines_added"],
                    "total_lines_removed": change_impact["total_lines_removed"],
                    "languages": list(change_impact["languages"]),
                    "file_types": list(change_impact["file_types"]),
                    "high_impact_files_count": len(change_impact["high_impact_files"])
                }

            return files_with_diffs

        except P4Exception as e:
            self._log_p4_error(e, f"get_changelist_diffs CL {cl_number}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in get_changelist_diffs for CL {cl_number}: {e}", exc_info=True)
            return []

    def _detect_language(self, extension: str) -> str:
        """
        Detect programming language based on file extension.

        Args:
            extension (str): File extension (e.g., '.py', '.js')

        Returns:
            str: Detected programming language or 'plaintext' if unknown
        """
        extension_map = {
            # Web
            '.html': 'html',
            '.htm': 'html',
            '.css': 'css',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.json': 'json',
            '.xml': 'xml',
            '.svg': 'svg',

            # Programming languages
            '.py': 'python',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.pl': 'perl',
            '.sh': 'bash',
            '.bat': 'batch',
            '.ps1': 'powershell',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.m': 'objectivec',
            '.mm': 'objectivec',
            '.scala': 'scala',
            '.dart': 'dart',

            # Data formats
            '.yml': 'yaml',
            '.yaml': 'yaml',
            '.toml': 'toml',
            '.ini': 'ini',
            '.csv': 'csv',
            '.md': 'markdown',
            '.sql': 'sql',

            # Config files
            '.gitignore': 'gitignore',
            '.dockerignore': 'dockerignore',
            'dockerfile': 'dockerfile',
            '.env': 'env',

            # Other
            '.txt': 'plaintext',
            '.log': 'log'
        }

        # Remove leading dot if present
        if extension.startswith('.'):
            clean_ext = extension
        else:
            clean_ext = f".{extension}"

        # Try to match the extension
        return extension_map.get(clean_ext.lower(), 'plaintext')

    # --- Helper Methods for Snippet Extraction ---

    def _get_diff_text(self, depot_file: str, rev1: int, rev2: int | str) -> Optional[str]:
        """DEPRECATED - Use get_diff_from_describe instead."""
        # Keeping old logic commented out for reference
        # try:
        #     context_lines = 3
        #     logger.debug(f"Running: p4 diff -du{context_lines} \"{depot_file}#{rev1}\" \"{depot_file}@{rev2}\"\")
        #     diff_result = self.p4.run("diff", f"-du{context_lines}", f"{depot_file}#{rev1}", f"{depot_file}@{rev2}")
        #     logger.debug(f"Raw diff result for {depot_file}: {diff_result}")
        #     if diff_result:
        #         formatted_diff = self._format_diff_content(diff_result if isinstance(diff_result, list) else [diff_result])
        #         if formatted_diff: return formatted_diff
        #         else: logger.warning(f"diff result for {depot_file} was not empty but could not be formatted: {diff_result}")
        #     else:
        #         logger.debug(f"diff result for {depot_file} was empty or None.")
        #     return None
        # except P4Exception as e:
        #     # ... (error handling) ...
        # except Exception as e:
        #     # ... (error handling) ...
        logger.warning("_get_diff_text is deprecated, use get_diff_from_describe.")
        return None  # Return None as this method is deprecated

    def get_diff_from_describe(self, cl_number: str, depot_file: str) -> Optional[str]:
        """Gets a unified diff for a specific file relative to its state in the given changelist.
        Uses 'p4 fstat' to get the file's revision and action in the CL, then 'p4 diff2' or 'p4 print'.
        """
        try:
            self._ensure_connected() # Ensure connection

            logger.debug(f"Getting specific diff for {depot_file} in CL {cl_number}")

            # Get file's revision and action in this specific changelist
            fstat_output = self.p4.run('fstat', f"{depot_file}@{cl_number}")

            if not fstat_output or not isinstance(fstat_output, list) or not fstat_output[0] or not isinstance(fstat_output[0], dict):
                logger.warning(f"Could not get fstat info for {depot_file}@{cl_number}")
                return None

            file_stat = fstat_output[0]
            # Fallback logic: some servers do not tag 'rev' / 'action' but expose 'headRev' / 'headAction'
            rev_in_cl = file_stat.get('rev') or file_stat.get('headRev')
            action_in_cl = file_stat.get('action') or file_stat.get('headAction')
            head_type = file_stat.get('headType', '') # Check headType as well

            if not rev_in_cl or not action_in_cl:
                logger.warning(f"Missing rev or action in fstat for {depot_file}@{cl_number}: {file_stat}")
                return None
            
            # Skip diff for clearly binary files based on headType if action is not add/delete
            # For add/delete, we might still want to show placeholder or attempt print
            if 'binary' in head_type.lower() and action_in_cl not in ['add', 'delete', 'branch', 'import', 'move/add']: # Be more specific for binary check
                logger.info(f"Skipping diff for binary file {depot_file} (type: {head_type}) in CL {cl_number}")
                return "[Binary file - diff not applicable]"

            diff_text = None

            # For EDIT actions, we need both current and previous content
            if action_in_cl in ['edit', 'integrate']:
                # For better results, get entire file content instead of just a diff
                # First get the current version
                current_rev = int(rev_in_cl)
                prev_rev = current_rev - 1
                
                if prev_rev > 0:
                    # Get current version content
                    current_content = "CURRENT VERSION:\n"
                    try:
                        print_output = self.p4.run('print', '-q', f"{depot_file}#{current_rev}")
                        raw_current = self._format_content(print_output if isinstance(print_output, list) else [print_output])
                        if raw_current is not None:
                            current_content += raw_current
                        else:
                            current_content += "[Failed to retrieve current content]"
                    except Exception as e:
                        logger.warning(f"Error getting current content for {depot_file}#{current_rev}: {e}")
                        current_content += f"[Error retrieving content: {str(e)}]"

                    # Get previous version content
                    prev_content = "\n\nPREVIOUS VERSION:\n"
                    try:
                        print_output = self.p4.run('print', '-q', f"{depot_file}#{prev_rev}")
                        raw_prev = self._format_content(print_output if isinstance(print_output, list) else [print_output])
                        if raw_prev is not None:
                            prev_content += raw_prev
                        else:
                            prev_content += "[Failed to retrieve previous content]"
                    except Exception as e:
                        logger.warning(f"Error getting previous content for {depot_file}#{prev_rev}: {e}")
                        prev_content += f"[Error retrieving content: {str(e)}]"

                    # Combine content (current first, then previous)
                    diff_text = current_content + prev_content
                    
                    # Still try to get a proper diff as a fallback
                    try:
                        logger.debug(f"Getting diff for {depot_file} between revs {prev_rev} and {current_rev}")
                        diff_output = self.p4.run("diff2", "-du", f"{depot_file}#{prev_rev}", f"{depot_file}#{current_rev}")
                        diff_formatted = self._format_diff_content(diff_output)
                        if diff_formatted and len(diff_formatted) > 50:  # Only use if we got meaningful diff
                            diff_text += "\n\nDIFF OUTPUT:\n" + diff_formatted
                    except Exception as diff_e:
                        logger.warning(f"Error getting diff for {depot_file}: {diff_e}")
                else:
                    # First revision edit (no previous version)
                    logger.debug(f"File {depot_file}#{rev_in_cl} is first revision. Getting full content.")
                    action_in_cl = 'add'  # Fallthrough to 'add' logic

            if action_in_cl in ['add', 'branch', 'import', 'move/add']: # Handles true adds and fallthrough from edit of rev 1
                logger.debug(f"Action is '{action_in_cl}'. Printing content of {depot_file}#{rev_in_cl} for CL {cl_number}")
                print_output = self.p4.run('print', '-q', f"{depot_file}#{rev_in_cl}")
                # _format_content expects a list of lines (after metadata)
                # p4.run_print already handles this by returning a list of strings (content lines) or single string
                raw_content = self._format_content(print_output if isinstance(print_output, list) else [print_output])

                if raw_content is not None: # Check for None explicitly
                    if 'binary' in head_type.lower(): # Check again if it's binary based on fstat
                         diff_text = f"[Added binary file: {depot_file}]"
                    else:
                        diff_text = f"ADDED FILE CONTENT:\n{raw_content}"
                else:
                    logger.warning(f"Could not print content for {depot_file}#{rev_in_cl} (action: {action_in_cl})")

            elif action_in_cl == 'delete':
                # For deleted files, try to show the content of the revision *before* it was deleted.
                # The 'rev' from fstat {depot_file}@{cl_number} for a delete action *is* the deleted revision.
                # So, we need to print rev_in_cl itself (which is the last existing version).
                # Perforce philosophy: a deleted rev still "exists" in history.
                logger.debug(f"Action is 'delete'. Printing content of {depot_file}#{rev_in_cl} for CL {cl_number} (content that was deleted)")
                print_output = self.p4.run('print', '-q', f"{depot_file}#{rev_in_cl}")
                raw_content = self._format_content(print_output if isinstance(print_output, list) else [print_output])
                if raw_content is not None:
                    if 'binary' in head_type.lower():
                        diff_text = f"[Deleted binary file: {depot_file}]"
                    else:
                        diff_text = f"DELETED FILE CONTENT:\n{raw_content}"
                else: # This case means the revision just before delete (or the deleted rev itself) had no content / couldn't be printed
                    diff_text = f"[File {depot_file} was deleted in CL {cl_number}. Prior content not available.]"

            if diff_text:
                logger.debug(f"Successfully generated content for {depot_file} in CL {cl_number} (action: {action_in_cl})")
                return diff_text
            else:
                logger.warning(f"No content generated for {depot_file} in CL {cl_number} (action: {action_in_cl})")
                return None # Fallback will be triggered by caller

        except P4Exception as e:
            logger.warning(f"P4Exception in get_diff_from_describe for {depot_file} in CL {cl_number}: {self._sanitize_error(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in get_diff_from_describe for {depot_file} in CL {cl_number}: {e}", exc_info=True)
            return None

    # --- Properly implement as async method ---
    async def get_relevant_code_snippets(self, cl_number: str, max_files: Optional[int] = None) -> List[Dict]:
        """Fetches relevant code diff snippets (+/- lines with context) or full file content on fallback.
           NOTE: This function no longer limits the number of lines per snippet."""
        logger.info(
            f"Fetching diff snippets for CL {cl_number} (max_files={max_files if max_files is not None else 'All'}, max_lines=All) using describe -duf with fallback")
        snippets = []
        files_needing_fallback = []  # Store (index, depot_file, revision, action) tuples

        try:
            # Get file list using the efficient method - run in executor since it's synchronous
            loop = asyncio.get_running_loop()
            files_info = await loop.run_in_executor(
                self._executor,
                self.get_changelist_files_efficient,
                cl_number
            )

            if not files_info:
                logger.warning(f"Could not get file list for CL {cl_number} to fetch snippets.")
                return []

            files_processed = 0

            for file_info in files_info:
                # Only break if max_files is set and reached
                if max_files is not None and files_processed >= max_files: break

                depot_file = file_info.get('depotFile')
                action = file_info.get('action', 'unknown')
                revision = file_info.get('rev')
                file_type = file_info.get('type', '')

                if not depot_file or 'binary' in file_type or action == 'delete':
                    continue  # Skip binary, deleted, or invalid entries

                # Initialize placeholder for snippet data
                snippet_entry = {
                    'file': depot_file,
                    'action': action,
                    'snippet': '[Snippet retrieval failed]',  # Default error
                    'language': self._detect_language(os.path.splitext(depot_file)[1].lower())
                }

                try:
                    if action == 'add':
                        # For added files, always try to get full content
                        # Defer fetching content to the fallback phase to avoid nested calls
                        files_needing_fallback.append((len(snippets), depot_file, cl_number, action))
                        snippet_entry['snippet'] = '[Fallback required for added file]'  # Placeholder
                    elif action in ['edit', 'integrate', 'branch', 'move/add']:
                        # Run get_diff_from_describe in the executor because it uses p4 commands
                        diff_content = await loop.run_in_executor(
                            self._executor,
                            self.get_diff_from_describe,
                            cl_number,
                            depot_file
                        )

                        if diff_content:
                            # Process diff if found - this is pure Python so no need for executor
                            snippet_entry['snippet'] = self._extract_diff_lines(diff_content)
                        else:
                            # Diff failed, mark for fallback content retrieval
                            logger.warning(f"Diff retrieval failed for {depot_file}@{revision}, marking for fallback.")
                            files_needing_fallback.append((len(snippets), depot_file, cl_number, action))
                            snippet_entry['snippet'] = '[Fallback required]'  # Placeholder
                    else:
                        snippet_entry['snippet'] = f'[Action \'{action}\' not handled for snippets]'

                except Exception as e:
                    logger.error(f"Unexpected error preparing snippet for {depot_file}@{revision}: {e}", exc_info=True)
                    snippet_entry['snippet'] = '[Unexpected error during snippet preparation]'

                snippets.append(snippet_entry)  # Add the entry (potentially with placeholder snippet)
                files_processed += 1

            # --- Fallback Phase: Process files needing content retrieval outside the loop ---
            if files_needing_fallback:
                logger.info(
                    f"Attempting fallback content retrieval for {len(files_needing_fallback)} files in CL {cl_number}")
                # Create a list of coroutines for fetching content
                fallback_tasks = []
                for index, f_path, rev_cl, f_action in files_needing_fallback:
                    # Use a separate helper async function to avoid blocking
                    fallback_tasks.append(self._fetch_fallback_content(index, f_path, rev_cl, f_action))

                # Run fallback fetches concurrently
                fallback_results = await asyncio.gather(*fallback_tasks, return_exceptions=True)

                # Update snippets list with results from fallback
                for result in fallback_results:
                    if isinstance(result, dict) and 'index' in result and 'snippet' in result:
                        snippet_index = result['index']
                        if 0 <= snippet_index < len(snippets):
                            snippets[snippet_index]['snippet'] = result['snippet']  # Update placeholder
                        else:
                            logger.error(f"Fallback returned invalid index {snippet_index} for CL {cl_number}")
                    elif isinstance(result, Exception):
                        logger.error(f"Error during fallback content fetch for CL {cl_number}: {result}")
                    # else: None result or malformed dict indicates failure, keep original placeholder

        except Exception as e:
            logger.error(f"Error processing files for snippets in CL {cl_number}: {e}", exc_info=True)
        return snippets

    async def _fetch_fallback_content(self, index: int, depot_file: str, revision_or_cl: str, action: str) -> Optional[
        Dict]:
        """Helper coroutine to fetch fallback content using _get_file_content_via_session."""
        try:
            # Run the synchronous _get_file_content_via_session in the executor
            loop = asyncio.get_running_loop()
            content = await loop.run_in_executor(
                self._executor,
                self._get_file_content_via_session,
                depot_file,
                revision_or_cl
            )

            snippet = "[Could not retrieve fallback content]"
            if content:
                if action == 'add':
                    formatted_added_content = '\n'.join([f"+ {line}" for line in content.splitlines()])
                    snippet = self._extract_diff_lines(formatted_added_content)
                else:  # Other actions (edit, integrate etc.) where diff failed
                    # Format as full file content fallback
                    snippet = "[FALLBACK - FULL FILE CONTENT]\n" + '\n'.join(
                        [f"  {line}" for line in content.splitlines()])  # Indent slightly
                if not snippet: snippet = '[Empty fallback content]'
            return {'index': index, 'snippet': snippet}
        except Exception as e:
            logger.error(f"Error in _fetch_fallback_content for {depot_file}@{revision_or_cl}: {e}", exc_info=True)
            # Return None or error indicator if needed, handled by gather exception processing
            return None

    def _extract_diff_lines(self, diff_text: str) -> str:
        """Extracts diff lines (+/-) along with context lines from unified diff text.
           NOTE: This function no longer truncates based on max_lines."""
        if not diff_text:
            return "[Empty diff text received]"

        extracted_lines = []
        
        # Escape any accidental escaped newlines in the diff_text
        if '\\n' in diff_text:
            diff_text = diff_text.replace('\\n', '\n')
        
        # Actual processing of diff lines
        for line in diff_text.splitlines():
            # Skip standard header lines, but keep other headers for context
            if (line.startswith('---') and '---' in line[:4]) or (line.startswith('+++') and '+++' in line[:4]) or line.startswith('@@'):
                continue
                
            # Keep ALL context lines and changes, not just those with specific prefixes
            # This ensures we capture meaningful context even when prefixes are missing
            extracted_lines.append(line)

        if not extracted_lines:
            # If no lines were extracted, use the original text with minimal cleanup
            # This ensures something is always displayed rather than "[No relevant diff/context lines found]"
            for line in diff_text.splitlines():
                if not (line.startswith('Index:') or line.startswith('====')):
                    extracted_lines.append(line)

        if not extracted_lines:
            return "[Diff available but no displayable content found]"

        return '\n'.join(extracted_lines)

    def _get_file_content_via_session(self, depot_file: str, revision_or_cl: str) -> Optional[str]:
        """Gets file content using a separate P4 session to avoid nested calls."""
        new_p4 = None
        try:
            logger.debug(f"Helper: retrieving file content via new P4 session for {depot_file}@{revision_or_cl}")
            new_p4 = P4()
            # Mirror connection settings
            new_p4.user = self.p4.user
            new_p4.client = self.p4.client
            new_p4.port = self.p4.port
            new_p4.password = self.p4.password
            new_p4.connect()
            raw_output = new_p4.run('print', '-q', f"{depot_file}@{revision_or_cl}")
            content = self._format_content(raw_output if isinstance(raw_output, list) else [raw_output])
            return content
        except P4Exception as p4e:
            logger.warning(f"Helper P4Exception for {depot_file}@{revision_or_cl}: {p4e}")
            return None
        except Exception as e:
            logger.warning(f"Helper extraction error for {depot_file}@{revision_or_cl}: {e}")
            return None
        finally:
            if new_p4:
                try:
                    new_p4.disconnect()
                except:
                    pass

    # --- END Helper Methods ---

    def get_file_details_with_snippets(self, cl_number, file_path, max_lines=50):
        """
        Get detailed information about a file in a changelist, including code snippets.

        Args:
            cl_number: The changelist number
            file_path: Path to the file
            max_lines: Maximum number of lines to include in snippet (default: 50)

        Returns:
            Dictionary with file details including:
            - path: File path
            - extension: File extension
            - language: Detected programming language
            - type: File type from Perforce
            - is_binary: Whether the file is binary
            - snippet: Code snippet or message if binary
            - error: Error message if there was a problem
        """
        result = {
            "path": file_path,
            "extension": os.path.splitext(file_path)[1],
            "language": "unknown",
            "type": "unknown",
            "is_binary": False,
            "snippet": "",
            "error": None
        }

        # Detect language based on file extension
        result["language"] = self._detect_language(result["extension"])

        try:
            # Connect to Perforce if not already connected
            if not hasattr(self, 'p4') or not self.p4.connected():
                self._connect_to_perforce()

            # Get file type
            try:
                file_info = self.p4.run('fstat', f"{file_path}@{cl_number}")
                if file_info and len(file_info) > 0:
                    result["type"] = file_info[0].get('headType', 'unknown')
                    result["is_binary"] = 'binary' in result["type"].lower()
                    logger.debug(f"File type for {file_path}@{cl_number}: {result['type']}")
                else:
                    logger.warning(f"No file info returned for {file_path}@{cl_number}")
            except P4Exception as e:
                self._log_p4_error(e, f"getting file type for {file_path}@{cl_number}")
                result["error"] = f"Error getting file type: {str(e)}"
                return result

            # If binary, don't attempt to get snippet
            if result["is_binary"]:
                result["snippet"] = f"[Binary file of type: {result['type']}]"
                return result

            # Try to get file content using describe first (more efficient)
            try:
                logger.debug(f"Getting file content for {file_path}@{cl_number} using describe")
                desc = self.p4.run('describe', '-s', str(cl_number))
                snippet_found = False

                if desc and len(desc) > 0 and 'depotFile' in desc[0]:
                    # Find the matching file in the changelist
                    for i, depot_file in enumerate(desc[0].get('depotFile', [])):
                        if depot_file == file_path:
                            # Extract the diff content for this file
                            if 'data' in desc[0] and i < len(desc[0].get('data', [])):
                                content = desc[0]['data'][i]
                                # Get a subset of lines if content is large
                                lines = content.split('\n')
                                if len(lines) > max_lines:
                                    # Take first and last lines to provide context
                                    first_part = lines[:max_lines // 2]
                                    last_part = lines[-(max_lines // 2):]
                                    result["snippet"] = '\n'.join(first_part + ["[...]"] + last_part)
                                else:
                                    result["snippet"] = content
                            snippet_found = True
                            break

                # If no snippet found via describe, fall back to direct file retrieval
                if not snippet_found:
                    logger.debug(f"Falling back to print for {file_path}@{cl_number}")
                    file_content = self.p4.run('print', f"{file_path}@{cl_number}")

                    if file_content and len(file_content) > 1:  # First item is usually metadata
                        # Skip the first item (metadata) and join the rest
                        content = ''.join(file_content[1:])

                        # Get a subset of lines if content is large
                        lines = content.split('\n')
                        if len(lines) > max_lines:
                            # Take first and last lines to provide context
                            first_part = lines[:max_lines // 2]
                            last_part = lines[-(max_lines // 2):]
                            result["snippet"] = '\n'.join(first_part + ["[...]"] + last_part)
                        else:
                            result["snippet"] = content
                    else:
                        result["error"] = "No content found in the file"

            except P4Exception as e:
                self._log_p4_error(e, f"getting content for {file_path}@{cl_number}")
                result["error"] = f"Error getting file content: {str(e)}"

            return result

        except Exception as e:
            logger.error(f"Unexpected error getting file details: {str(e)}")
            result["error"] = f"Unexpected error: {str(e)}"
            return result

    def generate_code_summary(self, files_with_snippets):
        """
        Generate a summary of code changes based on files with snippets.

        Args:
            files_with_snippets: List of file details with snippets

        Returns:
            Dictionary with summary information including:
            - file_count: Total number of files
            - languages: List of programming languages detected
            - extensions: List of file extensions
            - binary_count: Number of binary files
            - code_count: Number of code files
            - assessment: Overall assessment of code changes
        """
        summary = {
            "file_count": len(files_with_snippets),
            "languages": set(),
            "extensions": set(),
            "binary_count": 0,
            "code_count": 0,
            "assessment": ""
        }

        for file in files_with_snippets:
            # Skip files with errors
            if file.get("error"):
                continue

            # Add extension and language to sets
            if file.get("extension"):
                summary["extensions"].add(file["extension"])

            if file.get("language") and file["language"] != "unknown":
                summary["languages"].add(file["language"])

            # Count binary and code files
            if file.get("is_binary"):
                summary["binary_count"] += 1
            else:
                summary["code_count"] += 1

        # Convert sets to lists for JSON serialization
        summary["languages"] = list(summary["languages"])
        summary["extensions"] = list(summary["extensions"])

        # Generate assessment
        if summary["code_count"] > 0:
            summary[
                "assessment"] = f"Found {summary['code_count']} code files across {len(summary['languages'])} languages. "
            if summary["binary_count"] > 0:
                summary["assessment"] += f"Also detected {summary['binary_count']} binary files."
        elif summary["binary_count"] > 0:
            summary["assessment"] = f"Only found {summary['binary_count']} binary files. No code changes detected."
        else:
            summary["assessment"] = "No code changes detected."

        return summary

    def _format_content(self, content_lines: List) -> str:
        """
        Format the content lines into a single string.

        Args:
            content_lines: List of content lines from p4 print

        Returns:
            Formatted content string
        """
        if not content_lines:
            return ""

        formatted_lines = []

        for line in content_lines:
            if isinstance(line, dict):
                # Handle case where content might contain metadata
                logger.debug(f"Skipping dict line in content: {line}")
                continue
            elif isinstance(line, str):
                formatted_lines.append(line)
            else:
                # Decode if bytes, otherwise convert to string
                decoded_line = self._decode_binary_content(line)  # Use existing decoder
                # Downgrade to DEBUG to avoid log noise since the line is handled gracefully
                logger.debug(
                    "Unexpected content line type was %s, decoded/converted (first 100 chars): %s...",
                    type(line), decoded_line[:100]
                )
                formatted_lines.append(decoded_line)

        return "\n".join(formatted_lines)

    def process_changelist(self, changelist_data: dict, cl_number: str = None) -> dict:
        """
        Process a changelist description to extract details and build a structured result.

        Args:
            changelist_data: The changelist data from p4.run_describe
            cl_number: Optional changelist number if not present in changelist_data

        Returns:
            A dictionary with processed changelist information
        """
        try:
            # Extract the changelist number
            cl_num = cl_number or changelist_data.get("change")
            if not cl_num:
                logger.warning("No changelist number found in data")
                return None

            # Get basic metadata
            user = changelist_data.get("user", "unknown")
            date_str = changelist_data.get("time", "0")
            try:
                # Convert Perforce timestamp to human-readable format
                date = datetime.fromtimestamp(int(date_str))
                formatted_date = date.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                formatted_date = "Unknown date"

            # Get description
            description = changelist_data.get("desc", "")

            # Process files
            files = []
            depot_files = changelist_data.get("depotFile", [])
            actions = changelist_data.get("action", [])
            types = changelist_data.get("type", [])
            revs = changelist_data.get("rev", [])

            # Ensure all are lists for safe indexing
            if not isinstance(depot_files, list): depot_files = [depot_files]
            if not isinstance(actions, list): actions = [actions]
            if not isinstance(types, list): types = [types]
            if not isinstance(revs, list): revs = [revs]

            for i, file_path in enumerate(depot_files):
                # Skip sensitive files
                if self._is_sensitive_path(file_path):
                    files.append({
                        "path": "[SENSITIVE FILE]",
                        "action": actions[i] if i < len(actions) else "unknown",
                        "type": "sensitive",
                        "revision": "hidden"
                    })
                    continue

                # Get file details
                file_type = types[i] if i < len(types) else "unknown"
                action = actions[i] if i < len(actions) else "unknown"
                revision = revs[i] if i < len(revs) else "unknown"

                # Determine if binary
                is_binary = "binary" in file_type.lower()

                # Add file to the list
                files.append({
                    "path": file_path,
                    "action": action,
                    "type": file_type,
                    "revision": revision,
                    "is_binary": is_binary
                })

            # Build the result object
            result = {
                "changelist": cl_num,
                "user": user,
                "date": formatted_date,
                "description": description,
                "files": files,
                "file_count": len(files)
            }

            # Add Swarm URL if available
            if self.p4_swarm_url and cl_num and str(cl_num).isdigit():
                result["url"] = f"{self.p4_swarm_url}/changes/{cl_num}"

            return result

        except Exception as e:
            logger.error(f"Error processing changelist {cl_number}: {str(e)}")
            return None

    def _format_diff_content(self, diff_result: List) -> str:
        """
        Format the diff content into a readable string.

        Args:
            diff_result: The result of p4 diff command

        Returns:
            Formatted diff string
        """
        if not diff_result:
            return ""

        formatted_lines = []
        file_pair = []  # To store file pairs for generating a diff header

        for item in diff_result:
            if isinstance(item, dict):
                # Handle dictionary items which sometimes appear in diff results
                # Track file information to generate proper diff headers
                if 'depotFile' in item and 'rev' in item:
                    file_info = f"{item['depotFile']}#{item['rev']}"
                    file_pair.append(file_info)

                    # If we have both files in the pair, generate a diff header
                    if len(file_pair) == 2:
                        formatted_lines.append(f"--- {file_pair[0]}")
                        formatted_lines.append(f"+++ {file_pair[1]}")
                        file_pair = []  # Reset for next pair
                else:
                    # For other dict formats, just add a summary line
                    formatted_lines.append(
                        f"File: {item.get('depotFile', 'unknown')} (rev: {item.get('rev', 'unknown')})")
            elif isinstance(item, str):
                # Regular string lines in the diff output
                formatted_lines.append(item)
            else:
                # Convert other types to string but log warning
                logger.warning(f"Unexpected item type in diff result, converting to string: {type(item)} - {item}")
                formatted_lines.append(str(item))

        # If we have a single file info that wasn't paired, add it as a header
        if len(file_pair) == 1:
            formatted_lines.append(f"--- {file_pair[0]}")

        return "\n".join(formatted_lines)

    # ---------------- MTV SEARCH (DEFAULT 180 DAYS) ----------------

    async def _search_mtv_changes_core(self, mtv_number: str, days_back: int = 180) -> List[dict]:
        """Core MTV changes search logic with default 180-day lookback."""
        logger.info(f"Searching MTV {mtv_number} changes within {days_back} days in {self.depot_path}")
        results = []
        target_paths = os.getenv("P4_TARGET_PATHS", f"{self.depot_path}/...")  # Default to whole depot if not specified
        search_paths_list = [p.strip() for p in target_paths.split(',') if p.strip()]
        if not search_paths_list:
            logger.error("No valid P4_TARGET_PATHS found for searching.")
            return []

        try:
            self.connect()  # Ensure connected

            # Construct date range for query
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            # Perforce date format: @YYYY/MM/DD:HH:MM:SS or just @YYYY/MM/DD
            date_range = f"@{start_date:%Y/%m/%d},@{end_date:%Y/%m/%d}"

            # Run 'p4 changes' command
            # Use -L for long description, -s submitted status
            cmd_args = ["changes", "-L", "-s", "submitted", "-m", str(self.max_changes)]
            for path in search_paths_list:
                cmd_args.append(f"{path}{date_range}")

            logger.debug(f"Running P4 command: p4 {' '.join(cmd_args)}")
            changes = self.p4.run(cmd_args)

            # Filter results based on MTV number in description
            if changes and isinstance(changes, list):
                mtv_pattern_re = re.compile(r'\b' + re.escape(mtv_number) + r'\b', re.IGNORECASE)
                for change_dict in changes:
                    if isinstance(change_dict, dict):
                        desc = change_dict.get("desc", "")
                        if mtv_pattern_re.search(desc):
                            results.append({
                                'change': change_dict.get('change'),
                                'desc': desc,
                                'status': change_dict.get('status', ''),
                                'time': int(change_dict.get('time', 0)),  # Store as int
                                'user': change_dict.get('user', '')
                            })
            logger.info(
                f"Found {len(results)} changes containing '{mtv_number}' in description within {days_back} days.")
            return results

        except P4Exception as e:
            self._log_p4_error(e, f"MTV search for {mtv_number}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error searching MTV changes for {mtv_number}: {e}", exc_info=True)
            return []

    # --- Ensure this remains async ---
    async def get_mtv_changes(self, mtv_number: str, days_back: int = 180) -> List[Dict]:
        """Async method to fetch MTV changes over N days."""
        logger.info(f"Starting get_mtv_changes for MTV: {mtv_number}, days_back: {days_back}")
        try:
            # Use asyncio.to_thread to run the synchronous P4Python calls in the core function
            loop = asyncio.get_running_loop()
            # Wrap the call to the core logic
            raw_changes = await loop.run_in_executor(
                self._executor,
                lambda: self._search_mtv_changes_sync(mtv_number, days_back)  # Call a sync wrapper if needed
            )

            if not isinstance(raw_changes, list):
                logger.error(f"MTV changes result is not a list, but a {type(raw_changes)}")
                return []
            return raw_changes
        except Exception as e:
            logger.error(f"Error fetching MTV changes asynchronously: {str(e)}", exc_info=True)
            return []

    # --- Add a synchronous wrapper for the core logic if needed by executor ---
    def _search_mtv_changes_sync(self, mtv_number: str, days_back: int = 180) -> List[dict]:
        """Synchronous wrapper for the core MTV search logic."""
        # This method directly calls the P4Python library which is synchronous
        logger.debug(f"Executing synchronous MTV search for {mtv_number}")
        changelist_info = []  # Store basic CL info first
        try:
            self.connect()  # Ensure connected
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            date_range = f"@{start_date:%Y/%m/%d},@{end_date:%Y/%m/%d}"
            target_paths = os.getenv("P4_TARGET_PATHS", f"{self.depot_path}/...")
            search_paths_list = [p.strip() for p in target_paths.split(',') if p.strip()]

            cmd_args = ["changes", "-L", "-s", "submitted", "-m", str(self.max_changes)]
            for path in search_paths_list:
                cmd_args.append(f"{path}{date_range}")

            changes = self.p4.run(cmd_args)

            if changes and isinstance(changes, list):
                mtv_pattern_re = re.compile(r'\b' + re.escape(mtv_number) + r'\b', re.IGNORECASE)
                for change_dict in changes:
                    if isinstance(change_dict, dict):
                        desc = change_dict.get("desc", "")
                        if mtv_pattern_re.search(desc):
                            # Store basic info, fetch snippets later
                            changelist_info.append({
                                'change': change_dict.get('change'),
                                'desc': desc,
                                'status': change_dict.get('status', ''),
                                'time': int(change_dict.get('time', 0)),
                                'user': change_dict.get('user', '')
                            })

            logger.debug(
                f"Synchronous search found {len(changelist_info)} potential changes for {mtv_number}. Now fetching snippets sequentially.")

            # --- Fetch snippets sequentially AFTER p4 changes ---
            results_with_snippets = []
            for cl_info in changelist_info:
                cl_num = cl_info.get('change')
                if cl_num:
                    try:
                        # Use synchronous method instead of async
                        snippets = self.get_snippets_sync(cl_num)
                        if snippets:
                            cl_info['code_snippets'] = snippets  # Add snippets key
                    except Exception as snippet_err:
                        logger.warning(f"Error fetching snippets for CL {cl_num} during sync search: {snippet_err}")
                results_with_snippets.append(cl_info)  # Add processed CL info to final results
            # --- END Snippet Fetching ---

            return results_with_snippets  # Return list with snippets added

        except P4Exception as e:
            self._log_p4_error(e, f"Sync MTV search {mtv_number}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in sync MTV search {mtv_number}: {e}", exc_info=True)
            return []

    # ---------------- Searching Code References ----------------

    def search_changelists_by_keyword(self, keyword: str, status: Optional[str] = None,
                                      force_path: Optional[str] = None, max_results_override: Optional[int] = None) -> \
    List[Dict[str, Any]]:
        """
        Search for changelists that match the given keyword.
        This is a more sophisticated version of get_mtv_changes that can search by any keyword.

        Args:
            keyword (str): Keyword to search for
            status (Optional[str]): P4 status filter (e.g., "submitted", "pending")
            force_path (Optional[str]): If provided, overrides the default search path with this one
            max_results_override (Optional[int]): Override for max results (default from env or 1000)

        Returns:
            List[Dict[str, Any]]: List of changelists that match the keyword
        """
        if not keyword:
            logger.warning("No keyword provided to search_changelists_by_keyword.")
            return []

        keyword = str(keyword).strip()
        logger.debug(f"Searching CL descriptions for keyword: '{keyword}'")

        try:
            self.connect_if_needed()

            # If keyword is MTV#### pattern, add other potential format variations
            # This helps with different naming conventions used in changelist descriptions
            variations = [keyword]
            if keyword.upper().startswith('MTV') and re.match(r'MTV\d{4,}', keyword.upper()):
                base_pattern = keyword.upper()
                mtv_digits = base_pattern[3:]
                patterns_to_try = [
                    base_pattern,  # MTV####
                    f"MTV{mtv_digits}",  # MTV####
                    f"mtv{mtv_digits}",  # mtv####
                    f"M{mtv_digits}",  # M####
                    f"m{mtv_digits}"  # m####
                ]
                variations = list(set(patterns_to_try))  # Deduplicate
                logger.debug(f"Searching CL descriptions for variations: {variations}")

            # Configure the command
            cmd_args = ['changes', '-l']  # -l for long format (include descriptions)

            # Determine limit: use override or env var P4_MAX_CHANGES (default 1000)
            try:
                limit = max_results_override if max_results_override is not None else int(
                    os.getenv('P4_MAX_CHANGES', '1000'))
            except Exception:
                limit = int(os.getenv('P4_MAX_CHANGES', '1000'))
            if limit and limit > 0:
                cmd_args.extend(["-m", str(limit)])

            # Set the path to search
            search_path = None

            # Use force_path if provided
            if force_path:
                logger.info(f"Forcing search path to: {force_path}")
                search_path = force_path
            # Otherwise, use global DEPOT_PATH if available, or try P4_TARGET_PATHS
            else:
                depot_path = os.getenv('DEPOT_PATH')
                if depot_path:
                    search_path = depot_path
                else:
                    target_paths_str = os.getenv('P4_TARGET_PATHS')
                    if target_paths_str:
                        # Parse comma-separated target paths
                        # For now, just use the first one
                        target_paths = [p.strip() for p in target_paths_str.split(',')]
                        if target_paths:
                            search_path = target_paths[0]

            # Fallback to the client's view
            if not search_path:
                search_path = "//VFIT/..."  # Broad default if nothing else specified

            # Force trailing / or ... depending on what we have already
            if search_path:
                if search_path.endswith('...'):
                    pass  # Already has ... ending
                elif search_path.endswith('/'):
                    search_path += '...'
                else:
                    search_path += '/...'

            cmd_args.append(search_path)

            # Execute the command
            logger.debug(f"Running P4 command: p4 {' '.join(cmd_args)}")
            results = self.p4.run(*cmd_args)

            if not results:
                results = []

            # Search for the keyword in descriptions
            matched_cls = []
            for cl in results:
                desc = cl.get('desc', '').lower()

                # Look for variation matches
                match_found = False
                for variation in variations:
                    if variation.lower() in desc:
                        match_found = True
                        break

                if match_found:
                    matched_cls.append(cl)

            logger.info(f"Keyword search found {len(matched_cls)} potential CLs containing any of '{variations}'.")

            # If no results exactly match the keyword/variations, do a fallback scan
            # This does a more general scan though it's slower
            if not matched_cls:
                logger.info(
                    f"Fallback scan: checking full describe output for keyword '{keyword}' in all returned changes.")

                # For efficiency, handle different potential search methods
                cl_candidates = []

                # 1) If our results set is small (less than 50), just try them all
                if len(results) <= 50:
                    cl_candidates = results
                # 2) Otherwise, apply some heuristics to check most likely matches first
                else:
                    # First check CLs with short descriptions that returned in the changes output
                    for cl in results:
                        # If desc already in the change, check it
                        if 'desc' in cl and len(cl['desc']) < 1000:  # Check shorter descriptions first
                            desc_lower = cl['desc'].lower()
                            if any(var.lower() in desc_lower for var in variations):
                                cl_candidates.append(cl)

                    # Then, if we found fewer than 10 candidates, analyze commit messages
                    # of recent CLs (expensive fallback)
                    if len(cl_candidates) < 10 and len(results) > 0:
                        # Take a subset of CLs to check in detail - we can't check all for performance reasons
                        # Prefer more recent CLs in case the feature is newer
                        cl_subset = results[:min(100, len(results))]

                        # First using search pattern
                        for cl in cl_subset:
                            # If not already matched and we have a change number
                            if cl not in cl_candidates and 'change' in cl:
                                try:
                                    change_num = cl['change']
                                    # Get the full details for this CL (expensive call)
                                    describe_result = self.p4.run('describe', '-s', change_num)
                                    if describe_result:
                                        describe_data = describe_result[0]
                                        if 'desc' in describe_data:
                                            desc_lower = describe_data['desc'].lower()
                                            if any(var.lower() in desc_lower for var in variations):
                                                # Add the describe data with fuller info
                                                cl_candidates.append(describe_data)
                                except Exception as describe_err:
                                    logger.warning(f"Error describing CL {cl.get('change', 'unknown')}: {describe_err}")

                # Use candidates if we found any, otherwise use original results
                if cl_candidates:
                    matched_cls = cl_candidates
                elif not matched_cls and keyword.strip().lower() in ['jira', 'confluence', 'vfit']:
                    # Special case for common keywords that appear in almost every CL
                    logger.info(f"Not using fallback for very common keyword: '{keyword}'")
                    matched_cls = []

            logger.info(f"After fallback, total matched CLs: {len(matched_cls)}. Now fetching snippets.")

            # Now for each matched CL, get code snippets
            cls_with_snippets = []
            for cl in matched_cls:
                if 'change' in cl:
                    try:
                        # Extract code snippets for this changelist
                        cl_with_snippets = self.extract_code_snippets_for_cl(cl)
                        cls_with_snippets.append(cl_with_snippets)
                    except Exception as snippet_err:
                        logger.warning(f"Error extracting snippets for CL {cl.get('change', 'unknown')}: {snippet_err}")
                        # Still include the CL even without snippets
                        cls_with_snippets.append(cl)

            return cls_with_snippets

        except Exception as e:
            logger.error(f"Error searching changelists: {e}")
            if hasattr(e, 'errors') and e.errors:
                for err in e.errors:
                    logger.error(f"P4 error detail: {err}")

            # Log the error to a file for debugging
            self._log_error_to_file("search_changelists_by_keyword", keyword, e)

            return []

    def extract_code_snippets_for_cl(self, cl: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract code snippets for a changelist.

        Args:
            cl (Dict[str, Any]): The changelist dictionary from P4 output

        Returns:
            Dict[str, Any]: The changelist with snippets added
        """
        try:
            # Start with a copy of the CL to avoid modifying the original
            result = cl.copy()

            # Skip if no change number
            if "change" not in cl:
                logger.warning(f"No change number found in CL: {cl}")
                return result

            # Initialize empty snippets list if not exist
            if "code_snippets" not in result:
                result["code_snippets"] = []

            # If files are not yet fetched, get them
            if "files" not in cl or not cl["files"]:
                try:
                    describe_result = self.p4.run("describe", "-s", cl["change"])
                    if describe_result and len(describe_result) > 0:
                        describe_data = describe_result[0]

                        # Handle the case where "depotFile" is a list directly in the describe result
                        if "depotFile" in describe_data and isinstance(describe_data["depotFile"], list):
                            files = []
                            for i, depot_file in enumerate(describe_data["depotFile"]):
                                action = describe_data.get("action", [])[i] if i < len(
                                    describe_data.get("action", [])) else "unknown"
                                file_info = {"depotFile": depot_file, "action": action}
                                files.append(file_info)
                            result["files"] = files
                            logger.debug(f"Extracted {len(files)} files from describe for CL {cl['change']}")
                        elif "files" in describe_data:
                            result["files"] = describe_data["files"]
                            logger.debug(
                                f"Extracted {len(describe_data['files'])} files from describe for CL {cl['change']}")
                        else:
                            logger.debug(f"No files found in describe for CL {cl['change']}")
                            return result
                    else:
                        logger.warning(f"Describe command failed for CL {cl['change']}")
                        return result
                except Exception as e:
                    logger.error(f"Error getting files for CL {cl['change']}: {e}")
                    return result

            # Skip if no files or files isn't a list
            if "files" not in result or not result["files"]:
                logger.warning(f"No files found in CL {cl.get('change', 'unknown')} after attempts to retrieve them")
                return result

            logger.debug(f"Processing {len(result['files'])} files for CL {cl['change']}")

            file_count = 0
            for file_info in result["files"]:
                # Handle different file_info formats
                if isinstance(file_info, dict) and "depotFile" in file_info:
                    depot_file = file_info["depotFile"]
                    action = file_info.get("action", "unknown")
                elif isinstance(file_info, str):
                    depot_file = file_info
                    action = "unknown"
                else:
                    logger.warning(f"Invalid file_info format: {file_info}")
                    continue

                # Skip sensitive files
                if self._is_sensitive_file(depot_file):
                    continue

                # Create snippet entry
                snippet_entry = {
                    "file": depot_file,
                    "action": action,
                    "snippet": "[Snippet retrieval failed]",
                    "language": self._detect_language(os.path.splitext(depot_file)[1].lower())
                }

                # Handle based on action type
                try:
                    if action in ["add", "branch", "move/add"]:
                        try:
                            content = self.p4.run("print", f"{depot_file}@{cl['change']}")
                            if content and len(content) > 0:
                                file_content = ""
                                for i in range(1, len(content)):
                                    if isinstance(content[i], str):
                                        file_content += content[i]
                                    elif isinstance(content[i], bytes):
                                        try:
                                            file_content += content[i].decode("utf-8")
                                        except UnicodeDecodeError:
                                            logger.warning(f"Could not decode binary content for {depot_file}")
                                            file_content = "[Binary content]"
                                            break
                                if len(file_content) > 2000:
                                    file_content = file_content[:2000] + "... [truncated]"
                                snippet_entry["snippet"] = file_content
                        except Exception as print_err:
                            logger.warning(f"Error getting content for added file {depot_file}: {print_err}")
                            snippet_entry["snippet"] = "[Error retrieving content]"

                    elif action in ["edit", "integrate"]:
                        try:
                            diff_args = ["diff2"]
                            prev_rev = f"{depot_file}@{int(cl['change']) - 1}"
                            curr_rev = f"{depot_file}@{cl['change']}"
                            diff_args.extend([prev_rev, curr_rev])

                            diff_output = self.p4.run(*diff_args)
                            if diff_output:
                                diff_content = ""
                                for line in diff_output:
                                    if isinstance(line, str):
                                        diff_content += line + "\n"
                                    elif isinstance(line, dict) and "data" in line:
                                        diff_content += line["data"] + "\n"
                                if not diff_content.strip():
                                    logger.debug(f"diff2 produced no output for {depot_file}, trying regular diff")
                                    diff_args = ["diff", "-du", curr_rev]
                                    diff_output = self.p4.run(*diff_args)
                                    for line in diff_output:
                                        if isinstance(line, str):
                                            diff_content += line + "\n"
                                        elif isinstance(line, dict) and "data" in line:
                                            diff_content += line["data"] + "\n"
                                if diff_content.strip():
                                    if len(diff_content) > 2000:
                                        diff_content = diff_content[:2000] + "... [truncated]"
                                    snippet_entry["snippet"] = diff_content
                                else:
                                    logger.debug(f"Both diff methods failed for {depot_file}, getting current version")
                                    content = self.p4.run("print", curr_rev)
                                    if content and len(content) > 0:
                                        file_content = ""
                                        for i in range(1, len(content)):
                                            if isinstance(content[i], str):
                                                file_content += content[i]
                                            elif isinstance(content[i], bytes):
                                                try:
                                                    file_content += content[i].decode("utf-8")
                                                except UnicodeDecodeError:
                                                    file_content = "[Binary content]"
                                                    break
                                        if len(file_content) > 2000:
                                            file_content = file_content[:2000] + "... [truncated]"
                                        snippet_entry[
                                            "snippet"] = "[Changes not visible in diff] Current content:\n" + file_content
                                    else:
                                        snippet_entry[
                                            "snippet"] = "[No changes detected in diff and unable to get file content]"
                            else:
                                snippet_entry["snippet"] = "[No diff output available]"
                        except Exception as diff_err:
                            logger.warning(f"Error getting diff for {depot_file}: {diff_err}")
                            snippet_entry["snippet"] = f"[Error retrieving diff: {str(diff_err)}]"

                    elif action in ["delete", "move/delete"]:
                        snippet_entry["snippet"] = "[File deleted]"
                    else:
                        snippet_entry["snippet"] = f"[Action '{action}' not handled for snippets]"
                except Exception as e:
                    logger.error(f"Unexpected error preparing snippet for {depot_file}: {e}")
                    snippet_entry["snippet"] = f"[Error: {str(e)}]"

                result["code_snippets"].append(snippet_entry)
                file_count += 1

                # Limit number of files processed per CL
                if file_count >= 5:
                    result["code_snippets"].append({
                        "file": "TRUNCATED",
                        "action": "info",
                        "snippet": f"[Showing snippets for {file_count} of {len(result['files'])} files]",
                        "language": "plaintext"
                    })
                    break

            if file_count > 0:
                logger.info(f"Added snippets for {file_count} files in CL {cl['change']}")
            else:
                logger.warning(f"No snippets added for CL {cl['change']}")

            return result
        except Exception as e:
            logger.error(f"Unexpected error extracting code snippets for CL: {e}")
            return cl

    def _parse_describe_output(self, output: str, cl_number: str) -> Dict:
        """
        Parse the output from p4 describe command into a structured dictionary.

        Args:
            output: The output string from p4 describe command
            cl_number: The changelist number for reference

        Returns:
            Dictionary with parsed changelist details
        """
        if not output:
            return {}

        # Initialize the result structure
        result = {
            "change": cl_number,
            "user": "",
            "client": "",
            "time": 0,
            "description": "",
            "status": "unknown",
            "files": []
        }

        # Extract basic CL information
        change_match = re.search(
            r"Change\s+(\d+)\s+by\s+([^@]+)@([^\s]+)\s+on\s+(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})", output)
        if change_match:
            result["change"] = change_match.group(1)
            result["user"] = change_match.group(2).strip()
            result["client"] = change_match.group(3).strip()
            # Convert time string to timestamp if needed

        # Extract description
        desc_match = re.search(r"\n\n(.*?)\n\nAffected files", output, re.DOTALL)
        if desc_match:
            result["description"] = desc_match.group(1).strip()

        # Extract files
        file_lines = []
        files_section_match = re.search(r"Affected files\s+\.{3}\s+\n\n(.*?)(?:\n\nDifferences|$)", output, re.DOTALL)
        if files_section_match:
            file_lines = files_section_match.group(1).strip().splitlines()

        for line in file_lines:
            # Example file line format: //depot/path/to/file#1 edit
            file_match = re.search(r"(//[^#]+)#(\d+)\s+(\w+)", line)
            if file_match:
                depot_file = file_match.group(1)
                revision = file_match.group(2)
                action = file_match.group(3)

                result["files"].append({
                    "depotFile": depot_file,
                    "action": action,
                    "rev": revision,
                    "type": self._detect_file_type(depot_file)
                })

        # Extract status if present
        status_match = re.search(r"Status:\s+(\w+)", output)
        if status_match:
            result["status"] = status_match.group(1).strip()

        return result

    def _detect_file_type(self, file_path: str) -> str:
        """Detect file type based on file extension"""
        _, ext = os.path.splitext(file_path)
        if ext in ['.jpg', '.png', '.gif', '.bmp', '.tiff', '.ico']:
            return 'binary+image'
        elif ext in ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx']:
            return 'binary+document'
        elif ext in ['.zip', '.tar', '.gz', '.rar', '.7z']:
            return 'binary+archive'
        elif ext in ['.exe', '.dll', '.so', '.dylib']:
            return 'binary+executable'
        elif ext in ['.js', '.ts', '.jsx', '.tsx']:
            return 'text+javascript'
        elif ext in ['.py']:
            return 'text+python'
        elif ext in ['.java']:
            return 'text+java'
        elif ext in ['.c', '.cpp', '.h', '.hpp']:
            return 'text+c++'
        elif ext in ['.cs']:
            return 'text+csharp'
        elif ext in ['.html', '.htm']:
            return 'text+html'
        elif ext in ['.css', '.scss', '.sass', '.less']:
            return 'text+css'
        elif ext in ['.json']:
            return 'text+json'
        elif ext in ['.xml']:
            return 'text+xml'
        elif ext in ['.md', '.markdown']:
            return 'text+markdown'
        elif ext in ['.txt', '.log']:
            return 'text+plain'
        elif ext in ['.sql']:
            return 'text+sql'
        else:
            return 'unknown'

    def _ensure_connected(self):
        """
        Ensure we have a valid Perforce connection before running commands.
        Handles reconnection if needed.
        """
        if not self.p4:
            self.connect()
            return

        # Check if connected
        try:
            if not self.p4.connected():
                logger.info("Perforce connection lost, reconnecting...")
                self.connect()
        except Exception as e:
            logger.warning(f"Error checking Perforce connection state: {e}")
            # Force reconnection
            self.connect()

    def run_p4_command(self, command: str) -> Optional[str]:
        """
        Run a Perforce command and handle all error scenarios.

        Args:
            command: The p4 command to run (without 'p4' prefix)

        Returns:
            Command output as string or None if command failed
        """
        if not command:
            return None

        # Make sure we're connected
        self._ensure_connected()

        try:
            # Split the command into arguments
            args = command.split()

            # Run the command
            result = self.p4.run(args)

            # Handle different result types
            if isinstance(result, list):
                if len(result) == 0:
                    return ""
                elif all(isinstance(item, dict) for item in result):
                    # P4Python sometimes returns a list of dicts for commands like 'describe'
                    # We need to handle this specially
                    out_parts = []
                    for item in result:
                        if 'data' in item:
                            out_parts.append(item['data'])
                        else:
                            # If there's no 'data' key, serialize the entire dict
                            out_parts.append(str(item))
                    return "\n".join(out_parts)
                else:
                    # Just join all elements with newlines
                    return "\n".join(str(item) for item in result)
            elif isinstance(result, str):
                return result
            elif result is None:
                return ""
            else:
                # Try to convert whatever we got to a string
                return str(result)

        except P4Exception as e:
            error_msg = str(e)
            logger.error(
                f"Perforce Error ({command}): {error_msg}. Details logged to {self._get_error_log_path(error_msg)}")
            self._log_p4_error(e, command)
            return None
        except Exception as e:
            logger.error(f"Unexpected error running p4 command '{command}': {e}")
            return None

    def _get_error_log_path(self, error_msg: str) -> str:
        """
        Generate a path for logging Perforce errors with timestamp.

        Args:
            error_msg: The error message to include in the log

        Returns:
            Path to the log file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = str(self.error_log_path) if hasattr(self, 'error_log_path') else "perforce_logs"

        # Create the log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        return os.path.join(log_dir, f"p4_error_{timestamp}.log")

    def get_changelist_files_efficient(self, cl_number: str) -> List[Dict]:
        """
        Gets files from a changelist using the efficient 'p4 files @=' command.
        This is faster than using 'p4 describe' when you only need the file list.

        Args:
            cl_number: The changelist number

        Returns:
            A list of dictionaries with file information
        """
        if not cl_number or not cl_number.strip():
            logger.warning("Empty CL number provided to get_changelist_files_efficient")
            return []

        # Ensure connection
        self._ensure_connected()

        try:
            # Use the efficient 'files' command with the @= specifier
            command = f"files @={cl_number}"
            result = self.run_p4_command(command)

            if not result:
                logger.warning(f"No files found for CL {cl_number}")
                return []

            files = []

            # Handle the case where run_p4_command returns a string representing a dictionary
            if isinstance(result, str) and result.strip().startswith("{") and "depotFile" in result:
                depot_match = re.search(r"'depotFile':\s*'([^']+)'", result)
                rev_match = re.search(r"'rev':\s*'([^']+)'", result)
                action_match = re.search(r"'action':\s*'([^']+)'", result)
                type_match = re.search(r"'type':\s*'([^']+)'", result)

                if depot_match:
                    file_info = {
                        "depotFile": depot_match.group(1),
                        "rev": rev_match.group(1) if rev_match else "1",
                        "action": action_match.group(1) if action_match else "unknown",
                        "type": type_match.group(1) if type_match else "unknown"
                    }
                    files.append(file_info)
                    return files

            # Handle multiple lines case (multiple files)
            elif isinstance(result, str):
                for line in result.strip().split("\n"):
                    if line.strip().startswith("{") and "depotFile" in line:
                        # Extract fields with regex for each line
                        depot_match = re.search(r"'depotFile':\s*'([^']+)'", line)
                        rev_match = re.search(r"'rev':\s*'([^']+)'", line)
                        action_match = re.search(r"'action':\s*'([^']+)'", line)
                        type_match = re.search(r"'type':\s*'([^']+)'", line)

                        if depot_match:
                            file_info = {
                                "depotFile": depot_match.group(1),
                                "rev": rev_match.group(1) if rev_match else "1",
                                "action": action_match.group(1) if action_match else "unknown",
                                "type": type_match.group(1) if type_match else "unknown"
                            }
                            files.append(file_info)

            # Original parsing logic (for standard p4 files output format)
            if not files:
                for line in result.strip().split("\n"):
                    # Format: //depot/path/to/file#rev - action change rev_num (file_type)
                    match = re.match(r"(//[^#]+)#(\d+)\s+-\s+(\w+)\s+change\s+(\d+)\s+\((.+?)\)", line)
                    if match:
                        depot_file = match.group(1)
                        rev = match.group(2)
                        action = match.group(3)
                        file_type = match.group(5)

                        files.append({
                            "depotFile": depot_file,
                            "rev": rev,
                            "action": action,
                            "type": file_type
                        })

            return files
        except Exception as e:
            logger.error(f"Error getting files for CL {cl_number} using efficient method: {e}")
            return []

    def get_cl_content(self, cl_number: str) -> List[Dict]:
        """Retrieves file details + content for a specific changelist.

        Args:
            cl_number: The changelist number to retrieve

        Returns:
            A list of file info dictionaries with content
        """
        try:
            if not self.p4.connected():
                self.connect()

            logger.info(f"Getting content for CL {cl_number} using describe -du")

            try:
                # Use -du flag to get unified diffs
                describe_result = self.p4.run("describe", "-du", cl_number)
            except P4Exception as p4e:
                logger.warning(f"Error running describe -du for CL {cl_number}: {p4e}")
                # Try without -du flag as fallback
                describe_result = self.p4.run("describe", "-s", cl_number)

            if not describe_result:
                logger.warning(f"No describe output for CL {cl_number}")
                return []

            detail = describe_result[0]
            if isinstance(detail, dict):
                depot_files = detail.get('depotFile', [])
                actions = detail.get('action', [])
                rev_numbers = detail.get('rev', [])
            else:
                logger.error(f"Unexpected describe result format for CL {cl_number}")
                return []

            files_info = []

            # Ensure we handle them as lists
            if not isinstance(depot_files, list):
                depot_files = [depot_files]
            if not isinstance(actions, list):
                actions = [actions]
            if not isinstance(rev_numbers, list):
                rev_numbers = [rev_numbers]

            for idx, depot_file in enumerate(depot_files):
                file_info = {
                    'depotFile': depot_file,
                    'action': actions[idx] if idx < len(actions) else 'unknown',
                    'revision': rev_numbers[idx] if idx < len(rev_numbers) else '0',
                }

                try:
                    # If not deleted, fetch content
                    if file_info['action'] not in ['delete', 'move/delete']:
                        try:
                            print_result = self.p4.run('print', f"{depot_file}@{cl_number}")
                            if print_result and len(print_result) > 1:
                                # Process each line with the binary decoder
                                decoded_lines = [self._decode_binary_content(line) for line in print_result[1:]]
                                content = '\n'.join(decoded_lines)
                                if content:
                                    file_info['content'] = content
                        except P4Exception as e:
                            logger.warning(f"Could not get content for {depot_file}@{cl_number}: {e}")

                    # If edited or integrated, show diff
                    if file_info['action'] in ['edit', 'integrate']:
                        try:
                            prev_rev = int(file_info['revision']) - 1
                            if prev_rev > 0:
                                diff_result = self.p4.run('diff', '-du',
                                                          f"{depot_file}#{prev_rev}",
                                                          f"{depot_file}@{cl_number}")
                                if diff_result:
                                    # Process each line with the binary decoder
                                    decoded_diff = [self._decode_binary_content(line) for line in diff_result]
                                    diff_text = '\n'.join(decoded_diff)
                                    if diff_text:
                                        file_info['diff'] = diff_text
                        except P4Exception as e:
                            if "up-to-date" not in str(e):
                                logger.warning(f"Could not get diff for {depot_file}: {str(e)}")

                except P4Exception as e:
                    if "up-to-date" not in str(e):
                        logger.warning(f"Could not get content/diff for {depot_file}: {str(e)}")
                    continue

                files_info.append(file_info)

            return files_info

        except P4Exception as e:
            logger.error(f"Perforce error describing CL {cl_number}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in get_cl_content for CL {cl_number}: {e}")
        return []

    def _decode_binary_content(self, content) -> str:
        """
        Decode binary content from Perforce to a proper string.
        Will return a placeholder for binary data that can't be meaningfully decoded.

        Args:
            content: The content to decode, which may be binary or string

        Returns:
            Decoded string content or placeholder for binary data
        """
        # First check if input looks like binary data
        if isinstance(content, bytes):
            # Check if this looks like binary data by sampling
            sample_size = min(100, len(content))
            binary_chars = 0
            for byte in content[:sample_size]:
                if byte < 32 and byte not in (9, 10, 13):  # Not tab, LF, or CR
                    binary_chars += 1
            
            # If more than 10% of first 100 bytes are binary, treat as binary data
            if binary_chars > sample_size * 0.1:
                content_start = content[:20].hex()
                return f"[Binary data detected - first bytes: {content_start}...]"
                
            try:
                return content.decode('utf-8', errors='replace')
            except UnicodeDecodeError:
                return content.decode('latin1', errors='replace')
        elif isinstance(content, str):
            # Check if the string looks like it contains binary data
            if len(content) > 0:
                binary_chars = 0
                sample_size = min(100, len(content))
                for char in content[:sample_size]:
                    if ord(char) < 32 and ord(char) not in (9, 10, 13):
                        binary_chars += 1
                
                if binary_chars > sample_size * 0.1:
                    return f"[Binary data detected in string - length: {len(content)} bytes]"
                    
            return content
        else:
            return str(content)

    def connect_if_needed(self):
        """Connect to Perforce server if not already connected."""
        try:
            # Check if we're already connected
            if getattr(self, '_connected', False):
                logger.debug("Already connected to Perforce.")
                return

            # Connect to Perforce
            self.connect()
        except Exception as e:
            logger.error(f"Error connecting to Perforce: {e}")
            # Re-raise to be handled by caller
            raise

    def _log_error_to_file(self, method_name, query, exception):
        """
        Log a Perforce error to a file for debugging.

        Args:
            method_name (str): The name of the method that encountered the error
            query (str): The query or parameter that caused the error
            exception (Exception): The exception that was raised
        """
        try:
            # Create logs directory if it doesn't exist
            log_dir = Path(os.environ.get('P4_ERROR_LOG_DIR', './perforce_logs'))
            log_dir.mkdir(exist_ok=True)

            # Create a timestamped log file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = log_dir / f"p4_error_{timestamp}.log"

            # Gather environment info
            env_info = {
                'P4CLIENT': os.environ.get('P4CLIENT', 'Not set'),
                'P4PORT': os.environ.get('P4PORT', 'Not set'),
                'P4USER': os.environ.get('P4USER', 'Not set'),
                'DEPOT_PATH': os.environ.get('DEPOT_PATH', 'Not set'),
                'Method': method_name,
                'Query': query,
                'Exception Type': type(exception).__name__,
                'Exception Message': str(exception)
            }

            # Write to the log file
            with open(log_file_path, 'w', encoding='utf-8') as f:
                f.write(f"Perforce Error Log - {timestamp}\n")
                f.write("=" * 50 + "\n\n")

                # Write environment info
                f.write("ENVIRONMENT INFORMATION:\n")
                for key, value in env_info.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")

                # Write exception details
                f.write("EXCEPTION DETAILS:\n")
                import traceback
                f.write(traceback.format_exc())

            logger.info(f"Perforce error logged to {log_file_path}")
        except Exception as log_err:
            logger.error(f"Failed to log Perforce error to file: {log_err}")
            # Don't raise - this is auxiliary logging functionality

    def get_snippets_sync(self, cl_number: str, max_files: Optional[int] = None) -> List[Dict]:
        """
        Synchronous version of get_relevant_code_snippets for compatibility with code that can't use async.
        Fetches relevant code diff snippets (+/- lines with context) or full file content on fallback.
        
        Args:
            cl_number: The changelist number to get snippets for
            max_files: Optional maximum number of files to include
            
        Returns:
            List of dictionaries containing file and snippet information
        """
        logger.info(f"Fetching diff snippets synchronously for CL {cl_number} (max_files={max_files if max_files is not None else 'All'})")
        snippets = []
        
        try:
            # Ensure we're connected to Perforce
            self.connect()
            
            # Get file list using the efficient method
            files_info = self.get_changelist_files_efficient(cl_number)
            
            if not files_info:
                logger.warning(f"Could not get file list for CL {cl_number} to fetch snippets.")
                return []

            files_processed = 0
            
            # Process each file in the changelist
            for file_info in files_info:
                # Only break if max_files is set and reached
                if max_files is not None and files_processed >= max_files: 
                    break

                depot_file = file_info.get('depotFile')
                action = file_info.get('action', 'unknown')
                revision = file_info.get('rev')
                file_type = file_info.get('type', '')

                if not depot_file or 'binary' in file_type or action == 'delete':
                    continue  # Skip binary, deleted, or invalid entries

                # Initialize placeholder for snippet data
                snippet_entry = {
                    'file': depot_file,
                    'action': action,
                    'snippet': '[Snippet retrieval failed]',  # Default error
                    'language': self._detect_language(os.path.splitext(depot_file)[1].lower())
                }

                try:
                    if action == 'add':
                        # For added files, get full content
                        content = self._get_file_content_via_session(depot_file, cl_number)
                        if content:
                            formatted_added_content = '\n'.join([f"+ {line}" for line in content.splitlines()])
                            snippet_entry['snippet'] = self._extract_diff_lines(formatted_added_content)
                        else:
                            snippet_entry['snippet'] = '[Could not retrieve content for added file]'
                    elif action in ['edit', 'integrate', 'branch', 'move/add']:
                        # Get diff for modified files
                        diff_content = self.get_diff_from_describe(cl_number, depot_file)
                        
                        if diff_content:
                            snippet_entry['snippet'] = self._extract_diff_lines(diff_content)
                        else:
                            # Diff failed, try fallback content retrieval
                            logger.warning(f"Diff retrieval failed for {depot_file}@{revision}, trying fallback.")
                            content = self._get_file_content_via_session(depot_file, cl_number)
                            if content:
                                # Format as full file content fallback
                                snippet_entry['snippet'] = "[FALLBACK - FULL FILE CONTENT]\n" + '\n'.join(
                                    [f"  {line}" for line in content.splitlines()])  # Indent slightly
                            else:
                                snippet_entry['snippet'] = '[Fallback content retrieval failed]'
                    else:
                        snippet_entry['snippet'] = f'[Action \'{action}\' not handled for snippets]'

                except Exception as e:
                    logger.error(f"Error preparing snippet for {depot_file}@{revision}: {e}", exc_info=True)
                    snippet_entry['snippet'] = f'[Error: {str(e)}]'

                snippets.append(snippet_entry)
                files_processed += 1
                
        except Exception as e:
            logger.error(f"Error in get_snippets_sync for CL {cl_number}: {e}", exc_info=True)
            
        return snippets

    def _format_result_for_output(self, result) -> Dict[str, Any]:
        """Format perforce output for downstream consumption"""
        from src.assistant.utils.response_formatter import sanitize_binary_content, format_tool_response

        # Use our sanitize utility to handle binary content recursively
        return format_tool_response(result, "perforce")


# --- Example Main Usage ---
async def main_async():
    # --- ADDED: Configure logging for DEBUG level ---
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s')
    # Optionally set level for the specific logger used in the class
    # logging.getLogger("__main__").setLevel(logging.DEBUG) # Or use the actual logger name if different
    # --- END ADDED ---
    helper = None
    try:
        helper = PerforceHelper()
        async with helper.async_session():  # Use the async context manager
            print("Connected to Perforce server (async context)")
            test_mtv = "MTV2005"  # Example MTV
            print(f"\nAsync searching for {test_mtv} changes (last 180 days)...")

            changes = await helper.get_mtv_changes(test_mtv)
            print(f"Found {len(changes)} changes for {test_mtv}")

            if changes:
                # Fetch details and snippets concurrently
                detail_tasks = []
                snippet_tasks = []
                for change in changes[:5]:  # Limit for demonstration
                    cl = change.get('change')
                    if cl:
                        # Use asyncio.to_thread for synchronous get_change_details
                        detail_tasks.append(asyncio.to_thread(helper.get_change_details, cl))
                        # Use get_snippets_sync for snippet fetching
                        snippet_tasks.append(asyncio.to_thread(helper.get_snippets_sync, cl))

                details_list = await asyncio.gather(*detail_tasks, return_exceptions=True)
                snippets_list = await asyncio.gather(*snippet_tasks, return_exceptions=True)

                for i, change in enumerate(changes[:5]):
                    cl = change.get('change')
                    print("-" * 50)
                    print(f"CL: {cl}")
                    details = details_list[i] if i < len(details_list) and not isinstance(details_list[i],
                                                                                          Exception) else {}
                    print(f"User: {details.get('user', change.get('user'))}")
                    try:
                        time_str = datetime.fromtimestamp(int(details.get('time', change.get('time')))).strftime(
                            '%Y-%m-%d %H:%M:%S')
                    except:
                        time_str = "Unknown"
                    print(f"Date: {time_str}")
                    print(f"Description: {details.get('description', change.get('desc'))[:200]}...")

                    # Swarm Link
                    if helper.p4_swarm_url and cl and cl.isdigit():
                        print(f"Swarm Link: {helper.p4_swarm_url}/changes/{cl}")

                    # Print Files
                    files = details.get('files', [])
                    if files:
                        print(f"Files ({len(files)}):")
                        for f_info in files[:3]:  # Show first 3 files
                            print(f"  - {f_info.get('depotFile')} ({f_info.get('action')})")
                        if len(files) > 3: print("    ...")

                    # Print Snippets (Placeholder)
                    snippets = snippets_list[i] if i < len(snippets_list) and not isinstance(snippets_list[i],
                                                                                             Exception) else []
                    if snippets:
                        print("Code Snippets:")
                        for snip in snippets:
                            print(f"  File: {snip.get('file')}")
                            print(f"  ```\n{snip.get('snippet')[:200]}...\n  ```")  # Show limited snippet
                    print("-" * 50)

    except Exception as e:
        print(f"Error in async main: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the async main function
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("Execution interrupted.")
    finally:
        print("Perforce tool script finished.")
