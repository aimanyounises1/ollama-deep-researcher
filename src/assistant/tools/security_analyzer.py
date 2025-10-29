"""Security analysis tool for code and configuration files.
Provides functionality to detect and analyze potential security issues.
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SecurityAnalyzer:
    """Enhanced security analyzer for sensitive content detection in files and code."""
    
    # Enhanced patterns for sensitive file detection
    SENSITIVE_FILE_PATTERNS = [
        # Credential files
        r'.*password.*\.(txt|json|xml|properties|config|ini|env|yaml|yml)$',
        r'.*secret.*\.(txt|json|xml|properties|config|ini|env|yaml|yml)$',
        r'.*credential.*\.(txt|json|xml|properties|config|ini|env|yaml|yml)$',
        r'.*\.pem$',
        r'.*\.key$',
        r'.*\.p12$',
        r'.*\.pfx$',
        r'.*\.keystore$',
        r'.*\.jks$',
        r'.*id_rsa$',
        r'.*\.env$',
        r'.*\.npmrc$',
        r'.*\.htpasswd$',
        
        # Config files with potential secrets
        r'.*oauth.*\.(json|xml|properties|config|ini|env|yaml|yml)$',
        r'.*aws.*credentials.*',
        r'.*gcp.*credentials.*',
        r'.*azure.*credentials.*',
        
        # Security-sensitive paths
        r'.*/\.ssh/.*',
        r'.*/\.aws/.*',
        r'.*/\.config/gcloud/.*',
    ]
    
    # Enhanced patterns for sensitive content detection
    SENSITIVE_CONTENT_PATTERNS = [
        # API keys and tokens
        r'(?i)api[_\-\s]?key[_\-\s]?[:=]\s*[\'"`]?([a-zA-Z0-9_\-\.]{16,64})[\'"`]?',
        r'(?i)auth[_\-\s]?token[_\-\s]?[:=]\s*[\'"`]?([a-zA-Z0-9_\-\.]{16,64})[\'"`]?',
        r'(?i)access[_\-\s]?token[_\-\s]?[:=]\s*[\'"`]?([a-zA-Z0-9_\-\.]{16,64})[\'"`]?',
        r'(?i)secret[_\-\s]?key[_\-\s]?[:=]\s*[\'"`]?([a-zA-Z0-9_\-\.]{16,64})[\'"`]?',
        
        # AWS specific
        r'(?i)aws[_\-\s]?access[_\-\s]?key[_\-\s]?id[_\-\s]?[:=]\s*[\'"`]?([A-Z0-9]{20})[\'"`]?',
        r'(?i)aws[_\-\s]?secret[_\-\s]?access[_\-\s]?key[_\-\s]?[:=]\s*[\'"`]?([a-zA-Z0-9/+]{40})[\'"`]?',
        r'(?i)account[_\-\s]?key[_\-\s]?[:=]\s*[\'"`]?([a-zA-Z0-9/+]{40})[\'"`]?',
        
        # Database connection strings
        r'(?i)jdbc:(?:mysql|postgresql|oracle|sqlserver)://.*(?:password|passwd|pwd)=[^&\s]+',
        r'(?i)mongodb(?:\+srv)?://[^:]+:[^@]+@.+',
        
        # Password assignments
        r'(?i)password[_\-\s]?[:=]\s*[\'"`]?([^\'"`\s]{8,32})[\'"`]?',
        r'(?i)passwd[_\-\s]?[:=]\s*[\'"`]?([^\'"`\s]{8,32})[\'"`]?',
        r'(?i)pwd[_\-\s]?[:=]\s*[\'"`]?([^\'"`\s]{8,32})[\'"`]?',
        
        # Tokens and keys
        r'(?i)private[_\-\s]?key[_\-\s]?[:=]',
        r'-----BEGIN\s+(?:RSA\s+|ENCRYPTED\s+)?PRIVATE\s+KEY-----',
        
        # OAuth tokens
        r'(?i)oauth.*token[_\-\s]?[:=]\s*[\'"`]?([a-zA-Z0-9_\-\.]{16,64})[\'"`]?',
        
        # JWT tokens
        r'eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}',
    ]

    @classmethod
    def analyze_diff(cls, diff_content: str) -> List[Dict[str, Any]]:
        """
        Analyze a diff for potential security issues.
        
        Args:
            diff_content: The diff content to analyze
            
        Returns:
            List of security issues found
        """
        if not diff_content:
            return []
            
        issues = []
        
        # Split diff into lines
        lines = diff_content.splitlines()
        
        # Track line numbers in the diff
        line_num = 0
        file_path = "unknown"
        
        for line in lines:
            line_num += 1
            
            # Track file paths in diff
            file_match = re.match(r'^(?:---|\+\+\+)\s+([^\s]+)', line)
            if file_match:
                file_path = file_match.group(1)
                continue
                
            # Only analyze added lines (starting with +)
            if not line.startswith('+'):
                continue
                
            # Skip the diff marker
            content_line = line[1:]
            
            # Check for sensitive data in the added line
            for pattern in cls.SENSITIVE_CONTENT_PATTERNS:
                match = re.search(pattern, content_line)
                if match:
                    issues.append({
                        "type": "sensitive_data",
                        "line": line_num,
                        "file": file_path,
                        "content": content_line.replace(match.group(0), f"[SENSITIVE: {match.group(0)[:5]}...]"),
                        "severity": "high"
                    })
                    break
                    
        return issues

    @classmethod
    def analyze_file_content(cls, file_content: str, filename: str = '') -> List[Dict[str, Any]]:
        """
        Analyze file content for potential security issues.
        
        Args:
            file_content: The file content to analyze
            filename: Optional filename for context
            
        Returns:
            List of security issues found
        """
        if not file_content:
            return []
            
        issues = []
        
        # Check if this is a sensitive file type
        is_sensitive = cls.is_sensitive_file(filename)
        if is_sensitive:
            issues.append({
                "type": "sensitive_file_type",
                "file": filename,
                "severity": "medium",
                "description": f"File {filename} appears to be a sensitive configuration or credential file"
            })
            
        # Analyze content line by line
        lines = file_content.splitlines()
        for i, line in enumerate(lines):
            for pattern in cls.SENSITIVE_CONTENT_PATTERNS:
                match = re.search(pattern, line)
                if match:
                    issues.append({
                        "type": "sensitive_data",
                        "line": i + 1,
                        "file": filename,
                        "content": line.replace(match.group(0), f"[SENSITIVE: {match.group(0)[:5]}...]"),
                        "severity": "high",
                        "description": "Possible credential or sensitive data detected"
                    })
                    break
        
        # Look for additional security issues based on content
        cls._check_additional_security_issues(file_content, filename, issues)
        
        return issues
        
    @classmethod
    def _check_additional_security_issues(cls, content: str, filename: str, issues: List[Dict[str, Any]]) -> None:
        """
        Check for additional security issues beyond credentials.
        
        Args:
            content: The file content
            filename: Filename for context
            issues: Issues list to append findings to
        """
        # Check for hardcoded IP addresses
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        ip_matches = re.findall(ip_pattern, content)
        if ip_matches:
            unique_ips = set(ip_matches)
            # Filter out common non-sensitive IPs
            sensitive_ips = [ip for ip in unique_ips if not (
                ip.startswith('127.0.0.') or  # localhost
                ip.startswith('192.168.') or  # private
                ip.startswith('10.') or       # private
                ip == '0.0.0.0' or            # wildcard
                ip == '255.255.255.255'       # broadcast
            )]
            
            if sensitive_ips:
                issues.append({
                    "type": "hardcoded_ip",
                    "file": filename,
                    "severity": "low",
                    "description": f"Hardcoded IP addresses found: {', '.join(sensitive_ips[:5])}" + 
                                  (f" and {len(sensitive_ips) - 5} more" if len(sensitive_ips) > 5 else "")
                })
                
        # Check for insecure functions/patterns (language dependent)
        if filename.endswith(('.c', '.cpp', '.h')):
            c_insecure_funcs = [
                'strcpy', 'strcat', 'sprintf', 'gets', 'scanf', 'memcpy', 'system'
            ]
            for func in c_insecure_funcs:
                if re.search(rf'\b{func}\s*\(', content):
                    issues.append({
                        "type": "insecure_function",
                        "file": filename,
                        "severity": "medium",
                        "description": f"Potentially insecure function '{func}' detected"
                    })
                    
        # Check for weak encryption (e.g., MD5, SHA-1)
        if re.search(r'\b(?:MD5|md5|SHA1|sha1)\b', content):
            issues.append({
                "type": "weak_crypto",
                "file": filename,
                "severity": "medium",
                "description": "Potentially weak cryptographic algorithm (MD5 or SHA-1) detected"
            })

    @staticmethod
    def is_sensitive_file(filename: str) -> bool:
        """
        Enhanced check if a file might contain sensitive information based on its name or path.
        
        Args:
            filename: The file name or path to check
            
        Returns:
            True if the file might contain sensitive information, False otherwise
        """
        if not filename:
            return False
            
        # Normalize filename for consistent matching
        filename = filename.lower()
        
        # Check against all sensitive file patterns
        for pattern in SecurityAnalyzer.SENSITIVE_FILE_PATTERNS:
            if re.match(pattern, filename, re.IGNORECASE):
                logger.warning(f"Sensitive file detected: {filename}")
                return True
                
        return False

    @staticmethod
    def redact_sensitive_data(content: str) -> str:
        """
        Redact sensitive data like API keys, passwords, and tokens from text content.
        
        Args:
            content: The text content to redact
            
        Returns:
            The content with sensitive data redacted
        """
        if not content:
            return content
            
        redacted_content = content
        for pattern in SecurityAnalyzer.SENSITIVE_CONTENT_PATTERNS:
            # Replace full matches with redacted value, preserving the key name
            redacted_content = re.sub(
                pattern, 
                lambda m: m.group(0).replace(m.group(1), "[REDACTED]") if len(m.groups()) > 0 else "[REDACTED]",
                redacted_content
            )
            
        return redacted_content 