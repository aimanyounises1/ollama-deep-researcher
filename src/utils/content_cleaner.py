# src/utils/content_cleaner.py
import logging
import re
from typing import Optional

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Try to import trafilatura if available
try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    logger.warning("Trafilatura not installed. Install with 'pip install trafilatura' for better content cleaning.")
    TRAFILATURA_AVAILABLE = False

def clean_confluence_content(html_content: str) -> str:
    """
    Clean Confluence HTML content using trafilatura with fallbacks.

    Args:
        html_content: Raw HTML content from Confluence

    Returns:
        Cleaned text content
    """
    if not html_content:
        return ""

    # First attempt: Trafilatura (best quality)
    if TRAFILATURA_AVAILABLE:
        try:
            # Configure trafilatura for Confluence content
            extracted_text = trafilatura.extract(
                html_content,
                include_comments=False,
                include_tables=True,
                favor_precision=True,
                include_links=True, # Keep links for context
                include_formatting=False, # Usually better to remove formatting for LLMs
                # --- START FIX ---
                output_format='txt' # Use 'txt' for plain text output
                # --- END FIX ---
            )

            if extracted_text and len(extracted_text.strip()) > 50: # Lower threshold slightly
                logger.debug("Successfully cleaned content with trafilatura")
                return _post_process_text(extracted_text)
            else:
                 logger.debug("Trafilatura extracted minimal content, falling back.")
        except Exception as e:
            # Log the specific error from Trafilatura
            logger.warning(f"Trafilatura extraction failed: {e.__class__.__name__}: {e}")


    # Second attempt: BeautifulSoup + html2text (if available)
    try:
        # Use BeautifulSoup to clean HTML
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove Confluence-specific elements and common noise
        for element in soup.select(
            '.page-metadata, .confluence-information-macro, .aui-nav, '
            '.footer, #footer, .header, #header, .nav, .sidebar, '
            '.confluence-sidebar, script, style, noscript, .license, '
            '.comment, .author, .breadcrumbs, link[rel="stylesheet"], meta' # Added more noise tags
        ):
            if element: element.decompose()

        # Attempt html2text conversion
        try:
            import html2text
            h = html2text.HTML2Text()
            h.ignore_links = False # Keep links
            h.ignore_images = True # Remove images
            h.body_width = 0  # No wrapping
            h.unicode_snob = True # Improve unicode handling
            h.escape_snob = True # Avoid unnecessary backslash escapes

            cleaned_text = h.handle(str(soup))
            # Check if html2text produced meaningful output
            if cleaned_text and len(cleaned_text.strip()) > 50:
                logger.debug("Cleaned content with BeautifulSoup + html2text")
                return _post_process_text(cleaned_text)
            else:
                 logger.debug("html2text produced minimal content, trying direct text extraction.")
        except ImportError:
            logger.debug("html2text not available, using direct text extraction.")
            pass # Fall through to direct text extraction

        # Fallback: get text directly from BeautifulSoup
        cleaned_text = soup.get_text(separator='\n', strip=True)
        if cleaned_text and len(cleaned_text.strip()) > 20: # Lower threshold for basic text
             logger.debug("Cleaned content with BeautifulSoup direct text extraction.")
             return _post_process_text(cleaned_text)
        else:
             logger.warning("BeautifulSoup direct text extraction yielded minimal content.")


    except Exception as e:
        logger.error(f"BeautifulSoup extraction failed: {e}")

    # Final fallback: simple regex to strip tags (less reliable)
    try:
        text = re.sub(r'<style.*?</style>', ' ', html_content, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<script.*?</script>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text) # Strip remaining tags
        text = re.sub(r'\s+', ' ', text) # Collapse whitespace
        cleaned_text = text.strip()
        if cleaned_text:
            logger.debug("Cleaned content using basic regex tag stripping.")
            return _post_process_text(cleaned_text)
        else:
            logger.warning("Regex cleaning resulted in empty string.")

    except Exception as e:
        logger.error(f"Regex cleaning failed: {e}")

    # If all methods fail significantly, return a placeholder or original snippet
    logger.error("All content cleaning methods failed or yielded minimal content.")
    return html_content[:500] + "... (cleaning failed)" # Return original snippet as last resort

def _post_process_text(text: str) -> str:
    """Additional cleanup for extracted text."""
    if not text: return ""
    # Remove Confluence-specific patterns like [ ], |image|, etc.
    text = re.sub(r'\[\s*\]', '', text)
    text = re.sub(r'\|\s*image\s*\|', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[sep\]', ' | ', text) # Keep separator transformation
    text = re.sub(r'\[mtv\]', '', text) # Remove [mtv] tags

    # Fix excess newlines and leading/trailing whitespace on lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text) # Collapse multiple blank lines

    # Remove lines that are just symbols or very short (likely artifacts)
    text = '\n'.join(line for line in text.split('\n') if re.search(r'[a-zA-Z0-9]', line) and len(line) > 2)

    return text.strip()

def strip_sensitive_info(content: str) -> str:
    """
    Redact sensitive information from text content.
    
    Args:
        content: Text content that may contain sensitive information
        
    Returns:
        Text with sensitive information redacted
    """
    if not content or not isinstance(content, str):
        return ""
        
    # Regex patterns for sensitive information
    patterns = {
        'password': r'(password|passwd|pwd)[\s]*[=:]+[\s]*[\'"]?([^\s\'"<>]{6,})[\'"]?',
        'api_key': r'(api_key|apikey|api[-_]token|access[-_]token)[\s]*[=:]+[\s]*[\'"]?([^\s\'"<>]{8,})[\'"]?',
        'bearer_token': r'(bearer[\s]+)([A-Za-z0-9_\-\.=]{8,})',
        'aws_key': r'(AKIA[A-Z0-9]{16})',
        'jwt_token': r'(eyJ[a-zA-Z0-9_-]{5,}\.eyJ[a-zA-Z0-9_-]{5,})',
        'private_key': r'-----BEGIN[\s\S]*?PRIVATE KEY[\s\S]*?-----',
        'credit_card': r'(?:\d{4}[- ]?){3}\d{4}',
        'ssn': r'\d{3}-\d{2}-\d{4}',
        'email_with_password': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[\s,:]+[^\s,:]{6,}',
    }
    
    # Replace matches with redacted text
    redacted = content
    for key, pattern in patterns.items():
        if key == 'private_key':
            redacted = re.sub(pattern, '[REDACTED PRIVATE KEY]', redacted)
        elif key == 'email_with_password':
            # Keep email but redact password
            def redact_email_pwd(match):
                parts = re.split(r'[\s,:]+', match.group(0), 1)
                return f"{parts[0]} [REDACTED]"
            redacted = re.sub(pattern, redact_email_pwd, redacted)
        else:
            # Standard redaction for other patterns
            redacted = re.sub(pattern, 
                             lambda m: m.group(0).replace(m.group(2), '[REDACTED]') 
                                if len(m.groups()) > 1 else f"{m.group(1)}[REDACTED]", 
                             redacted)
    
    # Redact connection strings
    redacted = re.sub(
        r'(jdbc|mongodb|mysql|postgresql|db|data source):(//|:)[^\s\'"<>]{10,}',
        r'\1\2[REDACTED]',
        redacted,
        flags=re.IGNORECASE
    )
    
    # Redact IP addresses with credentials
    redacted = re.sub(
        r'([a-zA-Z0-9._%+-]+):([\S]{6,})@([a-zA-Z0-9.-]+)',
        r'\1:[REDACTED]@\3',
        redacted
    )
    
    logger.debug("Stripped sensitive information from content")
    return redacted

# Keep extract_main_content as an alias if used elsewhere
def extract_main_content(html_content: str) -> Optional[str]:
    """
    Extract the main content from HTML, ignoring navigation, headers, etc.
    Uses trafilatura with fallbacks.
    """
    return clean_confluence_content(html_content)