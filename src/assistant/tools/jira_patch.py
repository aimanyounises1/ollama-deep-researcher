# JIRA Session Monkey Patch
"""This module monkey patches the JIRA library to handle sessions correctly.
Import this module before importing JIRA to apply the patch.
"""
import logging
import os
import re
import base64
from typing import Dict, List, Any, Optional, Union

from jira import JIRA as OriginalJIRA

logger = logging.getLogger(__name__)

# Save the original __init__ method
original_init = OriginalJIRA.__init__

# Define a patched __init__ method
def patched_init(self, *args, **kwargs):
    # Remove the 'session' parameter if it exists
    if 'session' in kwargs:
        logger.debug("Removing 'session' parameter from JIRA constructor")
        session = kwargs.pop('session')
        # You might want to do something with the session object here
    
    # Call the original __init__ method
    original_init(self, *args, **kwargs)

# Apply the patch
OriginalJIRA.__init__ = patched_init

# Export the patched class
JIRA = OriginalJIRA

def get_field_mapping() -> Dict[str, str]:
    """
    Get a mapping of common field names to their Jira field IDs.
    Enhanced with additional field mappings from Jira REST API docs.
    
    Returns:
        A dictionary mapping user-friendly field names to Jira field IDs
    """
    return {
        "summary": "summary",
        "description": "description",
        "comment": "comment",
        "status": "status",
        "assignee": "assignee",
        "reporter": "reporter",
        "priority": "priority",
        "labels": "labels",
        "components": "components",
        "fixVersions": "fixVersions",
        "versions": "versions",
        "issuelinks": "issuelinks",
        "customfield_10000": "epic_link",
        "customfield_10001": "story_points",
        "customfield_10002": "business_value",
        "customfield_10003": "release_notes",
        "customfield_10004": "acceptance_criteria",
        "customfield_10005": "sprint",
        "customfield_10006": "epic_name",
        "customfield_10007": "team",
        "customfield_10008": "test_instructions",
        "duedate": "duedate",
        "resolution": "resolution",
        "resolutiondate": "resolutiondate",
        "created": "created",
        "updated": "updated",
        "issuetype": "issuetype",
        "project": "project",
        "attachment": "attachment",
        "worklog": "worklog",
        "subtasks": "subtasks",
        "environment": "environment",
    }

def extract_issue_id_from_url(url: str) -> Optional[str]:
    """
    Extract Jira issue ID from URL with enhanced regex patterns.
    Handles various Jira URL formats.
    
    Args:
        url: The URL potentially containing a Jira issue ID
        
    Returns:
        The extracted issue ID or None if not found
    """
    # Handle multiple Jira URL formats
    patterns = [
        r'browse/([A-Z]+-\d+)',              # Standard browse URL
        r'issues/([A-Z]+-\d+)',              # Issue path
        r'[?&]key=([A-Z]+-\d+)',             # Key parameter
        r'[?&]selectedIssue=([A-Z]+-\d+)',   # Selected issue parameter
        r'[?&]issueKey=([A-Z]+-\d+)',        # Issue key parameter
        r'/([A-Z]+-\d+)(?:\?|$|#)',          # Direct ID in path
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    # Also look for plain issue IDs in the URL
    plain_id_match = re.search(r'([A-Z]+-\d+)', url)
    if plain_id_match:
        return plain_id_match.group(1)
    
    return None

def encode_attachment_data(file_path: str) -> Optional[Dict[str, str]]:
    """
    Encode a file for Jira attachment upload.
    
    Args:
        file_path: Path to the file to encode
        
    Returns:
        Dictionary with file name and base64-encoded data, or None if failed
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
            
        with open(file_path, 'rb') as file:
            file_data = file.read()
            encoded_data = base64.b64encode(file_data).decode('utf-8')
            
        return {
            "filename": os.path.basename(file_path),
            "content": encoded_data
        }
    except Exception as e:
        logger.error(f"Error encoding attachment: {e}")
        return None

def parse_jira_changelog(changelog_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Parse Jira changelog data into a more usable format.
    
    Args:
        changelog_data: Raw changelog data from Jira API
        
    Returns:
        Parsed changelog list with cleaned and structured entries
    """
    if not changelog_data:
        return []
        
    result = []
    field_map = get_field_mapping()
    
    for entry in changelog_data:
        try:
            author = entry.get("author", {}).get("displayName", "Unknown")
            created = entry.get("created", "Unknown")
            
            # Process each change item
            items = []
            for item in entry.get("items", []):
                field = item.get("field")
                # Map to friendly field name if possible
                friendly_field = next((k for k, v in field_map.items() if v == field), field)
                
                change = {
                    "field": friendly_field,
                    "from": item.get("fromString", ""),
                    "to": item.get("toString", "")
                }
                items.append(change)
                
            result.append({
                "author": author,
                "date": created,
                "changes": items
            })
        except Exception as e:
            logger.warning(f"Error parsing changelog entry: {e}")
            continue
            
    return result

def extract_epic_data(issue_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract Epic-specific data from an issue.
    
    Args:
        issue_data: The issue data dictionary
        
    Returns:
        Dictionary with extracted Epic data
    """
    epic_data = {
        "is_epic": False,
        "epic_name": "",
        "epic_status": "",
        "epic_progress": {},
        "epic_color": ""
    }
    
    if not issue_data:
        return epic_data
    
    try:
        # Check if issue is an Epic
        issue_type = issue_data.get("fields", {}).get("issuetype", {}).get("name", "")
        if issue_type.lower() == "epic":
            epic_data["is_epic"] = True
            
            # Get Epic-specific fields
            fields = issue_data.get("fields", {})
            epic_data["epic_name"] = fields.get("customfield_10006", "") or fields.get("epic_name", "")
            epic_data["epic_status"] = fields.get("status", {}).get("name", "")
            
            # Get Epic progress info if available
            if "progress" in issue_data:
                epic_data["epic_progress"] = {
                    "percent": issue_data.get("progress", {}).get("percent", 0),
                    "total": issue_data.get("progress", {}).get("total", 0),
                    "resolved": issue_data.get("progress", {}).get("resolved", 0)
                }
                
            # Try to get Epic color
            if "color" in fields.get("customfield_10005", {}) or "color" in fields:
                epic_data["epic_color"] = fields.get("customfield_10005", {}).get("color", "") or fields.get("color", "")
    except Exception as e:
        logger.warning(f"Error extracting Epic data: {e}")
        
    return epic_data
