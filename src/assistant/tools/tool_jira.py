# tool_jira.py
import json
import logging
import os
import asyncio
import re
import time
from typing import List, Dict, Any, Optional, Union

from dotenv import load_dotenv
from langchain.agents import Tool

from src.assistant.tools.jira_tool import JiraProject, ConfigurationError
from src.assistant.utils.response_formatter import format_tool_response, format_error_response

logger = logging.getLogger(__name__)


async def search_jira(jql: Optional[str] = None, keyword: Optional[str] = None, field: Optional[str] = None, 
                 include_comments: bool = True, include_attachments: bool = True, 
                 max_results: int = 50, extract_mtv_id: bool = True,
                 mtv_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Search Jira for tickets matching the provided JQL or keyword and optionally extract MTV IDs.
    
    Args:
        jql: JQL query string (takes precedence if provided)
        keyword: Text to search for
        field: Specific field to search in
        include_comments: Whether to include comments
        include_attachments: Whether to include attachments
        max_results: Maximum number of results to return
        extract_mtv_id: Whether to identify and extract MTV IDs from the ticket
        mtv_id: Specific MTV ID to search for (simplifies building the query)
        
    Returns:
        Dictionary with search results and extracted MTV IDs
    """
    try:
        # Environment variables should already be loaded in run_graph_3.py
        
        if max_results <= 0:
            max_results = 50  # Enforce a reasonable default
            
        # Handle direct MTV ID search - takes precedence if provided
        if mtv_id:
            # Normalize MTV format by removing spaces and ensuring uppercase
            mtv_normalized = re.sub(r'\s+', '', mtv_id).upper()
            if not mtv_normalized.startswith('MTV'):
                mtv_normalized = f"MTV{mtv_normalized}"
                
            # Search in summary, description, and comments
            if jql:
                jql = f"({jql}) AND (summary ~ \"{mtv_normalized}\" OR description ~ \"{mtv_normalized}\" OR comment ~ \"{mtv_normalized}\")"
            else:
                jql = f"summary ~ \"{mtv_normalized}\" OR description ~ \"{mtv_normalized}\" OR comment ~ \"{mtv_normalized}\""
            
            logger.info(f"Searching for MTV ID: {mtv_normalized} with JQL: {jql}")
        
        # Initialize JiraProject with default configuration
        logger.debug("Initializing JiraProject")
        jira = JiraProject()
        
        # Use robust search for jira issues that handles timeouts
        result = await robust_search_issues(
            jira, 
            jql=jql, 
            keyword=keyword, 
            field=field, 
            max_results=max_results
        )
        
        # Process and enhance results
        processed_results = []
        found_mtv_ids = set()
        
        for issue in result.get("issues", []):
            # Create an enhanced issue representation
            enhanced_issue = {
                "key": issue.key,
                "summary": issue.fields.summary,
                "status": issue.fields.status.name if hasattr(issue.fields.status, 'name') else str(issue.fields.status),
                "created": issue.fields.created,
                "updated": issue.fields.updated,
                "description": issue.fields.description or "",
                "url": f"{jira.server}/browse/{issue.key}",
                "issuetype": issue.fields.issuetype.name if hasattr(issue.fields.issuetype, 'name') else str(issue.fields.issuetype),
                "components": [c.name for c in issue.fields.components] if hasattr(issue.fields, 'components') else [],
                "assignee": issue.fields.assignee.displayName if issue.fields.assignee else "Unassigned",
                "reporter": issue.fields.reporter.displayName if issue.fields.reporter else "Unknown",
                "priority": issue.fields.priority.name if hasattr(issue.fields.priority, 'name') else str(issue.fields.priority) if issue.fields.priority else "None",
            }
            
            # Add attachments if requested and available
            if include_attachments and hasattr(issue.fields, 'attachment') and issue.fields.attachment:
                attachments = []
                for attachment in issue.fields.attachment:
                    try:
                        attachment_data = {
                            "filename": attachment.filename,
                            "created": attachment.created if hasattr(attachment, 'created') else None,
                            "size": attachment.size if hasattr(attachment, 'size') else 0,
                            "content": None
                        }
                        
                        # Only download non-image attachments below a certain size
                        mime_type = attachment.mimeType if hasattr(attachment, 'mimeType') else ""
                        size = attachment.size if hasattr(attachment, 'size') else 0
                        
                        if size < 5000000 and not mime_type.startswith('image/'):  # Skip images and large files
                            try:
                                attachment_content = jira.download_attachment(attachment)
                                if attachment_content:
                                    # Convert content to text using appropriate parser based on file type
                                    converted_text = jira.convert_attachment_to_text(
                                        attachment.filename, attachment_content, mime_type
                                    )
                                    attachment_data["content"] = converted_text
                            except Exception as e:
                                logger.warning(f"Error downloading attachment {attachment.filename}: {e}")
                                
                        attachments.append(attachment_data)
                    except Exception as e:
                        logger.warning(f"Error processing attachment: {e}")
                        
                enhanced_issue["attachments"] = attachments
            
            # Add comments if requested
            if include_comments and hasattr(issue.fields, 'comment') and issue.fields.comment:
                comments = []
                for comment in issue.fields.comment.comments:
                    comments.append({
                        "author": comment.author.displayName if comment.author else "Unknown",
                        "created": comment.created,
                        "body": comment.body
                    })
                enhanced_issue["comments"] = comments
                
            # Add to results
            processed_results.append(enhanced_issue)
            
            # Extract MTV IDs if requested
            if extract_mtv_id:
                # Search in summary, description, and comments
                content_to_search = [
                    enhanced_issue.get("summary", ""),
                    enhanced_issue.get("description", "")
                ]
                
                # Add comment bodies if available
                if "comments" in enhanced_issue:
                    for comment in enhanced_issue["comments"]:
                        content_to_search.append(comment.get("body", ""))
                
                # Add attachment content if available
                if "attachments" in enhanced_issue:
                    for attachment in enhanced_issue["attachments"]:
                        if attachment.get("content"):
                            content_to_search.append(attachment.get("content", ""))
                
                # Find MTV IDs in all content
                for text in content_to_search:
                    if not text:
                        continue
                    
                    # Find MTV IDs using regex pattern
                    mtv_matches = re.finditer(r'MTV\d{4,}', text, re.IGNORECASE)
                    for match in mtv_matches:
                        found_mtv_ids.add(match.group(0).upper())  # Store in uppercase
        
        # Prepare final results
        final_result = {
            "query": {
                "jql": jql,
                "keyword": keyword,
                "field": field,
                "mtv_id": mtv_id
            },
            "issues": processed_results,
            "total_issues": len(processed_results),
            "extracted_mtv_ids": list(found_mtv_ids) if extract_mtv_id else []
        }
        
        return format_tool_response(final_result, "jira")
        
    except Exception as e:
        logger.error(f"Error searching Jira: {e}", exc_info=True)
        return format_error_response(e, "jira")


# The actual 'Tool' object that you pass to the agent:
jira_tool = Tool(
    name="jira_search",
    func=search_jira,
    description="""Use this tool to search JIRA issues by text query, MTV reference, or issue key.
    Particularly useful for finding MTV tickets, epics, and related issues.
    Input should be either an MTV number (e.g., 'MTV1492'), a JIRA key (e.g., 'VIT-50569'), or a text search query.
    
    For detailed information about a specific issue including acceptance criteria, field mappings, and parent/child relationships,
    set parameter get_hierarchy=True and provide an exact JIRA issue key (e.g., "VIT-50569", get_hierarchy=True).
    
    To control attachment retrieval, use download_attachments parameter:
    - download_attachments=True (default): Retrieves and processes attachments (PDFs, Excel, Word, images)
    - download_attachments=False: Skips downloading attachments content for faster response
    
    Additional advanced parameters:
    - fields: Comma-separated list of specific fields to retrieve (e.g., fields="summary,description,comment")
    - expand: Comma-separated list of fields to expand (e.g., expand="renderedFields,changelog")
    
    Example with all parameters: "VIT-50569", get_hierarchy=True, download_attachments=True, fields="summary,description,comment", expand="renderedFields"
    
    This will return comprehensive details including:
    - Critical acceptance criteria 
    - Business value and feature type
    - Technical integration points
    - Field mappings between systems
    - Parent/child/subtask relationships
    - Epic associations
    - Attachment content (text extraction, images, documents)"""
)
