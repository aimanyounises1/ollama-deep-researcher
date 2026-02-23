# src/assistant/processors/jira_processor.py

import logging
import asyncio
import json
import re
from typing import Dict, Any, List, Optional, Set, Union

from langchain_core.runnables import RunnableConfig

from src.assistant.core.types import ResearchState
from src.assistant.configuration import Configuration
from src.assistant.utils.helpers import traceable, extract_identifiers

logger = logging.getLogger(__name__)

# Constants for chunk processing
JIRA_CHUNK_SIZE = 10
JIRA_OVERLAP = 2

@traceable
async def process_jira_data(state: ResearchState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Process raw Jira data into structured format for further analysis.
    Extracts key information, handles embedded data, and creates cleaned version.
    """
    logger.info("--- Node: Processing Jira Data ---")
    
    # Get configuration
    cfg = Configuration.from_runnable_config(config) if config else Configuration()
    raw_jira = state.get("raw_jira_results", [])
    
    if not raw_jira:
        logger.info("No Jira data to process.")
        return {
            "pre_processed_summaries": state.get("pre_processed_summaries", {})
        }
    
    logger.info(f"Processing {len(raw_jira)} Jira items.")
    
    # Initialize storage for pre-processed bullets
    pre_processed = state.get("pre_processed_summaries", {})
    if "### Jira" not in pre_processed:
        pre_processed["### Jira"] = []
    
    # Track unique Jira keys to avoid duplicates
    processed_keys = set()
    
    # Helper function to extract text from complex fields
    def extract_text(field) -> str:
        if field is None:
            return ""
        if isinstance(field, str):
            return field
        if isinstance(field, dict):
            # Check for Jira's content structure
            if "content" in field and isinstance(field["content"], list):
                extracted = []
                for content_item in field["content"]:
                    if isinstance(content_item, dict):
                        # Handle paragraph type
                        if content_item.get("type") == "paragraph" and "content" in content_item:
                            paragraph_text = []
                            for text_item in content_item["content"]:
                                if isinstance(text_item, dict) and "text" in text_item:
                                    paragraph_text.append(text_item["text"])
                            extracted.append(" ".join(paragraph_text))
                        
                        # Handle bulletList type
                        elif content_item.get("type") == "bulletList" and "content" in content_item:
                            for list_item in content_item["content"]:
                                if isinstance(list_item, dict) and "content" in list_item:
                                    list_item_text = []
                                    for paragraph in list_item["content"]:
                                        if isinstance(paragraph, dict) and "content" in paragraph:
                                            for text in paragraph["content"]:
                                                if isinstance(text, dict) and "text" in text:
                                                    list_item_text.append(text["text"])
                                    extracted.append("- " + " ".join(list_item_text))
                return "\n".join(extracted)
            
            # Look for plain text representations
            for key in ["text", "value", "displayName", "name"]:
                if key in field and isinstance(field[key], str):
                    return field[key]
        return str(field)
    
    # Process each Jira item
    for item in raw_jira:
        if not isinstance(item, dict):
            continue
        
        try:
            # Extract key fields with fallbacks
            key = item.get("key", "")
            if not key and "id" in item:
                key = f"JIRA-{item['id']}"
            
            # Skip if already processed or no key
            if not key or key in processed_keys:
                continue
            
            processed_keys.add(key)
            
            # Extract fields with fallbacks
            fields = item.get("fields", {})
            
            # Normalize fields that might be in top level or in "fields"
            summary = item.get("summary") or fields.get("summary", "")
            description = item.get("description") or fields.get("description", "")
            status = extract_text(item.get("status") or fields.get("status"))
            priority = extract_text(item.get("priority") or fields.get("priority"))
            
            # Extract assignee
            assignee = item.get("assignee") or fields.get("assignee")
            assignee_name = extract_text(assignee)
            
            # Extract reporter
            reporter = item.get("reporter") or fields.get("reporter")
            reporter_name = extract_text(reporter)
            
            # Dates with fallbacks
            created = item.get("created") or fields.get("created", "")
            updated = item.get("updated") or fields.get("updated", "")
            
            # Format description text
            description_text = extract_text(description)
            if not description_text:
                description_text = "No description provided."
            
            # Extract MTV references
            mtv_refs = []
            mtv_pattern = re.compile(r'\b(MTV[-\s]?\d{4,})\b', re.IGNORECASE)
            for text in [key, summary, description_text]:
                mtv_matches = mtv_pattern.findall(text)
                for match in mtv_matches:
                    normalized = match.upper().replace(' ', '').replace('-', '')
                    if normalized not in mtv_refs:
                        mtv_refs.append(normalized)
            
            # Build URL
            url = f"https://jira/browse/{key}" if key else ""
            
            # Extract issue links
            issue_links = []
            links = fields.get("issuelinks", [])
            if isinstance(links, list):
                for link in links:
                    if isinstance(link, dict):
                        link_type = link.get("type", {}).get("name", "")
                        inward_issue = link.get("inwardIssue", {})
                        outward_issue = link.get("outwardIssue", {})
                        
                        if inward_issue:
                            link_key = inward_issue.get("key", "")
                            link_status = extract_text(inward_issue.get("fields", {}).get("status")) if "fields" in inward_issue else ""
                            link_summary = inward_issue.get("fields", {}).get("summary", "") if "fields" in inward_issue else ""
                            issue_links.append(f"{link_key} ({link_status}) - {link_summary} - {link_type}")
                        
                        if outward_issue:
                            link_key = outward_issue.get("key", "")
                            link_status = extract_text(outward_issue.get("fields", {}).get("status")) if "fields" in outward_issue else ""
                            link_summary = outward_issue.get("fields", {}).get("summary", "") if "fields" in outward_issue else ""
                            issue_links.append(f"{link_key} ({link_status}) - {link_summary} - {link_type}")
            
            # Extract attachments
            attachments = []
            attachment_list = fields.get("attachment", [])
            if isinstance(attachment_list, list):
                for attachment in attachment_list:
                    if isinstance(attachment, dict):
                        att_name = attachment.get("filename", "")
                        att_size = attachment.get("size", 0)
                        att_mime = attachment.get("mimeType", "")
                        attachments.append(f"{att_name} ({att_mime}, {att_size} bytes)")
            
            # Extract acceptance criteria - look for it in both custom fields and description
            acceptance_criteria = ""
            
            # Look in custom fields first
            for field_name, field_value in fields.items():
                if "acceptancecriteria" in field_name.lower() or "acceptance_criteria" in field_name.lower():
                    if field_value:
                        acceptance_criteria = extract_text(field_value)
                        break
                        
            # If not found in custom fields, try to extract from description
            if not acceptance_criteria and description_text:
                ac_patterns = [
                    r'(?i)acceptance criteria:?(.*?)(?:^#|\Z)', 
                    r'(?i)acceptance criteria:?(.*?)(?:\n\n|\Z)',
                    r'(?i)acceptance criteria:?(.*)',
                ]
                for pattern in ac_patterns:
                    match = re.search(pattern, description_text, re.DOTALL | re.MULTILINE)
                    if match:
                        acceptance_criteria = match.group(1).strip()
                        break
            
            # Create bullet point entries
            bullet = f"- **[{key}]({url})** {summary} ({status})"
            pre_processed["### Jira"].append(bullet)
            
            # Add description bullet
            desc_bullet = f"  - **Description**: {description_text[:500]}{'...' if len(description_text) > 500 else ''}"
            pre_processed["### Jira"].append(desc_bullet)
            
            # Add acceptance criteria if found
            if acceptance_criteria:
                ac_bullet = f"  - **Acceptance Criteria**: {acceptance_criteria[:500]}{'...' if len(acceptance_criteria) > 500 else ''}"
                pre_processed["### Jira"].append(ac_bullet)
            
            # Add MTV references if found
            if mtv_refs:
                mtv_bullet = f"  - **MTV References**: {', '.join(mtv_refs)}"
                pre_processed["### Jira"].append(mtv_bullet)
            
            # Add metadata bullets
            pre_processed["### Jira"].append(f"  - **Priority**: {priority}")
            pre_processed["### Jira"].append(f"  - **Assignee**: {assignee_name}")
            pre_processed["### Jira"].append(f"  - **Reporter**: {reporter_name}")
            pre_processed["### Jira"].append(f"  - **Created**: {created}")
            pre_processed["### Jira"].append(f"  - **Updated**: {updated}")
            
            # Add attachments if available
            if attachments:
                att_bullet = "  - **Attachments**: " + "; ".join(attachments[:3])
                if len(attachments) > 3:
                    att_bullet += f"; (and {len(attachments) - 3} more)"
                pre_processed["### Jira"].append(att_bullet)
            
            # Add issue links if available
            if issue_links:
                links_bullet = "  - **Linked Issues**: " + "; ".join(issue_links[:3])
                if len(issue_links) > 3:
                    links_bullet += f"; (and {len(issue_links) - 3} more)"
                pre_processed["### Jira"].append(links_bullet)
            
            # Add empty line for readability
            pre_processed["### Jira"].append("")
            
        except Exception as e:
            logger.error(f"Error processing Jira item {item.get('key', 'unknown')}: {e}")
    
    logger.info(f"Processed {len(processed_keys)} unique Jira issues into {len(pre_processed['### Jira'])} bullet points.")
    
    # Set up chunking for this source type
    state["jira_chunks"] = []
    state["jira_chunk_idx"] = 0
    state["jira_total_chunks"] = 0
    state["_last_jira_chunk_idx"] = -1
    state["_jira_stuck_count"] = 0
    
    return {
        "pre_processed_summaries": pre_processed
    }

@traceable
async def fetch_jira_data(state: ResearchState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Enhanced fetch_jira_data function that properly handles various JSON response formats
    from the Jira API and consolidates results for further processing.
    """
    logger.info("--- Node: Fetching Jira Data (Enhanced) ---")
    
    # Get raw results from state
    raw_results = state.get("raw_jira_results", [])
    
    # If we received a string, attempt to parse it as JSON
    if isinstance(raw_results, str):
        try:
            parsed_data = json.loads(raw_results)
            logger.info(f"Parsed Jira string response into JSON data")
            
            # Handle different possible JSON structures
            if isinstance(parsed_data, dict):
                # Handle case where response is a single issue
                if "key" in parsed_data:
                    raw_results = [parsed_data]
                    logger.info(f"Extracted single Jira issue with key: {parsed_data.get('key')}")
                
                # Handle case where response has an "issues" array
                elif "issues" in parsed_data and isinstance(parsed_data["issues"], list):
                    raw_results = parsed_data["issues"]
                    logger.info(f"Extracted {len(raw_results)} issues from 'issues' array")
                
                # Handle case where we have issue and hierarchy structure
                elif "issue" in parsed_data and isinstance(parsed_data["issue"], dict):
                    # Add the main issue
                    raw_results = [parsed_data["issue"]]
                    logger.info(f"Extracted main issue with key: {parsed_data['issue'].get('key')}")
                    
                    # Process hierarchy if available
                    if "hierarchy" in parsed_data and isinstance(parsed_data["hierarchy"], dict):
                        # Add parent if available
                        if "parent" in parsed_data["hierarchy"] and parsed_data["hierarchy"]["parent"]:
                            raw_results.append(parsed_data["hierarchy"]["parent"])
                            logger.info(f"Added parent issue: {parsed_data['hierarchy']['parent'].get('key')}")
                        
                        # Add children/subtasks if available
                        for field in ["children", "subtasks"]:
                            if field in parsed_data["hierarchy"] and isinstance(parsed_data["hierarchy"][field], list):
                                for item in parsed_data["hierarchy"][field]:
                                    if isinstance(item, dict) and "key" in item:
                                        raw_results.append(item)
                                logger.info(f"Added {len(parsed_data['hierarchy'][field])} {field}")
            
            # Handle case where response is directly an array of issues
            elif isinstance(parsed_data, list):
                raw_results = parsed_data
                logger.info(f"Parsed direct list of {len(raw_results)} Jira issues")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Jira string response as JSON: {e}")
    
    # If we're dealing with a dictionary, check common structures
    elif isinstance(raw_results, dict):
        # Extract issues array if present
        if "issues" in raw_results and isinstance(raw_results["issues"], list):
            raw_results = raw_results["issues"]
            logger.info(f"Extracted {len(raw_results)} issues from dictionary response")
        else:
            # Single issue response
            raw_results = [raw_results]
            logger.info("Converted single issue dictionary to list")
    
    # Process incoming results to ensure they have required fields and structure
    processed_results = []
    unique_keys = set()
    
    logger.info(f"Processing {len(raw_results)} raw Jira items")
    
    for item in raw_results:
        # Skip non-dict items
        if not isinstance(item, dict):
            continue
            
        # Ensure each item has a key
        key = item.get("key")
        if not key:
            # Check if key is in fields
            if "fields" in item and isinstance(item["fields"], dict):
                key = item["fields"].get("key")
                
            # Check if we have an id to use as fallback
            if not key and "id" in item:
                key = f"JIRA-{item['id']}"
                item["key"] = key  # Add key to item
                
        # Skip if still no key or already processed
        if not key or key in unique_keys:
            continue
            
        unique_keys.add(key)
        
        # Make sure fields is a dict if present
        if "fields" in item and not isinstance(item["fields"], dict):
            item["fields"] = {"raw_data": str(item["fields"])}
            
        # Make sure we have at least a basic fields dict
        if "fields" not in item:
            item["fields"] = {}
            
        # Normalize fields that might be at top level
        for field in ["summary", "description", "status", "priority", "assignee", "reporter", "created", "updated"]:
            if field in item and field not in item["fields"]:
                item["fields"][field] = item[field]
                
        processed_results.append(item)
    
    logger.info(f"Processed {len(processed_results)} unique Jira issues")
    
    # Store the processed results
    return {
        "raw_jira_results": processed_results
    }
