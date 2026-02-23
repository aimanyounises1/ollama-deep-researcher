# src/assistant/tools/tool_perforce.py
import asyncio
import logging
import os  # Import os module
import re
import ssl
from datetime import datetime
from typing import Dict, Optional, List, Any
import json

import requests
from langsmith import traceable
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.assistant.tools.PerforceTool import PerforceHelper  # Import the helper class
from src.utils.ssl_fix import with_ssl_disabled, patch_ssl_for_huggingface

# Setup logging
logger = logging.getLogger(__name__)

# Handle SSL verification disable if needed
if os.environ.get('DISABLE_SSL_VERIFICATION', 'False').lower() in ('true', '1', 't'):
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

# Regular expressions for finding MTV IDs and JIRA IDs in search queries
MTV_PATTERN = r'\b(MTV\d+)\b'
JIRA_PATTERN = r'[A-Z]+-\d+'

# Set verify to False if environment variable is set
verify_ssl = os.environ.get('SSL_VERIFY', 'true').lower() != 'false'


@traceable
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(
        (requests.exceptions.RequestException, ConnectionError, ssl.SSLError, asyncio.TimeoutError)
        # Added TimeoutError
    )
)
@with_ssl_disabled
async def search_perforce(query: str, extract_api_details: bool = False, max_results: int = 50, config: Optional[dict] = None) -> str:
    """
    Search Perforce for changes matching a keyword or MTV number with enhanced API and business logic extraction.
    
    Args:
        query (str): Text to search for (keyword, MTV number, etc.)
        extract_api_details (bool): Whether to perform deep API analysis on code changes
        max_results (int): Maximum number of changelists to return
        config (Optional[dict]): Additional configuration
        
    Returns:
        JSON string of changelist dictionaries with enhanced code context
    """
    from src.assistant.tools.PerforceTool import PerforceHelper
    
    results = []
    
    # Pre-process query to detect API-related searches
    is_api_search = any(term in query.lower() for term in ['api', 'endpoint', 'service', 'interface', 'rest', 'soap', 'graphql', 'rpc'])
    
    # If API-specific search is detected, enable advanced extraction even if not explicitly requested
    if is_api_search and not extract_api_details:
        extract_api_details = True
        logging.info(f"API-related search detected in query: '{query}'. Enabling advanced API extraction.")
    
    try:
        # Initialize Perforce helper
        helper = PerforceHelper()
        
        # Check if query looks like an MTV number
        mtv_match = re.match(r'(?i)(?:MTV[-\s]?)?(\d{4,})', query)
        
        if mtv_match:
            # Extract the number part
            mtv_number = mtv_match.group(1)
            mtv_formatted = f"MTV{mtv_number}"
            
            logging.info(f"MTV number detected ({mtv_formatted}), searching Perforce changes")
            
            # Get changes related to MTV
            try:
                # Use asynchronous MTV search
                raw_results = await helper.get_mtv_changes(mtv_formatted)
                logging.info(f"Found {len(raw_results)} changes for {mtv_formatted}")
            except Exception as e:
                logging.error(f"Error in async MTV search: {e}, falling back to sync method")
                # Fallback to synchronous search
                raw_results = helper._search_mtv_changes_sync(mtv_formatted)
        else:
            # Regular keyword search
            logging.info(f"Performing keyword search in Perforce: '{query}'")
            raw_results = helper.search_changelists_by_keyword(query, max_results_override=max_results)
            
        logging.info(f"Raw search returned {len(raw_results)} results")
        
        # Process results in parallel to improve performance
        enhanced_results = []
        
        # Define a task to fetch and enhance each result
        async def fetch_and_enhance(cl_num_str: str):
            try:
                cl_num = str(cl_num_str)
                result = next((r for r in raw_results if str(r.get('change', r.get('changelist', ''))) == cl_num), None)
                
                if not result:
                    logging.warning(f"Could not find result data for CL {cl_num} in raw results")
                    return None
                
                # Enhance with code snippets and analysis
                enhanced = await enhance_result_with_api_context(result, helper, extract_api_details)
                return enhanced
            except Exception as e:
                logging.error(f"Error enhancing CL {cl_num_str}: {e}")
                return result  # Return original result on error
        
        # Extract CL numbers for parallel processing
        cl_numbers = []
        for result in raw_results:
            cl_num = result.get('change', result.get('changelist', None))
            if cl_num:
                cl_numbers.append(str(cl_num))
        
        # Process in batches to avoid overwhelming the Perforce server
        BATCH_SIZE = 5
        for i in range(0, len(cl_numbers), BATCH_SIZE):
            batch = cl_numbers[i:i+BATCH_SIZE]
            tasks = [fetch_and_enhance(cl_num) for cl_num in batch]
            batch_results = await asyncio.gather(*tasks)
            
            # Filter out None results and add to enhanced_results
            enhanced_results.extend([r for r in batch_results if r])
            
            logging.info(f"Processed batch {i//BATCH_SIZE + 1}/{(len(cl_numbers) + BATCH_SIZE - 1)//BATCH_SIZE} ({len(batch)} CLs)")
            
        # Filter out None results and duplicates
        seen_cls = set()
        results = []
        for result in enhanced_results:
            cl_num = result.get('change', result.get('changelist', None))
            if cl_num and cl_num not in seen_cls:
                seen_cls.add(cl_num)
                results.append(result)
        
        # Standard log line for global tracking (used by run_graph_3 DataTrackingLogFilter)
        logging.info(f"Found {len(results)} Perforce changes for query: {query}")

        logging.info(f"Final result count after enhancement and deduplication: {len(results)}")

        # Align with other assistant tools (JIRA / Confluence) which return **JSON strings**.
        # Returning plain Python objects causes LangSmith to display them as raw unicode
        # escapes and prevents downstream chains from treating the output as text.  To
        # keep behaviour consistent across all tools we serialise the final results
        # to a JSON string *once* here.  This also guarantees the payload is
        # serialisable and avoids accidental binary data leaks.

        try:
            formatted = json.dumps(results, indent=2, default=str)
        except (TypeError, ValueError) as serialise_err:
            # Fallback – should never happen because we already coerce complex
            # objects inside `enhance_result_with_api_context`, but we handle just
            # in case.
            logger.warning("Non-serialisable object encountered when dumping Perforce results: %s", serialise_err)
            # Best effort: convert un-serialisable items to string representation.
            def _safe(o):
                try:
                    json.dumps(o)
                    return o
                except Exception:
                    return str(o)

            safe_results = [_safe(r) for r in results]
            formatted = json.dumps(safe_results, indent=2, default=str)

        return formatted
    
    except Exception as e:
        logging.error(f"Error in search_perforce: {e}")
        # Try to safely clean up resources
        try:
            if 'helper' in locals() and helper:
                helper.disconnect_all()
        except Exception as cleanup_err:
            logging.error(f"Error during cleanup: {cleanup_err}")

        # On error we still need to return a text payload to avoid downstream
        # type errors.
        try:
            return json.dumps(results, indent=2, default=str)
        except Exception:
            # As a last resort return an explanatory string.
            return "[]"  # empty JSON array to signal no results

def handle_content_line(line):
    """Handle both string and bytes content lines by decoding if necessary."""
    if isinstance(line, bytes):
        return line.decode('utf-8', errors='replace')
    return line

async def enhance_result_with_api_context(result: Dict[str, Any], p4: PerforceHelper, extract_api_details: bool = False) -> Dict[str, Any]:
    """
    Enhanced version that extracts API details from code changes including parameters,
    routes, business logic from comments, and correlates with documentation references.
    
    Args:
        result: The changelist result to enhance
        p4: PerforceHelper instance
        extract_api_details: Whether to perform deep API analysis
        
    Returns:
        Enhanced result dictionary with API context
    """
    if not isinstance(result, dict):
        return result
    
    # Get the changelist number
    cl_num = str(result.get('change', result.get('changelist', '')))
    
    # Skip if no CL number or already enhanced
    if not cl_num or result.get('enhanced', False):
        return result
    
    # Add API context details to each file in the changelist
    files = result.get('files', [])
    for file_idx, file_info in enumerate(files):
        if not isinstance(file_info, dict):
            continue
        
        file_path = file_info.get('depot_file', '')
        
        # Skip if no file path
        if not file_path:
            continue
            
        # Try to get the diff for this file
        try:
            logger.info(f"Getting diff for {file_path} in CL {cl_num}")
            diff_output = await p4.run_p4_async(['diff', '-du', f'{file_path}@{cl_num}'])
            
            # Process the diff output
            if diff_output:
                # Extract the diff content
                diff_content = ""
                for line in diff_output:
                    line_content = handle_content_line(line)
                    diff_content += line_content + "\n"
                
                # Store the diff
                file_info['diff'] = diff_content[:15000]  # Limit size
                
        except Exception as e:
            # Log warning and attempt fallback if diff fails
            logger.warning(f"Describe diff failed for {file_path} in CL {cl_num} due to likely incorrect argument (file path used instead of CL?): {str(e)}")
            try:
                # Fallback: try to get the full file content
                logger.warning(f"Attempting fallback: getting full file content for {file_path}@{cl_num}")
                file_content = await p4.run_p4_async(['print', f'{file_path}@{cl_num}'])
                
                content_text = ""
                for line in file_content:
                    line_content = handle_content_line(line)
                    logger.warning(f"Processed content line: {line_content[:100]}...")
                    content_text += line_content + "\n"
                
                # Store the full file content instead of diff
                file_info['content'] = content_text[:15000]  # Limit size
                logger.info(f"Successfully retrieved full content for {file_path}@{cl_num} after diff failure.")
                
            except Exception as inner_e:
                logger.error(f"Fallback also failed for {file_path}@{cl_num}: {str(inner_e)}")
    
    # Mark as enhanced
    result['enhanced'] = True
    return result

def extract_api_details(file_details: Dict[str, Any], language: str) -> Dict[str, Any]:
    """
    Extract API-specific details from file content:
    - Endpoint definitions
    - Parameter information
    - Business logic comments
    - Documentation references
    
    Args:
        file_details: The file details including snippet
        language: The programming language
        
    Returns:
        Dictionary with API details extracted
    """
    if not file_details or 'snippet' not in file_details:
        return {"endpoints": [], "comments": []}
    
    snippet = file_details.get('snippet', '')
    file_path = file_details.get('file', '')
    
    endpoints = []
    comments = []
    
    # Language-specific extraction patterns
    if language in ['java', 'kotlin']:
        # Extract REST endpoints (Spring, JAX-RS)
        endpoint_patterns = [
            # Spring annotations
            r'@(GetMapping|PostMapping|PutMapping|DeleteMapping|PatchMapping)(?:\s*\(\s*(?:value\s*=)?\s*[\"\'](.*?)[\"\'])?\s*\)',
            # JAX-RS annotations
            r'@(GET|POST|PUT|DELETE|PATCH)(?:\s*\(\s*(?:path\s*=)?\s*[\"\'](.*?)[\"\'])?\s*\)'
        ]
        
        # Extract business logic comments
        comment_patterns = [
            r'/\*\*(.*?)\*/',  # JavaDoc comments
            r'// TODO:(.*?)(?:\n|$)',  # TODO comments
            r'// NOTE:(.*?)(?:\n|$)',  # NOTE comments
            r'// IMPORTANT:(.*?)(?:\n|$)'  # IMPORTANT comments
        ]
        
    elif language in ['python']:
        # Extract endpoints (Flask, FastAPI, Django)
        endpoint_patterns = [
            # Flask routes
            r'@app\.route\([\'\"](.*?)[\'\"](?:,\s*methods=\[(.*?)\])?\)',
            r'@blueprint\.route\([\'\"](.*?)[\'\"](?:,\s*methods=\[(.*?)\])?\)',
            # FastAPI
            r'@app\.(get|post|put|delete|patch)\([\'\"](.*?)[\'\"]',
            # Django URL patterns
            r'path\([\'\"](.*?)[\'\"],\s*views\.(.*?),'
        ]
        
        # Extract business logic comments
        comment_patterns = [
            r'"""(.*?)"""',  # Docstrings
            r"'''(.*?)'''",  # Alternative docstrings
            r'# TODO:(.*?)(?:\n|$)',  # TODO comments
            r'# NOTE:(.*?)(?:\n|$)'  # NOTE comments
        ]
        
    elif language in ['javascript', 'typescript']:
        # Extract endpoints (Express, React Router, etc.)
        endpoint_patterns = [
            # Express routes
            r'(app|router)\.(get|post|put|delete|patch)\([\'\"](.*?)[\'\"]',
            # React Router
            r'<Route\s+path=[\'\"](.*?)[\'\"]',
            # Next.js API routes (by filename)
            r'pages/api/(.*?)\.(js|ts)$'
        ]
        
        # Extract business logic comments
        comment_patterns = [
            r'/\*\*(.*?)\*/',  # JSDoc comments
            r'// TODO:(.*?)(?:\n|$)',  # TODO comments
            r'// NOTE:(.*?)(?:\n|$)',  # NOTE comments
            r'// @ts-ignore(.*?)(?:\n|$)'  # TypeScript comments
        ]
        
    else:
        # Generic patterns for other languages
        endpoint_patterns = [
            r'route\([\'\"](.*?)[\'\"]',
            r'endpoint[\'\"](.*?)[\'\"]',
            r'api[\/\\\.]+(.*?)[\'\"]'
        ]
        
        comment_patterns = [
            r'/\*(.*?)\*/',  # Block comments
            r'(?://|#)(.*?)(?:\n|$)'  # Line comments
        ]
    
    # Extract endpoints
    for pattern in endpoint_patterns:
        matches = re.findall(pattern, snippet, re.DOTALL | re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                # Handle tuple results from regex groups
                if len(match) >= 2:
                    method = match[0]
                    path = match[1]
                else:
                    method = "unknown"
                    path = match[0]
            else:
                # Handle string results
                method = "unknown"
                path = match
                
            # Clean up the path
            path = path.strip()
            if not path:
                continue
                
            endpoints.append({
                "method": method,
                "path": path,
                "file": file_path
            })
    
    # Extract business logic comments
    for pattern in comment_patterns:
        matches = re.findall(pattern, snippet, re.DOTALL)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]  # Get first group
                
            comment_text = match.strip()
            if not comment_text or len(comment_text) < 10:
                continue  # Skip very short comments
                
            # Look for business logic indicators in comments
            if any(term in comment_text.lower() for term in [
                'business logic', 'validation', 'rule', 'policy', 'requirement', 
                'must', 'should', 'check', 'ensure', 'verify'
            ]):
                comments.append({
                    "text": comment_text,
                    "file": file_path
                })
    
    return {
        "endpoints": endpoints,
        "comments": comments
    }

def get_file_details_with_snippets(p4, cl_num, file_path, file_info, language):
    """Enhanced version with better snippet extraction and error handling"""
    # Existing implementation with improvements for readability and error handling
    if not file_path or not file_info:
        return None
        
    # Get action type
    action = file_info.get('action', '')
    
    # Initialize result
    result = {
        'file': file_path,
        'action': action,
        'language': language
    }
    
    try:
        # For added or edited files, get the content
        if action in ['add', 'edit', 'integrate', 'branch']:
            # Try to get the diff for edit actions
            if action == 'edit':
                diff_content = p4.get_diff_from_describe(cl_num, file_path)
                if diff_content:
                    result['snippet'] = diff_content
                    result['source'] = 'diff'
                    return result
            
            # If no diff or not an edit, try to get the file content
            file_content = None
            try:
                # For new files, get the content at the CL revision
                revision = f"{cl_num}"
                file_content = p4.get_file_content(file_path, revision)
            except Exception as e:
                logging.warning(f"Error getting content for {file_path}@{cl_num}: {e}")
                
            if file_content:
                # Limit the content size to avoid memory issues
                MAX_SNIPPET_SIZE = 10000
                if len(file_content) > MAX_SNIPPET_SIZE:
                    # Try to find a logical cut point (e.g., after a function or class)
                    cut_points = [
                        "}",  # End of a block
                        "\n\n",  # Double newline
                        "\n  }",  # Indented block end
                        ";",  # End of statement
                        ".\n"  # End of sentence
                    ]
                    
                    cut_index = MAX_SNIPPET_SIZE
                    for point in cut_points:
                        # Find the last occurrence of the cut point before the max size
                        last_point = file_content[:MAX_SNIPPET_SIZE].rfind(point)
                        if last_point > 0 and last_point < cut_index:
                            cut_index = last_point + len(point)
                            
                    # Use the cut point if found, otherwise use max size
                    truncated_content = file_content[:cut_index] + "\n\n... [CONTENT TRUNCATED] ..."
                    result['snippet'] = truncated_content
                else:
                    result['snippet'] = file_content
                    
                result['source'] = 'file'
                return result
        
        # For deleted files, handle differently
        elif action == 'delete':
            # Try to get previous version
            try:
                prev_revision = f"{int(cl_num) - 1}"
                file_content = p4.get_file_content(file_path, prev_revision)
                if file_content:
                    # Truncate if needed
                    if len(file_content) > 5000:
                        file_content = file_content[:5000] + "\n\n... [CONTENT TRUNCATED] ..."
                    
                    result['snippet'] = file_content
                    result['source'] = 'previous_version'
                    return result
            except Exception as e:
                logging.warning(f"Error getting previous content for deleted file {file_path}: {e}")
            
            # If we can't get content, just note that it was deleted
            result['snippet'] = f"[File {file_path} was deleted in this changelist]"
            result['source'] = 'info_only'
            return result
            
        # For other actions or if all else fails
        result['snippet'] = f"[Could not retrieve content for {file_path} with action '{action}']"
        result['source'] = 'none'
        return result
        
    except Exception as e:
        logging.error(f"Error getting details for {file_path} in CL {cl_num}: {e}")
        result['snippet'] = f"[Error retrieving content: {str(e)}]"
        result['source'] = 'error'
        return result

def generate_code_summary(files_with_snippets):
    """
    Generate a summary of code changes with enhanced API and business logic focus.
    
    Args:
        files_with_snippets: List of file dictionaries with snippets
        
    Returns:
        Dictionary summarizing the code changes
    """
    if not files_with_snippets:
        return {
            "files_changed": 0,
            "languages": [],
            "added_lines": 0,
            "removed_lines": 0,
            "api_related": False,
            "summary": "No code changes found."
        }
    
    # Count files by language
    languages = {}
    actions = {}
    added_lines = 0
    removed_lines = 0
    total_lines = 0
    api_endpoints_found = 0
    business_logic_changes = 0
    
    # Patterns to detect api endpoints
    api_patterns = {
        "java": [r'@(Get|Post|Put|Delete|Patch)Mapping', r'@(GET|POST|PUT|DELETE|PATCH)'],
        "python": [r'@app\.route', r'@blueprint\.route', r'@app\.(get|post|put|delete)'],
        "javascript": [r'app\.(get|post|put|delete)', r'router\.(get|post|put|delete)'],
        "typescript": [r'app\.(get|post|put|delete)', r'router\.(get|post|put|delete)'],
        "csharp": [r'\[Http(Get|Post|Put|Delete)\]', r'app\.Map(Get|Post|Put|Delete)']
    }
    
    # Patterns to detect business logic
    logic_patterns = [
        r'validate', r'business logic', r'rule', r'policy', r'requirement', 
        r'calcul', r'comput', r'determin', r'authorization'
    ]
    
    for file in files_with_snippets:
        language = file.get('language', 'unknown')
        action = file.get('action', 'unknown')
        snippet = file.get('snippet', '')
        
        # Update language stats
        languages[language] = languages.get(language, 0) + 1
        
        # Update action stats
        actions[action] = actions.get(action, 0) + 1
        
        # Count lines of code
        if snippet:
            lines = snippet.count('\n') + 1
            total_lines += lines
            
            # Estimate added/removed lines
            if action == 'add':
                added_lines += lines
            elif action == 'edit':
                # For edits, try to count diff markers
                added = len(re.findall(r'^\+[^+]', snippet, re.MULTILINE))
                removed = len(re.findall(r'^-[^-]', snippet, re.MULTILINE))
                added_lines += added
                removed_lines += removed
            elif action == 'delete':
                removed_lines += lines
            
            # Check for API endpoints
            for pattern in api_patterns.get(language, []) + api_patterns.get('any', []):
                if re.search(pattern, snippet, re.IGNORECASE):
                    api_endpoints_found += 1
                    break
            
            # Check for business logic
            for pattern in logic_patterns:
                if re.search(pattern, snippet, re.IGNORECASE):
                    business_logic_changes += 1
                    break
    
    # Generate summary
    api_related = api_endpoints_found > 0
    has_business_logic = business_logic_changes > 0
    
    # Construct the summary text
    summary_parts = []
    
    # Add file count
    file_count = len(files_with_snippets)
    summary_parts.append(f"Changed {file_count} file{'s' if file_count != 1 else ''}")
    
    # Add language information
    if languages:
        lang_list = [f"{count} {lang}" for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True)]
        summary_parts.append(f"using {', '.join(lang_list)}")
    
    # Add line counts
    if added_lines > 0 or removed_lines > 0:
        summary_parts.append(f"with {added_lines} line{'s' if added_lines != 1 else ''} added and {removed_lines} line{'s' if removed_lines != 1 else ''} removed")
    
    # Add API information
    if api_related:
        summary_parts.append(f"including {api_endpoints_found} API endpoint{'s' if api_endpoints_found != 1 else ''}")
    
    # Add business logic information
    if has_business_logic:
        summary_parts.append(f"containing business logic changes in {business_logic_changes} location{'s' if business_logic_changes != 1 else ''}")
    
    # Join all parts
    summary_text = " ".join(summary_parts) + "."
    
    return {
        "files_changed": file_count,
        "languages": list(languages.keys()),
        "actions": actions,
        "added_lines": added_lines,
        "removed_lines": removed_lines,
        "total_lines": total_lines,
        "api_related": api_related,
        "api_endpoints_found": api_endpoints_found,
        "business_logic_changes": business_logic_changes,
        "summary": summary_text
    }

# --- Example Main Usage ---
async def main_async():
    # --- ADDED: Configure logging for DEBUG level ---
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s')
    # --- END ADDED ---
    helper = None
    try:
        helper = PerforceHelper()
        async with helper.async_session():
            print("Connected to Perforce server (async context)")
            # --- Use relevant test query ---
            test_query = "MTV2005"  # Example keyword search
            # test_query = "27239429" # Example direct CL search
            print(f"\nAsync searching for query: '{test_query}'...")
            # --- Call the main search function ---
            results = await search_perforce(test_query)
            print(f"Found {len(results)} results for query '{test_query}'")

            if results:
                # --- Updated print logic to handle potentially enhanced results ---
                for i, result_data in enumerate(results[:5]):  # Limit output for demo
                    if isinstance(result_data, dict):
                        cl = result_data.get('change')
                        print("-" * 50)
                        print(f"CL: {cl}")
                        print(f"User: {result_data.get('user')}")
                        print(f"Date: {result_data.get('dateFormatted')}")
                        print(f"Description: {result_data.get('desc', '')[:200]}...")
                        if result_data.get('url'):
                            print(f"Swarm Link: {result_data.get('url')}")

                        files = result_data.get('files', [])
                        if files:
                            print(f"Files ({len(files)}):")
                            for f_info in files[:3]:
                                print(f"  - {f_info.get('path')} ({f_info.get('action')})")
                            if len(files) > 3: print("    ...")

                        snippets = result_data.get('code_snippets')
                        if snippets:
                            print("Code Snippets:")
                            for snip_info in snippets[:2]:  # Show first 2 snippets per CL
                                print(f"  File: {snip_info.get('file')}")
                                print(f"  Language: {snip_info.get('language')}")
                                print(f"  Action: {snip_info.get('action')}")
                                snippet_text = snip_info.get('snippet', '')[:200]
                                print(f"  ```\n{snippet_text}...\n  ```")
                        else:
                            print("  No code snippets found or extracted for this CL.")
                        print("-" * 50)
                    elif isinstance(result_data, str):  # Handle error string case
                        print(f"Error in result: {result_data}")

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
