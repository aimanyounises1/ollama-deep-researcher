"""
Tool Output Verification component that validates data coming from integration tools
before it is used for response generation.
"""

import asyncio
import logging
import re
import base64
import hashlib
from typing import Dict, List, Any, Optional, Union, Set, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolOutputVerifier:
    """
    Verifies outputs from external tools (Perforce, Jira, Confluence, etc.)
    to ensure they are valid and properly structured before being used in LLM reasoning.
    
    This helps prevent hallucinations by ensuring the model only works with
    validated external data sources.
    """
    
    def __init__(
        self, 
        llm: BaseChatModel,
        schema_validation: bool = True,
        content_validation: bool = True,
        reference_tracking: bool = True,
        enable_multimodal: bool = True,
        enable_statistical_validation: bool = True,
        enable_cross_tool_validation: bool = False
    ):
        """
        Initialize the tool output verifier.
        
        Args:
            llm: Language model to use for content validation
            schema_validation: Whether to validate output schema
            content_validation: Whether to validate content semantics
            reference_tracking: Whether to track references to tool outputs
            enable_multimodal: Whether to enable validation for non-text content
            enable_statistical_validation: Whether to validate numeric values statistically
            enable_cross_tool_validation: Whether to validate outputs across different tools
        """
        self.llm = llm
        self.schema_validation = schema_validation
        self.content_validation = content_validation
        self.reference_tracking = reference_tracking
        self.enable_multimodal = enable_multimodal
        self.enable_statistical_validation = enable_statistical_validation
        self.enable_cross_tool_validation = enable_cross_tool_validation
        self.reference_map = {}  # Maps reference IDs to source data
        self.tool_stats = {}  # Stores statistical information about tool outputs
    
    async def verify_tool_output(
        self,
        tool_name: str,
        tool_output: Any,
        expected_schema: Optional[Dict[str, Any]] = None,
        related_outputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Verify output from a tool to ensure it's valid.
        
        Args:
            tool_name: Name of the tool that generated the output
            tool_output: The output data from the tool
            expected_schema: Optional schema for validation
            related_outputs: Optional outputs from related tools for cross-validation
            
        Returns:
            Dictionary with verification results
        """
        verification_results = {
            "tool_name": tool_name,
            "is_valid": True,
            "issues": [],
            "reference_id": None,
            "modified_output": tool_output,
            "content_types": self._detect_content_types(tool_output)
        }
        
        # Schema validation
        if self.schema_validation and expected_schema:
            schema_issues = self._validate_schema(tool_output, expected_schema)
            if schema_issues:
                verification_results["issues"].extend(schema_issues)
                verification_results["is_valid"] = False
        
        # Content validation based on content type
        if self.content_validation:
            for content_type in verification_results["content_types"]:
                if content_type == "text":
                    text_issues = await self._validate_text_content(tool_name, tool_output)
                    if text_issues:
                        verification_results["issues"].extend(text_issues)
                elif content_type == "code" and self.enable_multimodal:
                    code_issues = await self._validate_code_content(tool_output)
                    if code_issues:
                        verification_results["issues"].extend(code_issues)
                elif content_type == "image" and self.enable_multimodal:
                    image_issues = await self._validate_image_content(tool_output)
                    if image_issues:
                        verification_results["issues"].extend(image_issues)
                elif content_type == "numeric" and self.enable_statistical_validation:
                    numeric_issues = self._validate_numeric_content(tool_name, tool_output)
                    if numeric_issues:
                        verification_results["issues"].extend(numeric_issues)
            
            # Mark as invalid if any high-severity issues
            if any(issue.get("severity") == "high" for issue in verification_results["issues"]):
                verification_results["is_valid"] = False
        
        # Cross-tool validation if enabled
        if self.enable_cross_tool_validation and related_outputs:
            cross_tool_issues = await self._validate_across_tools(tool_name, tool_output, related_outputs)
            if cross_tool_issues:
                verification_results["issues"].extend(cross_tool_issues)
                # Cross-tool issues don't necessarily invalidate the output
        
        # Reference tracking
        if self.reference_tracking:
            reference_id = self._generate_reference_id(tool_name, tool_output)
            verification_results["reference_id"] = reference_id
            self.reference_map[reference_id] = {
                "tool_name": tool_name,
                "output_summary": self._summarize_output(tool_output),
                "content_types": verification_results["content_types"],
                "timestamp": asyncio.get_event_loop().time()
            }
        
        # Update tool statistics
        self._update_tool_stats(tool_name, tool_output)
        
        # If there were issues, try to fix the output
        if not verification_results["is_valid"]:
            try:
                modified_output = await self._fix_output(tool_output, verification_results["issues"])
                verification_results["modified_output"] = modified_output
                verification_results["is_valid"] = True
                verification_results["was_fixed"] = True
            except Exception as e:
                logger.error(f"Error fixing tool output: {e}")
                verification_results["was_fixed"] = False
        
        return verification_results
    
    def _detect_content_types(self, output: Any) -> List[str]:
        """Detect content types present in the output."""
        content_types = []
        
        if isinstance(output, dict) or isinstance(output, list):
            # Check for text content
            content_types.append("text")
            
            # Check for code
            if self._contains_code(output):
                content_types.append("code")
            
            # Check for image data
            if self._contains_image(output):
                content_types.append("image")
            
            # Check for numeric data
            if self._contains_numeric(output):
                content_types.append("numeric")
        else:
            # Default to text for simple types
            content_types.append("text")
        
        return content_types
    
    def _contains_code(self, output: Any) -> bool:
        """Check if output contains code snippets."""
        if isinstance(output, str):
            # Simple heuristic - look for common code patterns
            code_patterns = [
                r"```[a-z]*\n[\s\S]*?\n```",  # Markdown code blocks
                r"def\s+\w+\s*\(",  # Python function definitions
                r"function\s+\w+\s*\(",  # JavaScript function definitions
                r"class\s+\w+",  # Class definitions
                r"import\s+[\w\s,]+;",  # Java/TypeScript imports
                r"#include\s+[<\"][\w\.]+[>\"]"  # C/C++ includes
            ]
            for pattern in code_patterns:
                if re.search(pattern, output):
                    return True
        elif isinstance(output, dict):
            # Check if any value contains code
            for key, value in output.items():
                if key in ["code", "snippet", "source"] or (isinstance(value, str) and self._contains_code(value)):
                    return True
        elif isinstance(output, list):
            # Check if any list item contains code
            for item in output:
                if self._contains_code(item):
                    return True
        return False
    
    def _contains_image(self, output: Any) -> bool:
        """Check if output contains image data."""
        if isinstance(output, dict):
            # Check for common image-related keys
            for key in output:
                if key in ["image", "img", "image_data", "screenshot", "thumbnail"]:
                    return True
                # Check for base64 encoded images
                if isinstance(output[key], str) and output[key].startswith(("data:image", "iVBORw0K")):
                    return True
        elif isinstance(output, list):
            # Check if any list item contains image data
            for item in output:
                if self._contains_image(item):
                    return True
        return False
    
    def _contains_numeric(self, output: Any) -> bool:
        """Check if output contains significant numeric data."""
        numeric_count = 0
        
        def count_numerics(obj):
            nonlocal numeric_count
            if isinstance(obj, (int, float)):
                numeric_count += 1
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    count_numerics(value)
            elif isinstance(obj, list):
                for item in obj:
                    count_numerics(item)
        
        count_numerics(output)
        return numeric_count >= 3  # Consider it numeric data if at least 3 numeric values
    
    def _validate_schema(
        self, 
        tool_output: Any, 
        expected_schema: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Validate the schema of tool output.
        
        Args:
            tool_output: The output data from the tool
            expected_schema: Expected schema for validation
            
        Returns:
            List of schema validation issues
        """
        issues = []
        
        # Convert to dict if not already
        if not isinstance(tool_output, dict) and not isinstance(tool_output, list):
            return [{
                "type": "schema_error",
                "message": f"Expected dict or list, got {type(tool_output)}",
                "severity": "high"
            }]
        
        # Handle list case
        if isinstance(tool_output, list):
            if isinstance(expected_schema.get("items"), dict):
                for i, item in enumerate(tool_output):
                    item_issues = self._validate_schema(item, expected_schema["items"])
                    for issue in item_issues:
                        issue["item_index"] = i
                        issues.append(issue)
            return issues
        
        # Check required fields
        for field in expected_schema.get("required", []):
            if field not in tool_output:
                issues.append({
                    "type": "missing_field",
                    "field": field,
                    "message": f"Required field '{field}' is missing",
                    "severity": "high"
                })
        
        # Check field types
        for field, field_schema in expected_schema.get("properties", {}).items():
            if field in tool_output:
                field_type = field_schema.get("type")
                if field_type and not self._check_type(tool_output[field], field_type):
                    issues.append({
                        "type": "type_error",
                        "field": field,
                        "expected_type": field_type,
                        "actual_type": type(tool_output[field]).__name__,
                        "message": f"Field '{field}' has wrong type: expected {field_type}, got {type(tool_output[field]).__name__}",
                        "severity": "medium"
                    })
                
                # Additional validation for format if specified
                if field_type == "string" and "format" in field_schema:
                    format_issues = self._validate_string_format(
                        tool_output[field], 
                        field_schema["format"], 
                        field
                    )
                    issues.extend(format_issues)
        
        return issues
    
    def _validate_string_format(
        self, 
        value: str, 
        format_type: str, 
        field_name: str
    ) -> List[Dict[str, Any]]:
        """Validate string against a specified format."""
        if not isinstance(value, str):
            return []
            
        issues = []
        
        if format_type == "email":
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, value):
                issues.append({
                    "type": "format_error",
                    "field": field_name,
                    "message": f"Field '{field_name}' is not a valid email address",
                    "severity": "medium"
                })
        elif format_type == "uri":
            uri_pattern = r'^(https?|ftp)://[^\s/$.?#].[^\s]*$'
            if not re.match(uri_pattern, value):
                issues.append({
                    "type": "format_error",
                    "field": field_name,
                    "message": f"Field '{field_name}' is not a valid URI",
                    "severity": "medium"
                })
        elif format_type == "date":
            date_pattern = r'^\d{4}-\d{2}-\d{2}$'
            if not re.match(date_pattern, value):
                issues.append({
                    "type": "format_error",
                    "field": field_name,
                    "message": f"Field '{field_name}' is not a valid ISO date (YYYY-MM-DD)",
                    "severity": "medium"
                })
        elif format_type == "date-time":
            # ISO 8601 date-time format
            datetime_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?$'
            if not re.match(datetime_pattern, value):
                issues.append({
                    "type": "format_error",
                    "field": field_name,
                    "message": f"Field '{field_name}' is not a valid ISO date-time",
                    "severity": "medium"
                })
        
        return issues
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if a value matches the expected type."""
        if expected_type == "string":
            return isinstance(value, str)
        elif expected_type == "number":
            return isinstance(value, (int, float))
        elif expected_type == "integer":
            return isinstance(value, int)
        elif expected_type == "boolean":
            return isinstance(value, bool)
        elif expected_type == "array":
            return isinstance(value, list)
        elif expected_type == "object":
            return isinstance(value, dict)
        return True  # Unknown type
    
    async def _validate_text_content(
        self, 
        tool_name: str, 
        tool_output: Any
    ) -> List[Dict[str, Any]]:
        """
        Validate the semantic content of tool output.
        
        Args:
            tool_name: Name of the tool that generated the output
            tool_output: The output data from the tool
            
        Returns:
            List of content validation issues
        """
        # For complex tools like Perforce, Jira, Confluence, we use LLM to check content
        if tool_name in ["perforce", "jira", "confluence", "web_search"]:
            return await self._llm_validate_content(tool_name, tool_output)
        
        # For simpler tools, do basic checks
        issues = []
        
        # Check for empty or null values in important fields
        if isinstance(tool_output, dict):
            for key, value in tool_output.items():
                if value is None or (isinstance(value, str) and not value.strip()):
                    issues.append({
                        "type": "empty_value",
                        "field": key,
                        "message": f"Field '{key}' has empty value",
                        "severity": "low"
                    })
        
        return issues
    
    async def _validate_code_content(self, tool_output: Any) -> List[Dict[str, Any]]:
        """Validate code content from tool output."""
        code_snippets = self._extract_code_snippets(tool_output)
        if not code_snippets:
            return []
            
        issues = []
        
        # Process each code snippet with LLM
        for i, (code, language) in enumerate(code_snippets):
            code_prompt = ChatPromptTemplate.from_template(
                "You are an expert code validator. Review this {language} code snippet and identify any issues:\n\n"
                "```{language}\n{code}\n```\n\n"
                "Check for these potential problems:\n"
                "1. Syntax errors\n"
                "2. Logic errors\n"
                "3. Security vulnerabilities\n"
                "4. Best practice violations\n"
                "5. Potential runtime errors\n\n"
                "Format your response as a JSON list of issues, each with:\n"
                "- 'type': issue type\n"
                "- 'line': line number/location of issue (if applicable)\n"
                "- 'message': detailed explanation\n"
                "- 'severity': 'high', 'medium', or 'low'\n\n"
                "If there are no issues, return an empty list []"
            )
            
            validate_chain = code_prompt | self.llm.with_structured_output(
                List[Dict[str, str]]
            )
            
            try:
                code_issues = await validate_chain.ainvoke({
                    "language": language or "code",
                    "code": code
                })
                
                # Add snippet index to each issue
                for issue in code_issues:
                    issue["snippet_index"] = i
                    issue["type"] = f"code_{issue.get('type', 'issue')}"
                
                issues.extend(code_issues)
            except Exception as e:
                logger.error(f"Error validating code snippet: {e}")
                issues.append({
                    "type": "code_validation_error",
                    "snippet_index": i,
                    "message": f"Error validating code: {str(e)}",
                    "severity": "medium"
                })
        
        return issues
    
    def _extract_code_snippets(self, tool_output: Any) -> List[Tuple[str, Optional[str]]]:
        """Extract code snippets from tool output with language info."""
        snippets = []
        
        def extract_from_str(text):
            # Extract markdown code blocks with language info
            markdown_pattern = r'```([a-z]*)\n([\s\S]*?)\n```'
            for match in re.finditer(markdown_pattern, text):
                language = match.group(1) or None
                code = match.group(2)
                snippets.append((code, language))
        
        def process_item(item):
            if isinstance(item, str):
                extract_from_str(item)
            elif isinstance(item, dict):
                # Check for explicit code fields
                for key, value in item.items():
                    if key in ["code", "snippet", "source"] and isinstance(value, str):
                        language = item.get("language") or None
                        snippets.append((value, language))
                    elif isinstance(value, (dict, list, str)):
                        process_item(value)
            elif isinstance(item, list):
                for subitem in item:
                    process_item(subitem)
        
        process_item(tool_output)
        return snippets
    
    async def _validate_image_content(self, tool_output: Any) -> List[Dict[str, Any]]:
        """Validate image content from tool output."""
        # For now, just check that image data is present and valid
        # A more sophisticated implementation would pass the image to a vision model
        issues = []
        
        image_data = self._extract_image_data(tool_output)
        for i, img_data in enumerate(image_data):
            if not img_data or len(img_data) < 100:
                issues.append({
                    "type": "invalid_image_data",
                    "image_index": i,
                    "message": "Image data is missing or too small to be valid",
                    "severity": "medium"
                })
        
        return issues
    
    def _extract_image_data(self, tool_output: Any) -> List[str]:
        """Extract image data from tool output."""
        images = []
        
        def process_item(item):
            if isinstance(item, str) and (item.startswith("data:image") or item.startswith("iVBORw0K")):
                images.append(item)
            elif isinstance(item, dict):
                for key, value in item.items():
                    if key in ["image", "img", "image_data", "screenshot", "thumbnail"]:
                        if isinstance(value, str):
                            images.append(value)
                    elif isinstance(value, (dict, list)):
                        process_item(value)
            elif isinstance(item, list):
                for subitem in item:
                    process_item(subitem)
        
        process_item(tool_output)
        return images
    
    def _validate_numeric_content(self, tool_name: str, tool_output: Any) -> List[Dict[str, Any]]:
        """Validate numeric values in tool output against statistical norms."""
        if not self.enable_statistical_validation:
            return []
            
        issues = []
        
        # Extract numeric values with their context
        numeric_values = self._extract_numeric_values(tool_output)
        
        # Check against stats if we have them
        if tool_name in self.tool_stats:
            tool_stats = self.tool_stats[tool_name]
            
            for field, value in numeric_values:
                if field in tool_stats:
                    field_stats = tool_stats[field]
                    min_val = field_stats.get("min")
                    max_val = field_stats.get("max")
                    mean = field_stats.get("mean")
                    stddev = field_stats.get("stddev")
                    
                    # Check for outliers (beyond 3 standard deviations)
                    if mean is not None and stddev is not None and stddev > 0:
                        z_score = abs(value - mean) / stddev
                        if z_score > 3:
                            issues.append({
                                "type": "statistical_outlier",
                                "field": field,
                                "value": value,
                                "mean": mean,
                                "stddev": stddev,
                                "z_score": z_score,
                                "message": f"Value {value} for field '{field}' is a statistical outlier (z-score: {z_score:.2f})",
                                "severity": "medium"
                            })
                    
                    # Check against known min/max
                    if min_val is not None and value < min_val:
                        issues.append({
                            "type": "value_below_minimum",
                            "field": field,
                            "value": value,
                            "min": min_val,
                            "message": f"Value {value} for field '{field}' is below the minimum observed value {min_val}",
                            "severity": "low"
                        })
                    
                    if max_val is not None and value > max_val:
                        issues.append({
                            "type": "value_above_maximum",
                            "field": field,
                            "value": value,
                            "max": max_val,
                            "message": f"Value {value} for field '{field}' is above the maximum observed value {max_val}",
                            "severity": "low"
                        })
        
        return issues
    
    def _extract_numeric_values(self, tool_output: Any) -> List[Tuple[str, float]]:
        """Extract numeric values with their field names from tool output."""
        numeric_values = []
        
        def extract_from_dict(obj, prefix=""):
            for key, value in obj.items():
                path = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (int, float)):
                    numeric_values.append((path, float(value)))
                elif isinstance(value, dict):
                    extract_from_dict(value, path)
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            extract_from_dict(item, f"{path}[{i}]")
                        elif isinstance(item, (int, float)):
                            numeric_values.append((f"{path}[{i}]", float(item)))
        
        if isinstance(tool_output, dict):
            extract_from_dict(tool_output)
        elif isinstance(tool_output, list):
            for i, item in enumerate(tool_output):
                if isinstance(item, dict):
                    extract_from_dict(item, f"[{i}]")
                elif isinstance(item, (int, float)):
                    numeric_values.append((f"[{i}]", float(item)))
        
        return numeric_values
    
    def _update_tool_stats(self, tool_name: str, tool_output: Any):
        """Update statistical information about tool outputs."""
        numeric_values = self._extract_numeric_values(tool_output)
        
        if tool_name not in self.tool_stats:
            self.tool_stats[tool_name] = {}
            
        for field, value in numeric_values:
            if field not in self.tool_stats[tool_name]:
                self.tool_stats[tool_name][field] = {
                    "count": 0,
                    "sum": 0,
                    "sum_squares": 0,
                    "min": value,
                    "max": value
                }
                
            stats = self.tool_stats[tool_name][field]
            stats["count"] += 1
            stats["sum"] += value
            stats["sum_squares"] += value * value
            stats["min"] = min(stats["min"], value)
            stats["max"] = max(stats["max"], value)
            
            # Update mean and stddev
            if stats["count"] > 0:
                stats["mean"] = stats["sum"] / stats["count"]
                
                if stats["count"] > 1:
                    variance = (stats["sum_squares"] - (stats["sum"] ** 2) / stats["count"]) / (stats["count"] - 1)
                    stats["stddev"] = max(0, variance) ** 0.5
                else:
                    stats["stddev"] = 0
    
    async def _validate_across_tools(
        self, 
        tool_name: str, 
        tool_output: Any,
        related_outputs: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Validate consistency across outputs from different tools."""
        if not self.enable_cross_tool_validation or not related_outputs:
            return []
            
        issues = []
        
        # Convert all outputs to text for LLM validation
        tool_output_str = self._output_to_string(tool_output)
        related_outputs_str = "\n\n".join([
            f"{name}:\n{self._output_to_string(output)}"
            for name, output in related_outputs.items()
        ])
        
        # Define prompt for cross-tool validation
        cross_validate_prompt = ChatPromptTemplate.from_template(
            "You are an expert in validating consistency across different data sources. "
            "Check whether the following output from {tool_name} is consistent with outputs from related tools:\n\n"
            "Output from {tool_name}:\n{output}\n\n"
            "Outputs from related tools:\n{related_outputs}\n\n"
            "Check for these potential inconsistencies:\n"
            "1. Conflicting information\n"
            "2. Mismatched identifiers or references\n"
            "3. Temporal inconsistencies (different timestamps for same events)\n"
            "4. Different values for the same properties\n\n"
            "Format your response as a JSON list of issues, each with:\n"
            "- 'type': issue type (e.g., 'conflicting_information')\n"
            "- 'message': detailed explanation of the inconsistency\n"
            "- 'severity': 'high', 'medium', or 'low'\n"
            "- 'affected_tools': list of tool names involved in the inconsistency\n\n"
            "If there are no inconsistencies, return an empty list []"
        )
        
        # Process with LLM
        cross_validate_chain = cross_validate_prompt | self.llm.with_structured_output(
            List[Dict[str, Any]]
        )
        
        try:
            cross_tool_issues = await cross_validate_chain.ainvoke({
                "tool_name": tool_name,
                "output": tool_output_str,
                "related_outputs": related_outputs_str
            })
            
            # Add cross-tool prefix to issue types
            for issue in cross_tool_issues:
                issue["type"] = f"cross_tool_{issue.get('type', 'inconsistency')}"
            
            issues.extend(cross_tool_issues)
        except Exception as e:
            logger.error(f"Error during cross-tool validation: {e}")
            issues.append({
                "type": "cross_tool_validation_error",
                "message": f"Error during cross-tool validation: {str(e)}",
                "severity": "medium",
                "affected_tools": [tool_name] + list(related_outputs.keys())
            })
        
        return issues
    
    async def _llm_validate_content(
        self, 
        tool_name: str, 
        tool_output: Any
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to validate tool output content.
        
        Args:
            tool_name: Name of the tool that generated the output
            tool_output: The output data from the tool
            
        Returns:
            List of content validation issues
        """
        # Convert output to string for LLM processing
        output_str = self._output_to_string(tool_output)
        
        # Define prompt for content validation
        validate_prompt = ChatPromptTemplate.from_template(
            "You are an expert validator for {tool_name} data. Review this output and identify any issues:\n\n"
            "{output}\n\n"
            "Check for these potential problems:\n"
            "1. Missing or incomplete information\n"
            "2. Inconsistent data\n"
            "3. Suspicious or unusual values\n"
            "4. Data that seems fabricated or hallucinated\n"
            "5. Formatting problems\n\n"
            "Format your response as a JSON list of issues, each with:\n"
            "- 'type': issue type\n"
            "- 'field': field/location of issue (if applicable)\n"
            "- 'message': detailed explanation\n"
            "- 'severity': 'high', 'medium', or 'low'\n\n"
            "If there are no issues, return an empty list []"
        )
        
        # Process with LLM
        validate_chain = validate_prompt | self.llm.with_structured_output(
            List[Dict[str, str]]
        )
        
        try:
            issues = await validate_chain.ainvoke({
                "tool_name": tool_name,
                "output": output_str
            })
            return issues
        except Exception as e:
            logger.error(f"Error validating content with LLM: {e}")
            return [{
                "type": "validation_error",
                "message": f"Error during validation: {str(e)}",
                "severity": "medium"
            }]
    
    def _output_to_string(self, output: Any) -> str:
        """Convert output to string format for LLM processing."""
        import json
        
        try:
            if isinstance(output, str):
                return output
            else:
                return json.dumps(output, indent=2)
        except Exception:
            return str(output)
    
    def _summarize_output(self, output: Any) -> str:
        """Create a brief summary of output for reference tracking."""
        if isinstance(output, dict):
            # Get top-level keys
            keys = list(output.keys())
            return f"Dict with keys: {', '.join(keys[:5])}" + ("..." if len(keys) > 5 else "")
        elif isinstance(output, list):
            return f"List with {len(output)} items"
        else:
            return str(output)[:100] + ("..." if len(str(output)) > 100 else "")
    
    def _generate_reference_id(self, tool_name: str, output: Any) -> str:
        """Generate a reference ID for tracking."""
        import hashlib
        import time
        
        # Create a hash based on tool name, timestamp, and a sample of the output
        content = f"{tool_name}_{time.time()}_{self._summarize_output(output)}"
        hash_obj = hashlib.md5(content.encode())
        return f"tool_{tool_name}_{hash_obj.hexdigest()[:8]}"
    
    async def _fix_output(
        self, 
        output: Any, 
        issues: List[Dict[str, Any]]
    ) -> Any:
        """
        Try to fix problematic tool output.
        
        Args:
            output: The original tool output
            issues: List of identified issues
            
        Returns:
            Fixed output if possible
        """
        # Group issues by type for targeted fixing
        issue_types = set(issue["type"] for issue in issues)
        
        # For basic schema issues, we can do simple fixes
        if all(issue_type in ["missing_field", "empty_value", "type_error"] for issue_type in issue_types):
            fixed_output = output.copy() if isinstance(output, (dict, list)) else output
            
            if isinstance(fixed_output, dict):
                # Fix missing or empty fields
                for issue in issues:
                    if issue["type"] in ["missing_field", "empty_value"]:
                        field = issue.get("field")
                        if field:
                            fixed_output[field] = "[No data available]"
                    elif issue["type"] == "type_error":
                        field = issue.get("field")
                        expected_type = issue.get("expected_type")
                        if field and expected_type:
                            # Try to convert the value to the expected type
                            try:
                                if expected_type == "string":
                                    fixed_output[field] = str(fixed_output[field])
                                elif expected_type == "number":
                                    fixed_output[field] = float(fixed_output[field])
                                elif expected_type == "integer":
                                    fixed_output[field] = int(float(fixed_output[field]))
                                elif expected_type == "boolean":
                                    fixed_output[field] = bool(fixed_output[field])
                                elif expected_type == "array" and not isinstance(fixed_output[field], list):
                                    fixed_output[field] = [fixed_output[field]]
                                elif expected_type == "object" and not isinstance(fixed_output[field], dict):
                                    fixed_output[field] = {"value": fixed_output[field]}
                            except (ValueError, TypeError):
                                # If conversion fails, use a placeholder
                                if expected_type == "string":
                                    fixed_output[field] = "[Invalid string]"
                                elif expected_type in ["number", "integer"]:
                                    fixed_output[field] = 0
                                elif expected_type == "boolean":
                                    fixed_output[field] = False
                                elif expected_type == "array":
                                    fixed_output[field] = []
                                elif expected_type == "object":
                                    fixed_output[field] = {}
            
            return fixed_output
        
        # For code issues, try to fix simple syntax problems
        if any(issue["type"].startswith("code_") for issue in issues):
            code_issues = [issue for issue in issues if issue["type"].startswith("code_")]
            if len(code_issues) > 0:
                fixed_output = await self._fix_code_issues(output, code_issues)
                # If we fixed code issues but other issues remain, continue with LLM fixing
                if fixed_output != output:
                    remaining_issues = [issue for issue in issues if not issue["type"].startswith("code_")]
                    if not remaining_issues:
                        return fixed_output
                    # Otherwise, proceed with LLM fixing using the partially fixed output
                    output = fixed_output
        
        # For more complex issues, use LLM to fix
        output_str = self._output_to_string(output)
        issues_str = "\n".join([f"- {issue['message']} (severity: {issue['severity']})" for issue in issues])
        
        # Define prompt for fixing output
        fix_prompt = ChatPromptTemplate.from_template(
            "Fix the following tool output that has these issues:\n\n"
            "ISSUES:\n{issues}\n\n"
            "ORIGINAL OUTPUT:\n{output}\n\n"
            "Return a fixed version of the output that addresses the issues. "
            "For missing information, use placeholder text clearly marked as unavailable data. "
            "Keep the same structure and format as the original."
        )
        
        # Process with LLM
        fix_chain = fix_prompt | self.llm | StrOutputParser()
        
        try:
            fixed_output_str = await fix_chain.ainvoke({
                "issues": issues_str,
                "output": output_str
            })
            
            # Convert back to original format if possible
            import json
            try:
                if isinstance(output, dict) or isinstance(output, list):
                    return json.loads(fixed_output_str)
                else:
                    return fixed_output_str
            except:
                return fixed_output_str
        except Exception as e:
            logger.error(f"Error fixing output with LLM: {e}")
            raise
    
    async def _fix_code_issues(self, output: Any, code_issues: List[Dict[str, Any]]) -> Any:
        """Fix code issues in the output."""
        # Extract all code snippets
        snippets = self._extract_code_snippets(output)
        if not snippets:
            return output
            
        # Group issues by snippet index
        issues_by_snippet = {}
        for issue in code_issues:
            snippet_idx = issue.get("snippet_index", 0)
            if snippet_idx not in issues_by_snippet:
                issues_by_snippet[snippet_idx] = []
            issues_by_snippet[snippet_idx].append(issue)
        
        # Fix each snippet with issues
        fixed_snippets = {}
        for idx, issues in issues_by_snippet.items():
            if idx >= len(snippets):
                continue
                
            code, language = snippets[idx]
            issues_str = "\n".join([f"- Line {issue.get('line', '?')}: {issue['message']}" for issue in issues])
            
            fix_code_prompt = ChatPromptTemplate.from_template(
                "Fix the following {language} code that has these issues:\n\n"
                "ISSUES:\n{issues}\n\n"
                "CODE:\n```{language}\n{code}\n```\n\n"
                "Return only the fixed code without any explanations or markdown formatting."
            )
            
            fix_code_chain = fix_code_prompt | self.llm | StrOutputParser()
            
            try:
                fixed_code = await fix_code_chain.ainvoke({
                    "language": language or "code",
                    "issues": issues_str,
                    "code": code
                })
                
                fixed_snippets[idx] = (fixed_code, language)
            except Exception as e:
                logger.error(f"Error fixing code snippet {idx}: {e}")
                # Keep original if fixing fails
                fixed_snippets[idx] = snippets[idx]
        
        # Replace snippets in the output
        return self._replace_code_snippets(output, fixed_snippets)
    
    def _replace_code_snippets(self, output: Any, fixed_snippets: Dict[int, Tuple[str, Optional[str]]]) -> Any:
        """Replace code snippets in the output with fixed versions."""
        if not fixed_snippets:
            return output
            
        # For simple string output with markdown code blocks
        if isinstance(output, str):
            result = output
            # Extract and replace markdown code blocks
            markdown_pattern = r'```([a-z]*)\n([\s\S]*?)\n```'
            matches = list(re.finditer(markdown_pattern, output))
            
            # Replace from end to start to avoid offset issues
            for i, match in reversed(list(enumerate(matches))):
                if i in fixed_snippets:
                    fixed_code, lang = fixed_snippets[i]
                    lang_str = lang or match.group(1) or ""
                    replacement = f"```{lang_str}\n{fixed_code}\n```"
                    start, end = match.span()
                    result = result[:start] + replacement + result[end:]
            
            return result
        
        # For dictionary output
        elif isinstance(output, dict):
            result = output.copy()
            
            # Track encountered snippets to match with the fixed ones
            snippet_count = 0
            
            def process_dict(d):
                nonlocal snippet_count
                for key, value in d.items():
                    if key in ["code", "snippet", "source"] and isinstance(value, str):
                        if snippet_count in fixed_snippets:
                            d[key] = fixed_snippets[snippet_count][0]
                        snippet_count += 1
                    elif isinstance(value, dict):
                        process_dict(value)
                    elif isinstance(value, list):
                        process_list(value)
                    elif isinstance(value, str) and ("```" in value):
                        # String with potential code blocks
                        d[key] = self._replace_code_snippets(value, fixed_snippets)
            
            def process_list(lst):
                nonlocal snippet_count
                for i, item in enumerate(lst):
                    if isinstance(item, dict):
                        process_dict(item)
                    elif isinstance(item, list):
                        process_list(item)
                    elif isinstance(item, str) and ("```" in item):
                        # String with potential code blocks
                        lst[i] = self._replace_code_snippets(item, fixed_snippets)
            
            process_dict(result)
            return result
        
        # For list output
        elif isinstance(output, list):
            result = output.copy()
            
            # Track encountered snippets to match with the fixed ones
            snippet_count = 0
            
            for i, item in enumerate(result):
                if isinstance(item, dict):
                    # Process dictionary items
                    for key, value in item.items():
                        if key in ["code", "snippet", "source"] and isinstance(value, str):
                            if snippet_count in fixed_snippets:
                                item[key] = fixed_snippets[snippet_count][0]
                            snippet_count += 1
                elif isinstance(item, str) and ("```" in item):
                    # String with potential code blocks
                    result[i] = self._replace_code_snippets(item, fixed_snippets)
            
            return result
        
        # For other types, return as is
        return output
    
    def get_reference_info(self, reference_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a referenced tool output.
        
        Args:
            reference_id: The reference ID to look up
            
        Returns:
            Reference information if found
        """
        return self.reference_map.get(reference_id)
    
    def get_all_references(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all tracked references.
        
        Returns:
            Dictionary of all reference information
        """
        return self.reference_map
    
    def get_tool_statistics(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistical information about tool outputs.
        
        Args:
            tool_name: Optional tool name to filter statistics
            
        Returns:
            Dictionary of tool statistics
        """
        if tool_name:
            return self.tool_stats.get(tool_name, {})
        return self.tool_stats 