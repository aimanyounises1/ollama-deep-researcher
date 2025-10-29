"""
Response Formatter utility that provides standardized formatting to reduce hallucination risk.

By clearly delineating between facts, opinions, and confidence levels,
this formatter helps both users and models distinguish between different types
of content, reducing the risk of presenting speculative information as factual.
"""

import re
import json
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from enum import Enum
import logging
import base64

logger = logging.getLogger(__name__)

class EpistemicStatus(str, Enum):
    """Epistemic status of a statement."""
    VERIFIED = "verified"
    PROBABLE = "probable"
    SPECULATIVE = "speculative"
    UNVERIFIED = "unverified"
    DISPUTED = "disputed"

class ResponseFormatter:
    """
    Formats responses with clear delineation between facts and opinions
    to reduce hallucination risk.
    """
    
    def __init__(
        self,
        include_confidence: bool = True,
        include_citations: bool = True,
        highlight_uncertainty: bool = True,
        clear_sections: bool = True,
        use_epistemic_tags: bool = True,
        use_color_coding: bool = False,
        show_source_quality: bool = True,
        markdown_format: bool = True
    ):
        """
        Initialize the response formatter.
        
        Args:
            include_confidence: Whether to include confidence ratings
            include_citations: Whether to include citation formatting
            highlight_uncertainty: Whether to highlight uncertainty markers
            clear_sections: Whether to use clear section delineation
            use_epistemic_tags: Whether to include epistemic status tags
            use_color_coding: Whether to use color coding for different confidence levels
            show_source_quality: Whether to show information about source quality
            markdown_format: Whether to use markdown formatting
        """
        self.include_confidence = include_confidence
        self.include_citations = include_citations
        self.highlight_uncertainty = highlight_uncertainty
        self.clear_sections = clear_sections
        self.use_epistemic_tags = use_epistemic_tags
        self.use_color_coding = use_color_coding
        self.show_source_quality = show_source_quality
        self.markdown_format = markdown_format
        
        # Define color scheme if using colors
        self.colors = {
            "high_confidence": "green",
            "medium_confidence": "blue",
            "low_confidence": "orange",
            "unverified": "red",
            "citation": "purple",
            "uncertainty": "magenta"
        }
    
    def format_response(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Format a response with anti-hallucination formatting.
        
        Args:
            content: The content to format
            metadata: Optional metadata about the content (verification results, etc.)
            
        Returns:
            Formatted content with anti-hallucination features
        """
        if not content:
            return ""
        
        formatted = content
        metadata = metadata or {}
        
        # Add section headers if content is long enough
        if self.clear_sections and len(content) > 500:
            formatted = self._add_section_headers(formatted)
        
        # Apply epistemic status tagging if enabled
        if self.use_epistemic_tags and "epistemic_statuses" in metadata:
            formatted = self._add_epistemic_tags(formatted, metadata["epistemic_statuses"])
        
        # Format citations if present and enabled
        if self.include_citations:
            formatted = self._format_citations(formatted, metadata.get("citations", {}))
        
        # Add confidence indicators if available
        if self.include_confidence and "confidence" in metadata:
            formatted = self._add_confidence_indicator(formatted, metadata["confidence"])
        
        # Highlight uncertainty markers
        if self.highlight_uncertainty:
            formatted = self._highlight_uncertainty(formatted, metadata.get("uncertainty_markers", {}))
        
        # Add source quality information if available and enabled
        if self.show_source_quality and "source_quality" in metadata:
            formatted = self._add_source_quality_info(formatted, metadata["source_quality"])
        
        # Wrap in a formatted structure
        if self.clear_sections:
            formatted = self._wrap_in_structure(formatted, metadata)
        
        return formatted
    
    def _add_section_headers(self, content: str) -> str:
        """Add clear section headers to content if not already present."""
        # Check if content already has headers
        if re.search(r'^#+\s+', content, re.MULTILINE):
            return content
        
        # Try to identify natural sections
        paragraphs = content.split('\n\n')
        if len(paragraphs) <= 2:
            return content
        
        # Identify potential section headers
        result = []
        current_section = []
        
        for para in paragraphs:
            if len(para.strip()) > 0 and len(para.strip()) < 100 and para.strip()[-1] not in '.,:;?!':
                # This could be a header
                if current_section:
                    result.append('\n\n'.join(current_section))
                    current_section = []
                result.append(f"## {para.strip()}")
            else:
                current_section.append(para)
        
        if current_section:
            result.append('\n\n'.join(current_section))
        
        return '\n\n'.join(result)
    
    def _add_epistemic_tags(self, content: str, epistemic_statuses: Dict[str, str]) -> str:
        """Add epistemic status tags to statements based on provided mapping."""
        if not epistemic_statuses:
            return content
            
        formatted = content
        
        # Replace each statement with a tagged version
        for statement, status in epistemic_statuses.items():
            # Skip very short statements to avoid false positives
            if len(statement) < 10:
                continue
                
            # Create tag based on status
            status_tag = self._create_epistemic_tag(status)
            
            # Replace the statement with tagged version
            # Use word boundaries to avoid partial matches
            pattern = re.escape(statement)
            formatted = re.sub(
                f"({pattern})", 
                f"\\1 {status_tag}", 
                formatted
            )
        
        # Add legend at the end if using epistemic tags
        legend = "\n\n**Epistemic Status Legend:**\n"
        legend += "- ✓ [Verified]: Confirmed by reliable sources\n"
        legend += "- ⓟ [Probable]: Likely correct based on evidence\n"
        legend += "- ? [Speculative]: Reasonable conjecture\n"
        legend += "- ⚠ [Unverified]: Not yet verified\n"
        legend += "- ✗ [Disputed]: Contested by other sources"
        
        if "Epistemic Status Legend" not in formatted:
            formatted += legend
        
        return formatted
    
    def _create_epistemic_tag(self, status: str) -> str:
        """Create an epistemic status tag based on the status."""
        if status.lower() == EpistemicStatus.VERIFIED:
            return "✓ [Verified]" if not self.use_color_coding else self._colorize("✓ [Verified]", "high_confidence")
        elif status.lower() == EpistemicStatus.PROBABLE:
            return "ⓟ [Probable]" if not self.use_color_coding else self._colorize("ⓟ [Probable]", "medium_confidence")
        elif status.lower() == EpistemicStatus.SPECULATIVE:
            return "? [Speculative]" if not self.use_color_coding else self._colorize("? [Speculative]", "low_confidence")
        elif status.lower() == EpistemicStatus.UNVERIFIED:
            return "⚠ [Unverified]" if not self.use_color_coding else self._colorize("⚠ [Unverified]", "unverified")
        elif status.lower() == EpistemicStatus.DISPUTED:
            return "✗ [Disputed]" if not self.use_color_coding else self._colorize("✗ [Disputed]", "unverified")
        else:
            return f"[{status}]"
    
    def _colorize(self, text: str, color_key: str) -> str:
        """Add color formatting to text based on color key."""
        if not self.use_color_coding:
            return text
            
        color = self.colors.get(color_key, "black")
        
        if self.markdown_format:
            # Using markdown doesn't support colors directly, so we use alternative formatting
            if color == "green":
                return f"**{text}**"
            elif color == "red":
                return f"*{text}*"
            elif color == "blue":
                return f"`{text}`"
            else:
                return text
        else:
            # If we're in an environment that supports ANSI colors
            color_codes = {
                "black": "\033[30m",
                "red": "\033[31m",
                "green": "\033[32m",
                "orange": "\033[33m",
                "blue": "\033[34m",
                "magenta": "\033[35m",
                "purple": "\033[35m",
                "cyan": "\033[36m",
                "white": "\033[37m",
                "reset": "\033[0m"
            }
            return f"{color_codes.get(color, color_codes['reset'])}{text}{color_codes['reset']}"
    
    def _format_citations(self, content: str, citations: Dict[str, Any]) -> str:
        """Format citations in the content with improved styling and source quality info."""
        # Look for citation patterns like [1], [2], etc.
        citation_pattern = r'\[(\d+)\]'
        
        def replace_citation(match):
            citation_num = match.group(1)
            citation_info = citations.get(citation_num)
            
            if citation_info:
                source = citation_info.get('source', 'Unknown source')
                quality = citation_info.get('quality', None)
                quality_indicator = ""
                
                if quality and self.show_source_quality:
                    if quality >= 0.8:
                        quality_indicator = " [High Quality]"
                    elif quality >= 0.5:
                        quality_indicator = " [Medium Quality]"
                    else:
                        quality_indicator = " [Low Quality]"
                
                citation_text = f"[{citation_num}: {source}{quality_indicator}]"
                return self._colorize(citation_text, "citation") if self.use_color_coding else citation_text
            return match.group(0)
        
        # Replace citations
        formatted = re.sub(citation_pattern, replace_citation, content)
        
        # Add citation section at the end if citations exist and aren't already in the content
        if citations and not re.search(r'##\s+References|##\s+Citations', content, re.IGNORECASE):
            citation_section = "\n\n## References\n"
            for num, info in sorted(citations.items(), key=lambda x: int(x[0])):
                source = info.get('source', 'Unknown source')
                url = info.get('url', '')
                year = info.get('year', '')
                quality = info.get('quality', None)
                quality_indicator = ""
                
                if quality and self.show_source_quality:
                    if quality >= 0.8:
                        quality_indicator = " [High Quality]"
                    elif quality >= 0.5:
                        quality_indicator = " [Medium Quality]"
                    else:
                        quality_indicator = " [Low Quality]"
                
                citation_entry = f"[{num}] {source}"
                if year:
                    citation_entry += f" ({year})"
                if quality_indicator:
                    citation_entry += quality_indicator
                if url:
                    citation_entry += f" - {url}"
                
                citation_section += citation_entry + "\n"
            
            formatted += citation_section
        
        return formatted
    
    def _add_confidence_indicator(self, content: str, confidence: float) -> str:
        """Add confidence indicator to the content with visual representation."""
        conf_text = ""
        conf_visual = ""
        
        if confidence >= 0.9:
            conf_text = "HIGH CONFIDENCE"
            conf_visual = "█████" if self.markdown_format else "■■■■■"
        elif confidence >= 0.7:
            conf_text = "MEDIUM CONFIDENCE"
            conf_visual = "████▒" if self.markdown_format else "■■■■□"
        elif confidence >= 0.5:
            conf_text = "MODERATE CONFIDENCE"
            conf_visual = "███▒▒" if self.markdown_format else "■■■□□"
        elif confidence >= 0.3:
            conf_text = "LOW CONFIDENCE"
            conf_visual = "██▒▒▒" if self.markdown_format else "■■□□□"
        else:
            conf_text = "VERY LOW CONFIDENCE"
            conf_visual = "█▒▒▒▒" if self.markdown_format else "■□□□□"
        
        confidence_header = f"CONFIDENCE LEVEL: {conf_text} ({confidence:.2f}) {conf_visual}\n\n"
        
        if self.use_color_coding:
            color_key = "high_confidence" if confidence >= 0.8 else "medium_confidence" if confidence >= 0.5 else "low_confidence"
            confidence_header = self._colorize(confidence_header, color_key)
        
        return confidence_header + content
    
    def _highlight_uncertainty(self, content: str, uncertainty_markers: Dict[str, List[str]]) -> str:
        """Highlight uncertainty markers in the content with improved visual indicators."""
        if not uncertainty_markers:
            return content
        
        formatted = content
        
        # Collect all uncertainty phrases
        all_phrases = []
        for marker_type, phrases in uncertainty_markers.items():
            all_phrases.extend(phrases)
        
        # Sort by length (descending) to avoid partial replacements
        all_phrases.sort(key=len, reverse=True)
        
        # Replace each phrase with highlighted version
        for phrase in all_phrases:
            if self.use_color_coding:
                replacement = self._colorize(f"[?{phrase}?]", "uncertainty")
            else:
                replacement = f"[?{phrase}?]"
            
            formatted = formatted.replace(phrase, replacement)
        
        # Add uncertainty types and count
        uncertainty_types = {
            "hedging": "Hedging language (might, perhaps, possibly)",
            "vague_quantifiers": "Vague quantifiers (some, many, few)",
            "modal_adverbs": "Modal adverbs (probably, likely, certainly)",
            "passive_voice": "Passive voice (avoiding attribution)",
            "subjunctive": "Subjunctive mood (would, could, should)"
        }
        
        total_markers = sum(len(markers) for markers in uncertainty_markers.values())
        uncertainty_counts = {k: len(v) for k, v in uncertainty_markers.items() if v}
        
        uncertainty_note = "\n\n---\n"
        uncertainty_note += f"*Uncertainty Analysis: {total_markers} markers detected*\n\n"
        
        if uncertainty_counts:
            uncertainty_note += "**Uncertainty Types:**\n"
            for marker_type, count in uncertainty_counts.items():
                description = uncertainty_types.get(marker_type, marker_type)
                uncertainty_note += f"- {description}: {count} instances\n"
        
        uncertainty_note += "\n*Note: Text with [?uncertainty markers?] indicates lower confidence statements.*"
        
        # Add uncertainty note if it doesn't already exist
        if "Uncertainty Analysis:" not in formatted:
            formatted += uncertainty_note
        
        return formatted
    
    def _add_source_quality_info(self, content: str, source_quality: Dict[str, Any]) -> str:
        """Add information about the quality of sources used."""
        if not source_quality:
            return content
            
        # Create a summary of source quality at the end
        quality_info = "\n\n## Source Quality Analysis\n\n"
        
        # Add overall quality score if available
        if "overall_score" in source_quality:
            score = source_quality["overall_score"]
            quality_info += f"**Overall Source Quality:** {score:.2f}/1.0\n\n"
        
        # Add source type distribution if available
        if "source_types" in source_quality:
            quality_info += "**Source Distribution:**\n"
            for source_type, percentage in source_quality["source_types"].items():
                quality_info += f"- {source_type}: {percentage:.1f}%\n"
        
        # Add source age information if available
        if "recency" in source_quality:
            quality_info += f"\n**Information Recency:** {source_quality['recency']}\n"
        
        # Make sure we're not duplicating this section
        if "Source Quality Analysis" not in content:
            return content + quality_info
        else:
            return content
    
    def _wrap_in_structure(self, content: str, metadata: Dict[str, Any]) -> str:
        """Wrap content in a structured format with enhanced verification status."""
        verification_status = ""
        if "is_verified" in metadata:
            is_verified = metadata["is_verified"]
            verification_status = "✓ VERIFIED" if is_verified else "⚠ UNVERIFIED"
            verification_status += f" (Confidence: {metadata.get('confidence', 0.0):.2f})"
        
        # Calculate knowledge quality indicator
        knowledge_quality = self._calculate_knowledge_quality(metadata)
        
        header = "="*80 + "\n"
        header += f"RESPONSE {verification_status}\n"
        if knowledge_quality:
            header += f"KNOWLEDGE QUALITY: {knowledge_quality}\n"
        header += "="*80 + "\n\n"
        
        footer = "\n" + "-"*80 + "\n"
        
        # Add any corrections if available
        if "corrections" in metadata and metadata["corrections"]:
            footer += "Potential inaccuracies detected:\n"
            for i, correction in enumerate(metadata["corrections"], 1):
                footer += f"{i}. {correction}\n"
        
        # Add hallucination score if available
        if "hallucination_score" in metadata:
            score = metadata["hallucination_score"]
            footer += f"\nHallucination likelihood assessment: {score:.2f}/1.0 (lower is better)\n"
        
        # Add epistemological distribution if available
        if "epistemological_distribution" in metadata:
            dist = metadata["epistemological_distribution"]
            if dist:
                footer += "\nKnowledge Distribution:\n"
                total = sum(dist.values())
                for status, count in dist.items():
                    if total > 0:
                        percentage = (count / total) * 100
                        footer += f"- {status.capitalize()}: {count} ({percentage:.1f}%)\n"
        
        footer += "-"*80
        
        return header + content + footer
    
    def _calculate_knowledge_quality(self, metadata: Dict[str, Any]) -> str:
        """Calculate an overall knowledge quality indicator based on metadata."""
        # Default to empty string
        quality = ""
        
        # Factors to consider
        confidence = metadata.get("confidence", 0.0)
        hallucination_score = metadata.get("hallucination_score", 0.5)
        source_quality = metadata.get("source_quality", {}).get("overall_score", 0.5)
        verifications_passed = metadata.get("verifications_passed", 0)
        verifications_total = metadata.get("verifications_total", 0)
        
        # Simple heuristic based on confidence and hallucination score
        quality_score = confidence * 0.4 + (1 - hallucination_score) * 0.4
        
        # Add source quality if available
        if "source_quality" in metadata:
            quality_score += source_quality * 0.2
        
        # Add verification ratio if available
        if verifications_total > 0:
            verification_ratio = verifications_passed / verifications_total
            quality_score = quality_score * 0.8 + verification_ratio * 0.2
        
        # Generate text label
        if quality_score >= 0.8:
            quality = "EXCELLENT"
        elif quality_score >= 0.6:
            quality = "GOOD"
        elif quality_score >= 0.4:
            quality = "FAIR"
        elif quality_score >= 0.2:
            quality = "POOR"
        else:
            quality = "VERY POOR"
        
        return quality
    
    @staticmethod
    def format_facts_vs_opinions(
        facts: List[str],
        opinions: List[str],
        context: Optional[str] = None,
        fact_sources: Optional[Dict[str, str]] = None,
        epistemic_statuses: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Format a clear separation between facts and opinions with enhanced source attribution.
        
        Args:
            facts: List of factual statements
            opinions: List of opinions or interpretations
            context: Optional context paragraph
            fact_sources: Optional mapping of facts to their sources
            epistemic_statuses: Optional mapping of facts to epistemic statuses
            
        Returns:
            Formatted content with clear separation
        """
        result = ""
        
        if context:
            result += f"{context}\n\n"
        
        if facts:
            result += "## Facts\n\n"
            for fact in facts:
                fact_text = f"- {fact}"
                
                # Add source if available
                if fact_sources and fact in fact_sources:
                    source = fact_sources[fact]
                    fact_text += f" (Source: {source})"
                
                # Add epistemic status if available
                if epistemic_statuses and fact in epistemic_statuses:
                    status = epistemic_statuses[fact]
                    # Create a simple tag based on status
                    if status == EpistemicStatus.VERIFIED:
                        fact_text += " ✓"
                    elif status == EpistemicStatus.PROBABLE:
                        fact_text += " ⓟ"
                    elif status == EpistemicStatus.SPECULATIVE:
                        fact_text += "?"
                    elif status == EpistemicStatus.DISPUTED:
                        fact_text += "✗"
                
                result += fact_text + "\n"
            result += "\n"
        
        if opinions:
            result += "## Analysis & Interpretations\n\n"
            for opinion in opinions:
                result += f"- {opinion}\n"
            result += "\n"
        
        result += "*Note: Facts are statements directly supported by evidence. Interpretations and analyses may involve judgment and inference.*"
        
        # Add epistemic status legend if used
        if epistemic_statuses:
            result += "\n\n**Symbols:**\n"
            result += "- ✓: Verified by reliable sources\n"
            result += "- ⓟ: Probable based on evidence\n"
            result += "- ?: Speculative or conjectural\n"
            result += "- ✗: Disputed or contradicted"
        
        return result
    
    @staticmethod
    def format_with_citations(
        content: str,
        citations: Dict[str, Dict[str, str]],
        show_quality: bool = True
    ) -> str:
        """
        Format content with academic-style citations.
        
        Args:
            content: The content to format
            citations: Dictionary of citation information
            show_quality: Whether to show source quality information
            
        Returns:
            Content with formatted citations
        """
        # Replace citation placeholders with formatted citations
        pattern = r'\[CITE:(\w+)\]'
        
        def citation_replacer(match):
            cite_key = match.group(1)
            if cite_key in citations:
                return f"[{cite_key}]"
            return match.group(0)
        
        formatted = re.sub(pattern, citation_replacer, content)
        
        # Add bibliography
        if citations:
            formatted += "\n\n## References\n\n"
            for key, info in citations.items():
                author = info.get('author', 'Unknown')
                title = info.get('title', 'Untitled')
                source = info.get('source', '')
                url = info.get('url', '')
                year = info.get('year', '')
                quality = info.get('quality', None)
                
                citation = f"[{key}] {author}. \"{title}\""
                if year:
                    citation += f" ({year})"
                if source:
                    citation += f", {source}"
                if show_quality and quality is not None:
                    quality_label = "High quality" if quality >= 0.8 else "Medium quality" if quality >= 0.5 else "Lower quality"
                    citation += f" [{quality_label}]"
                if url:
                    citation += f". Available at: {url}"
                
                formatted += citation + "\n"
        
        return formatted
    
    @staticmethod
    def format_uncertainty_analysis(
        text: str,
        uncertainty_scores: Dict[str, float],
        uncertainty_markers: Dict[str, List[str]]
    ) -> str:
        """
        Format a detailed analysis of uncertainty in text.
        
        Args:
            text: The text to analyze
            uncertainty_scores: Dictionary of uncertainty scores by category
            uncertainty_markers: Dictionary of uncertainty markers found in text
            
        Returns:
            Formatted uncertainty analysis
        """
        result = "# Uncertainty Analysis\n\n"
        
        # Overall uncertainty score
        total_score = sum(uncertainty_scores.values()) / len(uncertainty_scores) if uncertainty_scores else 0
        result += f"**Overall Uncertainty Score:** {total_score:.2f}/1.0\n\n"
        
        # Uncertainty by category
        if uncertainty_scores:
            result += "## Uncertainty by Category\n\n"
            for category, score in sorted(uncertainty_scores.items(), key=lambda x: x[1], reverse=True):
                result += f"- **{category.replace('_', ' ').title()}**: {score:.2f}/1.0\n"
            result += "\n"
        
        # Uncertainty markers found
        if uncertainty_markers:
            result += "## Uncertainty Markers Found\n\n"
            for category, markers in uncertainty_markers.items():
                if markers:
                    result += f"### {category.replace('_', ' ').title()}\n\n"
                    for marker in markers:
                        # Find the marker in context
                        context = ""
                        for line in text.split('\n'):
                            if marker in line:
                                # Extract context around the marker
                                index = line.find(marker)
                                start = max(0, index - 30)
                                end = min(len(line), index + len(marker) + 30)
                                context = f"...{line[start:end]}..." if start > 0 or end < len(line) else line[start:end]
                                break
                        
                        result += f"- \"{marker}\""
                        if context:
                            result += f" in context: \"{context}\"\n"
                        else:
                            result += "\n"
                    result += "\n"
        
        # Recommendations
        result += "## Recommendations\n\n"
        if total_score > 0.7:
            result += "- High uncertainty detected. Consider seeking additional sources or evidence.\n"
            result += "- Be cautious when making decisions based on this information.\n"
            result += "- Highlight the speculative nature when sharing this information.\n"
        elif total_score > 0.4:
            result += "- Moderate uncertainty detected. Supplement with additional sources where possible.\n"
            result += "- Acknowledge limitations when sharing this information.\n"
        else:
            result += "- Low uncertainty detected. Information appears to be presented with appropriate confidence.\n"
            result += "- Continue to verify critical facts with primary sources.\n"
        
        return result

def sanitize_binary_content(content: Any) -> Any:
    """
    Recursively sanitizes a data structure, replacing binary content with descriptive placeholders.
    Works on nested dicts, lists, and other JSON-serializable structures.
    
    Args:
        content: The content to sanitize (can be dict, list, string, etc.)
        
    Returns:
        The sanitized content with binary data replaced by descriptive placeholders
    """
    if isinstance(content, dict):
        # Process dictionary
        sanitized = {}
        for key, value in content.items():
            # Skip known binary fields or handle specially
            if key in ["content_text", "content_processed"] and isinstance(value, bytes):
                sanitized[key] = f"[Binary data: {len(value)} bytes]"
            elif key.endswith("_bytes") or key.startswith("binary_") or "image" in key.lower():
                if isinstance(value, bytes):
                    sanitized[key] = f"[Binary data: {len(value)} bytes]"
                else:
                    sanitized[key] = sanitize_binary_content(value)
            else:
                sanitized[key] = sanitize_binary_content(value)
        return sanitized
    
    elif isinstance(content, list):
        # Process list
        return [sanitize_binary_content(item) for item in content]
    
    elif isinstance(content, bytes):
        # Convert bytes to a descriptive string
        # Check if it's likely binary by examining first few bytes
        sample_size = min(100, len(content))
        binary_chars = 0
        for byte in content[:sample_size]:
            if byte < 32 and byte not in (9, 10, 13):  # Not tab, LF, or CR
                binary_chars += 1
                
        if binary_chars > sample_size * 0.1:
            return f"[Binary data: {len(content)} bytes]"
        else:
            try:
                # Try to decode as text if it doesn't look binary
                return content.decode('utf-8', errors='replace')
            except:
                return f"[Binary data: {len(content)} bytes]"
    
    elif isinstance(content, str):
        # Check if the string might contain base64 encoded binary data
        if len(content) > 100 and re.match(r'^[A-Za-z0-9+/]+={0,2}$', content):
            return f"[Possible encoded binary data: {len(content)} chars]"
            
        # Check if the string contains unusual or control characters
        control_chars = sum(1 for c in content[:100] if ord(c) < 32 and c not in '\t\n\r')
        if control_chars > 10:
            return f"[String with binary content: {len(content)} chars]"
            
        return content
    
    else:
        # Return other types unchanged
        return content

def format_tool_response(response: Any, tool_name: str) -> Dict[str, Any]:
    """
    Format and sanitize a response from a tool to ensure it's clean and consistent.
    
    Args:
        response: The raw response from the tool
        tool_name: The name of the tool that produced the response
        
    Returns:
        A sanitized and formatted response
    """
    try:
        # First sanitize to handle any binary content
        sanitized_response = sanitize_binary_content(response)
        
        # Convert to structured format with metadata
        formatted = {
            "tool": tool_name,
            "status": "success",
            "data": sanitized_response
        }
        
        return formatted
        
    except Exception as e:
        logger.error(f"Error formatting response from {tool_name}: {e}")
        return {
            "tool": tool_name,
            "status": "error",
            "error": str(e),
            "data": str(response)[:1000]  # Truncate to avoid massive error messages
        }

def format_error_response(error: Exception, tool_name: str) -> Dict[str, Any]:
    """
    Format an error response from a tool.
    
    Args:
        error: The exception that occurred
        tool_name: The name of the tool that produced the error
        
    Returns:
        A formatted error response
    """
    error_type = type(error).__name__
    error_message = str(error)
    
    return {
        "tool": tool_name,
        "status": "error",
        "error_type": error_type,
        "error": error_message
    }

def normalize_mtv_id(mtv_id: str) -> str:
    """
    Normalize MTV IDs to a standard format (MTV1234).
    
    Args:
        mtv_id: The MTV ID to normalize, which might be in different formats
        
    Returns:
        Normalized MTV ID
    """
    if not mtv_id:
        return ""
        
    # Extract digits, ignoring any prefix or separators
    digits = ''.join(filter(str.isdigit, mtv_id))
    
    # If we found digits, format as MTV followed by the digits
    if digits:
        return f"MTV{digits}"
    else:
        return mtv_id  # Return original if no digits found

def combine_source_results(jira_results: Optional[Dict] = None, 
                           perforce_results: Optional[Dict] = None,
                           confluence_results: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Combine and normalize results from different sources.
    
    Args:
        jira_results: Results from Jira
        perforce_results: Results from Perforce
        confluence_results: Results from Confluence
        
    Returns:
        Combined results in a structured format
    """
    combined = {
        "sources": {
            "jira": {
                "status": "not_executed",
                "data": None
            },
            "perforce": {
                "status": "not_executed",
                "data": None
            },
            "confluence": {
                "status": "not_executed",
                "data": None
            }
        },
        "cross_references": [],
        "timestamp": None
    }
    
    # Process Jira results
    if jira_results:
        if isinstance(jira_results, dict) and "status" in jira_results:
            combined["sources"]["jira"] = jira_results
        else:
            combined["sources"]["jira"] = {
                "status": "success",
                "data": sanitize_binary_content(jira_results)
            }
    
    # Process Perforce results
    if perforce_results:
        if isinstance(perforce_results, dict) and "status" in perforce_results:
            combined["sources"]["perforce"] = perforce_results
        else:
            combined["sources"]["perforce"] = {
                "status": "success",
                "data": sanitize_binary_content(perforce_results)
            }
    
    # Process Confluence results
    if confluence_results:
        if isinstance(confluence_results, dict) and "status" in confluence_results:
            combined["sources"]["confluence"] = confluence_results
        else:
            combined["sources"]["confluence"] = {
                "status": "success",
                "data": sanitize_binary_content(confluence_results)
            }
    
    # Identify cross-references between sources
    # This is a placeholder for more sophisticated cross-reference detection
    combined["cross_references"] = _extract_cross_references(combined["sources"])
    
    return combined

def _extract_cross_references(sources: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract cross-references between different sources.
    This is a placeholder for more sophisticated analysis.
    
    Args:
        sources: The combined source data
        
    Returns:
        List of cross-references
    """
    cross_references = []
    
    # Implementation would analyze content across sources to find connections
    # between Jira tickets, Perforce changelists, and Confluence pages
    
    return cross_references 