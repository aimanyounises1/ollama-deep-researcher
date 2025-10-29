# src/assistant/verifiers/chain_of_knowledge.py
"""
Chain of Knowledge (CoK) implementation that requires expert source citations for all factual claims.
"""

import asyncio
import logging
import re
import json  # Explicit json import
from typing import Dict, List, Any, Optional, Union, TypedDict, Set

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SourceClaim(BaseModel):
    """Model for a claim with its source."""
    claim: str = Field(description="The factual claim made")
    source_type: str = Field(description="The type of source (jira, confluence, perforce, web, etc.)")
    source_id: str = Field(description="Identifier for the source")
    source_content: str = Field(description="The specific content from the source supporting this claim")
    confidence: float = Field(description="Confidence in the claim-source match (0-1)")


class CitationMetrics(BaseModel):
    """Metrics about citations in the content."""
    total_claims: int = Field(description="Total number of factual claims")
    cited_claims: int = Field(description="Number of claims with citations")
    citation_coverage: float = Field(description="Proportion of claims with citations (0-1)")
    high_confidence_citations: int = Field(description="Number of citations with high confidence (>0.8)")
    source_distribution: Dict[str, int] = Field(description="Distribution of source types used")


# Define pydantic model for extraction outside the method
class ClaimExtraction(BaseModel):
    """Model for extracting claims with their sources."""
    claim: str = Field(description="The factual claim made")
    source_type: str = Field(description="The type of source")
    cited_detail: str = Field(description="The specific detail cited")


class ChainOfKnowledge:
    """
    Implements Chain of Knowledge (CoK) approach for creating content with expert citations.
    
    Chain of Knowledge is a technique that explicitly requires factual claims to be linked
    to source material, greatly reducing the likelihood of hallucinations.
    """
    
    def __init__(self, llm: BaseChatModel):
        """
        Initialize the Chain of Knowledge component.
        
        Args:
            llm: Language model to use for generation and verification.
        """
        self.llm = llm
    
    async def generate_with_expert_citations(
        self, topic: str, source_data: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Generate content with expert citations for all factual claims.
        
        Args:
            topic: The research topic.
            source_data: Dict mapping source types to their content.
            
        Returns:
            Dict containing generated content, claims with sources, and citation metrics.
        """
        try:
            # Format source data for the prompt
            formatted_sources = self._format_sources(source_data)
            
            # Generate initial content with citations
            content = await self._generate_cited_content(topic, formatted_sources)
            
            # Extract claims and their sources
            claims_with_sources = await self._extract_claims_with_sources(content, source_data)
            
            # Calculate citation metrics
            citation_metrics = self._calculate_citation_metrics(claims_with_sources)
            
            # If citation coverage is too low, try regenerating
            if citation_metrics["citation_coverage"] < 0.7 and len(claims_with_sources) > 0:
                logger.warning(f"Low citation coverage: {citation_metrics['citation_coverage']:.2f}. Regenerating...")
                content = await self._regenerate_with_better_citations(topic, formatted_sources, claims_with_sources)
                claims_with_sources = await self._extract_claims_with_sources(content, source_data)
                citation_metrics = self._calculate_citation_metrics(claims_with_sources)
            
            return {
                "cok_content": content,
                "claims_with_sources": claims_with_sources,
                "citation_metrics": citation_metrics
            }
            
        except Exception as e:
            logger.error(f"Error in Chain of Knowledge generation: {e}")
            return {
                "cok_content": f"Error generating content with citations: {str(e)}",
                "claims_with_sources": [],
                "citation_metrics": {
                    "total_claims": 0,
                    "cited_claims": 0,
                    "citation_coverage": 0.0,
                    "high_confidence_citations": 0,
                    "source_distribution": {}
                }
            }
    
    async def verify_citation_accuracy(
        self, claims_with_sources: List[Dict[str, Any]], source_data: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Verify the accuracy of citations against the source data.
        
        Args:
            claims_with_sources: List of claims with their sources.
            source_data: Dict mapping source types to their content.
            
        Returns:
            Verification results.
        """
        try:
            # Track verification results
            results = {
                "accurate_citations": 0,
                "inaccurate_citations": 0,
                "accuracy_rate": 0.0,
                "problem_claims": []
            }
            
            # Skip verification if no claims
            if not claims_with_sources:
                return results
            
            # Verify each claim against its source
            for claim_data in claims_with_sources:
                claim = claim_data.get("claim", "")
                source_type = claim_data.get("source_type", "")
                source_content = claim_data.get("source_content", "")
                
                # Skip claims without source data
                if not source_type or not source_content or source_type not in source_data:
                    results["inaccurate_citations"] += 1
                    results["problem_claims"].append({
                        "claim": claim,
                        "issue": "Missing source data"
                    })
                    continue
                
                # Verify the claim against the source
                is_accurate = await self._verify_claim_against_source(
                    claim, source_content, source_data[source_type]
                )
                
                if is_accurate:
                    results["accurate_citations"] += 1
                else:
                    results["inaccurate_citations"] += 1
                    results["problem_claims"].append({
                        "claim": claim,
                        "issue": "Claim not supported by the cited source"
                    })
            
            # Calculate accuracy rate
            total_claims = results["accurate_citations"] + results["inaccurate_citations"]
            results["accuracy_rate"] = results["accurate_citations"] / total_claims if total_claims > 0 else 0.0
            
            return results
            
        except Exception as e:
            logger.error(f"Error verifying citation accuracy: {e}")
            return {
                "accurate_citations": 0,
                "inaccurate_citations": 0,
                "accuracy_rate": 0.0,
                "problem_claims": [{"issue": f"Error during verification: {str(e)}"}]
            }
    
    def _format_sources(self, source_data: Dict[str, str]) -> str:
        """
        Format sources for inclusion in the prompt.
        
        Args:
            source_data: Dict mapping source types to their content.
            
        Returns:
            Formatted source data as string.
        """
        formatted = []
        for source_type, content in source_data.items():
            formatted.append(f"SOURCE TYPE: {source_type.upper()}")
            formatted.append(f"CONTENT:\n{content}\n")
        
        return "\n".join(formatted)
    
    async def _generate_cited_content(self, topic: str, formatted_sources: str) -> str:
        """
        Generate content with explicit citations.
        
        Args:
            topic: The research topic.
            formatted_sources: Formatted source material.
            
        Returns:
            Generated content with citations.
        """
        # Define prompt for generating content with citations
        generate_prompt = ChatPromptTemplate.from_template(
            "You are an expert researcher creating content on: {topic}\n\n"
            "You will create a comprehensive report using ONLY information from the provided sources. "
            "For EVERY factual claim, you must cite the source using this format: [SOURCE_TYPE: specific detail].\n\n"
            "Available sources:\n{sources}\n\n"
            "Guidelines:\n"
            "1. NEVER introduce information not present in the sources\n"
            "2. EVERY factual claim must have a citation\n"
            "3. Format citations as [SOURCE_TYPE: specific detail] where SOURCE_TYPE is the type of source\n"
            "4. Be precise and technical in your analysis\n"
            "5. If sources contradict, note this explicitly\n"
            "6. If information is missing on a subtopic, state this explicitly\n\n"
            "Create a well-structured report with appropriate sections and clear citations."
        )
        
        generate_chain = generate_prompt | self.llm | StrOutputParser()
        
        # Generate content
        content = await generate_chain.ainvoke({
            "topic": topic,
            "sources": formatted_sources
        })
        
        return content
    
    async def _extract_claims_with_sources(
        self, content: str, source_data: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Extract claims and their sources from the generated content.
        
        Args:
            content: The generated content with citations.
            source_data: Dict mapping source types to their content.
            
        Returns:
            List of claims with their sources.
        """
        # First, extract using regex for the citation format [SOURCE_TYPE: detail]
        citation_pattern = r'\[([\w_]+):\s*([^\]]+)\]'
        citations = re.findall(citation_pattern, content)
        
        # Then use LLM to extract the claims associated with each citation
        extract_prompt = ChatPromptTemplate.from_template(
            "Extract factual claims and their sources from this content:\n\n"
            "{content}\n\n"
            "For each claim with a citation in the format [SOURCE_TYPE: detail], extract:\n"
            "1. The factual claim being made\n"
            "2. The source type (e.g., jira, confluence, perforce, web)\n"
            "3. The specific cited detail\n\n"
            "Format your response as a list of JSON objects with these fields:\n"
            "- claim: The factual claim made\n"
            "- source_type: The type of source\n"
            "- cited_detail: The specific detail cited\n\n"
            "Only include claims that have explicit citations in the format [SOURCE_TYPE: detail]."
        )
        
        extract_chain = extract_prompt | self.llm | StrOutputParser()
        
        try:
            # Extract claims using string parsing first
            extracted_text = await extract_chain.ainvoke({"content": content})
            
            # Parse the JSON response manually
            import json
            
            # Clean the output for JSON parsing
            json_str = extracted_text.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
            
            json_str = json_str.strip()
            
            # Try to parse as JSON list
            try:
                extracted_claims = json.loads(json_str)
                if not isinstance(extracted_claims, list):
                    extracted_claims = [extracted_claims]
            except json.JSONDecodeError:
                # Fallback to regex if JSON parsing fails
                logger.warning("Failed to parse JSON from LLM output, using regex fallback")
                extracted_claims = []
                for i, (source_type, cited_detail) in enumerate(citations):
                    extracted_claims.append({
                        "claim": f"Claim {i+1} from citation {source_type}: {cited_detail}",
                        "source_type": source_type.lower(),
                        "cited_detail": cited_detail
                    })
            
            # Post-process extracted claims to add source content and confidence
            claims_with_sources = []
            for claim_data in extracted_claims:
                source_type = claim_data.get("source_type", "").lower()
                claim = claim_data.get("claim", "")
                cited_detail = claim_data.get("cited_detail", "")
                
                source_id = f"{source_type}_{len(claims_with_sources)}"
                
                # Find relevant source content
                source_content = ""
                confidence = 0.5  # Default confidence
                
                if source_type in source_data:
                    full_source = source_data[source_type]
                    # Try to find the cited detail in the source
                    if cited_detail and cited_detail in full_source:
                        # Extract wider context around the cited detail
                        start_idx = max(0, full_source.find(cited_detail) - 100)
                        end_idx = min(len(full_source), full_source.find(cited_detail) + len(cited_detail) + 100)
                        source_content = full_source[start_idx:end_idx]
                        confidence = 0.9  # High confidence if cited detail found
                    else:
                        # Use whole source if detail not found exactly
                        source_content = full_source[:500] + "..." if len(full_source) > 500 else full_source
                        confidence = 0.7  # Medium confidence
                
                claims_with_sources.append({
                    "claim": claim,
                    "source_type": source_type,
                    "source_id": source_id,
                    "source_content": source_content,
                    "confidence": confidence
                })
            
            return claims_with_sources
            
        except Exception as e:
            logger.error(f"Error extracting claims: {e}")
            # Fallback to manual parsing based on regex findings
            claims_with_sources = []
            
            # Use regex matches as fallback
            for i, (source_type, cited_detail) in enumerate(citations):
                source_type = source_type.lower()
                source_id = f"{source_type}_{i}"
                
                # Extract context around citation to estimate the claim
                citation_marker = f"[{source_type}: {cited_detail}]"
                start_idx = max(0, content.find(citation_marker) - 200)
                end_idx = min(len(content), content.find(citation_marker) + len(citation_marker) + 50)
                context = content[start_idx:end_idx]
                
                # Use simple heuristic to extract the claim (sentence containing citation)
                sentence_pattern = r'([^.!?]+' + re.escape(citation_marker) + r'[^.!?]*[.!?])'
                claim_matches = re.findall(sentence_pattern, context)
                claim = claim_matches[0] if claim_matches else context
                
                claims_with_sources.append({
                    "claim": claim,
                    "source_type": source_type,
                    "source_id": source_id,
                    "source_content": cited_detail,
                    "confidence": 0.5  # Lower confidence for regex fallback
                })
            
            return claims_with_sources
    
    def _calculate_citation_metrics(self, claims_with_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate metrics about citations in the content.
        
        Args:
            claims_with_sources: List of claims with their sources.
            
        Returns:
            Citation metrics.
        """
        # Count total factual claims (this could be more sophisticated)
        total_claims = len(claims_with_sources)
        
        # Count claims with citations
        cited_claims = total_claims  # All extracted claims have citations by definition
        
        # Count high confidence citations
        high_confidence_citations = sum(1 for claim in claims_with_sources if claim.get("confidence", 0) > 0.8)
        
        # Calculate source distribution
        source_types = [claim.get("source_type", "unknown") for claim in claims_with_sources]
        source_distribution = {}
        for source_type in source_types:
            source_distribution[source_type] = source_distribution.get(source_type, 0) + 1
        
        return {
            "total_claims": total_claims,
            "cited_claims": cited_claims,
            "citation_coverage": cited_claims / total_claims if total_claims > 0 else 0.0,
            "high_confidence_citations": high_confidence_citations,
            "source_distribution": source_distribution
        }
    
    async def _regenerate_with_better_citations(
        self, topic: str, formatted_sources: str, previous_claims: List[Dict[str, Any]]
    ) -> str:
        """
        Regenerate content with better citations based on previous attempt.
        
        Args:
            topic: The research topic.
            formatted_sources: Formatted source material.
            previous_claims: Previous claims with sources.
            
        Returns:
            Regenerated content with better citations.
        """
        # Format feedback about previous citations
        citation_feedback = "Problems with previous citations:\n"
        for claim in previous_claims:
            confidence = claim.get("confidence", 0)
            if confidence < 0.8:
                citation_feedback += f"- Claim: '{claim.get('claim', '')}' - citation confidence: {confidence:.2f}\n"
        
        # Define prompt for regenerating with better citations
        regenerate_prompt = ChatPromptTemplate.from_template(
            "You are an expert researcher creating content on: {topic}\n\n"
            "You will create a comprehensive report using ONLY information from the provided sources. "
            "For EVERY factual claim, you must cite the source using this format: [SOURCE_TYPE: specific detail].\n\n"
            "Available sources:\n{sources}\n\n"
            "{citation_feedback}\n\n"
            "Guidelines:\n"
            "1. NEVER introduce information not present in the sources\n"
            "2. EVERY factual claim must have a citation\n"
            "3. Format citations as [SOURCE_TYPE: specific detail] where SOURCE_TYPE is the type of source\n"
            "4. Be precise and technical in your analysis\n"
            "5. If sources contradict, note this explicitly\n"
            "6. If information is missing on a subtopic, state this explicitly\n\n"
            "Create a well-structured report with appropriate sections and clear citations."
        )
        
        regenerate_chain = regenerate_prompt | self.llm | StrOutputParser()
        
        # Regenerate content
        content = await regenerate_chain.ainvoke({
            "topic": topic,
            "sources": formatted_sources,
            "citation_feedback": citation_feedback
        })
        
        return content
    
    async def _verify_claim_against_source(
        self, claim: str, cited_content: str, full_source: str
    ) -> bool:
        """
        Verify if a claim is supported by the cited source.
        
        Args:
            claim: The claim to verify.
            cited_content: The cited portion of the source.
            full_source: The full source content.
            
        Returns:
            Whether the claim is supported by the source.
        """
        # Define prompt for verification
        verify_prompt = ChatPromptTemplate.from_template(
            "Verify if this claim is supported by the cited source:\n\n"
            "Claim: {claim}\n\n"
            "Cited content: {cited_content}\n\n"
            "Full source excerpt: {full_source_excerpt}\n\n"
            "Determine if the claim is directly supported by the source material. "
            "The claim must not contain information that isn't present in the source. "
            "Minor rewordings or paraphrasing are acceptable as long as the meaning is preserved.\n\n"
            "Is the claim supported by the source? Answer with just 'yes' or 'no'."
        )
        
        verify_chain = verify_prompt | self.llm | StrOutputParser()
        
        # Limit full source size to avoid token limits
        full_source_excerpt = full_source[:1000] + "..." if len(full_source) > 1000 else full_source
        
        # Verify claim
        result = await verify_chain.ainvoke({
            "claim": claim,
            "cited_content": cited_content,
            "full_source_excerpt": full_source_excerpt
        })
        
        # Parse result
        return result.strip().lower() == "yes"