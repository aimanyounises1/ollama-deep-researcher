# src/assistant/verifiers/self_verification.py
"""
Self-verification component that implements Meta AI's Chain-of-Verification approach.
"""

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional, Union, TypedDict, Tuple, Set

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VerificationResult(TypedDict):
    """Container for verification results"""
    is_verified: bool
    confidence: float
    corrections: Optional[List[str]]
    reasoning: Optional[str]
    uncertainty_markers: Optional[Dict[str, List[str]]]
    detailed_analysis: Optional[Dict[str, Any]]


class UncertaintyMarker(BaseModel):
    """Model for detected uncertainty markers in text."""
    marker_type: str = Field(description="Type of uncertainty marker (hedging, speculation, vagueness, etc.)")
    text: str = Field(description="The text containing the uncertainty marker")
    severity: str = Field(description="Severity level (high, medium, low)")
    suggested_correction: Optional[str] = Field(description="Suggested correction or clarification")


# NEW: Pydantic model to wrap a list of UncertaintyMarker objects
class UncertaintyMarkerList(BaseModel):
    markers: List[UncertaintyMarker] = Field(description="A list of detected uncertainty markers")


class VerificationStep(BaseModel):
    """Model for a single verification step."""
    step_number: int = Field(description="Step number in verification process")
    claim: str = Field(description="Claim being verified")
    analysis: str = Field(description="Analysis of the claim")
    is_verified: bool = Field(description="Whether the claim is verified")
    confidence: float = Field(description="Confidence in verification (0-1)")


class SelfVerificationOutput(BaseModel):
    """Structured output for self-verification."""
    is_verified: bool = Field(description="Whether the content passes verification")
    confidence: float = Field(description="Confidence level from 0.0 to 1.0")
    corrections: List[str] = Field(description="List of necessary corrections")
    reasoning: str = Field(description="Step-by-step reasoning for verification")
    uncertainty_markers: Optional[List[UncertaintyMarker]] = Field(description="Detected uncertainty markers")
    verification_steps: Optional[List[VerificationStep]] = Field(description="Detailed verification steps")


# NEW: Pydantic model for the output of _judge_verification
class JudgeVerificationOutput(BaseModel):
    is_verified: bool = Field(description="Whether the content passes verification")
    confidence: float = Field(description="Confidence level from 0.0 to 1.0, representing belief in the verification judgment itself.")
    reasoning: str = Field(description="A concise summary of the reasoning behind the verification judgment.")


class SelfVerifier:
    """
    Implements Meta AI's Chain-of-Verification (CoVe) approach for self-verification.
    
    Chain-of-Verification is a technique where the LLM verifies its own output in 
    a structured sequence of reasoning steps to catch and correct hallucinations.
    
    Reference: https://arxiv.org/abs/2309.11495
    
    Enhanced features:
    - Direct uncertainty detection in text
    - Detailed verification steps with debugging information
    - Confidence calibration based on multiple signals
    - Specialized detection for different hallucination types
    """
    
    def __init__(self, llm: BaseChatModel, enable_uncertainty_detection: bool = True, debug_mode: bool = False):
        """
        Initialize the self-verifier.
        
        Args:
            llm: Language model to use for verification.
            enable_uncertainty_detection: Whether to detect uncertainty markers in text.
            debug_mode: Whether to return detailed debugging information.
        """
        self.llm = llm
        self.enable_uncertainty_detection = enable_uncertainty_detection
        self.debug_mode = debug_mode
    
    async def verify_content(self, content: str) -> VerificationResult:
        """
        Verify content using Chain-of-Verification approach.
        
        Args:
            content: The content to verify.
            
        Returns:
            Verification result indicating whether content is verified.
        """
        try:
            # Initial verification reasoning with structured steps
            verification_steps = await self._generate_verification_steps(content)
            
            # Detect uncertainty markers if enabled
            uncertainty_markers = None
            if self.enable_uncertainty_detection:
                uncertainty_markers = await self._detect_uncertainty_markers(content)
            
            # Judge verification based on steps and uncertainty markers
            verification_result = await self._judge_verification(content, verification_steps, uncertainty_markers)
            
            # Generate corrections if needed
            if not verification_result["is_verified"]:
                corrections = await self._generate_corrections(content, verification_steps, uncertainty_markers)
                verification_result["corrections"] = corrections
            
            # Add uncertainty markers to the result if detected
            if uncertainty_markers:
                verification_result["uncertainty_markers"] = uncertainty_markers
            
            # Add detailed analysis if in debug mode
            if self.debug_mode:
                verification_result["detailed_analysis"] = await self._generate_debug_information(
                    content, verification_steps, uncertainty_markers
                )
            
            return verification_result
            
        except Exception as e:
            logger.error(f"Error in self-verification: {e}")
            return {
                "is_verified": False,
                "confidence": 0.0,
                "corrections": ["Error during self-verification process."],
                "reasoning": f"Error during verification: {str(e)}"
            }
    
    async def _generate_verification_steps(self, content: str) -> str:
        """
        Generate verification steps using Chain-of-Thought reasoning.
        
        Args:
            content: The content to verify.
            
        Returns:
            Verification steps as structured text.
        """
        # Define prompt for verification steps
        verify_prompt = ChatPromptTemplate.from_template(
            "You are a critical fact-checker tasked with verifying the accuracy of the following content:\n\n"
            "{content}\n\n"
            "Following the Chain-of-Verification approach, verify the accuracy of this content by:\n"
            "1. Identifying specific claims and statements that require verification\n"
            "2. For each claim, assess whether it contains potential inaccuracies, inconsistencies, or hallucinations\n"
            "3. For any identified issues, explain why they are problematic\n"
            "4. Consider both internal consistency and alignment with general knowledge\n"
            "5. Evaluate the specificity of claims - overly specific details without support may indicate hallucination\n"
            "6. Check for logical impossibilities or implausible combinations of facts\n"
            "7. Identify technical claims that seem questionable or unnecessarily specific\n\n"
            "Think step-by-step and be thorough in your verification. "
            "For each claim, rate your confidence in your verification (0-100%)."
        )
        
        verify_chain = verify_prompt | self.llm | StrOutputParser()
        
        # Generate verification steps
        verification_steps = await verify_chain.ainvoke({"content": content})
        
        return verification_steps
    
    async def _detect_uncertainty_markers(self, content: str) -> Dict[str, List[str]]:
        """
        Detect linguistic markers of uncertainty in the content.
        
        Args:
            content: The content to analyze.
            
        Returns:
            Dictionary mapping uncertainty types to text examples.
        """
        # Define prompt for uncertainty detection
        uncertainty_prompt = ChatPromptTemplate.from_template(
            "Analyze the following content for linguistic markers of uncertainty, ambiguity, or speculation:\n\n"
            "{content}\n\n"
            "Identify the following types of uncertainty markers:\n"
            "1. Hedging language (e.g., 'may', 'might', 'could', 'possibly', 'perhaps')\n"
            "2. Vague quantifiers (e.g., 'some', 'many', 'often', 'usually')\n"
            "3. Overly specific details without support\n"
            "4. Imprecise descriptions or measurements\n"
            "5. Speculation without clear foundation\n"
            "6. Subjective or value-laden language presented as fact\n\n"
            "For each uncertainty marker found, provide:\n"
            "- The type of marker\n"
            "- The exact text containing the marker\n"
            "- The severity (high, medium, low)\n"
            "- A suggested correction or clarification\n\n"
            "Format your response as a list of JSON objects."
        )
        
        # Define pydantic model for structured output
        uncertainty_chain = uncertainty_prompt | self.llm.with_structured_output(
            UncertaintyMarkerList
        )
        
        try:
            # Extract uncertainty markers
            marker_list_model = await uncertainty_chain.ainvoke({"content": content})
            
            # MODIFICATION START: Check if marker_list_model is valid and has markers
            markers = []
            if marker_list_model and isinstance(marker_list_model, UncertaintyMarkerList) and hasattr(marker_list_model, 'markers'):
                markers = marker_list_model.markers
            elif marker_list_model: # If it's not None but not the expected model, log a warning
                logger.warning(f"Unexpected type for marker_list_model: {type(marker_list_model)}. Expected UncertaintyMarkerList.")
            # If marker_list_model is None, markers remains an empty list, handled gracefully below.
            # MODIFICATION END
            
            # Reorganize by marker type
            result = {}
            for marker in markers:
                marker_type = marker.marker_type
                if marker_type not in result:
                    result[marker_type] = []
                
                result[marker_type].append(marker.text)
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting uncertainty markers: {e}")
            return {}
    
    async def _judge_verification(
        self, 
        content: str, 
        verification_steps: str, 
        uncertainty_markers: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Judge the verification based on verification steps and uncertainty markers.
        
        Args:
            content: The original content.
            verification_steps: The verification steps generated.
            uncertainty_markers: Optional dictionary of uncertainty markers.
            
        Returns:
            Verification result with judgment.
        """
        # Prepare uncertainty information if available
        uncertainty_info = ""
        if uncertainty_markers:
            uncertainty_info = "Detected uncertainty markers:\n"
            for marker_type, examples in uncertainty_markers.items():
                uncertainty_info += f"- {marker_type}: {', '.join(examples[:3])}"
                if len(examples) > 3:
                    uncertainty_info += f" (and {len(examples) - 3} more)"
                uncertainty_info += "\n"
        
        # Define prompt for judging verification
        judge_prompt = ChatPromptTemplate.from_template(
            "You are a high-precision fact verification judge. Evaluate the verification analysis below "
            "and determine if the original content passes verification.\n\n"
            "Original content:\n{content}\n\n"
            "Verification analysis:\n{verification_steps}\n\n"
            "{uncertainty_info}\n"
            "Based on this analysis, make a final judgment about whether the content passes verification.\n"
            "If there are any factual inaccuracies, logical inconsistencies, or unfounded claims, the content fails verification.\n"
            "If there are only minor issues that don't affect the core accuracy, the content may still pass verification.\n\n"
            "Provide your final judgment in JSON format with these fields:\n"
            "- is_verified (boolean): Whether the content passes verification\n"
            "- confidence (float between 0 and 1): Your confidence in this judgment\n"
            "- reasoning (string): A concise summary of your reasoning"
        )
        
        judge_chain = judge_prompt | self.llm.with_structured_output(
            JudgeVerificationOutput
        )
        
        # Generate judgment
        try:
            judgment_model = await judge_chain.ainvoke({
                "content": content,
                "verification_steps": verification_steps,
                "uncertainty_info": uncertainty_info
            })
        except Exception as e:
            logger.error(f"LLM error during verification judgment: {e}")
            judgment_model = None
        
        # Provide safe defaults if the LLM fails to return a structured output
        if judgment_model is None:
            logger.warning("judge_chain returned None, falling back to default unverified result")
            return {
                "is_verified": False,
                "confidence": 0.0,
                "reasoning": "Verification model did not return a result.",
                "corrections": []
            }
        
        # Adjust confidence based on uncertainty markers
        is_verified = judgment_model.is_verified
        confidence = judgment_model.confidence
        reasoning = judgment_model.reasoning
        
        if uncertainty_markers:
            # Reduce confidence if many uncertainty markers are present
            total_markers = sum(len(markers) for markers in uncertainty_markers.values())
            if total_markers > 5:
                confidence = max(0.1, confidence - 0.3)
            elif total_markers > 2:
                confidence = max(0.2, confidence - 0.2)
            elif total_markers > 0:
                confidence = max(0.3, confidence - 0.1)
        
        # Convert to expected format
        result = {
            "is_verified": is_verified,
            "confidence": confidence,
            "reasoning": reasoning,
            "corrections": []
        }
        
        return result
    
    async def _generate_corrections(
        self, 
        content: str, 
        verification_steps: str,
        uncertainty_markers: Optional[Dict[str, List[str]]] = None
    ) -> List[str]:
        """
        Generate corrections for issues identified in verification.
        
        Args:
            content: The original content.
            verification_steps: The verification steps generated.
            uncertainty_markers: Optional dictionary of uncertainty markers.
            
        Returns:
            List of specific corrections to address issues.
        """
        # Prepare uncertainty information if available
        uncertainty_info = ""
        if uncertainty_markers:
            uncertainty_info = "Detected uncertainty markers:\n"
            for marker_type, examples in uncertainty_markers.items():
                uncertainty_info += f"- {marker_type}: {', '.join(examples[:3])}"
                if len(examples) > 3:
                    uncertainty_info += f" (and {len(examples) - 3} more)"
                uncertainty_info += "\n"
        
        # Define prompt for generating corrections
        corrections_prompt = ChatPromptTemplate.from_template(
            "Based on the verification analysis, identify specific corrections needed for the content. "
            "For each issue identified, provide a clear correction instruction.\n\n"
            "Original content:\n{content}\n\n"
            "Verification analysis:\n{verification_steps}\n\n"
            "{uncertainty_info}\n"
            "List each specific correction needed as a separate bullet point. "
            "Be precise about what needs to be fixed and why. For each correction:\n"
            "1. Identify the problematic text\n"
            "2. Explain why it's problematic\n"
            "3. Suggest a specific correction"
        )
        
        corrections_chain = corrections_prompt | self.llm | StrOutputParser()
        
        # Generate corrections
        corrections_text = await corrections_chain.ainvoke({
            "content": content,
            "verification_steps": verification_steps,
            "uncertainty_info": uncertainty_info
        })
        
        # Parse bullet points
        corrections = []
        lines = corrections_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and (line.startswith('- ') or line.startswith('• ') or (line[0].isdigit() and '.' in line)):
                # Extract the correction (removing any bullet point or numbering)
                if line.startswith('- ') or line.startswith('• '):
                    correction = line[2:].strip()
                else:
                    correction = line.split('.', 1)[-1].strip()
                
                if correction:
                    corrections.append(correction)
        
        return corrections
    
    async def _generate_debug_information(
        self,
        content: str,
        verification_steps: str,
        uncertainty_markers: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate detailed debugging information for the verification process.
        
        Args:
            content: The original content.
            verification_steps: The verification steps generated.
            uncertainty_markers: Optional dictionary of uncertainty markers.
            
        Returns:
            Dictionary with detailed debugging information.
        """
        try:
            # Define prompt for extracting structured verification steps
            debug_prompt = ChatPromptTemplate.from_template(
                "Parse your verification analysis into structured steps for debugging purposes.\n\n"
                "Verification analysis:\n{verification_steps}\n\n"
                "For each claim you verified, extract:\n"
                "1. The step number\n"
                "2. The claim being verified\n"
                "3. Your analysis of the claim\n"
                "4. Whether the claim is verified (true/false)\n"
                "5. Your confidence in the verification (0-1)\n\n"
                "Format your response as a list of JSON objects, one per verification step."
            )
            
            debug_chain = debug_prompt | self.llm.with_structured_output(
                List[VerificationStep]
            )
            
            # Generate structured verification steps
            structured_steps = await debug_chain.ainvoke({
                "verification_steps": verification_steps
            })
            
            # Calculate statistics
            verified_claims = sum(1 for step in structured_steps if step.is_verified)
            total_claims = len(structured_steps)
            
            # Collect low-confidence claims
            low_confidence_claims = [
                {"claim": step.claim, "confidence": step.confidence}
                for step in structured_steps
                if step.confidence < 0.7
            ]
            
            # Build debug information
            debug_info = {
                "total_claims": total_claims,
                "verified_claims": verified_claims,
                "verification_rate": verified_claims / total_claims if total_claims > 0 else 0,
                "average_confidence": sum(step.confidence for step in structured_steps) / total_claims if total_claims > 0 else 0,
                "structured_steps": [step.model_dump() for step in structured_steps],
                "low_confidence_claims": low_confidence_claims,
            }
            
            # Add uncertainty information if available
            if uncertainty_markers:
                debug_info["uncertainty_markers"] = uncertainty_markers
                debug_info["total_uncertainty_markers"] = sum(len(markers) for markers in uncertainty_markers.values())
                
                # Group uncertainty markers by severity
                # This would require modifying the uncertainty detection to include severity
            
            return debug_info
            
        except Exception as e:
            logger.error(f"Error generating debug information: {e}")
            return {
                "error": str(e),
                "verification_steps": verification_steps
            }