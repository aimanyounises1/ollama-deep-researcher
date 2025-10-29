# src/assistant/verifiers/fact_checker.py
"""
Fact checking component that verifies claims against provided context.
"""

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional, Union, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
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


class FactCheck(BaseModel):
    """Output schema for fact checking."""
    is_verified: bool = Field(description="Whether the claim is verified by the context")
    confidence: float = Field(description="Confidence level from 0.0 to 1.0")
    corrections: List[str] = Field(description="List of corrections if the claim is not verified")
    reasoning: str = Field(description="Reasoning for the verification decision")


class FactChecker:
    """Verifies content against reference material to detect hallucinations."""
    
    def __init__(self, llm: BaseChatModel):
        """
        Initialize the fact checker.
        
        Args:
            llm: Language model to use for verification.
        """
        self.llm = llm
    
    async def verify_with_context(
        self, content: str, context: List[str], detailed: bool = False
    ) -> VerificationResult:
        """
        Verify content against provided context.
        
        Args:
            content: The content to verify.
            context: List of context passages to verify against.
            detailed: Whether to return detailed verification with reasoning.
            
        Returns:
            Verification result indicating whether content is verified.
        """
        try:
            # Extract claims from content for verification
            claims = await self._extract_claims(content)
            
            # Verify each claim against context
            verification_results = []
            for claim in claims:
                result = await self._verify_claim(claim, context)
                verification_results.append(result)
            
            # Aggregate results
            all_verified = all(result["is_verified"] for result in verification_results)
            avg_confidence = sum(result["confidence"] for result in verification_results) / len(verification_results) if verification_results else 0.0
            
            # Collect corrections for unverified claims
            corrections = []
            for result in verification_results:
                if not result["is_verified"] and result.get("corrections"):
                    corrections.extend(result["corrections"])
            
            # Format reasoning if detailed results requested
            reasoning = None
            if detailed:
                reasoning_parts = []
                for i, (claim, result) in enumerate(zip(claims, verification_results)):
                    verification_status = "✓ Verified" if result["is_verified"] else "✗ Not verified"
                    confidence = f"{result['confidence']:.2f}"
                    reasoning_parts.append(f"Claim {i+1}: \"{claim}\"\n{verification_status} (confidence: {confidence})\nReasoning: {result.get('reasoning', 'No reasoning provided')}")
                
                reasoning = "\n\n".join(reasoning_parts)
            
            return {
                "is_verified": all_verified,
                "confidence": avg_confidence,
                "corrections": corrections if corrections else None,
                "reasoning": reasoning
            }
            
        except Exception as e:
            logger.error(f"Error verifying content: {e}")
            return {
                "is_verified": False,
                "confidence": 0.0,
                "corrections": ["Error during verification: could not complete the verification process."],
                "reasoning": f"Error during verification: {str(e)}"
            }
    
    async def _extract_claims(self, content: str) -> List[str]:
        """
        Extract verifiable claims from content.
        
        Args:
            content: The content to extract claims from.
            
        Returns:
            List of claims for verification.
        """
        # Use the LLM to extract claims for verification
        extract_prompt = ChatPromptTemplate.from_template(
            "Extract the key factual claims from the following text. "
            "Focus on specific, verifiable statements about technical systems, processes, data, "
            "or historical facts. Ignore opinions, speculations, and subjective assessments.\n\n"
            "Text:\n{content}\n\n"
            "Return the claims as a numbered list, with each claim being a single factual statement."
        )
        
        extract_chain = extract_prompt | self.llm | StrOutputParser()
        
        response = await extract_chain.ainvoke({"content": content})
        
        # Parse the numbered list
        claims = []
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('- ')):
                # Extract the claim part (removing any numbering)
                claim_text = line.split('.', 1)[-1].strip() if '.' in line else line.split(' ', 1)[-1].strip()
                claims.append(claim_text)
        
        # If extraction fails, use sentence splitting as fallback
        if not claims:
            sentences = re.split(r'(?<=[.!?])\s+', content)
            claims = [s.strip() for s in sentences if len(s.strip()) > 20]  # Only reasonably long sentences
        
        return claims
    
    async def _verify_claim(self, claim: str, context: List[str]) -> Dict[str, Any]:
        """
        Verify a single claim against provided context.
        
        Args:
            claim: The claim to verify.
            context: List of context passages to verify against.
            
        Returns:
            Verification result for the claim.
        """
        # Concatenate context with reasonable length limit
        combined_context = "\n\n".join(context)
        if len(combined_context) > 12000:  # Limit context to avoid token limits
            combined_context = combined_context[:12000] + "..."
        
        # Create verification prompt
        verify_prompt = ChatPromptTemplate.from_template(
            "Your task is to verify whether a claim is supported by the provided context.\n\n"
            "Claim to verify:\n{claim}\n\n"
            "Context:\n{context}\n\n"
            "Verify if the claim is supported by the context. Apply these criteria:\n"
            "1. The claim must be explicitly supported by the context\n"
            "2. If the claim contradicts the context, it is not verified\n"
            "3. If the claim contains information not found in the context, it is not verified\n"
            "4. If the claim is a reasonable inference from the context, but has additional details, it is partially verified\n\n"
            "Provide a detailed analysis and return a JSON response with these fields:\n"
            "- is_verified (boolean): Whether the claim is verified by the context\n"
            "- confidence (float between 0 and 1): Your confidence in the verification\n"
            "- corrections (list of strings): Suggested corrections if the claim is not verified\n"
            "- reasoning (string): Your detailed reasoning for the verification decision"
        )
        
        # Create verification chain with JSON output
        verify_chain = verify_prompt | self.llm.with_structured_output(FactCheck)
        
        # Verify claim against context
        try:
            result = await verify_chain.ainvoke({
                "claim": claim,
                "context": combined_context
            })
            
            # Handle result - ensure it's converted to a dictionary
            if hasattr(result, "model_dump"):
                return result.model_dump()
            elif isinstance(result, dict):
                return result
            else:
                # Handle unexpected result type by converting to a basic dict
                return {
                    "is_verified": False,
                    "confidence": 0.0,
                    "corrections": [f"Unable to verify due to processing error: {claim}"],
                    "reasoning": "Error: Unexpected result format from verification model."
                }
        except Exception as e:
            logger.error(f"Error verifying claim: {e}")
            # Return error result
            return {
                "is_verified": False,
                "confidence": 0.0,
                "corrections": [f"Unable to verify: {claim}"],
                "reasoning": f"Error during verification: {str(e)}"
            }