# src/assistant/verifiers/multi_agent_verification.py
"""
Multi-Agent Verification System that uses specialized agents to verify different aspects of content.
"""

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional, Union, TypedDict, Tuple, Set

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VerificationIssue(BaseModel):
    """Model for a verification issue."""
    issue_type: str = Field(description="Type of issue (factual, logical, technical, etc.)")
    description: str = Field(description="Description of the issue")
    severity: str = Field(description="Severity of the issue (critical, high, medium, low)")
    location: str = Field(description="Where in the content the issue appears")
    confidence: float = Field(description="Confidence in this issue (0-1)")


class AgentVerificationResult(BaseModel):
    """Model for agent verification results."""
    agent_id: str = Field(description="ID of the verification agent")
    agent_role: str = Field(description="Role/expertise of the agent")
    is_verified: bool = Field(description="Whether content passed verification")
    confidence: float = Field(description="Confidence in the verification result (0-1)")
    issues: List[Dict[str, Any]] = Field(description="List of issues found")
    reasoning: str = Field(description="Agent's reasoning for the verification decision")


class VerificationAgent:
    """A specialized verification agent with a specific expertise."""
    
    def __init__(
        self, 
        llm: BaseChatModel, 
        agent_id: str,
        agent_role: str,
        instruction_template: str
    ):
        """
        Initialize a verification agent.
        
        Args:
            llm: Language model for the agent.
            agent_id: Unique identifier for the agent.
            agent_role: Role/expertise of the agent.
            instruction_template: Template for agent instructions.
        """
        self.llm = llm
        self.agent_id = agent_id
        self.agent_role = agent_role
        self.instruction_template = instruction_template
    
    async def verify(
        self, content: str, context: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Verify content based on agent's expertise.
        
        Args:
            content: Content to verify.
            context: Optional context for verification.
            
        Returns:
            Verification result from this agent.
        """
        try:
            # Format context if provided
            formatted_context = "\n\n".join(context) if context else "No additional context provided."
            
            # Create verification prompt
            verify_prompt = ChatPromptTemplate.from_template(
                "You are a specialized verification agent with expertise in {role}.\n\n"
                "{instructions}\n\n"
                "Content to verify:\n{content}\n\n"
                "Additional context:\n{context}\n\n"
                "Carefully verify the content according to your expertise. "
                "Find any issues related to {role}, and assess whether the content passes verification "
                "from your perspective.\n\n"
                "Provide your verification result in this JSON format:\n"
                "- is_verified (boolean): Whether the content passes your verification\n"
                "- confidence (float 0-1): Your confidence in this assessment\n"
                "- issues (list): List of issues found, each with these fields:\n"
                "  - issue_type (string): Type of issue (factual, logical, technical, etc.)\n"
                "  - description (string): Description of the issue\n"
                "  - severity (string): Severity (critical, high, medium, low)\n"
                "  - location (string): Where in the content the issue appears\n"
                "  - confidence (float 0-1): Confidence in this issue\n"
                "- reasoning (string): Your detailed reasoning for the verification decision"
            )
            
            # Define the output schema explicitly as a proper Pydantic model
            class AgentOutput(BaseModel):
                is_verified: bool = Field(description="Whether the content passes verification")
                confidence: float = Field(description="Confidence in the verification result (0-1)")
                issues: List[Dict[str, Any]] = Field(description="List of issues found")
                reasoning: str = Field(description="Detailed reasoning for the verification decision")
            
            # Create verification chain with structured output
            verify_chain = verify_prompt | self.llm.with_structured_output(AgentOutput)
            
            # Invoke verification
            result = await verify_chain.ainvoke({
                "role": self.agent_role,
                "instructions": self.instruction_template,
                "content": content,
                "context": formatted_context
            })
            
            # Convert Pydantic model to dict if needed
            if hasattr(result, "model_dump"):
                result = result.model_dump()
            
            # Add agent metadata
            result["agent_id"] = self.agent_id
            result["agent_role"] = self.agent_role
            
            return result
            
        except Exception as e:
            logger.error(f"Error in agent {self.agent_id} verification: {e}")
            # Return error result
            return {
                "agent_id": self.agent_id,
                "agent_role": self.agent_role,
                "is_verified": False,
                "confidence": 0.0,
                "issues": [{
                    "issue_type": "error",
                    "description": f"Error during verification: {str(e)}",
                    "severity": "critical",
                    "location": "verification process",
                    "confidence": 1.0
                }],
                "reasoning": f"Error during verification: {str(e)}"
            }


class MultiAgentVerifier:
    """
    Multi-Agent Verification System that uses specialized agents to verify different aspects of content.
    
    This approach uses multiple specialized agents with different verification expertise to conduct
    a comprehensive verification of content, similar to a panel of experts evaluating different aspects
    of the same material.
    """
    
    def __init__(self, llm: BaseChatModel, topic: Optional[str] = None):
        """
        Initialize the multi-agent verification system.
        
        Args:
            llm: Language model to use for verification.
            topic: Optional topic to specialize verification for.
        """
        self.llm = llm
        self.topic = topic
        
        # Initialize specialized verification agents
        self.agents = self._create_agents()
    
    def _create_agents(self) -> List[VerificationAgent]:
        """
        Create specialized verification agents.
        
        Returns:
            List of verification agents.
        """
        agents = []
        
        # Factual accuracy agent
        factual_template = (
            "Your role is to verify the factual accuracy of the content. "
            "Check for claims that are unsupported, contradictory, or inconsistent with general knowledge. "
            "Focus on specific facts, dates, numbers, names, and technical details. "
            "Flag any factual claims that cannot be verified from the provided context."
        )
        agents.append(VerificationAgent(
            self.llm, "factual_agent", "factual accuracy", factual_template
        ))
        
        # Logical consistency agent
        logical_template = (
            "Your role is to verify the logical consistency of the content. "
            "Check for contradictions, non-sequiturs, circular reasoning, and other logical fallacies. "
            "Ensure that arguments flow logically and that conclusions follow from premises. "
            "Flag any logical inconsistencies or reasoning errors."
        )
        agents.append(VerificationAgent(
            self.llm, "logical_agent", "logical consistency", logical_template
        ))
        
        # Technical correctness agent
        technical_template = (
            "Your role is to verify the technical correctness of the content. "
            "Check for technical inaccuracies, misuse of technical terms, implausible technical descriptions, "
            "and other technical errors. Focus on domain-specific technical details and ensure they are accurate. "
            "Flag any technical issues or implausible technical claims."
        )
        agents.append(VerificationAgent(
            self.llm, "technical_agent", "technical correctness", technical_template
        ))
        
        # Completeness agent
        completeness_template = (
            "Your role is to verify the completeness of the content. "
            "Check for missing important information, unexplained gaps, or incomplete discussions of important topics. "
            "Flag any areas where the content is incomplete or where important context is missing."
        )
        agents.append(VerificationAgent(
            self.llm, "completeness_agent", "completeness", completeness_template
        ))
        
        # Hallucination detection agent
        hallucination_template = (
            "Your role is to detect hallucinations in the content. "
            "Focus specifically on identifying content that appears to be fabricated, unnecessarily specific without support, "
            "or that makes claims beyond what the context would support. "
            "Flag any suspected hallucinations or unsupported speculations."
        )
        agents.append(VerificationAgent(
            self.llm, "hallucination_agent", "hallucination detection", hallucination_template
        ))
        
        return agents
    
    async def verify_content(
        self, content: str, context: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Verify content using multiple specialized agents.
        
        Args:
            content: Content to verify.
            context: Optional additional context for verification.
            
        Returns:
            Comprehensive verification results.
        """
        try:
            # Run verification with all agents in parallel
            tasks = [agent.verify(content, context) for agent in self.agents]
            agent_results = await asyncio.gather(*tasks)
            
            # Analyze consensus
            consensus_result = await self._analyze_consensus(agent_results)
            
            # Extract and categorize issues
            critical_issues, all_issues = self._extract_issues(agent_results)
            
            # Create final result
            result = {
                "is_verified": consensus_result["is_verified"],
                "consensus_score": consensus_result["consensus_score"],
                "weighted_confidence": consensus_result["weighted_confidence"],
                "individual_results": agent_results,
                "critical_issues": critical_issues,
                "all_issues": all_issues
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in multi-agent verification: {e}")
            return {
                "is_verified": False,
                "consensus_score": 0.0,
                "weighted_confidence": 0.0,
                "individual_results": [],
                "critical_issues": [{
                    "error": f"Error during verification: {str(e)}"
                }],
                "all_issues": [{
                    "error": f"Error during verification: {str(e)}"
                }]
            }
    
    async def _analyze_consensus(self, agent_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze consensus among verification agents.
        
        Args:
            agent_results: Results from individual agents.
            
        Returns:
            Consensus analysis.
        """
        # Calculate basic statistics
        verified_count = sum(1 for result in agent_results if result.get("is_verified", False))
        total_agents = len(agent_results)
        consensus_score = verified_count / total_agents if total_agents > 0 else 0.0
        
        # Calculate weighted confidence
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for result in agent_results:
            is_verified = result.get("is_verified", False)
            confidence = result.get("confidence", 0.5)
            
            # Assign weights based on agent roles (could be customized)
            weight = 1.0
            role = result.get("agent_role", "").lower()
            
            # Adjust weights based on agent specialization
            if "factual" in role:
                weight = 1.5  # Higher weight for factual accuracy
            elif "hallucination" in role:
                weight = 1.3  # Higher weight for hallucination detection
            
            # For verified content, add positive confidence; for unverified, subtract
            weighted_confidence += weight * (confidence if is_verified else -confidence)
            total_weight += weight
        
        # Normalize to -1.0 to 1.0 range, then convert to 0.0 to 1.0
        normalized_confidence = (weighted_confidence / total_weight) if total_weight > 0 else 0.0
        normalized_confidence = (normalized_confidence + 1.0) / 2.0  # Convert from [-1,1] to [0,1]
        
        # Determine verification threshold (could be configurable)
        verification_threshold = 0.6
        
        return {
            "consensus_score": consensus_score,
            "weighted_confidence": normalized_confidence,
            "is_verified": normalized_confidence >= verification_threshold
        }
    
    def _extract_issues(self, agent_results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Extract issues from agent results.
        
        Args:
            agent_results: Results from individual agents.
            
        Returns:
            Tuple of (critical issues, all issues).
        """
        critical_issues = []
        all_issues = []
        
        for result in agent_results:
            agent_role = result.get("agent_role", "unknown")
            issues = result.get("issues", [])
            
            for issue in issues:
                # Format issue information
                issue_type = issue.get("issue_type", "unknown")
                description = issue.get("description", "No description")
                severity = issue.get("severity", "medium").lower()
                location = issue.get("location", "unknown")
                confidence = issue.get("confidence", 0.5)
                
                formatted_issue = {
                    "agent": agent_role,
                    "type": issue_type,
                    "description": description,
                    "location": location
                }
                
                # Add to appropriate lists
                all_issues.append(formatted_issue)
                
                if severity == "critical" and confidence >= 0.7:
                    critical_issues.append(formatted_issue)
        
        return critical_issues, all_issues