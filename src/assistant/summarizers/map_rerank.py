# src/assistant/summarizers/map_rerank.py
"""
Map-Rerank pattern implementation for generating and selecting the most accurate summaries.

This module implements the Map-Rerank pattern where:
1. Multiple different summaries are generated using different prompting strategies
2. Each summary is scored based on various quality metrics
3. The highest-scoring summary is selected as the final output

This approach reduces hallucination by leveraging diverse generation strategies
and selecting the most reliable output.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, TypedDict, Tuple
import re

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.assistant.verifiers.fact_checker import FactChecker

logger = logging.getLogger(__name__)

class CandidateSummary(TypedDict):
    """Structure for candidate summaries with metadata"""
    summary: str
    generation_strategy: str
    scores: Dict[str, float]
    total_score: float

class MapRerankSummarizer:
    """
    Implements the Map-Rerank pattern for generating reliable summaries.
    
    This approach:
    1. Maps: Generates multiple candidate summaries using different strategies
    2. Scores: Evaluates each candidate on multiple quality criteria
    3. Reranks: Sorts candidates by score
    4. Selects: Returns the highest-scoring candidate
    
    This reduces hallucination by generating diverse alternatives and
    selecting the most reliable one.
    """
    
    def __init__(self, llm, fact_checker: Optional[FactChecker] = None):
        """
        Initialize the Map-Rerank summarizer
        
        Args:
            llm: Language model to use for generation and evaluation
            fact_checker: Optional FactChecker for verifying candidates
        """
        self.llm = llm
        self.fact_checker = fact_checker
        logger.info("Map-Rerank summarizer initialized")
    
    async def generate_summaries(self, 
                               topic: str, 
                               source_data: Dict[str, str],
                               context: Optional[List[str]] = None,
                               num_candidates: int = 3) -> Dict[str, Any]:
        """
        Generate and select the best summary using Map-Rerank pattern
        
        Args:
            topic: Research topic
            source_data: Dict of source data by type (jira, confluence, etc.)
            context: Optional additional context for fact checking
            num_candidates: Number of candidate summaries to generate
            
        Returns:
            Dict with selected summary and evaluation details
        """
        logger.info(f"Generating {num_candidates} candidate summaries for topic: {topic}")
        
        # Generate candidate summaries with different strategies
        candidates = await self._generate_candidate_summaries(topic, source_data, num_candidates)
        
        # Score each candidate
        scored_candidates = await self._score_candidates(candidates, topic, source_data, context)
        
        # Select best candidate
        selected = max(scored_candidates, key=lambda x: x["total_score"])
        
        logger.info(f"Selected best summary with score {selected['total_score']:.2f} using strategy: {selected['generation_strategy']}")
        
        return {
            "selected_summary": selected["summary"],
            "all_candidates": scored_candidates,
            "best_strategy": selected["generation_strategy"],
            "score": selected["total_score"]
        }
    
    async def _generate_candidate_summaries(self, 
                                         topic: str, 
                                         source_data: Dict[str, str],
                                         num_candidates: int) -> List[Dict[str, str]]:
        """
        Generate multiple candidate summaries using different prompting strategies
        
        Args:
            topic: Research topic
            source_data: Dict of source data by type
            num_candidates: Number of candidates to generate
            
        Returns:
            List of candidate summaries with generation strategy
        """
        # Define generation strategies - different ways to prompt the model
        strategies = [
            {
                "name": "comprehensive",
                "prompt": self._create_comprehensive_prompt(topic, source_data),
                "description": "Comprehensive strategy focusing on thoroughness and detail"
            },
            {
                "name": "conservative",
                "prompt": self._create_conservative_prompt(topic, source_data),
                "description": "Conservative strategy focusing on high-confidence facts only"
            },
            {
                "name": "structured",
                "prompt": self._create_structured_prompt(topic, source_data),
                "description": "Structured strategy using clear sections and organization"
            },
            {
                "name": "critical",
                "prompt": self._create_critical_prompt(topic, source_data),
                "description": "Critical strategy focusing on analysis and evaluation"
            }
        ]
        
        # Select strategies based on requested number of candidates
        selected_strategies = strategies[:num_candidates]
        
        # Generate summaries in parallel
        generation_tasks = []
        for strategy in selected_strategies:
            task = self._generate_summary_with_strategy(strategy["prompt"], strategy["name"])
            generation_tasks.append(task)
        
        results = await asyncio.gather(*generation_tasks)
        
        return results
    
    async def _generate_summary_with_strategy(self, prompt: str, strategy_name: str) -> Dict[str, str]:
        """
        Generate a summary using a specific prompt strategy
        
        Args:
            prompt: The prompt to use for generation
            strategy_name: Name of the strategy for tracking
            
        Returns:
            Dict with summary and strategy metadata
        """
        try:
            # Create chain
            chain = self.llm | StrOutputParser()
            
            # Generate summary
            summary = await chain.ainvoke(prompt)
            
            return {
                "summary": summary,
                "generation_strategy": strategy_name
            }
        except Exception as e:
            logger.error(f"Error generating summary with strategy {strategy_name}: {e}")
            return {
                "summary": f"Error generating summary: {str(e)}",
                "generation_strategy": strategy_name
            }
    
    async def _score_candidates(self, 
                             candidates: List[Dict[str, str]], 
                             topic: str, 
                             source_data: Dict[str, str],
                             context: Optional[List[str]] = None) -> List[CandidateSummary]:
        """
        Score candidate summaries on multiple quality criteria
        
        Args:
            candidates: List of candidate summaries
            topic: Research topic
            source_data: Source data used for generation
            context: Optional additional context for fact checking
            
        Returns:
            List of candidates with scores
        """
        scored_candidates = []
        
        for candidate in candidates:
            summary = candidate["summary"]
            strategy = candidate["generation_strategy"]
            
            # Calculate various quality scores
            coherence_score = await self._assess_coherence(summary)
            coverage_score = await self._assess_coverage(summary, source_data)
            specificity_score = await self._assess_specificity(summary)
            
            # Calculate fact checking score if fact checker available
            fact_check_score = 0.0
            if self.fact_checker:
                fact_check_result = await self.fact_checker.verify_with_context(
                    summary, 
                    context or []
                )
                fact_check_score = fact_check_result.get("confidence", 0.0)
                
                # If not verified, penalize score
                if not fact_check_result.get("is_verified", False):
                    fact_check_score *= 0.5
            else:
                # Fallback rating if no fact checker
                fact_check_score = await self._assess_factuality(summary, source_data)
            
            # Detect speculation/uncertainty markers
            speculation_score = await self._assess_speculation(summary)
            
            # Calculate total score with weights
            total_score = (
                coherence_score * 0.15 +
                coverage_score * 0.20 +
                specificity_score * 0.15 +
                fact_check_score * 0.40 +  # Factuality is most important
                speculation_score * 0.10
            )
            
            # Create scored candidate
            scored_candidate: CandidateSummary = {
                "summary": summary,
                "generation_strategy": strategy,
                "scores": {
                    "coherence": coherence_score,
                    "coverage": coverage_score,
                    "specificity": specificity_score,
                    "factuality": fact_check_score,
                    "low_speculation": speculation_score
                },
                "total_score": total_score
            }
            
            scored_candidates.append(scored_candidate)
        
        return scored_candidates
    
    async def _assess_coherence(self, summary: str) -> float:
        """Assess coherence and logical flow of the summary"""
        prompt = ChatPromptTemplate.from_template(
            "Rate the coherence and logical flow of the following summary on a scale from 0.0 to 1.0.\n\n"
            "Summary:\n{summary}\n\n"
            "Consider these factors:\n"
            "- Logical progression of ideas\n"
            "- Consistent narrative flow\n"
            "- Clear transitions between topics\n"
            "- Absence of contradictions\n\n"
            "Respond with ONLY a number between 0.0 and 1.0, where 1.0 indicates perfect coherence."
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            result = await chain.ainvoke({"summary": summary})
            # Extract numeric score
            score_match = re.search(r'(\d+\.\d+|\d+)', result)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0.0), 1.0)  # Ensure between 0 and 1
            return 0.5  # Default if parsing fails
        except Exception as e:
            logger.error(f"Error assessing coherence: {e}")
            return 0.5
    
    async def _assess_coverage(self, summary: str, source_data: Dict[str, str]) -> float:
        """Assess how well the summary covers key information from sources"""
        # Combine source data
        combined_sources = "\n\n".join(source_data.values())
        
        prompt = ChatPromptTemplate.from_template(
            "Your task is to evaluate how completely the summary covers the key information "
            "from the source material. Rate on a scale from 0.0 to 1.0.\n\n"
            "Source material key information:\n{sources}\n\n"
            "Summary to evaluate:\n{summary}\n\n"
            "Consider:\n"
            "- Are all major topics from the source included?\n"
            "- Are important details preserved?\n"
            "- Is any critical information missing?\n\n"
            "Respond with ONLY a number between 0.0 and 1.0, where 1.0 indicates perfect coverage."
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            result = await chain.ainvoke({
                "summary": summary,
                "sources": combined_sources[:2000]  # Limit size to avoid token issues
            })
            # Extract numeric score
            score_match = re.search(r'(\d+\.\d+|\d+)', result)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0.0), 1.0)  # Ensure between 0 and 1
            return 0.5  # Default if parsing fails
        except Exception as e:
            logger.error(f"Error assessing coverage: {e}")
            return 0.5
    
    async def _assess_specificity(self, summary: str) -> float:
        """Assess the level of specific, concrete details vs. vague generalities"""
        prompt = ChatPromptTemplate.from_template(
            "Rate the specificity and concreteness of the following summary on a scale from 0.0 to 1.0.\n\n"
            "Summary:\n{summary}\n\n"
            "Consider these factors:\n"
            "- Use of specific examples, numbers, dates, names\n"
            "- Precise technical details rather than vague descriptions\n"
            "- Concrete rather than abstract language\n"
            "- Absence of hedge words ('maybe', 'possibly', etc.)\n\n"
            "Respond with ONLY a number between 0.0 and 1.0, where 1.0 indicates high specificity."
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            result = await chain.ainvoke({"summary": summary})
            # Extract numeric score
            score_match = re.search(r'(\d+\.\d+|\d+)', result)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0.0), 1.0)  # Ensure between 0 and 1
            return 0.5  # Default if parsing fails
        except Exception as e:
            logger.error(f"Error assessing specificity: {e}")
            return 0.5
    
    async def _assess_factuality(self, summary: str, source_data: Dict[str, str]) -> float:
        """
        Assess factual accuracy when no dedicated fact checker is available
        
        This is a simplified version of fact checking for when the full
        FactChecker component is not available.
        """
        # Combine source data
        combined_sources = "\n\n".join(source_data.values())
        
        prompt = ChatPromptTemplate.from_template(
            "Evaluate the factual accuracy of the following summary compared to the source material. "
            "Rate on a scale from 0.0 to 1.0.\n\n"
            "Source material:\n{sources}\n\n"
            "Summary to evaluate:\n{summary}\n\n"
            "Consider:\n"
            "- Are all factual statements supported by the source material?\n"
            "- Are there any contradictions with the source material?\n"
            "- Are there any claims that go beyond what's in the source material?\n\n"
            "Respond with ONLY a number between 0.0 and 1.0, where 1.0 indicates perfect factual accuracy."
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            result = await chain.ainvoke({
                "summary": summary,
                "sources": combined_sources[:2000]  # Limit size to avoid token issues
            })
            # Extract numeric score
            score_match = re.search(r'(\d+\.\d+|\d+)', result)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0.0), 1.0)  # Ensure between 0 and 1
            return 0.5  # Default if parsing fails
        except Exception as e:
            logger.error(f"Error assessing factuality: {e}")
            return 0.5
    
    async def _assess_speculation(self, summary: str) -> float:
        """
        Assess the absence of speculation and uncertainty markers
        
        Returns a higher score when fewer speculation markers are present
        """
        # Define patterns indicating speculation or uncertainty
        speculation_patterns = [
            r'(?i)(?:might|may|could|perhaps|possibly|probably|likely|seems|appears|suggests|indicates)',
            r'(?i)(?:uncertain|unclear|unknown|not clear|not certain|not sure)',
            r'(?i)(?:assuming|assumption|presuming|hypothesis|speculate)',
            r'(?i)(?:believe|think|feel|guess|estimate|approximate)'
        ]
        
        # Count occurrences
        total_markers = 0
        for pattern in speculation_patterns:
            matches = re.findall(pattern, summary)
            total_markers += len(matches)
        
        # Calculate words in summary
        word_count = len(summary.split())
        
        # Calculate density (markers per 100 words)
        if word_count == 0:
            return 1.0  # Empty summary
        
        marker_density = (total_markers / word_count) * 100
        
        # Convert to score (lower density = higher score)
        # 0 markers = 1.0, 5+ per 100 words = 0.0
        score = max(0.0, 1.0 - (marker_density / 5.0))
        
        return score
    
    def _create_comprehensive_prompt(self, topic: str, source_data: Dict[str, str]) -> str:
        """Create a prompt for comprehensive summary generation"""
        return (
            f"Create a comprehensive research report on: {topic}\n\n"
            "Focus on thoroughness and including all relevant details from the sources.\n"
            "Prioritize depth of coverage over brevity.\n\n"
            "Include information from:\n" +
            "\n\n".join([f"{source_type.upper()}:\n{data}" for source_type, data in source_data.items()]) +
            "\n\nProvide a detailed, thorough analysis that covers all aspects of the topic."
        )
    
    def _create_conservative_prompt(self, topic: str, source_data: Dict[str, str]) -> str:
        """Create a prompt for conservative (high-confidence only) summary generation"""
        return (
            f"Create a research report on: {topic}\n\n"
            "Focus ONLY on high-confidence information that is explicitly supported by the sources.\n"
            "DO NOT speculate or draw conclusions beyond what is directly stated.\n"
            "It is better to say less than to risk including uncertain information.\n\n"
            "Include information from:\n" +
            "\n\n".join([f"{source_type.upper()}:\n{data}" for source_type, data in source_data.items()]) +
            "\n\nProvide a concise, factual report containing only information you are highly confident about."
        )
    
    def _create_structured_prompt(self, topic: str, source_data: Dict[str, str]) -> str:
        """Create a prompt for structured summary generation"""
        return (
            f"Create a well-structured research report on: {topic}\n\n"
            "Use clear sections and organization to present the information logically.\n"
            "Include section headings to separate different aspects of the topic.\n\n"
            "Structure the report with these sections:\n"
            "1. Executive Summary\n"
            "2. Key Findings (from all sources)\n"
            "3. Detailed Analysis\n"
            "4. Connections and Relationships\n"
            "5. Gaps and Uncertainties\n\n"
            "Include information from:\n" +
            "\n\n".join([f"{source_type.upper()}:\n{data}" for source_type, data in source_data.items()]) +
            "\n\nProvide a clearly structured report that organizes the information effectively."
        )
    
    def _create_critical_prompt(self, topic: str, source_data: Dict[str, str]) -> str:
        """Create a prompt for critical analysis summary generation"""
        return (
            f"Create an analytical research report on: {topic}\n\n"
            "Focus on critical analysis and evaluation of the information.\n"
            "Identify patterns, inconsistencies, and relationships between data points.\n"
            "Distinguish between well-supported facts and areas with limited evidence.\n\n"
            "Include information from:\n" +
            "\n\n".join([f"{source_type.upper()}:\n{data}" for source_type, data in source_data.items()]) +
            "\n\nProvide an analytical report that evaluates the quality and significance of the information."
        )