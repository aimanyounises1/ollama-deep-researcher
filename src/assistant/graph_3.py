# src/assistant/graph_3.py
# Enhanced implementation with advanced anti-hallucination techniques

import asyncio
import gc
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, TypedDict, Dict, Any, Optional, Union, Annotated, Literal, cast, Set, Tuple
from uuid import uuid4
import re

import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig, RunnablePassthrough
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END, START
from langgraph.constants import Send
from langgraph.checkpoint.memory import InMemorySaver

# Local imports - maintain compatibility with existing code structure
from src.assistant.sources.perforce_client import PerforceHelper

# Import modular components from our refactored structure
from src.assistant.rag.ollama_retrieval import RAGRetriever
from src.assistant.verifiers.fact_checker import FactChecker
from src.assistant.verifiers.self_verification import SelfVerifier
from src.assistant.verifiers.multi_agent_verification import MultiAgentVerifier
from src.assistant.verifiers.chain_of_knowledge import ChainOfKnowledge
from src.assistant.processors.chunk_processor import process_chunks
from src.assistant.knowledge.graph import TemporalKnowledgeGraph
from src.assistant.analyzers.cross_source_analyzer import analyze_cross_sources, apply_cross_source_analysis
from src.assistant.summarizers.map_rerank import MapRerankSummarizer
from src.assistant.utils.langgraph_helpers import (
    process_in_batches, map_reduce_text, extract_identifiers, clean_llm_output, MemoryOptimizer
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- State Types ---
class VerificationResult(TypedDict):
    """Container for verification results"""
    is_verified: bool
    confidence: float
    corrections: Optional[List[str]]
    reasoning: Optional[str]

class SourceData(TypedDict):
    """Container for data from a specific source"""
    raw_results: List[Dict[str, Any]]
    processed_results: List[str]
    current_chunk_index: int
    total_chunks: int
    summary: str
    verification: VerificationResult

class CrossSourceConnection(TypedDict):
    """Container for connections between sources"""
    source1_type: str
    source1_id: str
    source2_type: str
    source2_id: str
    connection_type: str
    confidence: float
    evidence: str

class MapRerankResult(TypedDict):
    """Container for map-rerank results"""
    selected_summary: str
    all_candidates: List[Dict[str, Any]]
    best_strategy: str
    score: float

class ChainOfKnowledgeResult(TypedDict):
    """Container for chain of knowledge results"""
    cok_content: str
    claims_with_sources: List[Dict[str, Any]]
    citation_metrics: Dict[str, Any]
    verification_results: Optional[Dict[str, Any]]

class MultiAgentVerificationResult(TypedDict):
    """Container for multi-agent verification results"""
    is_verified: bool
    consensus_score: float
    weighted_confidence: float
    individual_results: List[Dict[str, Any]]
    critical_issues: List[Dict[str, str]]
    all_issues: List[Dict[str, str]]

class ResearchState(TypedDict):
    """Structured state for the research graph with appropriate annotations"""
    # Core fields
    research_topic: str
    search_query: str
    running_summary: str
    research_loop_count: int
    messages: Annotated[List[BaseMessage], operator.add]
    
    # Source data
    jira_data: SourceData
    confluence_data: SourceData
    perforce_data: SourceData
    web_data: SourceData
    
    # RAG and verification
    retrieved_context: List[str]
    verification_results: List[VerificationResult]
    
    # Enhanced features
    map_rerank_results: Optional[MapRerankResult]
    chain_of_knowledge_results: Optional[ChainOfKnowledgeResult]
    multi_agent_verification_results: Optional[MultiAgentVerificationResult]
    
    # Cross-source analysis
    cross_source_analysis: Dict[str, Any]
    cross_source_connections: List[CrossSourceConnection]
    technical_validation: Dict[str, Any]
    security_findings: List[Dict[str, Any]]
    
    # Final output
    final_summary: str
    completed_research: bool
    hallucination_score: float  # Lower is better

# --- Configuration ---
class ResearchConfiguration:
    """Configuration for the research system"""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Default configuration
        self.ollama_base_url = os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434")
        self.local_llm = os.environ.get("LLM_MODEL", "qwen3:30b-a3b")
        self.enable_jira = True
        self.enable_confluence = True
        self.enable_perforce = True
        self.enable_web = True
        self.enable_rag = True
        self.enable_verification = True
        self.enable_cot = True  # Chain of Thought reasoning
        self.enable_cove = True  # Chain of Verification
        self.enable_map_rerank = True  # Map-Rerank pattern
        self.enable_cok = True  # Chain of Knowledge
        self.enable_multi_agent = True  # Multi-Agent Verification
        self.enable_cross_source = True  # Cross-source connections analysis
        self.max_iterations = 3
        self.temperature = 0.15  # Lower temperature for more factual outputs
        self.top_p = 0.7
        self.knowledge_graph_path = os.environ.get("KNOWLEDGE_GRAPH_PATH", "./knowledge_graph.json")
        
        # Fast-mode flag (reduced search scope & lighter LLM ops)
        self.fast_mode = False
        
        # Override with provided config
        if config:
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> 'ResearchConfiguration':
        """Create configuration from LangChain RunnableConfig"""
        if not config:
            return cls()
            
        # Extract configurable values
        config_dict = {}
        if isinstance(config, dict) and "configurable" in config:
            config_dict = config["configurable"]
            
        return cls(config_dict)

# --- LLM Setup ---
def get_llm(config: ResearchConfiguration, temperature: Optional[float] = None):
    """Initialize LLM with appropriate configuration"""
    return ChatOllama(
        base_url=config.ollama_base_url,
        model=config.local_llm,
        temperature=temperature if temperature is not None else config.temperature,
        top_p=config.top_p,
        num_ctx=8192,
        repeat_penalty=1.1
    )

# Define Pydantic model for structured output
class ResearchParameters(BaseModel):
    """Research parameters extracted from the user query."""
    research_topic: str = Field(description="The central topic to research (short phrase)")
    search_query: str = Field(description="The specific search query that would yield good results")

# --- MAIN GRAPH DEFINITION ---

def create_fact_verification_subgraph(config: ResearchConfiguration):
    """Create a subgraph for fact verification with hallucination detection"""
    
    # Define the state for this subgraph
    class VerificationState(TypedDict):
        content: str
        verified_content: str
        is_verified: bool
        corrections: List[str]
        verification_steps: List[str]
        
    # Define nodes
    async def initial_verification(state: VerificationState):
        """Initial quick verification of content"""
        verifier = SelfVerifier(get_llm(config, temperature=0.1))
        result = await verifier.verify_content(state["content"])
        
        return {
            "is_verified": result["is_verified"],
            "corrections": result.get("corrections", []),
            "verification_steps": [f"Initial verification: {result['is_verified']}"]
        }
    
    async def deep_verification(state: VerificationState):
        """Second-level verification with RAG and external sources"""
        fact_checker = FactChecker(get_llm(config, temperature=0.1))
        
        # Use RAG to ground verification in factual data
        retriever = RAGRetriever()
        related_facts = await retriever.retrieve(state["content"])
        
        result = await fact_checker.verify_with_context(state["content"], related_facts)
        
        # Add verification step
        verification_steps = state.get("verification_steps", [])
        verification_steps.append(f"Deep verification: {result['is_verified']}")
        
        return {
            "is_verified": result["is_verified"],
            "corrections": result.get("corrections", []),
            "verification_steps": verification_steps
        }
    
    async def revise_content(state: VerificationState):
        """Revise content based on verification results"""
        llm = get_llm(config, temperature=0.1)
        
        revise_prompt = ChatPromptTemplate.from_template(
            "The following content may contain inaccuracies:\n\n"
            "{content}\n\n"
            "Corrections needed:\n{corrections}\n\n"
            "Please revise the content to correct these issues while maintaining the same overall structure "
            "and information intent. Only fix the problematic parts."
        )
        
        corrections = "\n".join(state.get("corrections", ["No specific corrections identified"]))
        
        chain = revise_prompt | llm | StrOutputParser()
        revised = await chain.ainvoke({
            "content": state["content"],
            "corrections": corrections
        })
        
        # Add verification step
        verification_steps = state.get("verification_steps", [])
        verification_steps.append("Content revised based on verification results")
        
        return {
            "verified_content": revised,
            "verification_steps": verification_steps
        }
    
    def need_revision(state: VerificationState) -> str:
        """Determine if content needs revision"""
        if not state.get("is_verified", True) and state.get("corrections", []):
            return "revise_content"
        else:
            # No issues found or no corrections available
            state["verified_content"] = state["content"]
            return "output"
    
    # Create the verification graph
    graph = StateGraph(VerificationState)
    
    # Add nodes
    graph.add_node("initial_verification", initial_verification)
    graph.add_node("deep_verification", deep_verification)
    graph.add_node("revise_content", revise_content)
    
    # Add edges
    graph.add_edge(START, "initial_verification")
    graph.add_edge("initial_verification", "deep_verification")
    graph.add_conditional_edges("deep_verification", need_revision, {
        "revise_content": "revise_content",
        "output": END
    })
    graph.add_edge("revise_content", END)
    
    return graph.compile()

def create_research_graph() -> StateGraph:
    """Create and configure the research graph with hallucination prevention"""
    config = ResearchConfiguration()
    
    # Initialize verification subgraph
    verification_subgraph = create_fact_verification_subgraph(config)
    
    graph = StateGraph(ResearchState)
    
    # --- Define core nodes ---
    
    async def process_search_query(state: ResearchState, config: RunnableConfig = None) -> Dict[str, Any]:
        """Process the initial search query to extract research parameters with CoT reasoning"""
        cfg = ResearchConfiguration.from_runnable_config(config)
        llm = get_llm(cfg, temperature=0.2)
        
        # Extract search query from messages
        query = ""
        for msg in state.get("messages", []):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break

        # Heuristic extraction: if the user's query contains a JIRA or MTV identifier, skip LLM
        jira_match = re.search(r"([A-Z]+-\d+)", query)
        mtv_match = re.search(r"MTV\d{4,}", query, re.IGNORECASE)
        if mtv_match:
            extracted = mtv_match.group(0).upper()
            logger.info(f"Heuristically extracted MTV identifier from query: {extracted}")
            return {
                "research_topic": extracted,
                "search_query": extracted,
                "research_loop_count": 0,
                "running_summary": f"Researching identifier {extracted}"
            }
        if jira_match:
            extracted = jira_match.group(1)
            logger.info(f"Heuristically extracted JIRA key from query: {extracted}")
            return {
                "research_topic": extracted,
                "search_query": extracted,
                "research_loop_count": 0,
                "running_summary": f"Researching JIRA issue {extracted}"
            }

        if not query:
            logger.warning("No query found in messages. Using default values.")
            # Attempt heuristic extraction of potential identifiers (e.g. JIRA keys like ABC-123 or MTV2212)
            jira_match = re.search(r"([A-Z]+-\d+)", query)
            mtv_match = re.search(r"MTV\d{4,}", query, re.IGNORECASE)

            if jira_match:
                extracted = jira_match.group(1)
                logger.info(f"Heuristically extracted JIRA key from query: {extracted}")
                return {
                    "research_topic": extracted,
                    "search_query": extracted,
                    "research_loop_count": 0,
                    "running_summary": f"Researching JIRA issue {extracted}"
                }

            if mtv_match:
                extracted = mtv_match.group(0).upper()
                logger.info(f"Heuristically extracted MTV identifier from query: {extracted}")
                return {
                    "research_topic": extracted,
                    "search_query": extracted,
                    "research_loop_count": 0,
                    "running_summary": f"Researching identifier {extracted}"
                }

            # Attempt heuristic extraction when structured extraction fails
            jira_match2 = re.search(r"([A-Z]+-\d+)", query)
            mtv_match2 = re.search(r"MTV\d{4,}", query, re.IGNORECASE)

            if jira_match2:
                extracted = jira_match2.group(1)
                logger.info(f"Heuristically extracted JIRA key from query after extraction failure: {extracted}")
                return {
                    "research_topic": extracted,
                    "search_query": extracted,
                    "research_loop_count": 0,
                    "running_summary": f"Researching JIRA issue {extracted}"
                }

            if mtv_match2:
                extracted = mtv_match2.group(0).upper()
                logger.info(f"Heuristically extracted MTV identifier from query after extraction failure: {extracted}")
                return {
                    "research_topic": extracted,
                    "search_query": extracted,
                    "research_loop_count": 0,
                    "running_summary": f"Researching identifier {extracted}"
                }

            return {
                "research_topic": query,
                "search_query": query,
                "research_loop_count": 0,
                "running_summary": f"Researching: {query}"
            }
        
        # Define prompt for extracting research parameters with Chain-of-Thought
        extract_prompt = ChatPromptTemplate.from_template(
            "Extract the core research topic and specific search query from this user request:\n\n"
            "{query}\n\n"
            "Think step by step:\n"
            "1. What is the main subject or domain being asked about?\n"
            "2. What specific information is being requested?\n"
            "3. What would be the most effective search query to find this information?\n"
            "4. Is there any ambiguity I need to resolve?\n\n"
            "Based on this analysis, provide a JSON with two fields:\n"
            "1. 'research_topic': The central topic to research (short phrase)\n"
            "2. 'search_query': The specific search query that would yield good results\n"
        )
        
        # Create chain with structured output using Pydantic model
        extract_chain = extract_prompt | llm.with_structured_output(ResearchParameters)
        
        # Execute chain
        try:
            result = await extract_chain.ainvoke({"query": query})
            
            # Make sure we have a valid result
            if not result or not hasattr(result, 'research_topic') or not result.research_topic:
                logger.warning(f"Invalid or empty research parameters returned. Using query directly.")
                return {
                    "research_topic": query,
                    "search_query": query,
                    "research_loop_count": 0,
                    "running_summary": f"Researching: {query}"
                }
                
            return {
                "research_topic": result.research_topic,
                "search_query": result.search_query,
                "research_loop_count": 0,
                "running_summary": f"Researching: {result.research_topic}"
            }
        except Exception as e:
            logger.error(f"Error extracting research parameters: {e}")
            # Fallback: use query directly
            return {
                "research_topic": query,
                "search_query": query,
                "research_loop_count": 0,
                "running_summary": f"Researching: {query}"
            }
    
    async def retrieve_context(state: ResearchState, config: RunnableConfig = None) -> Dict[str, Any]:
        """Retrieve relevant context using RAG to ground the research"""
        cfg = ResearchConfiguration.from_runnable_config(config)
        
        if not cfg.enable_rag:
            logger.info("RAG disabled. Skipping context retrieval.")
            return {"retrieved_context": []}
        
        # Initialize retriever
        retriever = RAGRetriever()
        
        # Retrieve relevant context based on the search query
        try:
            context = await retriever.retrieve(state["search_query"])
            logger.info(f"Retrieved {len(context)} context items using RAG")
            
            return {"retrieved_context": context}
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return {"retrieved_context": []}
    
    # --- Source Processing Nodes ---
    # These would be fully implemented as separate modules
    # For brevity, we're only showing placeholder implementations
    
    async def fetch_jira_data(state: ResearchState, config: RunnableConfig = None) -> Dict[str, Any]:
        """Fetch data from Jira"""
        cfg = ResearchConfiguration.from_runnable_config(config)
        
        if not cfg.enable_jira:
            logger.info("Jira search disabled. Skipping.")
            return {
                "jira_data": {
                    "raw_results": [],
                    "current_chunk_index": 0,
                    "total_chunks": 0
                },
                "raw_jira_results": []
            }
        
        try:
            from src.assistant.sources.jira_runner import search_jira
            from src.assistant.utils.response_formatter import sanitize_binary_content
            
            search_query = state.get("search_query", "")
            if not search_query:
                logger.warning("No search query available for Jira search")
                return {
                    "jira_data": {
                        "raw_results": [],
                        "current_chunk_index": 0,
                        "total_chunks": 0
                    },
                    "raw_jira_results": []
                }
            
            # Limit result count for fast mode
            max_results = 10 if cfg.fast_mode else 30
            
            try:
                # Extract potential identifiers from query
                jira_id = None
                mtv_id = None
                
                jira_match = re.search(r'([A-Z]+-\d+)', search_query)
                if jira_match:
                    jira_id = jira_match.group(1)
                    
                mtv_match = re.search(r'MTV\d{4,}', search_query, re.IGNORECASE)
                if mtv_match:
                    mtv_id = mtv_match.group(0).upper()
                
                # Call search function
                result = await search_jira(
                    jql=None if jira_id is None else f'key = "{jira_id}"',
                    keyword=search_query if jira_id is None else None,
                    mtv_id=mtv_id,
                    max_results=max_results,
                    include_attachments=not cfg.fast_mode  # Skip attachments in fast mode
                )
                
                # Handle standardized response format
                if isinstance(result, dict) and "status" in result:
                    if result["status"] == "error":
                        logger.error(f"Jira search error: {result.get('error', 'Unknown error')}")
                        # Handle error but continue the pipeline
                        return {
                            "jira_data": {
                                "raw_results": [],
                                "current_chunk_index": 0,
                                "total_chunks": 0,
                                "error": result.get("error", "Unknown error")
                            },
                            "raw_jira_results": []
                        }
                    
                    # Extract the successful data
                    raw_results = result.get("data", {}).get("issues", [])
                else:
                    # Legacy format handling
                    raw_results = result.get("issues", []) if isinstance(result, dict) else []
                
                # Sanitize any binary content
                clean_results = sanitize_binary_content(raw_results) 
                
                # Calculate chunks
                chunked_results = list(process_in_batches(clean_results, batch_size=3))
                total_chunks = len(chunked_results)
                
                logger.info(f"Found {len(clean_results)} Jira issues for query: {search_query}")
                
                return {
                    "jira_data": {
                        "raw_results": clean_results,
                        "current_chunk_index": 0,
                        "total_chunks": total_chunks,
                        "chunked_results": chunked_results
                    },
                    "raw_jira_results": clean_results
                }
            except Exception as tool_error:
                logger.error(f"Error during Jira search: {tool_error}", exc_info=True)
                return {
                    "jira_data": {
                        "raw_results": [],
                        "current_chunk_index": 0,
                        "total_chunks": 0,
                        "error": str(tool_error)
                    }
                }
        except ImportError as e:
            logger.error(f"Jira module import error: {e}")
            return {
                "jira_data": {
                    "raw_results": [],
                    "current_chunk_index": 0,
                    "total_chunks": 0,
                    "error": f"Import error: {str(e)}"
                }
            }
                
    async def fetch_perforce_data(state: ResearchState, config: RunnableConfig = None) -> Dict[str, Any]:
        """Fetch data from Perforce"""
        cfg = ResearchConfiguration.from_runnable_config(config)
        
        if not cfg.enable_perforce:
            logger.info("Perforce search disabled. Skipping.")
            return {
                "perforce_data": {
                    "raw_results": [],
                    "current_chunk_index": 0,
                    "total_chunks": 0
                },
                "raw_perforce_results": []
            }
        
        try:
            from src.assistant.sources.perforce import search_perforce
            from src.assistant.utils.response_formatter import sanitize_binary_content
            
            search_query = state.get("search_query", "")
            if not search_query:
                logger.warning("No search query available for Perforce search")
                return {
                    "perforce_data": {
                        "raw_results": [],
                        "current_chunk_index": 0,
                        "total_chunks": 0
                    },
                    "raw_perforce_results": []
                }
            
            # Limit result count for fast mode
            max_results = 10 if cfg.fast_mode else 25
            
            try:
                # Extract potential identifiers from query
                mtv_id = None
                mtv_match = re.search(r'MTV\d{4,}', search_query, re.IGNORECASE)
                if mtv_match:
                    mtv_id = mtv_match.group(0).upper()
                
                # Initialize Perforce helper
                p4_helper = PerforceHelper()
                
                # Call search function
                result = await search_perforce(query=search_query, mtv_id=mtv_id, max_results=max_results)
                
                # Handle standardized response format
                if isinstance(result, dict) and "status" in result:
                    if result["status"] == "error":
                        logger.error(f"Perforce search error: {result.get('error', 'Unknown error')}")
                        # Handle error but continue the pipeline
                        return {
                            "perforce_data": {
                                "raw_results": [],
                                "current_chunk_index": 0,
                                "total_chunks": 0,
                                "error": result.get("error", "Unknown error")
                            },
                            "raw_perforce_results": []
                        }
                    
                    # Extract the successful data
                    raw_results = result.get("data", {}).get("perforce_data", [])
                else:
                    # Legacy format handling
                    raw_results = result
                
                # Sanitize any binary content
                clean_results = sanitize_binary_content(raw_results)
                
                # Calculate chunks
                chunked_results = list(process_in_batches(clean_results, batch_size=5))
                total_chunks = len(chunked_results)
                
                logger.info(f"Found {len(clean_results)} Perforce changelists for query: {search_query}")
                
                return {
                    "perforce_data": {
                        "raw_results": clean_results,
                        "current_chunk_index": 0,
                        "total_chunks": total_chunks,
                        "chunked_results": chunked_results
                    },
                    "raw_perforce_results": clean_results
                }
            except Exception as tool_error:
                logger.error(f"Error during Perforce search: {tool_error}", exc_info=True)
                return {
                    "perforce_data": {
                        "raw_results": [],
                        "current_chunk_index": 0,
                        "total_chunks": 0,
                        "error": str(tool_error)
                    }
                }
        except ImportError as e:
            logger.error(f"Perforce module import error: {e}")
            return {
                "perforce_data": {
                    "raw_results": [],
                    "current_chunk_index": 0,
                    "total_chunks": 0,
                    "error": f"Import error: {str(e)}"
                }
            }
    
    async def fetch_confluence_data(state: ResearchState, config: RunnableConfig = None) -> Dict[str, Any]:
        """Fetch data from Confluence"""
        cfg = ResearchConfiguration.from_runnable_config(config)
        
        if not cfg.enable_confluence:
            logger.info("Confluence search disabled. Skipping.")
            return {
                "confluence_data": {
                    "raw_results": [],
                    "current_chunk_index": 0,
                    "total_chunks": 0
                },
                "raw_confluence_results": []
            }
        
        try:
            from src.assistant.sources.confluence import search_confluence
            from src.assistant.utils.response_formatter import sanitize_binary_content
            
            search_query = state.get("search_query", "")
            if not search_query:
                logger.warning("No search query available for Confluence search")
                return {
                    "confluence_data": {
                        "raw_results": [],
                        "current_chunk_index": 0,
                        "total_chunks": 0
                    },
                    "raw_confluence_results": []
                }
            
            # Limit result count for fast mode
            max_results = 5 if cfg.fast_mode else 15
            
            try:
                # Call search function
                result = await search_confluence(
                    query=search_query,
                    max_results=max_results
                )
                
                # Handle standardized response format
                if isinstance(result, dict) and "status" in result:
                    if result["status"] == "error":
                        logger.error(f"Confluence search error: {result.get('error', 'Unknown error')}")
                        # Handle error but continue the pipeline
                        return {
                            "confluence_data": {
                                "raw_results": [],
                                "current_chunk_index": 0,
                                "total_chunks": 0,
                                "error": result.get("error", "Unknown error")
                            },
                            "raw_confluence_results": []
                        }
                    
                    # Extract the successful data
                    raw_results = result.get("data", {}).get("pages", [])
                else:
                    # Legacy format handling
                    raw_results = result.get("pages", []) if isinstance(result, dict) else []
                
                # Sanitize any binary content
                clean_results = sanitize_binary_content(raw_results)
                
                # Calculate chunks
                chunked_results = list(process_in_batches(clean_results, batch_size=3))
                total_chunks = len(chunked_results)
                
                logger.info(f"Found {len(clean_results)} Confluence pages for query: {search_query}")
                
                return {
                    "confluence_data": {
                        "raw_results": clean_results,
                        "current_chunk_index": 0,
                        "total_chunks": total_chunks,
                        "chunked_results": chunked_results
                    },
                    "raw_confluence_results": clean_results
                }
            except Exception as tool_error:
                logger.error(f"Error during Confluence search: {tool_error}", exc_info=True)
                return {
                    "confluence_data": {
                        "raw_results": [],
                        "current_chunk_index": 0,
                        "total_chunks": 0, 
                        "error": str(tool_error)
                    },
                    "raw_confluence_results": []
                }
        except ImportError as e:
            logger.error(f"Confluence module import error: {e}")
            return {
                "confluence_data": {
                    "raw_results": [],
                    "current_chunk_index": 0,
                    "total_chunks": 0,
                    "error": f"Import error: {str(e)}"
                }
            }
    
    async def generate_final_summary(state: ResearchState, config: RunnableConfig = None) -> Dict[str, Any]:
        """Generate final comprehensive research summary with fact verification"""
        cfg = ResearchConfiguration.from_runnable_config(config)
        llm = get_llm(cfg, temperature=0.1)  # Lower temperature for factual accuracy
        
        # If we already have a final summary from one of the enhanced methods, use it
        if state.get("final_summary"):
            logger.info("Using existing final summary from enhanced methods")
            
            # Calculate hallucination score if not already set
            hallucination_score = state.get("hallucination_score", 0.5)
            
            # Format response for user
            response_message = AIMessage(content=state["final_summary"])
            
            return {
                "messages": [response_message],
                "completed_research": True,
                "hallucination_score": hallucination_score
            }
        
        # Otherwise, generate a summary using the standard approach
        
        # Collect all source summaries - ensure we have default values to avoid undefined variables
        jira_summary = state.get("jira_data", {}).get("summary", "No Jira data available.")
        confluence_summary = state.get("confluence_data", {}).get("summary", "No Confluence data available.")
        perforce_summary = state.get("perforce_data", {}).get("summary", "No Perforce data available.")
        web_summary = state.get("web_data", {}).get("summary", "No web research data available.")
        cross_source = state.get("cross_source_analysis", {}).get("text", "No cross-source analysis available.")
        
        # Include retrieved context from RAG if available
        retrieved_context = "\n\n".join(state.get("retrieved_context", []))
        
        # -------------------- Build citation list --------------------
        citations_list: List[str] = []
        for _iss in state.get('jira_data', {}).get('raw_results', []):
            if isinstance(_iss, dict) and _iss.get('url') and _iss.get('key'):
                citations_list.append(f"[JIRA {_iss['key']}]({_iss['url']})")
        for _pg in state.get('confluence_data', {}).get('raw_results', []):
            if isinstance(_pg, dict) and _pg.get('url'):
                title = _pg.get('title') or _pg.get('content', {}).get('title') or 'Page'
                citations_list.append(f"[Confluence: {title}]({_pg['url']})")
        for _cl in state.get('perforce_data', {}).get('raw_results', []):
            if isinstance(_cl, dict) and _cl.get('url'):
                cl_id = _cl.get('change') or _cl.get('cl') or _cl.get('id') or 'CL'
                citations_list.append(f"[CL {cl_id}]({_cl['url']})")

        citations_text = "\n".join(f"{idx+1}. {c}" for idx, c in enumerate(citations_list)) or "No source links available."
        
        # -------------------- Build timeline / chronology --------------------
        def _parse_dt(val: Any) -> Optional[datetime]:
            if not val:
                return None
            if isinstance(val, (int, float)):
                try:
                    return datetime.fromtimestamp(float(val))
                except Exception:
                    return None
            if isinstance(val, str):
                for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                    try:
                        return datetime.strptime(val, fmt)
                    except Exception:
                        continue
            return None

        events: List[Tuple[datetime, str]] = []

        # JIRA dates
        for _iss in state.get('jira_data', {}).get('raw_results', []):
            if not isinstance(_iss, dict):
                continue
            dt_raw = _iss.get('fields', {}).get('created') or _iss.get('created')
            dt = _parse_dt(dt_raw)
            if dt:
                events.append((dt, f"JIRA {_iss.get('key', '')} created"))
        # Perforce dates
        for _cl in state.get('perforce_data', {}).get('raw_results', []):
            if not isinstance(_cl, dict):
                continue
            dt_raw = _cl.get('date') or _cl.get('time')
            dt = _parse_dt(dt_raw)
            if dt:
                cl_id = _cl.get('change') or _cl.get('cl') or _cl.get('id') or ''
                events.append((dt, f"CL {cl_id} submitted"))
        # Confluence dates
        for _pg in state.get('confluence_data', {}).get('raw_results', []):
            if not isinstance(_pg, dict):
                continue
            dt_raw = _pg.get('version', {}).get('when') or _pg.get('createdDate')
            dt = _parse_dt(dt_raw)
            if dt:
                title = _pg.get('title') or _pg.get('content', {}).get('title') or 'Page'
                events.append((dt, f"Confluence: {title} updated"))

        events.sort(key=lambda x: x[0])
        timeline_text = "\n".join([f"{dt.strftime('%Y-%m-%d')}: {label}" for dt, label in events][:30]) or "No dated events extracted."

        # -------------------- Build simple risk matrix --------------------
        risk_counts: Dict[str, int] = {}
        for _iss in state.get('jira_data', {}).get('raw_results', []):
            if not isinstance(_iss, dict):
                continue
            prio = (_iss.get('fields', {}).get('priority', {}) or {}).get('name') or _iss.get('priority', {}).get('name') if isinstance(_iss.get('priority'), dict) else _iss.get('priority')
            prio = (prio or 'Unspecified').title()
            risk_counts[prio] = risk_counts.get(prio, 0) + 1

        # build markdown table
        if risk_counts:
            header = "| Priority | # JIRA Issues |\n|---|---|"
            rows = [f"| {p} | {c} |" for p, c in sorted(risk_counts.items(), key=lambda x: x[0])]
            risk_matrix_text = "\n".join([header] + rows)
        else:
            risk_matrix_text = "No risk data available."

        # Build template text first to avoid relying on internal attributes
        base_template = (
            "Create a comprehensive research report on: {topic}\n\n"
            "I want you to think step-by-step to ensure accuracy and avoid hallucinations:\n\n"
            "1. First, examine what information we have from each source:\n"
            "JIRA DATA:\n{jira_summary}\n\n"
            "CONFLUENCE DATA:\n{confluence_summary}\n\n"
            "PERFORCE CODE DATA:\n{perforce_summary}\n\n"
            "WEB RESEARCH:\n{web_summary}\n\n"
            "CROSS-SOURCE ANALYSIS:\n{cross_source}\n\n"
            "ADDITIONAL CONTEXT:\n{retrieved_context}\n\n"
            "2. For each piece of information you include, verify it appears in at least one of the sources above.\n"
            "3. If there are contradictions between sources, acknowledge them explicitly.\n"
            "4. Do not add speculative information beyond what's in the sources.\n"
            "5. If information is missing on a particular subtopic, state this explicitly rather than making up details.\n\n"
            "Structure the report with these sections:\n"
            "1. Executive Summary\n"
            "2. Requirements Analysis (from Jira)\n"
            "3. Documentation Review (from Confluence)\n"
            "4. Implementation Analysis (from Perforce)\n"
            "5. External Context (from web research)\n"
            "6. Integrated Analysis (connections between sources)\n"
            "7. Timeline / Chronology\n{timeline}\n"
            "8. Risk Matrix\n{risk_matrix}\n"
            "9. Key Findings and Recommendations\n"
            "10. Information Gaps and Uncertainties\n"
            "11. Sources and References (markdown links):\n{citations}\n\n"
            "Focus on providing comprehensive technical details, highlighting contradictions, "
            "noting implementation gaps, and making specific technical recommendations."
        )

        if cfg.enable_cove:
            base_template += (
                "\n\nAfter drafting your report, verify each claim by asking:\n"
                "1. Is this statement directly supported by the source data?\n"
                "2. Am I making any assumptions not explicit in the data?\n"
                "3. Am I contradicting information from any source?\n"
                "Remove or flag any content that doesn't pass this verification."
            )

        summary_prompt = ChatPromptTemplate.from_template(base_template)
        
        summary_chain = summary_prompt | llm | StrOutputParser()
        
        try:
            # Generate final summary
            final_summary = await summary_chain.ainvoke({
                "topic": state["research_topic"],
                "jira_summary": jira_summary,
                "confluence_summary": confluence_summary,
                "perforce_summary": perforce_summary,
                "web_summary": web_summary,
                "cross_source": cross_source,
                "retrieved_context": retrieved_context or "No additional context available.",
                "citations": citations_text,
                "timeline": timeline_text,
                "risk_matrix": risk_matrix_text
            })
            
            # Verify the final summary
            hallucination_score = 0.5  # Default middle value
            
            if cfg.enable_verification:
                verifier = SelfVerifier(get_llm(cfg, temperature=0.1))
                verification_result = await verifier.verify_content(final_summary)
                
                # Calculate hallucination score (0-1, lower is better)
                # Based on verification confidence and any corrections
                if not verification_result["is_verified"]:
                    # Higher score indicates more potential hallucinations
                    hallucination_score = 1.0 - verification_result.get("confidence", 0.5)
                    
                    # If corrections were suggested, apply them for critical issues
                    if verification_result.get("corrections") and hallucination_score > 0.7:
                        logger.warning("High hallucination detected, applying corrections")
                        correction_prompt = ChatPromptTemplate.from_template(
                            "The following research summary may contain inaccuracies:\n\n"
                            "{summary}\n\n"
                            "Corrections needed:\n{corrections}\n\n"
                            "Please revise the summary to address these issues while maintaining the same structure."
                        )
                        
                        correction_chain = correction_prompt | llm | StrOutputParser()
                        final_summary = await correction_chain.ainvoke({
                            "summary": final_summary,
                            "corrections": "\n".join(verification_result["corrections"])
                        })
                        
                        # Reduce score after corrections
                        hallucination_score = max(0.3, hallucination_score - 0.3)
                else:
                    # Good verification result - low hallucination score
                    hallucination_score = 0.1
            
            # Format response for user
            response_message = AIMessage(content=final_summary)
            
            return {
                "final_summary": final_summary,
                "messages": [response_message],
                "completed_research": True,
                "hallucination_score": hallucination_score
            }
        except Exception as e:
            logger.error(f"Error generating final summary: {e}")
            error_msg = AIMessage(content=f"Research encountered an error: {str(e)}")
            return {
                "messages": [error_msg],
                "completed_research": True,
                "hallucination_score": 1.0  # Highest score indicates failure
            }

    # -------------------- Processing helpers --------------------
    def _safe_truncate(text: str, limit: int = 120) -> str:
        """Truncate text for log/summary purposes, ensuring no newlines."""
        if not text:
            return ""
        text = text.replace("\n", " ").replace("\r", " ")
        return text[:limit] + ("…" if len(text) > limit else "")

    # -------------------- JIRA --------------------
    async def process_jira_data(state: ResearchState, config: RunnableConfig = None) -> Dict[str, Any]:
        """Convert raw JIRA dicts into short human-readable strings and summary."""
        jira_data = state.get("jira_data", {}) or {}
        raw_results = jira_data.get("raw_results", []) or []

        cfg = ResearchConfiguration.from_runnable_config(config)
        llm = get_llm(cfg, temperature=0.15)

        bullet_prompt = ChatPromptTemplate.from_template(
            "Summarise the following JIRA issue JSON in one concise bullet (max 40 words, no line breaks). "
            "Include the issue key and a direct link if available:\n\n{issue_json}"
        )
        bullet_chain = bullet_prompt | llm | StrOutputParser()

        processed_results: List[str] = []
        limit = 3 if cfg.fast_mode else 10
        for idx, issue in enumerate(raw_results):
            if idx >= limit:
                break
            if not isinstance(issue, dict):
                continue
            try:
                bullet = await bullet_chain.ainvoke({"issue_json": json.dumps(issue)})
                processed_results.append(bullet.strip())
            except Exception as e:
                logger.debug(f"LLM jira bullet error: {e}")
                key = issue.get("key", "UNKN")
                summary = issue.get("summary") or issue.get("fields", {}).get("summary") or "No summary"
                processed_results.append(f"{key} – {_safe_truncate(summary)} (fallback)")
        
        # Create a concise summary
        summary_txt = (
            f"Processed {len(raw_results)} JIRA issues." if raw_results else "No JIRA data to process."
        )

        jira_data.update({
            "processed_results": processed_results,
            "summary": summary_txt,
        })

        return {"jira_data": jira_data}

    # -------------------- PERFORCE --------------------
    async def process_perforce_data(state: ResearchState, config: RunnableConfig = None) -> Dict[str, Any]:
        """Summarise Perforce change dictionaries."""
        p4_data = state.get("perforce_data", {}) or {}
        raw_results = p4_data.get("raw_results", []) or []

        cfg = ResearchConfiguration.from_runnable_config(config)
        llm = get_llm(cfg, temperature=0.15)

        bullet_prompt = ChatPromptTemplate.from_template(
            "Summarise the following Perforce changelist JSON in one concise bullet (max 40 words, no line breaks). "
            "Include the changelist number and a direct link if available:\n\n{cl_json}"
        )
        bullet_chain = bullet_prompt | llm | StrOutputParser()

        processed_results: List[str] = []
        limit = 3 if cfg.fast_mode else 10
        for idx, change in enumerate(raw_results):
            if idx >= limit:
                break
            if not isinstance(change, dict):
                continue
            try:
                bullet = await bullet_chain.ainvoke({"cl_json": json.dumps(change)})
                processed_results.append(bullet.strip())
            except Exception as e:
                logger.debug(f"LLM perforce bullet error: {e}")
                cl = change.get("change") or change.get("cl") or change.get("id") or "?"
                desc = change.get("desc") or change.get("description") or "No description"
                processed_results.append(f"CL {cl} – {_safe_truncate(desc)} (fallback)")

        summary_txt = (
            f"Processed {len(raw_results)} Perforce changelists." if raw_results else "No Perforce data to process."
        )

        p4_data.update({
            "processed_results": processed_results,
            "summary": summary_txt,
        })

        return {"perforce_data": p4_data}

    # -------------------- CONFLUENCE --------------------
    async def process_confluence_data(state: ResearchState, config: RunnableConfig = None) -> Dict[str, Any]:
        """Summarise Confluence page dictionaries."""
        conf_data = state.get("confluence_data", {}) or {}
        raw_results = conf_data.get("raw_results", []) or []

        cfg = ResearchConfiguration.from_runnable_config(config)
        llm = get_llm(cfg, temperature=0.15)

        bullet_prompt = ChatPromptTemplate.from_template(
            "Summarise the following Confluence page JSON in one concise bullet (max 40 words, no line breaks). "
            "Include the page title and a direct link if available:\n\n{page_json}"
        )
        bullet_chain = bullet_prompt | llm | StrOutputParser()

        processed_results: List[str] = []
        limit = 3 if cfg.fast_mode else 10
        for idx, page in enumerate(raw_results):
            if idx >= limit:
                break
            if not isinstance(page, dict):
                continue
            try:
                bullet = await bullet_chain.ainvoke({"page_json": json.dumps(page)})
                processed_results.append(bullet.strip())
            except Exception as e:
                logger.debug(f"LLM confluence bullet error: {e}")
                title = page.get("title") or page.get("content", {}).get("title") or "Untitled"
                space = page.get("space") or page.get("content", {}).get("space", {}).get("key")
                processed_results.append(f"{space or 'Space?'} – {_safe_truncate(title)} (fallback)")

        summary_txt = (
            f"Processed {len(raw_results)} Confluence pages." if raw_results else "No Confluence data to process."
        )

        conf_data.update({
            "processed_results": processed_results,
            "summary": summary_txt,
        })

        return {"confluence_data": conf_data}

    # -------------------- SUMMARIZE DATA --------------------
    async def summarize_data(state: ResearchState, config: RunnableConfig = None) -> Dict[str, Any]:
        """Combine individual source summaries into the running_summary."""
        parts = []
        for src_key, label in [
            ("jira_data", "JIRA"),
            ("confluence_data", "Confluence"),
            ("perforce_data", "Perforce"),
        ]:
            dat = state.get(src_key, {}) or {}
            parts.append(f"{label}: {dat.get('summary', 'No data')}")

        running_summary = " | ".join(parts)

        return {"running_summary": running_summary}

    # -------------------- Cross-Source Analysis --------------------
    from src.assistant.analyzers import cross_source_analyzer as _cs

    async def apply_cross_source_analysis(state: ResearchState, config: RunnableConfig = None) -> Dict[str, Any]:
        """Delegate to the full analyzer module to compute cross-source links."""
        try:
            return await _cs.apply_cross_source_analysis(state, config)
        except Exception as e:
            logger.error(f"Cross-source analysis failed: {e}")
            return {}

    # -------------------- Map-Rerank Summariser --------------------
    from src.assistant.summarizers.map_rerank import MapRerankSummarizer

    async def apply_map_rerank(state: ResearchState, config: RunnableConfig = None) -> Dict[str, Any]:
        cfg = ResearchConfiguration.from_runnable_config(config)
        if not cfg.enable_map_rerank:
            logger.info("Map-Rerank disabled. Skipping.")
            return {}

        try:
            llm = get_llm(cfg, temperature=0.15)
            summarizer = MapRerankSummarizer(llm)

            # Build source_data from processed results for richer context
            source_data = {
                "jira": "\n".join(state.get("jira_data", {}).get("processed_results", [])),
                "confluence": "\n".join(state.get("confluence_data", {}).get("processed_results", [])),
                "perforce": "\n".join(state.get("perforce_data", {}).get("processed_results", [])),
            }

            result = await summarizer.generate_summaries(
                topic=state.get("research_topic", state.get("search_query", "")),
                source_data=source_data,
                context=state.get("retrieved_context", []),
                num_candidates=3 if cfg.fast_mode else 4,
            )

            return {"map_rerank_results": result, "final_summary": result["selected_summary"]}
        except Exception as e:
            logger.error(f"Map-Rerank summariser failed: {e}")
            return {}

    # -------------------- Chain of Knowledge --------------------
    from src.assistant.verifiers.chain_of_knowledge import ChainOfKnowledge

    async def apply_chain_of_knowledge(state: ResearchState, config: RunnableConfig = None) -> Dict[str, Any]:
        cfg = ResearchConfiguration.from_runnable_config(config)
        if not cfg.enable_cok:
            logger.info("Chain-of-Knowledge disabled. Skipping.")
            return {}
        try:
            llm = get_llm(cfg, temperature=0.15)
            cok = ChainOfKnowledge(llm)

            source_data = {
                "jira": "\n".join(state.get("jira_data", {}).get("processed_results", [])),
                "confluence": "\n".join(state.get("confluence_data", {}).get("processed_results", [])),
                "perforce": "\n".join(state.get("perforce_data", {}).get("processed_results", [])),
            }

            result = await cok.generate_with_expert_citations(
                topic=state.get("research_topic", state.get("search_query", "")),
                source_data=source_data,
            )

            return {"chain_of_knowledge_results": result}
        except Exception as e:
            logger.error(f"Chain-of-Knowledge failed: {e}")
            return {}

    # -------------------- Multi-Agent Verification --------------------
    from src.assistant.verifiers.multi_agent_verification import MultiAgentVerifier

    async def apply_multi_agent_verification(state: ResearchState, config: RunnableConfig = None) -> Dict[str, Any]:
        cfg = ResearchConfiguration.from_runnable_config(config)
        if not cfg.enable_multi_agent:
            logger.info("Multi-Agent verification disabled. Skipping.")
            return {}
        try:
            llm = get_llm(cfg, temperature=0.1)
            verifier = MultiAgentVerifier(llm)

            content = state.get("final_summary") or state.get("running_summary", "")
            if not content:
                logger.info("No content available for multi-agent verification.")
                return {}

            result = await verifier.verify_content(content, context=state.get("retrieved_context", []))
            return {"multi_agent_verification_results": result}
        except Exception as e:
            logger.error(f"Multi-Agent verification failed: {e}")
            return {}

    # --- Add nodes to the research graph --------------------------------------
    graph.add_node("process_query", process_search_query)
    graph.add_node("retrieve_context", retrieve_context)

    # Source processing nodes
    graph.add_node("fetch_jira", fetch_jira_data)
    graph.add_node("fetch_perforce", fetch_perforce_data)
    graph.add_node("fetch_confluence", fetch_confluence_data)
    graph.add_node("process_jira", process_jira_data)
    graph.add_node("process_perforce", process_perforce_data)
    graph.add_node("process_confluence", process_confluence_data)
    graph.add_node("summarize_data", summarize_data)

    # Enhanced feature nodes
    graph.add_node("apply_cross_source_analysis", apply_cross_source_analysis)
    graph.add_node("apply_map_rerank", apply_map_rerank)
    graph.add_node("apply_chain_of_knowledge", apply_chain_of_knowledge)
    graph.add_node("apply_multi_agent_verification", apply_multi_agent_verification)

    # Final summary generation
    graph.add_node("generate_final_summary", generate_final_summary)

    # --- Connect nodes in the graph -------------------------------------------
    graph.add_edge(START, "process_query")
    graph.add_edge("process_query", "retrieve_context")
    graph.add_edge("retrieve_context", "fetch_jira")
    graph.add_edge("retrieve_context", "fetch_perforce")
    graph.add_edge("retrieve_context", "fetch_confluence")

    # Independent processing chains for each source
    graph.add_edge("fetch_jira", "process_jira")
    graph.add_edge("fetch_perforce", "process_perforce")
    graph.add_edge("fetch_confluence", "process_confluence")

    # Converge processed data into summarization node
    graph.add_edge("process_jira", "summarize_data")
    graph.add_edge("process_perforce", "summarize_data")
    graph.add_edge("process_confluence", "summarize_data")

    # Enhanced features pipeline
    graph.add_edge("summarize_data", "apply_cross_source_analysis")
    graph.add_edge("apply_cross_source_analysis", "apply_chain_of_knowledge")
    graph.add_edge("apply_chain_of_knowledge", "apply_map_rerank")
    graph.add_edge("apply_map_rerank", "apply_multi_agent_verification")
    graph.add_edge("apply_multi_agent_verification", "generate_final_summary")
    graph.add_edge("generate_final_summary", END)

    # Add checkpoint memory and compile the graph
    memory = InMemorySaver()
    return graph.compile(checkpointer=memory)

# --- Main Entry Point --------------------------------------------------------
async def research(query: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Main entry point for the research system"""
    # Initialize configuration
    research_config = ResearchConfiguration(config.get("configurable", {}) if config else None)
    # Build the research graph
    graph = create_research_graph()
    # Prepare initial state
    state = {
        "research_topic": "",
        "search_query": "",
        "running_summary": "",
        "research_loop_count": 0,
        "messages": [HumanMessage(content=query)],
        "jira_data": {"raw_results": [], "processed_results": [], "current_chunk_index": 0, "total_chunks": 0, "summary": ""},
        "confluence_data": {"raw_results": [], "processed_results": [], "current_chunk_index": 0, "total_chunks": 0, "summary": ""},
        "perforce_data": {"raw_results": [], "processed_results": [], "current_chunk_index": 0, "total_chunks": 0, "summary": ""},
        "web_data": {"raw_results": [], "processed_results": [], "current_chunk_index": 0, "total_chunks": 0, "summary": ""},
        "retrieved_context": [],
        "verification_results": [],
        "map_rerank_results": None,
        "chain_of_knowledge_results": None,
        "multi_agent_verification_results": None,
        "cross_source_analysis": {},
        "cross_source_connections": [],
        "technical_validation": {},
        "security_findings": [],
        "final_summary": "",
        "completed_research": False,
        "hallucination_score": 0.5
    }
    try:
        # Extract configurable for invocation
        invocation_config = {"configurable": config["configurable"]} if isinstance(config, dict) and "configurable" in config else {}
        result = await graph.ainvoke(state, invocation_config)
        return result
    except Exception as e:
        logger.error(f"Error in research: {e}")
        return {
            "messages": [AIMessage(content=f"Research encountered an error: {str(e)}")],
            "completed_research": True,
            "hallucination_score": 1.0
        }

# --- Module Exports ----------------------------------------------------------
__all__ = ['research', 'ResearchState', 'ResearchConfiguration', 'create_research_graph']
