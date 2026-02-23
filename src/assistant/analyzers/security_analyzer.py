# src/assistant/analyzers/security_analyzer.py

import logging
import asyncio
import json
import gc
from typing import Dict, Any, List, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

from src.assistant.core.types import ResearchState
from src.assistant.configuration import Configuration
from src.assistant.utils.helpers import traceable
from src.assistant.utils.langgraph_helpers import clean_llm_output
from src.assistant.sources.perforce_client import SecurityAnalyzer

logger = logging.getLogger(__name__)

# LLM Configuration
LLM_CONFIG_PARAMS = {
    "temperature": 0.15,
    "num_ctx": 8192,
    "top_p": 0.7,
    "repeat_penalty": 1.15,
}

# Constants
MAX_SUMMARY_CONTEXT = 30000

@traceable
async def analyze_security(state: ResearchState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Analyze security implications using static analysis and LLM.
    
    Examines both the raw content via static analysis tools and uses
    an LLM to identify potential security issues in the research content.
    """
    logger.info("--- Node: analyze_security ---")
    
    # Get configuration
    cfg = Configuration.from_runnable_config(config) if config else Configuration()
    findings = []
    
    # --- Static Scan ---
    try:
        # Combine content from formatted results for scanning
        # Use a sample to avoid excessive length
        content_to_scan = "\n\n".join(state.get("enterprise_research_results", [])[:5])  # Scan first 5 blocks
        if len(content_to_scan) > 10000:
            content_to_scan = content_to_scan[:10000]  # Limit length

        if content_to_scan:
            static_findings = SecurityAnalyzer.analyze_file_content(content_to_scan)
            if static_findings:
                logger.info(f"Static security scan found {len(static_findings)} potential issues.")
                findings.extend(static_findings)
        else:
            logger.info("No content available for static security scan.")
    except Exception as static_err:
        logger.error(f"Static security analysis failed: {static_err}", exc_info=True)

    # --- LLM Analysis ---
    summary_for_llm = state.get("running_summary", "")
    if summary_for_llm:
        try:
            # Initialize LLM with JSON format for structured output
            llm = ChatOllama(
                base_url=cfg.ollama_base_url,
                model=cfg.local_llm,
                format="json",  # Request JSON output for easier parsing
                **LLM_CONFIG_PARAMS
            )
            
            # Get security prompt from imported template
            from src.utils.prompts import security_analysis_instructions
            prompt = security_analysis_instructions
            
            # Prepare human message with content to analyze
            human_msg = f"Analyze security implications of:\n{summary_for_llm[:MAX_SUMMARY_CONTEXT]}"

            # Invoke LLM
            result = await asyncio.wait_for(
                llm.ainvoke([SystemMessage(content=prompt), SystemMessage(content=human_msg)]),
                timeout=300  # Adjust timeout for complex analysis
            )
            
            try:
                # Parse and validate LLM output
                llm_parsed = json.loads(result.content)
                llm_found = []
                
                if isinstance(llm_parsed, list):
                    llm_found = llm_parsed
                elif isinstance(llm_parsed, dict):
                    llm_found = llm_parsed.get("findings", []) or llm_parsed.get("issues", [])
                
                # Validate findings format
                validated_llm = [f for f in llm_found if isinstance(f, dict)]
                if validated_llm:
                    logger.info(f"LLM security analysis identified {len(validated_llm)} potential issues.")
                    findings.extend(validated_llm)
            except json.JSONDecodeError as decode_err:
                logger.error(f"Failed to parse LLM security findings JSON: {decode_err}")
        except Exception as llm_err:
            logger.error(f"LLM security analysis failed: {llm_err}", exc_info=True)

    # --- Deduplicate Findings ---
    deduped_findings = []
    seen_keys = set()
    
    for item in findings:
        # Create a unique key based on type, location (if available), and description/match
        key = (
            item.get("type", "unknown").lower(),
            item.get("line_number"),  # Can be None
            (item.get("match") or item.get("description", ""))[:50]  # Use first 50 chars
        )
        
        if key not in seen_keys:
            seen_keys.add(key)
            deduped_findings.append(item)

    logger.info(f"Security analysis complete. Total unique findings: {len(deduped_findings)}")
    
    # Clean up memory
    gc.collect()
    
    return {"security_findings": deduped_findings}
