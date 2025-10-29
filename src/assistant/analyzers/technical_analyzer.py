# src/assistant/analyzers/technical_analyzer.py

import logging
import asyncio
import json
from typing import Dict, Any, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

from src.assistant.state import ResearchState
from src.assistant.configuration import Configuration
from src.assistant.utils.helpers import traceable, clean_llm_output

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
async def validate_technical_aspects(state: ResearchState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Validate technical aspects of the research using LLM analysis.
    
    Focuses on technical correctness, implementation feasibility, and 
    potential technical conflicts or gaps in the research findings.
    """
    logger.info("--- Node: validate_technical_aspects ---")
    
    # Get configuration
    cfg = Configuration.from_runnable_config(config) if config else Configuration()
    summary = state.get("running_summary", "")
    validation_result = {}  # Default empty dict
    
    if not summary:
        logger.warning("No summary available for technical validation.")
        return {"technical_validation": validation_result}
    
    try:
        # Initialize LLM with JSON format for structured output
        llm = ChatOllama(
            base_url=cfg.ollama_base_url,
            model=cfg.local_llm,
            format="json",  # Request JSON output
            **LLM_CONFIG_PARAMS
        )
        
        # Get technical validation prompt from imported template
        from src.utils.prompts import technical_validation_instructions
        prompt = technical_validation_instructions
        
        # Prepare human message with content to validate
        human_msg = f"Validate technical aspects of:\n{summary[:MAX_SUMMARY_CONTEXT]}"
        
        # Invoke LLM
        result = await asyncio.wait_for(
            llm.ainvoke([SystemMessage(content=prompt), SystemMessage(content=human_msg)]),
            timeout=300  # Adjust timeout for complex analysis
        )
        
        try:
            # Parse and validate LLM output
            validation_result = json.loads(result.content)
            
            if not isinstance(validation_result, dict):
                logger.error("Technical validation did not return a JSON object.")
                validation_result = {"error": "Invalid format received from LLM"}
                
            # Ensure expected fields exist
            expected_fields = [
                "overall_assessment", 
                "technical_correctness", 
                "implementation_feasibility",
                "potential_issues",
                "recommendations"
            ]
            
            # Add any missing expected fields with default values
            for field in expected_fields:
                if field not in validation_result:
                    if field in ["potential_issues", "recommendations"]:
                        validation_result[field] = []
                    else:
                        validation_result[field] = "Not assessed"
            
        except json.JSONDecodeError as decode_err:
            logger.error(f"Failed to parse technical validation JSON: {decode_err}")
            validation_result = {"error": f"JSON parsing failed: {decode_err}"}
    
    except Exception as e:
        logger.error(f"Error in technical validation: {e}", exc_info=True)
        validation_result = {"error": f"Validation failed: {e}"}
    
    logger.info("Technical validation complete.")
    return {"technical_validation": validation_result}