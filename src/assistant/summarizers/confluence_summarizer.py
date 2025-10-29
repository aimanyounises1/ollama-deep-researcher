# src/assistant/summarizers/confluence_summarizer.py

import logging
import asyncio
import json
from typing import Dict, Any, List, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

from src.assistant.research.utils import clean_llm_output
from src.assistant.state import ResearchState
from src.assistant.configuration import Configuration
from src.assistant.utils.helpers import traceable

logger = logging.getLogger(__name__)

# LLM Configuration params
LLM_CONFIG_PARAMS = {
    "temperature": 0.15,
    "num_ctx": 8192,
    "top_p": 0.7,
    "repeat_penalty": 1.15,
}

# Constants
MAX_CHARS_PER_SUMMARIZE_CALL = 10000

@traceable
async def summarize_confluence_results(state: ResearchState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Summarize the pre-processed Confluence bullet points into a coherent narrative.
    
    Uses the modular extraction and summarization approach to handle large
    amounts of content by processing in batches if needed.
    """
    logger.info("--- Node: Summarizing Confluence ---")
    
    # Get configuration
    cfg = Configuration.from_runnable_config(config) if config else Configuration()
    topic = state.get("research_topic", "Unknown Topic")
    
    # Get pre-processed bullet points
    pre_processed_bullets = state.get("pre_processed_summaries", {}).get("### Confluence", [])
    
    # Default output if no data
    summary_output = "_No relevant Confluence information was found or processed._"
    
    if not pre_processed_bullets:
        logger.info("No pre-processed Confluence bullets found. Skipping summarization.")
        return {"confluence_summary": summary_output}
    
    # Calculate total length of bullet points
    total_bullet_chars = sum(len(b) + 1 for b in pre_processed_bullets)
    
    try:
        # Ensure LLM configuration is available
        if not hasattr(cfg, 'ollama_base_url') or not hasattr(cfg, 'local_llm'):
            raise ValueError("LLM configuration missing")
            
        # Initialize LLM
        llm = ChatOllama(
            base_url=cfg.ollama_base_url,
            model=cfg.local_llm,
            timeout=180,
            **LLM_CONFIG_PARAMS
        )
        
        # Single LLM call if content is small enough
        if total_bullet_chars <= MAX_CHARS_PER_SUMMARIZE_CALL:
            logger.info(f"Confluence bullet points fit within limit ({total_bullet_chars}/{MAX_CHARS_PER_SUMMARIZE_CALL}). Using single LLM call.")
            
            # Join bullets into a single string
            bullet_string = "\n".join(pre_processed_bullets)
            
            # Create prompt for summarization
            prompt = _get_confluence_summarization_prompt(topic, bullet_string)
            
            # Invoke LLM
            messages = [SystemMessage(content=prompt)]
            out = await llm.ainvoke(messages)
            cleaned_summary = clean_llm_output(out.content)
            
            # Validate output
            if len(cleaned_summary) < 30 or "based on the following" in cleaned_summary.lower() or "no specific findings" in cleaned_summary.lower():
                logger.warning("LLM returned a short or generic summary for Confluence, using default.")
            else:
                summary_output = cleaned_summary
                logger.info("Confluence summarization complete (single call).")
                
        else:
            # Iterative approach for large content
            logger.info(f"Confluence bullet points exceed limit ({total_bullet_chars}/{MAX_CHARS_PER_SUMMARIZE_CALL}). Using iterative LLM calls.")
            
            # Split into batches
            batches = []
            current_batch_chars = 0
            current_batch = []
            
            for bullet in pre_processed_bullets:
                bullet_len = len(bullet) + 1
                if current_batch_chars + bullet_len > MAX_CHARS_PER_SUMMARIZE_CALL and current_batch:
                    batches.append("\n".join(current_batch))
                    current_batch = [bullet]
                    current_batch_chars = bullet_len
                else:
                    current_batch.append(bullet)
                    current_batch_chars += bullet_len
                    
            if current_batch:
                batches.append("\n".join(current_batch))
                
            logger.info(f"Split Confluence bullets into {len(batches)} batches for LLM.")
            
            # Process batches iteratively
            running_summary = ""
            for i, batch_str in enumerate(batches):
                logger.info(f"Processing Confluence batch {i + 1}/{len(batches)}...")
                
                # Create appropriate prompt based on batch position
                if i == 0:
                    # First batch - standard prompt
                    prompt = _get_confluence_summarization_prompt(
                        topic, 
                        batch_str, 
                        is_iterative=True, 
                        batch_num=i+1, 
                        total_batches=len(batches)
                    )
                else:
                    # Subsequent batch - iterative prompt with previous summary
                    prompt = _get_confluence_iterative_prompt(
                        topic,
                        batch_str,
                        previous_summary=running_summary,
                        batch_num=i+1,
                        total_batches=len(batches)
                    )
                
                # Invoke LLM
                messages = [SystemMessage(content=prompt)]
                out = await llm.ainvoke(messages)
                
                # Update running summary
            if running_summary.strip() and "_Error summarizing" not in running_summary:
                summary_output = running_summary
                logger.info(f"Confluence iterative summarization complete ({len(batches)} calls).")
            else:
                logger.error("Iterative summarization for Confluence failed or produced empty result.")
                summary_output = "_Error during iterative summarization for Confluence._"
                
    except Exception as e:
        logger.error(f"Confluence summarization failed: {e}", exc_info=True)
        summary_output = f"_Error summarizing Confluence information: {e}_"
    
    return {"confluence_summary": summary_output.strip()}

def _get_confluence_summarization_prompt(topic: str, bullet_string: str, is_iterative: bool = False, batch_num: int = 1, total_batches: int = 1) -> str:
    """Generate the appropriate prompt for Confluence summarization."""
    
    base_instructions = (
        f"Objective: Generate a DETAILED 'Confluence' analysis section for a research report on '{topic}'."
        f"\nInput: Multiple bullet points below. Each starting `- **[Page Title](URL)** (Content Snippet N/M): ...` represents a CHUNK of content from the linked Confluence page. Some snippets contain prose, others contain table-like data describing system interactions."
        "\nInstructions:"
        "\n1. **Synthesize Deeply:** DO NOT just list pages/excerpts. Write a comprehensive narrative integrating information from ALL relevant snippets."
        "\n2. **Extract ALL Specifics:** Meticulously extract and incorporate specific details. **List or reference ALL mentioned API/Interface IDs, component names, API types, statuses, and configuration requirements.**"
        "\n3. **Structure Logically:** Organize findings by sub-topic or by summarizing each major document/page, ensuring smooth transitions."
        "\n4. **Incorporate Metadata:** Include author/date if available. Mention identifiers and explain the relevance of statuses."
        "\n5. **Use Links:** Ensure page titles are Markdown links `[Title](URL)`."
        "\n6. **Length & Detail:** Aim for a thorough, multi-paragraph section accurately reflecting the technical depth of the provided snippets."
        f"\n7. **Strict Relevance:** Base output *only* on provided snippets clearly related to '{topic}'. Do not infer information not present."
    )
    
    # Standard prompt for single batch
    if not is_iterative:
        return base_instructions + f"\n\nInput Bullet Points/Snippets from Confluence:\n{bullet_string}\n\nGenerate the comprehensive, technically detailed 'Confluence' analysis section below:"
    
    # First batch of iterative process
    if is_iterative and batch_num == 1:
        iterative_instruction = f"\n\n**IMPORTANT:** This is the first batch (1/{total_batches}) of bullet points for Confluence. Generate a summary based on this batch, but be prepared to integrate information from subsequent batches."
        return base_instructions + f"\n\nInput Bullet Points/Snippets from Confluence:\n{bullet_string}\n\n{iterative_instruction}\n\nGenerate the initial 'Confluence' summary section below:"
    
    # Should not reach here in this function, but provide fallback
    return base_instructions + f"\n\nInput Bullet Points/Snippets from Confluence:\n{bullet_string}\n\nGenerate the comprehensive 'Confluence' summary section below:"

def _get_confluence_iterative_prompt(topic: str, bullet_string: str, previous_summary: str, batch_num: int, total_batches: int) -> str:
    """Generate the appropriate prompt for iterative refinement of Confluence summary."""
    
    base_objective = f"Objective: Refine and extend the existing Confluence documentation analysis (related to '{topic}') using new information."
    
    iterative_instruction = (
        f"\n\n**IMPORTANT:** This is batch {batch_num}/{total_batches} for Confluence. You MUST integrate the information from the 'New Bullet Points' below with the 'Previous Summary'. "
        f"Update and refine the summary to incorporate the new details cohesively. Output ONLY the *complete, updated summary* incorporating all information processed so far."
        f"\n\n**Previous Summary (from batches 1 to {batch_num - 1}):**\\n{previous_summary}\\n\\n"
        f"**New Bullet Points (batch {batch_num}):**\\n{bullet_string}\\n\\n"
        f"Generate the *complete, updated* Confluence summary section below:"
    )
    
    return base_objective + iterative_instruction