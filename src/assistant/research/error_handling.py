"""
Error handling and recovery functionality for the research package.
"""

import logging
from typing import Dict, Any, List
from .types import ResearchState
from datetime import datetime

logger = logging.getLogger(__name__)

def error_recovery(state: ResearchState, config: Dict[str, Any]) -> ResearchState:
    """Handle errors and determine recovery actions."""
    try:
        errors = state.get("_errors", [])
        if not errors:
            return state
            
        # Group errors by stage
        errors_by_stage = {}
        for error in errors:
            stage = error.get("stage", "unknown")
            if stage not in errors_by_stage:
                errors_by_stage[stage] = []
            errors_by_stage[stage].append(error)
            
        # Determine recovery actions
        recovery_actions = []
        for stage, stage_errors in errors_by_stage.items():
            if stage == "search":
                recovery_actions.append("retry_enterprise")
            elif stage == "summarization":
                recovery_actions.append("continue")
            elif stage == "security_analysis":
                recovery_actions.append("continue")
            elif stage == "technical_validation":
                recovery_actions.append("continue")
            elif stage == "verification":
                recovery_actions.append("perform_web_search")
            else:
                recovery_actions.append("continue")
                
        state["_recovery_actions"] = recovery_actions
        state["_recovery_metadata"] = {
            "errors_by_stage": errors_by_stage,
            "timestamp": datetime.now().isoformat()
        }
        
        return state
    except Exception as e:
        logger.error(f"Error in error recovery: {e}")
        return state

def try_route_enterprise(state: ResearchState) -> str:
    """Determine routing for enterprise research."""
    if state.get("_errors"):
        return "error_recovery"
    return "continue"

def try_get_recovery_action(state: ResearchState) -> str:
    """Get the next recovery action."""
    actions = state.get("_recovery_actions", [])
    if not actions:
        return "continue"
    return actions[0]

def try_check_summarization(state: ResearchState) -> str:
    """Check if summarization is needed."""
    if not state.get("running_summary"):
        return "needs_summarization"
    return "continue"

def try_check_technical_needs(state: ResearchState) -> str:
    """Check if technical validation is needed."""
    if not state.get("technical_validation"):
        return "needs_technical_validation"
    return "continue"

def try_determine_continuation(state: ResearchState, config: Dict[str, Any] | None = None) -> str:
    """Determine whether to continue research or finalize."""
    if state.get("_errors"):
        return "has_errors"
        
    if state.get("research_loop_count", 0) >= 5:
        return "complete"
        
    summary = state.get("running_summary", "")
    if len(summary) < 200 and state.get("research_loop_count", 0) < 2:
        return "continue"
        
    topic = state.get("research_topic", "")
    deep_topics = ["technical", "detailed", "comprehensive", "analysis", "security"]
    needs_depth = any(word in topic.lower() for word in deep_topics)
    
    if needs_depth and state.get("research_loop_count", 0) < 3:
        return "continue"
        
    return "complete" 