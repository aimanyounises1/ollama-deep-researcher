"""
Analysis functions for the research package.
"""

import json
import logging
import re
from datetime import datetime
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langsmith import traceable
from src.assistant.configuration import Configuration
from src.assistant.core.types import ResearchState
from src.assistant.pipeline.nodes import should_process_next_confluence_chunk
from src.assistant.utils.langgraph_helpers import clean_llm_output
from src.assistant.pipeline.tools import tool_manager

logger = logging.getLogger(__name__)

@traceable
async def analyze_perforce_diffs(state: ResearchState, config: RunnableConfig) -> ResearchState:
    """Analyze Perforce diffs for security and technical insights."""
    if should_process_next_confluence_chunk(state):
        logger.info("Skipping Perforce analysis during Confluence chunk processing")
        return state
        
    try:
        results = state.get("enterprise_research_results", [])
        perforce_results = [r for r in results if "=== Perforce Results ===" in r]
                
        if not perforce_results:
            return state
            
        analysis = {
            "security_issues": [],
            "technical_changes": [],
            "affected_files": set()
        }
        
        for result in perforce_results:
            file_matches = re.findall(r'File: ([^\n]+)', result)
            for file_path in file_matches:
                analysis["affected_files"].add(file_path)
                
            security_patterns = [
                r'password', r'secret', r'key', r'token', r'auth',
                r'credential', r'permission', r'access', r'security'
            ]
            for pattern in security_patterns:
                if re.search(pattern, result, re.IGNORECASE):
                    analysis["security_issues"].append({
                        "pattern": pattern,
                        "context": result[:200] + "..."
                    })
                    
            if re.search(r'class|function|method|api|interface', result, re.IGNORECASE):
                analysis["technical_changes"].append({
                    "type": "code_change",
                    "context": result[:200] + "..."
                })
                
        analysis["affected_files"] = list(analysis["affected_files"])
        state["perforce_diff_analysis"] = analysis
        return state
    except Exception as e:
        logger.error(f"Error analyzing Perforce diffs: {e}")
        state.setdefault("_errors", []).append({
            "stage": "perforce_analysis",
            "message": str(e)
        })
        return state

@traceable
async def analyze_security(state: ResearchState, config: RunnableConfig) -> ResearchState:
    """Analyze security implications of the findings."""
    if should_process_next_confluence_chunk(state):
        logger.info("Skipping security analysis during Confluence chunk processing")
        return state
        
    try:
        summary = state.get("running_summary", "")
        perforce_analysis = state.get("perforce_diff_analysis", {})
        
        if not summary and not perforce_analysis:
            return state
            
        cfg = Configuration.from_runnable_config(config)
        llm = ChatOllama(
            base_url=cfg.ollama_base_url,
            model=cfg.local_llm,
            temperature=0
        )
        
        prompt = f"""Analyze the following information for security implications:

Summary:
{summary}

Perforce Analysis:
{json.dumps(perforce_analysis, indent=2)}

Security Analysis:"""
        
        response = await llm.ainvoke(prompt)
        state["security_findings"] = [{
            "analysis": clean_llm_output(response.content),
            "timestamp": datetime.now().isoformat()
        }]
        
        return state
    except Exception as e:
        logger.error(f"Error analyzing security: {e}")
        state.setdefault("_errors", []).append({
            "stage": "security_analysis",
            "message": str(e)
        })
        return state

@traceable
async def validate_technical_aspects(state: ResearchState, config: RunnableConfig) -> ResearchState:
    """Validate technical aspects of the findings."""
    if should_process_next_confluence_chunk(state):
        logger.info("Skipping technical validation during Confluence chunk processing")
        return state
        
    try:
        summary = state.get("running_summary", "")
        perforce_analysis = state.get("perforce_diff_analysis", {})
        
        if not summary and not perforce_analysis:
            return state
            
        cfg = Configuration.from_runnable_config(config)
        llm = ChatOllama(
            base_url=cfg.ollama_base_url,
            model=cfg.local_llm,
            temperature=0
        )
        
        prompt = f"""Validate the technical aspects of the following information:

Summary:
{summary}

Perforce Analysis:
{json.dumps(perforce_analysis, indent=2)}

Technical Validation:"""
        
        response = await llm.ainvoke(prompt)
        state["technical_validation"] = {
            "validation": clean_llm_output(response.content),
            "timestamp": datetime.now().isoformat()
        }
        
        return state
    except Exception as e:
        logger.error(f"Error validating technical aspects: {e}")
        state.setdefault("_errors", []).append({
            "stage": "technical_validation",
            "message": str(e)
        })
        return state

@traceable
async def verify_summary(state: ResearchState, config: RunnableConfig | None = None) -> ResearchState:
    """Verify the quality and completeness of the summary."""
    if should_process_next_confluence_chunk(state):
        logger.info("Skipping summary verification during Confluence chunk processing")
        return state
        
    try:
        summary = state.get("running_summary", "")
        if not summary:
            state.setdefault("_errors", []).append({
                "stage": "verification",
                "message": "No summary to verify"
            })
            return state
            
        if len(summary) < 100:
            state.setdefault("_errors", []).append({
                "stage": "verification",
                "message": "Summary too short"
            })
            
        key_components = ["security", "technical", "implementation"]
        missing_components = []
        for component in key_components:
            if component not in summary.lower():
                missing_components.append(component)
                
        if missing_components:
            state.setdefault("_errors", []).append({
                "stage": "verification",
                "message": f"Missing key components: {', '.join(missing_components)}"
            })
            
        return state
    except Exception as e:
        logger.error(f"Error verifying summary: {e}")
        state.setdefault("_errors", []).append({
            "stage": "verification",
            "message": str(e)
        })
        return state

@traceable
async def reflect_on_findings(state: ResearchState, config: RunnableConfig) -> ResearchState:
    """Reflect on the research findings and determine next steps."""
    try:
        summary = state.get("running_summary", "")
        security_findings = state.get("security_findings", [])
        technical_validation = state.get("technical_validation", {})
        errors = state.get("_errors", [])
        
        cfg = Configuration.from_runnable_config(config)
        llm = ChatOllama(
            base_url=cfg.ollama_base_url,
            model=cfg.local_llm,
            temperature=0
        )
        
        prompt = f"""Reflect on the following research findings:

Summary:
{summary}

Security Findings:
{json.dumps(security_findings, indent=2)}

Technical Validation:
{json.dumps(technical_validation, indent=2)}

Errors:
{json.dumps(errors, indent=2)}

Reflection:"""
        
        response = await llm.ainvoke(prompt)
        state["reflection_data"] = clean_llm_output(response.content)
        
        return state
    except Exception as e:
        logger.error(f"Error reflecting on findings: {e}")
        state.setdefault("_errors", []).append({
            "stage": "reflection",
            "message": str(e)
        })
        return state

@traceable
async def finalize_summary(state: ResearchState) -> ResearchState:
    """Finalize the research summary with all findings."""
    try:
        summary = state.get("running_summary", "")
        security_findings = state.get("security_findings", [])
        technical_validation = state.get("technical_validation", {})
        reflection = state.get("reflection_data", "")
        
        final_summary = f"""Research Summary:
{summary}

Security Findings:
{json.dumps(security_findings, indent=2)}

Technical Validation:
{json.dumps(technical_validation, indent=2)}

Reflection:
{reflection}"""
        
        state["running_summary"] = final_summary
        return state
    except Exception as e:
        logger.error(f"Error finalizing summary: {e}")
        state.setdefault("_errors", []).append({
            "stage": "finalization",
            "message": str(e)
        })
        return state

@traceable
async def analyze_security_tool(state: ResearchState, config: RunnableConfig) -> ResearchState:
    """Analyze security aspects of the research content using a tool."""
    try:
        # Get content to analyze
        content = state.get("running_summary", "")
        if not content:
            logger.info("No content to analyze for security")
            state["security_findings"] = []
            return state
        
        # Use security analyzer tool
        security_findings = await tool_manager.analyze_security(content)
        
        # If tool returns empty results, use LLM fallback
        if not security_findings:
            logger.info("Security tool returned no findings, using LLM fallback")
            
            # Use LLM to analyze security
            cfg = config.get("configurable", {})
            ollama_base_url = cfg.get("ollama_base_url", "http://localhost:11434")
            local_llm = cfg.get("local_llm", "deepseek:latest")
            
            llm = ChatOllama(
                base_url=ollama_base_url,
                model=local_llm,
                temperature=0,
                format="json"
            )
            
            prompt = f"""Analyze the following content for security implications and return a JSON array of findings:
{content[:4000]}  # Limit content to avoid token limits

Each finding should include:
1. "type": Type of security finding
2. "severity": high/medium/low
3. "description": Description of the issue
4. "recommendation": Recommended action
"""
            
            try:
                response = await llm.ainvoke([
                    SystemMessage(content="You are a security analyst. Identify security implications in the content."),
                    HumanMessage(content=prompt)
                ])
                
                try:
                    security_findings = json.loads(response.content)
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON from LLM response")
                    security_findings = []
            except Exception as e:
                logger.error(f"Error calling LLM for security analysis: {e}")
                security_findings = []
                
        # Store findings in state
        state["security_findings"] = security_findings
        logger.info(f"Found {len(security_findings)} security findings")
        
        return state
    except Exception as e:
        logger.error(f"Error in analyze_security_tool: {e}")
        state.setdefault("_errors", []).append({
            "stage": "security_analysis",
            "message": str(e)
        })
        state["security_findings"] = []
        return state

@traceable
async def validate_technical_aspects_tool(state: ResearchState, config: RunnableConfig) -> ResearchState:
    """Validate technical aspects of the research using a tool."""
    try:
        # Get content to validate
        content = state.get("running_summary", "")
        if not content:
            logger.info("No content to validate for technical aspects")
            state["technical_validation"] = {"valid": True, "findings": []}
            return state
        
        # Use LLM to validate technical aspects
        cfg = config.get("configurable", {})
        ollama_base_url = cfg.get("ollama_base_url", "http://localhost:11434")
        local_llm = cfg.get("local_llm", "deepseek:latest")
        
        llm = ChatOllama(
            base_url=ollama_base_url,
            model=local_llm,
            temperature=0,
            format="json"
        )
        
        # Get the research topic
        topic = state.get("research_topic", "")
        
        prompt = f"""Validate the technical aspects of the following research summary for the topic "{topic}":
{content[:4000]}  # Limit content to avoid token limits

Return a JSON object with:
1. "valid": boolean indicating if the content is technically valid
2. "findings": array of technical findings/issues
3. "improvement_suggestions": array of suggestions to improve technical accuracy
"""
        
        try:
            response = await llm.ainvoke([
                SystemMessage(content="You are a technical validator. Assess the technical accuracy of the content."),
                HumanMessage(content=prompt)
            ])
            
            try:
                validation_result = json.loads(response.content)
                state["technical_validation"] = validation_result
                logger.info(f"Technical validation complete: valid={validation_result.get('valid', False)}")
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from LLM response")
                state["technical_validation"] = {"valid": True, "findings": [], "improvement_suggestions": []}
        except Exception as e:
            logger.error(f"Error calling LLM for technical validation: {e}")
            state["technical_validation"] = {"valid": True, "findings": [], "improvement_suggestions": []}
        
        return state
    except Exception as e:
        logger.error(f"Error in validate_technical_aspects_tool: {e}")
        state.setdefault("_errors", []).append({
            "stage": "technical_validation",
            "message": str(e)
        })
        state["technical_validation"] = {"valid": True, "findings": [], "improvement_suggestions": []}
        return state

@traceable
async def verify_summary_tool(state: ResearchState, config: RunnableConfig) -> ResearchState:
    """Verify the quality of the summary using a tool."""
    try:
        # Get summary to verify
        summary = state.get("running_summary", "")
        if not summary:
            logger.info("No summary to verify")
            return state
        
        # Check for citations
        citation_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        citations = re.findall(citation_pattern, summary)
        
        if not citations:
            logger.warning("No citations found in summary")
            state.setdefault("_errors", []).append({
                "stage": "verification",
                "message": "No citations found in summary",
                "severity": "warning"
            })
        
        # Check for technical terms
        topic = state.get("research_topic", "")
        topic_terms = topic.lower().split()
        
        missing_terms = []
        for term in topic_terms:
            if len(term) > 3 and term not in summary.lower():
                missing_terms.append(term)
        
        if missing_terms:
            logger.warning(f"Summary missing key terms: {missing_terms}")
            state.setdefault("_errors", []).append({
                "stage": "verification",
                "message": f"Summary missing key terms: {missing_terms}",
                "severity": "warning"
            })
        
        return state
    except Exception as e:
        logger.error(f"Error in verify_summary_tool: {e}")
        state.setdefault("_errors", []).append({
            "stage": "verification",
            "message": str(e),
            "severity": "error"
        })
        return state

@traceable
async def reflect_on_findings_tool(state: ResearchState, config: RunnableConfig) -> ResearchState:
    """Reflect on research findings to identify gaps and next steps using a tool."""
    try:
        # Get research content
        summary = state.get("running_summary", "")
        if not summary:
            logger.info("No summary to reflect on")
            state["reflection_data"] = {
                "key_findings": [],
                "identified_gaps": [],
                "next_steps": []
            }
            return state
        
        # Use LLM for reflection
        cfg = config.get("configurable", {})
        ollama_base_url = cfg.get("ollama_base_url", "http://localhost:11434")
        local_llm = cfg.get("local_llm", "deepseek:latest")
        
        llm = ChatOllama(
            base_url=ollama_base_url,
            model=local_llm,
            temperature=0,
            format="json"
        )
        
        # Get the research topic
        topic = state.get("research_topic", "")
        
        prompt = f"""Reflect on the following research summary for the topic "{topic}":
{summary[:4000]}  # Limit content to avoid token limits

Return a JSON object with:
1. "key_findings": array of key findings
2. "identified_gaps": array of gaps in the research
3. "next_steps": array of suggested next steps
4. "alternative_search_terms": array of alternative search terms for further research
5. "priority": "HIGH", "MEDIUM", or "LOW" for continuing research
"""
        
        try:
            response = await llm.ainvoke([
                SystemMessage(content="You are a research analyst. Identify key findings, gaps, and next steps."),
                HumanMessage(content=prompt)
            ])
            
            try:
                reflection_data = json.loads(response.content)
                state["reflection_data"] = reflection_data
                
                # Extract alternative terms for next search iteration
                if "alternative_search_terms" in reflection_data:
                    state["alternative_terms"] = reflection_data["alternative_search_terms"]
                
                logger.info(f"Reflection complete: priority={reflection_data.get('priority', 'UNKNOWN')}")
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from LLM response")
                state["reflection_data"] = {
                    "key_findings": [],
                    "identified_gaps": [],
                    "next_steps": [],
                    "priority": "LOW"
                }
        except Exception as e:
            logger.error(f"Error calling LLM for reflection: {e}")
            state["reflection_data"] = {
                "key_findings": [],
                "identified_gaps": [],
                "next_steps": [],
                "priority": "LOW"
            }
        
        return state
    except Exception as e:
        logger.error(f"Error in reflect_on_findings_tool: {e}")
        state.setdefault("_errors", []).append({
            "stage": "reflection",
            "message": str(e)
        })
        state["reflection_data"] = {
            "key_findings": [],
            "identified_gaps": [],
            "next_steps": [],
            "priority": "LOW"
        }
        return state

@traceable
async def finalize_summary_tool(state: ResearchState, config: RunnableConfig = None) -> ResearchState:
    """Finalize the research summary with improved formatting using a tool."""
    try:
        # Get existing summary
        summary = state.get("running_summary", "")
        topic = state.get("research_topic", "")
        
        if not summary:
            logger.info("No summary to finalize")
            state["running_summary"] = f"# Research on {topic}\n\nNo research findings available."
            return state
        
        # Add sections that might be missing
        sections = ["## Overview", "## Key Findings", "## Technical Details", "## Security Implications", "## Recommendations"]
        existing_sections = set()
        
        for line in summary.split("\n"):
            if line.startswith("## "):
                existing_sections.add(line.strip())
        
        missing_sections = []
        for section in sections:
            if section not in existing_sections:
                missing_sections.append(section)
        
        if missing_sections:
            # Append missing sections
            for section in missing_sections:
                summary += f"\n\n{section}\nNo information available."
        
        # Clean up the format
        summary = re.sub(r'\n{3,}', '\n\n', summary)  # Remove excessive newlines
        
        # Add citations section if there are citations
        citations = state.get("citations", [])
        if citations and "## Sources" not in existing_sections:
            citation_section = "\n\n## Sources\n"
            for citation in citations:
                source_type = citation.get("source_type", "unknown")
                content = citation.get("content", "No content")
                url = citation.get("url", "#")
                citation_section += f"- [{source_type}]({url}): {content[:100]}...\n"
            
            summary += citation_section
        
        # Add timestamp
        timestamp = datetime.now().isoformat(timespec='minutes')
        summary += f"\n\n*Generated on: {timestamp}*"
        
        # Store the finalized summary
        state["running_summary"] = summary
        logger.info("Summary finalized successfully")
        
        return state
    except Exception as e:
        logger.error(f"Error in finalize_summary_tool: {e}")
        state.setdefault("_errors", []).append({
            "stage": "finalize_summary",
            "message": str(e)
        })
        return state 
