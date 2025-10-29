"""
Memory-Augmented Verification component that maintains factual consistency
across multiple interactions using a simple memory store.
"""

import asyncio
import logging
import json
import time
import networkx as nx
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Set, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Pydantic models for structured outputs
class ExtractedFacts(BaseModel):
    """List of extracted facts."""
    facts: List[str] = Field(description="List of extracted factual claims")

class ExtractedEntities(BaseModel):
    """List of extracted entities."""
    entities: List[str] = Field(description="List of extracted entities")

class ConsistencyCheck(BaseModel):
    """Result of consistency check between facts."""
    is_consistent: bool = Field(description="Whether the content is consistent")
    conflicts: List[Dict[str, str]] = Field(description="Conflicts between facts")
    confidence: float = Field(description="Confidence level in the assessment")
    reasoning: str = Field(description="Reasoning behind the assessment")

class CorrectionResult(BaseModel):
    """Suggested corrections for inconsistent content."""
    corrections: List[str] = Field(description="List of suggested corrections")
    explanation: str = Field(description="Explanation for the corrections")

class FactEntry(BaseModel):
    """Model for a fact stored in the fact registry."""
    fact_id: str = Field(description="Unique identifier for this fact")
    content: str = Field(description="The factual statement")
    source: str = Field(description="Where this fact came from")
    timestamp: float = Field(description="When this fact was added")
    confidence: float = Field(description="Confidence level (0-1)")
    context: Optional[str] = Field(description="Additional context about the fact")
    related_facts: List[str] = Field(description="IDs of related facts")
    epistemological_status: str = Field(description="Epistemological status of the fact", default="unverified")
    verification_history: List[Dict[str, Any]] = Field(description="History of verification attempts", default_factory=list)
    entities: List[str] = Field(description="Entities mentioned in the fact", default_factory=list)
    lifespan: Optional[Dict[str, Any]] = Field(description="Time validity range of the fact", default=None)

class TemporalConsistencyResult(BaseModel):
    """Result of temporal consistency check."""
    is_consistent: bool = Field(description="Whether the statement is temporally consistent")
    conflicts: List[Dict[str, Any]] = Field(description="Any conflicting facts")
    confidence: float = Field(description="Confidence in consistency assessment")
    reasoning: str = Field(description="Reasoning for the consistency assessment")

class MemoryAugmentedVerifier:
    """
    Verifies factual consistency across conversations using memory management.
    
    This verifier maintains a fact registry that persists across verification sessions
    allowing it to detect and prevent temporal inconsistencies
    and contradictions in LLM responses over time.
    """
    
    def __init__(
        self, 
        llm: BaseChatModel,
        fact_registry_size: int = 1000,
        enable_temporal_graph: bool = True,
        enable_epistemic_tagging: bool = True,
        enable_entity_extraction: bool = True,
        enable_user_feedback: bool = False
    ):
        """
        Initialize the memory-augmented verifier.
        
        Args:
            llm: Language model to use for verification
            fact_registry_size: Maximum number of facts to maintain in registry
            enable_temporal_graph: Whether to use graph structure for fact relationships
            enable_epistemic_tagging: Whether to tag facts with epistemological status
            enable_entity_extraction: Whether to extract entities from facts
            enable_user_feedback: Whether to allow incorporating user feedback
        """
        self.llm = llm
        self.fact_registry_size = fact_registry_size
        self.enable_temporal_graph = enable_temporal_graph
        self.enable_epistemic_tagging = enable_epistemic_tagging
        self.enable_entity_extraction = enable_entity_extraction
        self.enable_user_feedback = enable_user_feedback
        self.checkpointer = None  # Set this separately via the config
        
        # Use simple dictionaries for storage or checkpointer if provided
        self._fact_registry = {}  # thread_id -> list of facts
        self._temporal_graphs = {} if enable_temporal_graph else None
    
    async def _load_thread_facts(self, thread_id: str) -> List[Dict[str, Any]]:
        """Load facts from checkpointer if available."""
        if self.checkpointer is None:
            return self._fact_registry.get(thread_id, [])
            
        try:
            # Try to load from checkpointer
            facts_key = f"facts_{thread_id}"
            facts = await self.checkpointer.aget(facts_key)
            if facts:
                return facts
        except Exception as e:
            logger.warning(f"Failed to load facts from checkpointer: {e}")
            
        return self._fact_registry.get(thread_id, [])
    
    async def _save_thread_facts(self, thread_id: str, facts: List[Dict[str, Any]]) -> None:
        """Save facts to checkpointer if available."""
        # Update in-memory registry
        self._fact_registry[thread_id] = facts
        
        # Save to checkpointer if available
        if self.checkpointer is not None:
            try:
                facts_key = f"facts_{thread_id}"
                await self.checkpointer.aput(facts_key, facts)
            except Exception as e:
                logger.warning(f"Failed to save facts to checkpointer: {e}")
    
    async def verify_with_memory(
        self, 
        statement: str,
        thread_id: str,
        context: Optional[str] = None,
        source: str = "model_output",
        user_feedback: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Verify a statement against previously stored facts for temporal consistency.
        
        Args:
            statement: The statement to verify
            thread_id: Thread ID for conversation context
            context: Additional context for verification
            source: Source of the statement
            user_feedback: Optional feedback from the user about fact accuracy
            
        Returns:
            Verification results
        """
        # Extract key facts from the statement
        facts = await self._extract_facts(statement)
        
        # Get existing facts for this thread (using checkpointer if available)
        thread_facts_data = await self._load_thread_facts(thread_id)
        thread_facts = [FactEntry(**fact) for fact in thread_facts_data]
        
        # Check temporal consistency for each fact
        consistency_results = []
        new_facts = []
        
        for fact in facts:
            # Extract entities if enabled
            entities = []
            if self.enable_entity_extraction:
                entities = await self._extract_entities(fact)
            
            # Assign epistemological status if enabled
            epistemological_status = "unverified"
            if self.enable_epistemic_tagging:
                epistemological_status = await self._assign_epistemic_status(fact, source)
            
            # Check temporal consistency
            result = await self._check_temporal_consistency(fact, thread_facts, context)
            consistency_results.append({
                "fact": fact,
                "result": result.model_dump(),
                "entities": entities
            })
            
            # If consistent (or no conflicts), add to the fact registry
            if result.is_consistent:
                fact_id = self._generate_fact_id(fact)
                new_fact = FactEntry(
                    fact_id=fact_id,
                    content=fact,
                    source=source,
                    timestamp=time.time(),
                    confidence=result.confidence,
                    context=context,
                    related_facts=[],
                    epistemological_status=epistemological_status,
                    entities=entities,
                    lifespan=self._estimate_fact_lifespan(fact)
                )
                new_facts.append(new_fact)
        
        # Update fact registry with new facts
        thread_facts_data = [fact.model_dump() for fact in thread_facts]
        new_facts_data = [fact.model_dump() for fact in new_facts]
        updated_facts = thread_facts_data + new_facts_data
        
        # Limit size if needed
        if len(updated_facts) > self.fact_registry_size:
            updated_facts = sorted(updated_facts, key=lambda x: x.get('timestamp', 0), reverse=True)[:self.fact_registry_size]
        
        # Save using checkpointer if available
        await self._save_thread_facts(thread_id, updated_facts)
        
        # If temporal graph is enabled, update the graph with new facts
        if self.enable_temporal_graph and new_facts:
            await self._update_temporal_graph(thread_id, new_facts, thread_facts)
        
        # Apply user feedback if provided and enabled
        if self.enable_user_feedback and user_feedback:
            await self._apply_user_feedback(thread_id, user_feedback)
        
        # Determine overall consistency
        is_consistent = all(result["result"]["is_consistent"] for result in consistency_results)
        confidence = sum(result["result"]["confidence"] for result in consistency_results) / len(consistency_results) if consistency_results else 0.0
        
        # Collect conflicts
        all_conflicts = []
        for result in consistency_results:
            if result["result"]["conflicts"]:
                all_conflicts.extend(result["result"]["conflicts"])
        
        # Generate suggested corrections if inconsistent
        corrections = []
        if not is_consistent and all_conflicts:
            corrections = await self._generate_corrections(statement, all_conflicts)
        
        return {
            "is_consistent": is_consistent,
            "confidence": confidence,
            "fact_results": consistency_results,
            "conflicts": all_conflicts,
            "corrections": corrections,
            "new_facts": [fact.model_dump() for fact in new_facts],
            "epistemological_distribution": await self._get_epistemological_distribution(thread_id) if self.enable_epistemic_tagging else {}
        }
    
    async def _extract_facts(self, statement: str) -> List[str]:
        """
        Extract key factual claims from a statement.
        
        Args:
            statement: The statement to extract facts from
            
        Returns:
            List of extracted factual claims
        """
        # Define prompt for fact extraction
        extract_prompt = ChatPromptTemplate.from_template(
            "Extract the key factual claims from this statement:\n\n"
            "{statement}\n\n"
            "A factual claim is a specific assertion that can be verified as true or false. "
            "Focus on extracting claims that are:\n"
            "1. Specific and concrete\n"
            "2. About entities, events, relationships, or properties\n"
            "3. Independent assertions (one claim per fact)\n\n"
            "Format your response as a JSON list of strings, each containing a single factual claim. "
            "Keep the original wording where possible."
        )
        
        extract_chain = extract_prompt | self.llm.with_structured_output(
            ExtractedFacts
        )
        
        try:
            result = await extract_chain.ainvoke({"statement": statement})
            logger.info(f"Extracted {len(result.facts)} facts from statement")
            return result.facts
        except Exception as e:
            logger.error(f"Error extracting facts: {e}")
            # Fallback: treat the entire statement as one fact
            return [statement]
    
    async def _extract_entities(self, fact: str) -> List[str]:
        """Extract entities mentioned in a fact."""
        entity_prompt = ChatPromptTemplate.from_template(
            "Extract the key entities mentioned in this factual claim:\n\n"
            "{fact}\n\n"
            "An entity can be a person, organization, product, technology, place, or concept. "
            "Format your response as a JSON list of strings, each containing a single entity name."
        )
        
        entity_chain = entity_prompt | self.llm.with_structured_output(
            ExtractedEntities
        )
        
        try:
            result = await entity_chain.ainvoke({"fact": fact})
            return result.entities
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    async def _assign_epistemic_status(self, fact: str, source: str) -> str:
        """
        Assign an epistemological status to a fact.
        
        Possible statuses:
        - "verified": Fact has been verified through reliable sources
        - "probable": Fact is likely true but not fully verified
        - "speculative": Fact is a reasonable conjecture but lacks evidence
        - "unverified": Fact hasn't been verified yet
        - "disputed": Fact is contested or contradicted by other sources
        """
        if not self.enable_epistemic_tagging:
            return "unverified"
            
        # Simple heuristic based on source
        if source in ["user_provided", "system_verified", "external_database"]:
            return "verified"
        elif source in ["web_search", "documentation", "technical_reference"]:
            return "probable"
        elif source in ["model_reasoning", "inference"]:
            return "speculative"
        elif source in ["disputed_source", "conflicting_information"]:
            return "disputed"
        else:
            return "unverified"
    
    def _estimate_fact_lifespan(self, fact: str) -> Optional[Dict[str, Any]]:
        """Estimate the time period during which a fact is likely to remain valid."""
        # Default: facts don't expire
        lifespan = {
            "type": "permanent",
            "expiration": None
        }
        
        # Check for temporal indicators in the fact
        temporal_indicators = [
            "currently", "now", "at present", "at the moment", 
            "as of today", "as of now", "presently"
        ]
        
        if any(indicator in fact.lower() for indicator in temporal_indicators):
            # For facts that are explicitly about the current state
            now = time.time()
            # Set expiration to 30 days from now for "current" facts
            expiration = now + (30 * 24 * 60 * 60)
            lifespan = {
                "type": "temporary",
                "start": now,
                "expiration": expiration,
                "confidence": 0.7
            }
        
        return lifespan
    
    async def _check_temporal_consistency(
        self,
        fact: str,
        existing_facts: List[FactEntry],
        context: Optional[str] = None
    ) -> TemporalConsistencyResult:
        """
        Check a fact for temporal consistency with existing facts.
        
        Args:
            fact: The fact to check
            existing_facts: Existing facts to check against
            context: Additional context
            
        Returns:
            Temporal consistency result
        """
        if not existing_facts:
            # No existing facts to check against
            return TemporalConsistencyResult(
                is_consistent=True,
                conflicts=[],
                confidence=1.0,
                reasoning="No existing facts to compare with"
            )
        
        # Find potentially related facts
        related_facts = self._find_related_facts(fact, existing_facts)
        if not related_facts:
            return TemporalConsistencyResult(
                is_consistent=True,
                conflicts=[],
                confidence=0.9,
                reasoning="No closely related facts found"
            )
        
        # Format related facts for the LLM, including epistemological status if available
        related_facts_str = "\n".join([
            f"- {rf.content} (from {rf.source}, confidence: {rf.confidence:.2f}" + 
            (f", status: {rf.epistemological_status}" if self.enable_epistemic_tagging else "") + ")"
            for rf in related_facts
        ])
        
        # Check for consistency with LLM
        consistency_prompt = ChatPromptTemplate.from_template(
            "Determine if the new fact is temporally consistent with existing facts.\n\n"
            "New fact: {fact}\n\n"
            "Existing related facts:\n{related_facts}\n\n"
            "{context_str}\n\n"
            "Check for:\n"
            "1. Direct contradictions\n"
            "2. Temporal inconsistencies (facts that can't both be true at the same time)\n"
            "3. Logical incompatibilities\n"
            "4. Changes in state over time (e.g., someone changing jobs)\n\n"
            "First, reason step-by-step whether the new fact is consistent with the existing facts. "
            "Consider that facts may have changed over time - what was true in the past may not be true now. "
            "Then provide your final assessment and list any specific conflicts."
        )
        
        # For structured output
        consistency_chain = consistency_prompt | self.llm.with_structured_output(
            ConsistencyCheck
        )
        
        try:
            context_str = f"Additional context: {context}" if context else "No additional context provided."
            
            result = await consistency_chain.ainvoke({
                "fact": fact,
                "related_facts": related_facts_str,
                "context_str": context_str
            })
            
            return TemporalConsistencyResult(
                is_consistent=result.is_consistent,
                conflicts=result.conflicts,
                confidence=result.confidence,
                reasoning=result.reasoning
            )
        except Exception as e:
            logger.error(f"Error checking temporal consistency: {e}")
            return TemporalConsistencyResult(
                is_consistent=False,
                conflicts=[{"existing_fact": "Error during consistency check", "conflict_type": "verification_error"}],
                confidence=0.5,
                reasoning=f"Error during consistency check: {str(e)}"
            )
    
    def _find_related_facts(self, fact: str, facts: List[FactEntry]) -> List[FactEntry]:
        """
        Find facts related to the given fact using keyword matching and entity overlap.
        
        Args:
            fact: The fact to find related facts for
            facts: List of facts to search
            
        Returns:
            List of related facts
        """
        # If temporal graph is enabled, use it for relation finding
        if self.enable_temporal_graph:
            # Extract entities from this fact
            fact_entities = self._extract_entities_simple(fact)
            
            # Look for facts with overlapping entities
            entity_related = []
            for existing_fact in facts:
                # Check entity overlap
                if hasattr(existing_fact, 'entities') and existing_fact.entities:
                    overlap = set(fact_entities).intersection(set(existing_fact.entities))
                    if overlap:
                        entity_related.append((existing_fact, len(overlap)))
            
            # Sort by entity overlap and recency
            if entity_related:
                # Sort by overlap first, then by recency
                entity_related.sort(key=lambda x: (x[1], x[0].timestamp), reverse=True)
                return [item[0] for item in entity_related[:5]]
        
        # Fallback: Simple keyword-based approach
        # Extract keywords from the fact
        import re
        keywords = set(re.findall(r'\b\w{4,}\b', fact.lower()))
        
        related = []
        for existing_fact in facts:
            # Check for keyword overlap
            existing_keywords = set(re.findall(r'\b\w{4,}\b', existing_fact.content.lower()))
            overlap = keywords.intersection(existing_keywords)
            
            # If enough keyword overlap, consider it related
            if len(overlap) >= 2 or (len(overlap) >= 1 and len(keywords) <= 3):
                related.append(existing_fact)
        
        # Sort by timestamp (most recent first)
        related.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Limit to most relevant
        return related[:5]
    
    def _extract_entities_simple(self, text: str) -> List[str]:
        """Simple entity extraction without using LLM."""
        import re
        # Look for capitalized words and phrases as a simple heuristic
        entities = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        return list(set(entities))
    
    async def _update_temporal_graph(
        self, 
        thread_id: str, 
        new_facts: List[FactEntry],
        existing_facts: List[FactEntry]
    ) -> None:
        """
        Update the temporal knowledge graph with new facts.
        
        Args:
            thread_id: Thread ID
            new_facts: New facts to add
            existing_facts: Existing facts in registry
        """
        if not self.enable_temporal_graph:
            return
            
        # Get or create graph for this thread
        if thread_id not in self._temporal_graphs:
            self._temporal_graphs[thread_id] = {"nodes": {}, "edges": []}
        
        graph = self._temporal_graphs[thread_id]
        
        # Add new facts as nodes
        for fact in new_facts:
            # Node properties
            node_props = {
                "id": fact.fact_id,
                "content": fact.content,
                "source": fact.source,
                "timestamp": fact.timestamp,
                "confidence": fact.confidence,
                "epistemological_status": fact.epistemological_status,
                "entities": fact.entities
            }
            
            # Add to nodes dictionary
            graph["nodes"][fact.fact_id] = node_props
            
            # Find related existing facts and create edges
            related_facts = self._find_related_facts(fact.content, existing_facts)
            for related_fact in related_facts:
                relation_type = await self._determine_relation_type(fact.content, related_fact.content)
                
                # Create edge in both directions
                edge = {
                    "source": fact.fact_id,
                    "target": related_fact.fact_id,
                    "type": relation_type,
                    "timestamp": time.time()
                }
                
                reverse_edge = {
                    "source": related_fact.fact_id,
                    "target": fact.fact_id,
                    "type": relation_type,
                    "timestamp": time.time()
                }
                
                graph["edges"].append(edge)
                graph["edges"].append(reverse_edge)
        
        # Update entity relationships if enabled
        if self.enable_entity_extraction:
            await self._update_entity_relationships(graph, new_facts)
    
    async def _determine_relation_type(self, fact1: str, fact2: str) -> str:
        """Determine the relationship type between two facts."""
        # For simple implementation, we'll use a few basic relation types
        relation_types = [
            "related_to",        # Generic relation
            "supports",          # Fact1 supports fact2
            "contradicts",       # Fact1 contradicts fact2
            "elaborates_on",     # Fact1 elaborates on fact2
            "temporal_before",   # Fact1 temporally precedes fact2
            "temporal_after",    # Fact1 temporally follows fact2
            "same_as"            # Fact1 and fact2 are essentially the same
        ]
        
        # Simple heuristic based on keyword matching
        if fact1.lower() == fact2.lower():
            return "same_as"
            
        # For more complex determinations, we'll need NLP
        # Here we're using a simplified approach
        if "not" in fact1 and fact1.replace("not", "") in fact2:
            return "contradicts"
            
        return "related_to"  # Default relation type
    
    async def _update_entity_relationships(self, graph: Dict[str, Any], facts: List[FactEntry]) -> None:
        """Update the graph with entity relationship information."""
        if not self.enable_entity_extraction:
            return
            
        # Extract all entities from new facts
        all_entities = set()
        for fact in facts:
            if hasattr(fact, 'entities') and fact.entities:
                all_entities.update(fact.entities)
        
        # Add entity nodes if they don't exist
        for entity in all_entities:
            entity_id = f"entity_{self._hash_string(entity)}"
            
            if entity_id not in graph["nodes"]:
                graph["nodes"][entity_id] = {
                    "id": entity_id,
                    "name": entity,
                    "type": "entity",
                    "timestamp": time.time()
                }
            
            # Connect facts to this entity
            for fact in facts:
                if hasattr(fact, 'entities') and entity in fact.entities:
                    # Create edge from fact to entity
                    edge = {
                        "source": fact.fact_id,
                        "target": entity_id,
                        "type": "mentions",
                        "timestamp": time.time()
                    }
                    
                    # Create edge from entity to fact
                    reverse_edge = {
                        "source": entity_id,
                        "target": fact.fact_id,
                        "type": "mentioned_in",
                        "timestamp": time.time()
                    }
                    
                    graph["edges"].append(edge)
                    graph["edges"].append(reverse_edge)
    
    def _hash_string(self, text: str) -> str:
        """Create a simple hash for a string."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()[:8]
    
    async def _generate_corrections(
        self,
        statement: str,
        conflicts: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate suggested corrections for an inconsistent statement.
        
        Args:
            statement: The inconsistent statement
            conflicts: List of detected conflicts
            
        Returns:
            List of suggested corrections
        """
        if not conflicts:
            return []
            
        # Format conflicts for the LLM
        conflicts_str = "\n".join([
            f"- {c.get('existing_fact', 'Unknown fact')}, conflict: {c.get('conflict_type', 'Unknown conflict')}"
            for c in conflicts
        ])
        
        # Generate corrections
        corrections_prompt = ChatPromptTemplate.from_template(
            "The following statement has consistency issues:\n\n"
            "{statement}\n\n"
            "Conflicts detected:\n{conflicts}\n\n"
            "Please generate specific corrections that would resolve these issues. "
            "For each suggestion, provide the corrected statement and a brief explanation. "
            "Focus on minimal changes that preserve the original meaning while fixing inconsistencies."
        )
        
        corrections_chain = corrections_prompt | self.llm.with_structured_output(
            CorrectionResult
        )
        
        try:
            result = await corrections_chain.ainvoke({
                "statement": statement,
                "conflicts": conflicts_str
            })
            
            return result.corrections
        except Exception as e:
            logger.error(f"Error generating corrections: {e}")
            return [f"Unable to generate specific corrections due to: {str(e)}"]
    
    def _get_thread_facts(self, thread_id: str) -> List[FactEntry]:
        """
        Get facts associated with a specific thread.
        
        Args:
            thread_id: The thread ID
            
        Returns:
            List of facts for the thread
        """
        # Get thread-specific facts from our registry
        thread_facts = self._fact_registry.get(thread_id, [])
        
        # Convert to FactEntry objects if they're dictionaries
        result = []
        for fact in thread_facts:
            if isinstance(fact, dict):
                result.append(FactEntry(**fact))
            else:
                result.append(fact)
        
        return result
    
    async def _apply_user_feedback(self, thread_id: str, feedback: Dict[str, Any]) -> None:
        """
        Apply user feedback to facts in the registry.
        
        Args:
            thread_id: Thread ID
            feedback: User feedback dictionary with fact IDs and corrections
        """
        if not self.enable_user_feedback or not feedback:
            return
            
        thread_facts_data = await self._load_thread_facts(thread_id)
        updated_facts = []
        
        for fact_dict in thread_facts_data:
            fact_id = fact_dict.get('fact_id', '')
            
            if fact_id in feedback:
                # Apply feedback
                user_correction = feedback[fact_id]
                
                # Update with user feedback
                if "corrected_content" in user_correction:
                    fact_dict["content"] = user_correction["corrected_content"]
                
                if "user_confidence" in user_correction:
                    fact_dict["confidence"] = user_correction["user_confidence"]
                
                # Mark as verified by user if specified
                if user_correction.get("mark_as_verified", False):
                    fact_dict["epistemological_status"] = "verified"
                
                # Add to verification history
                if "verification_history" not in fact_dict:
                    fact_dict["verification_history"] = []
                
                fact_dict["verification_history"].append({
                    "timestamp": time.time(),
                    "source": "user_feedback",
                    "action": "correction" if "corrected_content" in user_correction else "verification"
                })
            
            updated_facts.append(fact_dict)
        
        # Update registry and save to checkpointer
        await self._save_thread_facts(thread_id, updated_facts)
    
    async def _get_epistemological_distribution(self, thread_id: str) -> Dict[str, int]:
        """Get the distribution of epistemological statuses for a thread."""
        if not self.enable_epistemic_tagging:
            return {}
        
        thread_facts_data = await self._load_thread_facts(thread_id)
        facts = [FactEntry(**fact) for fact in thread_facts_data]
        
        distribution = {
            "verified": 0,
            "probable": 0,
            "speculative": 0,
            "unverified": 0,
            "disputed": 0
        }
        
        for fact in facts:
            if hasattr(fact, 'epistemological_status'):
                status = fact.epistemological_status
                if status in distribution:
                    distribution[status] += 1
        
        return distribution
    
    def get_all_thread_facts(self, thread_id: str) -> List[Dict[str, Any]]:
        """
        Get all facts for a specific thread.
        
        Args:
            thread_id: The thread ID
            
        Returns:
            All facts for the thread
        """
        facts = self._get_thread_facts(thread_id)
        return [fact.model_dump() if hasattr(fact, 'model_dump') else fact for fact in facts]
    
    def get_temporal_graph(self, thread_id: str) -> Dict[str, Any]:
        """
        Get the temporal knowledge graph for a thread.
        
        Args:
            thread_id: The thread ID
            
        Returns:
            Temporal graph for the thread
        """
        if not self.enable_temporal_graph:
            return {"nodes": {}, "edges": []}
            
        return self._temporal_graphs.get(thread_id, {"nodes": {}, "edges": []})
    
    def get_facts_by_entity(self, thread_id: str, entity: str) -> List[Dict[str, Any]]:
        """
        Get all facts related to a specific entity.
        
        Args:
            thread_id: The thread ID
            entity: Entity to look for
            
        Returns:
            List of facts mentioning the entity
        """
        facts = self._get_thread_facts(thread_id)
        entity_related = []
        
        for fact in facts:
            if hasattr(fact, 'entities') and entity in fact.entities:
                entity_related.append(fact.model_dump() if hasattr(fact, 'model_dump') else fact)
        
        return entity_related
    
    def clear_thread_facts(self, thread_id: str) -> None:
        """
        Clear all facts for a specific thread.
        
        Args:
            thread_id: The thread ID
        """
        if thread_id in self._fact_registry:
            del self._fact_registry[thread_id]
        
        # Also clear temporal graph if enabled
        if self.enable_temporal_graph and thread_id in self._temporal_graphs:
            del self._temporal_graphs[thread_id]
    
    def get_fact_verification_history(self, thread_id: str, fact_id: str) -> List[Dict[str, Any]]:
        """
        Get the verification history for a specific fact.
        
        Args:
            thread_id: Thread ID
            fact_id: Fact ID
            
        Returns:
            Verification history for the fact
        """
        facts = self._get_thread_facts(thread_id)
        
        for fact in facts:
            if hasattr(fact, 'fact_id') and fact.fact_id == fact_id:
                return fact.verification_history if hasattr(fact, 'verification_history') else []
        
        return [] 