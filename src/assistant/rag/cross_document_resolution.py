"""
Cross-document coreference resolution system that identifies and links
entity mentions across document boundaries.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrossDocumentResolver:
    """
    Cross-document coreference resolution system that identifies and links entities
    across document boundaries to create a unified understanding.
    
    This component:
    1. Extracts entity mentions from documents
    2. Clusters mentions that refer to the same entity
    3. Builds an entity graph connecting mentions across documents
    4. Provides context enhancement by resolving ambiguous references
    """
    
    def __init__(self, llm: BaseChatModel):
        """
        Initialize the cross-document resolver.
        
        Args:
            llm: Language model for entity extraction and resolution
        """
        self.llm = llm
        self.entity_mentions = defaultdict(list)  # entity_id -> [(doc_id, text_span)]
        self.mention_clusters = {}  # mention_id -> entity_id
        self.entity_metadata = {}  # entity_id -> metadata
        self.document_entities = defaultdict(set)  # doc_id -> {entity_ids}
        
        self.next_entity_id = 1
        self.next_mention_id = 1
        
        logger.info("Initialized CrossDocumentResolver")
    
    async def process_document(self, doc_id: str, content: str) -> List[Dict[str, Any]]:
        """
        Process a document to extract entities and integrate with existing knowledge.
        
        Args:
            doc_id: Document identifier
            content: Document content
            
        Returns:
            List of extracted entity mentions with metadata
        """
        logger.info(f"Processing document {doc_id} for entity extraction")
        
        try:
            # Extract entities from document
            entities = await self._extract_entities(content)
            
            # Register each entity and assign to document
            processed_entities = []
            for entity in entities:
                entity_mention_id = self._get_next_mention_id()
                entity_text = entity.get("text", "")
                entity_type = entity.get("type", "unknown")
                
                # Try to resolve entity against existing entities
                entity_id = await self._resolve_entity(entity, doc_id)
                
                # Register the mention
                self.entity_mentions[entity_id].append((doc_id, entity_text))
                self.mention_clusters[entity_mention_id] = entity_id
                self.document_entities[doc_id].add(entity_id)
                
                # Update entity metadata if available
                if entity_id in self.entity_metadata:
                    # Update existing metadata with new information
                    existing_metadata = self.entity_metadata[entity_id]
                    
                    # Merge mentions
                    if "mentions" in existing_metadata:
                        existing_metadata["mentions"].append(entity_text)
                    else:
                        existing_metadata["mentions"] = [entity_text]
                    
                    # Update type if more specific
                    if entity_type != "unknown" and existing_metadata.get("type") == "unknown":
                        existing_metadata["type"] = entity_type
                    
                    # Add document to doc_ids
                    if "doc_ids" in existing_metadata:
                        existing_metadata["doc_ids"].add(doc_id)
                    else:
                        existing_metadata["doc_ids"] = {doc_id}
                else:
                    # Create new metadata
                    self.entity_metadata[entity_id] = {
                        "type": entity_type,
                        "mentions": [entity_text],
                        "doc_ids": {doc_id},
                        "canonical_name": entity.get("canonical_name", entity_text)
                    }
                
                # Add to processed entities
                processed_entities.append({
                    "mention_id": entity_mention_id,
                    "entity_id": entity_id,
                    "text": entity_text,
                    "type": entity_type,
                    "canonical_name": self.entity_metadata[entity_id].get("canonical_name", entity_text),
                    "doc_id": doc_id
                })
            
            logger.info(f"Extracted {len(processed_entities)} entity mentions from document {doc_id}")
            return processed_entities
            
        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {e}")
            return []
    
    async def resolve_references(self, text: str, doc_id: Optional[str] = None) -> str:
        """
        Resolve entity references in text by replacing ambiguous mentions
        with canonical entity names when possible.
        
        Args:
            text: Text with potential entity references
            doc_id: Optional document ID for context
            
        Returns:
            Text with resolved references
        """
        try:
            # If document ID is provided, use its entities as context
            context_entities = set()
            if doc_id and doc_id in self.document_entities:
                context_entities = self.document_entities[doc_id]
                
                # Add entities from related documents
                for entity_id in context_entities:
                    for related_doc_id, _ in self.entity_mentions[entity_id]:
                        if related_doc_id != doc_id:
                            context_entities.update(self.document_entities[related_doc_id])
            
            # If no document context, use all entities
            if not context_entities:
                context_entities = set(self.entity_metadata.keys())
            
            # Prepare context for resolution
            context_entities_data = []
            for entity_id in context_entities:
                metadata = self.entity_metadata.get(entity_id, {})
                context_entities_data.append({
                    "id": entity_id,
                    "name": metadata.get("canonical_name", f"Entity_{entity_id}"),
                    "type": metadata.get("type", "unknown"),
                    "mentions": metadata.get("mentions", [])
                })
            
            # Prompt for entity resolution
            prompt = ChatPromptTemplate.from_template(
                """You are a language processing system specializing in entity reference resolution.
                
                Your task is to identify ambiguous entity references in the text and resolve them to their canonical forms.
                
                Entity context:
                {entity_context}
                
                Text to process:
                {text}
                
                Instructions:
                1. Identify references (pronouns, abbreviations, partial names, etc.) that likely refer to entities in the context
                2. Replace ambiguous references with clearer ones using [ENTITY: canonical_name] format
                3. Only replace references when you're confident of the correct entity
                4. Don't add information not in the original text, just resolve references
                5. Preserve the original meaning and tone of the text
                
                Return the text with resolved references.
                """
            )
            
            # Format entity context
            entity_context_text = ""
            for entity in context_entities_data:
                entity_context_text += f"- ID: {entity['id']}, Name: {entity['name']}, Type: {entity['type']}\n"
                entity_context_text += f"  Mentions: {', '.join(entity['mentions'][:5])}\n"
            
            # Execute entity resolution
            chain = prompt | self.llm | StrOutputParser()
            resolved_text = await chain.ainvoke({
                "text": text,
                "entity_context": entity_context_text
            })
            
            return resolved_text
            
        except Exception as e:
            logger.error(f"Error resolving references: {e}")
            return text  # Return original text on error
    
    async def get_entity_context(self, entity_id: int, max_contexts: int = 3) -> str:
        """
        Get contextual information about an entity from across documents.
        
        Args:
            entity_id: Entity identifier
            max_contexts: Maximum number of context snippets to return
            
        Returns:
            Combined entity context from multiple documents
        """
        if entity_id not in self.entity_metadata:
            return ""
        
        try:
            # Get entity mentions across documents
            mentions = self.entity_mentions.get(entity_id, [])
            if not mentions:
                return ""
            
            # Get metadata
            metadata = self.entity_metadata[entity_id]
            canonical_name = metadata.get("canonical_name", f"Entity_{entity_id}")
            entity_type = metadata.get("type", "unknown")
            
            # Build context header
            context = f"Entity: {canonical_name} (Type: {entity_type})\n\n"
            
            # Add contexts from different documents
            doc_contexts = []
            seen_docs = set()
            
            for doc_id, mention_text in mentions:
                if doc_id in seen_docs:
                    continue
                
                # TODO: Retrieve actual document context around the mention
                # For now, just use the mention
                doc_contexts.append(f"Document {doc_id}: ... {mention_text} ...")
                seen_docs.add(doc_id)
                
                if len(doc_contexts) >= max_contexts:
                    break
            
            # Add document contexts
            context += "\n".join(doc_contexts)
            return context
            
        except Exception as e:
            logger.error(f"Error getting entity context: {e}")
            return ""
    
    def get_entity_graph(self) -> Dict[str, Any]:
        """
        Get the current entity graph with cross-document connections.
        
        Returns:
            Entity graph data structure
        """
        return {
            "entities": self.entity_metadata,
            "document_entities": {doc_id: list(entities) for doc_id, entities in self.document_entities.items()},
            "statistics": {
                "entity_count": len(self.entity_metadata),
                "mention_count": sum(len(mentions) for mentions in self.entity_mentions.values()),
                "document_count": len(self.document_entities)
            }
        }
    
    async def _extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract entity mentions from document content.
        
        Args:
            content: Document content
            
        Returns:
            List of extracted entities with metadata
        """
        try:
            # Define prompt for entity extraction
            prompt = ChatPromptTemplate.from_template(
                """Extract all entity mentions from the following text. Focus on named entities, technical terms, and important concepts.
                
                Text:
                {content}
                
                For each entity, provide:
                1. The exact text of the mention
                2. The entity type (PERSON, ORGANIZATION, TECHNOLOGY, CONCEPT, etc.)
                3. A canonical name (standardized version of the name)
                
                Format your response as a list of JSON objects, one per line:
                {"text": "mention text", "type": "entity type", "canonical_name": "canonical name"}
                {"text": "mention text", "type": "entity type", "canonical_name": "canonical name"}
                
                Only include the JSON objects, with no additional text.
                """
            )
            
            # Execute entity extraction
            chain = prompt | self.llm | StrOutputParser()
            response = await chain.ainvoke({"content": content})
            
            # Parse extraction results
            entities = []
            for line in response.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Handle JSON-like output that might not be perfectly formatted
                    # Extract fields using regex
                    text_match = re.search(r'"text"\s*:\s*"([^"]+)"', line)
                    type_match = re.search(r'"type"\s*:\s*"([^"]+)"', line)
                    canonical_match = re.search(r'"canonical_name"\s*:\s*"([^"]+)"', line)
                    
                    if text_match:
                        entity = {
                            "text": text_match.group(1),
                            "type": type_match.group(1) if type_match else "unknown",
                            "canonical_name": canonical_match.group(1) if canonical_match else text_match.group(1)
                        }
                        entities.append(entity)
                except Exception as e:
                    logger.warning(f"Error parsing entity from line '{line}': {e}")
                    continue
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    async def _resolve_entity(self, entity: Dict[str, Any], doc_id: str) -> int:
        """
        Resolve an entity mention against existing entities.
        
        Args:
            entity: Entity mention data
            doc_id: Document identifier
            
        Returns:
            Entity ID (existing or new)
        """
        try:
            entity_text = entity.get("text", "")
            canonical_name = entity.get("canonical_name", entity_text)
            
            # Look for exact matches in canonical names
            for entity_id, metadata in self.entity_metadata.items():
                if metadata.get("canonical_name") == canonical_name:
                    return entity_id
            
            # Look for fuzzy matches in mentions
            for entity_id, mentions in self.entity_mentions.items():
                for mention_doc_id, mention_text in mentions:
                    # If texts are very similar or entity text contains/is contained in mention
                    if (mention_text.lower() == entity_text.lower() or
                        mention_text.lower() in entity_text.lower() or
                        entity_text.lower() in mention_text.lower()):
                        return entity_id
            
            # No match found, create new entity
            new_entity_id = self._get_next_entity_id()
            return new_entity_id
            
        except Exception as e:
            logger.error(f"Error resolving entity: {e}")
            # Return new entity ID on error
            return self._get_next_entity_id()
    
    def _get_next_entity_id(self) -> int:
        """Get next available entity ID"""
        entity_id = self.next_entity_id
        self.next_entity_id += 1
        return entity_id
    
    def _get_next_mention_id(self) -> int:
        """Get next available mention ID"""
        mention_id = self.next_mention_id
        self.next_mention_id += 1
        return mention_id 