# src/assistant/memory/knowledge_graph.py
"""
Temporal Knowledge Graph for AI memory

Implements a knowledge graph that can track entities and relationships over time,
providing long-term memory capabilities for AI agents.

This is inspired by Zep's temporal knowledge graph approach mentioned in:
https://www.getzep.com/ai-agents/reducing-llm-hallucinations
"""

import logging
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class TemporalKnowledgeGraph:
    """
    Implements a temporal knowledge graph for AI memory
    
    This knowledge graph tracks:
    - Entities (nodes) with types and properties
    - Relationships (edges) between entities
    - Temporal information about when entities/relationships were created or modified
    
    This provides grounding for AI systems, reducing hallucinations by giving
    access to previously established facts and relationships.
    """
    
    def __init__(self, storage_path: str = "./knowledge_graph.json"):
        """
        Initialize the temporal knowledge graph
        
        Args:
            storage_path: Path to store the knowledge graph data
        """
        self.storage_path = storage_path
        self.entities = {}  # type: Dict[str, Dict[str, Any]]
        self.relationships = []  # type: List[Dict[str, Any]]
        self.latest_update = 0  # Unix timestamp of latest update
        
        # Load existing graph if available
        self._load_graph()
        
        logger.info(f"Temporal Knowledge Graph initialized with {len(self.entities)} entities and {len(self.relationships)} relationships")
    
    def _load_graph(self) -> bool:
        """
        Load the knowledge graph from storage
        
        Returns:
            Success flag
        """
        if not os.path.exists(self.storage_path):
            logger.info(f"No existing knowledge graph found at {self.storage_path}")
            return False
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            self.entities = data.get('entities', {})
            self.relationships = data.get('relationships', [])
            self.latest_update = data.get('latest_update', 0)
            
            logger.info(f"Loaded knowledge graph with {len(self.entities)} entities and {len(self.relationships)} relationships")
            return True
            
        except Exception as e:
            logger.error(f"Error loading knowledge graph: {e}")
            return False
    
    def _save_graph(self) -> bool:
        """
        Save the knowledge graph to storage
        
        Returns:
            Success flag
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            # Update timestamp
            self.latest_update = int(time.time())
            
            # Save data
            data = {
                'entities': self.entities,
                'relationships': self.relationships,
                'latest_update': self.latest_update
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved knowledge graph with {len(self.entities)} entities and {len(self.relationships)} relationships")
            return True
            
        except Exception as e:
            logger.error(f"Error saving knowledge graph: {e}")
            return False
    
    def add_entity(self, entity_id: str, entity_type: str, properties: Dict[str, Any] = None) -> bool:
        """
        Add an entity to the knowledge graph
        
        Args:
            entity_id: Unique identifier for the entity
            entity_type: Type of entity (e.g., "Person", "Organization", "Document")
            properties: Additional properties of the entity
            
        Returns:
            Success flag
        """
        timestamp = int(time.time())
        
        # Check if entity already exists
        if entity_id in self.entities:
            # Update existing entity
            entity = self.entities[entity_id]
            
            # Save previous version in history
            if 'history' not in entity:
                entity['history'] = []
            
            entity['history'].append({
                'type': entity['type'],
                'properties': entity.get('properties', {}),
                'timestamp': entity['timestamp']
            })
            
            # Update entity
            entity['type'] = entity_type
            entity['properties'] = properties or {}
            entity['timestamp'] = timestamp
            entity['updated_at'] = timestamp
            
            logger.info(f"Updated entity: {entity_id} (type: {entity_type})")
        else:
            # Create new entity
            self.entities[entity_id] = {
                'id': entity_id,
                'type': entity_type,
                'properties': properties or {},
                'timestamp': timestamp,
                'created_at': timestamp,
                'updated_at': timestamp,
                'history': []
            }
            
            logger.info(f"Added new entity: {entity_id} (type: {entity_type})")
        
        # Save changes
        return self._save_graph()
    
    def add_relationship(self, from_entity: str, to_entity: str, relationship_type: str, properties: Dict[str, Any] = None) -> bool:
        """
        Add a relationship between entities
        
        Args:
            from_entity: ID of the source entity
            to_entity: ID of the target entity
            relationship_type: Type of relationship (e.g., "KNOWS", "CONTAINS", "PART_OF")
            properties: Additional properties of the relationship
            
        Returns:
            Success flag
        """
        timestamp = int(time.time())
        
        # Check if both entities exist
        if from_entity not in self.entities or to_entity not in self.entities:
            logger.error(f"Cannot create relationship: one or both entities do not exist ({from_entity}, {to_entity})")
            return False
        
        # Check for existing relationship
        existing_idx = -1
        for idx, rel in enumerate(self.relationships):
            if (rel['from'] == from_entity and
                rel['to'] == to_entity and
                rel['type'] == relationship_type):
                existing_idx = idx
                break
        
        if existing_idx >= 0:
            # Update existing relationship
            relationship = self.relationships[existing_idx]
            
            # Save previous version in history
            if 'history' not in relationship:
                relationship['history'] = []
            
            relationship['history'].append({
                'properties': relationship.get('properties', {}),
                'timestamp': relationship['timestamp']
            })
            
            # Update relationship
            relationship['properties'] = properties or {}
            relationship['timestamp'] = timestamp
            relationship['updated_at'] = timestamp
            
            logger.info(f"Updated relationship: {from_entity} -{relationship_type}-> {to_entity}")
        else:
            # Create new relationship
            self.relationships.append({
                'from': from_entity,
                'to': to_entity,
                'type': relationship_type,
                'properties': properties or {},
                'timestamp': timestamp,
                'created_at': timestamp,
                'updated_at': timestamp,
                'history': []
            })
            
            logger.info(f"Added new relationship: {from_entity} -{relationship_type}-> {to_entity}")
        
        # Save changes
        return self._save_graph()
    
    def get_entity(self, entity_id: str, include_history: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get an entity by ID
        
        Args:
            entity_id: ID of the entity to retrieve
            include_history: Whether to include historical versions
            
        Returns:
            Entity data or None if not found
        """
        if entity_id not in self.entities:
            return None
        
        entity = dict(self.entities[entity_id])
        
        if not include_history:
            entity.pop('history', None)
        
        return entity
    
    def get_relationships(self, entity_id: str, direction: str = 'both', relationship_type: str = None) -> List[Dict[str, Any]]:
        """
        Get relationships for an entity
        
        Args:
            entity_id: ID of the entity
            direction: 'outgoing', 'incoming', or 'both'
            relationship_type: Optional filter by relationship type
            
        Returns:
            List of relationships
        """
        result = []
        
        for rel in self.relationships:
            # Check direction
            is_outgoing = rel['from'] == entity_id
            is_incoming = rel['to'] == entity_id
            
            if ((direction == 'outgoing' and is_outgoing) or
                (direction == 'incoming' and is_incoming) or
                (direction == 'both' and (is_outgoing or is_incoming))):
                
                # Check relationship type if specified
                if relationship_type is None or rel['type'] == relationship_type:
                    result.append(dict(rel))
        
        return result
    
    def query_entities(self, entity_type: str = None, properties: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Query entities by type and properties
        
        Args:
            entity_type: Optional entity type to filter by
            properties: Optional properties to filter by
            
        Returns:
            List of matching entities
        """
        result = []
        
        for entity_id, entity in self.entities.items():
            # Check entity type if specified
            if entity_type is not None and entity['type'] != entity_type:
                continue
            
            # Check properties if specified
            if properties is not None:
                match = True
                for key, value in properties.items():
                    if key not in entity.get('properties', {}) or entity['properties'][key] != value:
                        match = False
                        break
                
                if not match:
                    continue
            
            # Entity matches all criteria
            result.append(dict(entity))
        
        return result
    
    def get_entity_at_time(self, entity_id: str, timestamp: int) -> Optional[Dict[str, Any]]:
        """
        Get an entity as it existed at a specific time
        
        Args:
            entity_id: ID of the entity
            timestamp: Unix timestamp
            
        Returns:
            Entity data at that time or None if not found
        """
        if entity_id not in self.entities:
            return None
        
        entity = self.entities[entity_id]
        
        # If entity was created after the timestamp, it didn't exist then
        if entity['created_at'] > timestamp:
            return None
        
        # If current version was created before or at the timestamp, return it
        if entity['created_at'] <= timestamp and ('updated_at' not in entity or entity['updated_at'] <= timestamp):
            return dict(entity)
        
        # Otherwise, search history for the version that existed at that time
        if 'history' in entity:
            # Sort history by timestamp (newest first)
            sorted_history = sorted(entity['history'], key=lambda x: x['timestamp'], reverse=True)
            
            for historical_version in sorted_history:
                if historical_version['timestamp'] <= timestamp:
                    # Found the version that existed at that time
                    result = {
                        'id': entity_id,
                        'type': historical_version['type'],
                        'properties': historical_version['properties'],
                        'timestamp': historical_version['timestamp'],
                        'created_at': entity['created_at']
                    }
                    return result
        
        # If we got here, entity existed but we don't have the version at that time
        return None
    
    def get_temporal_path(self, from_entity: str, to_entity: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        """
        Find a path between entities, showing how their relationship evolved over time
        
        Args:
            from_entity: Starting entity ID
            to_entity: Target entity ID
            max_depth: Maximum path length to search
            
        Returns:
            List of relationships forming the path, with temporal information
        """
        # This is a simplified BFS pathfinding implementation
        # In a production system, more sophisticated graph algorithms would be used
        
        if from_entity not in self.entities or to_entity not in self.entities:
            return []
        
        # Initialize queue with starting entity
        queue = [(from_entity, [])]
        visited = set([from_entity])
        
        while queue:
            current_id, path = queue.pop(0)
            
            # If we've reached the target, return the path
            if current_id == to_entity:
                return path
            
            # If we've reached max depth, skip this branch
            if len(path) >= max_depth:
                continue
            
            # Find all relationships from this entity
            for rel in self.relationships:
                if rel['from'] == current_id and rel['to'] not in visited:
                    next_id = rel['to']
                    next_path = path + [dict(rel)]
                    queue.append((next_id, next_path))
                    visited.add(next_id)
                
                # Also check incoming relationships
                if rel['to'] == current_id and rel['from'] not in visited:
                    next_id = rel['from']
                    # Mark this as a reversed relationship
                    rel_copy = dict(rel)
                    rel_copy['direction'] = 'incoming'
                    next_path = path + [rel_copy]
                    queue.append((next_id, next_path))
                    visited.add(next_id)
        
        # If we get here, no path was found
        return []
    
    def get_subgraph(self, entity_ids: List[str], include_relationships: bool = True) -> Dict[str, Any]:
        """
        Get a subgraph containing specified entities and their relationships
        
        Args:
            entity_ids: List of entity IDs to include
            include_relationships: Whether to include relationships between entities
            
        Returns:
            Subgraph data
        """
        entities = {}
        relationships = []
        
        # Include specified entities
        for entity_id in entity_ids:
            if entity_id in self.entities:
                entities[entity_id] = dict(self.entities[entity_id])
        
        # Include relationships if requested
        if include_relationships:
            for rel in self.relationships:
                if rel['from'] in entity_ids and rel['to'] in entity_ids:
                    relationships.append(dict(rel))
        
        return {
            'entities': entities,
            'relationships': relationships
        }
    
    def export_graph(self, format: str = 'json') -> str:
        """
        Export the knowledge graph in specified format
        
        Args:
            format: Output format ('json' or 'cypher')
            
        Returns:
            Exported graph data as string
        """
        if format == 'json':
            data = {
                'entities': self.entities,
                'relationships': self.relationships,
                'latest_update': self.latest_update
            }
            return json.dumps(data, indent=2)
        
        elif format == 'cypher':
            # Export as Cypher queries for Neo4j
            cypher = []
            
            # Create entities
            for entity_id, entity in self.entities.items():
                properties_str = ', '.join([f'{k}: {json.dumps(v)}' for k, v in entity.get('properties', {}).items()])
                cypher.append(f"CREATE (e:{entity['type']} {{id: '{entity_id}', {properties_str}}})")
            
            # Create relationships
            for rel in self.relationships:
                properties_str = ', '.join([f'{k}: {json.dumps(v)}' for k, v in rel.get('properties', {}).items()])
                cypher.append(
                    f"MATCH (a), (b) WHERE a.id = '{rel['from']}' AND b.id = '{rel['to']}' "
                    f"CREATE (a)-[r:{rel['type']} {{{properties_str}}}]->(b)"
                )
            
            return '\n'.join(cypher)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def clear(self) -> bool:
        """
        Clear the knowledge graph
        
        Returns:
            Success flag
        """
        self.entities = {}
        self.relationships = []
        self.latest_update = int(time.time())
        
        logger.info("Knowledge graph cleared")
        return self._save_graph()
    
    def stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph
        
        Returns:
            Statistical information
        """
        entity_types = {}
        for entity in self.entities.values():
            entity_type = entity['type']
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        relationship_types = {}
        for rel in self.relationships:
            rel_type = rel['type']
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
        
        return {
            'entity_count': len(self.entities),
            'relationship_count': len(self.relationships),
            'entity_types': entity_types,
            'relationship_types': relationship_types,
            'latest_update': self.latest_update,
            'latest_update_formatted': datetime.fromtimestamp(self.latest_update).isoformat() if self.latest_update else None
        }
