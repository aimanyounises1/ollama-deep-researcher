# src/assistant/rag/hybrid_retriever.py
"""
Hybrid Retrieval implementation that combines multiple retrieval methods
for more comprehensive document exploration.
"""

import logging
import os
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Union, Set, Tuple
from collections import defaultdict

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.assistant.rag.retrieval import RAGRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Advanced hybrid retrieval system that combines multiple retrieval methods
    for more comprehensive document exploration.
    
    Features:
    - Multi-query expansion for different perspectives
    - Hybrid retrieval combining dense and sparse approaches
    - Parent-child document relationship preservation
    - Cross-document coreference resolution
    - Recursive retrieval for deeper exploration
    - Hypothetical Document Embeddings (HyDE)
    - RAG-Fusion with Reciprocal Rank Fusion
    - Step-back prompting for high-level context
    """
    
    def __init__(
        self, 
        semantic_retriever: RAGRetriever,
        llm: Optional[BaseChatModel] = None,
        ollama_base_url: str = None,
        local_llm_name: str = None,
        enable_multi_query: bool = True,
        enable_recursive: bool = True,
        enable_hyde: bool = True,
        enable_rag_fusion: bool = True,
        enable_step_back: bool = True,
        max_recursive_depth: int = 2,
        relevance_threshold: float = 0.7
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            semantic_retriever: Base RAG retriever for semantic search
            llm: Language model for query expansion (optional)
            ollama_base_url: Base URL for Ollama (if llm not provided)
            local_llm_name: Model name for Ollama (if llm not provided)
            enable_multi_query: Whether to use multi-query expansion
            enable_recursive: Whether to use recursive retrieval
            enable_hyde: Whether to use Hypothetical Document Embeddings
            enable_rag_fusion: Whether to use RAG-Fusion
            enable_step_back: Whether to use step-back prompting
            max_recursive_depth: Maximum recursion depth for recursive retrieval
            relevance_threshold: Minimum relevance score for documents
        """
        self.semantic_retriever = semantic_retriever
        
        if llm:
            self.llm = llm
        else:
            # Initialize LLM for query expansion if not provided
            self.ollama_base_url = ollama_base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            self.local_llm_name = local_llm_name or os.environ.get("LOCAL_LLM", "deepseek-coder:33b-instruct")
            
            self.llm = ChatOllama(
                base_url=self.ollama_base_url,
                model=self.local_llm_name,
                temperature=0.7,  # Higher temperature for diverse queries
            )
        
        # Configuration options
        self.enable_multi_query = enable_multi_query
        self.enable_recursive = enable_recursive
        self.enable_hyde = enable_hyde
        self.enable_rag_fusion = enable_rag_fusion
        self.enable_step_back = enable_step_back
        self.max_recursive_depth = max_recursive_depth
        self.relevance_threshold = relevance_threshold
        
        # Document relationship tracking
        self.document_relationships = defaultdict(set)  # parent -> {children}
        self.entity_mentions = defaultdict(set)  # entity -> {document_ids}
        
        logger.info(f"Initialized HybridRetriever with multi_query={enable_multi_query}, recursive={enable_recursive}, "
                   f"hyde={enable_hyde}, rag_fusion={enable_rag_fusion}, step_back={enable_step_back}")
    
    async def retrieve(
        self, 
        query: str, 
        max_docs: int = 8,
        include_parent_context: bool = True
    ) -> List[str]:
        """
        Perform hybrid retrieval using multiple methods.
        
        Args:
            query: The search query
            max_docs: Maximum number of documents to retrieve
            include_parent_context: Whether to include parent document context
            
        Returns:
            List of relevant document contents
        """
        logger.info(f"Hybrid retrieval for query: '{query}'")
        
        # Step-back prompting if enabled
        if self.enable_step_back:
            step_back_query = await self._generate_step_back_query(query)
            if step_back_query:
                logger.info(f"Generated step-back query: '{step_back_query}'")
                step_back_docs = await self.semantic_retriever.retrieve(step_back_query, max_docs=3)
            else:
                step_back_docs = []
        else:
            step_back_docs = []
        
        # HyDE approach if enabled
        if self.enable_hyde:
            hyde_docs = await self._perform_hyde_retrieval(query, max_docs=max_docs // 2)
        else:
            hyde_docs = []
        
        # Multi-query expansion (if enabled)
        queries = [query]
        if self.enable_multi_query:
            expanded_queries = await self._generate_multiple_queries(query)
            queries.extend(expanded_queries)
            logger.info(f"Expanded to {len(queries)} queries: {queries}")
        
        # Retrieve documents for each query
        all_docs = set()
        all_docs_with_scores = []
        
        # Keep track of doc-query pairs for potential RAG-Fusion
        doc_query_scores = []
        
        for i, q in enumerate(queries):
            # Get semantic search results
            docs = await self.semantic_retriever.retrieve(q, max_docs=max_docs)
            
            # Add to results (with deduplication)
            for j, doc in enumerate(docs):
                all_docs.add(doc)
                
                # Store for RAG-Fusion - assign a rank-based score
                # Higher ranks (lower position numbers) get higher scores
                rank = j + 1
                score = 1.0 / rank
                doc_query_scores.append((doc, q, score))
            
            # Track for potential recursive retrieval
            all_docs_with_scores.extend([
                {"content": doc, "query": q, "score": 0.9 - (0.1 * i)}  # Lower score for later queries
                for doc in docs
            ])
        
        # Apply RAG-Fusion if enabled and we have multiple queries
        if self.enable_rag_fusion and len(queries) > 1:
            fusion_docs = self._apply_rag_fusion(doc_query_scores, max_docs=max_docs)
            # Add fusion results
            for doc in fusion_docs:
                all_docs.add(doc)
        
        # Add HyDE results
        for doc in hyde_docs:
            all_docs.add(doc)
            
        # Add step-back results
        for doc in step_back_docs:
            all_docs.add(doc)
        
        # Recursive retrieval (if enabled)
        if self.enable_recursive and len(all_docs) > 0:
            recursive_docs = await self._perform_recursive_retrieval(
                all_docs_with_scores, 
                original_query=query,
                depth=1
            )
            
            # Add recursive results (with deduplication)
            for doc in recursive_docs:
                all_docs.add(doc)
        
        # Include parent context (if enabled and available)
        if include_parent_context:
            parent_docs = self._get_parent_contexts(all_docs)
            for doc in parent_docs:
                all_docs.add(doc)
        
        # Return combined results
        logger.info(f"Retrieved {len(all_docs)} unique documents")
        return list(all_docs)
    
    async def _generate_multiple_queries(self, query: str, num_queries: int = 3) -> List[str]:
        """
        Generate multiple query variations using a local LLM.
        
        Args:
            query: The original query
            num_queries: Number of query variations to generate
            
        Returns:
            List of query variations
        """
        try:
            # Prompt for generating multiple query variations
            prompt = ChatPromptTemplate.from_template(
                f"""Generate {num_queries} different search queries that would help find information to answer this question: "{query}"
                
                The queries should:
                1. Approach the question from different angles and perspectives
                2. Use different keywords and phrasings
                3. Include specific technical terms when appropriate
                4. Be concise and focused
                5. Cover different aspects of the question
                
                Format your response as a numbered list with ONLY the queries, one per line:
                1. First query
                2. Second query
                3. Third query
                """
            )
            
            # Generate variations
            chain = prompt | self.llm | StrOutputParser()
            response = await chain.ainvoke({"query": query})
            
            # Parse response
            lines = response.strip().split('\n')
            queries = []
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('- ')):
                    # Extract the query part (removing any numbering)
                    query_text = line.split('.', 1)[-1].strip() if '.' in line else line.split(' ', 1)[-1].strip()
                    queries.append(query_text)
            
            # Exclude the original query to avoid duplication
            queries = [q for q in queries if q != query]
            
            return queries[:num_queries]  # Limit to requested number
            
        except Exception as e:
            logger.error(f"Error generating multiple queries: {e}")
            # Return empty list on failure
            return []
    
    async def _perform_recursive_retrieval(
        self, 
        base_docs_with_scores: List[Dict[str, Any]],
        original_query: str,
        depth: int = 1
    ) -> List[str]:
        """
        Perform recursive retrieval to explore related concepts.
        
        Args:
            base_docs_with_scores: Base documents with relevance scores
            original_query: The original query
            depth: Current recursion depth
            
        Returns:
            Additional documents from recursive exploration
        """
        if depth > self.max_recursive_depth or not base_docs_with_scores:
            return []
        
        try:
            # Extract key concepts from most relevant documents
            concepts = await self._extract_key_concepts(base_docs_with_scores, original_query)
            
            # Generate targeted queries for each concept
            concept_queries = []
            for concept in concepts[:3]:  # Limit to top concepts
                concept_query = f"{original_query} related to {concept}"
                concept_queries.append(concept_query)
            
            # Retrieve documents for concept queries
            recursive_docs = set()
            for query in concept_queries:
                docs = await self.semantic_retriever.retrieve(query, max_docs=3)  # Fewer docs per recursive query
                for doc in docs:
                    recursive_docs.add(doc)
            
            # Continue recursion if needed
            if depth < self.max_recursive_depth:
                next_level_docs = [
                    {"content": doc, "query": original_query, "score": 0.8}  # Lower score for recursive docs
                    for doc in recursive_docs
                ]
                
                deeper_docs = await self._perform_recursive_retrieval(
                    next_level_docs,
                    original_query,
                    depth + 1
                )
                
                for doc in deeper_docs:
                    recursive_docs.add(doc)
            
            return list(recursive_docs)
            
        except Exception as e:
            logger.error(f"Error in recursive retrieval at depth {depth}: {e}")
            return []
    
    async def _extract_key_concepts(
        self, 
        docs_with_scores: List[Dict[str, Any]],
        original_query: str
    ) -> List[str]:
        """
        Extract key concepts from retrieved documents for further exploration.
        
        Args:
            docs_with_scores: List of documents with relevance scores
            original_query: The original query
            
        Returns:
            List of key concepts for recursive retrieval
        """
        try:
            # Sort by relevance score
            sorted_docs = sorted(docs_with_scores, key=lambda x: x.get("score", 0), reverse=True)
            
            # Take top docs for concept extraction
            top_docs = sorted_docs[:3]
            combined_text = "\n\n".join([doc.get("content", "") for doc in top_docs])
            
            # Use LLM to extract key concepts
            prompt = ChatPromptTemplate.from_template(
                f"""Based on these document excerpts related to the query "{original_query}", identify 3-5 key concepts that would be valuable to explore further.
                
                Documents:
                {combined_text}
                
                Extract specific key concepts that:
                1. Are central to understanding the query topic
                2. Would benefit from additional information
                3. Are mentioned but not fully explained in these documents
                
                Format your response as a simple list with ONLY the concepts, one per line:
                - First concept
                - Second concept
                - Third concept
                """
            )
            
            # Generate concepts
            chain = prompt | self.llm | StrOutputParser()
            response = await chain.ainvoke({"query": original_query})
            
            # Parse response
            lines = response.strip().split('\n')
            concepts = []
            for line in lines:
                line = line.strip()
                if line and line.startswith('- '):
                    concepts.append(line[2:].strip())
                elif line and line:
                    concepts.append(line)
            
            return concepts
            
        except Exception as e:
            logger.error(f"Error extracting key concepts: {e}")
            return []
    
    async def _perform_hyde_retrieval(self, query: str, max_docs: int = 4) -> List[str]:
        """
        Implement Hypothetical Document Embeddings (HyDE) approach.
        
        HyDE generates a hypothetical document that would answer the query,
        then uses the embedding of that document to find similar documents.
        
        Args:
            query: The original query
            max_docs: Maximum number of documents to retrieve
            
        Returns:
            List of documents retrieved via HyDE
        """
        try:
            # Generate a hypothetical document that would answer the query
            hyde_prompt = ChatPromptTemplate.from_template(
                """Generate a detailed passage that would serve as a perfect answer to the following question:
                
                Question: {query}
                
                Create a concise yet comprehensive hypothetical document (2-3 paragraphs) that directly answers
                this question with specific details, technical information, and clear explanations.
                Do not include phrases like "I would answer" or "the answer is" - write the passage as if
                it's from an authoritative document that contains the answer.
                """
            )
            
            # Generate hypothetical document
            chain = hyde_prompt | self.llm | StrOutputParser()
            hypothetical_doc = await chain.ainvoke({"query": query})
            
            # Use this hypothetical document as a query for semantic search
            hyde_docs = await self.semantic_retriever.retrieve(hypothetical_doc, max_docs=max_docs)
            
            logger.info(f"HyDE retrieval found {len(hyde_docs)} documents")
            return hyde_docs
            
        except Exception as e:
            logger.error(f"Error in HyDE retrieval: {e}")
            return []
    
    def _apply_rag_fusion(
        self, 
        doc_query_scores: List[Tuple[str, str, float]],
        max_docs: int = 8,
        k: float = 60.0  # RRF constant
    ) -> List[str]:
        """
        Apply Reciprocal Rank Fusion to combine results from multiple queries.
        
        Args:
            doc_query_scores: List of (doc, query, score) tuples
            max_docs: Maximum documents to return
            k: RRF constant parameter
            
        Returns:
            Fused document list
        """
        try:
            # Group by document
            doc_scores = defaultdict(list)
            for doc, query, score in doc_query_scores:
                doc_scores[doc].append(score)
            
            # Apply RRF formula: score = sum(1 / (k + rank_i))
            # Since we already converted ranks to scores as 1/rank,
            # we'll use score_rrf = sum(1 / (k + 1/score))
            fusion_scores = {}
            for doc, scores in doc_scores.items():
                rrf_score = sum(1.0 / (k + (1.0 / score)) for score in scores)
                fusion_scores[doc] = rrf_score
            
            # Sort by fusion score and return top docs
            sorted_docs = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
            result = [doc for doc, score in sorted_docs[:max_docs]]
            
            logger.info(f"RAG-Fusion returned {len(result)} documents")
            return result
            
        except Exception as e:
            logger.error(f"Error in RAG-Fusion: {e}")
            # Return empty list on error
            return []
    
    async def _generate_step_back_query(self, query: str) -> str:
        """
        Generate a "step-back" query that looks at higher-level concepts.
        
        This approach helps to retrieve broader context before diving into specifics.
        
        Args:
            query: The original query
            
        Returns:
            A step-back query focusing on higher-level concepts
        """
        try:
            # Prompt for generating step-back query
            step_back_prompt = ChatPromptTemplate.from_template(
                """For the following specific question, create a "step-back" question that asks about the
                broader topic or concept that would help establish foundational context.
                
                Original question: {query}
                
                For example:
                - If the original question is "How does React's virtual DOM improve performance?",
                  a good step-back question would be "What is the virtual DOM and how does it work?"
                - If the original question is "What are the best techniques for preventing hallucinations in LLMs?",
                  a good step-back question would be "What are hallucinations in language models and why do they occur?"
                
                Generate a step-back question that would help establish important context for the original question.
                Respond with ONLY the step-back question, nothing else.
                """
            )
            
            # Generate step-back query
            chain = step_back_prompt | self.llm | StrOutputParser()
            step_back_query = await chain.ainvoke({"query": query})
            
            # Clean up
            step_back_query = step_back_query.strip().strip('"').strip()
            
            return step_back_query
            
        except Exception as e:
            logger.error(f"Error generating step-back query: {e}")
            return ""
    
    def _get_parent_contexts(self, docs: Set[str]) -> List[str]:
        """
        Get parent document contexts for child documents.
        
        Args:
            docs: Set of document contents
            
        Returns:
            List of parent document contents
        """
        # This would connect to your document storage system
        # and retrieve parent documents for any children in the result set
        # Simplified implementation for now
        return []
    
    def add_document_relationship(self, parent_id: str, child_id: str):
        """
        Register a parent-child relationship between documents.
        
        Args:
            parent_id: Parent document ID
            child_id: Child document ID
        """
        self.document_relationships[parent_id].add(child_id)
    
    def add_entity_mention(self, entity: str, document_id: str):
        """
        Register an entity mention in a document.
        
        Args:
            entity: Entity name or identifier
            document_id: Document where the entity is mentioned
        """
        self.entity_mentions[entity].add(document_id)
    
    async def add_documents(self, documents: List[Union[str, Dict[str, Any]]]):
        """
        Add documents to the underlying retriever.
        
        Args:
            documents: List of documents or document dicts to add
        """
        await self.semantic_retriever.add_documents(documents) 