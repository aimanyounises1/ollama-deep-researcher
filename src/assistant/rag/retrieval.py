# src/assistant/rag/retrieval.py
"""
RAG (Retrieval Augmented Generation) implementation for ollama-deep-researcher.
Retrieves relevant context to ground LLM generations in factual data.
"""

import logging
import os
import ssl
from typing import List, Dict, Any, Optional, Union
import asyncio

from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from sentence_transformers import SentenceTransformer

# Import our custom Ollama embeddings implementation
from src.assistant.rag.ollama_retrieval import OllamaEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGRetriever:
    """Retrieval Augmented Generation component for finding relevant context."""
    
    def __init__(self, vector_store_path: Optional[str] = None, model_name: str = "nomic-embed-text:latest"):
        """
        Initialize the RAG retriever.
        
        Args:
            vector_store_path: Path to store/load the vector database.
            model_name: Name of the embedding model to use.
        """
        self.vector_store_path = vector_store_path or os.environ.get(
            "VECTOR_STORE_PATH", "./vector_store"
        )
        
        # Apply proxy settings if configured
        self._setup_proxy_environment()
        
        # Initialize embeddings with Ollama
        self.model_name = model_name or os.environ.get("EMBEDDING_MODEL", "nomic-embed-text:latest")
        self.ollama_base_url = os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434")
        
        try:
            # Use Ollama embeddings instead of HuggingFace
            self.embeddings = OllamaEmbeddings(
                model_name=self.model_name,
                base_url=self.ollama_base_url
            )
            logger.info(f"Successfully initialized Ollama embeddings with model {self.model_name}")
                
        except Exception as e:
            logger.warning(f"Error initializing Ollama embeddings: {e}")
            self.embeddings = None
            
        self._vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # For multi-query generation
        self.ollama_base_url = os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434")
        self.local_llm_name = os.environ.get("LLM_MODEL", "deepseek-coder:33b-instruct")
    
    def _setup_proxy_environment(self):
        """Set up proxy environment variables if not already set."""
        # Proxy configuration from environment or defaults
        proxy_host = os.environ.get("PROXY_HOST", "10.232.233.70")
        proxy_port = os.environ.get("PROXY_PORT", "8080")
        
        # Set up environment variables if not already set
        if not os.environ.get("HTTP_PROXY"):
            os.environ["HTTP_PROXY"] = f"http://{proxy_host}:{proxy_port}"
        if not os.environ.get("HTTPS_PROXY"):
            os.environ["HTTPS_PROXY"] = f"http://{proxy_host}:{proxy_port}"
        
        logger.info(f"Proxy settings: HTTP_PROXY={os.environ.get('HTTP_PROXY')}, HTTPS_PROXY={os.environ.get('HTTPS_PROXY')}")
    
    def _setup_ssl_environment(self):
        """Set up SSL environment for handling certificate issues."""
        # Disable SSL verification warnings
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except (ImportError, AttributeError):
            pass
        
        # Handle SSL certificate issues for macOS
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Set environment variables to disable SSL verification
        os.environ['CURL_CA_BUNDLE'] = ''
        os.environ['REQUESTS_CA_BUNDLE'] = ''
        os.environ['SSL_CERT_FILE'] = ''
    
    async def retrieve(self, query: str, max_docs: int = 5) -> List[str]:
        """
        Retrieve relevant context based on a query.
        
        Args:
            query: The query to search for.
            max_docs: Maximum number of documents to retrieve.
            
        Returns:
            List of relevant context passages.
        """
        # Simplified implementation for testing
        logger.info(f"RAGRetriever.retrieve called with query: {query}")
        
        # For testing, just return some placeholder content
        placeholder_docs = [
            f"This is a placeholder document for the query: {query}",
            f"Additional context information would be retrieved here.",
            f"The system would normally search for information related to: {query}"
        ]
        
        return placeholder_docs
    
    async def _get_vector_store(self) -> Any:
        """
        Simplified implementation of vector store retrieval.
        
        Returns:
            A basic vector store or None
        """
        logger.info("Using simplified vector store implementation for testing")
        return None
    
    async def add_documents(self, documents: List[Union[str, Dict[str, Any]]]):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents or document dicts to add.
        """
        vector_store = await self._get_vector_store()
        
        # Process documents
        docs_to_add = []
        for doc in documents:
            if isinstance(doc, str):
                # Split text into chunks
                chunks = self.text_splitter.split_text(doc)
                docs_to_add.extend([Document(page_content=chunk) for chunk in chunks])
            elif isinstance(doc, dict):
                # Process dictionary with metadata
                content = doc.get("content", "")
                metadata = {k: v for k, v in doc.items() if k != "content"}
                
                # Split text into chunks
                chunks = self.text_splitter.split_text(content)
                docs_to_add.extend([
                    Document(page_content=chunk, metadata=metadata) 
                    for chunk in chunks
                ])
        
        # Add to vector store
        if docs_to_add:
            vector_store.add_documents(docs_to_add)
            # Save updated vector store
            vector_store.save_local(self.vector_store_path)
            logger.info(f"Added {len(docs_to_add)} documents to vector store")
    
    async def _generate_multiple_queries(self, query: str, num_queries: int = 3) -> List[str]:
        """
        Generate multiple query variations using a local LLM.
        
        Args:
            query: The original query.
            num_queries: Number of query variations to generate.
            
        Returns:
            List of query variations.
        """
        try:
            # Initialize local LLM for query generation
            llm = ChatOllama(
                base_url=self.ollama_base_url,
                model=self.local_llm_name,
                temperature=0.7,  # Higher temperature for diverse queries
            )
            
            # Prompt for generating multiple query variations
            prompt = f"""Generate {num_queries} different search queries that would help find information to answer this question: "{query}"
            
            The queries should:
            1. Approach the question from different angles
            2. Use different keywords and phrasings
            3. Include specific technical terms when appropriate
            4. Be concise and focused
            
            Format your response as a numbered list with ONLY the queries, one per line:
            1. First query
            2. Second query
            3. Third query
            """
            
            # Generate variations
            response = await llm.ainvoke(prompt)
            content = response.content
            
            # Parse response
            lines = content.strip().split('\n')
            queries = []
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('- ')):
                    # Extract the query part (removing any numbering)
                    query_text = line.split('.', 1)[-1].strip() if '.' in line else line.split(' ', 1)[-1].strip()
                    queries.append(query_text)
            
            # Always include the original query
            if query not in queries:
                queries.append(query)
            
            return queries[:num_queries]  # Limit to requested number
            
        except Exception as e:
            logger.error(f"Error generating multiple queries: {e}")
            # Fallback to just the original query
            return [query]


class HybridRetriever:
    """Hybrid retrieval combining semantic search with keyword-based search."""
    
    def __init__(self, semantic_retriever: RAGRetriever):
        """
        Initialize the hybrid retriever.
        
        Args:
            semantic_retriever: The semantic retriever component.
        """
        self.semantic_retriever = semantic_retriever
        # Other retrievers could be added here (BM25, etc.)
    
    async def retrieve(self, query: str, max_docs: int = 8) -> List[str]:
        """
        Perform hybrid retrieval using multiple methods.
        
        Args:
            query: The search query.
            max_docs: Maximum number of documents to retrieve.
            
        Returns:
            List of relevant document contents.
        """
        # Get semantic search results
        semantic_results = await self.semantic_retriever.retrieve(query, max_docs=max_docs)
        
        # Here you could add additional retrieval methods
        # For example, keyword-based search using BM25
        # keyword_results = keyword_retriever.retrieve(query, max_docs)
        
        # Combine and deduplicate results
        all_results = set(semantic_results)
        
        # Return combined results
        return list(all_results)