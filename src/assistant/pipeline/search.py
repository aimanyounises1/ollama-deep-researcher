"""Search functionality for the research package.
"""

import asyncio
import logging
from typing import Any, Dict, List

# Import with fallbacks
try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None
    
try:
    from langchain.docstore.document import Document
except ImportError:
    # Create a simple Document class if it's not available
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

# Use try-except for potentially problematic imports
vectorstore = None
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.retrievers import ParentDocumentRetriever
    from langchain_community.storage import InMemoryStore
    from langchain_community.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    
    # Initialize these in a try block to handle potential errors
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(embedding_function=embeddings)
        docstore = InMemoryStore()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    except Exception as e:
        logging.warning(f"Could not initialize vector search components: {e}")
except ImportError as e:
    logging.warning(f"Could not import search dependencies: {e}")

from src.assistant.pipeline.tools import tool_manager

logger = logging.getLogger(__name__)

async def search_jira(query: str) -> str:
    """Search JIRA for relevant information."""
    try:
        results = await tool_manager.search_jira(query)
        if not results:
            return "No JIRA results found."
            
        formatted_results = []
        for result in results:
            formatted_results.append(f"Issue: {result.get('key', 'N/A')}")
            formatted_results.append(f"Title: {result.get('title', 'N/A')}")
            formatted_results.append(f"Status: {result.get('status', 'N/A')}")
            formatted_results.append(f"Description: {result.get('description', 'N/A')}")
            formatted_results.append("-" * 50)
            
        return "\n".join(formatted_results)
    except Exception as e:
        logger.error(f"Error searching JIRA: {e}")
        return f"Error searching JIRA: {str(e)}"

async def search_perforce(query: str) -> str:
    """Search Perforce for relevant information."""
    try:
        results = await tool_manager.search_perforce(query)
        if not results:
            return "No Perforce results found."
            
        formatted_results = []
        for result in results:
            formatted_results.append(f"File: {result.get('file', 'N/A')}")
            formatted_results.append(f"Revision: {result.get('revision', 'N/A')}")
            formatted_results.append(f"Change: {result.get('change', 'N/A')}")
            formatted_results.append(f"Content: {result.get('content', 'N/A')}")
            formatted_results.append("-" * 50)
            
        return "\n".join(formatted_results)
    except Exception as e:
        logger.error(f"Error searching Perforce: {e}")
        return f"Error searching Perforce: {str(e)}"

async def search_confluence(query: str) -> str:
    """Search Confluence for relevant information."""
    try:
        results = await tool_manager.search_confluence(query)
        if not results:
            return "No Confluence results found."
            
        formatted_results = []
        for result in results:
            formatted_results.append(f"Title: {result.get('title', 'N/A')}")
            formatted_results.append(f"Space: {result.get('space', 'N/A')}")
            formatted_results.append(f"Content: {result.get('content', 'N/A')}")
            formatted_results.append("-" * 50)
            
        return "\n".join(formatted_results)
    except Exception as e:
        logger.error(f"Error searching Confluence: {e}")
        return f"Error searching Confluence: {str(e)}"

async def search_vectorstore(query: str) -> str:
    """Search vectorstore for relevant information."""
    try:
        # Check if vectorstore is initialized
        if not vectorstore:
            return "Vectorstore is not available."
            
        # Search the vector store
        docs = vectorstore.similarity_search(query, k=3)
        
        formatted_results = []
        for doc in docs:
            formatted_results.append(
                f"Content: {doc.page_content}\n"
                f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            )
        
        return f"=== Vectorstore Results ===\nQuery: {query}\n\n" + "\n".join(formatted_results)
    except Exception as e:
        logger.error(f"Error searching vectorstore: {e}")
        return f"Error searching vectorstore: {str(e)}"

async def search_web(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search the web for relevant information."""
    try:
        # Check if DDGS is available
        if not DDGS:
            logger.warning("DuckDuckGo search not available")
            return [{
                "title": "Web search unavailable",
                "snippet": "The required library for web search is not installed.",
                "url": "",
                "source": "web"
            }]
            
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "title": result.get("title", ""),
                "snippet": result.get("body", ""),
                "url": result.get("link", ""),
                "source": "web"
            })
            
        return formatted_results
    except Exception as e:
        logger.error(f"Error searching web: {e}")
        return [{
            "title": "Error in web search",
            "snippet": f"Error: {str(e)}",
            "url": "",
            "source": "web"
        }]

async def search_all_sources(query: str) -> str:
    """Search all available sources for information."""
    try:
        # Run searches in parallel
        tasks = [
            search_jira(query),
            search_perforce(query),
            search_confluence(query),
            search_vectorstore(query)
        ]
        results = await asyncio.gather(*tasks)
        
        # Combine results
        combined_results = "\n\n".join(results)
        
        return combined_results
    except Exception as e:
        logger.error(f"Error searching all sources: {e}")
        return f"Error searching all sources: {str(e)}"

def add_documents_to_retrieval(documents: List[Document]) -> None:
    """Add documents to the retrieval system."""
    try:
        # Check if vectorstore is available
        if not vectorstore:
            logger.warning("Vectorstore not available, can't add documents")
            return
            
        # Split documents into chunks
        if not text_splitter:
            logger.warning("Text splitter not available, can't add documents")
            return
            
        split_docs = text_splitter.split_documents(documents)
        
        # Add to vector store
        vectorstore.add_documents(split_docs)
        
        # Add to document store
        for doc in documents:
            docstore.mset([(doc.metadata.get("id", str(hash(doc.page_content))), doc)])
            
        logger.info(f"Added {len(documents)} documents to retrieval system")
    except Exception as e:
        logger.error(f"Error adding documents to retrieval: {e}")

def ensure_vectorstore_has_documents() -> None:
    """Ensure the vectorstore has documents for search."""
    try:
        # Check if vectorstore is available
        if not vectorstore:
            logger.warning("Vectorstore not available, can't ensure documents")
            return
            
        # Check if vectorstore is empty
        try:
            if not vectorstore.get():
                # Initialize with some default documents
                default_docs = [
                    Document(
                        page_content="Default documentation",
                        metadata={"source": "default", "id": "default1"}
                    )
                ]
                add_documents_to_retrieval(default_docs)
        except:
            # If get() fails, just add the default documents
            default_docs = [
                Document(
                    page_content="Default documentation",
                    metadata={"source": "default", "id": "default1"}
                )
            ]
            add_documents_to_retrieval(default_docs)
            
        logger.info("Vectorstore initialized with documents")
    except Exception as e:
        logger.error(f"Error initializing vectorstore: {e}")

hybrid_retriever = None

def initialize_hybrid_retriever() -> None:
    """Initialize the hybrid retriever system."""
    global hybrid_retriever
    try:
        # Check if dependencies are available
        if not vectorstore or not docstore or not text_splitter:
            logger.warning("Dependencies not available, can't initialize hybrid retriever")
            return
            
        # Create parent document retriever - try multiple import locations for compatibility
        retriever = None
        try:
            # New package structure
            from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
            retriever = ParentDocumentRetriever(
                vectorstore=vectorstore,
                docstore=docstore,
                child_splitter=text_splitter,
            )
        except ImportError:
            try:
                # Try alternate location
                from langchain_core.retrievers import ParentDocumentRetriever
                retriever = ParentDocumentRetriever(
                    vectorstore=vectorstore,
                    docstore=docstore,
                    child_splitter=text_splitter,
                )
            except ImportError:
                # Fallback to normal RAG if ParentDocumentRetriever isn't available
                logger.warning("ParentDocumentRetriever not available in langchain - using basic retriever")
                retriever = vectorstore.as_retriever()
        
        # Store retriever in global scope
        hybrid_retriever = retriever
        
        logger.info("Hybrid retriever initialized")
    except Exception as e:
        logger.error(f"Error initializing hybrid retriever: {e}")

def create_ensemble_retriever(documents: List[Document]) -> None:
    """Create an ensemble retriever from documents."""
    try:
        # Check if dependencies are available
        if not vectorstore:
            logger.warning("Vectorstore not available, can't create ensemble retriever")
            return
            
        # Add documents to retrieval system
        add_documents_to_retrieval(documents)
        
        # Initialize hybrid retriever if not already done
        if 'hybrid_retriever' not in globals() or hybrid_retriever is None:
            initialize_hybrid_retriever()
            
        logger.info(f"Created ensemble retriever with {len(documents)} documents")
    except Exception as e:
        logger.error(f"Error creating ensemble retriever: {e}")

async def hybrid_retrieval(query: str) -> str:
    """Perform hybrid retrieval using multiple methods."""
    try:
        # Check if hybrid_retriever is available
        if hybrid_retriever is None:
            # Fallback to simple vectorstore search
            vector_results = await search_vectorstore(query)
        else:
            # Use hybrid retriever
            docs = hybrid_retriever.get_relevant_documents(query)
            vector_results = "\n\n".join([doc.page_content for doc in docs])
            
        # Get web results
        web_results = await search_web(query)
        
        # Format web results
        web_formatted = []
        for result in web_results:
            web_formatted.append(
                f"Title: {result['title']}\n"
                f"Snippet: {result['snippet']}\n"
                f"URL: {result['url']}\n"
            )
        
        # Combine results
        combined_results = f"""=== Hybrid Retrieval Results ===
Query: {query}

Vector Store Results:
{vector_results}

Web Results:
{chr(10).join(web_formatted)}"""
        
        return combined_results
    except Exception as e:
        logger.error(f"Error performing hybrid retrieval: {e}")
        return f"Error performing hybrid retrieval: {str(e)}" 
