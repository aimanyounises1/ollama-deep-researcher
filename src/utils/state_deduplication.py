"""Enhanced state deduplication tools with semantic similarity.
"""
import json
import logging
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Try to import sentence-transformers
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    SEMANTIC_DEDUP_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not installed. Install with 'pip install sentence-transformers' for semantic deduplication.")
    SEMANTIC_DEDUP_AVAILABLE = False

# Global model for embedding generation - load once
_embedding_model = None

def get_embedding_model():
    """Get or initialize the embedding model for semantic deduplication."""
    global _embedding_model, SEMANTIC_DEDUP_AVAILABLE
    if not SEMANTIC_DEDUP_AVAILABLE:
        return None
        
    if _embedding_model is None:
        try:
            # Use a small, fast model
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded embedding model for semantic deduplication")
        except Exception as e:
            logger.warning(f"Could not load embedding model: {e}")
            SEMANTIC_DEDUP_AVAILABLE = False
    return _embedding_model

def _hashable(item: Any):
    """Return a hashable representation of *item* for deduplication purposes."""
    # Normalize strings/bytes by collapsing whitespace so formatting differences don't break deduplication
    if isinstance(item, (str, bytes)):
        try:
            text = item.decode() if isinstance(item, bytes) else item
            # Collapse all consecutive whitespace (spaces, tabs, newlines) into a single space and strip
            normalized = re.sub(r"\s+", " ", text).strip()
            return normalized
        except Exception:
            # Fallback to original value if decoding fails
            return item

    if isinstance(item, (int, float, bool, type(None))):
        return item

    try:
        if isinstance(item, dict):
            return json.dumps(item, sort_keys=True, default=str)
        if isinstance(item, (list, tuple, set)):
            return tuple(_hashable(x) for x in item)
    except Exception:
        pass
    return repr(item)

def _lexical_dedup_list(seq):
    """Deduplicate *seq* while preserving order using lexical comparison."""
    seen = set()
    deduped = []
    for item in seq:
        marker = _hashable(item)
        if marker not in seen:
            deduped.append(item)
            seen.add(marker)
    return deduped

def _dedup_nested(seq):
    """Recursively deduplicate *seq* (which may contain nested lists)."""
    if not isinstance(seq, list):
        return seq  # Non‑lists are returned as‑is

    seen: set = set()
    deduped: list = []

    for item in seq:
        # First, ensure inner lists are de‑duplicated before hashing
        if isinstance(item, list):
            item = _dedup_nested(item)

        marker = _hashable(item)
        if marker not in seen:
            seen.add(marker)
            deduped.append(item)
    return deduped

def semantic_dedup_list(items, threshold=0.85):
    """Deduplicate list using semantic similarity."""
    if not items or len(items) <= 1:
        return items
    
    if not SEMANTIC_DEDUP_AVAILABLE:
        # Fall back to lexical dedup if semantic not available
        return _lexical_dedup_list(items)
    
    model = get_embedding_model()
    if model is None:
        # Fall back to lexical dedup if model isn't available
        return _lexical_dedup_list(items)
    
    # Generate embeddings for all items
    texts = [str(item) for item in items]
    try:
        embeddings = model.encode(texts)
        
        # Calculate similarity matrix
        similarity_matrix = np.inner(embeddings, embeddings)
        
        # Select unique items
        unique_indices = []
        for i in range(len(items)):
            # Check if this item is too similar to any already-selected item
            if not any(similarity_matrix[i][j] > threshold for j in unique_indices):
                unique_indices.append(i)
        
        # Return unique items in original order
        logger.info(f"Semantic deduplication reduced items from {len(items)} to {len(unique_indices)}")
        return [items[i] for i in unique_indices]
    except Exception as e:
        logger.warning(f"Semantic deduplication failed: {e}, falling back to lexical dedup")
        return _lexical_dedup_list(items)

# ---------------------------------------------------------------------------
#  universal, size-bounded de-duplication
# ---------------------------------------------------------------------------
_NORMALISE = re.compile(r"\s+")

def deduplicate_state(state: Dict[str, Any],
                      *,
                      max_blocks: int = 250,
                      max_jira:   int = 200) -> Dict[str, Any]:
    """In-place de-duplication **and** pruning of all heavy list fields that
    live in the LangGraph state.

    ▸ identical items (after whitespace normalisation) are removed
    ▸ only the newest `max_blocks` items are kept in
        •   enterprise_research_results
    ▸ `filtered_jira_results` is deduped by `"key"` and capped at `max_jira`
    ▸ Ensures sources_gathered references enterprise_research_results directly instead of duplicating
    ▸ Maintains content hash tracking for efficient deduplication
    """
    # Initialize content hash tracking if not present
    if "_content_hashes" not in state:
        state["_content_hashes"] = set()
        # Populate with hashes of existing content
        for item in state.get("enterprise_research_results", []):
            state["_content_hashes"].add(hash(str(item)))

    # Deduplicate enterprise_research_results with content hashing
    if "enterprise_research_results" in state and isinstance(state["enterprise_research_results"], list):
        seen_hashes = set()
        deduplicated = []
        
        for item in state["enterprise_research_results"]:
            item_hash = hash(str(item))
            if item_hash not in seen_hashes:
                seen_hashes.add(item_hash)
                deduplicated.append(item)
        
        removed_count = len(state["enterprise_research_results"]) - len(deduplicated)
        if removed_count > 0:
            logger.info(f"Deduplicated enterprise_research_results: removed {removed_count} duplicates")
        
        state["enterprise_research_results"] = deduplicated
        
        # Update global content hash tracking
        state["_content_hashes"].update(seen_hashes)
    
    # Keep sources_gathered as a direct reference to enterprise_research_results
    if "enterprise_research_results" in state:
        state["sources_gathered"] = state["enterprise_research_results"]
        logger.debug("sources_gathered now references enterprise_research_results directly")

    # 1 — run the old lexical / semantic passes first for other fields
    list_fields = {
        # "enterprise_research_results",  # Already handled above
        # "sources_gathered",             # Directly references enterprise_research_results
        "web_search_results",
        "alternative_terms",
        "citations",
    }
    for key in list_fields:
        if key in state and isinstance(state[key], list):
            cleaned = _lexical_dedup_list(state[key])
            cleaned = semantic_dedup_list(cleaned)
            cleaned = _dedup_nested(cleaned)
            state[key] = cleaned

    # fallback for every other list-typed field
    for key, value in list(state.items()):
        if isinstance(value, list) and key not in list_fields and key != "enterprise_research_results" and key != "sources_gathered":
            state[key] = _lexical_dedup_list(value)

    # 2 — *now* apply hard caps
    def _cap(field: str, cap: int) -> None:
        if field in state and isinstance(state[field], list):
            if len(state[field]) > cap:
                state[field] = state[field][-cap:]          # newest N

    # Apply size caps to enterprise_research_results
    if "enterprise_research_results" in state and len(state["enterprise_research_results"]) > max_blocks:
        old_length = len(state["enterprise_research_results"])
        state["enterprise_research_results"] = state["enterprise_research_results"][-max_blocks:]
        logger.info(f"Capped enterprise_research_results from {old_length} to {max_blocks} items")
        
        # Since sources_gathered is now a direct reference, we don't need to update it separately

    # bounded, unique flattened JIRA list
    if "filtered_jira_results" in state and isinstance(state["filtered_jira_results"], list):
        seen, uniq = set(), []
        for item in reversed(state["filtered_jira_results"]):   # newest-first
            key = isinstance(item, dict) and item.get("key")
            if key and key not in seen:
                seen.add(key)
                uniq.append(item)
            if len(uniq) >= max_jira:
                break
        state["filtered_jira_results"] = list(reversed(uniq))

    return state
