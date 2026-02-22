# src/assistant/rag/hybrid_retriever.py
"""
HybridRetriever: true hybrid search combining dense (semantic) and sparse (BM25)
retrieval, plus LLM-powered query-expansion techniques.

Dense path  : FAISS vector store via OllamaEmbeddings (semantic similarity)
Sparse path : BM25 keyword index (rank_bm25)
Fusion      : Reciprocal Rank Fusion (RRF) across all result lists
Extras      : Multi-query expansion, HyDE, Step-back prompting, Recursive retrieval
"""

import logging
import os
import asyncio
import math
import re
from typing import List, Dict, Any, Optional, Union, Set, Tuple
from collections import defaultdict

from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.assistant.rag.ollama_retrieval import RAGRetriever

# --------------------------------------------------------------------------- #
#  Optional BM25 import – gracefully degrade if rank_bm25 is not installed    #
# --------------------------------------------------------------------------- #
try:
        from rank_bm25 import BM25Okapi
        _BM25_AVAILABLE = True
except ImportError:  # pragma: no cover
        _BM25_AVAILABLE = False
        BM25Okapi = None  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================================================================== #
#  BM25 sparse index                                                           #
# =========================================================================== #

def _tokenize(text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer for BM25."""
        return re.sub(r"[^\w\s]", " ", text.lower()).split()


class BM25Index:
        """
            Lightweight wrapper around rank_bm25.BM25Okapi.

                Documents are added incrementally; the index is (re)built lazily on the
                    first call to ``search``.
                        """

    def __init__(self) -> None:
                self._corpus: List[str] = []
                self._tokenized: List[List[str]] = []
                self._index: Optional["BM25Okapi"] = None  # type: ignore[name-defined]
        self._dirty: bool = False

    # ---------------------------------------------------------------------- #

    def add_documents(self, documents: List[str]) -> None:
                for doc in documents:
                                self._corpus.append(doc)
                                self._tokenized.append(_tokenize(doc))
                            self._dirty = True

    def _rebuild(self) -> None:
                if not _BM25_AVAILABLE:
                                logger.warning(
                                                    "rank_bm25 is not installed – BM25 search is disabled. "
                                                    "Install it with: pip install rank-bm25"
                                )
                                return
                            if self._tokenized:
                                            self._index = BM25Okapi(self._tokenized)
                                        self._dirty = False

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
                """
                        Return up to *top_k* (document, score) pairs ranked by BM25.
                                Returns an empty list when no documents have been indexed or
                                        rank_bm25 is unavailable.
                                                """
        if not self._corpus:
                        return []

        if self._dirty or self._index is None:
                        self._rebuild()

        if self._index is None:          # rank_bm25 unavailable
                        return []

        tokens = _tokenize(query)
        scores = self._index.get_scores(tokens)

        ranked = sorted(
                        zip(self._corpus, scores),
                        key=lambda x: x[1],
                        reverse=True,
        )
        return [(doc, float(score)) for doc, score in ranked[:top_k] if score > 0]

    @property
    def size(self) -> int:
                return len(self._corpus)


# =========================================================================== #
#  HybridRetriever                                                             #
# =========================================================================== #

class HybridRetriever:
        """
            Advanced hybrid retrieval system combining:

                1. **Dense retrieval**  – FAISS vector search via ``RAGRetriever``
                    2. **Sparse retrieval** – BM25 keyword search via ``BM25Index``
                        3. **Multi-query expansion** – LLM generates query variants
                            4. **HyDE** – LLM generates a hypothetical answer, used as a dense query
                                5. **Step-back prompting** – LLM generates a broader context query
                                    6. **Recursive retrieval** – expands results by drilling into key concepts
                                        7. **RAG-Fusion (RRF)** – merges all ranked lists via Reciprocal Rank Fusion
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
                enable_bm25: bool = True,
                sparse_weight: float = 0.3,
                max_recursive_depth: int = 2,
                relevance_threshold: float = 0.7,
    ) -> None:
                self.semantic_retriever = semantic_retriever

        # LLM for query-expansion techniques
        if llm:
                        self.llm = llm
else:
            self.ollama_base_url = ollama_base_url or os.environ.get(
                                "OLLAMA_ENDPOINT", "http://localhost:11434"
            )
            self.local_llm_name = local_llm_name or os.environ.get(
                                "LLM_MODEL", "qwen3:30b-a3b"
            )
            self.llm = ChatOllama(
                                base_url=self.ollama_base_url,
                                model=self.local_llm_name,
                                temperature=0.7,
            )

        # Feature flags
        self.enable_multi_query = enable_multi_query
        self.enable_recursive = enable_recursive
        self.enable_hyde = enable_hyde
        self.enable_rag_fusion = enable_rag_fusion
        self.enable_step_back = enable_step_back
        self.enable_bm25 = enable_bm25 and _BM25_AVAILABLE

        # Weight given to BM25 scores when pre-normalising before RRF
        # (0 = pure dense, 1 = pure sparse; RRF handles the actual merging)
        self.sparse_weight = sparse_weight
        self.max_recursive_depth = max_recursive_depth
        self.relevance_threshold = relevance_threshold

        # Sparse index (populated lazily from documents added via add_documents)
        self._bm25: BM25Index = BM25Index()

        # Document relationship tracking
        self.document_relationships: Dict[str, Set[str]] = defaultdict(set)
        self.entity_mentions: Dict[str, Set[str]] = defaultdict(set)

        logger.info(
                        "Initialized HybridRetriever  "
                        f"bm25={self.enable_bm25}  "
                        f"multi_query={enable_multi_query}  "
                        f"hyde={enable_hyde}  "
                        f"rag_fusion={enable_rag_fusion}  "
                        f"step_back={enable_step_back}  "
                        f"recursive={enable_recursive}"
        )

    # =================================================================== #
    #  Public API                                                           #
    # =================================================================== #

    async def retrieve(
                self,
                query: str,
                max_docs: int = 8,
                include_parent_context: bool = True,
    ) -> List[str]:
                """
                        Perform hybrid retrieval.

                                Pipeline:
                                          step-back query  →  dense retrieval
                                                    HyDE             →  dense retrieval
                                                              multi-query      →  dense + sparse retrieval  →  RRF fusion
                                                                        recursive        →  dense retrieval on derived concept queries
                                                                                  parent context   →  stub (extend as needed)
                                                                                          """
        logger.info(f"HybridRetriever.retrieve: '{query}'")

        # ------------------------------------------------------------------ #
        #  1. Step-back: broad context docs                                   #
        # ------------------------------------------------------------------ #
        step_back_docs: List[str] = []
        if self.enable_step_back:
                        sb_query = await self._generate_step_back_query(query)
                        if sb_query:
                                            logger.info(f"Step-back query: '{sb_query}'")
                                            step_back_docs = await self.semantic_retriever.retrieve(sb_query, max_docs=3)

                    # ------------------------------------------------------------------ #
                    #  2. HyDE: hypothetical document → dense retrieval                  #
        # ------------------------------------------------------------------ #
        hyde_docs: List[str] = []
        if self.enable_hyde:
                        hyde_docs = await self._perform_hyde_retrieval(query, max_docs=max(2, max_docs // 2))

        # ------------------------------------------------------------------ #
        #  3. Query expansion                                                 #
        # ------------------------------------------------------------------ #
        queries = [query]
        if self.enable_multi_query:
                        expanded = await self._generate_multiple_queries(query)
                        queries.extend(expanded)
                        logger.info(f"Expanded to {len(queries)} queries")

        # ------------------------------------------------------------------ #
        #  4. Dense + sparse retrieval for every query                        #
        # ------------------------------------------------------------------ #
        all_docs: Set[str] = set()
        all_docs_with_scores: List[Dict[str, Any]] = []
        doc_query_scores: List[Tuple[str, str, float]] = []   # for RRF

        for i, q in enumerate(queries):
                        # -- dense --
                        dense_docs = await self.semantic_retriever.retrieve(q, max_docs=max_docs)
                        for rank, doc in enumerate(dense_docs, start=1):
                                            all_docs.add(doc)
                                            score = 1.0 / rank
                                            doc_query_scores.append((doc, q, score))
                                            all_docs_with_scores.append(
                                                {"content": doc, "query": q, "score": 0.9 - 0.1 * i}
                                            )

                        # -- sparse (BM25) --
                        if self.enable_bm25 and self._bm25.size > 0:
                                            sparse_hits = self._bm25.search(q, top_k=max_docs)
                                            for rank, (doc, _bm25_score) in enumerate(sparse_hits, start=1):
                                                                    all_docs.add(doc)
                                                                    # Use a slightly discounted score so sparse complements dense
                                                                    score = self.sparse_weight / rank
                                                                    doc_query_scores.append((doc, q + "__sparse__", score))
                                                                    all_docs_with_scores.append(
                                                                        {"content": doc, "query": q, "score": self.sparse_weight * (0.9 - 0.1 * i)}
                                                                    )

                                    # ------------------------------------------------------------------ #
                                    #  5. RAG-Fusion (RRF) across all ranked lists                        #
                                    # ------------------------------------------------------------------ #
                                    if self.enable_rag_fusion and len(queries) > 1:
                                                    fused = self._apply_rag_fusion(doc_query_scores, max_docs=max_docs)
                                                    all_docs.update(fused)

        # ------------------------------------------------------------------ #
        #  6. Merge HyDE + step-back results                                  #
        # ------------------------------------------------------------------ #
        all_docs.update(hyde_docs)
        all_docs.update(step_back_docs)

        # ------------------------------------------------------------------ #
        #  7. Recursive retrieval                                             #
        # ------------------------------------------------------------------ #
        if self.enable_recursive and all_docs_with_scores:
                        recursive_docs = await self._perform_recursive_retrieval(
                                            all_docs_with_scores, original_query=query, depth=1
                        )
                        all_docs.update(recursive_docs)

        # ------------------------------------------------------------------ #
        #  8. Parent context enrichment                                       #
        # ------------------------------------------------------------------ #
        if include_parent_context:
                        all_docs.update(self._get_parent_contexts(all_docs))

        logger.info(f"HybridRetriever retrieved {len(all_docs)} unique documents")
        return list(all_docs)

    # =================================================================== #
    #  Document management                                                  #
    # =================================================================== #

    async def add_documents(self, documents: List[Union[str, Dict[str, Any]]]) -> None:
                """Add documents to both the dense vector store and BM25 index."""
        texts: List[str] = []
        for doc in documents:
                        if isinstance(doc, str):
                                            texts.append(doc)
elif isinstance(doc, dict):
                texts.append(doc.get("content", ""))

        # Update BM25 sparse index
        if self.enable_bm25 and texts:
                        self._bm25.add_documents(texts)
                        logger.info(f"BM25 index now contains {self._bm25.size} documents")

        # Update dense vector store
        await self.semantic_retriever.add_documents(documents)

    def add_document_relationship(self, parent_id: str, child_id: str) -> None:
                self.document_relationships[parent_id].add(child_id)

    def add_entity_mention(self, entity: str, document_id: str) -> None:
                self.entity_mentions[entity].add(document_id)

    # =================================================================== #
    #  Private helpers                                                      #
    # =================================================================== #

    # ------------------------------------------------------------------ #
    #  RRF fusion                                                          #
    # ------------------------------------------------------------------ #

    def _apply_rag_fusion(
                self,
                doc_query_scores: List[Tuple[str, str, float]],
                max_docs: int = 8,
                k: float = 60.0,
    ) -> List[str]:
                """
                        Reciprocal Rank Fusion over all (doc, query, rank_score) triples.

                                score_rrf(doc) = Σ_q  1 / (k + 1/rank_score_q)
                                        """
        try:
                        doc_scores: Dict[str, List[float]] = defaultdict(list)
                        for doc, _query, score in doc_query_scores:
                                            doc_scores[doc].append(score)

                        fusion: Dict[str, float] = {
                            doc: sum(1.0 / (k + (1.0 / s)) for s in scores)
                            for doc, scores in doc_scores.items()
                        }
                        ranked = sorted(fusion.items(), key=lambda x: x[1], reverse=True)
                        result = [doc for doc, _ in ranked[:max_docs]]
                        logger.info(f"RRF fusion: {len(result)} documents")
                        return result
except Exception as exc:
            logger.error(f"RRF fusion error: {exc}")
            return []

    # ------------------------------------------------------------------ #
    #  HyDE                                                                #
    # ------------------------------------------------------------------ #

    async def _perform_hyde_retrieval(self, query: str, max_docs: int = 4) -> List[str]:
                """
                        Hypothetical Document Embeddings (HyDE):
                                Generate a synthetic ideal-answer passage, embed it, and use that
                                        embedding to retrieve real documents from the dense index.
                                                """
        try:
                        hyde_prompt = ChatPromptTemplate.from_template(
                                            "Generate a detailed passage (2-3 paragraphs) that would be a perfect answer "
                                            "to the following question. Write as if from an authoritative technical document "
                                            "— do NOT say 'the answer is', just write the passage directly.\n\n"
                                            "Question: {query}"
                        )
                        chain = hyde_prompt | self.llm | StrOutputParser()
                        hypothetical_doc = await chain.ainvoke({"query": query})
                        docs = await self.semantic_retriever.retrieve(hypothetical_doc, max_docs=max_docs)
                        logger.info(f"HyDE retrieved {len(docs)} documents")
                        return docs
except Exception as exc:
            logger.error(f"HyDE error: {exc}")
            return []

    # ------------------------------------------------------------------ #
    #  Multi-query expansion                                               #
    # ------------------------------------------------------------------ #

    async def _generate_multiple_queries(
                self, query: str, num_queries: int = 3
    ) -> List[str]:
                """Generate *num_queries* diverse paraphrases of the original query."""
        try:
                        prompt = ChatPromptTemplate.from_template(
                                            f"Generate {num_queries} different search queries that approach the following "
                                            f"question from different angles, using varied keywords and phrasings.\n\n"
                                            f"Question: {{query}}\n\n"
                                            f"Format: numbered list, one query per line, no extra text.\n"
                                            f"1.\n2.\n3."
                        )
                        chain = prompt | self.llm | StrOutputParser()
                        response = await chain.ainvoke({"query": query})
                        queries: List[str] = []
                        for line in response.strip().splitlines():
                                            line = line.strip()
                                            if not line:
                                                                    continue
                                                                # Strip leading "1." / "- " etc.
                                                                cleaned = re.sub(r"^[\d]+[.)]\s*|^[-*]\s*", "", line).strip()
                                            if cleaned and cleaned != query:
                                                                    queries.append(cleaned)
                                                            return queries[:num_queries]
        except Exception as exc:
            logger.error(f"Multi-query expansion error: {exc}")
            return []

    # ------------------------------------------------------------------ #
    #  Step-back prompting                                                 #
    # ------------------------------------------------------------------ #

    async def _generate_step_back_query(self, query: str) -> str:
                """Return a broader, higher-level version of *query* for context priming."""
        try:
                        prompt = ChatPromptTemplate.from_template(
                                            "For the following specific question, write a broader 'step-back' question "
                                            "that asks about the underlying concept or foundational topic.\n\n"
                                            "Specific question: {query}\n\n"
                                            "Respond with ONLY the step-back question, nothing else."
                        )
                        chain = prompt | self.llm | StrOutputParser()
                        result = await chain.ainvoke({"query": query})
                        return result.strip().strip('"')
except Exception as exc:
            logger.error(f"Step-back query error: {exc}")
            return ""

    # ------------------------------------------------------------------ #
    #  Recursive retrieval                                                 #
    # ------------------------------------------------------------------ #

    async def _perform_recursive_retrieval(
                self,
                base_docs_with_scores: List[Dict[str, Any]],
                original_query: str,
                depth: int = 1,
    ) -> List[str]:
                """Recursively expand retrieval by drilling into key concepts."""
        if depth > self.max_recursive_depth or not base_docs_with_scores:
                        return []
                    try:
                                    concepts = await self._extract_key_concepts(
                                                        base_docs_with_scores, original_query
                                    )
                                    recursive_docs: Set[str] = set()
                                    for concept in concepts[:3]:
                                                        concept_query = f"{original_query} related to {concept}"
                                                        docs = await self.semantic_retriever.retrieve(concept_query, max_docs=3)
                                                        # Also search BM25 for the concept query
                                                        if self.enable_bm25 and self._bm25.size > 0:
                                                                                for doc, _ in self._bm25.search(concept_query, top_k=3):
                                                                                                            docs.append(doc)
                                                                                                    recursive_docs.update(docs)

                                                    if depth < self.max_recursive_depth and recursive_docs:
                                                                        next_level = [
                                                                                                {"content": d, "query": original_query, "score": 0.8}
                                                                                                for d in recursive_docs
                                                                        ]
                                                                        deeper = await self._perform_recursive_retrieval(
                                                                            next_level, original_query, depth + 1
                                                                        )
                                                                        recursive_docs.update(deeper)

            return list(recursive_docs)
except Exception as exc:
            logger.error(f"Recursive retrieval error at depth {depth}: {exc}")
            return []

    async def _extract_key_concepts(
                self,
                docs_with_scores: List[Dict[str, Any]],
                original_query: str,
    ) -> List[str]:
                """Extract 3-5 key concepts from the top-ranked retrieved documents."""
        try:
                        top_docs = sorted(
                                            docs_with_scores, key=lambda x: x.get("score", 0), reverse=True
                        )[:3]
            combined = "\n\n".join(d.get("content", "") for d in top_docs)
            prompt = ChatPromptTemplate.from_template(
                                f'From the documents below (related to "{original_query}"), '
                                f"list 3-5 key concepts worth exploring further.\n\n"
                                f"Documents:\n{{combined}}\n\n"
                                f"Format: bullet list, one concept per line, no extra text."
            )
            chain = prompt | self.llm | StrOutputParser()
            response = await chain.ainvoke({"combined": combined})
            concepts: List[str] = []
            for line in response.strip().splitlines():
                                cleaned = re.sub(r"^[-*•]\s*", "", line).strip()
                if cleaned:
                                        concepts.append(cleaned)
                                return concepts
except Exception as exc:
            logger.error(f"Key concept extraction error: {exc}")
            return []

    # ------------------------------------------------------------------ #
    #  Parent context (stub – extend with your doc-store implementation)  #
    # ------------------------------------------------------------------ #

    def _get_parent_contexts(self, docs: Set[str]) -> List[str]:
                """
                        Return parent document contents for any child chunks in *docs*.
                                Extend this method to connect to your document storage system.
                                        """
        return []
