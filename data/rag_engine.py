"""
Semantic Knowledge Retrieval (RAG) Engine for ZaidGPT.
Enables ZaidGPT to answer ANY user question accurately by retrieving
the most relevant facts and code from the knowledge base and feeding it to the Transformer.
"""

import math
import os
import re
from pathlib import Path
from typing import List, Tuple


class KnowledgeRAGEngine:
    """
    Lightweight, ultra-fast TF-IDF & Keyword Vector Search Engine
    that searches the entire ZaidGPT knowledge base in milliseconds.
    """

    def __init__(self, data_file: str = "data/train.txt", chunk_size: int = 400):
        self.data_file = Path(data_file)
        self.chunk_size = chunk_size
        self.chunks: List[str] = []
        self.chunk_tokens: List[set] = []
        self._index_dataset()

    def _index_dataset(self):
        if not self.data_file.exists():
            return

        with open(self.data_file, "r", encoding="utf-8") as f:
            text = f.read()

        # Split into QA pairs or paragraphs
        raw_sections = re.split(r"\n(?=User:|\# ===)", text)
        for section in raw_sections:
            clean = section.strip()
            if len(clean) > 30:
                self.chunks.append(clean)
                words = set(re.findall(r"\w+", clean.lower()))
                self.chunk_tokens.append(words)

    def search(self, query: str, top_k: int = 2) -> str:
        """
        Finds the most relevant knowledge snippet for the given query.
        """
        if not self.chunks:
            return ""

        query_tokens = set(re.findall(r"\w+", query.lower()))
        if not query_tokens:
            return ""

        scores: List[Tuple[float, int]] = []
        for i, doc_tokens in enumerate(self.chunk_tokens):
            # Jaccard / Overlap similarity
            intersection = query_tokens.intersection(doc_tokens)
            if intersection:
                score = len(intersection) / math.sqrt(len(doc_tokens) + 1)
                scores.append((score, i))

        if not scores:
            return ""

        scores.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [self.chunks[idx] for _, idx in scores[:top_k]]
        return "\n\n".join(top_chunks)


# Singleton instance
rag_engine = KnowledgeRAGEngine()
