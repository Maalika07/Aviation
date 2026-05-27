from __future__ import annotations

from collections import Counter
from math import log
import os
import sys
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

try:
    from Project.common import KnowledgeDocument, RetrievedDocument, tokenize
except ImportError:
    from common import KnowledgeDocument, RetrievedDocument, tokenize


DOMAIN_SYNONYMS = {
    "delay": ["delay", "turnaround", "hold", "propagation"],
    "gate": ["gate", "stand", "terminal"],
    "congestion": ["congestion", "queue", "bottleneck"],
    "fuel": ["fuel", "refuel", "fueling", "truck"],
    "baggage": ["baggage", "bag", "vehicle", "loader"],
    "weather": ["weather", "rain", "fog", "thunderstorm", "visibility", "wind"],
    "staff": ["staff", "crew", "ramp", "operator"],
}


def expand_query(query: str) -> str:
    tokens = set(tokenize(query))
    expanded = [query]
    for canonical, variants in DOMAIN_SYNONYMS.items():
        if tokens.intersection(variants):
            expanded.extend(variants)
            expanded.append(canonical)
    return " ".join(expanded)


def _running_in_colab() -> bool:
    return "google.colab" in sys.modules


def _dense_retriever_requested() -> bool:
    if os.getenv("ENABLE_LOCAL_DENSE_RETRIEVER") == "1":
        return True
    return _running_in_colab() and os.getenv("COLAB_USE_DENSE_RETRIEVER", "1") == "1"


def _exact_match_bonus(doc: KnowledgeDocument, query_tokens: set[str]) -> float:
    doc_tokens = set(tokenize(doc.combined_text()))
    overlap = query_tokens.intersection(doc_tokens)
    if not overlap:
        return 0.0

    bonus = 0.0
    for token in overlap:
        if any(char.isdigit() for char in token) and any(char.isalpha() for char in token):
            bonus += 1.5
        elif "-" in token or "/" in token:
            bonus += 1.25
        elif token in {"fog", "rain", "thunderstorm", "weather", "gate", "runway", "fuel", "baggage", "staff"}:
            bonus += 0.35
    return bonus


class VectorRetriever:
    def __init__(self, docs: Iterable[KnowledgeDocument]) -> None:
        self.docs = list(docs)
        if not self.docs:
            raise ValueError("VectorRetriever requires at least one knowledge document.")
        self.backend = "hashing"
        self.backend_details = "sparse hashing vectorizer"
        self.texts = [doc.combined_text() for doc in self.docs]
        self.vectorizer = None
        self.doc_matrix = None
        self.model = None
        self.doc_embeddings = None
        self._build_index()

    def _build_index(self) -> None:
        if _dense_retriever_requested():
            try:
                from sentence_transformers import SentenceTransformer
                import torch

                model_name = os.getenv("DENSE_RETRIEVER_MODEL", "all-MiniLM-L6-v2")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                allow_download = os.getenv(
                    "ALLOW_EMBEDDING_DOWNLOAD",
                    "1" if _running_in_colab() else "0",
                ) == "1"
                self.model = SentenceTransformer(
                    model_name,
                    device=device,
                    local_files_only=not allow_download,
                )
                self.doc_embeddings = self.model.encode(
                    self.texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                self.backend = f"sentence_transformer:{device}"
                self.backend_details = model_name
                return
            except Exception:
                self.model = None
                self.doc_embeddings = None

        self.vectorizer = HashingVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            n_features=4096,
            alternate_sign=False,
            norm="l2",
        )
        self.doc_matrix = self.vectorizer.transform(self.texts)

    def search(self, query: str, top_k: int = 6) -> list[RetrievedDocument]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query = expand_query(query)
        query_tokens = set(tokenize(query))
        if self.backend.startswith("sentence_transformer") and self.model is not None and self.doc_embeddings is not None:
            query_vec = self.model.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )[0]
            scores = np.dot(self.doc_embeddings, query_vec)
        else:
            query_vec = self.vectorizer.transform([query])
            scores = (self.doc_matrix @ query_vec.T).toarray().ravel()

        ranked_idx = np.argsort(scores)[::-1][:top_k]
        results: list[RetrievedDocument] = []
        for rank, idx in enumerate(ranked_idx, start=1):
            score = float(scores[idx]) + _exact_match_bonus(self.docs[int(idx)], query_tokens)
            if score <= 0:
                continue
            results.append(
                RetrievedDocument(
                    doc=self.docs[int(idx)],
                    score=score,
                    retriever=f"vector:{self.backend}",
                    rank=rank,
                )
            )
        return results


class KeywordRetriever:
    def __init__(self, docs: Iterable[KnowledgeDocument]) -> None:
        self.docs = list(docs)
        if not self.docs:
            raise ValueError("KeywordRetriever requires at least one knowledge document.")
        self.doc_tokens = [tokenize(doc.combined_text()) for doc in self.docs]
        self.doc_counters = [Counter(tokens) for tokens in self.doc_tokens]
        self.idf = self._compute_idf(self.doc_tokens)

    def _compute_idf(self, documents: list[list[str]]) -> dict[str, float]:
        df_counter: Counter[str] = Counter()
        for tokens in documents:
            df_counter.update(set(tokens))
        total_docs = len(documents)
        return {token: log((1 + total_docs) / (1 + count)) + 1 for token, count in df_counter.items()}

    def search(self, query: str, top_k: int = 6) -> list[RetrievedDocument]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        expanded = tokenize(expand_query(query))
        query_tokens = set(expanded)
        scores = []
        for idx, counter in enumerate(self.doc_counters):
            score = 0.0
            for token in expanded:
                tf = counter.get(token, 0)
                if tf:
                    score += (1 + log(tf)) * self.idf.get(token, 0.0)
            score += _exact_match_bonus(self.docs[idx], query_tokens)
            if score > 0:
                scores.append((idx, score))
        scores.sort(key=lambda item: item[1], reverse=True)

        results: list[RetrievedDocument] = []
        for rank, (idx, score) in enumerate(scores[:top_k], start=1):
            results.append(
                RetrievedDocument(
                    doc=self.docs[idx],
                    score=float(score),
                    retriever="keyword",
                    rank=rank,
                )
            )
        return results
