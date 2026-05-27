from __future__ import annotations

from collections import defaultdict

try:
    from Project.common import RetrievedDocument
except ImportError:
    from common import RetrievedDocument


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedDocument]],
    k: int = 60,
    top_k: int = 8,
) -> list[RetrievedDocument]:
    if k < 1:
        raise ValueError("RRF constant k must be at least 1.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    fused_scores: dict[str, float] = defaultdict(float)
    representative_docs: dict[str, RetrievedDocument] = {}
    trace: dict[str, set[str]] = defaultdict(set)

    for result_list in ranked_lists:
        for rank, item in enumerate(result_list, start=1):
            fused_scores[item.doc.doc_id] += 1.0 / (k + rank)
            representative_docs[item.doc.doc_id] = item
            trace[item.doc.doc_id].add(item.retriever)

    ranked_doc_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]
    fused_results: list[RetrievedDocument] = []
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        item = representative_docs[doc_id]
        fused_results.append(
            RetrievedDocument(
                doc=item.doc,
                score=float(fused_scores[doc_id]),
                retriever=" + ".join(sorted(trace[doc_id])),
                rank=rank,
            )
        )
    return fused_results
