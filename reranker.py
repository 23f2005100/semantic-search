# reranker.py

import cohere
import os

co = cohere.Client(os.getenv("CO_API_KEY"))

def rerank_results(query, documents, top_k=5):
    docs = [doc["content"] for doc in documents]

    response = co.rerank(
    query=query,
    documents=docs,
    top_n=top_k,
    model="rerank-english-v3.0"
)


    reranked = []
    for result in response.results:
        original_doc = documents[result.index]
        
        # Normalize relevance score (already ~0-1 but ensure)
        # Normalize dynamically across returned results
    scores = [r.relevance_score for r in response.results]
    min_s = min(scores)
    max_s = max(scores)

    for result in response.results:
        original_doc = documents[result.index]

        if max_s == min_s:
            normalized = 1.0
        else:
            normalized = (result.relevance_score - min_s) / (max_s - min_s)

        reranked.append({
            "id": original_doc["id"],
            "score": float(round(normalized, 4)),
            "content": original_doc["content"],
            "metadata": original_doc["metadata"]
        })

    # Sort descending
    reranked.sort(key=lambda x: x["score"], reverse=True)

    return reranked
