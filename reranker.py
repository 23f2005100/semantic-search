# reranker.py

import cohere
import os

co = cohere.Client(os.getenv("COHERE_API_KEY"))

def rerank_results(query, documents, top_k=5):
    docs = [doc["content"] for doc in documents]

    response = co.rerank(
        query=query,
        documents=docs,
        top_n=top_k,
        model="rerank-english-v2.0"
    )

    reranked = []
    for result in response.results:
        original_doc = documents[result.index]
        
        # Normalize relevance score (already ~0-1 but ensure)
        score = max(0, min(1, float(result.relevance_score)))

        reranked.append({
            "id": original_doc["id"],
            "score": score,
            "content": original_doc["content"],
            "metadata": original_doc["metadata"]
        })

    # Sort descending
    reranked.sort(key=lambda x: x["score"], reverse=True)

    return reranked
