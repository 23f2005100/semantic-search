# main.py

from fastapi import FastAPI
import time
import os

from data_loader import fetch_scientific_abstracts
from vector_store import VectorStore
from reranker import rerank_results

app = FastAPI()

# Load documents at startup
documents = fetch_scientific_abstracts()
vector_store = VectorStore(documents)

@app.post("/search")
async def search(payload: dict):
    start_time = time.time()

    query = payload.get("query")
    k = payload.get("k", 8)
    rerank = payload.get("rerank", True)
    rerank_k = payload.get("rerankK", 5)

    if not query:
        return {"error": "Query is required"}

    # Stage 1: Vector search
    initial_results = vector_store.search(query, k=k)

    if not initial_results:
        return {
            "results": [],
            "reranked": False,
            "metrics": {
                "latency": 0,
                "totalDocs": len(documents)
            }
        }

    # Stage 2: Re-ranking
    if rerank:
        final_results = rerank_results(query, initial_results, top_k=rerank_k)
        reranked_flag = True
    else:
        final_results = initial_results[:rerank_k]
        reranked_flag = False

    latency = int((time.time() - start_time) * 1000)

    return {
        "results": final_results,
        "reranked": reranked_flag,
        "metrics": {
            "latency": latency,
            "totalDocs": len(documents)
        }
    }
