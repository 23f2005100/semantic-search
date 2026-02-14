# main.py

from fastapi import FastAPI
import time
import os
from fastapi.middleware.cors import CORSMiddleware

from data_loader import fetch_scientific_abstracts
from vector_store import VectorStore
from reranker import rerank_results

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load documents at startup
documents = fetch_scientific_abstracts()
vector_store = VectorStore(documents)
print("Total documents fetched:", len(documents))

@app.post("/search")
async def search(payload: dict):
    start_time = time.time()

    query = payload.get("query")
    k = int(payload.get("k", 8))
    rerank = bool(payload.get("rerank", True))
    rerank_k = int(payload.get("rerankK", 5))

    if not query or not isinstance(query, str):
        return {
            "results": [],
            "reranked": False,
            "metrics": {"latency": 0, "totalDocs": len(documents)}
        }

    initial_results = vector_store.search(query, k=k)

    if not initial_results:
        return {
            "results": [],
            "reranked": False,
            "metrics": {"latency": 0, "totalDocs": len(documents)}
        }

    # ---------- RERANK ----------
    if rerank:
        final_results = rerank_results(query, initial_results, top_k=rerank_k)
        reranked_flag = True
    else:
        # Normalize FAISS scores manually to 0-1
        scores = [r["score"] for r in initial_results]
        min_s = min(scores)
        max_s = max(scores)

        normalized = []
        for r in initial_results:
            if max_s == min_s:
                score = 1.0
            else:
                score = (r["score"] - min_s) / (max_s - min_s)

            normalized.append({
                "id": r["id"],
                "score": float(round(score, 4)),
                "content": r["content"],
                "metadata": r["metadata"]
            })

        final_results = sorted(
            normalized, key=lambda x: x["score"], reverse=True
        )[:rerank_k]

        reranked_flag = False

    # ---- FINAL SANITY CHECK ----
    cleaned_results = []
    for r in final_results:
        cleaned_results.append({
            "id": int(r["id"]),
            "score": float(max(0.0, min(1.0, r["score"]))),
            "content": r["content"],
            "metadata": r["metadata"]
        })

    latency = int((time.time() - start_time) * 1000)

    return {
        "results": cleaned_results,
        "reranked": reranked_flag,
        "metrics": {
            "latency": latency,
            "totalDocs": len(documents)
        }
    }
