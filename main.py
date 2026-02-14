from fastapi import FastAPI
import time
from fastapi.middleware.cors import CORSMiddleware

from data_loader import fetch_scientific_abstracts
from vector_store import VectorStore
from reranker import rerank_results

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load documents
documents = fetch_scientific_abstracts()
vector_store = VectorStore(documents)


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
            "metrics": {
                "latency": 0,
                "totalDocs": len(documents)
            }
        }

    # Ensure enough candidates
    required_k = max(k, rerank_k, 10)
    initial_results = vector_store.search(query, k=required_k)

    if not initial_results:
        return {
            "results": [],
            "reranked": False,
            "metrics": {
                "latency": 0,
                "totalDocs": len(documents)
            }
        }

    # ---------- RERANK ----------
    if rerank:
        final_results = rerank_results(query, initial_results, top_k=rerank_k)
        reranked_flag = True
    else:
        # Normalize FAISS scores to 0–1
        scores = [r["score"] for r in initial_results]
        min_s = min(scores)
        max_s = max(scores)

        normalized = []
        for r in initial_results:
            if max_s == min_s:
                score = 0.5
            else:
                score = (r["score"] - min_s) / (max_s - min_s)

            normalized.append({
                "id": r["id"],
                "score": float(score),
                "content": r["content"],
                "metadata": r["metadata"]
            })

        normalized.sort(key=lambda x: x["score"], reverse=True)
        final_results = normalized[:rerank_k]
        reranked_flag = False

    # ---------- CLEAN + STRICT SCORE CLAMP ----------
    cleaned_results = []

    for r in final_results[:rerank_k]:
        safe_score = float(r["score"])

        # STRICT (0,1) range
        if safe_score <= 0:
            safe_score = 0.0001
        elif safe_score >= 1:
            safe_score = 0.9999

        cleaned_results.append({
            "id": int(r["id"]),
            "score": round(safe_score, 4),
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
