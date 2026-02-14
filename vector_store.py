# vector_store.py

import cohere
import faiss
import numpy as np
import os

co = cohere.Client(os.getenv("COHERE_API_KEY"))

class VectorStore:
    def __init__(self, documents):
        self.documents = documents
        self.dimension = 384  # embed-english-light-v3.0 dimension
        self.index = faiss.IndexFlatIP(self.dimension)
        self.embeddings = None
        self._build_index()

    def _embed_texts(self, texts):
        response = co.embed(
            texts=texts,
            model="embed-english-light-v3.0",
            input_type="search_document"
        )
        return np.array(response.embeddings)

    def _build_index(self):
        texts = [doc["content"] for doc in self.documents]
        embeddings = self._embed_texts(texts)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        self.embeddings = embeddings
        self.index.add(embeddings)

    def search(self, query, k=8):
        query_embedding = co.embed(
            texts=[query],
            model="embed-english-light-v3.0",
            input_type="search_query"
        )

        query_vector = np.array(query_embedding.embeddings)
        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    "id": self.documents[idx]["id"],
                    "score": float(score),
                    "content": self.documents[idx]["content"],
                    "metadata": self.documents[idx]["metadata"]
                })

        return results
