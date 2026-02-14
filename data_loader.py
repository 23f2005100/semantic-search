# data_loader.py

import requests

def fetch_scientific_abstracts(query="machine learning", limit=120):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract"
    }

    response = requests.get(url, params=params)
    data = response.json()

    documents = []
    for i, paper in enumerate(data.get("data", [])):
        if paper.get("abstract"):
            documents.append({
                "id": i,
                "content": paper["abstract"],
                "metadata": {
                    "title": paper.get("title", ""),
                    "source": "SemanticScholar"
                }
            })

    return documents
