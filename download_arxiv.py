import requests
import xml.etree.ElementTree as ET
import json

def fetch_arxiv_abstracts(query="machine learning", max_results=120):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
    
    response = requests.get(url)
    root = ET.fromstring(response.content)

    documents = []
    for i, entry in enumerate(root.findall("{http://www.w3.org/2005/Atom}entry")):
        abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
        title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()

        documents.append({
            "id": i,
            "content": abstract,
            "metadata": {
                "title": title,
                "source": "arXiv"
            }
        })

    return documents


docs = fetch_arxiv_abstracts()

with open("documents.json", "w") as f:
    json.dump(docs, f, indent=2)

print("Saved", len(docs), "documents")
