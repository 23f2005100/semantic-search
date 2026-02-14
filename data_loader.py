import json

def fetch_scientific_abstracts():
    with open("documents.json", "r") as f:
        return json.load(f)
