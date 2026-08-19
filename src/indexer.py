import os
import json
import faiss
import numpy as np
import networkx as nx
from src.config import settings

class CatalogIndexer:
    def __init__(self):
        self.dim = settings.EMBEDDING_DIM
        self.threshold = settings.SIMILARITY_THRESHOLD
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadata = []
        self._load_persisted_index()

    def _load_persisted_index(self):
        if os.path.exists(settings.FAISS_INDEX_PATH) and os.path.exists(settings.METADATA_PATH):
            self.index = faiss.read_index(settings.FAISS_INDEX_PATH)
            with open(settings.METADATA_PATH, "r") as f:
                self.metadata = json.load(f)

    def persist(self):
        os.makedirs(os.path.dirname(settings.FAISS_INDEX_PATH), exist_ok=True)
        faiss.write_index(self.index, settings.FAISS_INDEX_PATH)
        with open(settings.METADATA_PATH, "w") as f:
            json.dump(self.metadata, f)

    def add_item(self, item_id: str, title: str, vector: np.ndarray):
        self.index.add(vector.astype("float32"))
        self.metadata.append({"item_id": item_id, "title": title})

    def search_duplicates(self, query_vec: np.ndarray, top_k: int = 5):
        if self.index.ntotal == 0:
            return []
        scores, indices = self.index.search(query_vec.astype("float32"), min(top_k, self.index.ntotal))
        
        matches = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score >= self.threshold:
                match_data = self.metadata[idx].copy()
                match_data["similarity_score"] = float(score)
                matches.append(match_data)
        return matches