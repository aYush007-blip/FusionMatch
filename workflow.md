[Raw Catalog Data] 
       │ (Images: front/side/back + Title/Description)
       ▼
[Preprocessing & Multi-View Feature Extraction]
       │ ──► SigLIP Vision Encoder ──► Mean-Pool Angle Embeddings ──► Normalized Vector (512-d)
       │ ──► SigLIP Text Encoder   ──► Normalized Text Vector (512-d)
       ▼
[Early Fusion / Late Fusion]
       │ ──► Combined Product Embedding: v_prod = Normalize(0.6 * v_img + 0.4 * v_text)
       ▼
[Local Vector Indexing (FAISS)]
       │ ──► FlatIP / HNSW Index (Cosine Similarity Search)
       ▼
[Deduplication & Clustering Engine]
       │ ──► Cosine Similarity Thresholding (e.g., score > 0.88)
       │ ──► Connected Components / Agglomerative Clustering
       ▼
[Evaluation & Lightweight Demo]
       └──► Ground Truth Evaluation (F1-score) + Streamlit / FastAPI UI