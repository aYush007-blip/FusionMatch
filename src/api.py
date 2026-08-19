import io
from PIL import Image
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# pyrefly: ignore [missing-import]
from pydantic import BaseModel
from src.model import DedupModelEngine
from src.indexer import CatalogIndexer

app = FastAPI(
    title="Multimodal Product Deduplication Microservice",
    version="1.0.0",
    docs_url="/docs"
)

engine = DedupModelEngine()
indexer = CatalogIndexer()

class ItemRegistrationResponse(BaseModel):
    item_id: str
    status: str
    total_indexed: int

class SearchDuplicateResponse(BaseModel):
    is_duplicate: bool
    highest_similarity: float
    matches: list[dict]

@app.get("/healthz")
def healthcheck():
    return {"status": "healthy", "indexed_items": indexer.index.ntotal}

@app.post("/items/index", response_model=ItemRegistrationResponse)
async def register_item(
    item_id: str = Form(...),
    title: str = Form(...),
    images: list[UploadFile] = File(...)
):
    if not images:
        raise HTTPException(status_code=400, detail="At least one image angle is required.")
    
    pil_images = [Image.open(io.BytesIO(await img.read())).convert("RGB") for img in images]
    vec = engine.extract_fused_vector(pil_images, title).cpu().numpy()
    
    indexer.add_item(item_id=item_id, title=title, vector=vec)
    indexer.persist()
    
    return {
        "item_id": item_id,
        "status": "indexed",
        "total_indexed": indexer.index.ntotal
    }

@app.post("/items/check-duplicate", response_model=SearchDuplicateResponse)
async def check_duplicate(
    title: str = Form(...),
    images: list[UploadFile] = File(...)
):
    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required.")
    
    pil_images = [Image.open(io.BytesIO(await img.read())).convert("RGB") for img in images]
    query_vec = engine.extract_fused_vector(pil_images, title).cpu().numpy()
    
    duplicates = indexer.search_duplicates(query_vec, top_k=5)
    highest_score = duplicates[0]["similarity_score"] if duplicates else 0.0
    
    return {
        "is_duplicate": len(duplicates) > 0,
        "highest_similarity": highest_score,
        "matches": duplicates
    }