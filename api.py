import os
import tempfile
from typing import Dict, List

import numpy as np
import requests
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from PIL import Image
from dotenv import load_dotenv
from azure.cosmos import CosmosClient

# ─────────────────────────────────────────────
# LOAD YOUR EXISTING PIPELINE + EMBEDDER
# ─────────────────────────────────────────────
from query.pipeline import run_pipeline
import features.dino_embedder as dino_embedder

# preload model once
dino_embedder.get_model()

load_dotenv()

app = FastAPI(title="StoneX API")

# ─────────────────────────────────────────────
# CONFIG FROM ENV
# ─────────────────────────────────────────────
API_TOKEN = os.getenv("API_KEY")

COSMOS_URL       = os.getenv("COSMOS_URL")
COSMOS_KEY       = os.getenv("COSMOS_KEY")
COSMOS_DB_NAME   = os.getenv("COSMOS_DB_NAME", "stonexaiservicepreprod")
COSMOS_CONTAINER = os.getenv("COSMOS_CONTAINER", "images1")

# ─────────────────────────────────────────────
# COSMOS CLIENT
# ─────────────────────────────────────────────
cosmos_client = CosmosClient(COSMOS_URL, credential=COSMOS_KEY)
cosmos_container = (
    cosmos_client
    .get_database_client(COSMOS_DB_NAME)
    .get_container_client(COSMOS_CONTAINER)
)

# ─────────────────────────────────────────────
# CMD OFFICE MAPPING
# ─────────────────────────────────────────────
from cmd_mapping import resolve_family_name, is_cmd_class


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def normalize_name(name: str) -> str:
    return name.lower().replace("_", " ").strip()


def get_best_image_from_results(family_name, model_images):
    """
    Find the best-scoring image for a given family from model results.
    Matches by the raw folder name (before CMD mapping) since paths
    on disk still use the original folder names.
    """
    best_path = None
    best_score = -1.0

    for path, score in model_images:
        parts = path.replace("\\", "/").split("/")
        fam = normalize_name(parts[-2]) if len(parts) >= 2 else ""

        if fam == normalize_name(family_name):
            if score > best_score:
                best_path = path
                best_score = score

    return best_path, best_score


def validate_token(authorization: str):
    """Shared Bearer token validation."""
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header"
        )
    token = authorization.split(" ")[1]
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")


# ─────────────────────────────────────────────
# ENDPOINT 1: /predict
# ─────────────────────────────────────────────

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    authorization: str = Header(None)
) -> Dict:
    temp_path = None

    try:
        validate_token(authorization)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            contents = await file.read()
            tmp.write(contents)
            temp_path = tmp.name

        results = run_pipeline(
            temp_path,
            layer_order=["model"],
            top_k_families=5,
            top_k_images=800,
            first_layer_fetch=60,
        )

        families     = results.get("families", [])[:5]
        model_images = results.get("images", {}).get("model", [])

        output = []

        for raw_family, fam_score in families:
            img_path, img_score = get_best_image_from_results(raw_family, model_images)

            display_name = resolve_family_name(raw_family)
            cmd_flag     = is_cmd_class(raw_family)

            output.append({
                "family":       display_name,
                "raw_class":    raw_family,
                "is_cmd_class": cmd_flag,
                "family_score": float(fam_score),
                "best_image":   img_path,
                "image_score":  float(img_score) if img_score is not None else None,
            })

        return {"status": "success", "results": output}

    except HTTPException as e:
        raise e

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ─────────────────────────────────────────────
# ENDPOINT 2: /embedding
# ─────────────────────────────────────────────

@app.post("/embedding")
async def get_embedding(
    files: List[UploadFile] = File(...),
    authorization: str = Header(None)
) -> Dict:
    temp_paths = []

    try:
        validate_token(authorization)

        embeddings = []

        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                contents = await file.read()
                tmp.write(contents)
                temp_path = tmp.name
                temp_paths.append(temp_path)

            emb = dino_embedder.embed_image(temp_path)

            if emb is None:
                continue

            embeddings.append({
                "filename":  file.filename,
                "embedding": emb.tolist(),
                "dimension": len(emb),
            })

        return {
            "status":  "success",
            "count":   len(embeddings),
            "results": embeddings,
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        for p in temp_paths:
            if os.path.exists(p):
                os.remove(p)


# ─────────────────────────────────────────────
# ENDPOINT 3: /search
# Vector similarity search against Cosmos DB
# Index  : DiskANN
# Path   : /embedding  (float32, cosine, 1024-dim)
# Partition key: /stone_family
# ─────────────────────────────────────────────

@app.post("/search")
async def search(
    image: UploadFile = File(...),
    stone_families: List[str] = Form(...),
    top_k_per_lot: int = Form(20),
    db_top_n: int = Form(1000),
    authorization: str = Header(None)
) -> Dict:
    temp_path = None

    try:
        validate_token(authorization)

        # 1️⃣  Embed the query image locally (reuse dino_embedder — no extra HTTP hop)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            contents = await image.read()
            tmp.write(contents)
            temp_path = tmp.name

        query_embedding = dino_embedder.embed_image(temp_path)

        if query_embedding is None:
            return {"status": "error", "message": "Failed to generate embedding for the uploaded image."}

        query_embedding = query_embedding.tolist()   # Cosmos expects a plain list

        # 2️⃣  Cosmos DB vector search
        # VectorDistance with cosine returns a distance (0 = identical, lower = better)
        cosmos_query = """
            SELECT TOP @topN
                c.img_stone_family,
                c.img_lot_no,
                c.img_slab_no,
                c.img_blob_path,
                VectorDistance(c.img_embedding, @embedding) AS score
            FROM c
            WHERE ARRAY_CONTAINS(@families, c.img_stone_family)
            ORDER BY VectorDistance(c.img_embedding, @embedding)
        """

        items = list(cosmos_container.query_items(
            query=cosmos_query,
            parameters=[
                {"name": "@embedding", "value": query_embedding},
                {"name": "@families", "value": stone_families},
                {"name": "@topN",     "value": db_top_n},
            ],
            enable_cross_partition_query=True,
        ))

        # 3️⃣  Group by family → lot
        grouped: Dict[str, Dict[str, list]] = {}

        for item in items:
            fam = item["img_stone_family"]
            lot = item["img_lot_no"]

            grouped.setdefault(fam, {})
            grouped[fam].setdefault(lot, [])

            grouped[fam][lot].append({
                "slab_no":    item["img_slab_no"],
                "image_path": item["img_blob_path"],
                "score":      float(item["score"]),   # cosine distance (lower = closer)
            })

        # 4️⃣  Per-lot top-k (sorted by ascending distance)
        final_output = []

        for fam, lots in grouped.items():
            fam_entry = {"stone_family": fam, "lots": []}

            for lot, slabs in lots.items():
                slabs_sorted = sorted(slabs, key=lambda x: x["score"])
                fam_entry["lots"].append({
                    "lot_no": lot,
                    "slabs":  slabs_sorted[:top_k_per_lot],
                })

            final_output.append(fam_entry)

        return {"status": "success", "results": final_output}

    except HTTPException as e:
        raise e

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)