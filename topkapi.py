import requests
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from typing import List
from azure.cosmos import CosmosClient

app = FastAPI()

# -----------------------------
# COSMOS CONFIG
# -----------------------------
COSMOS_URL = "YOUR_URL"
COSMOS_KEY = "YOUR_KEY"
DB_NAME = "stonexaiservicepreprod"
CONTAINER_NAME = "images1"

client = CosmosClient(COSMOS_URL, credential=COSMOS_KEY)
container = client.get_database_client(DB_NAME).get_container_client(CONTAINER_NAME)

# -----------------------------
# EMBEDDING API
# -----------------------------
EMBEDDING_API = "http://localhost:8000/embedding"

def get_embedding(image: UploadFile):
    files = {"files": (image.filename, image.file, image.content_type)}
    res = requests.post(EMBEDDING_API, files=files)

    if res.status_code != 200:
        raise Exception("Embedding API failed")

    return res.json()["results"][0]["embedding"]

# -----------------------------
# MAIN API
# -----------------------------
@app.post("/search")
async def search(
    image: UploadFile = File(...),
    stone_families: List[str] = Form(...),
    top_k_per_lot: int = Form(20),
    db_top_n: int = Form(1000)   # IMPORTANT
):
    
    # 1️⃣ Get query embedding
    query_embedding = get_embedding(image)

    # 2️⃣ Cosmos Vector Search
    query = """
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

    items = list(container.query_items(
        query=query,
        parameters=[
            {"name": "@embedding", "value": query_embedding},
            {"name": "@families", "value": stone_families},
            {"name": "@topN", "value": db_top_n}
        ],
        enable_cross_partition_query=True
    ))

    # 3️⃣ GROUPING
    grouped = {}

    for item in items:
        fam = item["img_stone_family"]
        lot = item["img_lot_no"]

        grouped.setdefault(fam, {})
        grouped[fam].setdefault(lot, [])

        grouped[fam][lot].append({
            "slab_no": item["img_slab_no"],
            "image_path": item["img_blob_path"],
            "score": float(item["score"])   # distance
        })

    # 4️⃣ PER LOT TOP K
    final_output = []

    for fam, lots in grouped.items():
        fam_entry = {
            "stone_family": fam,
            "lots": []
        }

        for lot, slabs in lots.items():
            # lower score = better (distance)
            slabs_sorted = sorted(slabs, key=lambda x: x["score"])

            top_slabs = slabs_sorted[:top_k_per_lot]

            fam_entry["lots"].append({
                "lot_no": lot,
                "slabs": top_slabs
            })

        final_output.append(fam_entry)

    return {
        "status": "success",
        "results": final_output
    }