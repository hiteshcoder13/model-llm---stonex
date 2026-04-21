"""
StoneX FastAPI application.

Start with:
    uvicorn api:app --reload

Endpoints
---------
POST /predict           — visual model: top-5 stone families for an image
POST /embedding         — generate DINOv2 embeddings for one or more images
POST /search            — vector search: find visually similar slabs in Cosmos DB
POST /ingest-inventory  — pull inventory from Stonex Galleria and embed into Cosmos
POST /rerank            — Gemini visual RAG: re-rank candidate families against a slab image
POST /discover          — combined pipeline: predict → rerank → similarity search (top-2 families)
"""
import re
import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from dotenv import load_dotenv
from azure.cosmos import CosmosClient

from query.pipeline import run_pipeline
import features.dino_embedder as dino_embedder
from cmd_mapping import resolve_family_name, is_cmd_class
from ingest_worker import process_lot_parallel
from stone_reranker import rerank_stone_families   # Gemini visual reranker

load_dotenv()

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────

dino_embedder.get_model()

app = FastAPI(title="StoneX API")

API_TOKEN = os.getenv("API_KEY")

COSMOS_URL       = os.getenv("COSMOS_URL")
COSMOS_KEY       = os.getenv("COSMOS_KEY")
COSMOS_DB_NAME   = os.getenv("COSMOS_DB_NAME", "stonexaiservicepreprod")
COSMOS_CONTAINER = os.getenv("COSMOS_CONTAINER", "images1")

INVENTORY_API_URL = os.getenv("INVENTORY_API_URL", "https://app.stonexgalleria.com/ai")
INVENTORY_API_KEY = os.getenv("INVENTORY_API_KEY")

cosmos_client = CosmosClient(COSMOS_URL, credential=COSMOS_KEY)
cosmos_container = (
    cosmos_client
    .get_database_client(COSMOS_DB_NAME)
    .get_container_client(COSMOS_CONTAINER)
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def validate_token(authorization: str) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing/Invalid Authorization header")
    token = authorization.split(" ", 1)[1]
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")


def normalize_inventory_response(resp: Any) -> List[dict]:
    """
    Handles multiple possible API response formats safely.

    Supported shapes:
      1. Direct list:              [{ lot }, ...]
      2. Flat wrapped:             { "data": [{ lot }, ...] }
      3. Nested wrapped (actual):  { "data": { "lots": [{ lot }, ...] } }
      4. Top-level lots key:       { "lots": [{ lot }, ...] }
    """
    if resp is None:
        return []
    if isinstance(resp, list):
        return resp
    if isinstance(resp, dict):
        data = resp.get("data")
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            lots = data.get("lots")
            if isinstance(lots, list):
                return lots
        lots = resp.get("lots")
        if isinstance(lots, list):
            return lots
    return []


def get_best_image_from_results(family_name: str, model_images: list):
    best_path, best_score = None, -1.0
    for path, score in model_images:
        try:
            parts = path.replace("\\", "/").split("/")
            fam = parts[-2].lower().replace("_", " ").strip() if len(parts) >= 2 else ""
            if fam == family_name.lower().replace("_", " ").strip() and score > best_score:
                best_path, best_score = path, score
        except Exception:
            continue
    return best_path, best_score


def normalize_stone_family(name: str) -> str:
    if not name:
        return ""

    name = name.lower().strip()

    # Remove 'tile' prefix
    name = re.sub(r"^tile[_\s]*", "", name)

    # Replace underscores with spaces so \b word boundaries work
    name = name.replace("_", " ")

    # Remove variation words (all types)
    name = re.sub(r"\b(all\s*)?variations?\b", "", name)

    # Remove numbers
    name = re.sub(r"\d+", "", name)

    # Collapse extra spaces and uppercase
    return " ".join(name.split()).upper()


def _cosmos_similarity_search(
    query_vec: list,
    stone_families: List[str],
    db_top_n: int = 2000,
    top_k_per_lot: int = 20,
) -> List[Dict]:
    """
    Shared helper: run a vector-distance query against Cosmos DB for the given
    families and return results grouped and ranked as:
        family → lots (ranked by best slab distance) → slabs (ranked by distance).
    """
    family_conditions = []
    parameters = [{"name": "@embedding", "value": query_vec}]

    for i, family in enumerate(stone_families):
        param_name = f"@family{i}"
        family_conditions.append(f"UPPER(c.stone_family) = {param_name}")
        parameters.append({"name": param_name, "value": family})

    where_clause = " OR ".join(family_conditions)

    query = f"""
    SELECT TOP {db_top_n}
        c.stone_family,
        c.img_lot_no,
        c.img_slab_no,
        c.img_blob_path,
        c.img_color_signature,
        VectorDistance(c.embedding, @embedding) AS score
    FROM c
    WHERE {where_clause}
    ORDER BY VectorDistance(c.embedding, @embedding)
    """

    items = list(cosmos_container.query_items(
        query=query,
        parameters=parameters,
        enable_cross_partition_query=True,
    ))

    # ── Group by family → lot → slabs ────────────────────────────────────────
    grouped: Dict[str, Dict[str, list]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        fam = item.get("stone_family")
        lot = item.get("img_lot_no")
        if not fam or not lot:
            continue
        grouped.setdefault(fam, {}).setdefault(lot, []).append({
            "slab_no":    item.get("img_slab_no"),
            "image_path": item.get("img_blob_path"),
            "distance":   float(item.get("score", 1.0)),
        })

    # ── Rank within each family ───────────────────────────────────────────────
    family_results: Dict[str, Dict] = {}
    for fam, lots in grouped.items():
        lot_entries = []
        for lot_no, slabs in lots.items():
            slabs_ranked       = sorted(slabs, key=lambda x: x["distance"])
            best_slab_distance = slabs_ranked[0]["distance"] if slabs_ranked else 1.0
            lot_entries.append({
                "lot_no":        lot_no,
                "best_distance": best_slab_distance,
                "slabs":         slabs_ranked[:top_k_per_lot],
            })
        lot_entries_ranked = sorted(lot_entries, key=lambda x: x["best_distance"])
        best_lot_distance  = lot_entries_ranked[0]["best_distance"] if lot_entries_ranked else 1.0
        family_results[fam.upper()] = {
            "stone_family":  fam,
            "best_distance": best_lot_distance,
            "lots":          lot_entries_ranked,
        }

    return family_results


# ─────────────────────────────────────────────
# 1️⃣  PREDICT
# ─────────────────────────────────────────────

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    authorization: str = Header(None)
) -> Dict:

    temp_path = None

    try:
        validate_token(authorization)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await image.read())
            temp_path = tmp.name

        results = run_pipeline(
            temp_path,
            layer_order=["model"],
            top_k_families=5,
            top_k_images=800,
            first_layer_fetch=100,
        )

        families    = results.get("families", [])[:5]
        model_images = results.get("images", {}).get("model", [])

        output = []
        for raw_family, fam_score in families:
            _img_path, _img_score = get_best_image_from_results(raw_family, model_images)

            if is_cmd_class(raw_family):
                clean_family = resolve_family_name(raw_family).upper()
            else:
                clean_family = normalize_stone_family(raw_family)

            output.append({
                "family":       clean_family,
                "family_score": round(float(fam_score), 4)
            })

        return {"status": "success", "results": output}

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ─────────────────────────────────────────────
# 2️⃣  EMBEDDING
# ─────────────────────────────────────────────

@app.post("/embedding")
async def get_embedding(
    files: List[UploadFile] = File(...),
    authorization: str = Header(None),
) -> Dict:
    temp_paths = []
    try:
        validate_token(authorization)
        results = []

        for upload in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(await upload.read())
                temp_paths.append(tmp.name)

            emb = dino_embedder.embed_image(tmp.name)
            if emb is not None:
                results.append({
                    "filename":  upload.filename,
                    "embedding": emb.tolist(),
                    "dimension": len(emb),
                })

        return {"status": "success", "count": len(results), "results": results}

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        for p in temp_paths:
            if os.path.exists(p):
                os.remove(p)


# ─────────────────────────────────────────────
# 3️⃣  SEARCH
# ─────────────────────────────────────────────

@app.post("/search")
async def search(
    image: UploadFile = File(...),
    stone_families: List[str] = Form(...),
    top_k_per_lot: int = Form(20),
    db_top_n: int = Form(2000),
    authorization: str = Header(None),
) -> Dict:
    """
    Search for the most visually similar stone slabs to the uploaded image.

    Returns results grouped by stone_family → lots, where:
    - Slabs within each lot are ranked by similarity (most similar first).
    - Lots within each family are ranked by their best slab's similarity score.
    - Families are ranked by their best lot's best slab similarity score.

    Score field: VectorDistance (cosine distance). Lower = more similar to query image.
    """
    temp_path = None
    try:
        validate_token(authorization)

        stone_families = [
            f.strip().upper()
            for f in stone_families
            if f and f.strip()
        ]

        print("🔍 Received families:", stone_families)

        if not stone_families:
            return {"status": "error", "message": "No valid stone families provided"}

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await image.read())
            temp_path = tmp.name

        query_embedding = dino_embedder.embed_image(temp_path)
        if query_embedding is None:
            return {
                "status":  "error",
                "message": "Embedding failed — could not generate vector for the uploaded image.",
            }

        family_results = _cosmos_similarity_search(
            query_vec=query_embedding.tolist(),
            stone_families=stone_families,
            db_top_n=db_top_n,
            top_k_per_lot=top_k_per_lot,
        )

        final_ranked = sorted(family_results.values(), key=lambda x: x["best_distance"])
        return {"status": "success", "results": final_ranked}

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ─────────────────────────────────────────────
# 4️⃣  INGEST INVENTORY
# ─────────────────────────────────────────────

@app.post("/ingest-inventory")
async def ingest_inventory(
    limit: int = 1000,
    offset: int = 0,
    max_workers: int = 8,
    authorization: str = Header(None),
) -> Dict:
    try:
        validate_token(authorization)

        url      = f"{INVENTORY_API_URL}?limit={limit}&offset={offset}"
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {INVENTORY_API_KEY}"},
            timeout=30,
        )

        if response.status_code != 200:
            return {"status": "error", "message": response.text}

        raw_data = response.json()
        lots     = normalize_inventory_response(raw_data)

        if not lots:
            return {
                "status":   "error",
                "message":  "No lots found in inventory response. Check API shape.",
                "raw_keys": list(raw_data.keys()) if isinstance(raw_data, dict) else str(type(raw_data)),
            }

        total_inserted = total_skipped = total_failed = 0

        for lot in lots:
            if not isinstance(lot, dict):
                continue
            if lot.get("lot_status") != "A":
                total_skipped += 1
                continue
            try:
                results = process_lot_parallel(lot, cosmos_container, max_workers=max_workers)
                for r in results:
                    status = r.get("status")
                    if status == "success":
                        total_inserted += 1
                    elif status == "skip":
                        total_skipped += 1
                    else:
                        total_failed += 1
            except Exception as e:
                print(f"[ingest-inventory] ❌ lot={lot.get('lotno')} error: {e}")
                total_failed += 1

        return {
            "status":     "success",
            "total_lots": len(lots),
            "inserted":   total_inserted,
            "skipped":    total_skipped,
            "failed":     total_failed,
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# ─────────────────────────────────────────────
# 5️⃣  RERANK  (Gemini visual RAG)
# ─────────────────────────────────────────────

@app.post("/rerank")
async def rerank(
    image: UploadFile = File(...),
    candidates: str = Form(...),
    authorization: str = Header(None),
) -> Dict:
    """
    Visually re-rank stone family candidates against an uploaded slab image
    using Gemini multimodal intelligence + the Stonex 200-stone knowledge base.

    ---
    ### How it works
    1. You pass a slab image and a JSON list of candidate stone family names
       (typically the top results from `/predict` or `/search`).
    2. Each family name is looked up in `stones_db.json` — a 200-stone knowledge base
       containing color, veining, texture, grain, pattern type, and differentiator text.
    3. The image + all descriptors are sent together to **Gemini 2.5 Flash Lite** as a
       structured multimodal prompt.
    4. Gemini visually inspects the slab, matches it against each descriptor, and
       returns all candidates re-ranked from best match to worst, with explanations.

    ---
    ### Request fields
    | Field         | Type           | Required | Description |
    |---------------|----------------|----------|-------------|
    | image         | file (binary)  | yes      | Slab photo — JPG, PNG, or WEBP |
    | candidates    | JSON string    | yes      | Array of candidate objects (see below) |
    | Authorization | Bearer <token> | yes      | API token |

    ### candidates JSON format
    ```json
    [
      {"family_name": "Carrara"},
      {"family_name": "Statuario"},
      {"family_name": "Bianco Lasa"}
    ]
    ```
    - `family_name` — exact stone family name (matched case-insensitively against the KB).
    - `score` — original vector-distance score (optional).

    ---
    ### Response
    ```json
    {
      "status": "success",
      "image_summary": "White marble with fine grey directional veining and soft movement",
      "ranked_families": [
        {
          "rank": 1,
          "family_name": "Carrara",
          "confidence": 0.91,
          "match_reason": "Fine grey veining on white base matches Carrara exactly.",
          "mismatch_notes": ""
        }
      ],
      "kb_hits":   ["Carrara", "Statuario", "Bianco Lasa"],
      "kb_misses": []
    }
    ```
    """
    try:
        validate_token(authorization)

        try:
            candidate_list = json.loads(candidates)
        except json.JSONDecodeError:
            return {
                "status":  "error",
                "message": (
                    "candidates must be a valid JSON string. "
                    'Example: [{"family_name": "Carrara"}]'
                ),
            }

        if not isinstance(candidate_list, list) or not candidate_list:
            return {"status": "error", "message": "candidates must be a non-empty JSON array."}

        for item in candidate_list:
            if not isinstance(item, dict) or "family_name" not in item:
                return {
                    "status":  "error",
                    "message": (
                        "Each candidate must be a JSON object with at least 'family_name'. "
                        'Example: {"family_name": "Carrara"}'
                    ),
                }

        image_bytes = await image.read()
        if not image_bytes:
            return {"status": "error", "message": "Uploaded image file is empty."}

        mime = image.content_type or ""
        if mime not in {"image/jpeg", "image/png", "image/webp"}:
            ext  = (image.filename or "").rsplit(".", 1)[-1].lower()
            mime = {
                "jpg":  "image/jpeg",
                "jpeg": "image/jpeg",
                "png":  "image/png",
                "webp": "image/webp",
            }.get(ext, "image/jpeg")

        # Strip scores before sending to Gemini — only family_name is passed
        gemini_candidates = [{"family_name": c["family_name"]} for c in candidate_list]

        result = await rerank_stone_families(
            image_bytes=image_bytes,
            image_mime=mime,
            candidates=gemini_candidates,
        )

        return {
            "status":          "success",
            "image_summary":   result["image_summary"],
            "ranked_families": result["ranked_families"],
            "kb_hits":         result["kb_hits"],
            "kb_misses":       result["kb_misses"],
        }

    except RuntimeError as exc:
        logger.error("Gemini error in /rerank: %s", exc)
        return {"status": "error", "message": str(exc)}

    except Exception as exc:
        logger.exception("Unexpected error in /rerank")
        return {"status": "error", "message": str(exc)}


# ─────────────────────────────────────────────
# 6️⃣  DISCOVER  (predict → rerank → search)
# ─────────────────────────────────────────────
@app.post("/discover")
async def discover(
    image: UploadFile = File(...),
    top_k_per_lot: int = Form(20),
    db_top_n: int = Form(2000),
    authorization: str = Header(None),
) -> Dict:

    temp_path = None
    image_bytes: Optional[bytes] = None

    try:
        validate_token(authorization)

        # ── Read image ─────────────────────────────────────────────────────
        image_bytes = await image.read()
        if not image_bytes:
            return {"status": "error", "message": "Uploaded image file is empty."}

        # MIME resolve
        mime = image.content_type or ""
        if mime not in {"image/jpeg", "image/png", "image/webp"}:
            ext = (image.filename or "").rsplit(".", 1)[-1].lower()
            mime = {
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "png": "image/png",
                "webp": "image/webp",
            }.get(ext, "image/jpeg")

        # Temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(image_bytes)
            temp_path = tmp.name

        # ────────────────────────────────────────────────────────────────────
        # STEP 1 — Predict (DINOv2)
        # ────────────────────────────────────────────────────────────────────
        print("🔮 Step 1: running predict pipeline …")

        pipeline_results = run_pipeline(
            temp_path,
            layer_order=["model"],
            top_k_families=5,
            top_k_images=800,
            first_layer_fetch=100,
        )

        raw_families = pipeline_results.get("families", [])[:5]

        predicted_families: List[str] = []
        for raw_family, _ in raw_families:
            if is_cmd_class(raw_family):
                clean = resolve_family_name(raw_family).upper()
            else:
                clean = normalize_stone_family(raw_family)
            if clean:
                predicted_families.append(clean)

        if not predicted_families:
            return {
                "status": "error",
                "message": "Predict step returned no stone families.",
            }

        print(f"✅ Predicted families: {predicted_families}")

        # ────────────────────────────────────────────────────────────────────
        # 🚀 STEP 2 — NO GEMINI → just pick top-2 directly
        # ────────────────────────────────────────────────────────────────────
        top2_families = predicted_families[:2]

        print(f"🎯 Top-2 families (model): {top2_families}")

        # Fake rerank structure (to keep response consistent)
        reranked_families = [
            {
                "rank": i + 1,
                "family_name": f,
                "confidence": None,
                "match_reason": "Selected directly from model prediction",
                "mismatch_notes": "",
            }
            for i, f in enumerate(predicted_families)
        ]

        image_summary = ""
        kb_hits = []
        kb_misses = []

        # ────────────────────────────────────────────────────────────────────
        # STEP 3 — Embedding + Cosmos Search
        # ────────────────────────────────────────────────────────────────────
        print("🔍 Embedding image …")

        query_embedding = dino_embedder.embed_image(temp_path)
        if query_embedding is None:
            return {
                "status": "error",
                "message": "Embedding failed.",
            }

        print("🔍 Querying Cosmos DB …")

        family_results = _cosmos_similarity_search(
            query_vec=query_embedding.tolist(),
            stone_families=[f.upper() for f in top2_families],
            db_top_n=db_top_n,
            top_k_per_lot=top_k_per_lot,
        )

        print(f"✅ Cosmos results: {list(family_results.keys())}")

        # ────────────────────────────────────────────────────────────────────
        # FINAL RESPONSE
        # ────────────────────────────────────────────────────────────────────
        final_results = []
        for i, fam in enumerate(top2_families):
            cosmos_data = family_results.get(fam.upper(), {})

            final_results.append({
                "stone_family": fam,
                "gemini_rank": i + 1,  # keeping field name same
                "confidence": None,
                "match_reason": "Selected directly from model prediction",
                "mismatch_notes": "",
                "best_distance": cosmos_data.get("best_distance"),
                "lots": cosmos_data.get("lots", []),
            })

        return {
            "status": "success",
            "image_summary": image_summary,
            "predicted_families": predicted_families,
            "reranked_families": reranked_families,
            "kb_hits": kb_hits,
            "kb_misses": kb_misses,
            "results": final_results,
        }

    except Exception as exc:
        logger.exception("Unexpected error in /discover")
        return {"status": "error", "message": str(exc)}

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


