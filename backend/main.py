"""
MedGuard — FastAPI entry point
==============================
Startup sequence (lifespan):
  1. Load Bio_ClinicalBERT tokenizer + MedGuardModel
  2. Load pre-built DrugKnowledgeGraph (knowledge_graph.pkl) or demo graph
  3. Load LipinskiProcessor (DB_compounds_lipinski.csv)
  4. Inject all singletons into routes module

Project layout (main.py lives inside backend/):
  backend/
    main.py
    data/
    app/
      static/
        demo.html
      api/
        routes.py
      models/
        checkpoints/
          best_model_3heads.pt
      knowledge_graph/
        graph_builder.py

API prefix:  /api
Demo UI:     http://127.0.0.1:8000/
"""

import os
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ── Internal imports ──────────────────────────────────────────────────────────
from app.api import routes as routes_module
from app.models.medguard_model import load_model, load_tokenizer
from app.knowledge_graph.graph_builder import DrugKnowledgeGraph, build_demo_graph
from app.data.lipinski_processor import LipinskiProcessor

# ── Path configuration ────────────────────────────────────────────────────────
# main.py lives at:  backend/main.py
# BASE_DIR        =  .../medguard/backend
# Everything else lives under backend/app/
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
APP_DIR         = os.path.join(BASE_DIR, "app")
DATA_DIR        = os.path.join(BASE_DIR, "data")
STATIC_DIR      = os.path.join(APP_DIR, "static")
MODELS_DIR      = os.path.join(APP_DIR, "models", "checkpoints")

KG_PATH         = os.path.join(DATA_DIR, "knowledge_graph.pkl")
LIPINSKI_PATH   = os.path.join(DATA_DIR, "DB_compounds_lipinski.csv")
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "best_model_3heads.pt")

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"


# ── Lifespan: load all heavy resources once at startup ────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model, KG and Lipinski data at startup; inject into routes module.
    Everything is loaded into the routes module's global namespace so the
    endpoints can access them without re-loading on every request.
    """
    print("=" * 60)
    print("MedGuard — starting up")
    print("=" * 60)
    print(f"BASE_DIR        : {BASE_DIR}")
    print(f"APP_DIR         : {APP_DIR}")
    print(f"STATIC_DIR      : {STATIC_DIR}   (exists={os.path.isdir(STATIC_DIR)})")
    print(f"CHECKPOINT_PATH : {CHECKPOINT_PATH}   (exists={os.path.exists(CHECKPOINT_PATH)})")
    print(f"KG_PATH         : {KG_PATH}   (exists={os.path.exists(KG_PATH)})")
    print(f"LIPINSKI_PATH   : {LIPINSKI_PATH}   (exists={os.path.exists(LIPINSKI_PATH)})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device          : {device}")

    # ── 1. Tokenizer ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading Bio_ClinicalBERT tokenizer...")
    try:
        tokenizer = load_tokenizer(MODEL_NAME)
        routes_module.tokenizer = tokenizer
        print("      Tokenizer ready.")
    except Exception as exc:
        print(f"      ERROR loading tokenizer: {exc}")
        raise

    # ── 2. Model ──────────────────────────────────────────────────────────────
    print("\n[2/4] Loading MedGuard model...")
    try:
        model = load_model(MODEL_NAME)
        if os.path.exists(CHECKPOINT_PATH):
            print(f"      Found checkpoint: {CHECKPOINT_PATH}")
            state = torch.load(CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(state)
            print("      ✅ Fine-tuned weights loaded successfully.")
        else:
            print(f"      ⚠️  No checkpoint found at: {CHECKPOINT_PATH}")
            print("      Using pretrained Bio_ClinicalBERT weights only.")
            print("      → Confidence scores will be near-uniform (~25%) until trained weights load.")
        model.to(device)
        model.eval()
        routes_module.model = model
        print("      Model ready.")
    except Exception as exc:
        print(f"      ERROR loading model: {exc}")
        raise

    # ── 3. Knowledge Graph ────────────────────────────────────────────────────
    print("\n[3/4] Loading Knowledge Graph...")
    try:
        kg = DrugKnowledgeGraph()
        if os.path.exists(KG_PATH):
            kg.load(KG_PATH)
            print(f"      ✅ Full KG loaded from {KG_PATH}")
        else:
            print(f"      ⚠️  knowledge_graph.pkl not found at: {KG_PATH}")
            print("      Loading 10-node demo graph instead.")
            print("      → Run app/knowledge_graph/kg_builder_full.py to build the full graph.")
            kg = build_demo_graph()
        routes_module.kg = kg
        print(f"      KG ready — {kg.graph.number_of_nodes()} nodes, "
              f"{kg.graph.number_of_edges()} edges.")
    except Exception as exc:
        print(f"      WARNING: KG load failed ({exc}) — continuing without KG.")
        routes_module.kg = None

    # ── 4. Lipinski processor ─────────────────────────────────────────────────
    print("\n[4/4] Loading Lipinski physicochemical data...")
    try:
        if os.path.exists(LIPINSKI_PATH):
            lip = LipinskiProcessor()
            lip.load(LIPINSKI_PATH)
            lip.normalize_features()
            routes_module.lipinski = lip
            print(f"      ✅ Lipinski ready — {len(lip.drug_id_to_features)} compounds.")
        else:
            print(f"      ⚠️  DB_compounds_lipinski.csv not found at: {LIPINSKI_PATH}")
            print("      Continuing without Lipinski features.")
            routes_module.lipinski = None
    except Exception as exc:
        print(f"      WARNING: Lipinski load failed ({exc}) — continuing without it.")
        routes_module.lipinski = None

    print("\n" + "=" * 60)
    print("MedGuard startup complete — API is ready.")
    print("Open your browser at: http://127.0.0.1:8000")
    print("=" * 60 + "\n")

    yield  # ── application runs here ─────────────────────────────────────────

    print("MedGuard shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="MedGuard DDI Detection API",
    description=(
        "Drug-Drug Interaction detection using Bio_ClinicalBERT with "
        "multi-task learning (NER + DDI classification + Severity prediction) "
        "enriched by a DrugBank Knowledge Graph."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API routes ────────────────────────────────────────────────────────────────
app.include_router(routes_module.router, prefix="/api")

# ── Static files + demo UI ────────────────────────────────────────────────────
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def serve_demo():
        """Serve demo.html at the root URL."""
        demo_path = os.path.join(STATIC_DIR, "demo.html")
        if os.path.exists(demo_path):
            return FileResponse(demo_path)
        return {"message": f"demo.html not found inside {STATIC_DIR}"}
else:
    @app.get("/", include_in_schema=False)
    def root():
        return {
            "message": f"static/ folder not found — expected: {STATIC_DIR}",
            "fix": "Place demo.html at backend/app/static/demo.html",
            "docs": "/docs",
            "health": "/api/health",
        }


# ── Dev entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)