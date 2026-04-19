from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# ── Default model configs (defined early for lifespan use) ───────────────────
DEFAULT_MODELS = [
    {
        "name": "COCO-SSD Detection",
        "task": "detect",
        "url": "",
        "description": "Real-time object detection on 80 COCO classes using COCO-SSD. Runs in-browser via TF.js WebGL backend.",
        "labels": ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
        "is_builtin": True,
    },
    {
        "name": "MoveNet Pose Lightning",
        "task": "pose",
        "url": "",
        "description": "Single-person pose estimation detecting 17 keypoints. Ultra-fast (50ms) using MoveNet Lightning architecture.",
        "labels": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder"],
        "is_builtin": True,
    },
    {
        "name": "BodyPix Segmentation",
        "task": "segment",
        "url": "",
        "description": "Person segmentation separating foreground from background. Uses MobileNetV1 backbone for browser-optimized performance.",
        "labels": ["person", "background"],
        "is_builtin": True,
    },
    {
        "name": "MobileNet Classification",
        "task": "classify",
        "url": "",
        "description": "ImageNet classification on 1000 classes using MobileNetV2. Optimized for mobile and browser deployment.",
        "labels": ["1000 ImageNet classes"],
        "is_builtin": True,
    },
    {
        "name": "COCO-SSD Tracking",
        "task": "track",
        "url": "",
        "description": "Multi-object tracking using COCO-SSD detection + centroid-based tracker. Assigns persistent IDs across frames.",
        "labels": ["person", "bicycle", "car", "motorcycle", "airplane", "bus"],
        "is_builtin": True,
    },
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: seed default models
    count = await db.model_configs.count_documents({"is_builtin": True})
    if count == 0:
        for m in DEFAULT_MODELS:
            mc = ModelConfig(**m)
            await db.model_configs.insert_one(mc.model_dump())
        logger.info("Seeded %d default model configs", len(DEFAULT_MODELS))
    yield
    # Shutdown
    client.close()

app = FastAPI(
    title="YOLO26 CV Platform",
    description="Professional Computer Vision Platform powered by YOLO26 & TensorFlow.js",
    version="1.0.0",
    lifespan=lifespan,
)
api_router = APIRouter(prefix="/api")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Helpers ──────────────────────────────────────────────────────────────────

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def new_id():
    return str(uuid.uuid4())

# ── Pydantic Models ───────────────────────────────────────────────────────────

class ProjectCreate(BaseModel):
    name: str
    description: str = ""
    task_types: List[str] = []

class Project(ProjectCreate):
    id: str = Field(default_factory=new_id)
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    task_types: Optional[List[str]] = None

class RunCreate(BaseModel):
    project_id: Optional[str] = None
    task: str
    model_name: str
    source_type: str  # image, video, webcam
    results_count: int = 0
    detections: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {}
    thumbnail: Optional[str] = None

class Run(RunCreate):
    id: str = Field(default_factory=new_id)
    created_at: str = Field(default_factory=now_iso)

class DatasetCreate(BaseModel):
    name: str
    description: str = ""
    project_id: Optional[str] = None
    classes: List[str] = []
    image_count: int = 0

class Dataset(DatasetCreate):
    id: str = Field(default_factory=new_id)
    created_at: str = Field(default_factory=now_iso)

class ModelConfigCreate(BaseModel):
    name: str
    task: str
    url: str = ""
    description: str = ""
    labels: List[str] = []
    is_builtin: bool = False

class ModelConfig(ModelConfigCreate):
    id: str = Field(default_factory=new_id)
    created_at: str = Field(default_factory=now_iso)

# ── Projects ──────────────────────────────────────────────────────────────────

@api_router.get("/projects", response_model=List[Project])
async def list_projects():
    docs = await db.projects.find({}, {"_id": 0}).sort("created_at", -1).to_list(200)
    return docs

@api_router.post("/projects", response_model=Project)
async def create_project(data: ProjectCreate):
    proj = Project(**data.model_dump())
    await db.projects.insert_one(proj.model_dump())
    return proj

@api_router.get("/projects/{project_id}", response_model=Project)
async def get_project(project_id: str):
    doc = await db.projects.find_one({"id": project_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Project not found")
    return doc

@api_router.put("/projects/{project_id}", response_model=Project)
async def update_project(project_id: str, data: ProjectUpdate):
    doc = await db.projects.find_one({"id": project_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Project not found")
    update_data = {k: v for k, v in data.model_dump().items() if v is not None}
    update_data["updated_at"] = now_iso()
    await db.projects.update_one({"id": project_id}, {"$set": update_data})
    doc.update(update_data)
    return doc

@api_router.delete("/projects/{project_id}")
async def delete_project(project_id: str):
    result = await db.projects.delete_one({"id": project_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"deleted": True}

# ── Runs ──────────────────────────────────────────────────────────────────────

@api_router.get("/runs", response_model=List[Run])
async def list_runs(project_id: Optional[str] = Query(None), task: Optional[str] = Query(None), limit: int = Query(100)):
    query = {}
    if project_id:
        query["project_id"] = project_id
    if task:
        query["task"] = task
    docs = await db.runs.find(query, {"_id": 0}).sort("created_at", -1).to_list(limit)
    return docs

@api_router.post("/runs", response_model=Run)
async def create_run(data: RunCreate):
    run = Run(**data.model_dump())
    await db.runs.insert_one(run.model_dump())
    return run

@api_router.get("/runs/{run_id}", response_model=Run)
async def get_run(run_id: str):
    doc = await db.runs.find_one({"id": run_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Run not found")
    return doc

@api_router.delete("/runs/{run_id}")
async def delete_run(run_id: str):
    result = await db.runs.delete_one({"id": run_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"deleted": True}

# ── Datasets ──────────────────────────────────────────────────────────────────

@api_router.get("/datasets", response_model=List[Dataset])
async def list_datasets(project_id: Optional[str] = Query(None)):
    query = {}
    if project_id:
        query["project_id"] = project_id
    docs = await db.datasets.find(query, {"_id": 0}).sort("created_at", -1).to_list(200)
    return docs

@api_router.post("/datasets", response_model=Dataset)
async def create_dataset(data: DatasetCreate):
    ds = Dataset(**data.model_dump())
    await db.datasets.insert_one(ds.model_dump())
    return ds

@api_router.get("/datasets/{dataset_id}", response_model=Dataset)
async def get_dataset(dataset_id: str):
    doc = await db.datasets.find_one({"id": dataset_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return doc

@api_router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    result = await db.datasets.delete_one({"id": dataset_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"deleted": True}

# ── Model Configs ─────────────────────────────────────────────────────────────

@api_router.get("/model-configs", response_model=List[ModelConfig])
async def list_model_configs(task: Optional[str] = Query(None)):
    query = {}
    if task:
        query["task"] = task
    docs = await db.model_configs.find(query, {"_id": 0}).sort("created_at", 1).to_list(100)
    return docs

@api_router.post("/model-configs", response_model=ModelConfig)
async def create_model_config(data: ModelConfigCreate):
    mc = ModelConfig(**data.model_dump())
    await db.model_configs.insert_one(mc.model_dump())
    return mc

@api_router.delete("/model-configs/{config_id}")
async def delete_model_config(config_id: str):
    doc = await db.model_configs.find_one({"id": config_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Model config not found")
    if doc.get("is_builtin"):
        raise HTTPException(status_code=400, detail="Cannot delete built-in model")
    result = await db.model_configs.delete_one({"id": config_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Model config not found")
    return {"deleted": True}

# ── Stats ─────────────────────────────────────────────────────────────────────

@api_router.get("/stats")
async def get_stats():
    projects_count = await db.projects.count_documents({})
    runs_count = await db.runs.count_documents({})
    datasets_count = await db.datasets.count_documents({})
    models_count = await db.model_configs.count_documents({})
    
    # Recent runs (last 5)
    recent_runs = await db.runs.find({}, {"_id": 0}).sort("created_at", -1).to_list(5)
    
    # Task distribution
    pipeline = [{"$group": {"_id": "$task", "count": {"$sum": 1}}}]
    task_dist_raw = await db.runs.aggregate(pipeline).to_list(10)
    task_dist = {d["_id"]: d["count"] for d in task_dist_raw}
    
    return {
        "projects": projects_count,
        "runs": runs_count,
        "datasets": datasets_count,
        "models": models_count,
        "recent_runs": recent_runs,
        "task_distribution": task_dist
    }

@api_router.get("/")
async def root():
    return {"message": "YOLO26 CV Platform API v1.0", "docs": "/docs"}

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup: seed default model configs ──────────────────────────────────────



