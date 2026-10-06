"""Small job based API for the Cycling Biomechanics Analyzer.

The service intentionally keeps job state in memory for the portfolio MVP.  A
production deployment can replace ``JOBS`` with a durable queue without
changing the HTTP contract.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict


ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", ROOT / "runtime" / "uploads"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", ROOT / "runtime" / "outputs"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


class JobResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    analysis_id: str
    status: str
    stage: str | None = None
    error: str | None = None
    created_at: str
    updated_at: str
    result_url: str | None = None


JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _update(job_id: str, **values: Any) -> None:
    with _LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(values, updated_at=_now())
            # Keep a small recovery/debug record alongside the job artifacts.
            # The in-memory map remains the source for active requests.
            status_file = JOBS[job_id].get("status_file")
            if status_file:
                snapshot = {k: _jsonable(v) for k, v in JOBS[job_id].items() if k != "status_file"}
                Path(status_file).parent.mkdir(parents=True, exist_ok=True)
                Path(status_file).write_text(json.dumps(snapshot, indent=2), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    """Convert numpy/path values from analysis packages to JSON primitives."""
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _run_analysis(job_id: str, video_path: Path, output_path: Path) -> None:
    try:
        _update(job_id, status="processing", stage="extracting_2d_pose")
        from pipeline.video_pipeline import analyze_video

        def status_callback(stage: str, *_args: Any, **_kwargs: Any) -> None:
            # Accommodate callbacks that report either a stage name or a
            # status/message pair.
            label = str(stage).lower().replace(" ", "_")
            _update(job_id, stage=label)

        result = analyze_video(str(video_path), output_dir=str(output_path), status_callback=status_callback)
        result = _jsonable(result or {})
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / "results.json"
        result_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
        _update(job_id, status="completed", stage="completed", result=result, result_url=f"/api/analyses/{job_id}/results")
    except Exception as exc:  # surfaced through the status endpoint
        _update(job_id, status="failed", stage="failed", error=f"{type(exc).__name__}: {exc}")


def _run_pose_analysis(job_id: str, pose_path: Path, output_path: Path, fps: float) -> None:
    try:
        _update(job_id, status="processing", stage="analyzing")
        from cycling_analysis.pipeline import analyze_pose
        result = analyze_pose(pose_path, fps=fps, output_dir=output_path)
        _update(job_id, status="completed", stage="completed", result=result,
                result_url=f"/api/analyses/{job_id}/results")
    except Exception as exc:
        _update(job_id, status="failed", stage="failed", error=f"{type(exc).__name__}: {exc}")


app = FastAPI(title="Cycling Biomechanics Analyzer", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "cycling-biomechanics-analyzer"}


@app.post("/api/analyses", response_model=JobResponse, status_code=202)
async def create_analysis(background_tasks: BackgroundTasks, video: UploadFile = File(...)) -> JobResponse:
    filename = Path(video.filename or "video").name
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Upload a supported video file (.mp4, .mov, .avi, .mkv, or .webm).")
    job_id = uuid.uuid4().hex
    job_upload = UPLOAD_DIR / job_id
    job_output = OUTPUT_DIR / job_id
    job_upload.mkdir(parents=True)
    destination = job_upload / filename
    with destination.open("wb") as handle:
        shutil.copyfileobj(video.file, handle)
    if destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded video is empty.")
    now = _now()
    record = {
        "analysis_id": job_id,
        "status": "queued",
        "stage": "queued",
        "error": None,
        "created_at": now,
        "updated_at": now,
        "result_url": None,
        "status_file": str(job_output / "status.json"),
    }
    job_output.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        JOBS[job_id] = record
        (job_output / "status.json").write_text(json.dumps({k: v for k, v in record.items() if k != "status_file"}, indent=2), encoding="utf-8")
    background_tasks.add_task(_run_analysis, job_id, destination, job_output)
    return JobResponse(**record)


@app.post("/api/poses", response_model=JobResponse, status_code=202)
async def create_pose_analysis(background_tasks: BackgroundTasks, pose: UploadFile = File(...), fps: float = Form(60.0)) -> JobResponse:
    """Analyze a saved MotionBERT pose without the legacy inference stacks."""
    if Path(pose.filename or "").suffix.lower() != ".npy":
        raise HTTPException(status_code=415, detail="Upload a MotionBERT X3D.npy file")
    if not 0 < fps <= 240:
        raise HTTPException(status_code=422, detail="FPS must be between 0 and 240")
    job_id = uuid.uuid4().hex
    job_upload = UPLOAD_DIR / job_id
    job_output = OUTPUT_DIR / job_id
    job_upload.mkdir(parents=True)
    job_output.mkdir(parents=True)
    destination = job_upload / "X3D.npy"
    with destination.open("wb") as handle:
        shutil.copyfileobj(pose.file, handle)
    if destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded pose is empty")
    now = _now()
    record = {"analysis_id": job_id, "status": "queued", "stage": "queued", "error": None,
              "created_at": now, "updated_at": now, "result_url": None,
              "status_file": str(job_output / "status.json")}
    with _LOCK:
        JOBS[job_id] = record
        (job_output / "status.json").write_text(json.dumps({k: v for k, v in record.items() if k != "status_file"}, indent=2))
    background_tasks.add_task(_run_pose_analysis, job_id, destination, job_output, fps)
    return JobResponse(**record)


@app.get("/api/analyses/{analysis_id}", response_model=JobResponse)
def get_analysis(analysis_id: str) -> JobResponse:
    with _LOCK:
        job = JOBS.get(analysis_id)
        snapshot = dict(job) if job else None
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Analysis was not found")
    return JobResponse(**snapshot)


@app.get("/api/analyses/{analysis_id}/results")
def get_results(analysis_id: str) -> dict[str, Any]:
    with _LOCK:
        job = JOBS.get(analysis_id)
        snapshot = dict(job) if job else None
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Analysis was not found")
    if snapshot["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"Analysis is {snapshot['status']}")
    result_file = OUTPUT_DIR / analysis_id / "results.json"
    if result_file.exists():
        return json.loads(result_file.read_text(encoding="utf-8"))
    return snapshot.get("result", {})


@app.get("/api/analyses/{analysis_id}/charts/{chart_path:path}")
def get_chart(analysis_id: str, chart_path: str) -> FileResponse:
    root = (OUTPUT_DIR / analysis_id).resolve()
    path = (root / chart_path).resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Chart was not found")
    return FileResponse(path)
