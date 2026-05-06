"""
REST API for querying available dataset labels and sign videos.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from loguru import logger
from config import DATASET_DIR
from modules import gru_model

router = APIRouter(prefix="/api", tags=["dataset"])


@router.get("/labels")
def get_labels():
    """Return all available sign labels."""
    labels = gru_model.get_labels()
    return {"count": len(labels), "labels": labels}


@router.get("/signs")
def list_signs():
    """List all classes that have a video.mp4 in the dataset."""
    if not DATASET_DIR.exists():
        return {"signs": []}
    signs = []
    for class_dir in sorted(DATASET_DIR.iterdir()):
        if class_dir.is_dir():
            video = class_dir / "video.mp4"
            kp = class_dir / "keypoints.npy"
            signs.append({
                "label": class_dir.name,
                "has_video": video.exists(),
                "has_keypoints": kp.exists(),
                "video_url": f"/static/signs/{class_dir.name}/video.mp4" if video.exists() else None,
            })
    return {"count": len(signs), "signs": signs}


@router.get("/signs/{label}/video")
def get_sign_video(label: str):
    """Stream a sign language video for a given label."""
    video_path = DATASET_DIR / label / "video.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found for label '{label}'")
    return FileResponse(str(video_path), media_type="video/mp4")


@router.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": gru_model.is_loaded(),
        "dataset_dir": str(DATASET_DIR),
        "labels_count": len(gru_model.get_labels()),
    }
