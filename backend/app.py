"""
FastAPI Server for APMAN-Net v2.0
Authors: Madala Leela Venkata Siva Spoorthi, Bhargav Gopi Anagani, Ramagiri Vardhan, Paturu Venkat Sujay Reddy
Guide: Dr. Anil Vitthalrao Turukmane (VIT-AP University)
"""

import os
import io
import re
import time
import base64
from pathlib import Path
from typing import Optional, List
from PIL import Image

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn

from inference import run_apman_enhancement, get_model, DEVICE
import metrics

# Base directories
ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"
RAW_DIR = ROOT_DIR / "raw-890"
REF_DIR = ROOT_DIR / "reference-890"

app = FastAPI(
    title="APMAN-Net v2.0 API",
    description="Adaptive Physics-Guided Multi-Stage Attention Network for Underwater Image Enhancement",
    version="2.0.0"
)

# Enable CORS for local dev or external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Curate matched pairs lookup
def get_matched_pairs():
    raw_files = [f for f in RAW_DIR.glob("*.*") if f.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    ref_files = [f for f in REF_DIR.glob("*.*") if f.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    
    raw_map = {int(re.findall(r"\d+", f.name)[-1]): f for f in raw_files if re.findall(r"\d+", f.name)}
    ref_map = {int(re.findall(r"\d+", f.name)[-1]): f for f in ref_files if re.findall(r"\d+", f.name)}
    
    pairs = {}
    for num_id in sorted(raw_map.keys()):
        if num_id in ref_map:
            pairs[num_id] = {
                "id": num_id,
                "raw_path": str(raw_map[num_id]),
                "ref_path": str(ref_map[num_id]),
                "filename": raw_map[num_id].name
            }
    return pairs

CACHED_PAIRS = get_matched_pairs()


def decode_base64_image(b64_str: str) -> Image.Image:
    """Helper to decode a base64 string or data URL into PIL Image."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


class EnhanceRequest(BaseModel):
    image: str  # Base64 string or data URL
    reference_image: Optional[str] = None  # Optional Base64 reference
    resolution: Optional[str] = "4k"
    red_gain: Optional[int] = 100
    dehaze_strength: Optional[int] = 85
    backscatter_suppression: Optional[bool] = True


@app.on_event("startup")
def startup_event():
    """Preload model weights into memory on server boot."""
    try:
        print("Preloading APMAN-Net model...")
        get_model()
        print("Model preloaded successfully!")
    except Exception as e:
        print(f"Model preload warning: {e}")


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "model": "APMAN-Net v2.0",
        "device": str(DEVICE),
        "available_pairs": len(CACHED_PAIRS)
    }


@app.get("/api/model-info")
def get_model_info():
    """Returns architecture overview, team credits, and benchmark metrics."""
    return {
        "project": {
            "title": "APMAN-Net v2.0",
            "full_name": "Adaptive Physics-Guided Multi-Stage Attention Network for Underwater Image Enhancement",
            "institution": "School of Computer Science and Engineering (SCOPE), VIT-AP University, Amaravati, India",
            "guide": "Dr. Anil Vitthalrao Turukmane (Head, Dept. of Security; Professor Grade 2, EPT & Networking)",
            "authors": [
                "Madala Leela Venkata Siva Spoorthi",
                "Bhargav Gopi Anagani",
                "Ramagiri Vardhan",
                "Paturu Venkat Sujay Reddy"
            ]
        },
        "specifications": {
            "parameters": "0.52 Million (522,963 parameters)",
            "architecture": "Physics-guided U-Net + MSFE + CBAM + ECA + Edge Attention + Confidence Fusion + Residual Refiner",
            "input_resolution": "256 x 256 x 3",
            "speed": "~41 FPS on RTX 3090, Real-time capable",
            "best_epoch": 48,
            "trained_epochs": 50,
            "best_train_psnr": "23.10 dB",
            "best_train_ssim": "0.9288",
            "reported_test_psnr": "21.53 dB",
            "reported_test_ssim": "0.9162"
        },
        "baselines_comparison": [
            {"method": "Water-Net (TIP 2020)", "psnr": 19.36, "ssim": 0.8309, "params": "0.78M", "fps": 18},
            {"method": "FUnIE-GAN (RA-L 2020)", "psnr": 20.12, "ssim": 0.8540, "params": "7.02M", "fps": 25},
            {"method": "UColor (TIP 2021)", "psnr": 20.89, "ssim": 0.8672, "params": "3.50M", "fps": 12},
            {"method": "DiffWater (JSTARS 2024)", "psnr": 21.36, "ssim": 0.8810, "params": "14.2M", "fps": 3},
            {"method": "APMAN-Net v2.0 (Ours)", "psnr": 21.53, "ssim": 0.9162, "params": "0.52M", "fps": 41, "highlight": True}
        ]
    }


@app.get("/api/samples")
def list_sample_images(limit: int = 14):
    """Returns a curated subset of representative test samples from UIEB dataset."""
    preferred_ids = [100, 101, 102, 104, 105, 106, 107, 108, 109, 11052, 11064, 11374, 1225, 9547, 9554, 9896]
    results = []
    
    for pid in preferred_ids:
        if pid in CACHED_PAIRS:
            results.append({
                "id": pid,
                "filename": CACHED_PAIRS[pid]["filename"],
                "has_reference": True
            })
            if len(results) >= limit:
                break
                
    if len(results) < limit:
        for pid, data in CACHED_PAIRS.items():
            if pid not in preferred_ids:
                results.append({
                    "id": pid,
                    "filename": data["filename"],
                    "has_reference": True
                })
                if len(results) >= limit:
                    break
                    
    return {"total_dataset_pairs": len(CACHED_PAIRS), "samples": results}


@app.get("/api/sample/{img_id}/raw")
def get_sample_raw(img_id: int):
    if img_id not in CACHED_PAIRS:
        raise HTTPException(status_code=404, detail="Sample ID not found")
    return FileResponse(CACHED_PAIRS[img_id]["raw_path"])


@app.get("/api/sample/{img_id}/reference")
def get_sample_reference(img_id: int):
    if img_id not in CACHED_PAIRS:
        raise HTTPException(status_code=404, detail="Sample ID not found")
    return FileResponse(CACHED_PAIRS[img_id]["ref_path"])


@app.post("/api/enhance")
def enhance_image(req: EnhanceRequest):
    """
    Enhances an uploaded underwater image using APMAN-Net v2.0 PyTorch model.
    """
    try:
        raw_pil = decode_base64_image(req.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid raw image data: {str(e)}")

    ref_pil = None
    if req.reference_image:
        try:
            ref_pil = decode_base64_image(req.reference_image)
        except Exception:
            ref_pil = None

    try:
        results = run_apman_enhancement(raw_pil, ref_pil=ref_pil)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/enhance-sample/{img_id}")
def enhance_dataset_sample(img_id: int):
    """
    Runs enhancement on a pre-existing UIEB dataset sample ID using APMAN-Net v2.0.
    """
    if img_id not in CACHED_PAIRS:
        raise HTTPException(status_code=404, detail="Sample ID not found")
    
    pair = CACHED_PAIRS[img_id]
    raw_pil = Image.open(pair["raw_path"]).convert("RGB")
    ref_pil = Image.open(pair["ref_path"]).convert("RGB")
    
    try:
        results = run_apman_enhancement(raw_pil, ref_pil=ref_pil)
        results["sample_id"] = img_id
        results["filename"] = pair["filename"]
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/evaluate-batch")
def evaluate_batch(count: int = Query(5, ge=1, le=20)):
    """
    Evaluates a batch of dataset samples and calculates average PSNR, SSIM, UCIQE, UIQM gains.
    """
    sample_ids = list(CACHED_PAIRS.keys())[:count]
    records = []
    
    for sid in sample_ids:
        pair = CACHED_PAIRS[sid]
        raw_pil = Image.open(pair["raw_path"]).convert("RGB")
        ref_pil = Image.open(pair["ref_path"]).convert("RGB")
        res = run_apman_enhancement(raw_pil, ref_pil=ref_pil)
        m = res["metrics"]
        records.append({
            "id": sid,
            "filename": pair["filename"],
            "raw_psnr": m["raw_psnr"],
            "enhanced_psnr": m["enhanced_psnr"],
            "psnr_gain": m["psnr_gain"],
            "raw_ssim": m["raw_ssim"],
            "enhanced_ssim": m["enhanced_ssim"],
            "ssim_gain": m["ssim_gain"],
            "raw_uciqe": m["raw_uciqe"],
            "enhanced_uciqe": m["enhanced_uciqe"],
            "uciqe_gain": m["uciqe_gain"],
            "raw_uiqm": m["raw_uiqm"],
            "enhanced_uiqm": m["enhanced_uiqm"],
            "uiqm_gain": m["uiqm_gain"],
            "time_ms": res["performance"]["inference_time_ms"]
        })
        
    avg_psnr_gain = round(sum(r["psnr_gain"] for r in records) / len(records), 2)
    avg_ssim_gain = round(sum(r["ssim_gain"] for r in records) / len(records), 4)
    avg_uciqe_gain = round(sum(r["uciqe_gain"] for r in records) / len(records), 4)
    avg_uiqm_gain = round(sum(r["uiqm_gain"] for r in records) / len(records), 4)
    avg_time = round(sum(r["time_ms"] for r in records) / len(records), 2)
    
    return {
        "count": len(records),
        "averages": {
            "avg_psnr_gain_db": avg_psnr_gain,
            "avg_ssim_gain": avg_ssim_gain,
            "avg_uciqe_gain": avg_uciqe_gain,
            "avg_uiqm_gain": avg_uiqm_gain,
            "avg_inference_time_ms": avg_time,
            "avg_fps": round(1000.0 / max(avg_time, 1e-4), 1)
        },
        "records": records
    }


# Mount frontend static directory if exists
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
