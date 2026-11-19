# APMAN-Net v2.0: Adaptive Physics-Guided Multi-Stage Attention Network for Underwater Image Enhancement

APMAN-Net v2.0 is an underwater image enhancement system that integrates physical optical formation modeling with multi-stage neural attention.

## Repository Overview

- **backend/**: FastAPI web service, PyTorch model definitions, optical processing, and evaluation metrics.
- **frontend/**: Interactive web interface with before/after split slider, intermediate transmission visualization, and image quality metrics.
- **raw-890/**: Raw underwater images from the UIEB dataset.
- **reference-890/**: Corresponding ground truth reference images.
- **run.py**: Entry script to launch the application.

## Getting Started

### Prerequisites
- Python 3.10+
- PyTorch

### Installation
```bash
pip install -r backend/requirements.txt
```

### Running the Application
```bash
python run.py
```
Open `http://localhost:8000` in your browser.

## Key Features
- **Physics-Guided Dehazing**: Inverts underwater optical attenuation based on estimated medium transmission.
- **Multi-Stage Attention**: Uses spatial, channel, and edge-guided attention for detail preservation.
- **Interactive UI**: Split slider, transmission map inspection, and image quality metrics (UIQM, UCIQE).
- **Fast Execution**: Compact model architecture designed for real-time inference.
