"""
APMAN-Net v2.0 - One-Click Full-Stack Launcher
Starts the FastAPI backend and serves the interactive frontend at http://localhost:8000
"""

import sys
import os
import webbrowser
from pathlib import Path

# Add backend directory to sys.path
BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import uvicorn

def main():
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 70)
    print("APMAN-Net v2.0: Adaptive Physics-Guided Multi-Stage Attention Network")
    print("Underwater Image Enhancement | Full-Stack Application")
    print("=" * 70)
    print("Institution   : School of Computer Science and Engineering (SCOPE), VIT-AP")
    print("Project Guide : Dr. Anil Vitthalrao Turukmane")
    print("Team Members  : Madala Leela Venkata Siva Spoorthi, Bhargav Gopi Anagani,")
    print("                Ramagiri Vardhan, Paturu Venkat Sujay Reddy")
    print("-" * 70)
    print("Starting FastAPI Server...")
    print("Web Application URL : http://localhost:8000")
    print("Swagger API Docs    : http://localhost:8000/docs")
    print("-" * 70)
    print("Press CTRL+C to stop the server.\n")

    # Start uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, app_dir=str(BACKEND_DIR))

if __name__ == "__main__":
    main()
