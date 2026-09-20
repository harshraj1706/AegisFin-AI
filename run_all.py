"""
AegisFin-AI Phase 1 — Runner Utility
Allows launching the Streamlit Web Application and/or the FastAPI backend server.

Usage:
    # Run Streamlit Web Application (Default)
    python run_all.py

    # Run FastAPI Backend Server only
    python run_all.py --api

    # Run both FastAPI backend and Streamlit Web UI concurrently
    python run_all.py --all
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent


def run_streamlit():
    print("🚀 Launching AegisFin-AI Streamlit Web Application (http://localhost:8501)...")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(BASE_DIR / "streamlit_app.py"),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        "8501",
    ]
    subprocess.run(cmd, cwd=str(BASE_DIR))


def run_api():
    print("🌐 Launching AegisFin-AI FastAPI Backend Server (http://127.0.0.1:8000)...")
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--reload",
    ]
    subprocess.run(cmd, cwd=str(BASE_DIR))


def run_all():
    print("🌟 Starting both FastAPI Backend and Streamlit Frontend concurrently...")
    api_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    ui_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(BASE_DIR / "streamlit_app.py"),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        "8501",
    ]

    p_api = subprocess.Popen(api_cmd, cwd=str(BASE_DIR))
    time.sleep(2)
    p_ui = subprocess.Popen(ui_cmd, cwd=str(BASE_DIR))

    try:
        p_ui.wait()
    except KeyboardInterrupt:
        print("\nStopping services...")
    finally:
        p_ui.terminate()
        p_api.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AegisFin-AI Launcher")
    parser.add_argument("--api", action="store_true", help="Launch FastAPI backend only")
    parser.add_argument("--all", action="store_true", help="Launch both API and Streamlit UI")
    args = parser.parse_args()

    if args.all:
        run_all()
    elif args.api:
        run_api()
    else:
        run_streamlit()
