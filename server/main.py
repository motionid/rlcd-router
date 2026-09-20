"""Persistent entry point for the local RLCD router service."""

import os
import sys

import uvicorn

# Ensure project root is in PYTHONPATH when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


if __name__ == "__main__":
    uvicorn.run(
        "server.app:app",
        host=os.environ.get("RLCD_ROUTER_HOST", "127.0.0.1"),
        port=int(os.environ.get("RLCD_ROUTER_PORT", "8001")),
        log_level=os.environ.get("RLCD_LOG_LEVEL", "info"),
    )
