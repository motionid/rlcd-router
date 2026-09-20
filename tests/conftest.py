"""Test isolation for filesystem-backed router telemetry."""

import os
import shutil
import tempfile
from pathlib import Path


_TELEMETRY_DIR = Path(tempfile.mkdtemp(prefix="rlcd-router-tests-"))
os.environ["ROUTER_TELEMETRY_PATH"] = str(_TELEMETRY_DIR / "events.jsonl")


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TELEMETRY_DIR, ignore_errors=True)
