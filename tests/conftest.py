from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("ASAN_PROVIDER", "localfree")
os.environ.setdefault("ASAN_LOCAL_LLM_BACKEND", "heuristic")
os.environ.setdefault("ASAN_REPOSITORY_BACKEND", "memory")


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def client():
    from app.config import get_settings
    from app.main import create_app

    evidence_root = Path(tempfile.mkdtemp(prefix="asanappeal-evidence-"))
    privacy_export_root = Path(tempfile.mkdtemp(prefix="asanappeal-privacy-export-"))
    os.environ["ASAN_EVIDENCE_ROOT"] = str(evidence_root)
    os.environ["ASAN_EVIDENCE_SIGNING_SECRET"] = "test-evidence-signing-secret"
    os.environ["ASAN_PRIVACY_EXPORT_DIR"] = str(privacy_export_root)
    get_settings.cache_clear()
    with TestClient(create_app()) as client:
        login = client.post(
            "/v1/auth/login",
            json={
                "username": os.getenv("ASAN_AUTH_DEMO_ADMIN_USERNAME", "admin.demo"),
                "password": os.getenv("ASAN_AUTH_DEMO_ADMIN_PASSWORD", "admin-demo-pass"),
            },
        )
        assert login.status_code == 200
        client.headers.update(
            {"Authorization": f"Bearer {login.json()['access_token']}"}
        )
        yield client
    shutil.rmtree(evidence_root, ignore_errors=True)
    shutil.rmtree(privacy_export_root, ignore_errors=True)
    get_settings.cache_clear()
