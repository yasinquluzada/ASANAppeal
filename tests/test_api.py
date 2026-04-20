from io import BytesIO
import os
from pathlib import Path
from urllib.parse import urlsplit
import zipfile

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw


def authenticate_test_client(
    client: TestClient,
    *,
    username: str | None = None,
    password: str | None = None,
) -> None:
    login = client.post(
        "/v1/auth/login",
        json={
            "username": username or os.getenv("ASAN_AUTH_DEMO_ADMIN_USERNAME", "admin.demo"),
            "password": password or os.getenv("ASAN_AUTH_DEMO_ADMIN_PASSWORD", "admin-demo-pass"),
        },
    )
    assert login.status_code == 200
    client.headers.update({"Authorization": f"Bearer {login.json()['access_token']}"})


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["repository_backend"] == "memory"
    assert response.json().get("degraded_dependencies") is None


def test_portal_route_renders_citizen_and_operator_sections(client: TestClient) -> None:
    response = client.get("/app")

    assert response.status_code == 200
    assert "Citizen Intake UI" in response.text
    assert "Reviewer Desk" in response.text
    assert "Analytics UI" in response.text
    assert "Institution Response UI" in response.text


def test_portal_static_assets_are_served(client: TestClient) -> None:
    response = client.get("/app-assets/app.js")

    assert response.status_code == 200
    assert "refreshReviewQueue" in response.text
    assert "submitCitizenForm" in response.text


def test_api_routes_require_authentication() -> None:
    from app.main import create_app

    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/cases/process",
            json={"submission": {"citizen_text": "Broken pavement", "location_hint": "Block 7"}},
        )

    assert response.status_code == 401
    assert "Authentication is required" in response.json()["detail"]


def test_rate_limiting_blocks_excess_requests() -> None:
    from app.config import Settings
    from app.main import create_app

    settings = Settings(
        repository_backend="memory",
        request_rate_limit_max_requests=2,
        request_rate_limit_window_seconds=60,
    )

    with TestClient(create_app(settings)) as client:
        first = client.get("/health")
        second = client.get("/health")
        third = client.get("/health")

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    assert "Rate limit exceeded" in third.json()["detail"]


def test_payload_size_cap_rejects_large_request() -> None:
    from app.config import Settings
    from app.main import create_app

    settings = Settings(
        repository_backend="memory",
        request_max_body_bytes=256,
    )

    with TestClient(create_app(settings)) as client:
        authenticate_test_client(client)
        response = client.post(
            "/v1/cases/process",
            json={
                "submission": {
                    "citizen_text": "x" * 500,
                    "location_hint": "Oversized payload check",
                }
            },
        )

    assert response.status_code == 413
    assert "payload limit" in response.json()["detail"].lower()


def test_citizen_accounts_can_register_and_only_access_their_own_cases() -> None:
    from app.main import create_app

    with TestClient(create_app()) as client:
        first_register = client.post(
            "/v1/auth/register",
            json={
                "username": "citizen.alpha",
                "password": "citizen-alpha-pass",
                "display_name": "Citizen Alpha",
            },
        )
        second_register = client.post(
            "/v1/auth/register",
            json={
                "username": "citizen.beta",
                "password": "citizen-beta-pass",
                "display_name": "Citizen Beta",
            },
        )
        assert first_register.status_code == 200
        assert second_register.status_code == 200

        client.headers.update({"Authorization": f"Bearer {first_register.json()['access_token']}"})
        alpha_case = client.post(
            "/v1/cases/process",
            json={
                "submission": {
                    "citizen_text": "Broken street light near block A.",
                    "location_hint": "Block A",
                }
            },
        )
        assert alpha_case.status_code == 200

        client.headers.update({"Authorization": f"Bearer {second_register.json()['access_token']}"})
        denied = client.get(f"/v1/cases/{alpha_case.json()['case']['case_id']}")

    assert denied.status_code == 403
    assert "only access their own cases" in denied.json()["detail"]


def test_institution_accounts_are_scoped_to_their_routed_institution(client: TestClient) -> None:
    road_case = client.post(
        "/v1/cases/process",
        json={
            "submission": {
                "citizen_text": "Large pothole on the service road by the hospital.",
                "location_hint": "Hospital service road",
            }
        },
    )
    lighting_case = client.post(
        "/v1/cases/process",
        json={
            "submission": {
                "citizen_text": "Street light outage outside the pharmacy.",
                "location_hint": "Pharmacy corner",
            }
        },
    )

    authenticate_test_client(
        client,
        username=os.getenv("ASAN_AUTH_DEMO_INSTITUTION_USERNAME", "institution.roads"),
        password=os.getenv("ASAN_AUTH_DEMO_INSTITUTION_PASSWORD", "institution-demo-pass"),
    )
    allowed = client.get(f"/v1/cases/{road_case.json()['case']['case_id']}")
    denied = client.get(f"/v1/cases/{lighting_case.json()['case']['case_id']}")

    assert allowed.status_code == 200
    assert denied.status_code == 403
    assert "routed to their institution" in denied.json()["detail"]


def test_citizen_accounts_cannot_open_review_queue(client: TestClient) -> None:
    register = client.post(
        "/v1/auth/register",
        json={
            "username": "citizen.queue",
            "password": "citizen-queue-pass",
            "display_name": "Citizen Queue",
        },
    )
    assert register.status_code == 200
    client.headers.update({"Authorization": f"Bearer {register.json()['access_token']}"})

    response = client.get("/v1/review-queue")

    assert response.status_code == 403
    assert "does not have permission" in response.json()["detail"]


def test_case_audit_log_tracks_ai_outputs_and_human_overrides(client: TestClient) -> None:
    created = client.post(
        "/v1/cases/process",
        json={
            "submission": {
                "citizen_text": "Large pothole beside the school gate causing drivers to swerve.",
                "location_hint": "School gate road",
            }
        },
    )
    case_id = created.json()["case"]["case_id"]

    approved = client.post(
        f"/v1/cases/{case_id}/workflow-actions",
        json={
            "action": "approve",
            "actor_id": "reviewer-77",
            "note": "Reviewed and approved for dispatch.",
        },
    )
    audit_log = client.get(f"/v1/cases/{case_id}/audit-log")
    audit_verification = client.get("/v1/audit/verify")

    assert approved.status_code == 200
    assert audit_log.status_code == 200
    assert audit_verification.status_code == 200
    items = audit_log.json()["items"]
    assert items[0]["event_type"] == "case_created"
    assert "ai_outputs" in items[0]["payload"]
    assert items[1]["event_type"] == "workflow_action"
    assert items[1]["human_override"] is True
    assert "human_override" in items[1]["payload"]
    assert items[1]["payload"]["human_override"]["workflow_entry"]["action"] == "approve"
    assert audit_verification.json()["verification"]["verified"] is True


def test_abuse_protection_blocks_excess_evidence_and_spam_content(client: TestClient) -> None:
    too_many_evidence = client.post(
        "/v1/cases/process",
        json={
            "submission": {
                "citizen_text": "Road collapse beside the service lane.",
                "location_hint": "Service lane",
                "evidence": [{"kind": "image", "description": f"photo {index}"} for index in range(9)],
            }
        },
    )
    spam_content = client.post(
        "/v1/cases/process",
        json={
            "submission": {
                "citizen_text": "Buy now, join my telegram channel, huge discount, crypto bonus.",
                "location_hint": "N/A",
            }
        },
    )

    assert too_many_evidence.status_code == 400
    assert "maximum is" in too_many_evidence.json()["detail"]
    assert spam_content.status_code == 400
    assert "spam-like" in spam_content.json()["detail"]


def test_duplicate_submission_spam_detection_blocks_repeat_flood() -> None:
    from app.config import Settings
    from app.main import create_app

    settings = Settings(
        repository_backend="memory",
        abuse_duplicate_threshold=2,
        abuse_spam_window_seconds=3600,
    )

    with TestClient(create_app(settings)) as client:
        authenticate_test_client(client)
        payload = {
            "submission": {
                "citizen_text": "Pothole beside school gate causing traffic swerves.",
                "location_hint": "School gate service road",
            }
        }
        first = client.post("/v1/cases/process", json=payload)
        second = client.post("/v1/cases/process", json=payload)
        third = client.post("/v1/cases/process", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    assert "duplicate submissions" in third.json()["detail"].lower()

def test_end_to_end_case_processing(client: TestClient) -> None:
    payload = {
        "submission": {
            "citizen_text": (
                "There is a deep pothole on the main road near the school entrance. "
                "Cars are swerving to avoid it at night."
            ),
            "location_hint": "Main road near School No. 12, Baku",
            "time_hint": "2026-04-18 09:00",
            "evidence": [
                {"kind": "image", "filename": "pothole-1.jpg", "description": "deep road pothole"},
                {"kind": "image", "filename": "pothole-2.jpg", "description": "cars avoiding the hole"},
            ],
        }
    }
    response = client.post("/v1/cases/process", json=payload)
    assert response.status_code == 200
    case = response.json()["case"]
    assert case["structured_issue"]["category"] == "road_damage"
    assert case["routing"]["institution"] == "ASAN Road Maintenance Agency"
    assert case["priority"]["level"] in {"high", "critical"}
    assert "Road Damage" in case["draft"]["title"]
    assert case["model_context"]["provenance_schema_version"] == "case-provenance.v1"
    assert case["model_context"]["prompt_bundle_version"] != "not_applicable"
    assert "intake" in case["model_context"]["stage_provenance"]
    assert "review" in case["model_context"]["stage_provenance"]
    assert response.json()["original_request"]["submission"]["location_hint"] != payload["submission"]["location_hint"]
    assert response.json()["original_request"]["submission"]["evidence"][0]["privacy"]["retention_delete_after"] is not None


def test_case_detail_exposes_original_request_payload(client: TestClient) -> None:
    payload = {
        "submission": {
            "citizen_text": "Broken street light near the pharmacy.",
            "language": "en",
            "location_hint": "Nizami Street near the pharmacy",
            "time_hint": "2026-04-18 20:15",
            "evidence": [{"kind": "image", "filename": "lamp.jpg", "description": "dark street lamp"}],
        }
    }
    created = client.post("/v1/cases/process", json=payload)
    case_id = created.json()["case"]["case_id"]

    detail = client.get(f"/v1/cases/{case_id}")
    original_request = client.get(f"/v1/cases/{case_id}/original-request")

    assert detail.status_code == 200
    assert detail.json()["original_request"]["submission"]["language"] == payload["submission"]["language"]
    assert detail.json()["original_request"]["submission"]["location_hint"] == payload["submission"]["location_hint"]
    assert detail.json()["original_request"]["submission"]["citizen_text"] != payload["submission"]["citizen_text"]
    assert original_request.status_code == 200
    assert original_request.json()["original_request"]["submission"]["language"] == payload["submission"]["language"]


def test_evidence_upload_round_trip_supports_signed_download_and_thumbnail(
    client: TestClient,
) -> None:
    buffer = BytesIO()
    Image.new("RGB", (64, 48), color=(220, 40, 40)).save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    upload = client.post(
        "/v1/evidence/upload",
        params={
            "kind": "image",
            "filename": "pothole.png",
            "description": "Citizen uploaded pothole photo.",
            "metadata_json": '{"camera":"mobile"}',
        },
        content=image_bytes,
        headers={"content-type": "image/png"},
    )

    assert upload.status_code == 200
    payload = upload.json()
    evidence_id = payload["evidence"]["evidence_id"]
    assert payload["evidence"]["mime_type"] == "image/png"
    assert payload["evidence"]["size_bytes"] == len(image_bytes)
    assert payload["evidence"]["sha256"]
    assert payload["evidence_item"]["evidence_id"] == evidence_id
    assert payload["thumbnail_url"] is not None

    metadata = client.get(f"/v1/evidence/{evidence_id}")
    assert metadata.status_code == 200
    assert metadata.json()["evidence"]["evidence_id"] == evidence_id
    assert metadata.json()["evidence_item"]["thumbnail_available"] is True

    download_url = urlsplit(payload["download_url"])
    downloaded = client.get(f"{download_url.path}?{download_url.query}")
    assert downloaded.status_code == 200
    assert downloaded.content == image_bytes
    assert downloaded.headers["content-type"] == "image/png"

    thumbnail_url = urlsplit(str(payload["thumbnail_url"]))
    thumbnail = client.get(f"{thumbnail_url.path}?{thumbnail_url.query}")
    assert thumbnail.status_code == 200
    assert thumbnail.headers["content-type"] == "image/png"
    assert len(thumbnail.content) > 0


def test_evidence_upload_rejects_invalid_image_payload(client: TestClient) -> None:
    response = client.post(
        "/v1/evidence/upload",
        params={"kind": "image", "filename": "not-an-image.png"},
        content=b"not really an image",
        headers={"content-type": "image/png"},
    )

    assert response.status_code == 400
    assert "could not be decoded" in response.json()["detail"]


def test_evidence_download_rejects_tampered_signature(client: TestClient) -> None:
    response = client.post(
        "/v1/evidence/upload",
        params={"kind": "text", "filename": "note.txt"},
        content=b"Citizen note about damaged road barrier.",
        headers={"content-type": "text/plain"},
    )
    download_url = response.json()["download_url"]
    parsed = urlsplit(download_url)
    tampered = parsed.query.replace("signature=", "signature=bad")

    denied = client.get(f"{parsed.path}?{tampered}")

    assert denied.status_code == 403
    assert "signature" in denied.json()["detail"].lower()


def test_text_evidence_upload_redacts_pii_and_sets_retention(client: TestClient) -> None:
    upload = client.post(
        "/v1/evidence/upload",
        params={"kind": "text", "filename": "complaint.txt"},
        content=(
            "Call me at +994 50 123 45 67 or email citizen@example.com. "
            "Apartment 12, Building 8, Nizami Street 42."
        ).encode("utf-8"),
        headers={"content-type": "text/plain"},
    )

    assert upload.status_code == 200
    payload = upload.json()
    assert payload["evidence"]["privacy"]["redaction_applied"] is True
    assert payload["evidence"]["privacy"]["pii_detected"] is True
    assert payload["evidence"]["privacy"]["retention_delete_after"] is not None

    download_url = urlsplit(payload["download_url"])
    downloaded = client.get(f"{download_url.path}?{download_url.query}")
    assert downloaded.status_code == 200
    downloaded_text = downloaded.text
    assert "[redacted-phone]" in downloaded_text
    assert "[redacted-email]" in downloaded_text
    assert "Apartment 12" not in downloaded_text


def test_image_evidence_upload_masks_face_and_plate_candidates(client: TestClient) -> None:
    buffer = BytesIO()
    image = Image.new("RGB", (240, 140), color=(70, 70, 70))
    draw = ImageDraw.Draw(image)
    draw.ellipse((36, 28, 96, 102), fill=(214, 168, 132))
    draw.rectangle((150, 92, 216, 114), fill=(235, 235, 228))
    image.save(buffer, format="PNG")
    original_bytes = buffer.getvalue()

    upload = client.post(
        "/v1/evidence/upload",
        params={"kind": "image", "filename": "scene.png"},
        content=original_bytes,
        headers={"content-type": "image/png"},
    )

    assert upload.status_code == 200
    payload = upload.json()
    assert payload["evidence"]["privacy"]["redaction_applied"] is True
    assert payload["evidence"]["privacy"]["pii_detected"] is True
    pii_types = {region["pii_type"] for region in payload["evidence"]["privacy"]["image_regions"]}
    assert "face_candidate" in pii_types or "license_plate_candidate" in pii_types

    download_url = urlsplit(payload["download_url"])
    downloaded = client.get(f"{download_url.path}?{download_url.query}")
    assert downloaded.status_code == 200
    assert downloaded.content != original_bytes


def test_zero_key_image_reasoning_uses_uploaded_image_bytes(client: TestClient) -> None:
    buffer = BytesIO()
    image = Image.new("RGB", (180, 110), color=(118, 118, 118))
    draw = ImageDraw.Draw(image)
    draw.ellipse((58, 28, 126, 82), fill=(18, 18, 18))
    draw.line((20, 72, 70, 40), fill=(60, 60, 60), width=4)
    draw.line((110, 34, 156, 74), fill=(52, 52, 52), width=4)
    image.save(buffer, format="PNG")

    upload = client.post(
        "/v1/evidence/upload",
        params={"kind": "image", "filename": "scene.png"},
        content=buffer.getvalue(),
        headers={"content-type": "image/png"},
    )
    assert upload.status_code == 200
    uploaded_item = upload.json()["evidence_item"]

    response = client.post(
        "/v1/cases/process",
        json={
            "submission": {
                "citizen_text": "Please inspect this problem.",
                "language": "en",
                "location_hint": "Zone 4 block 17",
                "evidence": [uploaded_item],
            }
        },
    )

    assert response.status_code == 200
    case = response.json()["case"]
    assert case["structured_issue"]["category"] == "road_damage"
    assert "damaged_surface_candidate" in case["structured_issue"]["extracted_signals"]
    assert case["routing"]["institution"] == "ASAN Road Maintenance Agency"


def test_case_processing_redacts_original_request_and_exposes_privacy_summary(client: TestClient) -> None:
    payload = {
        "submission": {
            "citizen_text": (
                "My phone is +994 50 123 45 67 and my email is person@example.com. "
                "There is a pothole outside Apartment 12, Building 8."
            ),
            "language": "en",
            "location_hint": "Nizami Street 42, Apartment 12, Baku",
            "evidence": [],
        }
    }

    created = client.post("/v1/cases/process", json=payload)
    assert created.status_code == 200
    case_id = created.json()["case"]["case_id"]

    assert "[redacted-phone]" in created.json()["original_request"]["submission"]["citizen_text"]
    assert "[redacted-email]" in created.json()["original_request"]["submission"]["citizen_text"]
    assert "42" not in (created.json()["original_request"]["submission"]["location_hint"] or "")
    assert created.json()["case"]["privacy"]["redaction_applied"] is True
    assert created.json()["case"]["privacy"]["pii_detected"] is True

    privacy = client.get(f"/v1/cases/{case_id}/privacy")
    audit_log = client.get(f"/v1/cases/{case_id}/audit-log")

    assert privacy.status_code == 200
    assert len(privacy.json()["findings"]) >= 2
    original_request = audit_log.json()["items"][0]["payload"]["original_request"]
    assert "[redacted-phone]" in original_request["submission"]["citizen_text"]
    assert "[redacted-email]" in original_request["submission"]["citizen_text"]


def test_privacy_export_and_delete_workflows(client: TestClient) -> None:
    note_upload = client.post(
        "/v1/evidence/upload",
        params={"kind": "text", "filename": "note.txt"},
        content=b"Citizen note with +994 50 123 45 67 and citizen@example.com",
        headers={"content-type": "text/plain"},
    )
    evidence_item = note_upload.json()["evidence_item"]
    evidence_id = note_upload.json()["evidence"]["evidence_id"]

    created = client.post(
        "/v1/cases/process",
        json={
            "submission": {
                "citizen_text": "Pothole near the school gate.",
                "location_hint": "School gate service road",
                "evidence": [evidence_item],
            }
        },
    )
    case_id = created.json()["case"]["case_id"]

    exported = client.post(f"/v1/cases/{case_id}/privacy-export")
    assert exported.status_code == 200
    archive_path = Path(exported.json()["export_path"])
    assert archive_path.exists()
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        assert "case.json" in names
        assert "original_request.json" in names
        assert "audit_log.json" in names

    deleted = client.post(
        f"/v1/cases/{case_id}/privacy-delete",
        json={"note": "Citizen requested erasure."},
    )
    assert deleted.status_code == 200
    assert deleted.json()["deleted_evidence_count"] >= 1
    assert deleted.json()["case"]["operations"]["final_disposition"] == "privacy_deleted"
    assert deleted.json()["case"]["privacy"]["deleted_at"] is not None
    assert deleted.json()["original_request"]["privacy_deleted_at"] is not None

    evidence_metadata = client.get(f"/v1/evidence/{evidence_id}")
    assert evidence_metadata.status_code == 404


def test_privacy_retention_enforcement_deletes_expired_case(tmp_path) -> None:
    from app.config import Settings
    from app.main import create_app

    settings = Settings(
        provider="localfree",
        repository_backend="memory",
        evidence_root=str(tmp_path / "evidence"),
        privacy_export_dir=str(tmp_path / "privacy_exports"),
        privacy_case_retention_days=0,
        privacy_evidence_retention_days=0,
    )

    with TestClient(create_app(settings)) as client:
        authenticate_test_client(client)
        created = client.post(
            "/v1/cases/process",
            json={
                "submission": {
                    "citizen_text": "Urgent pothole beside apartment block.",
                    "location_hint": "Apartment block 12",
                }
            },
        )
        case_id = created.json()["case"]["case_id"]
        retained = client.post("/v1/privacy/retention/enforce")
        case_after = client.get(f"/v1/cases/{case_id}")

    assert retained.status_code == 200
    assert retained.json()["cases_privacy_deleted"] >= 1
    assert case_id in retained.json()["affected_case_ids"]
    assert case_after.status_code == 200
    assert case_after.json()["case"]["operations"]["final_disposition"] == "privacy_deleted"


def test_observability_metrics_and_request_headers_are_exposed(client: TestClient) -> None:
    health = client.get("/health")
    created = client.post(
        "/v1/cases/process",
        json={
            "submission": {
                "citizen_text": "Broken lamp and damaged pavement near clinic.",
                "location_hint": "Clinic road",
            }
        },
    )
    metrics = client.get("/v1/observability/metrics")

    assert health.status_code == 200
    assert health.headers["X-Request-ID"]
    assert health.headers["X-Trace-ID"]
    assert created.status_code == 200
    assert metrics.status_code == 200
    metrics_payload = metrics.json()
    assert metrics_payload["request"]["total"] >= 3
    assert "/v1/cases/process" in metrics_payload["request"]["paths"]
    assert metrics_payload["provider"]["items"]["LocalFreeProvider:intake"]["calls"] >= 1
    assert "review_turnaround_avg_hours" in metrics_payload["reviews"]


def test_uncertain_case_enters_review_queue(client: TestClient) -> None:
    payload = {
        "submission": {
            "citizen_text": "Problem.",
            "evidence": [],
        }
    }
    process_response = client.post("/v1/cases/process", json=payload)
    assert process_response.status_code == 200
    assert process_response.json()["case"]["human_review"]["needed"] is True
    assert process_response.json()["case"]["human_review"]["queue"] in {
        "evidence-quality-review",
        "triage-review",
    }

    queue_response = client.get("/v1/review-queue")
    assert queue_response.status_code == 200
    assert len(queue_response.json()["items"]) >= 1
    assert queue_response.json()["meta"]["sort_by"] == "sla"


def test_analytics_summary_is_available_for_memory_backend(client: TestClient) -> None:
    client.post(
        "/v1/cases/process",
        json={
            "submission": {
                "citizen_text": "Broken street light near the pharmacy.",
                "location_hint": "Nizami Street near the pharmacy",
            }
        },
    )

    response = client.get("/v1/analytics/summary")

    assert response.status_code == 200
    assert response.json()["summary"]["total_cases"] >= 1
    assert "street_lighting" in response.json()["summary"]["counts_by_category"]


def test_local_ml_retrain_endpoint_exports_feedback_and_reloads_models(tmp_path) -> None:
    from app.config import Settings
    from app.main import create_app

    settings = Settings(
        provider="localfree",
        repository_backend="sqlite",
        sqlite_path=str(tmp_path / "retrain.db"),
        local_ml_enabled=True,
        local_ml_include_sqlite_feedback=False,
        local_ml_feedback_dir=str(tmp_path / "feedback_exports"),
    )

    with TestClient(create_app(settings)) as client:
        authenticate_test_client(client)
        initial_health = client.get("/health").json()
        created = client.post(
            "/v1/cases/process",
            json={
                "submission": {
                    "citizen_text": "Retaining wall and carriageway collapse beside library underpass.",
                    "location_hint": "Library underpass",
                    "time_hint": "2026-04-19 13:00",
                }
            },
        )
        case_id = created.json()["case"]["case_id"]

        updated = client.post(
            f"/v1/cases/{case_id}/operations",
            json={"reviewer_id": "reviewer-17", "final_disposition": "closed"},
        )
        retrained = client.post("/v1/local-ml/retrain")
        health_after = client.get("/health").json()

    assert updated.status_code == 200
    assert retrained.status_code == 200
    assert retrained.json()["routing_exported_examples"] >= 1
    assert retrained.json()["priority_exported_examples"] >= 1
    assert "exported_sqlite_feedback" in retrained.json()["routing_training_sources"]
    assert health_after["routing_model_training_examples"] > initial_health["routing_model_training_examples"]
    assert health_after["priority_model_training_examples"] > initial_health["priority_model_training_examples"]
    assert health_after["local_ml_last_retrain"]["routing_exported_examples"] >= 1


def test_eval_endpoints_generate_latest_report_and_health_summary(tmp_path) -> None:
    from app.config import Settings
    from app.main import create_app

    settings = Settings(
        provider="localfree",
        repository_backend="memory",
        eval_artifact_dir=str(tmp_path / "eval_artifacts"),
    )

    with TestClient(create_app(settings)) as client:
        authenticate_test_client(client)
        missing = client.get("/v1/evals/latest")
        run = client.post("/v1/evals/run")
        latest = client.get("/v1/evals/latest")
        health = client.get("/health")

    assert missing.status_code == 404
    assert run.status_code == 200
    assert run.json()["overall_passed"] is True
    assert latest.status_code == 200
    assert latest.json()["report_path"].endswith("latest_evaluation_report.json")
    assert health.status_code == 200
    assert health.json()["evaluation_latest_passed"] is True


def test_verification_flags_location_mismatch(client: TestClient) -> None:
    payload = {
        "original_submission": {
            "citizen_text": "Large pothole on Nizami Street near the metro station.",
            "location_hint": "Nizami Street near metro station",
            "evidence": [{"kind": "image", "filename": "original.jpg"}],
        },
        "structured_issue": {
            "category": "road_damage",
            "issue_type": "Road Damage",
            "summary": "Large pothole on Nizami Street near the metro station.",
            "extracted_signals": ["pothole"],
            "missing_information": [],
            "confidence": 0.82,
        },
        "institution_response": {
            "response_text": "Work completed and area repaired.",
            "location_hint": "Different district near airport highway",
            "evidence": [{"kind": "image", "filename": "response.jpg"}],
        },
    }
    response = client.post("/v1/verify", json=payload)
    assert response.status_code == 200
    verification = response.json()["verification"]
    assert "location_mismatch" in verification["mismatch_flags"]
    assert verification["same_place"] == "no"


def test_verification_uses_coordinate_distance_for_same_place_check(client: TestClient) -> None:
    response = client.post(
        "/v1/verify",
        json={
            "original_submission": {
                "citizen_text": "Large pothole on the service road.",
                "location_hint": "40.4093, 49.8671 service road",
                "evidence": [],
            },
            "structured_issue": {
                "category": "road_damage",
                "issue_type": "Road Damage",
                "summary": "Large pothole on the service road.",
                "extracted_signals": ["pothole"],
                "missing_information": [],
                "confidence": 0.82,
            },
            "institution_response": {
                "response_text": "Work completed and area repaired.",
                "location_hint": "40.5000, 49.9500 service road",
                "evidence": [],
            },
        },
    )

    assert response.status_code == 200
    verification = response.json()["verification"]
    assert verification["same_place"] == "no"
    assert "coordinate_mismatch" in verification["mismatch_flags"]


def test_verification_different_place_keeps_resolution_unverified(client: TestClient) -> None:
    response = client.post(
        "/v1/verify",
        json={
            "original_submission": {
                "citizen_text": "The street light outside the pharmacy has been dark for three nights.",
                "location_hint": "Nizami Street near the pharmacy",
                "evidence": [{"kind": "image", "filename": "dark-lamp.jpg"}],
            },
            "structured_issue": {
                "category": "street_lighting",
                "issue_type": "Street Lighting",
                "summary": "Street light outside the pharmacy is dark.",
                "extracted_signals": ["street light"],
                "missing_information": [],
                "confidence": 0.84,
            },
            "institution_response": {
                "response_text": "The lamp was fixed and restored.",
                "location_hint": "Sahil Boulevard near the seafront",
                "evidence": [{"kind": "image", "filename": "other-lamp.jpg"}],
            },
        },
    )

    assert response.status_code == 200
    verification = response.json()["verification"]
    assert verification["same_place"] == "no"
    assert verification["issue_resolved"] == "uncertain"
    assert "resolution_policy_not_satisfied" in verification["mismatch_flags"]


def test_verification_uses_before_after_images_for_resolution(client: TestClient) -> None:
    original_buffer = BytesIO()
    original_image = Image.new("RGB", (180, 110), color=(118, 118, 118))
    original_draw = ImageDraw.Draw(original_image)
    original_draw.rectangle((0, 78, 179, 109), fill=(105, 105, 105))
    original_draw.ellipse((62, 34, 122, 78), fill=(18, 18, 18))
    original_image.save(original_buffer, format="PNG")

    repaired_buffer = BytesIO()
    repaired_image = Image.new("RGB", (180, 110), color=(122, 122, 122))
    repaired_draw = ImageDraw.Draw(repaired_image)
    repaired_draw.rectangle((0, 78, 179, 109), fill=(108, 108, 108))
    repaired_draw.rectangle((62, 34, 122, 78), fill=(124, 124, 124))
    repaired_image.save(repaired_buffer, format="PNG")

    original_upload = client.post(
        "/v1/evidence/upload",
        params={"kind": "image", "filename": "before.png"},
        content=original_buffer.getvalue(),
        headers={"content-type": "image/png"},
    )
    repaired_upload = client.post(
        "/v1/evidence/upload",
        params={"kind": "image", "filename": "after.png"},
        content=repaired_buffer.getvalue(),
        headers={"content-type": "image/png"},
    )

    assert original_upload.status_code == 200
    assert repaired_upload.status_code == 200

    response = client.post(
        "/v1/verify",
        json={
            "original_submission": {
                "citizen_text": "Large pothole on Road 14 in Sector 7.",
                "location_hint": "Road 14, Sector 7",
                "evidence": [original_upload.json()["evidence_item"]],
            },
            "structured_issue": {
                "category": "road_damage",
                "issue_type": "Road Damage",
                "summary": "Large pothole on Road 14 in Sector 7.",
                "extracted_signals": ["pothole", "damaged_surface_candidate"],
                "missing_information": [],
                "confidence": 0.86,
            },
            "institution_response": {
                "response_text": "Field team attended the site and documented the result.",
                "location_hint": "Sector 7 road 14",
                "evidence": [repaired_upload.json()["evidence_item"]],
            },
        },
    )

    assert response.status_code == 200
    verification = response.json()["verification"]
    assert verification["same_place"] == "yes"
    assert verification["issue_resolved"] == "yes"
    assert "visual scene score" in verification["summary"]


def test_verification_missing_response_location_stays_place_uncertain(client: TestClient) -> None:
    response = client.post(
        "/v1/verify",
        json={
            "original_submission": {
                "citizen_text": "The bus stop display is dead near the metro station.",
                "location_hint": "Bus stop beside Sahil metro station",
                "evidence": [{"kind": "image", "filename": "display.jpg"}],
            },
            "structured_issue": {
                "category": "public_transport",
                "issue_type": "Public Transport",
                "summary": "Bus stop display is dead near the metro station.",
                "extracted_signals": ["bus stop", "metro"],
                "missing_information": [],
                "confidence": 0.8,
            },
            "institution_response": {
                "response_text": "The issue is scheduled for review.",
                "location_hint": "",
                "evidence": [],
            },
        },
    )

    assert response.status_code == 200
    verification = response.json()["verification"]
    assert verification["same_place"] == "uncertain"
    assert verification["issue_resolved"] == "no"
    assert "pending_resolution_language" in verification["mismatch_flags"]


def test_verification_policy_allows_resolution_without_response_evidence_for_waste(client: TestClient) -> None:
    response = client.post(
        "/v1/verify",
        json={
            "original_submission": {
                "citizen_text": "Overflowing garbage bins are blocking the courtyard entrance.",
                "location_hint": "Courtyard entrance behind Block C",
                "evidence": [{"kind": "image", "filename": "bins-before.jpg"}],
            },
            "structured_issue": {
                "category": "waste_management",
                "issue_type": "Waste Management",
                "summary": "Overflowing bins blocking the courtyard entrance.",
                "extracted_signals": ["garbage"],
                "missing_information": [],
                "confidence": 0.87,
            },
            "institution_response": {
                "response_text": "Crew removed waste from the area.",
                "location_hint": "Courtyard behind Block C",
                "evidence": [],
            },
        },
    )

    assert response.status_code == 200
    verification = response.json()["verification"]
    assert verification["same_place"] == "yes"
    assert verification["issue_resolved"] == "yes"


def test_existing_case_can_be_verified_by_case_id(client: TestClient) -> None:
    created = client.post(
        "/v1/cases/process",
        json={
            "submission": {
                "citizen_text": "Large pothole on the hospital access road causing traffic risk.",
                "location_hint": "Hospital access road",
            }
        },
    )
    case_id = created.json()["case"]["case_id"]

    assigned = client.post(
        f"/v1/cases/{case_id}/workflow-actions",
        json={
            "action": "assign",
            "actor_id": "lead-1",
            "assignee_id": "field-agent-7",
            "note": "Assign directly for repair.",
        },
    )
    verified = client.post(
        f"/v1/cases/{case_id}/verify",
        json={
            "actor_id": "field-agent-7",
            "note": "Verification from stored case context.",
            "institution_response": {
                "response_text": "Work completed and area repaired.",
                "location_hint": "Hospital access road",
                "evidence": [{"kind": "image", "filename": "repaired-road.jpg"}],
            },
        },
    )

    assert assigned.status_code == 200
    assert verified.status_code == 200
    assert verified.json()["case"]["status"] == "resolved"
    assert verified.json()["case"]["verification"]["issue_resolved"] == "yes"
    assert verified.json()["original_request"]["submission"]["location_hint"] == "Hospital access road"
    assert verified.json()["case"]["model_context"]["stage_provenance"]["verification"]["model_version"] == "geo-visual-policy.v2"


def test_case_processing_exposes_specialized_legal_review_queue(client: TestClient) -> None:
    response = client.post(
        "/v1/cases/process",
        json={
            "submission": {
                "citizen_text": "Large pothole on Nizami Street near the metro station.",
                "location_hint": "Nizami Street near metro station",
                "evidence": [{"kind": "image", "filename": "original.jpg"}],
            },
            "institution_response": {
                "response_text": "Work completed and area repaired.",
                "location_hint": "Different district near airport highway",
                "evidence": [{"kind": "image", "filename": "response.jpg"}],
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["case"]["human_review"]["needed"] is True
    assert response.json()["case"]["human_review"]["queue"] == "legal-review"
    assert "review-legal" in response.json()["case"]["human_review"]["candidate_groups"]


def test_review_queue_supports_paging_filters_and_sla_sorting(client: TestClient) -> None:
    high_priority = client.post(
        "/v1/cases/process",
        json={
            "submission": {
                "citizen_text": "Large pothole on the hospital access road causing traffic risk.",
                "location_hint": "Hospital access road",
            }
        },
    ).json()["case"]
    low_priority = client.post(
        "/v1/cases/process",
        json={"submission": {"citizen_text": "Problem.", "location_hint": "Unknown"}},
    ).json()["case"]

    client.post(
        f"/v1/cases/{high_priority['case_id']}/operations",
        json={"reviewer_id": "reviewer-queue-1", "final_disposition_reason": "Triaging high-risk case."},
    )

    queue_page = client.get(
        "/v1/review-queue",
        params={"page": 1, "page_size": 1, "sort_by": "sla"},
    )
    assigned_only = client.get(
        "/v1/review-queue",
        params={"assignment_state": "assigned", "assignee_id": "reviewer-queue-1"},
    )
    unassigned_only = client.get(
        "/v1/review-queue",
        params={"assignment_state": "unassigned"},
    )

    assert queue_page.status_code == 200
    assert queue_page.json()["meta"]["page"] == 1
    assert queue_page.json()["meta"]["page_size"] == 1
    assert queue_page.json()["meta"]["total_items"] >= 2
    assert queue_page.json()["meta"]["total_pages"] >= 2
    assert queue_page.json()["items"][0]["case_id"] == high_priority["case_id"]
    assert queue_page.json()["items"][0]["priority_level"] == "high"
    assert queue_page.json()["items"][0]["sla_deadline_at"]

    assert assigned_only.status_code == 200
    assert len(assigned_only.json()["items"]) >= 1
    assert assigned_only.json()["items"][0]["assignee_id"] == "reviewer-queue-1"
    assert assigned_only.json()["items"][0]["assignment_state"] == "assigned"

    assert unassigned_only.status_code == 200
    assert any(item["case_id"] == low_priority["case_id"] for item in unassigned_only.json()["items"])


def test_case_workflow_actions_cover_review_dispatch_verify_and_reopen(client: TestClient) -> None:
    process_response = client.post(
        "/v1/cases/process",
        json={
            "submission": {
                "citizen_text": "Large pothole on the hospital access road causing traffic risk.",
                "location_hint": "Hospital access road",
            }
        },
    )
    case_id = process_response.json()["case"]["case_id"]

    approve = client.post(
        f"/v1/cases/{case_id}/workflow-actions",
        json={"action": "approve", "actor_id": "lead-1", "note": "Manual review approved."},
    )
    assign = client.post(
        f"/v1/cases/{case_id}/workflow-actions",
        json={
            "action": "assign",
            "actor_id": "lead-1",
            "assignee_id": "field-agent-7",
            "note": "Assign to the road response crew.",
        },
    )
    comment = client.post(
        f"/v1/cases/{case_id}/workflow-actions",
        json={"action": "comment", "actor_id": "field-agent-7", "note": "Crew has acknowledged the case."},
    )
    dispatch = client.post(
        f"/v1/cases/{case_id}/workflow-actions",
        json={
            "action": "dispatch",
            "actor_id": "lead-1",
            "assignee_id": "field-agent-7",
            "note": "Dispatched to field operations.",
        },
    )
    verify = client.post(
        f"/v1/cases/{case_id}/workflow-actions",
        json={
            "action": "verify",
            "actor_id": "field-agent-7",
            "institution_response": {
                "response_text": "Work completed and area repaired.",
                "location_hint": "Hospital access road",
                "evidence": [{"kind": "image", "filename": "repaired-road.jpg"}],
            },
        },
    )
    close = client.post(
        f"/v1/cases/{case_id}/workflow-actions",
        json={"action": "close", "actor_id": "lead-1", "note": "Verified and closing case."},
    )
    reopen = client.post(
        f"/v1/cases/{case_id}/workflow-actions",
        json={"action": "reopen", "actor_id": "lead-1", "note": "Reopening for follow-up audit."},
    )

    assert approve.status_code == 200
    assert approve.json()["case"]["status"] == "ready_for_dispatch"
    assert approve.json()["case"]["human_review"]["needed"] is False

    assert assign.status_code == 200
    assert assign.json()["case"]["status"] == "assigned"
    assert assign.json()["case"]["operations"]["reviewer_id"] == "field-agent-7"

    assert comment.status_code == 200
    assert comment.json()["case"]["status"] == "assigned"
    assert comment.json()["case"]["operations"]["workflow_history"][-1]["action"] == "comment"

    assert dispatch.status_code == 200
    assert dispatch.json()["case"]["operations"]["final_disposition"] == "dispatched"

    assert verify.status_code == 200
    assert verify.json()["case"]["status"] == "resolved"
    assert verify.json()["case"]["verification"]["issue_resolved"] == "yes"

    assert close.status_code == 200
    assert close.json()["case"]["status"] == "closed"

    assert reopen.status_code == 200
    assert reopen.json()["case"]["status"] == "reopened"
    assert reopen.json()["case"]["human_review"]["needed"] is True


def test_case_workflow_actions_support_claim_and_reject(client: TestClient) -> None:
    claim_case = client.post(
        "/v1/cases/process",
        json={
            "submission": {
                "citizen_text": "Street light is out near the neighborhood pharmacy.",
                "location_hint": "Nizami Street near the pharmacy",
            }
        },
    ).json()["case"]

    claim = client.post(
        f"/v1/cases/{claim_case['case_id']}/workflow-actions",
        json={"action": "claim", "actor_id": "reviewer-2", "note": "Taking ownership of this case."},
    )

    reject_case = client.post(
        "/v1/cases/process",
        json={"submission": {"citizen_text": "Problem.", "location_hint": "Unknown"}},
    ).json()["case"]

    reject = client.post(
        f"/v1/cases/{reject_case['case_id']}/workflow-actions",
        json={"action": "reject", "actor_id": "reviewer-3", "note": "Insufficient information to proceed."},
    )

    assert claim.status_code == 200
    assert claim.json()["case"]["status"] == "assigned"
    assert claim.json()["case"]["operations"]["reviewer_id"] == "reviewer-2"
    assert claim.json()["case"]["operations"]["workflow_history"][-1]["action"] == "claim"

    assert reject.status_code == 200
    assert reject.json()["case"]["status"] == "rejected"
    assert reject.json()["case"]["operations"]["final_disposition"] == "rejected"


def test_sqlite_startup_hardens_invalid_configured_path(
    monkeypatch, client: TestClient
) -> None:
    from app.config import get_settings
    from app.main import create_app

    monkeypatch.setenv("ASAN_REPOSITORY_BACKEND", "sqlite")
    monkeypatch.setenv("ASAN_SQLITE_PATH", "/dev/null/asanappeal.db")
    get_settings.cache_clear()

    with TestClient(create_app()) as sqlite_client:
        authenticate_test_client(sqlite_client)
        response = sqlite_client.get("/health")

    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["repository"] == "SQLiteCaseRepository"
    assert payload["sqlite_fallback_used"] is True
    assert payload["sqlite_db_path"] != "/dev/null/asanappeal.db"


def test_health_reports_degraded_when_ollama_is_unreachable(monkeypatch) -> None:
    from app.config import get_settings
    from app.main import create_app
    from app.ollama_client import OllamaProbeResult
    from app.providers.local_provider import LocalFreeProvider

    monkeypatch.setenv("ASAN_PROVIDER", "ollama")
    monkeypatch.setenv("ASAN_LOCAL_LLM_BACKEND", "ollama")
    monkeypatch.setenv("ASAN_REPOSITORY_BACKEND", "memory")
    get_settings.cache_clear()

    monkeypatch.setattr(
        LocalFreeProvider,
        "_ollama_probe",
        lambda self, force_refresh=False: OllamaProbeResult(
            status="unreachable",
            base_url="http://127.0.0.1:11434",
            model_requested="gemma3:4b",
            server_reachable=False,
            model_available=False,
            dependency_ok=False,
            available_model_count=0,
        ),
    )

    with TestClient(create_app()) as ollama_client:
        payload = ollama_client.get("/health").json()

    assert payload["status"] == "degraded"
    assert payload["degraded_dependencies"] == ["local_llm"]
    assert payload["local_llm_status"] == "unreachable"
    assert payload["local_llm_dependency_ok"] is False


def test_health_reports_ok_when_ollama_is_ready(monkeypatch) -> None:
    from app.config import get_settings
    from app.main import create_app
    from app.ollama_client import OllamaProbeResult
    from app.providers.local_provider import LocalFreeProvider

    monkeypatch.setenv("ASAN_PROVIDER", "ollama")
    monkeypatch.setenv("ASAN_LOCAL_LLM_BACKEND", "ollama")
    monkeypatch.setenv("ASAN_REPOSITORY_BACKEND", "memory")
    get_settings.cache_clear()

    monkeypatch.setattr(
        LocalFreeProvider,
        "_ollama_probe",
        lambda self, force_refresh=False: OllamaProbeResult(
            status="ready",
            base_url="http://127.0.0.1:11434",
            model_requested="gemma3:4b",
            server_reachable=True,
            model_available=True,
            dependency_ok=True,
            available_model_count=2,
        ),
    )

    with TestClient(create_app()) as ollama_client:
        payload = ollama_client.get("/health").json()

    assert payload["status"] == "ok"
    assert payload["local_llm_status"] == "ready"
    assert payload["local_llm_dependency_ok"] is True


def test_create_app_defers_settings_and_runtime_until_startup(monkeypatch) -> None:
    from app import main as main_module

    fake_settings = main_module.Settings(
        app_name="Deferred App",
        provider="localfree",
        repository_backend="memory",
    )
    events: list[str] = []
    original_build_runtime = main_module.build_runtime

    def tracking_get_settings():
        events.append("get_settings")
        return fake_settings

    def tracking_build_runtime(settings=None):
        events.append("build_runtime")
        return original_build_runtime(settings)

    monkeypatch.setattr(main_module, "get_settings", tracking_get_settings)
    monkeypatch.setattr(main_module, "build_runtime", tracking_build_runtime)

    app = main_module.create_app()
    assert events == []
    assert not hasattr(app.state, "runtime")

    with TestClient(app) as test_client:
        payload = test_client.get("/health").json()

    assert events == ["get_settings", "build_runtime"]
    assert payload["status"] == "ok"
    assert payload["requested_provider"] == "localfree"


def test_storage_backup_and_restore_endpoints(monkeypatch, tmp_path) -> None:
    from app.config import get_settings
    from app.main import create_app

    monkeypatch.setenv("ASAN_REPOSITORY_BACKEND", "sqlite")
    monkeypatch.setenv("ASAN_SQLITE_PATH", str(tmp_path / "asanappeal.db"))
    monkeypatch.setenv("ASAN_SQLITE_BACKUP_DIR", str(tmp_path / "backups"))
    get_settings.cache_clear()

    initial_payload = {
        "submission": {
            "citizen_text": "Deep pothole near the school gate causing vehicles to swerve.",
            "location_hint": "School gate road",
        }
    }
    later_payload = {
        "submission": {
            "citizen_text": "Street light outage at the pharmacy corner for three nights.",
            "location_hint": "Pharmacy corner",
        }
    }

    with TestClient(create_app()) as sqlite_client:
        authenticate_test_client(sqlite_client)
        first_case = sqlite_client.post("/v1/cases/process", json=initial_payload).json()["case"]
        backup_response = sqlite_client.post("/v1/storage/backup", json={"label": "baseline"})
        assert backup_response.status_code == 200
        backup_path = backup_response.json()["operation"]["backup_path"]

        second_case = sqlite_client.post("/v1/cases/process", json=later_payload).json()["case"]
        assert sqlite_client.get(f"/v1/cases/{second_case['case_id']}").status_code == 200

        restore_response = sqlite_client.post(
            "/v1/storage/restore",
            json={"source_path": backup_path},
        )
        restored_first = sqlite_client.get(f"/v1/cases/{first_case['case_id']}")
        restored_second = sqlite_client.get(f"/v1/cases/{second_case['case_id']}")
        backups_response = sqlite_client.get("/v1/storage/backups")

    get_settings.cache_clear()

    assert restore_response.status_code == 200
    assert restored_first.status_code == 200
    assert restored_second.status_code == 404
    assert backups_response.status_code == 200
    assert len(backups_response.json()["items"]) >= 2
    assert restore_response.json()["operation"]["source_integrity"] == "ok"
    assert first_case["case_id"] != second_case["case_id"]


def test_case_operations_and_analytics_endpoints(monkeypatch, tmp_path) -> None:
    from app.config import get_settings
    from app.main import create_app

    monkeypatch.setenv("ASAN_REPOSITORY_BACKEND", "sqlite")
    monkeypatch.setenv("ASAN_SQLITE_PATH", str(tmp_path / "asanappeal.db"))
    monkeypatch.setenv("ASAN_SQLITE_BACKUP_DIR", str(tmp_path / "backups"))
    get_settings.cache_clear()

    payload = {
        "submission": {
            "citizen_text": "Large pothole on the hospital access road causing traffic risk.",
            "location_hint": "Hospital access road",
        }
    }

    with TestClient(create_app()) as sqlite_client:
        authenticate_test_client(sqlite_client)
        case = sqlite_client.post("/v1/cases/process", json=payload).json()["case"]
        update_response = sqlite_client.post(
            f"/v1/cases/{case['case_id']}/operations",
            json={
                "reviewer_id": "reviewer-7",
                "final_disposition": "triaged",
                "final_disposition_reason": "Assigned for field follow-up.",
            },
        )
        transition_response = sqlite_client.post(
            f"/v1/cases/{case['case_id']}/transition",
            json={
                "transition": "assign",
                "actor_id": "reviewer-7",
                "reviewer_id": "reviewer-7",
            },
        )
        list_response = sqlite_client.get(
            "/v1/cases",
            params={
                "category": "road_damage",
                "reviewer_id": "reviewer-7",
                "final_disposition": "assigned",
            },
        )
        analytics_response = sqlite_client.get("/v1/analytics/summary")

    get_settings.cache_clear()

    assert update_response.status_code == 200
    assert transition_response.status_code == 200
    updated_case = transition_response.json()["case"]
    assert updated_case["status"] == "assigned"
    assert updated_case["operations"]["reviewer_id"] == "reviewer-7"
    assert updated_case["operations"]["final_disposition"] == "assigned"
    assert updated_case["model_context"]["provider"] == "LocalFreeProvider"
    assert len(updated_case["operations"]["transition_history"]) >= 2

    assert list_response.status_code == 200
    assert len(list_response.json()["items"]) == 1
    assert list_response.json()["items"][0]["operations"]["reviewer_id"] == "reviewer-7"
    assert "start_progress" in list_response.json()["items"][0]["allowed_transitions"]

    assert analytics_response.status_code == 200
    summary = analytics_response.json()["summary"]
    assert summary["counts_by_status"]["assigned"] >= 1
    assert summary["counts_by_final_disposition"]["assigned"] >= 1


def test_case_transition_endpoint_rejects_invalid_transition(monkeypatch, tmp_path) -> None:
    from app.config import get_settings
    from app.main import create_app

    monkeypatch.setenv("ASAN_REPOSITORY_BACKEND", "sqlite")
    monkeypatch.setenv("ASAN_SQLITE_PATH", str(tmp_path / "asanappeal.db"))
    monkeypatch.setenv("ASAN_SQLITE_BACKUP_DIR", str(tmp_path / "backups"))
    get_settings.cache_clear()

    with TestClient(create_app()) as sqlite_client:
        authenticate_test_client(sqlite_client)
        case = sqlite_client.post(
            "/v1/cases/process",
            json={"submission": {"citizen_text": "Unclear service issue.", "location_hint": "Unknown"}},
        ).json()["case"]
        response = sqlite_client.post(
            f"/v1/cases/{case['case_id']}/transition",
            json={"transition": "resolve", "actor_id": "reviewer-1"},
        )

    get_settings.cache_clear()

    assert response.status_code == 400
    assert "Invalid lifecycle transition" in response.json()["detail"]
