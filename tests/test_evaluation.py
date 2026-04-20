from __future__ import annotations

from app.config import Settings
from app.evaluation import load_latest_evaluation_report, run_evaluation_suite
from app.providers.local_provider import LocalFreeProvider


def test_run_evaluation_suite_generates_reports_and_passes_acceptance_gates(tmp_path) -> None:
    settings = Settings(
        provider="localfree",
        repository_backend="memory",
        eval_artifact_dir=str(tmp_path / "eval_artifacts"),
    )

    report = run_evaluation_suite(settings, provider=LocalFreeProvider(settings))

    assert report["benchmark_name"] == "asanappeal-regression-suite"
    assert report["overall_passed"] is True
    assert set(report["tasks"]) == {"intake", "routing", "priority", "verification"}
    assert report["tasks"]["intake"]["accuracy"] >= settings.eval_intake_min_accuracy
    assert report["tasks"]["routing"]["accuracy"] >= settings.eval_routing_min_accuracy
    assert report["tasks"]["priority"]["accuracy"] >= settings.eval_priority_min_accuracy
    assert report["tasks"]["verification"]["accuracy"] >= settings.eval_verification_min_accuracy
    assert report["tasks"]["intake"]["calibration"]
    assert report["tasks"]["routing"]["calibration"]
    assert report["tasks"]["priority"]["calibration"]
    assert report["tasks"]["verification"]["calibration"]
    assert (tmp_path / "eval_artifacts" / "latest_evaluation_report.json").exists()
    assert (tmp_path / "eval_artifacts" / "latest_threshold_calibration.json").exists()

    latest = load_latest_evaluation_report(settings)
    assert latest is not None
    assert latest["overall_passed"] is True
