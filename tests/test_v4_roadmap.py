from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_v4_roadmap import (
    PhaseSpec,
    RoadmapOptions,
    _parse_phase_commands,
    artifacts_complete,
    phase_specs,
    status_report,
    validate_sealed_guard,
)


def _options(tmp_path: Path) -> RoadmapOptions:
    return RoadmapOptions(
        run_dir=tmp_path / "roadmap",
        checkpoint=tmp_path / "checkpoint.pt",
    )


def test_phase_contract_exposes_unsupported_gates_and_final_sealed_boundary(tmp_path: Path):
    specs = phase_specs(_options(tmp_path))

    assert specs["broad_generation"].supported
    assert not specs["teacher_audit"].supported
    assert not specs["gru_context"].supported
    assert not specs["supervised_ablation"].supported
    assert not specs["expert_iteration"].supported
    assert not specs["architecture"].supported
    assert specs["final_evaluation"].final_only
    assert specs["final_evaluation"].consumes_sealed
    assert specs["final_evaluation"].dependencies == ("architecture",)


def test_sealed_data_is_rejected_before_final_phase():
    intermediate = PhaseSpec(name="intermediate", description="test", consumes_sealed=False)
    with pytest.raises(RuntimeError, match="sealed data"):
        validate_sealed_guard(intermediate, ["python", "--sealed-entries", "sealed.pt"])

    final = PhaseSpec(name="final", description="test", consumes_sealed=True, final_only=True)
    validate_sealed_guard(final, ["python", "--sealed-entries", "sealed.pt"])


def test_artifact_status_is_complete_only_when_all_declared_files_exist(tmp_path: Path):
    options = _options(tmp_path)
    spec = phase_specs(options)["broad_generation"]
    assert not artifacts_complete(options.run_dir, spec)

    for relative in spec.artifacts:
        path = options.run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("artifact", encoding="utf-8")
    assert artifacts_complete(options.run_dir, spec)


def test_status_report_detects_existing_artifacts_without_claiming_unsupported_phases(
    tmp_path: Path,
):
    options = _options(tmp_path)
    specs = phase_specs(options)
    for relative in specs["broad_generation"].artifacts:
        path = options.run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("artifact", encoding="utf-8")
    summary = options.run_dir / "broad_generation" / "summary.json"
    summary.write_text(json.dumps({"counts": {"train": 227, "sealed": 93}}), encoding="utf-8")

    report = status_report(options)

    assert report["phases"]["broad_generation"]["status"] == "complete_artifacts"
    assert report["phases"]["teacher_audit"]["supported_in_repo"] is False
    assert report["phases"]["teacher_audit"]["status"] == "pending"
    assert report["data_counts"] == {"train": 227, "sealed": 93}
    assert report["required_data_counts"] == {"train": 1000, "sealed": 400}


def test_external_phase_command_supports_explicit_placeholders_and_rejects_duplicates():
    commands = _parse_phase_commands([
        "teacher_audit=python audit.py --out {phase_dir} --train {train_entries}",
    ])
    assert commands["teacher_audit"].startswith("python audit.py")
    with pytest.raises(RuntimeError, match="duplicate"):
        _parse_phase_commands(["teacher_audit=one", "teacher_audit=two"])
