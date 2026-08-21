from __future__ import annotations

import json
from pathlib import Path

import pytest

from simulator.physical_lab import (
    CalibrationArtifact,
    EvidenceStatus,
    ExperimentSpec,
    Frame,
    LIFECYCLE_PATH,
    LifecycleMachine,
    LifecyclePolicy,
    LifecycleState,
    ScreenStateDetectionError,
    ScriptedLifecycleDetector,
    SplitLock,
    SyncMarker,
    TemplateLifecycleDetector,
    estimate_clock_alignment,
    hog_cannon_probe,
    ingest_extracted_observations,
    offline_runner,
    run_simulator_replay,
    assign_capture_group_split,
    finalize_retention_records,
)
from simulator.physical_lab.artifacts import ArtifactRef, hash_file, register_retention_records
from simulator.physical_lab.cli import main as physical_lab_main
from simulator.physical_lab.devices import sha256_bytes
from simulator.physical_lab.devices import AdbPhoneController
from simulator.physical_lab.cache import ReplayCacheError, seal_replay_cache
from simulator.physical_lab.planner import load_readiness_questions
from simulator.physical_lab.observation import ObservationManifest
from simulator.physical_lab.replay import replay_hash_pair
from simulator.physical_lab.schema import canonical_hash


def test_experiment_is_canonical_and_round_trips(tmp_path: Path) -> None:
    spec = hog_cannon_probe()
    assert spec.experiment_hash() == ExperimentSpec.from_dict(spec.to_dict(include_hash=True)).experiment_hash()
    path = tmp_path / "experiment.json"
    spec.save(path)
    assert ExperimentSpec.load(path).experiment_hash() == spec.experiment_hash()


def test_calibration_uses_the_shared_18_by_32_grid() -> None:
    calibration = CalibrationArtifact.for_screen(
        device_label="test",
        screen_width_px=1080,
        screen_height_px=2400,
    )
    point = calibration.cell_to_pixel((3, 20))
    assert calibration.pixel_to_cell(point) == (3, 20)
    assert calibration.slot_to_pixel(0)[1] > point[1]


def test_sync_gate_accepts_precise_markers_and_rejects_uncertain_markers() -> None:
    accepted = estimate_clock_alignment(
        (
            SyncMarker("countdown", "A", 1_000_000, uncertainty_us=100),
            SyncMarker("countdown", "B", 1_000_004, uncertainty_us=100),
        ),
        declared_tolerance_us=10_000,
    )
    assert accepted.accepted
    assert accepted.alignment_for("B").offset_us == 4

    rejected = estimate_clock_alignment(
        (
            SyncMarker("countdown", "A", 1_000_000, uncertainty_us=20_000),
            SyncMarker("countdown", "B", 1_000_004, uncertainty_us=20_000),
        ),
        declared_tolerance_us=10_000,
    )
    assert not rejected.accepted
    assert rejected.rejection_reasons


def test_offline_runner_completes_lifecycle_and_replay_without_evidence() -> None:
    spec = hog_cannon_probe()
    result = offline_runner(observation_times={"hog_crosses_y_mtile": 17_000}).run(spec)
    assert result.status is EvidenceStatus.CANDIDATE_ONLY
    assert not result.rejection_reasons
    assert result.lifecycle is not None and result.lifecycle.passed
    assert tuple(transition.to_state for transition in result.lifecycle.transitions) == tuple(LIFECYCLE_PATH[1:])
    assert result.replay is not None
    assert replay_hash_pair(spec, action_times={"deploy-cannon": 17_000})


def test_offline_replay_preserves_requested_initial_elixir() -> None:
    spec = hog_cannon_probe()
    replay = run_simulator_replay(spec, action_times={"deploy-cannon": 17_000})
    assert [player.elixir_milli for player in replay.snapshots[0].players] == [10_000, 10_000]


def test_lifecycle_requires_both_devices_to_agree() -> None:
    clock_value = 0

    def clock() -> int:
        return clock_value

    def sleep(_interval: int) -> None:
        nonlocal clock_value
        clock_value += 1

    lifecycle = LifecycleMachine(
        {
            "A": ScriptedLifecycleDetector(LIFECYCLE_PATH),
            "B": ScriptedLifecycleDetector(
                [LifecycleState.RECOVERY, LifecycleState.LOBBY, LifecycleState.BATTLE]
            ),
        },
        policy=LifecyclePolicy(
            timeout_us={state: 3 for state in LIFECYCLE_PATH},
            poll_interval_us=1,
        ),
        clock=clock,
        sleep=sleep,
    )
    report = lifecycle.run()
    assert not report.passed
    assert report.failure is not None
    assert "expected challenge_sent" in report.failure.reason


def test_lifecycle_report_retains_detector_provenance() -> None:
    class ProvenanceDetector(ScriptedLifecycleDetector):
        def provenance(self) -> dict[str, object]:
            return {"kind": "test-detector", "manifest_sha256": "sha256:" + "a" * 64}

    lifecycle = LifecycleMachine(
        ProvenanceDetector(LIFECYCLE_PATH[1:]),
        policy=LifecyclePolicy(
            timeout_us={state: 1 for state in LIFECYCLE_PATH},
            poll_interval_us=1,
        ),
        sleep=lambda _interval: None,
    )
    report = lifecycle.run()
    assert report.passed
    assert report.detector_provenance["device"]["kind"] == "test-detector"
    assert report.to_dict()["detector_provenance"]["device"]["manifest_sha256"] == (
        "sha256:" + "a" * 64
    )


def _write_screen_template_manifest(tmp_path: Path, *, device_id: str = "A") -> tuple[Path, dict[LifecycleState, bytes]]:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    template_payloads: dict[LifecycleState, bytes] = {}
    templates: dict[str, list[dict[str, str]]] = {}
    for index, state in enumerate(LifecycleState):
        image = np.random.default_rng(10_000 + index).integers(
            0,
            256,
            size=(48, 64, 3),
            dtype=np.uint8,
        )
        path = tmp_path / f"{state.value}.png"
        assert cv2.imwrite(str(path), image)
        success, encoded = cv2.imencode(".png", image)
        assert success
        template_payloads[state] = encoded.tobytes()
        templates[state.value] = [{"path": path.name, "sha256": hash_file(path)}]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "device_id": device_id,
        "score_threshold": 0.95,
        "margin_threshold": 0.02,
        "templates": templates,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    manifest_path = tmp_path / "screen-templates.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest_path, template_payloads


def test_reviewed_screen_detector_classifies_sealed_templates(tmp_path: Path) -> None:
    manifest_path, payloads = _write_screen_template_manifest(tmp_path)
    frames = [
        Frame(
            source_device="A",
            frame_index=index,
            workstation_monotonic_us=index,
            payload=payloads[state],
        )
        for index, state in enumerate(LifecycleState)
    ]
    detector = TemplateLifecycleDetector(
        lambda: frames.pop(0),
        manifest_path,
        expected_device_id="A",
    )
    assert [detector.detect() for _ in LifecycleState] == list(LifecycleState)
    provenance = detector.provenance()
    assert provenance["kind"] == "reviewed_screen_template_detector"
    assert provenance["device_id"] == "A"
    assert provenance["template_count"] == len(LifecycleState)


def test_reviewed_screen_detector_rejects_tampered_template(tmp_path: Path) -> None:
    manifest_path, _ = _write_screen_template_manifest(tmp_path)
    template_path = tmp_path / "lobby.png"
    template_path.write_bytes(template_path.read_bytes() + b"tampered")
    with pytest.raises(ScreenStateDetectionError, match="hash mismatch"):
        TemplateLifecycleDetector(lambda: None, manifest_path, expected_device_id="A")


def test_ingest_retains_low_confidence_rows_as_rejections() -> None:
    spec = hog_cannon_probe()
    result = ingest_extracted_observations(
        {
            "entities": [
                {
                    "stable_observation_id": "A:hog-rider:0",
                    "card_id": "hog-rider",
                    "owner": "A",
                    "confidence": 0.5,
                    "samples": [],
                }
            ],
            "events": [],
        },
        run_id="offline-run",
        experiment_hash=spec.experiment_hash(),
        capture_group_id=spec.capture_group_id,
        evidence_split=spec.evidence_split,
        synchronization={"accepted": True},
    )
    assert result.status is EvidenceStatus.REJECTED
    assert result.rejected
    restored = ObservationManifest.from_dict(result.to_dict(include_hash=True))
    assert restored.manifest_hash() == result.manifest_hash()


def test_capture_group_split_lock_cannot_be_relabelled(tmp_path: Path) -> None:
    lock = SplitLock(tmp_path / "splits.json")
    split = assign_capture_group_split("session-a")
    assert lock.lock("session-a", split) is split
    with pytest.raises(ValueError):
        lock.lock("session-a", "heldout" if split.value != "heldout" else "calibration")


def test_replay_cache_rejection_is_explicit(tmp_path: Path) -> None:
    path = tmp_path / "opaque-cache.bin"
    path.write_bytes(b"not a recognized replay cache")
    with pytest.raises(ReplayCacheError):
        seal_replay_cache(path)


def test_readiness_planner_does_not_consume_heldout_rows(tmp_path: Path) -> None:
    path = tmp_path / "readiness.json"
    path.write_text(
        '{"mechanics": [{"id": "calibration-mechanic", "ready": false}, '
        '{"id": "heldout-mechanic", "ready": false, "split": "heldout"}]}'
    )
    rows = load_readiness_questions(path)
    assert [row["id"] for row in rows] == ["calibration-mechanic"]


def test_physical_raw_media_is_promoted_only_after_sealed_ingest(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    run_id = "hog-cannon-run"
    spec = hog_cannon_probe(capture_group_id="connected-group", evidence_split="validation")
    raw_root = root / "outputs/simulator/fidelity_media/physical_lab" / run_id / "raw"
    raw_root.mkdir(parents=True)
    media = {}
    for side in ("A", "B"):
        path = raw_root / f"{side}.mp4"
        path.write_bytes(f"capture-{side}".encode("ascii"))
        media[side] = (path, hash_file(path))

    retention_path = root / "outputs/simulator/fidelity_media/retention.json"
    register_retention_records(
        retention_path,
        run_id=run_id,
        experiment_hash=spec.experiment_hash(),
        workspace_root=root,
        artifacts=[
            ArtifactRef(
                artifact_id=f"capture-{side}",
                kind="raw_video",
                path=str(path),
                sha256=media_hash,
                size_bytes=path.stat().st_size,
            )
            for side, (path, media_hash) in media.items()
        ],
    )

    run_path = root / "run.json"
    run_payload = {
        "kind": "physical_lab_run",
        "run_id": run_id,
        "status": "candidate_only",
        "experiment_hash": spec.experiment_hash(),
        "experiment": spec.to_dict(include_hash=True),
        "device_info": {
            side: {"serial_hash": spec.devices[side].serial_hash, "connected": True}
            for side in ("A", "B")
        },
        "captures": {
            side: {
                "capture_id": f"capture-{side}",
                "media_path": str(path),
                "media_sha256": media_hash,
                "stream_verified": True,
                "status": "complete",
                "frame_count": 1,
            }
            for side, (path, media_hash) in media.items()
        },
        "lifecycle": {
            "initial_state": "recovery",
            "final_state": "recovery",
            "passed": True,
            "transitions": [
                {"to": state.value}
                for state in LIFECYCLE_PATH[1:]
            ],
            "detector_provenance": {
                side: {
                    "kind": "reviewed_screen_template_detector",
                    "device_id": side,
                    "manifest_sha256": f"sha256:{('a' if side == 'A' else 'b') * 64}",
                }
                for side in ("A", "B")
            },
        },
    }
    run_payload["run_hash"] = canonical_hash(run_payload)
    run_path.write_text(json.dumps(run_payload), encoding="utf-8")
    replay_cache = root / "replay-cache.json"
    replay_cache.write_bytes(b"recognized-replay-cache")
    observation_path = root / "observation.json"
    observation_payload = {
        "kind": "physical_lab_observation_manifest",
        "run_id": run_id,
        "experiment_hash": spec.experiment_hash(),
        "capture_group_id": "connected-group",
        "evidence_split": "validation",
        "status": "validation",
        "capture_ids": ["capture-A", "capture-B"],
        "media_hashes": {side: media_hash for side, (_, media_hash) in media.items()},
        "synchronization": {"accepted": True},
        "entities": [],
        "events": [],
        "rejected": [],
        "replay_cache_hash": hash_file(replay_cache),
    }
    observation_payload["manifest_hash"] = canonical_hash(observation_payload)
    observation_path.write_text(json.dumps(observation_payload), encoding="utf-8")

    incomplete_run = dict(run_payload)
    incomplete_run["lifecycle"] = None
    incomplete_run["run_hash"] = canonical_hash(
        {key: value for key, value in incomplete_run.items() if key != "run_hash"}
    )
    run_path.write_text(json.dumps(incomplete_run), encoding="utf-8")
    with pytest.raises(ValueError, match="lifecycle"):
        finalize_retention_records(
            retention_path,
            run_id=run_id,
            run_manifest_path=run_path,
            observation_manifest_path=observation_path,
            workspace_root=root,
            audit_paths=(replay_cache,),
        )
    run_path.write_text(json.dumps(run_payload), encoding="utf-8")

    result = finalize_retention_records(
        retention_path,
        run_id=run_id,
        run_manifest_path=run_path,
        observation_manifest_path=observation_path,
        workspace_root=root,
        audit_paths=(replay_cache,),
    )
    assert result["eviction_eligible"] is True
    saved = json.loads(retention_path.read_text(encoding="utf-8"))
    assert all(row["eviction_eligible"] for row in saved["artifacts"])
    assert all(not Path(row["path"]).is_absolute() for row in saved["artifacts"])
    assert saved["artifacts"][0]["evidence_group_id"] == "connected-group"
    assert "observation.json" in saved["artifacts"][0]["generated_paths"]

    media["A"][0].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash changed"):
        finalize_retention_records(
            retention_path,
            run_id=run_id,
            run_manifest_path=run_path,
            observation_manifest_path=observation_path,
            workspace_root=root,
            audit_paths=(replay_cache,),
        )


def test_connected_plan_binds_hashed_device_identities(tmp_path: Path) -> None:
    plan_path = tmp_path / "connected-plan.jsonl"
    assert physical_lab_main(
        [
            "plan",
            "--hog-cannon-only",
            "--serial-a",
            "physical-a",
            "--serial-b",
            "physical-b",
            "--json-out",
            str(plan_path),
        ]
    ) == 0
    planned = json.loads(plan_path.read_text(encoding="utf-8").splitlines()[0])
    assert planned["devices"]["A"]["serial_hash"] == sha256_bytes(b"physical-a")
    assert planned["devices"]["B"]["serial_hash"] == sha256_bytes(b"physical-b")
    assert "physical-a" not in plan_path.read_text(encoding="utf-8")
    assert "physical-b" not in plan_path.read_text(encoding="utf-8")


def test_adb_frame_and_receipt_use_the_sealed_device_label(monkeypatch) -> None:
    controller = AdbPhoneController("raw-serial", device_label="A", monotonic_clock=lambda: 10)
    monkeypatch.setattr(
        controller,
        "_run",
        lambda *arguments, **kwargs: b"png-bytes" if kwargs.get("binary") else "",
    )
    frame = controller.screenshot()
    assert frame.source_device == "A"
    receipt = controller.tap_screen(4, 5)
    assert receipt.device_id == "A"
    assert "raw-serial" not in receipt.receipt_id


def test_physical_run_blocks_before_capture_when_reserved_space_is_unavailable(tmp_path: Path) -> None:
    output = tmp_path / "blocked.json"
    (tmp_path / "existing.bin").write_bytes(b"x")
    assert physical_lab_main(
        [
            "run",
            "--mode",
            "offline",
            "--repository-root",
            str(tmp_path),
            "--reserve-bytes",
            "2",
            "--max-workspace-bytes",
            "2",
            "--low-water-bytes",
            "0",
            "--json-out",
            str(output),
        ]
    ) == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked_workspace_budget"
