from __future__ import annotations

import gzip
import json
import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from simulator.physical_lab import (
    EvidenceStatus,
    NormalizedEvent,
    ObservationCertainty,
    ObservationManifest,
    build_fidelity_corpus_payload,
    hog_cannon_probe,
    offline_runner,
    seal_replay_cache,
    write_run_artifacts,
)
from simulator.physical_lab.cli import main as physical_lab_main
from simulator.physical_lab.devices import sha256_bytes
from simulator.physical_lab.schema import PhysicalLabError
from simulator.ruleset import load_ruleset
from simulator.validation import normalized_state_events


def _recognized_cache(path: Path):
    with gzip.open(path, "wb") as handle:
        pickle.dump({"schema_version": 1}, handle)
        pickle.dump(
            SimpleNamespace(frame_idx=0, video_time_s=0.0, frame_png=b"encoded-frame"),
            handle,
        )
    return seal_replay_cache(path)


def _connected_fixture(tmp_path: Path):
    spec = hog_cannon_probe(evidence_split="validation")
    result = offline_runner().run(spec, run_id="bridge-run")
    captures = {}
    for side, capture in result.captures.items():
        media_path = tmp_path / f"{side}.mp4"
        media_path.write_bytes(side.encode("ascii"))
        captures[side] = replace(
            capture,
            media_path=str(media_path),
            media_sha256=sha256_bytes(side.encode("ascii")),
        )
    result = replace(result, captures=captures)
    artifacts = write_run_artifacts(result, repository_root=tmp_path)
    run_path = Path(artifacts["run_path"])
    run = json.loads(run_path.read_text(encoding="utf-8"))
    cache_seal = _recognized_cache(tmp_path / "replay-cache.pkl.gz")
    replay = result.replay
    assert replay is not None
    ruleset = load_ruleset(spec.ruleset_id)
    simulated_event = next(
        event
        for event in normalized_state_events(replay.final_state)
        if event.kind == "target_changed" and event.values.get("card_id") == "hog-rider"
    )
    observation = ObservationManifest(
        run_id=result.run_id,
        experiment_hash=spec.experiment_hash(),
        capture_group_id=spec.capture_group_id,
        evidence_split=spec.evidence_split,
        status=EvidenceStatus.VALIDATION,
        capture_ids=tuple(row["capture_id"] for row in run["captures"].values()),
        media_hashes={side: row["media_sha256"] for side, row in run["captures"].items()},
        synchronization={"accepted": True, "uncertainty_us": 0},
        replay_cache_hash=cache_seal.sha256,
        events=(
            NormalizedEvent(
                event_id="target-change-1",
                kind=simulated_event.kind,
                video_time_us=simulated_event.tick * ruleset.tick_us,
                match_time_us=simulated_event.tick * ruleset.tick_us,
                confidence=1.0,
                certainty=ObservationCertainty.DIRECT,
                source_frame_indices=(1,),
                evidence_refs=("capture-A:1",),
                card_id="hog-rider",
                owner="A",
                target_card_id="cannon",
            ),
        ),
    )
    observation_path = tmp_path / "observation.json"
    observation.save(observation_path)
    return result, run_path, observation_path, cache_seal, observation


def test_run_artifacts_emit_a_sealed_observation_handoff(tmp_path: Path) -> None:
    result = offline_runner().run(hog_cannon_probe(), run_id="handoff-run")

    artifacts = write_run_artifacts(result, repository_root=tmp_path)
    handoff_path = Path(artifacts["observation_handoff_path"])
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    run = json.loads(Path(artifacts["run_path"]).read_text(encoding="utf-8"))

    assert handoff["kind"] == "physical_lab_observation_handoff"
    assert handoff["run_id"] == run["run_id"]
    assert handoff["run_manifest"]["run_hash"] == run["run_hash"]
    assert set(handoff["captures"]) == {"A", "B"}
    assert handoff["primary_observation_side"] == "A"
    assert handoff["auxiliary_capture_side"] == "B"
    extractor_command = handoff["captures"]["A"]["extractor_command"]
    cache_argument = extractor_command.index("--write-replay-cache") + 1
    assert extractor_command[cache_argument].endswith("replay-cache-A.pkl.gz")
    declared_hash = handoff.pop("handoff_hash")
    from simulator.physical_lab.schema import canonical_hash

    assert declared_hash == canonical_hash(handoff)


def test_admitted_physical_observation_flows_to_standard_fidelity_report(
    tmp_path: Path,
) -> None:
    _result, run_path, observation_path, cache_seal, _observation = _connected_fixture(tmp_path)
    corpus_path = tmp_path / "fidelity-corpus.json"
    report_path = tmp_path / "fidelity-report.json"

    assert physical_lab_main(
        [
            "fidelity",
            str(observation_path),
            "--run",
            str(run_path),
            "--replay-cache",
            str(cache_seal.path),
            "--corpus-out",
            str(corpus_path),
            "--json-out",
            str(report_path),
            "--require-mechanic",
            "hog_cannon_pull_targeting",
        ]
    ) == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["dataset_split"] == "validation"
    assert report["mechanics"]["hog_cannon_pull_targeting"]["traces"]["count"] == 1
    assert report["gate"]["passed"] is True
    assert json.loads(corpus_path.read_text(encoding="utf-8"))["cases"]


def test_fidelity_bridge_rejects_a_hash_without_the_recognized_cache(
    tmp_path: Path,
) -> None:
    _result, run_path, _observation_path, cache_seal, observation = _connected_fixture(tmp_path)
    run = json.loads(run_path.read_text(encoding="utf-8"))

    with pytest.raises(PhysicalLabError, match="does not match the recognized cache"):
        build_fidelity_corpus_payload(
            observation,
            run,
            replay_cache_hash="sha256:" + "0" * 64,
        )

    assert cache_seal.recognized is True
