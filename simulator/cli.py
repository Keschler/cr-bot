"""Command-line entry points for simulations, validation, and smoke checks."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m simulator",
        description="Deterministic, versioned Level-11 Hog-cycle simulator",
    )
    # Resolve the default after parsing so every command that consumes the
    # simulator uses the immutable V1 artifact.  Date-stamped rulesets remain
    # available for compatibility and source-build audits via --ruleset.
    parser.add_argument("--ruleset", default=None, help="ruleset ID (V1 is the runtime default)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ruleset = subparsers.add_parser("ruleset", help="validate and describe the pinned ruleset")
    ruleset.add_argument("--json-out", type=Path)

    roster = subparsers.add_parser(
        "roster",
        help="validate the fixed player deck and complete eligible opponent roster",
    )
    roster.add_argument("--json-out", type=Path)
    roster.add_argument(
        "--require-release-verification",
        action="store_true",
        help="fail until every eligible card has exact reproducible release-date evidence",
    )
    roster.add_argument(
        "--require-coverage",
        action="store_true",
        help="fail until every eligible card has an implemented mechanic graph",
    )

    run = subparsers.add_parser("run", help="run one complete headless match")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--no-shuffle", action="store_true")
    run.add_argument("--passive", action="store_true", help="play no cards; exercise clock/tiebreak")
    run.add_argument("--max-ticks", type=int)
    run.add_argument("--state-out", type=Path, help="write final state and full event log")
    run.add_argument("--json-out", type=Path, help="write the compact match summary")

    scenario = subparsers.add_parser("scenario", help="run a canonical JSON scenario")
    scenario.add_argument("path", type=Path)
    scenario.add_argument("--state-out", type=Path)
    scenario.add_argument("--json-out", type=Path)

    fidelity = subparsers.add_parser("fidelity", help="run a pre-split observed fidelity corpus")
    fidelity.add_argument("corpus", type=Path)
    fidelity.add_argument(
        "--split",
        choices=("calibration", "validation", "regression", "heldout"),
        default="heldout",
    )
    fidelity.add_argument("--json-out", type=Path, required=True)
    fidelity.add_argument("--expected-corpus-hash")
    fidelity.add_argument("--min-observations", type=int, default=1)
    fidelity.add_argument("--min-agreement-rate", type=float)
    fidelity.add_argument("--require-mechanic", action="append", default=[])

    readiness = subparsers.add_parser(
        "readiness",
        help="fail-closed per-mechanic training-readiness report",
    )
    readiness.add_argument("report", nargs="*", type=Path, help="pre-split fidelity report JSON")
    readiness.add_argument(
        "--candidate-report",
        action="append",
        type=Path,
        default=[],
        help="conservative discovery report; reported separately and never satisfies held-out gates",
    )
    readiness.add_argument("--json-out", type=Path, required=True)
    readiness.add_argument("--min-heldout-observations", type=int, default=20)
    readiness.add_argument("--min-heldout-agreement-rate", type=float, default=0.98)
    readiness.add_argument("--min-heldout-groups", type=int, default=2)

    mine = subparsers.add_parser(
        "mine-corpus",
        help="compile confidence-gated offline tracks into a sealed fidelity corpus",
    )
    mine.add_argument("manifest", type=Path)
    mine.add_argument("--json-out", type=Path, required=True)
    mine.add_argument("--discarded-out", type=Path)
    mine.add_argument("--confidence-threshold", type=float)

    mine_replay = subparsers.add_parser(
        "mine-replay",
        help="mine isolated movement truth directly from a replay-analysis cache",
    )
    mine_replay.add_argument("cache", type=Path)
    mine_replay.add_argument("--corpus-id", required=True)
    mine_replay.add_argument("--group-id", required=True)
    mine_replay.add_argument("--source-level", type=int, required=True)
    mine_replay.add_argument(
        "--evidence-split", choices=("calibration", "validation", "heldout")
    )
    mine_replay.add_argument("--json-out", type=Path, required=True)
    mine_replay.add_argument("--discarded-out", type=Path)
    mine_replay.add_argument("--confidence-threshold", type=float, default=0.98)
    mine_replay.add_argument("--minimum-track-frames", type=int, default=20)
    mine_replay.add_argument("--minimum-displacement-mtile", type=int, default=750)
    mine_replay.add_argument("--isolation-radius-mtile", type=int, default=3500)
    mine_replay.add_argument(
        "--contamination-confidence-threshold",
        type=float,
        default=0.25,
        help=(
            "lower confidence gate for nearby objects that invalidate an "
            "otherwise high-confidence movement track"
        ),
    )
    mine_replay.add_argument("--minimum-speed-ratio-permille", type=int, default=500)
    mine_replay.add_argument("--maximum-speed-ratio-permille", type=int, default=1500)
    mine_replay.add_argument(
        "--level-invariant-current-ruleset",
        action="store_true",
        help=(
            "permit movement/path evidence from another card level only when the "
            "capture is known to use the current ruleset"
        ),
    )
    mine_replay.add_argument(
        "--expected-support-tower-hp",
        type=int,
        help="exact full support-tower HP used to confirm a cross-level source",
    )
    mine_replay.add_argument(
        "--kinematic-only-gate",
        action="store_true",
        help=(
            "select continuous motion without consulting expected card speed; "
            "required for validation and held-out evidence"
        ),
    )

    mine_replay_tracks = subparsers.add_parser(
        "mine-replay-tracks",
        help="convert one extractor replay cache into confidence-gated video tracks",
    )
    mine_replay_tracks.add_argument("source_manifest", type=Path)
    mine_replay_tracks.add_argument("--video-id", required=True)
    mine_replay_tracks.add_argument("--cache", type=Path, required=True)
    mine_replay_tracks.add_argument(
        "--hud-variant", choices=("standard", "alternative"), required=True
    )
    mine_replay_tracks.add_argument("--json-out", type=Path, required=True)
    mine_replay_tracks.add_argument("--confidence-threshold", type=float, default=0.98)
    mine_replay_tracks.add_argument("--minimum-track-frames", type=int, default=20)
    mine_replay_tracks.add_argument("--isolation-radius-mtile", type=int, default=3500)

    mine_replay_batch = subparsers.add_parser(
        "mine-replay-batch",
        help="mine every available replay cache for one HUD profile",
    )
    mine_replay_batch.add_argument("source_manifest", type=Path)
    mine_replay_batch.add_argument(
        "--extractor-root",
        dest="extractor_roots",
        type=Path,
        action="append",
        required=True,
        help="extractor root; repeat to merge resumptions deterministically",
    )
    mine_replay_batch.add_argument(
        "--hud-variant", choices=("standard", "alternative", "both"), required=True
    )
    mine_replay_batch.add_argument("--json-out", type=Path, required=True)
    mine_replay_batch.add_argument("--confidence-threshold", type=float, default=0.98)
    mine_replay_batch.add_argument("--minimum-track-frames", type=int, default=20)
    mine_replay_batch.add_argument("--isolation-radius-mtile", type=int, default=3500)

    mine_pulls = subparsers.add_parser(
        "mine-pulls",
        help="mine action-anchored Hog/Cannon pull trajectories from a replay cache",
    )
    mine_pulls.add_argument("cache", type=Path)
    mine_pulls.add_argument("--ground-truth", type=Path, required=True)
    mine_pulls.add_argument("--corpus-id", required=True)
    mine_pulls.add_argument("--group-id", required=True)
    mine_pulls.add_argument("--source-level", type=int, required=True)
    mine_pulls.add_argument(
        "--evidence-split", choices=("calibration", "validation", "heldout")
    )
    mine_pulls.add_argument("--json-out", type=Path, required=True)
    mine_pulls.add_argument("--confidence-threshold", type=float, default=0.80)
    mine_pulls.add_argument("--minimum-track-frames", type=int, default=5)

    mine_lifetime = subparsers.add_parser(
        "discover-cannon-lifetime",
        help="find conservative undamaged Cannon lifetime candidates in a replay cache",
    )
    mine_lifetime.add_argument("cache", type=Path)
    mine_lifetime.add_argument(
        "--ground-truth",
        type=Path,
        help="optional localized Cannon actions; without it track onset is only a hypothesis",
    )
    mine_lifetime.add_argument("--source-level", type=int, required=True)
    mine_lifetime.add_argument("--json-out", type=Path, required=True)
    mine_lifetime.add_argument("--confidence-threshold", type=float, default=0.90)
    mine_lifetime.add_argument("--maximum-track-gap-s", type=float, default=0.25)

    mine_interactions = subparsers.add_parser(
        "discover-replay-interactions",
        help=(
            "mine action-free bridge, Cannon-lifetime, Hog/Cannon, and "
            "track-onset candidates from replay caches"
        ),
    )
    mine_interactions.add_argument(
        "cache",
        type=Path,
        nargs="+",
        help="one or more sealed replay caches; sources are merged deterministically",
    )
    mine_interactions.add_argument("--source-level", type=int, required=True)
    mine_interactions.add_argument("--json-out", type=Path, required=True)
    mine_interactions.add_argument(
        "--level-invariant-current-ruleset",
        action="store_true",
        help="allow cross-level topology evidence with an explicit tower-HP sentinel",
    )
    mine_interactions.add_argument(
        "--expected-support-tower-hp",
        type=int,
        help="exact source support-tower HP required for cross-level evidence",
    )
    mine_interactions.add_argument(
        "--level-proof-cache",
        dest="level_proof_caches",
        type=Path,
        action="append",
        default=[],
        help=(
            "sealed full-cache proof of the declared source level; repeat for "
            "multiple videos so bounded windows from the same video may inherit proof"
        ),
    )
    mine_interactions.add_argument("--confidence-threshold", type=float, default=0.90)
    mine_interactions.add_argument(
        "--contamination-confidence-threshold", type=float, default=0.25
    )
    mine_interactions.add_argument("--minimum-track-frames", type=int, default=5)
    mine_interactions.add_argument("--maximum-track-gap-s", type=float, default=0.35)
    mine_interactions.add_argument("--isolation-radius-mtile", type=int, default=3500)

    merge_interactions = subparsers.add_parser(
        "merge-replay-interactions",
        help="reconcile standard/alternative HUD interaction candidates without promoting truth",
    )
    merge_interactions.add_argument(
        "report",
        type=Path,
        nargs="+",
        help="autonomous interaction candidate reports to merge",
    )
    merge_interactions.add_argument("--json-out", type=Path, required=True)
    merge_interactions.add_argument("--onset-tolerance-ms", type=int, default=250)
    merge_interactions.add_argument("--position-tolerance-mtile", type=int, default=1500)
    merge_interactions.add_argument(
        "--require-both-hud",
        action="store_true",
        help="fail closed when no source has both standard and alternative candidates",
    )

    mine_damage = subparsers.add_parser(
        "discover-tower-damage",
        help="find exact supported-card damage and repeat intervals from stable tower HP plateaus",
    )
    mine_damage.add_argument("cache", type=Path)
    mine_damage.add_argument("--source-level", type=int, required=True)
    mine_damage.add_argument("--json-out", type=Path, required=True)
    mine_damage.add_argument("--confidence-threshold", type=float, default=0.80)
    mine_damage.add_argument("--minimum-plateau-frames", type=int, default=3)

    mine_log = subparsers.add_parser(
        "discover-log-motion",
        help="find action-anchored monotonic Log rolling-speed candidates",
    )
    mine_log.add_argument("cache", type=Path)
    mine_log.add_argument("--ground-truth", type=Path, required=True)
    mine_log.add_argument("--source-level", type=int, required=True)
    mine_log.add_argument("--json-out", type=Path, required=True)
    mine_log.add_argument("--confidence-threshold", type=float, default=0.75)
    mine_log.add_argument("--minimum-moving-steps", type=int, default=5)

    mine_fireball = subparsers.add_parser(
        "discover-fireball-flight",
        help="find localized action-to-impact Fireball timing candidates",
    )
    mine_fireball.add_argument("cache", type=Path)
    mine_fireball.add_argument("--ground-truth", type=Path, required=True)
    mine_fireball.add_argument("--source-level", type=int, required=True)
    mine_fireball.add_argument("--json-out", type=Path, required=True)
    mine_fireball.add_argument("--confidence-threshold", type=float, default=0.75)
    mine_fireball.add_argument("--minimum-flight-samples", type=int, default=6)

    video_truth = subparsers.add_parser(
        "mine-video-truth",
        help="filter detector tracks into sealed pre-evolution video truth",
    )
    video_truth.add_argument(
        "manifest",
        type=Path,
        help="JSON source/track manifest emitted by the offline vision pipeline",
    )
    video_truth.add_argument("--json-out", type=Path, required=True)
    video_truth.add_argument("--retention-out", type=Path)
    video_truth.add_argument(
        "--raw-root-relative",
        default="outputs/simulator/fidelity_media/raw",
        help="workspace-relative raw-video root recorded in retention metadata",
    )
    video_truth.add_argument("--confidence-threshold", type=float, default=0.98)
    video_truth.add_argument("--minimum-track-frames", type=int, default=20)
    video_truth.add_argument(
        "--minimum-displacement-mtile",
        type=int,
        default=500,
        help="reject static/lingering tracks below this endpoint displacement",
    )
    video_truth.add_argument(
        "--minimum-elapsed-s",
        type=float,
        default=0.25,
        help="reject tracks shorter than this observed duration",
    )
    video_truth.add_argument(
        "--minimum-moving-interval-fraction",
        type=float,
        default=0.5,
        help="minimum fraction of sample intervals with generic motion",
    )
    video_truth.add_argument(
        "--moving-speed-floor-mtile-per-s",
        type=int,
        default=250,
        help="generic movement floor used only for detector-quality gating",
    )
    video_truth.add_argument(
        "--maximum-step-speed-mtile-per-s",
        type=int,
        default=6000,
        help="reject discontinuous detector teleports above this speed",
    )
    video_truth.add_argument("--maximum-frame-gap-factor", type=float, default=4.0)
    video_truth.add_argument("--maximum-frame-gap", type=int, default=60)
    video_truth.add_argument(
        "--maximum-path-to-displacement-ratio",
        type=float,
        default=3.0,
        help="reject detector tracks whose travelled path is too irregular versus endpoint displacement",
    )
    video_truth.add_argument(
        "--maximum-speed-iqr-ratio",
        type=float,
        default=2.0,
        help="reject tracks with unstable interval speed using interquartile spread",
    )
    video_truth.add_argument(
        "--split-salt",
        default="simulator-v1-video-split",
        help="stable group-disjoint split salt; record it in the sealed truth manifest",
    )

    compile_video_truth = subparsers.add_parser(
        "compile-video-truth",
        help="compile mined video tracks into the leakage-safe fidelity corpus schema",
    )
    compile_video_truth.add_argument("truth_manifest", type=Path)
    compile_video_truth.add_argument("--source-manifest", type=Path)
    compile_video_truth.add_argument("--corpus-id", default="yersoncz-video-truth-v1")
    compile_video_truth.add_argument("--position-tolerance-mtile", type=int, default=200)
    compile_video_truth.add_argument(
        "--speed-estimator",
        choices=("endpoint", "path_length", "median_step"),
        default="endpoint",
        help="measurement model for detector-track movement speed; path_length is preferred for curved paths",
    )
    compile_video_truth.add_argument("--json-out", type=Path, required=True)

    discover_video = subparsers.add_parser(
        "discover-video-source",
        help="discover and seal verified pre-Evolution YersonCz source metadata",
    )
    discover_video.add_argument("--source", type=str)
    discover_video.add_argument("--max-videos", type=int)
    discover_video.add_argument("--cookies-from-browser", type=str)
    discover_video.add_argument("--json-out", type=Path, required=True)

    extract_video = subparsers.add_parser(
        "extract-video",
        help="plan or run both HUD-profile vision extraction jobs for a source manifest",
    )
    extract_video.add_argument("manifest", type=Path)
    extract_video.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/simulator/fidelity_media/extractor"),
    )
    extract_video.add_argument("--sample-interval", type=float, default=0.1)
    extract_video.add_argument("--video-start-time", type=float)
    extract_video.add_argument("--video-duration", type=float)
    extract_video.add_argument("--no-yolo", action="store_true")
    extract_video.add_argument("--workspace-root", type=Path, default=Path.cwd())
    extract_video.add_argument(
        "--retention-manifest",
        type=Path,
        default=Path("outputs/simulator/fidelity_media/retention.json"),
    )
    extract_video.add_argument(
        "--raw-media-root",
        type=Path,
        default=Path("outputs/simulator/fidelity_media/raw"),
    )
    extract_video.add_argument("--reserve-bytes", type=int, default=1_000_000_000)
    extract_video.add_argument(
        "--job-timeout-s",
        type=float,
        default=1_800.0,
        help="maximum wall-clock time per extractor job; timed-out jobs remain auditable",
    )
    extract_video.add_argument(
        "--evict",
        action="store_true",
        help="evict only registered truth-extracted raw videos if the reserve requires it",
    )
    extract_video.add_argument(
        "--execute",
        action="store_true",
        help="run the neural extractor; without this flag only emit a dry-run plan",
    )
    extract_video.add_argument(
        "--rerun-existing",
        action="store_true",
        help="recompute replay caches even when an output cache already exists",
    )
    extract_video.add_argument("--stop-on-error", action="store_true")
    extract_video.add_argument("--json-out", type=Path, required=True)

    action_windows = subparsers.add_parser(
        "plan-action-windows",
        help=(
            "select confidence-gated card-action windows from offline candidates "
            "and optionally run both HUD extractors"
        ),
    )
    action_windows.add_argument("manifest", type=Path, help="sealed pre-Evolution source manifest")
    action_windows.add_argument(
        "candidates_root",
        type=Path,
        help="directory containing <video_id>.jsonl action-candidate files",
    )
    action_windows.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/simulator/fidelity_media/extractor-action-windows"),
    )
    action_windows.add_argument("--sample-interval", type=float, default=0.1)
    action_windows.add_argument("--confidence-threshold", type=float, default=0.85)
    action_windows.add_argument("--max-windows-per-video", type=int, default=8)
    action_windows.add_argument("--window-before", type=float, default=2.0)
    action_windows.add_argument("--window-after", type=float, default=8.0)
    action_windows.add_argument("--minimum-window-separation", type=float, default=3.0)
    action_windows.add_argument("--split-salt", default="simulator-v1-video-split")
    action_windows.add_argument(
        "--card",
        action="append",
        default=[],
        help="restrict candidate windows to one or more fixed V1 card IDs",
    )
    action_windows.add_argument("--no-yolo", action="store_true")
    action_windows.add_argument("--workspace-root", type=Path, default=Path.cwd())
    action_windows.add_argument(
        "--retention-manifest",
        type=Path,
        default=Path("outputs/simulator/fidelity_media/retention.json"),
    )
    action_windows.add_argument(
        "--raw-media-root",
        type=Path,
        default=Path("outputs/simulator/fidelity_media/raw"),
    )
    action_windows.add_argument("--reserve-bytes", type=int, default=1_000_000_000)
    action_windows.add_argument(
        "--job-timeout-s",
        type=float,
        default=1_800.0,
        help="maximum wall-clock time per extractor job; timed-out jobs remain auditable",
    )
    action_windows.add_argument("--evict", action="store_true")
    action_windows.add_argument(
        "--execute",
        action="store_true",
        help="run extractor jobs; without this flag only create a dry-run plan",
    )
    action_windows.add_argument("--rerun-existing", action="store_true")
    action_windows.add_argument("--stop-on-error", action="store_true")
    action_windows.add_argument("--json-out", type=Path, required=True)

    generate = subparsers.add_parser(
        "generate-scenarios",
        help="generate deterministic synthetic cases for every eligible opponent card",
    )
    generate.add_argument("--json-out", type=Path, required=True)
    generate.add_argument("--per-mechanic", type=int, default=1)
    generate.add_argument("--card", action="append", default=[])

    generate_interactions = subparsers.add_parser(
        "generate-interactions",
        help="generate fixed Hog-cycle-card × eligible-opponent interaction probes",
    )
    generate_interactions.add_argument("--json-out", type=Path, required=True)
    generate_interactions.add_argument("--variants", type=int, default=1)
    generate_interactions.add_argument(
        "--opponent-card",
        action="append",
        default=[],
        help="restrict the matrix to one or more eligible opponent cards",
    )
    generate_interactions.add_argument(
        "--player-card",
        action="append",
        default=[],
        help="restrict the fixed player deck column to one or more player cards",
    )

    generate_opponent_pairs = subparsers.add_parser(
        "generate-opponent-pairs",
        help="generate unordered two-opponent-card probes against the fixed Hog deck",
    )
    generate_opponent_pairs.add_argument("--json-out", type=Path, required=True)
    generate_opponent_pairs.add_argument("--variants", type=int, default=1)
    generate_opponent_pairs.add_argument(
        "--opponent-card",
        action="append",
        default=[],
        help="restrict pair generation to two or more eligible opponent cards",
    )

    validate_generated = subparsers.add_parser(
        "validate-generated",
        help="execute every generated scenario and audit repeated state/event hashes",
    )
    validate_generated.add_argument("manifest", type=Path)
    validate_generated.add_argument("--json-out", type=Path, required=True)
    validate_generated.add_argument("--repeats", type=int, default=2)
    validate_generated.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel isolated validation workers for large synthetic matrices",
    )
    validate_generated.add_argument(
        "--no-tick-validation",
        action="store_true",
        help="use final-state invariant validation for large generated matrices",
    )
    validate_generated.add_argument(
        "--require-complete",
        action="store_true",
        help=(
            "require complete V1 matrix coverage and a behavioral event/state "
            "obligation for every generated case"
        ),
    )

    reconcile = subparsers.add_parser(
        "reconcile-data",
        help="compare every opponent-card field with Level-11 structured data and official overrides",
    )
    reconcile.add_argument(
        "--source-json",
        type=Path,
        help="optional Level-11 source snapshot; defaults to simulator/sources/level11_card_stats.json",
    )
    reconcile.add_argument(
        "--additional-source-json",
        action="append",
        type=Path,
        default=[],
        help="additional structured source snapshot; may be repeated (DeckShop is included by default)",
    )
    reconcile.add_argument("--json-out", type=Path, required=True)
    reconcile.add_argument(
        "--strict",
        action="store_true",
        help="fail unless every compared field is fully verified and the ruleset is training-ready",
    )

    check = subparsers.add_parser("check-determinism", help="repeat a match and compare state hashes")
    check.add_argument("--seed", type=int, default=0)
    check.add_argument("--repeats", type=int, default=3)

    audit = subparsers.add_parser("audit", help="lockstep legal-action determinism fuzz audit")
    audit.add_argument("--seeds", type=int, default=4)
    audit.add_argument("--seed-start", type=int, default=0)
    audit.add_argument("--max-ticks", type=int, default=1_000)
    audit.add_argument("--decision-ticks", type=int)
    audit.add_argument("--json-out", type=Path)

    soak = subparsers.add_parser("soak", help="bounded lockstep determinism soak audit")
    soak.add_argument("--seeds", type=int, default=16)
    soak.add_argument("--seed-start", type=int, default=0)
    soak.add_argument("--tick-budget", type=int, default=100_000)
    soak.add_argument("--max-ticks", type=int)
    soak.add_argument("--decision-ticks", type=int)
    soak.add_argument("--json-out", type=Path)

    benchmark = subparsers.add_parser("benchmark", help="measure deterministic Python reference throughput")
    benchmark.add_argument("--matches", type=int, default=3)
    benchmark.add_argument("--seed", type=int, default=0)
    benchmark.add_argument(
        "--strict-validation",
        action="store_true",
        help="include the full per-tick state-schema audit (training mode skips it)",
    )

    benchmark_vector = subparsers.add_parser(
        "benchmark-vector",
        help="measure batched policy-step throughput and deterministic lane hashes",
    )
    benchmark_vector.add_argument("--envs", type=int, default=16)
    benchmark_vector.add_argument("--steps", type=int, default=20)
    benchmark_vector.add_argument("--seed", type=int, default=0)
    benchmark_vector.add_argument(
        "--backend",
        choices=("reference", "process", "packed-process", "persistent-process"),
        default="reference",
        help=(
            "reference lanes, serialized process lanes, packed-state process lanes, "
            "or persistent worker lanes"
        ),
    )
    benchmark_vector.add_argument("--workers", type=int)
    benchmark_vector.add_argument("--json-out", type=Path)

    train = subparsers.add_parser(
        "train",
        help="run a bounded NumPy PPO smoke-training job against the simulator",
    )
    train.add_argument("--steps", type=int, default=10_000, help="requested policy transitions")
    train.add_argument("--envs", type=int, default=8, help="parallel simulator lanes")
    train.add_argument("--rollout-steps", type=int, default=128)
    train.add_argument("--update-epochs", type=int, default=2)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--entropy-coef", type=float, default=0.01)
    train.add_argument(
        "--opponent",
        choices=("scripted", "deterministic-cycle", "self-play"),
        default="scripted",
        help="deterministic-cycle is an alias for scripted",
    )
    train.add_argument(
        "--backend",
        choices=("reference", "process", "packed-process"),
        default="reference",
    )
    train.add_argument("--workers", type=int)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--checkpoint", type=Path, help="resume from a compatible .npz checkpoint")
    train.add_argument(
        "--checkpoint-out",
        type=Path,
        default=Path("outputs/simulator/training/ppo-smoke.npz"),
    )
    train.add_argument("--checkpoint-every", type=int, default=10_000)
    train.add_argument("--eval-every", type=int, default=10_000)
    train.add_argument("--eval-episodes", type=int, default=8)
    train.add_argument(
        "--eval-max-decisions",
        type=int,
        default=None,
        help="evaluation decision cap; omit to evaluate a complete regulation-plus-overtime match",
    )
    train.add_argument(
        "--allow-provisional-smoke",
        action="store_true",
        help="allow bounded smoke training even though the bundled ruleset is not fidelity-ready",
    )
    train.add_argument(
        "--training-profile",
        type=Path,
        help="JSON scope/readiness profile for a serious or explicitly scoped smoke run",
    )
    train.add_argument("--json-out", type=Path)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="evaluate a saved NumPy PPO checkpoint on deterministic held-out seeds",
    )
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--episodes", type=int, default=8)
    evaluate.add_argument("--seed", type=int, default=10_000)
    evaluate.add_argument(
        "--opponent",
        choices=("scripted", "deterministic-cycle", "self-play"),
        default=None,
        help="defaults to the opponent recorded in the checkpoint",
    )
    evaluate.add_argument(
        "--max-decisions",
        type=int,
        default=None,
        help="decision cap; omit to evaluate a complete regulation-plus-overtime match",
    )
    evaluate.add_argument(
        "--training-profile",
        type=Path,
        help="JSON scope/readiness profile required for a scoped serious evaluation",
    )
    evaluate.add_argument("--json-out", type=Path)

    recurrent_prototype = subparsers.add_parser(
        "recurrent-prototype",
        help="run the PyTorch public-observation recurrent PPO prototype",
    )
    recurrent_prototype.add_argument(
        "prototype_args",
        nargs=argparse.REMAINDER,
        help="arguments forwarded to `python -m simulator.rl.prototype`",
    )

    media_budget = subparsers.add_parser(
        "media-budget",
        help="inspect or enforce the 200 GB workspace cap using registered raw videos",
    )
    media_budget.add_argument("--workspace-root", type=Path, default=Path.cwd())
    media_budget.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/simulator/fidelity_media/retention.json"),
    )
    media_budget.add_argument(
        "--raw-media-root",
        type=Path,
        default=Path("outputs/simulator/fidelity_media"),
    )
    media_budget.add_argument("--max-bytes", type=int, default=200_000_000_000)
    media_budget.add_argument("--low-water-bytes", type=int, default=190_000_000_000)
    media_budget.add_argument("--reserve-bytes", type=int, default=0)
    media_budget.add_argument(
        "--evict",
        action="store_true",
        help="delete oldest eligible registered raw videos when the cap requires it",
    )
    media_budget.add_argument("--json-out", type=Path)

    lab = subparsers.add_parser(
        "lab",
        help="run the physical-fidelity lab planner, capture, ingest, comparison, or fidelity tools",
    )
    lab.add_argument(
        "lab_args",
        nargs=argparse.REMAINDER,
        help="arguments for `lab plan|run|ingest|compare|fidelity|status`",
    )

    return parser


def _write_json(path: Path | None, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path is None:
        print(encoded, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _state_summary(state: object) -> dict[str, Any]:
    events = getattr(state, "events")
    players = getattr(state, "players")
    entities = getattr(state, "entities")
    tower_hp = {
        f"player_{entity.owner}_{entity.role}": entity.hp
        for entity in entities.values()
        if entity.kind == "tower"
    }
    return {
        "schema_version": 1,
        "engine_version": state.engine_version,
        "ruleset_id": state.ruleset_id,
        "ruleset_hash": state.ruleset_hash,
        "seed": state.seed,
        "tick": state.tick,
        "elapsed_us": state.elapsed_us,
        "terminal": state.terminal,
        "winner": state.winner,
        "terminal_reason": state.terminal_reason,
        "crowns": [player.crowns for player in players],
        "tower_hp": dict(sorted(tower_hp.items())),
        "event_counts": dict(sorted(Counter(event.kind for event in events).items())),
        "state_hash": state.state_hash(),
        "event_log_hash": state.event_log_hash(),
        "replay_hash": state.replay_hash(),
    }


def _ruleset_summary(ruleset: object, *, engine_version: str) -> dict[str, Any]:
    uncertainties = []
    for owner, rows in [
        ("ruleset", ruleset.uncertainties),
        *[(f"card:{key}", value.uncertainties) for key, value in ruleset.cards.items()],
        *[(f"tower:{key}", value.uncertainties) for key, value in ruleset.towers.items()],
    ]:
        for row in rows:
            uncertainties.append(
                {
                    "owner": owner,
                    "field": row.field,
                    "impact": row.impact,
                    "reason": row.reason,
                    "resolution": row.resolution,
                }
            )
    return {
        "schema_version": ruleset.schema_version,
        "engine_version": engine_version,
        "ruleset_id": ruleset.ruleset_id,
        "content_hash": ruleset.content_hash,
        "level": ruleset.level,
        "tick_us": ruleset.tick_us,
        "interaction_set": list(ruleset.interaction_set),
        "cards": sorted(ruleset.cards),
        "towers": sorted(ruleset.towers),
        "status": ruleset.metadata.get("status"),
        "uncertainty_count": len(uncertainties),
        "uncertainties": uncertainties,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "lab":
        from .physical_lab.cli import main as physical_lab_main

        return physical_lab_main(args.lab_args)
    if args.command == "recurrent-prototype":
        from .rl.prototype import main as recurrent_prototype_main

        return recurrent_prototype_main(args.prototype_args)
    if args.ruleset is None:
        # Coverage, simulation, and release/readiness commands target the
        # fixed V1 artifact.  Corpus evaluation keeps the historical default
        # so an old manifest remains replayable; pass --ruleset v1 when
        # evaluating a V1 corpus explicitly.
        args.ruleset = (
            "v1"
            if args.command
            in {
                "ruleset",
                "roster",
                "run",
                "scenario",
                "readiness",
                "generate-scenarios",
                "generate-interactions",
                "generate-opponent-pairs",
                "validate-generated",
                "reconcile-data",
                "compile-video-truth",
                "check-determinism",
                "audit",
                "soak",
                "benchmark",
                "benchmark-vector",
                "train",
                "evaluate",
                "recurrent-prototype",
            }
            else "2026-08-04"
        )
    if args.command == "media-budget":
        from .storage import enforce_workspace_budget

        report = enforce_workspace_budget(
            args.workspace_root,
            manifest_path=args.manifest,
            raw_media_root=args.raw_media_root,
            max_bytes=args.max_bytes,
            low_water_bytes=args.low_water_bytes,
            reserve_bytes=args.reserve_bytes,
            evict=args.evict,
        )
        _write_json(args.json_out, report)
        if args.json_out is not None:
            _write_json(None, report)
        return 0 if report["passed"] else 2

    if args.command == "mine-video-truth":
        from .video_pipeline import (
            filter_source_manifest,
            mine_clean_tracks,
            retention_records,
            write_json,
        )

        try:
            raw = json.loads(args.manifest.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("accepted"), list):
                source_manifest = raw
            elif isinstance(raw, dict):
                source_manifest = filter_source_manifest(
                    raw.get("entries") or raw.get("videos") or []
                )
            elif isinstance(raw, list):
                source_manifest = filter_source_manifest(raw)
            else:
                raise ValueError("manifest must be an object or array")
            truth = mine_clean_tracks(
                source_manifest,
                confidence_threshold=args.confidence_threshold,
                minimum_track_frames=args.minimum_track_frames,
                split_salt=args.split_salt,
                minimum_displacement_mtile=args.minimum_displacement_mtile,
                minimum_elapsed_s=args.minimum_elapsed_s,
                minimum_moving_interval_fraction=args.minimum_moving_interval_fraction,
                moving_speed_floor_mtile_per_s=args.moving_speed_floor_mtile_per_s,
                maximum_step_speed_mtile_per_s=args.maximum_step_speed_mtile_per_s,
                maximum_frame_gap_factor=args.maximum_frame_gap_factor,
                maximum_frame_gap=args.maximum_frame_gap,
                maximum_path_to_displacement_ratio=args.maximum_path_to_displacement_ratio,
                maximum_speed_iqr_ratio=args.maximum_speed_iqr_ratio,
            )
            write_json(args.json_out, truth)
            if args.retention_out:
                retention = {
                    "schema_version": 1,
                    "artifacts": retention_records(
                        source_manifest,
                        truth_manifest_path=args.json_out,
                        raw_root_relative=args.raw_root_relative,
                        truth_manifest=truth,
                    ),
                }
                write_json(args.retention_out, retention)
            _write_json(
                None,
                {
                    "kind": truth["kind"],
                    "json_out": str(args.json_out),
                    "accepted_case_count": truth["summary"]["accepted_case_count"],
                    "discarded_track_count": truth["summary"]["discarded_track_count"],
                },
            )
            return 0 if truth["summary"]["truth_ready"] else 2
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise SystemExit(str(error)) from error

    if args.command == "mine-replay-tracks":
        from .video_pipeline import (
            filter_source_manifest,
            replay_cache_track_manifest,
            write_json,
        )

        try:
            raw = json.loads(args.source_manifest.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("accepted"), list):
                source_manifest = raw
            elif isinstance(raw, dict):
                source_manifest = filter_source_manifest(
                    raw.get("entries") or raw.get("videos") or []
                )
            elif isinstance(raw, list):
                source_manifest = filter_source_manifest(raw)
            else:
                raise ValueError("source manifest must be an object or array")
            source = next(
                (
                    row
                    for row in source_manifest["accepted"]
                    if row.get("video_id") == args.video_id
                ),
                None,
            )
            if source is None:
                raise ValueError(f"video_id {args.video_id!r} is not in the accepted manifest")
            manifest = replay_cache_track_manifest(
                source,
                args.cache,
                hud_variant=args.hud_variant,
                confidence_threshold=args.confidence_threshold,
                minimum_track_frames=args.minimum_track_frames,
                isolation_radius_mtile=args.isolation_radius_mtile,
            )
            write_json(args.json_out, manifest)
            _write_json(
                None,
                {
                    "kind": manifest["kind"],
                    "json_out": str(args.json_out),
                    "accepted_source_count": len(manifest["accepted"]),
                    "track_count": sum(
                        len(row.get("tracks", [])) for row in manifest["accepted"]
                    ),
                },
            )
            return 0
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
            raise SystemExit(str(error)) from error

    if args.command == "mine-replay-batch":
        from .video_pipeline import (
            batch_replay_cache_track_manifest,
            filter_source_manifest,
            merge_hud_track_manifests,
            merge_track_manifests,
            write_json,
        )

        try:
            raw = json.loads(args.source_manifest.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("accepted"), list):
                source_manifest = raw
            elif isinstance(raw, dict):
                source_manifest = filter_source_manifest(
                    raw.get("entries") or raw.get("videos") or []
                )
            elif isinstance(raw, list):
                source_manifest = filter_source_manifest(raw)
            else:
                raise ValueError("source manifest must be an object or array")
            if args.hud_variant == "both":
                manifests = []
                for variant in ("standard", "alternative"):
                    roots = [
                        batch_replay_cache_track_manifest(
                            source_manifest,
                            extractor_root,
                            hud_variant=variant,
                            confidence_threshold=args.confidence_threshold,
                            minimum_track_frames=args.minimum_track_frames,
                            isolation_radius_mtile=args.isolation_radius_mtile,
                        )
                        for extractor_root in args.extractor_roots
                    ]
                    manifests.append(merge_track_manifests(roots, hud_variant=variant))
                manifest = merge_hud_track_manifests(manifests)
            else:
                roots = [
                    batch_replay_cache_track_manifest(
                        source_manifest,
                        extractor_root,
                        hud_variant=args.hud_variant,
                        confidence_threshold=args.confidence_threshold,
                        minimum_track_frames=args.minimum_track_frames,
                        isolation_radius_mtile=args.isolation_radius_mtile,
                    )
                    for extractor_root in args.extractor_roots
                ]
                manifest = merge_track_manifests(roots, hud_variant=args.hud_variant)
            write_json(args.json_out, manifest)
            _write_json(
                None,
                {
                    "kind": manifest["kind"],
                    "json_out": str(args.json_out),
                    **manifest["summary"],
                },
            )
            return 0 if manifest["summary"]["accepted_source_count"] else 2
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
            raise SystemExit(str(error)) from error

    if args.command == "discover-video-source":
        from .video_pipeline import discover_source_manifest, write_json

        try:
            manifest = discover_source_manifest(
                args.source,
                max_videos=args.max_videos,
                cookies_from_browser=args.cookies_from_browser,
            )
            write_json(args.json_out, manifest)
            _write_json(
                None,
                {
                    "kind": manifest["kind"],
                    "json_out": str(args.json_out),
                    "accepted_count": len(manifest["accepted"]),
                    "rejected_count": len(manifest["rejected"]),
                },
            )
            return 0
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
            raise SystemExit(str(error)) from error

    if args.command == "plan-action-windows":
        from .video_pipeline import (
            build_action_window_extractor_jobs,
            build_action_window_manifest,
            filter_source_manifest,
            run_extractor_jobs,
            write_json,
        )

        try:
            raw = json.loads(args.manifest.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("accepted"), list):
                source_manifest = raw
            elif isinstance(raw, dict):
                source_manifest = filter_source_manifest(
                    raw.get("entries") or raw.get("videos") or []
                )
            elif isinstance(raw, list):
                source_manifest = filter_source_manifest(raw)
            else:
                raise ValueError("source manifest must be an object or array")
            windows = build_action_window_manifest(
                source_manifest,
                args.candidates_root,
                confidence_threshold=args.confidence_threshold,
                max_windows_per_video=args.max_windows_per_video,
                window_before_s=args.window_before,
                window_after_s=args.window_after,
                minimum_window_separation_s=args.minimum_window_separation,
                split_salt=args.split_salt,
                supported_cards=args.card or None,
            )
            plan = build_action_window_extractor_jobs(
                windows,
                output_root=args.output_root,
                sample_interval_s=args.sample_interval,
                yolo_detections=not args.no_yolo,
            )
            run = run_extractor_jobs(
                plan,
                execute=args.execute,
                skip_existing=not args.rerun_existing,
                stop_on_error=args.stop_on_error,
                workspace_root=args.workspace_root,
                retention_manifest_path=args.retention_manifest,
                raw_media_root=args.raw_media_root,
                reserve_bytes=args.reserve_bytes,
                evict=args.evict,
                job_timeout_s=args.job_timeout_s,
            )
            payload = {"windows": windows, "plan": plan, "run": run}
            write_json(args.json_out, payload)
            _write_json(
                None,
                {
                    "kind": run["kind"],
                    "json_out": str(args.json_out),
                    "window_count": windows["summary"]["window_count"],
                    "job_count": len(run["jobs"]),
                    "executed": bool(args.execute),
                    "failed_count": run["failed_count"],
                },
            )
            return 0 if run["failed_count"] == 0 else 2
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
            raise SystemExit(str(error)) from error

    if args.command == "extract-video":
        from .video_pipeline import (
            build_extractor_jobs,
            filter_source_manifest,
            run_extractor_jobs,
            write_json,
        )

        try:
            raw = json.loads(args.manifest.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("accepted"), list):
                source_manifest = raw
            elif isinstance(raw, dict):
                source_manifest = filter_source_manifest(
                    raw.get("entries") or raw.get("videos") or []
                )
            elif isinstance(raw, list):
                source_manifest = filter_source_manifest(raw)
            else:
                raise ValueError("manifest must be an object or array")
            plan = build_extractor_jobs(
                source_manifest,
                output_root=args.output_root,
                sample_interval_s=args.sample_interval,
                yolo_detections=not args.no_yolo,
                video_start_time_s=args.video_start_time,
                video_duration_s=args.video_duration,
            )
            run = run_extractor_jobs(
                plan,
                execute=args.execute,
                skip_existing=not args.rerun_existing,
                stop_on_error=args.stop_on_error,
                workspace_root=args.workspace_root,
                retention_manifest_path=args.retention_manifest,
                raw_media_root=args.raw_media_root,
                reserve_bytes=args.reserve_bytes,
                evict=args.evict,
                job_timeout_s=args.job_timeout_s,
            )
            payload = {"plan": plan, "run": run}
            write_json(args.json_out, payload)
            _write_json(
                None,
                {
                    "kind": run["kind"],
                    "json_out": str(args.json_out),
                    "job_count": len(run["jobs"]),
                    "executed": bool(args.execute),
                    "failed_count": run["failed_count"],
                },
            )
            return 0 if run["failed_count"] == 0 else 2
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
            raise SystemExit(str(error)) from error

    # Imports stay here so `--help` does not initialize NumPy or the feature
    # stack and the pure headless core remains cheap to inspect.
    from .engine import ENGINE_VERSION, BattleEngine, DeterministicCycleController
    from .ruleset import load_ruleset

    ruleset = load_ruleset(args.ruleset)
    engine = BattleEngine(ruleset)

    if args.command == "generate-scenarios":
        from .scenario_factory import generate_roster_scenarios, write_generated_manifest

        rows = generate_roster_scenarios(
            ruleset,
            per_mechanic=args.per_mechanic,
            card_ids=args.card or None,
        )
        write_generated_manifest(args.json_out, rows)
        _write_json(
            None,
            {
                "kind": "simulator_generated_scenario_manifest",
                "json_out": str(args.json_out),
                "scenario_count": len(rows),
            },
        )
        return 0

    if args.command == "generate-interactions":
        from .scenario_factory import generate_interaction_scenarios, write_generated_manifest

        if args.player_card:
            rows = generate_interaction_scenarios(
                ruleset,
                opponent_card_ids=args.opponent_card or None,
                player_card_ids=args.player_card,
                variants=args.variants,
            )
        else:
            rows = generate_interaction_scenarios(
                ruleset,
                opponent_card_ids=args.opponent_card or None,
                variants=args.variants,
            )
        write_generated_manifest(args.json_out, rows)
        _write_json(
            None,
            {
                "kind": "simulator_generated_interaction_manifest",
                "json_out": str(args.json_out),
                "scenario_count": len(rows),
            },
        )
        return 0

    if args.command == "generate-opponent-pairs":
        from .scenario_factory import (
            generate_opponent_pair_scenarios,
            write_generated_manifest,
        )

        rows = generate_opponent_pair_scenarios(
            ruleset,
            opponent_card_ids=args.opponent_card or None,
            variants=args.variants,
        )
        write_generated_manifest(args.json_out, rows)
        _write_json(
            None,
            {
                "kind": "simulator_generated_opponent_pair_manifest",
                "json_out": str(args.json_out),
                "scenario_count": len(rows),
            },
        )
        return 0

    if args.command == "validate-generated":
        from .generated_validation import (
            load_generated_manifest,
            validate_complete_generated_coverage,
            validate_generated_behavioral_obligations,
            validate_generated_scenarios,
            write_generated_validation_report,
        )

        try:
            manifest_payload, scenarios = load_generated_manifest(args.manifest)
            validation_engine = BattleEngine(
                ruleset,
                validate_every_tick=not args.no_tick_validation,
            )
            report = validate_generated_scenarios(
                validation_engine,
                scenarios,
                repeats=args.repeats,
                workers=args.workers,
            )
            coverage_gate = None
            behavioral_gate = None
            if args.require_complete:
                coverage_gate = validate_complete_generated_coverage(
                    manifest_payload,
                    scenarios,
                    ruleset=ruleset,
                )
                report["coverage_gate"] = coverage_gate
                behavioral_gate = validate_generated_behavioral_obligations(scenarios)
                report["behavioral_obligation_gate"] = behavioral_gate
            write_generated_validation_report(args.json_out, report)
            _write_json(
                None,
                {
                    "kind": report["kind"],
                    "json_out": str(args.json_out),
                    "scenario_count": report["scenario_count"],
                    "passed_count": report["passed_count"],
                    "failed_count": report["failed_count"],
                    "determinism_failures": report["determinism_failures"],
                    "coverage_passed": (
                        coverage_gate is None or coverage_gate["passed"]
                    ),
                    "behavioral_obligations_passed": (
                        behavioral_gate is None or behavioral_gate["passed"]
                    ),
                },
            )
            return 0 if report["failed_count"] == 0 and (
                coverage_gate is None or coverage_gate["passed"]
            ) and (behavioral_gate is None or behavioral_gate["passed"]) else 2
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise SystemExit(str(error)) from error

    if args.command == "reconcile-data":
        from .data_reconciliation import (
            DECKSHOP_CORE_SOURCE_PATH,
            DECKSHOP_SOURCE_PATH,
            DECKSHOP_HEAL_SPIRIT_SOURCE_PATH,
            load_level11_source,
            reconcile_ruleset,
            report_is_strictly_verified,
        )

        try:
            # A bare reconciliation command audits the complete V1 opponent
            # roster.  Pass ``--ruleset 2026-08-04`` explicitly only when the
            # eight-card base interaction set is desired.
            audit_ruleset = (
                load_ruleset("2026-08-04-roster")
                if args.ruleset == "2026-08-04"
                else engine.ruleset
            )
            additional_paths = list(args.additional_source_json)
            if DECKSHOP_SOURCE_PATH not in additional_paths:
                additional_paths.insert(0, DECKSHOP_SOURCE_PATH)
            if DECKSHOP_CORE_SOURCE_PATH not in additional_paths:
                additional_paths.insert(1, DECKSHOP_CORE_SOURCE_PATH)
            if DECKSHOP_HEAL_SPIRIT_SOURCE_PATH not in additional_paths:
                additional_paths.append(DECKSHOP_HEAL_SPIRIT_SOURCE_PATH)
            additional_sources = tuple(
                (
                    str((payload := load_level11_source(path)).get("source_id", path)),
                    payload,
                )
                for path in additional_paths
            )
            # The DeckShop snapshot uses the same compact card schema; if a
            # future source has a different schema, its loader should be
            # upgraded here instead of silently treating it as evidence.
            report = reconcile_ruleset(
                audit_ruleset,
                source_path=args.source_json,
                additional_sources=additional_sources,
            )
            _write_json(args.json_out, report)
            _write_json(
                None,
                {
                    "kind": report["kind"],
                    "json_out": str(args.json_out),
                    **report["summary"],
                },
            )
            return 0 if not args.strict or report_is_strictly_verified(report) else 2
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise SystemExit(str(error)) from error

    if args.command == "compile-video-truth":
        from .mining import compile_observation_manifest, corpus_to_dict
        from .video_pipeline import video_truth_to_observation_manifest

        try:
            truth = json.loads(args.truth_manifest.read_text(encoding="utf-8"))
            if not isinstance(truth, dict):
                raise ValueError("truth manifest must be an object")
            source = None
            if args.source_manifest is not None:
                source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
                if not isinstance(source, dict):
                    raise ValueError("source manifest must be an object")
            mining_manifest = video_truth_to_observation_manifest(
                truth,
                source_manifest=source,
                corpus_id=args.corpus_id,
                tick_us=ruleset.tick_us,
                position_tolerance_mtile=args.position_tolerance_mtile,
                speed_estimator=args.speed_estimator,
            )
            result = compile_observation_manifest(mining_manifest, engine=engine)
            payload = corpus_to_dict(result.corpus)
            _write_json(args.json_out, payload)
            _write_json(
                None,
                {
                    "kind": "simulator_video_fidelity_corpus",
                    "json_out": str(args.json_out),
                    "accepted_cases": len(result.corpus.cases),
                    "discarded_cases": len(result.discarded),
                    "corpus_hash": result.corpus.content_hash,
                },
            )
            return 0
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
            raise SystemExit(str(error)) from error

    if args.command == "ruleset":
        _write_json(
            args.json_out,
            _ruleset_summary(ruleset, engine_version=ENGINE_VERSION),
        )
        return 0

    if args.command == "roster":
        from cr_bot.domain.card_metadata import CARD_METADATA

        from .roster import (
            build_mechanic_coverage,
            load_opponent_roster,
            validate_roster_against_catalog,
        )

        roster_contract = load_opponent_roster()
        catalog_report = validate_roster_against_catalog(
            roster_contract,
            CARD_METADATA,
            require_release_verification=args.require_release_verification,
        )
        coverage = build_mechanic_coverage(
            roster_contract,
            {
                card_id: {
                    "kind": definition.kind,
                    "is_air": definition.mechanics.get("movement_layer") == "air",
                    "is_splash": definition.area_radius_mtile is not None,
                    "mechanics": definition.mechanics,
                }
                for card_id, definition in ruleset.cards.items()
                if card_id in roster_contract.eligible_cards
            },
            set(ruleset.cards),
            fidelity_ready_cards=(
                set(ruleset.cards)
                if ruleset.metadata.get("training_ready") is True
                else set()
            ),
        )
        report = {
            "kind": "simulator_roster_contract",
            "ruleset_id": ruleset.ruleset_id,
            "ruleset_hash": ruleset.content_hash,
            "catalog": catalog_report,
            "coverage": coverage,
            "passed": bool(
                catalog_report["complete"]
                and (
                    not args.require_coverage
                    or (
                        coverage["all_cards_implemented"]
                        and coverage["all_cards_fidelity_ready"]
                    )
                )
            ),
        }
        _write_json(args.json_out, report)
        if args.json_out is not None:
            _write_json(None, report)
        return 0 if report["passed"] else 2

    if args.command == "run":
        state = engine.new_battle(seed=args.seed, shuffle_decks=not args.no_shuffle)
        controllers = None if args.passive else (
            DeterministicCycleController(),
            DeterministicCycleController(),
        )
        engine.run_match(state, controllers, max_ticks=args.max_ticks)
        if args.state_out:
            _write_json(args.state_out, state.to_primitive(include_events=True))
        _write_json(args.json_out, _state_summary(state))
        return 0

    if args.command == "scenario":
        from .runner import run_scenario
        from .scenario import load_scenario

        state = run_scenario(engine, load_scenario(args.path))
        if args.state_out:
            _write_json(args.state_out, state.to_primitive(include_events=True))
        _write_json(args.json_out, _state_summary(state))
        return 0

    if args.command == "fidelity":
        from .validation import apply_fidelity_gate, run_fidelity_corpus

        report = run_fidelity_corpus(
            engine,
            args.corpus,
            split=args.split,
            expected_corpus_hash=args.expected_corpus_hash,
        )
        report = apply_fidelity_gate(
            report,
            min_observations=args.min_observations,
            min_agreement_rate=args.min_agreement_rate,
            required_mechanics=args.require_mechanic,
        )
        report.write_json(args.json_out)
        print(report.to_json(), end="")
        assert report.gate is not None
        return 0 if report.gate["passed"] else 2

    if args.command == "readiness":
        from .readiness import (
            build_training_readiness_report,
            declared_mechanics_for_ruleset,
        )

        report = build_training_readiness_report(
            args.report,
            candidate_report_paths=args.candidate_report,
            ruleset_id=ruleset.ruleset_id,
            ruleset_hash=ruleset.content_hash,
            engine_version=ENGINE_VERSION,
            minimum_heldout_observations=args.min_heldout_observations,
            minimum_heldout_agreement_rate=args.min_heldout_agreement_rate,
            minimum_heldout_groups=args.min_heldout_groups,
            requirements=declared_mechanics_for_ruleset(ruleset),
        )
        if ruleset.metadata.get("training_ready") is not True:
            report["summary"]["ready"] = False
            report["summary"]["failures"].append(
                "ruleset metadata training_ready is not true; provisional definitions cannot train"
            )
        _write_json(args.json_out, report)
        _write_json(None, report)
        return 0 if report["summary"]["ready"] else 2

    if args.command == "mine-corpus":
        from .mining import (
            compile_observation_manifest,
            corpus_to_dict,
            load_observation_manifest,
        )

        result = compile_observation_manifest(
            load_observation_manifest(args.manifest),
            engine=engine,
            confidence_threshold=args.confidence_threshold,
        )
        _write_json(args.json_out, corpus_to_dict(result.corpus))
        summary = result.summary()
        if args.discarded_out:
            _write_json(args.discarded_out, summary)
        _write_json(None, summary)
        return 0

    if args.command == "mine-replay":
        from .mining import compile_replay_cache_movement, corpus_to_dict

        result = compile_replay_cache_movement(
            args.cache,
            corpus_id=args.corpus_id,
            group_id=args.group_id,
            source_level=args.source_level,
            evidence_split=args.evidence_split,
            engine=engine,
            confidence_threshold=args.confidence_threshold,
            minimum_track_frames=args.minimum_track_frames,
            minimum_displacement_mtile=args.minimum_displacement_mtile,
            isolation_radius_mtile=args.isolation_radius_mtile,
            contamination_confidence_threshold=args.contamination_confidence_threshold,
            minimum_speed_ratio_permille=args.minimum_speed_ratio_permille,
            maximum_speed_ratio_permille=args.maximum_speed_ratio_permille,
            level_invariant_current_ruleset=args.level_invariant_current_ruleset,
            expected_support_tower_hp=args.expected_support_tower_hp,
            use_expected_speed_gate=not args.kinematic_only_gate,
        )
        _write_json(args.json_out, corpus_to_dict(result.corpus))
        summary = result.summary()
        if args.discarded_out:
            _write_json(args.discarded_out, summary)
        _write_json(None, summary)
        return 0

    if args.command == "mine-pulls":
        from .mining import compile_replay_cache_hog_cannon_pulls, corpus_to_dict

        result = compile_replay_cache_hog_cannon_pulls(
            args.cache,
            ground_truth_path=args.ground_truth,
            corpus_id=args.corpus_id,
            group_id=args.group_id,
            source_level=args.source_level,
            evidence_split=args.evidence_split,
            engine=engine,
            confidence_threshold=args.confidence_threshold,
            minimum_track_frames=args.minimum_track_frames,
        )
        _write_json(args.json_out, corpus_to_dict(result.corpus))
        _write_json(None, result.summary())
        return 0

    if args.command == "discover-cannon-lifetime":
        from .mining import discover_replay_cache_cannon_lifetimes

        report = discover_replay_cache_cannon_lifetimes(
            args.cache,
            ground_truth_path=args.ground_truth,
            source_level=args.source_level,
            engine=engine,
            confidence_threshold=args.confidence_threshold,
            maximum_track_gap_s=args.maximum_track_gap_s,
        )
        _write_json(args.json_out, report)
        _write_json(
            None,
            {
                "kind": report["kind"],
                "candidate_count": len(report["candidates"]),
                "rejected_count": len(report["rejected"]),
                "json_out": str(args.json_out),
            },
        )
        return 0

    if args.command == "discover-replay-interactions":
        from .mining import discover_replay_cache_interactions_batch

        report = discover_replay_cache_interactions_batch(
            args.cache,
            source_level=args.source_level,
            engine=engine,
            level_proof_paths=args.level_proof_caches,
            level_invariant_current_ruleset=args.level_invariant_current_ruleset,
            expected_support_tower_hp=args.expected_support_tower_hp,
            confidence_threshold=args.confidence_threshold,
            contamination_confidence_threshold=args.contamination_confidence_threshold,
            minimum_track_frames=args.minimum_track_frames,
            maximum_track_gap_s=args.maximum_track_gap_s,
            isolation_radius_mtile=args.isolation_radius_mtile,
        )
        _write_json(args.json_out, report)
        _write_json(
            None,
            {
                "kind": report["kind"],
                "source_count": report["source_count"],
                "failed_source_count": report["failed_source_count"],
                "candidate_count": report["candidate_count"],
                "rejected_count": report["rejected_count"],
                "json_out": str(args.json_out),
            },
        )
        # A batch can still be useful when one cache is malformed, but a
        # completely failed batch must fail closed for automation/CI.
        return 0 if report["source_count"] and not (
            report["failed_source_count"] and not report["candidate_count"]
        ) else 2

    if args.command == "merge-replay-interactions":
        from .mining import merge_replay_interaction_reports

        report = merge_replay_interaction_reports(
            args.report,
            engine=engine,
            onset_tolerance_ms=args.onset_tolerance_ms,
            position_tolerance_mtile=args.position_tolerance_mtile,
            require_both_hud=args.require_both_hud,
        )
        _write_json(args.json_out, report)
        _write_json(
            None,
            {
                "kind": report["kind"],
                "source_count": report["source_count"],
                "failed_source_count": report["failed_source_count"],
                "candidate_count": report["candidate_count"],
                "rejected_count": report["rejected_count"],
                "json_out": str(args.json_out),
            },
        )
        if args.require_both_hud:
            return 0 if report["candidate_count"] and not report["failed_source_count"] else 2
        return 0 if report["source_count"] else 2

    if args.command == "discover-tower-damage":
        from .mining import discover_replay_cache_tower_damage

        report = discover_replay_cache_tower_damage(
            args.cache,
            source_level=args.source_level,
            engine=engine,
            confidence_threshold=args.confidence_threshold,
            minimum_plateau_frames=args.minimum_plateau_frames,
        )
        _write_json(args.json_out, report)
        _write_json(
            None,
            {
                "kind": report["kind"],
                "candidate_count": len(report["candidates"]),
                "interval_count": len(report["intervals"]),
                "rejected_count": len(report["rejected"]),
                "json_out": str(args.json_out),
            },
        )
        return 0

    if args.command == "discover-log-motion":
        from .mining import discover_replay_cache_log_motion

        report = discover_replay_cache_log_motion(
            args.cache,
            ground_truth_path=args.ground_truth,
            source_level=args.source_level,
            engine=engine,
            confidence_threshold=args.confidence_threshold,
            minimum_moving_steps=args.minimum_moving_steps,
        )
        _write_json(args.json_out, report)
        _write_json(
            None,
            {
                "kind": report["kind"],
                "candidate_count": len(report["candidates"]),
                "rejected_count": len(report["rejected"]),
                "json_out": str(args.json_out),
            },
        )
        return 0

    if args.command == "discover-fireball-flight":
        from .mining import discover_replay_cache_fireball_flights

        report = discover_replay_cache_fireball_flights(
            args.cache,
            ground_truth_path=args.ground_truth,
            source_level=args.source_level,
            engine=engine,
            confidence_threshold=args.confidence_threshold,
            minimum_flight_samples=args.minimum_flight_samples,
        )
        _write_json(args.json_out, report)
        _write_json(
            None,
            {
                "kind": report["kind"],
                "candidate_count": len(report["candidates"]),
                "rejected_count": len(report["rejected"]),
                "json_out": str(args.json_out),
            },
        )
        return 0

    if args.command == "check-determinism":
        if args.repeats < 2:
            raise SystemExit("--repeats must be at least 2")
        hashes = []
        replay_hashes = []
        for _ in range(args.repeats):
            state = engine.new_battle(seed=args.seed)
            engine.run_match(
                state,
                (DeterministicCycleController(), DeterministicCycleController()),
            )
            hashes.append(state.state_hash())
            replay_hashes.append(state.replay_hash())
        result = {
            "engine_version": ENGINE_VERSION,
            "ruleset_id": ruleset.ruleset_id,
            "ruleset_hash": ruleset.content_hash,
            "seed": args.seed,
            "repeats": args.repeats,
            "deterministic": len(set(hashes)) == 1 and len(set(replay_hashes)) == 1,
            "state_hashes": hashes,
            "replay_hashes": replay_hashes,
        }
        _write_json(None, result)
        return 0 if result["deterministic"] else 1

    if args.command in {"audit", "soak"}:
        from .audit import run_determinism_audit, run_soak_audit

        if args.command == "audit":
            report = run_determinism_audit(
                seed_count=args.seeds,
                seed_start=args.seed_start,
                max_ticks_per_seed=args.max_ticks,
                decision_interval_ticks=args.decision_ticks,
                engine_factory=lambda: BattleEngine(ruleset),
            )
        else:
            report = run_soak_audit(
                seed_count=args.seeds,
                seed_start=args.seed_start,
                tick_budget=args.tick_budget,
                max_ticks_per_seed=args.max_ticks,
                decision_interval_ticks=args.decision_ticks,
                engine_factory=lambda: BattleEngine(ruleset),
            )
        _write_json(args.json_out, report.to_dict())
        return 0

    if args.command == "benchmark":
        if args.matches <= 0:
            raise SystemExit("--matches must be positive")
        benchmark_engine = BattleEngine(
            ruleset,
            validate_every_tick=args.strict_validation,
        )
        started = perf_counter()
        ticks = 0
        hashes = []
        for offset in range(args.matches):
            state = benchmark_engine.new_battle(seed=args.seed + offset)
            benchmark_engine.run_match(
                state,
                (DeterministicCycleController(), DeterministicCycleController()),
            )
            ticks += state.tick
            hashes.append(state.state_hash())
        duration = perf_counter() - started
        _write_json(
            None,
            {
                "engine_version": ENGINE_VERSION,
                "ruleset_id": ruleset.ruleset_id,
                "ruleset_hash": ruleset.content_hash,
                "matches": args.matches,
                "strict_validation": args.strict_validation,
                "physics_ticks": ticks,
                "wall_seconds": duration,
                "ticks_per_second": ticks / duration,
                "state_hashes": hashes,
                "note": "Python deterministic reference throughput; not a live-game fidelity metric.",
            },
        )
        return 0

    if args.command == "benchmark-vector":
        if args.envs <= 0 or args.steps <= 0:
            raise SystemExit("--envs and --steps must be positive")
        from .env import VectorSimulatorEnv

        started = perf_counter()
        with VectorSimulatorEnv.create(
            args.envs,
            backend=args.backend,
            workers=args.workers,
        ) as vector:
            vector.reset(tuple(args.seed + index for index in range(args.envs)))
            completed_steps = 0
            for _ in range(args.steps):
                if any(environment.state is not None and environment.state.terminal for environment in vector.environments):
                    break
                vector.step(tuple((None, None) for _ in range(args.envs)))
                completed_steps += 1
            state_hashes = [
                environment.state.state_hash()
                for environment in vector.environments
                if environment.state is not None
            ]
        duration = perf_counter() - started
        result = {
            "engine_version": ENGINE_VERSION,
            "ruleset_id": ruleset.ruleset_id,
            "ruleset_hash": ruleset.content_hash,
            "backend": args.backend,
            "workers": args.workers,
            "environments": args.envs,
            "requested_steps": args.steps,
            "completed_steps": completed_steps,
            "wall_seconds": duration,
            "environment_steps_per_second": (
                args.envs * completed_steps / duration if duration else 0.0
            ),
            "state_hashes": state_hashes,
            "note": "Deterministic policy-boundary throughput; not a live-game fidelity metric.",
        }
        _write_json(args.json_out, result)
        return 0

    if args.command == "train":
        from .trainer import PPOConfig, PPOTrainer, TrainingConfigurationError
        from .training_profiles import TrainingProfile, TrainingProfileError

        opponent = "scripted" if args.opponent == "deterministic-cycle" else args.opponent
        training_profile = None
        if args.training_profile is not None:
            try:
                training_profile = TrainingProfile.from_json(args.training_profile)
            except TrainingProfileError as error:
                raise SystemExit(str(error)) from error
        config = PPOConfig(
            ruleset_id=ruleset.ruleset_id,
            training_profile=training_profile,
            num_envs=args.envs,
            backend=args.backend,
            workers=args.workers,
            rollout_steps=args.rollout_steps,
            total_steps=args.steps,
            update_epochs=args.update_epochs,
            learning_rate=args.learning_rate,
            entropy_coef=args.entropy_coef,
            seed=args.seed,
            opponent=opponent,
            checkpoint_out=args.checkpoint_out,
            checkpoint_every=args.checkpoint_every,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
            eval_max_decisions=args.eval_max_decisions,
            allow_provisional_smoke=args.allow_provisional_smoke,
        )
        try:
            trainer = PPOTrainer(config)
            if args.checkpoint is not None:
                trainer.load_checkpoint(args.checkpoint)
            report = trainer.train()
        except TrainingConfigurationError as error:
            raise SystemExit(str(error)) from error
        _write_json(args.json_out, report)
        if args.json_out is not None:
            _write_json(
                None,
                {
                    "kind": report["kind"],
                    "ruleset_id": report["ruleset_id"],
                    "total_steps": report["total_steps"],
                    "episodes": report["episodes"],
                    "checkpoint": report["checkpoint"],
                    "json_out": str(args.json_out),
                },
            )
        return 0

    if args.command == "evaluate":
        from .trainer import FactorizedPolicy, evaluate_policy
        from .observation import PINNED_OBSERVATION_CONTRACT_HASH
        from .training_profiles import TrainingProfile, TrainingProfileError, validate_training_profile

        policy, metadata = FactorizedPolicy.load(
            args.checkpoint,
            expected_metadata={
                "ruleset_id": ruleset.ruleset_id,
                "ruleset_hash": ruleset.content_hash,
                "engine_version": ENGINE_VERSION,
                "observation_contract_hash": PINNED_OBSERVATION_CONTRACT_HASH,
            },
        )
        opponent_value = args.opponent or str(metadata.get("opponent", "scripted"))
        if opponent_value == "deterministic-cycle":
            opponent_value = "scripted"
        if opponent_value not in {"scripted", "self-play"}:
            raise SystemExit(f"unsupported checkpoint opponent: {opponent_value!r}")
        reward_version = str(metadata.get("reward_version", "terminal-outcome-v1"))
        if reward_version not in {"terminal-outcome-v1", "tower-damage-crowns-v1"}:
            raise SystemExit(f"unsupported checkpoint reward version: {reward_version!r}")
        profile_result = None
        if args.training_profile is not None:
            try:
                profile = TrainingProfile.from_json(args.training_profile)
                profile_result = validate_training_profile(profile, ruleset=ruleset)
            except TrainingProfileError as error:
                raise SystemExit(str(error)) from error
            checkpoint_profile_id = metadata.get("training_profile_id")
            if checkpoint_profile_id not in {None, profile.profile_id}:
                raise SystemExit("checkpoint belongs to a different training profile")
        elif metadata.get("training_profile_purpose") in {"training", "evaluation"}:
            raise SystemExit(
                "this checkpoint has a serious training profile; pass --training-profile for evaluation"
            )
        report = evaluate_policy(
            policy,
            ruleset=ruleset,
            opponent=opponent_value,
            episodes=args.episodes,
            seed_start=args.seed,
            max_decisions=args.max_decisions,
            reward_version=reward_version,
        )
        report.update(
            {
                "kind": "simulator_ppo_checkpoint_evaluation",
                "checkpoint": str(args.checkpoint),
                "ruleset_id": ruleset.ruleset_id,
                "ruleset_hash": ruleset.content_hash,
                "engine_version": ENGINE_VERSION,
                "observation_contract_hash": PINNED_OBSERVATION_CONTRACT_HASH,
                "reward_version": reward_version,
                "training_profile": profile_result,
            }
        )
        _write_json(args.json_out, report)
        if args.json_out is not None:
            _write_json(None, report)
        return 0
    raise AssertionError(args.command)
