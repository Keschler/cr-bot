from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_pipeline import (
    MODEL_COST_MULTIPLIERS,
    normalize_enemy_unit_decision_roles,
    normalize_enemy_spell_decision_artifacts,
    normalize_enemy_spell_confirmation_artifacts,
    validate_own_semantic_decisions,
    validate_own_adjudication_decisions,
    validate_own_release_review_decisions,
    validate_own_slot_interval_decisions,
    validate_enemy_existence_decisions,
    validate_enemy_card_decisions,
    validate_enemy_identity_decisions,
    validate_enemy_side_check_decisions,
    validate_enemy_spell_decisions,
    validate_enemy_spell_confirmation_decisions,
    validate_enemy_unit_scan_decisions,
    validate_enemy_unit_decisions,
)
from cr_bot.own_localization import validate_own_localization_decisions


QUOTA_MARKERS = (
    "You've hit your usage limit",
    "usage limit",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def model_cost_multiplier(model: str) -> float:
    normalized = model.lower()
    for model_name, multiplier in MODEL_COST_MULTIPLIERS.items():
        if model_name in normalized:
            return multiplier
    raise ValueError(
        f"unknown model pricing for {model!r}; add an explicit multiplier"
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_image_paths(package_path: Path, run_dir: Path) -> list[Path]:
    """Resolve every review image cited by a worker package.

    Codex CLI workers cannot be trusted to call an image-viewing tool merely
    because the prompt contains filesystem paths. Attach the package's exact
    images to the initial multimodal prompt instead.
    """

    package = json.loads(package_path.read_text(encoding="utf-8"))
    resolved_run_dir = run_dir.resolve()
    images: list[Path] = []
    seen: set[Path] = set()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                visit(nested)
            return
        if isinstance(value, list):
            for nested in value:
                visit(nested)
            return
        if not isinstance(value, str):
            return
        path = Path(value)
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            return
        candidate = path if path.is_absolute() else resolved_run_dir / path
        candidate = candidate.resolve()
        if not candidate.is_relative_to(resolved_run_dir):
            raise ValueError(
                f"package image escapes run directory: {value!r}"
            )
        if not candidate.is_file():
            raise FileNotFoundError(f"package image does not exist: {candidate}")
        if candidate not in seen:
            seen.add(candidate)
            images.append(candidate)

    explicit_images = package.get("attached_images")
    if explicit_images is not None:
        if not isinstance(explicit_images, list):
            raise ValueError("attached_images must be a list when present")
        visit(explicit_images)
    else:
        visit(package)
    return images


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _token_count(transcript: str) -> int | None:
    match = re.search(r"tokens used\s*\n\s*([\d,]+)", transcript)
    return None if match is None else int(match.group(1).replace(",", ""))


def recover_json_object(message: str) -> dict[str, object] | None:
    """Recover a worker JSON object from its final response.

    Workers are instructed to write the output file, but Codex may occasionally
    return the requested object as its final message instead. Recovering that
    object before semantic validation avoids wasting an otherwise complete
    multimodal review. Validation remains the authority on whether it is usable.
    """

    candidates = [message.strip()]
    candidates.extend(
        match.group(1).strip()
        for match in re.finditer(
            r"```(?:json)?\s*([\s\S]*?)```", message, flags=re.IGNORECASE
        )
    )
    decoder = json.JSONDecoder()
    for candidate in reversed(candidates):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            value = None
            for offset, char in enumerate(candidate):
                if char != "{":
                    continue
                try:
                    parsed, _ = decoder.raw_decode(candidate[offset:])
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    value = parsed
        if isinstance(value, dict):
            return value
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Launch one model-pinned Codex CLI worker and retain its transcript. "
            "This uses the local Codex subscription, not the OpenAI API."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high", "xhigh"],
        default="medium",
    )
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--expected-output", type=Path)
    parser.add_argument("--expected-stage")
    parser.add_argument(
        "--expected-package",
        type=Path,
        help="Validate worker run_id and target_range against this package.",
    )
    parser.add_argument("--promote-to", type=Path)
    parser.add_argument("--result-file", type=Path)
    parser.add_argument(
        "--prompt-var",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional placeholder available to the prompt template.",
    )
    args = parser.parse_args()

    prompt_vars = {}
    for item in args.prompt_var:
        if "=" not in item:
            parser.error("--prompt-var must use KEY=VALUE")
        key, value = item.split("=", maxsplit=1)
        if not key or key in prompt_vars:
            parser.error(f"invalid or duplicate prompt variable {key!r}")
        prompt_vars[key] = value
    prompt = args.prompt_file.read_text(encoding="utf-8").format(
        RUN_DIR=str(args.run_dir.resolve()),
        SESSION_ID=args.session_id,
        MODEL=args.model,
        REASONING_EFFORT=args.reasoning_effort,
        **prompt_vars,
    )
    args.log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    transcript = args.log_dir / f"{timestamp}-{args.label}.log"
    final_message = args.log_dir / f"{timestamp}-{args.label}.final.txt"
    attached_images = (
        package_image_paths(args.expected_package, args.run_dir)
        if args.expected_package is not None
        else []
    )
    command = [
        "codex",
        "exec",
        # Worker transcripts and outputs are persisted by this harness under
        # RUN_DIR.  Ephemeral Codex sessions avoid creating an additional
        # persistent conversation outside the repository while retaining the
        # model's local-subscription authentication.
        "--ephemeral",
        "-C",
        str(args.workdir.resolve()),
        "--model",
        args.model,
        "-c",
        f'model_reasoning_effort="{args.reasoning_effort}"',
        "--sandbox",
        "workspace-write",
        "--output-last-message",
        str(final_message.resolve()),
    ]
    if attached_images:
        command.extend(["--image", *(str(path) for path in attached_images)])
    with transcript.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            input=prompt,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    transcript_text = transcript.read_text(encoding="utf-8")
    quota_exhausted = any(marker in transcript_text for marker in QUOTA_MARKERS)
    raw_tokens = _token_count(transcript_text)
    multiplier = model_cost_multiplier(args.model)
    status = "succeeded" if completed.returncode == 0 else "failed"
    output_sha256 = None
    validation_error = None
    if quota_exhausted:
        status = "paused_quota"
    if status == "succeeded" and args.expected_output is not None:
        try:
            if not args.expected_output.is_file() and final_message.is_file():
                recovered = recover_json_object(
                    final_message.read_text(encoding="utf-8")
                )
                if recovered is not None:
                    _atomic_json(args.expected_output, recovered)
            document = json.loads(
                args.expected_output.read_text(encoding="utf-8")
            )
            if not isinstance(document, dict):
                raise ValueError("worker output must be a JSON object")
            if (
                args.expected_stage is not None
                and document.get("stage") != args.expected_stage
            ):
                raise ValueError(
                    f"expected stage {args.expected_stage!r}, "
                    f"got {document.get('stage')!r}"
                )
            if args.expected_package is not None:
                package = json.loads(
                    args.expected_package.read_text(encoding="utf-8")
                )
                # Package metadata is harness-owned, not a semantic model
                # decision. Normalize it before validating the actual result.
                document["run_id"] = package.get("run_id")
                if "target_range" in package:
                    document["target_range"] = package["target_range"]
                document["annotation_session_id"] = args.session_id
                document["model"] = args.model
                document["reasoning_effort"] = args.reasoning_effort
                _atomic_json(args.expected_output, document)
                if document.get("run_id") != package.get("run_id"):
                    raise ValueError("worker output run_id does not match package")
                if (
                    "target_range" in package
                    and document.get("target_range")
                    != package.get("target_range")
                ):
                    raise ValueError(
                        "worker output target_range does not match package"
                    )
                if args.expected_stage in {
                    "enemy_unit_onsets_chunk",
                    "enemy_unit_completeness_chunk",
                }:
                    if normalize_enemy_unit_decision_roles(document, package):
                        _atomic_json(args.expected_output, document)
                    validate_enemy_unit_decisions(document, package)
                elif args.expected_stage == "own_semantics_chunk":
                    validate_own_semantic_decisions(
                        document,
                        package,
                        require_candidate_coverage=True,
                    )
                elif args.expected_stage == "own_completeness_chunk":
                    validate_own_semantic_decisions(
                        document,
                        package,
                        require_candidate_coverage=False,
                    )
                elif args.expected_stage == "own_adjudication_chunk":
                    validate_own_adjudication_decisions(document, package)
                elif args.expected_stage == "own_release_review_chunk":
                    validate_own_release_review_decisions(document, package)
                elif args.expected_stage == "own_slot_intervals_chunk":
                    for row in document.get("decisions", []):
                        if (
                            isinstance(row, dict)
                            and row.get("card") == "the-log"
                        ):
                            row["card"] = "log"
                    _atomic_json(args.expected_output, document)
                    validate_own_slot_interval_decisions(document, package)
                elif args.expected_stage == "enemy_identities_chunk":
                    validate_enemy_identity_decisions(document, package)
                elif (
                    args.expected_stage
                    == "enemy_overlap_adjudication_chunk"
                ):
                    validate_enemy_existence_decisions(document, package)
                elif args.expected_stage == "enemy_side_check_chunk":
                    validate_enemy_side_check_decisions(document, package)
                elif args.expected_stage == "enemy_spell_onsets_chunk":
                    if normalize_enemy_spell_decision_artifacts(
                        document, package
                    ):
                        _atomic_json(args.expected_output, document)
                    validate_enemy_spell_decisions(document, package)
                elif (
                    args.expected_stage
                    == "enemy_spell_confirmation_chunk"
                ):
                    if normalize_enemy_spell_confirmation_artifacts(
                        document, package
                    ):
                        _atomic_json(args.expected_output, document)
                    validate_enemy_spell_confirmation_decisions(
                        document, package
                    )
                elif args.expected_stage == "enemy_unit_scan_chunk":
                    validate_enemy_unit_scan_decisions(document, package)
                elif args.expected_stage == "enemy_cards_chunk":
                    validate_enemy_card_decisions(document, package)
                elif args.expected_stage == "own_localization_chunk":
                    validate_own_localization_decisions(document, package)
            output_sha256 = _sha256_file(args.expected_output)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            status = "invalid_output"
            validation_error = str(error)
    if status == "succeeded" and args.promote_to is not None:
        if args.expected_output is None:
            parser.error("--promote-to requires --expected-output")
        args.promote_to.parent.mkdir(parents=True, exist_ok=True)
        args.expected_output.replace(args.promote_to)
        output_sha256 = _sha256_file(args.promote_to)
    result = {
        "status": status,
        "label": args.label,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "session_id": args.session_id,
        "prompt_sha256": _sha256_text(prompt),
        "attached_images": [str(path) for path in attached_images],
        "attached_image_sha256s": {
            str(path): _sha256_file(path) for path in attached_images
        },
        "exit_code": completed.returncode,
        "transcript": str(transcript.resolve()),
        "final_message": str(final_message.resolve()),
        "raw_tokens": raw_tokens,
        "cost_multiplier": multiplier,
        "weighted_tokens": (
            None if raw_tokens is None else raw_tokens * multiplier
        ),
        "output_sha256": output_sha256,
        "validation_error": validation_error,
    }
    if args.result_file is not None:
        _atomic_json(args.result_file, result)
    print(
        f"worker={args.label} model={args.model} effort={args.reasoning_effort} "
        f"status={status} exit_code={completed.returncode} "
        f"raw_tokens={raw_tokens} weighted_tokens={result['weighted_tokens']} "
        f"transcript={transcript} final={final_message}"
    )
    if status == "succeeded":
        raise SystemExit(0)
    if status == "paused_quota":
        raise SystemExit(75)
    if status == "invalid_output":
        raise SystemExit(65)
    raise SystemExit(completed.returncode or 1)


if __name__ == "__main__":
    main()
