#!/usr/bin/env python3
"""Create and re-evaluate the immutable physical-fidelity interaction campaign.

Plan the deterministic deck/action matrix:

    outputs/venv/bin/python scripts/run_physical_fidelity_campaign.py plan

Re-evaluate every admitted case corpus after a simulator change:

    outputs/venv/bin/python scripts/run_physical_fidelity_campaign.py evaluate \
      --campaign outputs/simulator/fidelity_media/physical_lab/campaigns/\
physical-fidelity-interaction-sweep-v1/campaign.json \
      --results-root outputs/simulator/fidelity_media/physical_lab/campaigns/\
physical-fidelity-interaction-sweep-v1/results \
      --json-out outputs/simulator/fidelity_media/physical_lab/campaign-evaluation.json

The plan and each case file are write-once artifacts.  A different simulator
result must use a new evaluation output path; existing physical evidence is
never rewritten in place.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator.physical_lab.campaign import (  # noqa: E402
    DEFAULT_CAMPAIGN_ID,
    DEFAULT_CAMPAIGN_ROOT,
    build_default_campaign,
    evaluate_campaign,
    load_campaign,
    materialize_campaign,
    write_campaign_evaluation,
)
from simulator.physical_lab.schema import PhysicalLabError  # noqa: E402
from simulator.storage import (  # noqa: E402
    DEFAULT_LOW_WATER_BYTES,
    DEFAULT_MAX_WORKSPACE_BYTES,
    enforce_workspace_budget,
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _rooted(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def _write_json(path: Path | None, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path is None:
        print(encoded, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Physical-fidelity interaction campaign manager")
    parser.add_argument("--repository-root", type=Path, default=_repository_root())
    parser.add_argument("--max-workspace-bytes", type=int, default=DEFAULT_MAX_WORKSPACE_BYTES)
    parser.add_argument("--low-water-bytes", type=int, default=DEFAULT_LOW_WATER_BYTES)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="create the immutable simple-to-complex campaign")
    plan.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID)
    plan.add_argument("--evidence-split", default="calibration")
    plan.add_argument("--output-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    plan.add_argument("--json-out", type=Path)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="re-run every stored case corpus with the current simulator",
    )
    evaluate.add_argument("--campaign", type=Path, required=True)
    evaluate.add_argument("--results-root", type=Path, required=True)
    evaluate.add_argument("--split")
    evaluate.add_argument("--min-observations", type=int, default=1)
    evaluate.add_argument("--min-agreement-rate", type=float)
    evaluate.add_argument("--json-out", type=Path, required=True)
    return parser


def _budget(args: argparse.Namespace, repository_root: Path) -> dict[str, object]:
    if args.max_workspace_bytes > DEFAULT_MAX_WORKSPACE_BYTES:
        raise PhysicalLabError("--max-workspace-bytes cannot exceed 200,000,000,000")
    if args.max_workspace_bytes <= 0 or not 0 <= args.low_water_bytes <= args.max_workspace_bytes:
        raise PhysicalLabError("invalid workspace budget values")
    return enforce_workspace_budget(
        workspace_root=repository_root,
        manifest_path=repository_root / "outputs/simulator/fidelity_media/retention.json",
        raw_media_root=repository_root / "outputs/simulator/fidelity_media",
        max_bytes=args.max_workspace_bytes,
        low_water_bytes=args.low_water_bytes,
        reserve_bytes=0,
        evict=False,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = args.repository_root.resolve()
    try:
        before = _budget(args, repository_root)
        if not before["passed"]:
            payload = {"kind": "physical_lab_campaign", "status": "blocked_workspace_budget", "budget": before}
            _write_json(args.json_out, payload)
            return 2
        if args.command == "plan":
            campaign = build_default_campaign(
                campaign_id=args.campaign_id,
                evidence_split=args.evidence_split,
            )
            materialized = materialize_campaign(campaign, _rooted(repository_root, args.output_root))
            payload = {
                "kind": "physical_lab_campaign_plan",
                "status": "planned",
                **materialized,
                "case_count": len(campaign.cases),
                "budget": before,
            }
            _write_json(args.json_out, payload)
            return 0

        campaign = load_campaign(_rooted(repository_root, args.campaign))
        result = evaluate_campaign(
            campaign,
            results_root=_rooted(repository_root, args.results_root),
            split=args.split,
            min_observations=args.min_observations,
            min_agreement_rate=args.min_agreement_rate,
        )
        result["budget_before"] = before
        result["budget_after"] = _budget(args, repository_root)
        result["evaluation_hash"] = write_campaign_evaluation(
            _rooted(repository_root, args.json_out),
            result,
        )
        _write_json(None, result)
        return 0 if result["missing_or_rejected_case_count"] == 0 else 2
    except (OSError, PhysicalLabError, ValueError, json.JSONDecodeError) as error:
        _write_json(None, {"kind": "physical_lab_campaign_error", "status": "rejected", "error": str(error)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
