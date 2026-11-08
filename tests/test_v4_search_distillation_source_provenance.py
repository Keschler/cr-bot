"""Entry metadata must match the source matches used for replay."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_v4_search_distillation.py"
spec = importlib.util.spec_from_file_location("search_distillation_runner_provenance", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize("field", ("state_hash", "match_seed"))
def test_disjoint_source_check_rejects_entry_provenance_detached_from_source(
    tmp_path: Path, field: str,
) -> None:
    roots = (tmp_path / "train", tmp_path / "validation")
    entries = []
    for index, root in enumerate(roots):
        root.mkdir()
        seed = 100 + index
        state_hash = f"source-state-{index}"
        match = {
            "config": {
                "archetype": "hog",
                "match_index": 0,
                "match_seed": seed,
            },
            "decisions": [{"state_hash": "initial"}, {"state_hash": state_hash}],
        }
        (root / "train_hog_0_match.json").write_text(json.dumps(match))
        provenance = {"state_hash": state_hash, "match_seed": seed}
        if index == 0:
            provenance[field] = "detached" if field == "state_hash" else seed + 999
        entries.append([{
            "entry_id": "train:hog#0:t1",
            "provenance": provenance,
        }])

    args = SimpleNamespace(
        dataset_root=roots[0], validation_root=roots[1],
        max_retention=0, validation_retention=0,
    )
    with pytest.raises(ValueError, match=f"entry {field} differs from its source"):
        runner._check_disjoint_sources(args, entries[0], entries[1])
