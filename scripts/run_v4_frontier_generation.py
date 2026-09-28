"""Generate disjoint V4 frontier datasets with resumable per-match workers."""
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import sys
from dataclasses import asdict
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="V4 actor checkpoint produced by the bootstrap or a previous training run")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--reuse-matches-from", type=Path, default=None,
        help="copy matching source match JSONs and rescore them; source verdicts are never reused",
    )
    p.add_argument("--train-matches-per-archetype", type=positive, default=20)
    p.add_argument("--train-profile", choices=("pinned", "mixed", "real"), default="pinned")
    p.add_argument("--sealed-matches-per-archetype", type=positive, default=8)
    p.add_argument("--train-pool-seed", type=int, default=20261001)
    p.add_argument("--sealed-pool-seed", type=int, default=20261002)
    p.add_argument("--candidates-per-match", type=positive, default=16)
    p.add_argument("--replicas", type=positive, default=8,
                   help="discovery replicas per candidate")
    p.add_argument("--confirmation-replicas", type=positive, default=None,
                   help="independent held-out replicas (default: same as --replicas)")
    p.add_argument("--branches", type=positive, default=7)
    p.add_argument("--horizon", type=positive, default=128)
    p.add_argument("--max-decisions", type=positive, default=1200)
    p.add_argument("--max-entries-per-split", type=positive, default=512)
    p.add_argument("--per-archetype-cap", type=positive, default=96)
    p.add_argument("--workers", type=positive, default=3)
    return p


def save(path,data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data,indent=2,default=str)+'\n')
    temporary.replace(path)


def _load_compatible_verdict(path):
    verdict = json.loads(path.read_text())
    required = {
        "version", "best_action", "replica_indices", "discovery_replica_indices",
        "confirmation_replica_indices", "frozen_confirmation", "confirmation",
    }
    missing = sorted(required.difference(verdict))
    if missing:
        raise ValueError(
            f"existing verdict {path} lacks independent-confirmation fields {missing}; "
            "use a new --out or rescore this candidate"
        )
    return verdict

def run_match(task):
    import torch
    from simulator.rl.v4.frontier import PoolMatchConfig, FilterConfig, collect_pool_match, select_candidates, frontier_model_config
    from simulator.rl.v4.frontier_eval import (
        EvalConfig, evaluate_candidate, materialize_entries, replay_candidate_states,
    )
    from simulator.rl.v4.v4_ppo import load_frozen_anchor
    config, split,seed,arch,index=task
    real_profile = config["train_profile"] == "real"
    allow_variants = not real_profile and (split == "sealed" or config["train_profile"] == "mixed")
    ROOT = Path(config["out"])
    policy=load_frozen_anchor(config["checkpoint"],config=frontier_model_config(),device='cpu')
    key=f'{arch}#{index}'
    path=ROOT/f'{split}_{arch}_{index}_match.json'
    if path.exists():
        match=json.loads(path.read_text())
    else:
        reuse_root = config.get("reuse_matches_from")
        source_path = (
            Path(reuse_root) / f'{split}_{arch}_{index}_match.json'
            if reuse_root else None
        )
        if source_path is not None and source_path.is_file():
            match=json.loads(source_path.read_text())
            source_config = match.get("config", {})
            expected = {
                "archetype": arch,
                "match_index": index,
                "pool_seed": seed,
                "allow_variants": allow_variants,
                "deck_profile": "real" if real_profile else "scripted",
                "real_deck_split": ("eval" if split == "sealed" else "train") if real_profile else None,
            }
            mismatches = {
                key: (source_config.get(key), value)
                for key, value in expected.items()
                if source_config.get(key) != value
            }
            if mismatches:
                raise ValueError(
                    f"cannot reuse {source_path}: source match config mismatches {mismatches}"
                )
            if int(source_config.get("max_decisions", -1)) != int(config["max_decisions"]):
                raise ValueError(
                    f"cannot reuse {source_path}: max_decisions differs from requested generation"
                )
        else:
            match=collect_pool_match(policy=policy,cfg=PoolMatchConfig(
                archetype=arch,match_index=index,pool_seed=seed,
                max_decisions=config["max_decisions"],allow_variants=allow_variants,
                deck_profile="real" if real_profile else "scripted",
                real_deck_split="eval" if split == "sealed" else "train",
            ),device='cpu')
        save(path,match)
    candidates=select_candidates(match,FilterConfig(max_per_match=config["candidates_per_match"]))
    prepared_states = replay_candidate_states(
        match, [int(candidate["t"]) for candidate in candidates], policy, device='cpu'
    )
    verdicts=[]
    for candidate in candidates:
        path=ROOT/f'{split}_{arch}_{index}_t{candidate["t"]}_verdict.json'
        if path.exists():
            verdict=_load_compatible_verdict(path)
        else:
            discovery_replicas = int(config["replicas"])
            confirmation_replicas = int(config.get("confirmation_replicas") or discovery_replicas)
            verdict=evaluate_candidate(
                match=match,
                candidate=candidate,
                policy=policy,
                eval_cfg=EvalConfig(
                    horizon=config["horizon"],
                    n_replicas=discovery_replicas,
                    max_branches=config["branches"],
                    replica_seed_base=seed,
                    replica_indices=tuple(range(discovery_replicas)),
                ),
                device='cpu',
                prepared_state=prepared_states[int(candidate["t"])],
            )
            # Freeze the discovery winner, then evaluate that same complete
            # action identity on replicas whose seeds were never used for
            # selection.  The merged verdict keeps discovery aggregates in
            # the legacy fields and records both replica sets explicitly.
            confirmation=evaluate_candidate(
                match=match,
                candidate=candidate,
                policy=policy,
                eval_cfg=EvalConfig(
                    horizon=config["horizon"],
                    n_replicas=confirmation_replicas,
                    max_branches=config["branches"],
                    replica_seed_base=seed,
                    replica_indices=tuple(range(discovery_replicas, discovery_replicas + confirmation_replicas)),
                    frozen_action=verdict.get("best_action"),
                ),
                device='cpu',
                prepared_state=prepared_states[int(candidate["t"])],
            )
            from simulator.rl.v4.frontier_eval import merge_independent_confirmation
            verdict=merge_independent_confirmation(
                verdict,
                confirmation,
                frozen_action=verdict.get("best_action"),
            )
            save(path,verdict)
        verdicts.append(verdict)
    entries=materialize_entries(
        {key:match},[v for v in verdicts if v['class']=='frontier'],policy,
        split=split,device='cpu',prepared_states=prepared_states,
    )
    torch.save(entries,ROOT/f'{split}_{arch}_{index}_entries.pt')
    return split,key,verdicts,len(entries)

def main():
    import torch
    from simulator.rl.v4.frontier import DISCOVERY_ARCHETYPES
    from simulator.rl.v4.raw_ppo import MIXED_ARCHETYPES
    from simulator.rl.v4.real_decks import REAL_DECK_ARCHETYPES, catalog_sha256
    from simulator.rl.v4.frontier_eval import build_distribution
    from simulator.rl.v4.learning_curve import validate_entry_split
    from simulator.rl.v4.frontier import FilterConfig
    from simulator.rl.v4.frontier_eval import ClassifyConfig
    args = parser().parse_args()
    if args.train_pool_seed == args.sealed_pool_seed:
        raise ValueError("train and sealed pool seeds must differ")
    if args.branches < 2:
        raise ValueError("at least two branches are required")
    if args.confirmation_replicas is not None and args.confirmation_replicas < 1:
        raise ValueError("confirmation replicas must be positive")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if args.reuse_matches_from is not None:
        args.reuse_matches_from = args.reuse_matches_from.expanduser().resolve()
        if not args.reuse_matches_from.is_dir():
            raise FileNotFoundError(args.reuse_matches_from)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if args.reuse_matches_from is not None:
        source_manifest = args.reuse_matches_from / "manifest.json"
        if not source_manifest.is_file():
            raise ValueError("reused matches require a source manifest with checkpoint identity")
        source_hash = json.loads(source_manifest.read_text()).get("checkpoint_sha256")
        if source_hash != checkpoint_sha256:
            raise ValueError("reused matches were generated by a different checkpoint")
        if args.train_profile == "real":
            source_catalog_hash = json.loads(source_manifest.read_text()).get(
                "config", {}
            ).get("real_deck_catalog_sha256")
            if source_catalog_hash != catalog_sha256():
                raise ValueError("reused matches use a different real-deck catalog")
    ROOT = args.out.expanduser().resolve()
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["confirmation_replicas"] = int(
        args.confirmation_replicas if args.confirmation_replicas is not None else args.replicas
    )
    config["reuse_matches_from"] = (
        str(args.reuse_matches_from) if args.reuse_matches_from is not None else None
    )
    config.update(checkpoint=str(args.checkpoint), out=str(ROOT))
    config["real_deck_catalog_sha256"] = (
        catalog_sha256() if args.train_profile == "real" else None
    )
    sources = [Path(__file__).resolve()]
    sources += [p for p in (REPO / "simulator").rglob("*")
                if p.is_file() and p.suffix in (".py", ".json", ".yaml", ".yml")
                and not {"tests", "dist", "__pycache__"}.intersection(p.parts)]
    sources += list((REPO / "src/cr_bot/domain").glob("*.py"))
    manifest = dict(config=config,
                    checkpoint_sha256=checkpoint_sha256,
                    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    filter=asdict(FilterConfig(max_per_match=args.candidates_per_match)),
                    classification=asdict(ClassifyConfig()))
    manifest_path = ROOT / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("existing output uses different settings, checkpoint, or sources; choose a new --out")
    else:
        if ROOT.exists() and any(ROOT.iterdir()):
            raise ValueError("output is non-empty without a matching manifest; choose a new --out")
        ROOT.mkdir(parents=True, exist_ok=True)
        save(manifest_path, manifest)
    train_archetypes = (
        REAL_DECK_ARCHETYPES if args.train_profile == "real"
        else DISCOVERY_ARCHETYPES if args.train_profile == "pinned"
        else MIXED_ARCHETYPES
    )
    sealed_archetypes = (
        REAL_DECK_ARCHETYPES if args.train_profile == "real" else DISCOVERY_ARCHETYPES
    )
    tasks=[(config, split,seed,arch,index)
           for split,seed,count,archetypes in [
               ("train",args.train_pool_seed,args.train_matches_per_archetype,train_archetypes),
               ("sealed",args.sealed_pool_seed,args.sealed_matches_per_archetype,sealed_archetypes),
           ]
           for arch in archetypes for index in range(count)]
    by_split={'train':[],'sealed':[]}
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(run_match,t) for t in tasks]
        for future in as_completed(futures):
            split,key,verdicts,n=future.result()
            by_split[split].extend(verdicts)
            print(f'{split} {key}: evaluated {len(verdicts)}, frontier {n}',flush=True)
    all_entries={}
    for split in by_split:
        verdicts=sorted(by_split[split],key=lambda v:(v['match_id'],v['t']))
        selected,report=build_distribution(
            verdicts,
            max_train=args.max_entries_per_split,
            per_archetype_cap=args.per_archetype_cap,
        )
        selected_keys={(v['match_id'],v['t']) for v in selected}
        entries=[]
        for _,s,seed,arch,index in tasks:
            if s==split:
                entries.extend(torch.load(ROOT/f'{split}_{arch}_{index}_entries.pt',weights_only=False))
        selected_ids={f'{split}:{match}:t{t}' for match,t in selected_keys}
        entries=[entry for entry in entries if entry['entry_id'] in selected_ids]
        headrooms={f'{split}:{v["match_id"]}:t{v["t"]}':v['headroom'] for v in selected}
        for entry in entries:
            entry['headroom']=headrooms[entry['entry_id']]
        entries.sort(key=lambda entry:entry['entry_id'])
        torch.save(entries,ROOT/f'frontier_{split}.pt')
        save(ROOT/f'{split}_distribution.json',report)
        save(ROOT/f'{split}_verdicts.json',verdicts)
        all_entries[split]=entries
    save(ROOT/'split_validation.json',validate_entry_split(all_entries['train'],all_entries['sealed']))
    counts = {k:len(v) for k,v in all_entries.items()}
    save(ROOT / "summary.json", dict(counts=counts, candidates={k:len(v) for k,v in by_split.items()}))
    print("COMPLETE", counts, flush=True)
    if counts["train"] < 100 or counts["sealed"] < 30:
        print("Small frontier set: inspect coverage before interpreting PPO results.", flush=True)

if __name__=='__main__':
    main()
