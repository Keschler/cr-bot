"""Paired, resumable local screens for the broad Hog policy strategy.

Uses the native complete-match evaluator. Training deck/style identities and
reserved identities remain distinct; no actor is promoted by this module.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[3]
DEVELOPMENT_VERSION = "hog-policy-development-1"


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path: Path | str) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def runtime_source_fingerprint() -> dict[str, Any]:
    """Explicit native runtime scope, excluding capture apps and source tests.

    Exact file bytes and additions/removals are checked. Git status and unrelated
    phone/frontend edits are not execution identity. The declared scope travels
    with every run; this is not the older broad-source fingerprint.
    """
    excluded = {"physical_lab", "tests", "build", "dist", "__pycache__", "cache"}
    paths = [path for path in (ROOT / "simulator").rglob("*.py")
             if not excluded.intersection(path.relative_to(ROOT / "simulator").parts)]
    paths += list((ROOT / "simulator/rulesets").glob("*.json"))
    paths += [ROOT / "simulator/rl/v4/real_decks.json"]
    for directory in ("src/cr_bot/domain", "src/cr_bot/features"):
        paths += list((ROOT / directory).glob("*.py"))
    paths += [ROOT / "src/cr_bot/__init__.py"]
    paths += [ROOT / "src/cr_bot/trackers/__init__.py",
              ROOT / "src/cr_bot/trackers/direct_unit_to_card.py"]
    paths += list((ROOT / "scripts").glob("run_v4_*.py"))
    files = {str(path.relative_to(ROOT)): file_sha(path) for path in sorted(set(paths))}
    return dict(schema="native-development-source-scope-1", files=files, sha256=digest(files),
                excluded_subtrees=sorted(excluded),
                note="Byte identity of native simulator, public inputs, rules, catalog and V4 runners; capture/frontend/tests excluded explicitly.")


@dataclass(frozen=True)
class DevelopmentArm:
    name: str
    kind: str
    checkpoint: str | None = None
    sha256: str | None = None
    model_config: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in self.name):
            raise ValueError("arm name must be a nonempty safe directory name")
        if self.kind not in ("defense", "cycle", "plans", "actor"):
            raise ValueError("arm kind must be defense, cycle, plans or actor")
        if self.kind == "actor":
            if not self.checkpoint or not self.sha256 or not self.model_config:
                raise ValueError("actor requires explicit checkpoint/hash/model configuration")
            if file_sha(self.checkpoint) != self.sha256:
                raise ValueError("actor checkpoint hash differs")
        elif any(value is not None for value in (self.checkpoint, self.sha256, self.model_config)):
            raise ValueError("teacher arms do not load actor checkpoints")


def training_opponents(styles: Sequence[str] = (), *, style_split: str = "train") -> list[dict[str, Any]]:
    from .catalog_evaluation import catalog_opponents
    base = catalog_opponents()["train"]
    if not styles:
        return [dict(spec) for spec in base]
    from simulator.rl.foundation.opponent_styles import TRAIN_STYLE_IDS, DEV_STYLE_IDS, style_identity
    allowed = {"train": TRAIN_STYLE_IDS, "dev": DEV_STYLE_IDS}
    if style_split not in allowed or len(set(styles)) != len(styles) or any(style not in allowed[style_split] for style in styles):
        raise ValueError("style pool is invalid or crosses the train/dev boundary")
    result = []
    for spec in base:
        for style in styles:
            identity = style_identity(spec["strategy"], style, seed=spec["controller_seed"])
            result.append(dict(spec, style=style, style_split=style_split, style_identity=identity))
    return result


def validate_opponents(opponents: Sequence[Mapping[str, Any]], *, catalog_split: str = "train") -> None:
    from .catalog_evaluation import catalog_opponents
    from simulator.rl.foundation.opponent_styles import style_identity
    if catalog_split not in ("train", "eval"):
        raise ValueError("catalog_split must be train or eval")
    pinned = {spec["deck_id"]: spec for spec in catalog_opponents()[catalog_split]}
    if not opponents:
        raise ValueError("screens require pinned catalog opponents")
    for spec in opponents:
        base = pinned.get(spec.get("deck_id"))
        if base is None or any(spec.get(key) != value for key, value in base.items()):
            raise ValueError("opponent differs from pinned catalog identity")
        if catalog_split == "eval":
            if dict(spec) != base:
                raise ValueError("reserved catalog opponents must retain exact identity without styles")
            continue
        allowed = set(base) | {"style", "style_config", "style_identity", "style_split"}
        if set(spec) - allowed:
            raise ValueError("unrecognized development opponent execution fields")
        if "style" in spec or "style_config" in spec:
            identity = style_identity(spec["strategy"], spec.get("style"),
                seed=spec["controller_seed"], config=spec.get("style_config"))
            if spec.get("style_identity") != identity or spec.get("style_split") != identity["split"]:
                raise ValueError("development opponent style/split identity differs")
        elif set(spec) != set(base):
            raise ValueError("style metadata requires explicit style configuration")


def _job_catalog_split(job: Mapping[str, Any]) -> str:
    split = job.get("catalog_split", "train")
    validate_opponents([job["opponent"]], catalog_split=split)
    if split == "eval" and job["arm"]["kind"] != "actor":
        raise ValueError("reserved catalog evaluation requires frozen actor arms")
    return split


def actual_family(record: Mapping[str, Any]) -> str:
    cards = set(record["opponent"]["deck"])
    for card, name in (("lava-hound", "lavaloon"), ("golem", "golem"), ("giant", "giant"),
                       ("x-bow", "xbow"), ("pekka", "pekka-bridge"), ("hog-rider", "hog"),
                       ("goblin-barrel", "bait")):
        if card in cards:
            return name
    raise ValueError("unclassified catalog deck")


class _TraceEnvironment:
    def __init__(self, env: Any):
        self.env = env
        self.decisions = 0
        self.plays = Counter()
        self.accepted_cards = Counter()
        self.rejected_actions = 0
        self.waits = 0
        self.trace = hashlib.sha256()
        self.on_action_executed = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def step_v2(self, actions: Any) -> Any:
        from simulator.actions import PlayCardAction, action_to_dict
        learner = actions[0]
        if isinstance(learner, PlayCardAction):
            self.plays[self.env.state.players[0].hand[learner.card_slot]] += 1
        else:
            self.waits += 1
        self.decisions += 1
        self.trace.update(canonical([action_to_dict(action) for action in actions]).encode() + b"\n")
        event_start = len(self.env.state.events)
        result = self.env.step_v2(actions)
        accepted_play = rejected = False
        for event in self.env.state.events[event_start:]:
            if event.get("player") == 0:
                if event.kind == "card_played":
                    self.accepted_cards[event.get("card_id")] += 1
                    accepted_play = True
                elif event.kind == "action_rejected":
                    self.rejected_actions += 1
                    rejected = True
        if self.on_action_executed is not None:
            self.on_action_executed(learner, accepted=not rejected and (
                not isinstance(learner, PlayCardAction) or accepted_play))
        return result


class _PublicTeacherPolicy:
    def __init__(self, kind: str, env_box: list[Any]):
        import torch
        from .model_v4 import ModelConfigV4
        from .simulator_teacher import defense_teacher_label, cycle_teacher_label
        self.torch, self.config, self.env_box = torch, ModelConfigV4(), env_box
        self.labeler = {"defense": defense_teacher_label, "cycle": cycle_teacher_label,
                        "plans": cycle_teacher_label}[kind]
        self.plan_teacher = None
        if kind == "plans":
            from .tactical_plans import PlanTeacherLabeler
            self.plan_teacher = PlanTeacherLabeler()
        self.stall_frames = self.current_stall = self.max_stall = self.cycle_rescues = 0

    def _acknowledge_plan_action(self, action, *, accepted):
        from simulator.actions import PlayCardAction, WaitAction
        from .simulator_teacher import TeacherTarget
        if isinstance(action, PlayCardAction):
            target = TeacherTarget(1, action.card_slot, action.cell[1], action.cell[0], 0)
        elif isinstance(action, WaitAction):
            target = TeacherTarget(0, 0, 0, 0, 0)
        else:
            target = None
        self.plan_teacher.acknowledge(target, accepted=accepted)

    def eval(self):
        return self

    def initial_hidden(self, batch_size, device=None):
        return self.torch.zeros(1, batch_size, 1, device=device)

    def forward_step(self, args, hidden=None, reset_mask=None):
        return args, None, hidden

    def act_deterministic(self, args, masks):
        from .model_v4 import V4ActionBatch
        from .simulator_teacher import defense_teacher_context, defense_teacher_label, _hand_card
        hand = args["hand_tokens"][0, 0].numpy()
        entities, entity_mask = args["entities"][0, 0].numpy(), args["entity_mask"][0, 0].numpy()
        # Own elixir is public. The adapter never reads enemy hand/elixir/locks.
        elixir = self.env_box[0].state.players[0].elixir_milli / 1000
        kwargs = dict(hand_tokens=hand, entity_tokens=entities, entity_mask=entity_mask,
                      legal_play=masks.placement[0, 0].numpy(),
                      legal_wait=bool(masks.mode[0, 0, 0]), own_elixir=elixir)
        if self.plan_teacher is None:
            target = self.labeler(**kwargs)
        else:
            trace = self.env_box[0]
            trace.on_action_executed = self._acknowledge_plan_action
            target = self.plan_teacher.label(
                decision_step=trace.decisions,
                decision_interval_us=trace.decision_interval_ticks * trace.engine.ruleset.tick_us,
                **kwargs,
            )
        context = defense_teacher_context(entities, entity_mask)
        ground_only = not any(_hand_card(card) in ("musketeer", "fireball", "ice-spirit") for card in hand)
        opportunity = context.air_threats > 0 and ground_only and elixir >= 4
        if opportunity and target.mode == 0:
            self.stall_frames += 1
            self.current_stall += 1
            self.max_stall = max(self.max_stall, self.current_stall)
        else:
            self.current_stall = 0
        if opportunity and target.mode == 1 and defense_teacher_label(**kwargs).mode == 0:
            self.cycle_rescues += 1
        return V4ActionBatch(mode=self.torch.tensor([[target.mode]]),
                            card_slot=self.torch.tensor([[target.card_slot]]),
                            placement=self.torch.tensor([[[target.row, target.col]]]),
                            wait_duration=self.torch.zeros(1, 1, dtype=self.torch.long))


@lru_cache(maxsize=4)
def _actor(path: str, sha256: str, config_json: str):
    import torch
    from .model_v4 import ModelConfigV4
    from .v4_ppo.runtime import load_frozen_anchor, configure_v4_runtime
    with torch.random.fork_rng(devices=[]):
        policy = load_frozen_anchor(path, ModelConfigV4(**json.loads(config_json)),
                                    expected_sha256=sha256, device="cpu")
    configure_v4_runtime(policy)
    return policy


def _worker_init() -> None:
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    import torch
    torch.set_num_threads(1)


def execute_match(job: Mapping[str, Any]) -> dict[str, Any]:
    from .frontier_ppo import evaluation as native
    _job_catalog_split(job)
    arm = DevelopmentArm(**job["arm"])
    box = []
    original_factory = native.make_frontier_env
    def factory():
        env = _TraceEnvironment(original_factory())
        box.append(env)
        return env
    native.make_frontier_env = factory
    try:
        policy = (_actor(arm.checkpoint, arm.sha256, canonical(arm.model_config)) if arm.kind == "actor"
                  else _PublicTeacherPolicy(arm.kind, box))
        records = native.full_match_record(policy=policy, opponent_specs=[job["opponent"]],
            seeds=[job["seed"]], device="cpu", deterministic=arm.kind != "actor", max_decisions=1440)
    finally:
        native.make_frontier_env = original_factory
    if len(records) != 1 or len(box) != 1:
        raise RuntimeError("native evaluator returned an unexpected match grid")
    record, trace = records[0], box[0]
    diagnostics = dict(decisions=trace.decisions, attempted_cards=dict(trace.plays), waits=trace.waits,
                       accepted_cards=dict(trace.accepted_cards), rejected_actions=trace.rejected_actions,
                       action_trace_sha256=trace.trace.hexdigest())
    if arm.kind != "actor":
        diagnostics.update(ground_only_air_wait_seconds=policy.stall_frames * .25,
                           longest_ground_only_air_wait_seconds=policy.max_stall * .25,
                           cycle_rescue_plays=policy.cycle_rescues)
    if arm.kind == "plans":
        from .tactical_plans import TACTICAL_PLANS_VERSION
        diagnostics.update(tactical_plans_version=TACTICAL_PLANS_VERSION,
                           tactical_plan_counters=asdict(policy.plan_teacher.counters))
    row = dict(job=job, native_record=record, diagnostics=diagnostics)
    validate_match(row, job)
    return row


def validate_match(row: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    from simulator.roster import PLAYER_DECK
    from simulator.engine import ENGINE_VERSION
    from simulator.observation_v3 import OBSERVATION_V3_CONTRACT_HASH
    from simulator.ruleset import load_fixed_ruleset
    from .frontier import _stable_seed
    from .frontier_ppo.constants import FRONTIER_PPO_VERSION

    catalog_split = _job_catalog_split(job)
    record, diag = row["native_record"], row["diagnostics"]
    if row["job"] != job or record["seed"] != job["seed"] or record["deck_id"] != job["opponent"]["deck_id"]:
        raise ValueError("saved match job/native identity differs")
    if (not record["terminated"] or record["truncated"] or record["cap_reached"]
            or record["result"] not in ("win", "loss", "draw") or record["invalid_actions"][0]):
        raise ValueError("incomplete or illegal learner match")
    expected_result = {0: "win", 1: "loss", None: "draw"}.get(record["winner"])
    if expected_result != record["result"] or type(record["winner"]) not in (int, type(None)):
        raise ValueError("native winner and result differ")
    if (record["decision_interval_us"] != 250000 or record["max_decisions"] != 1440
            or type(record["decisions"]) is not int or not 0 < record["decisions"] <= 1440):
        raise ValueError("native decision cadence or cap differs")
    if record["learner_player"] != 0 or record["learner_deck"] != list(PLAYER_DECK):
        raise ValueError("native learner seat or deck differs")
    if (record["engine_version"] != ENGINE_VERSION or record["runtime_version"] != FRONTIER_PPO_VERSION
            or record["observation_contract_hash"] != OBSERVATION_V3_CONTRACT_HASH
            or record["ruleset_hash"] != load_fixed_ruleset().content_hash):
        raise ValueError("native execution contract differs")
    for field in ("initial_state_hash", "final_state_hash", "event_log_hash", "action_trace_sha256"):
        value = record[field]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("native state/event/trace digest is invalid")
    if (record["decisions"] != diag["decisions"]
            or record["action_trace_sha256"] != diag["action_trace_sha256"]
            or diag["attempted_cards"] != diag["accepted_cards"] or diag["rejected_actions"] != 0
            or record["card_plays"][0] != sum(diag["attempted_cards"].values())
            or diag["waits"] + sum(diag["attempted_cards"].values()) != diag["decisions"]):
        raise ValueError("native/diagnostic trace, decisions or accepted-card parity differs")
    if record["deterministic_policy"] != (job["arm"]["kind"] != "actor"):
        raise ValueError("actor/teacher inference mode differs")
    if job["arm"]["kind"] == "plans":
        from .tactical_plans import PlanCounters, TACTICAL_PLANS_VERSION
        counts = diag.get("tactical_plan_counters", {})
        if (diag.get("tactical_plans_version") != TACTICAL_PLANS_VERSION
                or set(counts) != set(PlanCounters.__dataclass_fields__)
                or any(type(value) is not int or value < 0 for value in counts.values())
                or counts["accepted_plays"] > record["card_plays"][0]
                or counts["waits"] > diag["waits"]
                or counts["rejected_plays"] > diag["rejected_actions"]):
            raise ValueError("tactical-plan execution diagnostics differ")
    spec = job["opponent"]
    expected_opponent = dict(kind=spec.get("kind", "scripted"), strategy=spec.get("strategy"),
        deck_id=spec["deck_id"], deck=spec["deck"], controller_seed=spec["controller_seed"])
    if "style_identity" in spec:
        expected_opponent["style"] = spec["style_identity"]
    if (record["opponent"] != expected_opponent or record["controller_seed"] != spec["controller_seed"]
            or record["strategy"] != spec["strategy"] or record["archetype"] != spec["archetype"]
            or record["real_deck_split"] != catalog_split):
        raise ValueError("native opponent/controller/catalog identity differs")
    expected_policy_seed = _stable_seed("full-match", json.dumps(expected_opponent, sort_keys=True), job["seed"]) & 0xFFFFFFFF
    if record["policy_seed"] != expected_policy_seed:
        raise ValueError("native learner sampling seed differs")


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = dict(games=len(rows), wins=0, losses=0, draws=0, families={},
                  ground_only_air_wait_seconds=0., cycle_rescue_plays=0)
    for row in rows:
        native = row["native_record"]
        field = {"win": "wins", "loss": "losses", "draw": "draws"}[native["result"]]
        result[field] += 1
        family = actual_family(native)
        bucket = result["families"].setdefault(family, dict(games=0, wins=0, losses=0, draws=0))
        bucket["games"] += 1
        bucket[field] += 1
        result["ground_only_air_wait_seconds"] += row["diagnostics"].get("ground_only_air_wait_seconds", 0.)
        result["cycle_rescue_plays"] += row["diagnostics"].get("cycle_rescue_plays", 0)
        if "tactical_plan_counters" in row["diagnostics"]:
            counts = result.setdefault("tactical_plan_counters", {})
            for key, value in row["diagnostics"]["tactical_plan_counters"].items():
                counts[key] = counts.get(key, 0) + value
    return result


def paired_summary(base: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    a, b = {row["job"]["cell_id"]: row for row in base}, {row["job"]["cell_id"]: row for row in candidate}
    if len(a) != len(base) or len(b) != len(candidate):
        raise ValueError("paired match grid contains duplicate cell identities")
    if set(a) != set(b):
        raise ValueError("paired match grids differ")
    fields = ("initial_state_hash", "opponent", "controller_seed", "policy_seed", "seed", "deck_id",
              "learner_deck", "learner_player", "decision_interval_us", "engine_version", "ruleset_hash",
              "observation_contract_hash", "runtime_version", "real_deck_split")
    net = favorable = unfavorable = 0
    families = Counter()
    for key, left in a.items():
        ln, rn = left["native_record"], b[key]["native_record"]
        if any(ln[field] != rn[field] for field in fields):
            raise ValueError("paired initial state/opponent/seed/native contract differs")
        delta = {"win": 1., "draw": .5, "loss": 0.}[rn["result"]] - {"win": 1., "draw": .5, "loss": 0.}[ln["result"]]
        net += delta
        favorable += delta > 0
        unfavorable += delta < 0
        families[actual_family(ln)] += delta
    return dict(pairs=len(a), net_wins=net, favorable=favorable, unfavorable=unfavorable,
                family_net=dict(families), mean_score_delta=net / len(a) if a else None)


def cycle_gate(comparison: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    regressions = {name: delta for name, delta in comparison["family_net"].items()
                   if delta <= gate["reject_any_family_delta_at_or_below"]}
    eligible = (comparison["net_wins"] >= gate["minimum_net_wins"]
                and comparison["family_net"].get("lavaloon", 0) >= gate["minimum_lavaloon_net_wins"]
                and not regressions)
    return dict(eligible=eligible, regressions=regressions, gate=dict(gate))


def run_development_screen(*, out: Path, arms: Sequence[DevelopmentArm], opponents: Sequence[Mapping[str, Any]],
                           seeds: Sequence[int], workers: int = 3, gate: Mapping[str, Any] | None = None,
                           catalog_split: str = "train") -> dict[str, Any]:
    if not arms or len({arm.name for arm in arms}) != len(arms):
        raise ValueError("unique nonempty arm list required")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("unique non-negative match seeds required")
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("workers must be in 1..16")
    validate_opponents(opponents, catalog_split=catalog_split)
    if catalog_split == "eval" and (any(arm.kind != "actor" for arm in arms) or gate is not None):
        raise ValueError("reserved catalog evaluation requires frozen actor arms and no selection gate")
    source = runtime_source_fingerprint()
    identity = dict(version=DEVELOPMENT_VERSION, arms=[asdict(arm) for arm in arms], opponents=list(opponents),
                    seeds=list(seeds), workers=workers, source=source, gate=gate, decision_interval_us=250000,
                    max_decisions=1440, warning_policy="retain_all_with_symmetric_exclusion",
                    catalog_split=catalog_split)
    run_hash = digest(identity)
    out = out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != dict(identity=identity, run_hash=run_hash):
        raise ValueError("resume protocol/source/actor differs; choose a new output directory")
    atomic_json(manifest, dict(identity=identity, run_hash=run_hash))
    jobs, cells = [], set()
    for spec in opponents:
        for seed in seeds:
            cell = f"{spec['deck_id']}__{spec.get('style_identity', {}).get('style_id', 'fixed')}__{seed}"
            if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in cell):
                raise ValueError("unsafe opponent/seed cell identity")
            if cell in cells:
                raise ValueError("opponent/seed cell repeats")
            cells.add(cell)
            for arm in arms:
                job = dict(cell_id=cell, arm=asdict(arm), opponent=dict(spec), seed=seed)
                if catalog_split == "eval":
                    job["catalog_split"] = "eval"
                jobs.append(job)
    expected_paths = {f"{job['arm']['name']}/{job['cell_id']}.json" for job in jobs}
    actual_paths = {str(path.relative_to(out)) for path in out.rglob("*.json")
                    if path.parent.name in {arm.name for arm in arms}}
    if actual_paths - expected_paths:
        raise ValueError("unexpected saved raw match outside the declared grid")
    if (out / "summary.json").exists():
        digest_path = out / "raw_digests.json"
        if not digest_path.exists():
            raise ValueError("completed resume is missing its raw digest seal")
        saved_digests = json.loads(digest_path.read_text())
        actual_digests = {name: file_sha(out / name) for name in sorted(actual_paths)}
        saved_summary = json.loads((out / "summary.json").read_text())
        if (actual_paths != expected_paths or saved_digests != actual_digests
                or saved_summary.get("raw_digest_sha256") != digest(actual_digests)
                or saved_summary.get("run_hash") != run_hash):
            raise ValueError("completed resume raw digest seal differs")
    completed, pending = {}, []
    for job in jobs:
        path = out / job["arm"]["name"] / (job["cell_id"] + ".json")
        if path.exists():
            row = json.loads(path.read_text())
            if row.get("run_hash") != run_hash:
                raise ValueError("saved match run identity differs")
            validate_match(row, job)
            completed[(job["arm"]["name"], job["cell_id"])] = row
        else:
            pending.append(job)
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        futures = {pool.submit(execute_match, job): job for job in pending}
        for future in as_completed(futures):
            job = futures[future]
            row = dict(future.result(), run_hash=run_hash)
            validate_match(row, job)
            atomic_json(out / job["arm"]["name"] / (job["cell_id"] + ".json"), row)
            completed[(job["arm"]["name"], job["cell_id"])] = row
            if len(completed) % 10 == 0:
                print(f"native matches {len(completed)}/{len(jobs)}", flush=True)
    all_rows = {arm.name: [completed[(arm.name, cell)] for cell in sorted(cells)] for arm in arms}
    base = arms[0].name
    comparisons = {name: paired_summary(all_rows[base], rows) for name, rows in all_rows.items() if name != base}
    warnings = {name: [row["job"]["cell_id"] for row in rows if row["native_record"]["invalid_actions"][1]]
                for name, rows in all_rows.items()}
    excluded = set().union(*warnings.values())
    sensitivity = {name: paired_summary([row for row in all_rows[base] if row["job"]["cell_id"] not in excluded],
                                       [row for row in rows if row["job"]["cell_id"] not in excluded])
                   for name, rows in all_rows.items() if name != base} if excluded else {}
    if runtime_source_fingerprint() != source:
        raise RuntimeError("native execution source changed; raw games preserved, summary not accepted")
    for arm in arms:
        if arm.kind == "actor" and file_sha(arm.checkpoint) != arm.sha256:
            raise RuntimeError("actor changed during evaluation")
    raw_hashes = {str(path.relative_to(out)): file_sha(path) for path in sorted(out.rglob("*.json"))
                  if path.parent.name in all_rows}
    report = dict(status="complete_with_opponent_warning" if excluded else "complete", run_hash=run_hash,
        arms={name: summarize(rows) for name, rows in all_rows.items()}, comparisons=comparisons,
        eligibility={name: cycle_gate(value, gate) for name, value in comparisons.items()} if gate else {},
        opponent_warnings=warnings, paired_exclusion=dict(cells=sorted(excluded), comparisons=sensitivity),
        raw_records=len(completed), raw_digest_sha256=digest(raw_hashes),
        wall_seconds_this_invocation=time.monotonic() - started, selected_actor_changed=False,
        limitation=("Train-deck development evidence against declared scripted styles; not human strength. Correlated stall frames are durations, not independent mistakes."
                    if catalog_split == "train" else
                    "Reserved-catalog final actor comparison against fixed scripted opponents; not human strength. No actor selection gate is applied."))
    atomic_json(out / "summary.json", report)
    atomic_json(out / "raw_digests.json", raw_hashes)
    return report
