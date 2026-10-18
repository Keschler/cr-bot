"""Bounded discovery of local V4 artifacts, without deserializing model files."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any

from cr_bot.paths import REPO_ROOT

REPORT_NAMES = ("progress.json", "learning_curve.json", "training_history.json", "manifest.json", "report.json", "summary.json")
PRUNED_DIRS = {".git", "__pycache__", "evaluation_cache", "prefix_cache", "cache", "worktrees", "node_modules"}


def artifact_id(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:24]


@dataclass(frozen=True)
class ViewerPaths:
    roots: tuple[Path, ...]
    storage: Path

    @classmethod
    def from_environment(cls) -> ViewerPaths:
        configured = os.environ.get("CR_SIM_VIEWER_ROOTS", "")
        roots = [Path(value).expanduser().resolve() for value in configured.split(os.pathsep) if value]
        if not roots:
            roots = [(REPO_ROOT / "outputs/v4").resolve()]
        recorder = os.environ.get("CR_SIM_VIEWER_DIR")
        if recorder:
            roots.append(Path(recorder).expanduser().resolve())
        storage = (REPO_ROOT / "outputs/v4/sim_viewer/matches").resolve()
        roots.append(storage)
        return cls(tuple(dict.fromkeys(roots)), storage)

    def resolve(self, value: str | Path, *, suffixes: tuple[str, ...]) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        path = path.resolve()
        if not any(path.is_relative_to(root.resolve()) for root in self.roots):
            raise ValueError("Artifact path must be inside a configured V4 data directory")
        if path.suffix.lower() not in suffixes:
            raise ValueError("Unsupported artifact file extension")
        if not path.is_file():
            raise FileNotFoundError(path.name)
        return path


def read_json(path: Path, *, max_bytes: int = 32_000_000) -> Any:
    if path.stat().st_size > max_bytes:
        raise ValueError(f"Report exceeds {max_bytes // 1_000_000} MB: {path.name}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _recording_info(path: Path) -> tuple[str, bool] | None:
    try:
        with path.open("rb") as handle:
            first = handle.readline(262_144)
            if not first.endswith(b"\n"):
                return None
            header = json.loads(first)
            if header.get("type") != "header" or header.get("schema_version") != 1:
                return None
            handle.seek(max(0, path.stat().st_size - 16_384))
            tail = handle.read().splitlines()
            # A writer has not published its last row until the newline lands.
            handle.seek(max(0, path.stat().st_size - 1))
            if handle.read() != b"\n":
                tail = tail[:-1]
        ended = False
        for line in tail[1:]:
            if b'"end"' not in line:
                continue
            try:
                ended = ended or json.loads(line).get("type") == "end"
            except ValueError:
                continue
        source = str(header.get("source", "training"))
        return "frontier" if source == "replay" else source, not ended
    except (OSError, ValueError, AttributeError):
        return None


def _is_actor(path: Path) -> bool:
    name = path.stem.lower()
    if name in {"train", "dev", "entries", "validation", "test", "dataset"}:
        return False
    if any(word in name for word in ("critic", "optimizer", "frontier_", "prefix_cache", "training_state", "training-bank", "train_bank", "sealed", "starts")):
        return False
    # Distillation checkpoints contain optimizers; matching plain actors live
    # in the sibling actors/ directory. PPO checkpoint_update files are actors.
    if path.parent.name == "checkpoints" and (path.parent.parent / "actors").is_dir():
        return False
    return True


class ArtifactLibrary:
    def __init__(self, paths: ViewerPaths) -> None:
        self.paths = paths
        self._lock = threading.Lock()
        self._expires = 0.0
        self._matches: list[dict[str, Any]] = []
        self._runs: list[dict[str, Any]] = []
        self._checkpoints: list[dict[str, Any]] = []

    def refresh(self, *, force: bool = False) -> None:
        with self._lock:
            if not force and time.monotonic() < self._expires:
                return
            matches, checkpoints = [], []
            reports: dict[Path, dict[str, Path]] = {}
            separate_reports: dict[Path, Path] = {}
            seen: set[Path] = set()
            scanned = 0
            for root in self.paths.roots:
                if not root.is_dir():
                    continue
                for base, dirs, filenames in os.walk(root, followlinks=False):
                    dirs[:] = [name for name in dirs if name not in PRUNED_DIRS and not (Path(base) / name).is_symlink()]
                    for name in filenames:
                        scanned += 1
                        if scanned > 100_000:
                            break
                        path = Path(base) / name
                        if path.suffix not in {".pt", ".json", ".jsonl"} or path.is_symlink():
                            continue
                        resolved = path.resolve()
                        if resolved in seen:
                            continue
                        seen.add(resolved)
                        try:
                            stat = path.stat()
                        except OSError:
                            continue
                        item = {"id": artifact_id(path), "path": str(resolved), "name": str(path.relative_to(root)), "modified_at": stat.st_mtime, "size": stat.st_size}
                        if path.suffix == ".pt" and _is_actor(path):
                            step = re.search(r"(update_|epoch[_-]?)(\d+)", path.stem)
                            if step:
                                item["step"] = int(step.group(2))
                                item["step_kind"] = "update" if step.group(1) == "update_" else "epoch"
                            checkpoints.append(item)
                        elif name.endswith("_match.json"):
                            matches.append({**item, "source": "frontier", "active": False})
                        elif path.suffix == ".jsonl":
                            info = _recording_info(path)
                            if info is not None:
                                source, unfinished = info
                                matches.append({**item, "source": source, "active": unfinished and time.time() - stat.st_mtime < 30, "unfinished": unfinished})
                        if name in REPORT_NAMES:
                            report_directory = resolved.parent
                            if name == "progress.json":
                                try:
                                    data = read_json(path)
                                    if not isinstance(data, dict) or data.get("source") not in {"training", "distillation"}:
                                        continue
                                    if isinstance(data, dict) and data.get("output_dir"):
                                        candidate = Path(data["output_dir"]).expanduser()
                                        candidate = (candidate if candidate.is_absolute() else REPO_ROOT / candidate).resolve()
                                        if any(candidate.is_relative_to(allowed) for allowed in self.paths.roots):
                                            report_directory = candidate
                                    if data.get("source") == "distillation" and isinstance(data.get("label"), str) and Path(data["label"]).name == data["label"]:
                                        identity = report_directory / f"{data['label']}.json"
                                        previous = separate_reports.get(identity)
                                        if previous is None or previous.stat().st_mtime < resolved.stat().st_mtime:
                                            separate_reports[identity] = resolved
                                        continue
                                except (OSError, ValueError, TypeError):
                                    continue
                            if name in {"manifest.json", "summary.json"}:
                                try:
                                    data = read_json(path, max_bytes=8_000_000)
                                except (OSError, ValueError):
                                    continue
                                if not isinstance(data, dict) or not any(key in data for key in ("history", "points", "reports")):
                                    continue
                            reports.setdefault(report_directory, {})[name] = resolved
                        elif name.startswith("context") and name.endswith(".json"):
                            # Named search-distillation arms share one output
                            # directory; keep each arm independently comparable.
                            try:
                                data = read_json(path, max_bytes=8_000_000)
                            except (OSError, ValueError):
                                continue
                            if isinstance(data, dict) and isinstance(data.get("history"), list):
                                previous = separate_reports.get(resolved)
                                if previous is None or previous.stat().st_mtime < resolved.stat().st_mtime:
                                    separate_reports[resolved] = resolved
                    if scanned > 100_000:
                        dirs.clear()
                        break
            runs = []
            for directory, files in reports.items():
                report = next(files[name] for name in REPORT_NAMES if name in files)
                name = next((str(directory.relative_to(root)) for root in self.paths.roots if directory.is_relative_to(root) and directory != root), directory.name)
                runs.append({"id": artifact_id(directory), "path": str(directory), "name": name, "report_path": str(report), "modified_at": max(p.stat().st_mtime for p in files.values())})
            for identity, report in separate_reports.items():
                runs.append({"id": artifact_id(identity), "path": str(identity.parent), "name": f"{identity.parent.name}/{identity.stem}", "report_path": str(report), "modified_at": report.stat().st_mtime})
            self._matches = sorted(matches, key=lambda row: row["modified_at"], reverse=True)
            self._runs = sorted(runs, key=lambda row: row["modified_at"], reverse=True)
            self._checkpoints = sorted(checkpoints, key=lambda row: row["modified_at"], reverse=True)
            self._expires = time.monotonic() + 3.0

    def listing(self, query: str = "", limit: int = 100) -> dict[str, Any]:
        self.refresh()
        query = query.casefold()
        with self._lock:
            matches = [dict(row) for row in self._matches if query in row["name"].casefold() or query in row["path"].casefold()]
            runs = [dict(row) for row in self._runs if query in row["name"].casefold() or query in row["path"].casefold()]
        return {"matches": matches[:limit], "runs": runs[:limit], "match_count": len(matches), "run_count": len(runs)}

    def checkpoints(self) -> list[dict[str, Any]]:
        self.refresh()
        with self._lock:
            return [dict(row) for row in self._checkpoints]

    def run(self, run_id: str) -> dict[str, Any]:
        self.refresh()
        with self._lock:
            row = next((dict(row) for row in self._runs if row["id"] == run_id), None)
        if row is None:
            raise KeyError(run_id)
        path = self.paths.resolve(row["report_path"], suffixes=(".json",))
        raw = read_json(path)
        series = normalise_series(raw)
        directory = Path(row["path"])
        checkpoints = [
            item for item in self.checkpoints()
            if Path(item["path"]).parent == directory
            or (Path(item["path"]).parent.name in {"actors", "checkpoints"} and Path(item["path"]).parent.parent == directory)
        ]
        arm_label = raw.get("label") if isinstance(raw, dict) else None
        if arm_label is None and path.stem.startswith("context"):
            arm_label = path.stem
        if arm_label:
            checkpoints = [item for item in checkpoints if Path(item["path"]).stem == arm_label or Path(item["path"]).stem.startswith(arm_label + "_")]
        by_step = {point["step"]: point for point in series}
        for checkpoint in checkpoints:
            step = checkpoint.get("step")
            # Distillation history is zero-indexed; epoch-N actor exports are
            # written after N completed epochs. Epoch zero is the untouched
            # anchor, and has no training-loss measurement.
            if checkpoint.get("step_kind") == "epoch" and isinstance(raw, dict) and raw.get("method"):
                step = step - 1 if step is not None else None
            if step in by_step:
                checkpoint["metrics"] = {key: value for key, value in by_step[step].items() if key != "step"}
        summary: dict[str, Any] = {"points": len(series), "report": path.name, "reward_label": "Mean return"}
        if series:
            summary.update({key: value for key, value in series[-1].items() if key != "step"})
            summary["step"] = series[-1]["step"]
        if isinstance(raw, dict):
            summary["state"] = raw.get("state", raw.get("status", "recorded"))
            summary["config"] = raw.get("config", {})
        return {**row, "series": series, "checkpoints": checkpoints, "summary": summary, "raw": raw}


def _number(value: Any) -> float | None:
    if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def normalise_series(raw: Any) -> list[dict[str, Any]]:
    """Expose recorded measurements; missing/censored values remain absent."""
    rows: dict[float, dict[str, Any]] = {}

    def row_at(step: Any) -> dict[str, Any]:
        step = _number(step)
        if step is None:
            raise ValueError("Training point has no numeric step")
        return rows.setdefault(step, {"step": step})

    def metrics(target: dict[str, Any], source: Any) -> None:
        if not isinstance(source, dict):
            return
        aliases = {"policy_loss": ("policy_loss",), "value_loss": ("value_loss", "critic_loss"), "entropy": ("entropy",), "kl": ("approx_kl", "kl"), "reward": ("returns_mean", "mean_return", "mean_reward"), "loss": ("loss",), "accuracy": ("exact_accuracy", "accuracy"), "target_probability": ("mean_target_probability",)}
        for key, names in aliases.items():
            for name in names:
                value = _number(source.get(name))
                if value is not None:
                    target[key] = value
                    break

    def update_point(step: Any, update: Any) -> None:
        if not isinstance(update, dict):
            return
        target = row_at(step)
        metrics(target, update.get("update", update))
        rollout = update.get("rollout", {})
        if isinstance(rollout, dict):
            metrics(target, rollout)
            for key in ("wait_fraction", "mean_episode_len"):
                value = _number(rollout.get(key))
                if value is not None:
                    target[key] = value

    history = raw if isinstance(raw, list) else raw.get("history", raw.get("series", [])) if isinstance(raw, dict) else []
    if isinstance(history, list):
        for index, item in enumerate(history):
            if isinstance(item, dict):
                target = row_at(item.get("step", item.get("epoch", index)))
                metrics(target, item)
                for key in ("win_rate", "score", "truncated", "completed"):
                    value = _number(item.get(key))
                    if value is not None:
                        target[key] = value
    if isinstance(raw, dict):
        updates = raw.get("updates", raw.get("update_reports", {}))
        if isinstance(updates, dict):
            for step, item in updates.items():
                try:
                    update_point(float(step), item)
                except (ValueError, TypeError):
                    continue
        for point in raw.get("points", []):
            if not isinstance(point, dict):
                continue
            target = row_at(point.get("update", point.get("step")))
            metrics(target, point)
            for step, item in point.get("updates_since_previous_point", {}).items():
                update_point(float(step), item)
            candidate = point.get("evaluation", {}).get("full_match", {}).get("candidate", {})
            if isinstance(candidate, dict):
                completed = _number(candidate.get("completed"))
                if completed is not None:
                    target["completed"] = completed
                    if completed > 0:
                        wins = _number(candidate.get("wins"))
                        if wins is not None:
                            target["win_rate"] = wins / completed
                        score = _number(candidate.get("score_mean_completed"))
                        if score is not None:
                            target["score"] = score
                truncated = _number(candidate.get("truncated"))
                if truncated is not None:
                    target["truncated"] = truncated
    return [rows[key] for key in sorted(rows)]
