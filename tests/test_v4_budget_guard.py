import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_v4_budget_guard.py"
spec = importlib.util.spec_from_file_location("v4_budget_guard", SCRIPT)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def test_cost_budget_requires_actual_billing_stop_hook(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), "--state", str(tmp_path / "state.json"),
                             "--budget-usd", "4.75", "--hourly-rate", "1.13", "--", sys.executable, "-c", "pass"],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "stop provider billing" in result.stderr
    assert not (tmp_path / "state.json").exists()


def test_deadline_survives_resume_and_runs_cleanup(tmp_path):
    state = tmp_path / "state.json"
    marker = tmp_path / "started"
    stopped = tmp_path / "stopped"
    command = [sys.executable, "-c", f"from pathlib import Path; import time; Path({str(marker)!r}).write_text('yes'); time.sleep(30)"]
    stop = [sys.executable, "-c", f"from pathlib import Path; Path({str(stopped)!r}).write_text('yes')"]
    args = [sys.executable, str(SCRIPT), "--state", str(state), "--max-runtime-seconds", "1",
            "--reserve-seconds", "0.5", "--stop-command-json", json.dumps(stop), "--", *command]
    result = subprocess.run(args, timeout=10)
    assert result.returncode == 124
    first = json.loads(state.read_text())
    assert first["status"] == "budget_exhausted"
    assert marker.exists() and stopped.exists()
    mtime = marker.stat().st_mtime_ns
    assert subprocess.run(args, timeout=10).returncode == 124
    assert json.loads(state.read_text())["deadline_epoch"] == first["deadline_epoch"]
    assert marker.stat().st_mtime_ns == mtime  # expired resumes never start compute


def test_success_is_not_rerun_and_exports_deadline(tmp_path):
    state = tmp_path / "state.json"
    marker = tmp_path / "deadline"
    command = [sys.executable, "-c", f"import os; from pathlib import Path; Path({str(marker)!r}).write_text(os.environ['V4_DEADLINE_EPOCH'])"]
    args = [sys.executable, str(SCRIPT), "--state", str(state), "--max-runtime-seconds", "10",
            "--reserve-seconds", "1", "--", *command]
    assert subprocess.run(args, timeout=5).returncode == 0
    assert float(marker.read_text()) > time.time()
    mtime = marker.stat().st_mtime_ns
    assert subprocess.run(args, timeout=5).returncode == 0
    assert marker.stat().st_mtime_ns == mtime


def test_reserve_is_subtracted_from_money_budget():
    assert guard.budget_seconds(4.75, 1.13, None, 300) == 4.75 / 1.13 * 3600


def test_archive_failure_still_runs_provider_stop(tmp_path):
    state = tmp_path / "state.json"
    stopped = tmp_path / "stopped"
    stop = [sys.executable, "-c", f"from pathlib import Path; Path({str(stopped)!r}).touch()"]
    args = [sys.executable, str(SCRIPT), "--state", str(state), "--max-runtime-seconds", "10",
            "--reserve-seconds", "2", "--archive-command-json", json.dumps([sys.executable, "-c", "raise SystemExit(7)"]),
            "--stop-command-json", json.dumps(stop), "--", sys.executable, "-c", "pass"]
    assert subprocess.run(args, timeout=5).returncode == 1
    result = json.loads(state.read_text())
    assert result["status"] == "cleanup_failed"
    assert result["archive_returncode"] == 7
    assert result["stop_returncode"] == 0
    assert stopped.exists()
