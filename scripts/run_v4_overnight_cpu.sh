#!/usr/bin/env bash
set -euo pipefail

repo="${V4_REPO:-/workspace/cr-bot}"
python="${V4_PYTHON:-$repo/.venv/bin/python}"
out="${V4_RUN_DIR:-$repo/outputs/v4/runpod_corrected_20260922}"
main_pool="${V4_AUDIT_SOURCE:-$repo/outputs/v4/runpod_frontier_dataset_20260921}"
small_val="${V4_VALIDATION_POOL:-$repo/outputs/v4/runpod_validation_pool_20260921}"
full_pool="${V4_FULL_POOL:-$repo/outputs/v4/runpod_full_pool_corrected_20260922}"
large_val="${V4_CONFIRMED_VALIDATION_POOL:?Supply an independently confirmed validation pool}"
checkpoint="${V4_CHECKPOINT:-$repo/outputs/v4/stage2_12k_v2_e12_s12.pt}"

# Run through run_v4_budget_guard.py. Its persistent deadline prevents restarts
# from granting a new budget and covers collection worker subprocesses too.
: "${V4_DEADLINE_EPOCH:?Start this chain with scripts/run_v4_budget_guard.py}"
"$python" -c 'import os,time; assert float(os.environ["V4_DEADLINE_EPOCH"]) > time.time(), "budget exhausted"'
# The historical validation pool has discovery-only labels. Reusing those as
# independently confirmed labels would invalidate the corrected comparison.
[[ -f "$large_val/frontier_train.pt" ]] || { echo "Missing confirmed validation entries: $large_val" >&2; exit 1; }

mkdir -p "$out/logs"
exec > >(tee -a "$out/chain.log") 2>&1
export PYTHONPATH="$repo:$repo/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

status() {
  "$python" -c 'import json,sys,time,pathlib; p=pathlib.Path(sys.argv[1]); t=p.with_suffix(".tmp"); t.write_text(json.dumps({"status":sys.argv[2],"time":time.time(),"detail":sys.argv[3]},indent=2)+"\n"); t.replace(p)' "$out/status.json" "$1" "$2"
}
trap 'rc=$?; status failed "line=$LINENO exit=$rc"; exit $rc' ERR
trap 'status budget_interrupted "guard deadline or external signal; saved artifacts retained"; exit 124' TERM INT

status running "waiting for disjoint validation pool"
while [[ ! -f "$small_val/summary.json" ]]; do
  if ! pgrep -f 'runpod_validation_pool_20260921' >/dev/null; then
    status failed "validation pool exited without summary.json"
    exit 1
  fi
  sleep 30
done

status running "teacher reliability audit"
"$python" -u "$repo/scripts/run_v4_teacher_audit.py" \
  --source "$main_pool:train" \
  --source "$small_val:train" \
  --checkpoint "$checkpoint" \
  --out "$out/teacher_audit" \
  --workers 26 | tee "$out/logs/teacher_audit.log"

"$python" - "$out/teacher_audit/summary.json" <<'PY'
import json, sys
report=json.load(open(sys.argv[1]))
if report.get("schema") != "teacher-audit-v2":
    raise SystemExit("teacher gate requires corrected v2 audit; old confirmation may name wrong action")
s=report["summaries"]["phase1_16rep_h128"]["frozen_confirmation"]
n=s["n_available"]
c=s["classifications"]
if n < 250 or (s["mean_delta"] or 0) <= 0 or c.get("negative",0) > n * 0.25:
    raise SystemExit(f"teacher gate failed: {s}")
gate=report["summaries"]["horizon_stability"].get("fixed_action_gate", {})
if not gate.get("pass") or set(gate.get("required_horizons", [])) != {28,56} or gate.get("reference") != 112:
    raise SystemExit(f"independent fixed-action horizon gate failed: {gate}")
print("TEACHER_GATE_PASS", s)
PY

status running "collecting full train and untouched sealed pools"
reuse_args=()
if [[ -n "${V4_REUSE_MATCHES_FROM:-}" ]]; then
  reuse_args=(--reuse-matches-from "$V4_REUSE_MATCHES_FROM")
fi
"$python" -u "$repo/scripts/run_v4_frontier_generation.py" \
  --checkpoint "$checkpoint" --out "$full_pool" --train-matches-per-archetype 32 \
  --sealed-matches-per-archetype 10 --train-pool-seed 20261005 \
  --sealed-pool-seed 20261006 --candidates-per-match 16 --replicas 8 \
  --branches 7 --horizon 128 --max-decisions 1200 \
  --max-entries-per-split 400 --per-archetype-cap 70 --workers 26 \
  "${reuse_args[@]}" \
  | tee "$out/logs/full_pool.log"

"$python" - "$full_pool/summary.json" "$large_val/summary.json" <<'PY'
import json,sys
full=json.load(open(sys.argv[1]))["counts"]
val=json.load(open(sys.argv[2]))["counts"]
if full["train"] < 300 or full["sealed"] < 80 or val["train"] < 80:
    raise SystemExit(f"dataset gate failed: full={full} validation={val}")
print("DATASET_GATE_PASS", full, val)
PY

status running "GRU context and hard-soft-retention ablations"
"$python" -u "$repo/scripts/run_v4_search_distillation.py" \
  --dataset-root "$full_pool" --validation-root "$large_val" \
  --checkpoint "$checkpoint" --out "$out/search_distillation" \
  --context 32 --contexts 0,32 \
  --methods hard-none,hard-replay,soft-replay,soft-kl \
  --seeds 0,1,2 --epochs 8 --batch-size 8 \
  --max-corrections 300 --max-retention 1200 \
  --validation-retention 500 --min-train-corrections 30 \
  --min-validation-corrections 10 --device cpu \
  | tee "$out/logs/search_distillation.log"

status cpu_complete "teacher audit, target datasets, GRU and four-way ablation complete; GPU/Expert-Iteration stages pending"
echo CPU_CHAIN_COMPLETE
