#!/usr/bin/env bash
# Complete September 2026 revision run, laid out for a quiet multi-core machine.
#
#   bash scripts/run_revision_full.sh                 # everything, 100 random seeds, fullef up to 6 h
#   SEEDS=200 bash scripts/run_revision_full.sh       # more sampling seeds
#   SKIP_FULLEF=1 bash scripts/run_revision_full.sh   # stop before the monolithic LP
#   FULLEF_TIME_LIMIT=43200 bash scripts/run_revision_full.sh
#
# Phase 1 is TIMED: gate, rerun, ccgmax and benders produce the same-session
# wall-clock table in Appendix B, so they run one at a time with solver
# threading pinned to one thread and nothing else on the machine.  Phase 2 is
# untimed and runs its three stages concurrently.  Phase 3 is the monolithic
# extensive form, which wants the whole machine and may exhaust memory; it runs
# last and in its own process so a crash cannot take the other results with it.
#
# Nothing under results/08-11 is read-write; every stage writes its own new
# directory with an environment record and a SHA-256 manifest.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/full_${STAMP}"
mkdir -p "$LOG"
PY="${PYTHON:-python3}"
SEEDS="${SEEDS:-100}"
FULLEF_TIME_LIMIT="${FULLEF_TIME_LIMIT:-21600}"
SKIP_FULLEF="${SKIP_FULLEF:-0}"
SKIP_TESTS="${SKIP_TESTS:-0}"
STATUS="$LOG/STATUS.txt"

say() { printf '%s  %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$STATUS"; }

say "repository : $ROOT"
say "logs       : $LOG"
say "seeds      : $SEEDS      fullef limit: ${FULLEF_TIME_LIMIT}s   skip fullef: $SKIP_FULLEF"

# ---------------------------------------------------------------- preflight --
{
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(hostname)"
  uname -a
  sysctl -n hw.model hw.ncpu hw.memsize 2>/dev/null || true
  "$PY" -c 'import sys; print("python:", sys.executable, sys.version.split()[0])'
  df -h . | tail -1
} > "$LOG/environment.txt" 2>&1
cat "$LOG/environment.txt" | sed 's/^/    /' | tee -a "$STATUS" > /dev/null

"$PY" - <<'EOF' || { say "PREFLIGHT FAILED"; exit 2; }
import importlib, sys
for mod in ("numpy", "scipy", "pandas"):
    m = importlib.import_module(mod)
    print(f"    {mod} {m.__version__}")
import scipy
if tuple(int(x) for x in scipy.__version__.split(".")[:2]) < (1, 9):
    sys.exit("scipy >= 1.9 required for scipy.optimize.milp")
EOF

for f in src/revision_common.py src/run_proano_revision.py src/make_revision_numbers.py \
         results/09_burst_ccg/ccg_summary.csv results/11_dominance_pruned_oracle/dominance_summary.json; do
  [ -f "$f" ] || { say "MISSING $f — is this the right repository copy?"; exit 2; }
done

if [ "$SKIP_TESTS" -eq 0 ] && "$PY" -c "import pytest" 2>/dev/null; then
  say "unit tests (small instance)"
  "$PY" -m pytest tests/test_revision_experiments.py -q > "$LOG/tests.log" 2>&1 \
    && say "  tests PASS" || { say "  tests FAILED — see $LOG/tests.log"; exit 3; }
fi

# ------------------------------------------------- phase 1: timed, serial ----
# One thread per solver so the four same-session wall-clock rows are mutually
# comparable and reproducible.  The environment record stores this setting.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
say "PHASE 1 (timed, serial, 1 solver thread): gate rerun ccgmax benders"
for stage in gate rerun ccgmax benders; do
  say "  stage $stage ..."
  t0=$(date +%s)
  "$PY" src/run_proano_revision.py --stage "$stage" > "$LOG/p1_${stage}.log" 2>&1
  rc=$?
  say "  stage $stage done rc=$rc in $(( $(date +%s) - t0 ))s"
done
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS \
      VECLIB_MAXIMUM_THREADS NUMEXPR_NUM_THREADS

# --------------------------------------------- phase 2: untimed, parallel ----
say "PHASE 2 (untimed, concurrent): verify dual seeds(${SEEDS})"
"$PY" src/run_proano_revision.py --stage verify                > "$LOG/p2_verify.log" 2>&1 &
P_VERIFY=$!
"$PY" src/run_proano_revision.py --stage dual                  > "$LOG/p2_dual.log"   2>&1 &
P_DUAL=$!
"$PY" src/run_proano_revision.py --stage seeds --seeds "$SEEDS" > "$LOG/p2_seeds.log" 2>&1 &
P_SEEDS=$!
wait $P_VERIFY; say "  verify rc=$?"
wait $P_DUAL;   say "  dual   rc=$?"
wait $P_SEEDS;  say "  seeds  rc=$?"

if [ -d paper/ieee_aerospace_2027/sections ]; then
  say "generating LaTeX macros from results/12-17"
  "$PY" src/make_revision_numbers.py > "$LOG/make_numbers.log" 2>&1 \
    && say "  macros written" || say "  MACRO GENERATION FAILED — see $LOG/make_numbers.log"
else
  say "no paper/ directory in this copy; macros will be generated where the manuscript lives"
fi

# ------------------------------------------------ phase 3: monolithic LP ----
if [ "$SKIP_FULLEF" -eq 0 ]; then
  say "PHASE 3 (alone): monolithic extensive form over all distinct matrices"
  say "  ~15.4M variables, ~8.4M rows, ~50M nonzeros; may exhaust memory, which is a reportable outcome"
  "$PY" src/run_proano_revision.py --stage fullef --fullef-method highs-ipm \
        --fullef-time-limit "$FULLEF_TIME_LIMIT" > "$LOG/p3_fullef_ipm.log" 2>&1
  say "  interior point rc=$?"
  if ! "$PY" - <<'EOF'
import json, os, sys
p = "results/18_full_ef_attempt/full_ef_result.json"
sys.exit(0 if os.path.exists(p) and json.load(open(p)).get("success") else 1)
EOF
  then
    say "  interior point did not return a solution; retrying with dual simplex"
    mkdir -p results/18_full_ef_attempt
    [ -f results/18_full_ef_attempt/full_ef_result.json ] && \
      mv results/18_full_ef_attempt/full_ef_result.json results/18_full_ef_attempt/full_ef_result_ipm.json
    "$PY" src/run_proano_revision.py --stage fullef --fullef-method highs \
          --fullef-time-limit "$FULLEF_TIME_LIMIT" > "$LOG/p3_fullef_simplex.log" 2>&1
    say "  dual simplex rc=$?"
  fi
fi

# ------------------------------------------------------------- acceptance ----
say "ACCEPTANCE SUMMARY"
"$PY" - <<'EOF' 2>&1 | tee -a "$STATUS"
import json, os
rows = [("gate",    "12_revision_gate/gate_result.json"),
        ("rerun",   "12_revision_gate/rerun_result.json"),
        ("verify",  "13_fixed_design_verification/verify_result.json"),
        ("dual",    "14_dual_oracle/dual_result.json"),
        ("benders", "15_benders_cross_check/benders_result.json"),
        ("ccgmax",  "16_ccg_maximal_separation/ccgmax_result.json"),
        ("seeds",   "17_sampling_seeds/seeds_result.json"),
        ("fullef",  "18_full_ef_attempt/fullef_result.json")]
for stage, rel in rows:
    path = os.path.join("results", rel)
    if not os.path.exists(path):
        print(f"    {stage:8s} MISSING"); continue
    d = json.load(open(path))
    if "all_checks_pass" in d:
        bad = [k for k, v in d.get("checks", {}).items() if not v]
        print(f"    {stage:8s} {'PASS' if d['all_checks_pass'] else 'CHECK: ' + ','.join(bad)}"
              f"   ({d.get('wall_time_s', 0):.0f}s)")
    elif stage == "seeds":
        s = d["summary"]
        print(f"    {stage:8s} even-index regression ok={s['even_index_regression']['complete_library_ok']}; "
              f"{s['random_samples']} random: {s['random_samples_with_gap_over_1e9']} with gap > $1B, "
              f"{s['random_samples_containing_1295']} contain encoding 1295   ({d.get('wall_time_s', 0):.0f}s)")
    elif stage == "fullef":
        f = d["full_ef"]
        print(f"    {stage:8s} success={f.get('success')} status={f.get('status')} "
              f"solve={f.get('solve_time_s', 0):.0f}s dims={f.get('dims', {}).get('variables')}")
    else:
        print(f"    {stage:8s} done ({d.get('wall_time_s', 0):.0f}s)")
EOF

# ------------------------------------------------------------------ build ----
if [ -d paper/ieee_aerospace_2027 ] && { command -v pdflatex >/dev/null 2>&1 || [ -x /Library/TeX/texbin/pdflatex ]; }; then
  export PATH="/Library/TeX/texbin:$PATH"
  say "building the manuscript"
  BUILD="build_revision_${STAMP}"
  ( cd paper/ieee_aerospace_2027 && mkdir -p "$BUILD" \
    && pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD" main.tex > "$BUILD/pass1.log" 2>&1 \
    && bibtex "$BUILD/main" > "$BUILD/bibtex.log" 2>&1 \
    && pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD" main.tex > "$BUILD/pass2.log" 2>&1 \
    && pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD" main.tex > "$BUILD/pass3.log" 2>&1 ) \
    && say "  built paper/ieee_aerospace_2027/${BUILD}/main.pdf" \
    || say "  BUILD FAILED — see paper/ieee_aerospace_2027/${BUILD}/pass*.log"
else
  say "no manuscript here or pdflatex missing; skipping the build (build on the machine holding paper/)"
fi

say "ALL DONE.  Status file: $STATUS"
