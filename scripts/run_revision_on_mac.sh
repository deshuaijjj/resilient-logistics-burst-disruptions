#!/usr/bin/env bash
# One-shot driver for the September 2026 revision experiments (results/12-17),
# the generated LaTeX macros, and a build of the manuscript.
#
#   bash scripts/run_revision_on_mac.sh            # gate rerun verify dual benders ccgmax seeds (~30-40 min)
#   bash scripts/run_revision_on_mac.sh --fullef   # additionally the monolithic EF attempt (hours; run overnight)
#   bash scripts/run_revision_on_mac.sh --skip-tests
#
# Run from any directory; the script cds to the repository root.  Every step
# logs to logs/revision_<timestamp>/.  Nothing under results/08-11 is touched.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="logs/revision_${STAMP}"
mkdir -p "$LOGDIR"
PY="${PYTHON:-python3}"
RUN_FULLEF=0
SKIP_TESTS=0
for arg in "$@"; do
  case "$arg" in
    --fullef) RUN_FULLEF=1 ;;
    --skip-tests) SKIP_TESTS=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

echo "== repository: $ROOT"
echo "== logs:       $LOGDIR"
echo "== python:     $($PY -c 'import sys; print(sys.executable, sys.version.split()[0])')"
$PY - <<'EOF'
import importlib, sys
missing = []
for mod in ("numpy", "scipy", "pandas"):
    try:
        m = importlib.import_module(mod); print(f"   {mod} {m.__version__}")
    except Exception:
        missing.append(mod)
import scipy
major, minor = (int(x) for x in scipy.__version__.split(".")[:2])
if (major, minor) < (1, 9):
    missing.append("scipy>=1.9 (scipy.optimize.milp)")
if missing:
    sys.exit("missing: " + ", ".join(missing) + "  -> pip install numpy scipy pandas")
EOF

if [ "$SKIP_TESTS" -eq 0 ]; then
  echo "== unit tests (small instance, ~30 s)"
  if $PY -c "import pytest" 2>/dev/null; then
    $PY -m pytest tests/test_revision_experiments.py -q 2>&1 | tee "$LOGDIR/tests.log"
  else
    echo "   pytest not installed; skipping tests (pip install pytest to enable)"
  fi
fi

echo "== experiments: gate rerun verify dual benders ccgmax seeds"
$PY src/run_proano_revision.py --all 2>&1 | tee "$LOGDIR/run_all.log"

if [ "$RUN_FULLEF" -eq 1 ]; then
  echo "== optional: monolithic extensive form over all distinct matrices (time limit 3 h)"
  $PY src/run_proano_revision.py --stage fullef --fullef-time-limit 10800 2>&1 | tee "$LOGDIR/run_fullef.log"
fi

echo "== generated LaTeX macros"
$PY src/make_revision_numbers.py 2>&1 | tee "$LOGDIR/make_numbers.log" | tail -5

echo "== acceptance summary"
$PY - <<'EOF'
import json, os
root = "results"
stages = {"gate": "12_revision_gate/gate_result.json", "rerun": "12_revision_gate/rerun_result.json",
          "verify": "13_fixed_design_verification/verify_result.json", "dual": "14_dual_oracle/dual_result.json",
          "benders": "15_benders_cross_check/benders_result.json", "ccgmax": "16_ccg_maximal_separation/ccgmax_result.json",
          "seeds": "17_sampling_seeds/seeds_result.json"}
for stage, rel in stages.items():
    path = os.path.join(root, rel)
    if not os.path.exists(path):
        print(f"   {stage:8s} MISSING {rel}"); continue
    d = json.load(open(path))
    if "all_checks_pass" in d:
        flag = "PASS" if d["all_checks_pass"] else "CHECK"
        print(f"   {stage:8s} {flag}  {json.dumps(d.get('checks', {}))}")
    elif stage == "seeds":
        s = d["summary"]; print(f"   {stage:8s} even-index regression ok={s['even_index_regression']['complete_library_ok']}  "
                                f"random {s['random_samples']}: gap>1e9 in {s['random_samples_with_gap_over_1e9']}, contain 1295: {s['random_samples_containing_1295']}")
    else:
        print(f"   {stage:8s} done ({d.get('wall_time_s', 0):.0f}s)")
EOF

if command -v pdflatex >/dev/null 2>&1; then
  echo "== manuscript build (paper/ieee_aerospace_2027/build_revision_${STAMP})"
  ( cd paper/ieee_aerospace_2027 && mkdir -p "build_revision_${STAMP}" \
    && pdflatex -interaction=nonstopmode -halt-on-error -output-directory="build_revision_${STAMP}" main.tex > "build_revision_${STAMP}/pass1.log" 2>&1 \
    && bibtex "build_revision_${STAMP}/main" > "build_revision_${STAMP}/bibtex.log" 2>&1 \
    && pdflatex -interaction=nonstopmode -halt-on-error -output-directory="build_revision_${STAMP}" main.tex > "build_revision_${STAMP}/pass2.log" 2>&1 \
    && pdflatex -interaction=nonstopmode -halt-on-error -output-directory="build_revision_${STAMP}" main.tex > "build_revision_${STAMP}/pass3.log" 2>&1 \
    && echo "   built: paper/ieee_aerospace_2027/build_revision_${STAMP}/main.pdf" \
    && (command -v pdfinfo >/dev/null 2>&1 && pdfinfo "build_revision_${STAMP}/main.pdf" | grep Pages || true) )
else
  echo "== pdflatex not on PATH (macOS: export PATH=/Library/TeX/texbin:\$PATH); skipping the build"
fi
echo "== done. Logs in $LOGDIR"
