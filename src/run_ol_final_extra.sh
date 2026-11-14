#!/bin/zsh
# Further official runs for the Optimization Letters manuscript, started by
# hand after src/run_ol_final_all.sh has finished.
#
#   zsh src/run_ol_final_extra.sh                   the first two stages
#   zsh src/run_ol_final_extra.sh --run --with-k3   also the three-outage attempt
#   zsh src/run_ol_final_extra.sh --dry-run         print the stages only
#
# The machine is shared. Before each stage the script waits while another
# experiment of the owner is running (a run_redraws.py driver, or a busy
# llama-server), and it notes in the log when one was running at the end of a
# stage, because the timings of that stage may then be disturbed.
#
# Stages: the variants with late demand; the multi-stage problem over every
# scenario with the dual simplex method (about two hours on a laptop); the
# multi-stage problem with three outages over the scenarios of Theorem 2
# (285,077 tree nodes; an attempt, with a time limit of eight hours).
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
PY=${PYTHON:-/usr/bin/python3}
LOG=logs/ol_final_official.log
DRY=0
[ "$1" = "--dry-run" ] && DRY=1

busy() {
  pgrep -f "[r]un_redraws.py" >/dev/null 2>&1 && return 0
  ps -Ao %cpu,command | awk '$2 ~ /llama-server/ && $1 > 25 {found = 1} END {exit !found}'
}

stage() {
  if [ $DRY -eq 1 ]; then echo "$PY $*"; return; fi
  local waited=0
  while busy; do
    [ $waited -eq 0 ] && echo "=== $(date '+%F %T') waiting: another experiment is running" >> $LOG
    waited=1
    sleep 300
  done
  echo "=== $(date '+%F %T') START $* | load $(sysctl -n vm.loadavg)" >> $LOG
  "$PY" "$@" >> $LOG 2>&1
  local code=$?
  local note=""
  busy && note=" | ANOTHER EXPERIMENT WAS RUNNING AT THE END"
  echo "=== $(date '+%F %T') END (exit $code) $* | load $(sysctl -n vm.loadavg)$note" >> $LOG
}

stage src/run_ol_final.py --stage variants --time-limit 7200
stage src/run_ol_final.py --stage learns --k 2 --only-all --method highs-ds --repeats 1 --time-limit 21600
[ "$2" = "--with-k3" ] && stage src/run_ol_final.py --stage learns --k 3 --only splus --method highs-ds --repeats 1 --time-limit 28800
[ $DRY -eq 1 ] || echo "=== $(date '+%F %T') EXTRA STAGES DONE" >> $LOG
