#!/bin/zsh
# Official runs for the Optimization Letters manuscript, one stage after
# another, on the machine whose timings are reported.
#
#   zsh src/run_ol_final_all.sh            all stages
#   zsh src/run_ol_final_all.sh --dry-run  print the stages only
#
# The machine is shared. Before each stage the script waits while another
# experiment of the owner is running (a run_redraws.py driver, or a busy
# llama-server), and it notes in the log when one was running at the end of a
# stage, because the timings of that stage may then be disturbed.
#
# Not in this list: the attempt to solve the learning operator's problem over
# every scenario (--stage learns --only-all), which is started by hand once the
# method and the time limit are chosen.
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

stage src/ol_scenarios.py
stage src/run_ol_final.py --stage counts
stage src/run_ol_final.py --stage milpsep --ks 2 --repeats 3 --time-limit 1800
stage src/run_ol_final.py --stage milpsep --ks 4 --repeats 3 --time-limit 1800
stage src/run_ol_final.py --stage milpsep --ks 3 --repeats 3 --time-limit 1800
stage src/run_ol_final.py --stage learns --k 2 --repeats 3 --method highs-ds --time-limit 7200
stage src/run_ol_final.py --stage family --instances 30 --repeats 3 --time-limit 900
stage src/run_ol_final.py --stage verify --ks 2,3 --jobs 8
stage src/run_ol_final.py --stage learnsfam --instances 30 --time-limit 7200
[ $DRY -eq 1 ] || echo "=== $(date '+%F %T') ALL STAGES DONE" >> $LOG
