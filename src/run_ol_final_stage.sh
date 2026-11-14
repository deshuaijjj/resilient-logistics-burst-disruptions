#!/bin/zsh
# One more official run for the Optimization Letters manuscript on the shared
# Mac Studio: a single stage of src/run_ol_final.py, with the same waiting and
# logging as src/run_ol_final_all.sh.
#
#   zsh src/run_ol_final_stage.sh --stage learnsccg --k 3 --method highs-ds --time-limit 28800 --jobs 8
#
# Before the stage the script waits while another experiment of the owner is
# running (a run_redraws.py driver, or a busy llama-server), and it notes in
# the log when one was running at the end, because the timings may then be
# disturbed. The last log line of the run starts with "=== ... SINGLE STAGE DONE".
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
PY=${PYTHON:-/usr/bin/python3}
LOG=logs/ol_final_official.log

busy() {
  pgrep -f "[r]un_redraws.py" >/dev/null 2>&1 && return 0
  ps -Ao %cpu,command | awk '$2 ~ /llama-server/ && $1 > 25 {found = 1} END {exit !found}'
}

waited=0
while busy; do
  [ $waited -eq 0 ] && echo "=== $(date '+%F %T') waiting: another experiment is running" >> $LOG
  waited=1
  sleep 300
done
echo "=== $(date '+%F %T') START src/run_ol_final.py $* | load $(sysctl -n vm.loadavg)" >> $LOG
"$PY" src/run_ol_final.py "$@" >> $LOG 2>&1
code=$?
note=""
busy && note=" | ANOTHER EXPERIMENT WAS RUNNING AT THE END"
echo "=== $(date '+%F %T') END (exit $code) src/run_ol_final.py $* | load $(sysctl -n vm.loadavg)$note" >> $LOG
echo "=== $(date '+%F %T') SINGLE STAGE DONE (exit $code) $*" >> $LOG
