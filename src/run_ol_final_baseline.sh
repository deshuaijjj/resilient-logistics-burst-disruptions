#!/bin/zsh
# Official run of the baseline that needs no proof (every outage replaced by
# the longest outage with the same first period), on the shared Mac Studio.
# Started by hand after src/run_ol_final_random_rerun.sh has finished:
#
#   zsh src/run_ol_final_baseline.sh
#
# It waits for that rerun, then runs one stage with the usual check that the
# owner's experiments are not running (src/run_ol_final_stage.sh). The last
# log line starts with "=== ... BASELINE DONE".
cd "$(dirname "$0")/.." || exit 1
LOG=logs/ol_final_official.log
while pgrep -f "[r]un_ol_final_random_rerun.sh" >/dev/null 2>&1; do sleep 120; done
zsh src/run_ol_final_stage.sh --stage samefirst --ks 2,4,3 --only 2,3 --repeats 3 --method highs-ds --time-limit 28800
echo "=== $(date '+%F %T') BASELINE DONE" >> $LOG
