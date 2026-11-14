#!/bin/zsh
# Reruns on the shared Mac Studio for the Optimization Letters manuscript,
# started by hand on 2026-11-12 after the random examples were restricted to
# seeds whose demand can all be served when no line fails and capacity is
# unlimited (run_ol_revision.select_random_instances):
#
#   treesize    nodes of the scenario trees (counting only)
#   worstfloor  the worst scenarios operated with unlimited capacity
#   famselect   which seeds are used
#   family      column-and-constraint generation on the 30 random examples,
#               with and without Proposition 1, three runs each
#   learnsfam   the two-stage and the multi-stage problem on the same examples
#
#   zsh src/run_ol_final_random_rerun.sh
#
# The results of the earlier 30 examples (seeds 0 to 29) are moved to
# v1_seeds_0_to_29/ inside their directories before anything is written. Each
# stage waits while another experiment of the owner is running
# (src/run_ol_final_stage.sh). The last log line starts with
# "=== ... RANDOM RERUN DONE".
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
LOG=logs/ol_final_official.log

for d in results/37_random_examples results/39_learning_operator_random; do
  if [ -f $d/family.json ] && [ ! -d $d/v1_seeds_0_to_29 ]; then
    mkdir -p $d/v1_seeds_0_to_29
    mv $d/family.json $d/v1_seeds_0_to_29/family.json
    echo "=== $(date '+%F %T') moved $d/family.json to $d/v1_seeds_0_to_29/" >> $LOG
  fi
done

zsh src/run_ol_final_stage.sh --stage treesize
zsh src/run_ol_final_stage.sh --stage worstfloor --ks 2,3,4
zsh src/run_ol_final_stage.sh --stage famselect --instances 30
zsh src/run_ol_final_stage.sh --stage family --instances 30 --repeats 3 --time-limit 900
zsh src/run_ol_final_stage.sh --stage learnsfam --instances 30 --time-limit 7200
echo "=== $(date '+%F %T') RANDOM RERUN DONE" >> $LOG
