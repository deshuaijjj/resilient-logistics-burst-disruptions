#!/bin/zsh
# Runs on the shared Mac Studio for the third revision of the Optimization
# Letters manuscript (2026-11-12, after the fourth simulated review):
#
#   variantsregret  the design found with the scenarios of Proposition 1 in the
#                   late-demand variants where they understate the cost,
#                   operated over the scenarios of Theorem 1
#   trees k=2       the three tree linear programs of Table 4 in one session,
#                   three runs, with simplex iterations
#   famtrees        the same three trees on the thirty random examples
#   family          column-and-constraint generation with all outages, with the
#                   longest ones and with those of Proposition 1, in one session
#   milpsep         the same three columns for the lunar example, k = 2, 3
#   trees k=3       the three trees of Table 4 for k = 3, one run each
#
#   zsh src/run_ol_final_overnight.sh
#
# Each stage waits while another experiment of the owner is running
# (src/run_ol_final_stage.sh). The last log line starts with
# "=== ... OVERNIGHT DONE".
cd "$(dirname "$0")/.." || exit 1
LOG=logs/ol_final_official.log
S=src/run_ol_final_stage.sh
V=all_single_outages,longest_any_start,longest_ending_inside
zsh $S --stage variantsregret --time-limit 7200
zsh $S --stage trees --ks 2 --repeats 3 --method highs-ds --time-limit 7200
zsh $S --stage famtrees --instances 30 --method highs-ds --time-limit 7200
zsh $S --stage family --instances 30 --repeats 3 --time-limit 900 --variants $V --out-suffix _three_columns
zsh $S --stage milpsep --ks 2,3 --repeats 3 --time-limit 1800 --variants $V --out-suffix _three_columns
zsh $S --stage trees --ks 3 --repeats 1 --method highs-ds --time-limit 28800 --out-suffix _k3
echo "=== $(date '+%F %T') OVERNIGHT DONE" >> $LOG
