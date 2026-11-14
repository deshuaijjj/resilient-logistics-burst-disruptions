"""Numbers for the Optimization Letters text, computed from results/34..43.

Run:  python3 paper/optimization_letters/checks/make_numbers_final.py [--suffix _laptop]
Writes numbers_final.tex next to main.tex and prints a summary. A results
file that does not exist yet is skipped and reported, so the script can be
run while the official runs are still going.

The results are written by src/run_ol_final.py (scenario definition of
2026-11-11: an outage may start in any period and is cut off by the end of
the horizon). Every macro starts with \\F.
"""
import argparse
import json
import os
import statistics as st

def _repo_root(start):
    """The repository root: the first directory above this file that holds src/ and results/."""
    d = start
    while True:
        if os.path.isdir(os.path.join(d, "src")) and os.path.isdir(os.path.join(d, "results")):
            return d
        up = os.path.dirname(d)
        if up == d:
            raise RuntimeError("repository root (with src/ and results/) not found above " + start)
        d = up


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _repo_root(HERE)
# numbers_final.tex is written next to main.tex when the script sits in the
# manuscript folder, otherwise next to the script.
OUT = (os.path.join(HERE, "..", "numbers_final.tex") if os.path.exists(os.path.join(HERE, "..", "main.tex"))
       else os.path.join(HERE, "numbers_final.tex"))
WORD = {1: "One", 2: "Two", 3: "Three", 4: "Four"}
macros, notes = {}, []


def put(name, value):
    assert name not in macros, name
    macros[name] = str(value)


def load(directory, filename, suffix):
    path = os.path.join(ROOT, "results", directory + suffix, filename)
    if not os.path.exists(path):
        notes.append(f"missing: results/{directory}{suffix}/{filename}")
        return None
    return json.load(open(path))


def secs(x):
    """Seconds as they are printed: one decimal below 100 s, whole seconds above."""
    return f"{x:.1f}" if x < 99.95 else f"{x:,.0f}"


def counts(suffix):
    d = load("34_scenario_counts", "counts.json", suffix)
    if d is None:
        return
    put("FSingleAll", d["single_outages"]["all"])
    put("FSingleEnd", d["single_outages"]["thm1"])
    put("FSingleStart", d["single_outages"]["splus"])
    for r in d["rows"]:
        w = WORD[r["k"]]
        put(f"FAll{w}", f"{r['scenarios']:,}")
        put(f"FEnd{w}", f"{r['thm1']:,}")
        put(f"FStart{w}", f"{r['splus']:,}")
        put(f"FEndPct{w}", f"{r['thm1_share_pct']:.1f}" if r["thm1_share_pct"] >= 1 else f"{r['thm1_share_pct']:.2f}")
        put(f"FStartPct{w}", f"{r['splus_share_pct']:.1f}" if r["splus_share_pct"] >= 1 else f"{r['splus_share_pct']:.2f}")
        assert r["T_at_least_k_times_longest"] and r["k_at_most_sum_of_ceil_T_over_longest"]
        if "pairwise_filter_s" in r:
            assert r["filter_equals_thm1"]
            put(f"FList{w}S", secs(r["list_all_s"]))
            put(f"FFilter{w}S", secs(r["pairwise_filter_s"]))
        put(f"FWriteEnd{w}S", f"{r['write_down_thm1_s']:.2f}")
        put(f"FWriteStart{w}S", f"{r['write_down_splus_s']:.2f}")
    put("FAllFourMillions", f"{d['rows'][3]['scenarios'] / 1e6:.0f}")
    # at most k longest outages, each beginning in any period: the empty scenario and exactly j of them,
    # j = 1..k (different j give different matrices, see the proof of Theorem 1(ii))
    base = 1
    for r in d["rows"]:
        base += r["splus"]
        w = WORD[r["k"]]
        put(f"FBaseCount{w}", f"{base:,}")
        pct = 100 * base / r["scenarios"]
        put(f"FBaseCountPct{w}", f"{pct:.1f}" if pct >= 1 else f"{pct:.2f}")


def milp(suffix):
    names = {"all_single_outages": "All", "longest_ending_inside": "End", "longest_ending_inside_exactly_k": "EndExact"}
    spreads, longests = [], []
    for k in (2, 3, 4):
        d = load("35_milp_separation", f"milp_separation_k{k}.json", suffix)
        if d is None:
            continue
        w = WORD[k]
        costs = []
        for key, short in names.items():
            s = d["summary"].get(key)
            if s is None:
                continue
            put(f"FMilp{short}{w}Vars", s["zero_one_variables"])
            if s["finished_runs"]:
                put(f"FMilp{short}{w}Rounds", s["rounds"])
                put(f"FMilp{short}{w}S", secs(s["median_wall_s"]))
                put(f"FMilp{short}{w}Runs", s["finished_runs"])
                costs.append(s["total_cost"])
            else:
                put(f"FMilp{short}{w}Rounds", f"not finished ({s['rounds_before_stop'] - 1} rounds done)")
                put(f"FMilp{short}{w}S", f"stopped after {s['wall_s_before_stop'] / 60:.0f} min")
                put(f"FMilp{short}{w}Runs", 0)
        if costs:
            assert max(costs) - min(costs) <= 1.0, (k, costs)
            # one "rounds" column in the table: the variants take the same number of rounds
            assert len({s["rounds"] for s in d["summary"].values() if s["finished_runs"]}) == 1, k
            put(f"FCost{w}M", f"{costs[0] / 1e6:,.2f}")
            put(f"FCost{w}MExact", f"{costs[0] / 1e6:,.3f}")
            put(f"FCost{w}", f"{costs[0]:,.2f}")
        put(f"FMilpLimit{w}Min", f"{d['time_limit_per_milp_s'] / 60:.0f}")
        sa, se = d["summary"].get("all_single_outages"), d["summary"].get("longest_ending_inside")
        if sa and se and sa["finished_runs"] and se["finished_runs"]:
            put(f"FMilpRatio{w}", f"{sa['median_wall_s'] / se['median_wall_s']:.1f}")
        # slowest run over fastest run of the same variant, largest over the variants and over k
        spreads += [max(s["wall_s_all_runs"]) / min(s["wall_s_all_runs"]) - 1
                    for s in d["summary"].values() if s["finished_runs"] > 1]
        # the longest single mixed-integer program of any run, against the time limit
        longest = max(r["milp_s"] for runs in d["runs"].values() for run in runs for r in run["round_log"])
        assert longest < d["time_limit_per_milp_s"], (k, longest)
        put(f"FMilpLongest{w}Min", f"{longest / 60:.1f}")
        longests.append(longest)
    if spreads:
        put("FMilpSpreadPctMax", f"{100 * max(spreads):.1f}")
    if longests:
        put("FMilpLongestMin", f"{max(longests) / 60:.0f}")


def verify(suffix):
    for k in (2, 3):
        d = load("36_all_scenarios_check", f"verify_k{k}.json", suffix)
        if d is None:
            continue
        w = WORD[k]
        assert abs(d["difference"]) <= 1.0, d["difference"]
        put(f"FVerify{w}N", f"{d['scenarios_checked']:,}")
        put(f"FVerify{w}Worst", f"{d['worst_operating_cost_all_scenarios']:,.2f}")
        put(f"FVerify{w}Min", f"{d['wall_s'] / 60:.0f}" if d["wall_s"] >= 90 else f"{d['wall_s'] / 60:.1f}")


def family(suffix):
    d = load("37_random_examples", "family.json", suffix)
    if d is None:
        return
    rows = d["rows"]
    done = [r for r in rows if "time_ratio" in r]
    put("FFamN", len(rows))
    put("FFamBothFinished", len(done))
    if done:
        assert all(r["same_cost"] for r in done)
        ratio = [r["time_ratio"] for r in done]
        a = [r["summary"]["all_single_outages"]["median_wall_s"] for r in done]
        b = [r["summary"]["longest_ending_inside"]["median_wall_s"] for r in done]
        put("FFamRatioMin", f"{min(ratio):.1f}"); put("FFamRatioMed", f"{st.median(ratio):.1f}"); put("FFamRatioMax", f"{max(ratio):.1f}")
        put("FFamAllSMin", secs(min(a))); put("FFamAllSMed", secs(st.median(a))); put("FFamAllSMax", secs(max(a)))
        put("FFamEndSMin", secs(min(b))); put("FFamEndSMed", secs(st.median(b))); put("FFamEndSMax", secs(max(b)))
        put("FFamSlowerWithTheorem", sum(1 for x in ratio if x < 1.0))
        put("FFamSumRatio", f"{sum(a) / sum(b):.1f}")
        va = [r["summary"]["all_single_outages"]["zero_one_variables"] for r in done]
        vb = [r["summary"]["longest_ending_inside"]["zero_one_variables"] for r in done]
        put("FFamAllVarsMin", min(va)); put("FFamAllVarsMax", max(va))
        put("FFamEndVarsMin", min(vb)); put("FFamEndVarsMax", max(vb))
        rounds = [r["summary"]["all_single_outages"]["rounds"] for r in done]
        assert rounds == [r["summary"]["longest_ending_inside"]["rounds"] for r in done]
        put("FFamRoundsMin", min(rounds)); put("FFamRoundsMax", max(rounds))
        put("FFamMilpLimitMin", f"{d['time_limit_per_milp_s'] / 60:.0f}")
    selected_seeds(suffix, [r["seed"] for r in rows], "37_random_examples")


def learns(suffix):
    d = None
    for name in ("lunar_k2_ds.json", "lunar_k2.json"):
        if os.path.exists(os.path.join(ROOT, "results", "38_learning_operator" + suffix, name)):
            d = load("38_learning_operator", name, suffix)
            break
    if d is None:
        notes.append(f"missing: results/38_learning_operator{suffix}/lunar_k2*.json")
    else:
        r = d["runs"]
        a, b = r["knows_scenario__thm1_set"], r["learns_as_outages_happen__splus_set"]
        c = r["learns_as_outages_happen__thm1_set"]
        e = r["design_for_knowing__operated_learning__splus_set"]
        put("FKnowsCost", f"{a['total_cost']:,.0f}"); put("FLearnsCost", f"{b['total_cost']:,.0f}")
        put("FLearnsMinusKnows", f"{b['total_cost'] - a['total_cost']:,.0f}")
        put("FLearnsOnEndMinusLearns", f"{c['total_cost'] - b['total_cost']:.2f}")
        put("FKnowingDesignExtra", f"{e['total_cost'] - b['total_cost']:,.0f}")
        put("FKnowsQTwo", f"{a['q'][1]:.2f}"); put("FLearnsQTwo", f"{b['q'][1]:.2f}")
        put("FLearnsNodes", f"{b['nodes']:,}"); put("FLearnsOnEndNodes", f"{c['nodes']:,}")
        put("FLearnsVarsMillions", f"{b['variables'] / 1e6:.2f}")
        put("FLearnsOnEndVarsMillions", f"{c['variables'] / 1e6:.2f}")
        put("FLearnsCostM", f"{b['total_cost'] / 1e6:.3f}"); put("FKnowsCostM", f"{a['total_cost'] / 1e6:.3f}")
        put("FLearnsOnEndCostM", f"{c['total_cost'] / 1e6:.3f}")
        put("FKnowingDesignPromiseExtra", f"{e['total_cost'] - a['total_cost']:,.0f}")
        put("FKnowsS", secs(a.get("median_solve_s", a["solve_s"])))
        put("FLearnsS", secs(b.get("median_solve_s", b["solve_s"])))
        put("FLearnsOnEndS", secs(c.get("median_solve_s", c["solve_s"])))
    for name in sorted(os.listdir(os.path.join(ROOT, "results", "38_learning_operator" + suffix))
                       if os.path.isdir(os.path.join(ROOT, "results", "38_learning_operator" + suffix)) else []):
        if not name.startswith("lunar_k2_all"):
            continue
        x = json.load(open(os.path.join(ROOT, "results", "38_learning_operator" + suffix, name)))["runs"]
        full = x.get("learns_as_outages_happen__all_scenarios") or x.get("learns_as_he_goes__all_scenarios")
        if full is None:
            continue
        tag = "Ipm" if "ipm" in name else "Ds"
        put(f"FLearnsAllNodes{tag}", f"{full['nodes']:,}")
        put(f"FLearnsAllVarsMillions{tag}", f"{full['variables'] / 1e6:.1f}")
        if full["status"] == 0:
            put(f"FLearnsAll{tag}", f"solved in {full['solve_s'] / 60:.0f} min, cost {full['total_cost']:,.2f}")
            put(f"FLearnsAllMin{tag}", f"{full['solve_s'] / 60:.0f}")
            put(f"FLearnsAllS{tag}", secs(full["solve_s"]))
            put(f"FLearnsAllRuns{tag}", len(full.get("solve_s_all_runs", [full["solve_s"]])))
            put(f"FLearnsAllHours{tag}", f"{full['solve_s'] / 3600:.1f}")
            put(f"FLearnsAllScenarios{tag}", f"{full['scenarios']:,}")
            if d is not None:
                # the same optimal cost over every scenario and over the set of the multi-stage theorem
                assert abs(full["total_cost"] - b["total_cost"]) <= 1e-6 * b["total_cost"], (full["total_cost"], b["total_cost"])
                put(f"FLearnsAllMinusLearns{tag}", f"{round(full['total_cost'] - b['total_cost'], 2) + 0.0:,.2f}")
        else:
            put(f"FLearnsAll{tag}", f"not solved in {full['solve_s'] / 3600:.1f} h")
    # the same linear program over every scenario on a second machine (kept as a second record)
    second = os.path.join(ROOT, "results", "38_learning_operator_laptop", "lunar_k2_all_ds.json")
    if suffix == "" and os.path.exists(second):
        x = json.load(open(second))
        full = x["runs"].get("learns_as_outages_happen__all_scenarios") or x["runs"].get("learns_as_he_goes__all_scenarios")
        if full["status"] == 0 and d is not None:
            assert abs(full["total_cost"] - b["total_cost"]) <= 1e-6 * b["total_cost"]
            put("FLearnsAllMinSecondMachine", f"{full['solve_s'] / 60:.0f}")


def learns_family(suffix):
    d = load("39_learning_operator_random", "family.json", suffix)
    if d is None:
        return
    rows = d["rows"]
    diff = [r["learns_total_cost"] - r["knows_total_cost"] for r in rows]
    rel = [100 * x / r["knows_total_cost"] for x, r in zip(diff, rows)]
    short = [r["learns_total_cost"] - r["learns_total_cost_on_thm1_set"] for r in rows]
    regret = [r["design_for_knowing_operated_learning_total_cost"] - r["learns_total_cost"] for r in rows]
    put("FLearnFamN", len(rows))
    put("FLearnFamEqual", sum(1 for x in diff if abs(x) < 1.0))
    put("FLearnFamDiffMax", f"{max(diff):,.0f}"); put("FLearnFamDiffMed", f"{st.median(diff):,.0f}")
    put("FLearnFamRelMaxPct", f"{max(rel):.2f}"); put("FLearnFamRelMedPct", f"{st.median(rel):.3f}")
    assert min(diff) > -1.0, min(diff)          # the multi-stage optimum is never below the two-stage one
    put("FLearnFamRelMinPct", "0" if min(diff) < 1.0 else f"{min(rel):.3f}")
    put("FLearnFamDiffMin", f"{max(min(diff), 0.0):,.0f}")
    put("FLearnFamEndSetShort", sum(1 for x in short if x > 1.0))
    put("FLearnFamRegretMax", f"{max(regret):,.0f}")
    put("FLearnFamRegretMaxPct", f"{max(100 * x / r['learns_total_cost'] for x, r in zip(regret, rows)):.2f}")
    put("FLearnFamCostMinM", f"{min(r['knows_total_cost'] for r in rows) / 1e6:.0f}")
    put("FLearnFamCostMaxM", f"{max(r['knows_total_cost'] for r in rows) / 1e6:,.0f}")
    selected_seeds(suffix, [r["seed"] for r in rows], "39_learning_operator_random")


def selected_seeds(suffix, seeds, where):
    """The random examples are the seeds of run_ol_final.py --stage famselect: all of their
    demand can be served when no line fails and capacity is unlimited."""
    path = os.path.join(ROOT, "results", "37_random_examples" + suffix, "seed_selection.json")
    if not os.path.exists(path):          # one selection serves every stage of the random examples
        path = os.path.join(ROOT, "results", "37_random_examples", "seed_selection.json")
    if not os.path.exists(path):
        if "missing: seed selection" not in notes:
            notes.append("missing: seed selection")
            notes.append(f"missing: results/37_random_examples{suffix}/seed_selection.json "
                         "(the random examples are then seeds 0..n-1, some with demand that no design can serve)")
        return
    f = json.load(open(path))
    if len(seeds) < len(f["seeds_used"]) and seeds == f["seeds_used"][:len(seeds)]:
        notes.append(f"unfinished: results/{where}{suffix} has {len(seeds)} of {len(f['seeds_used'])} selected seeds")
        return
    assert seeds == f["seeds_used"], f"results/{where}{suffix}: seeds {seeds} are not the selected seeds {f['seeds_used']}"
    assert all(r["unmet_tons"] <= 1e-6 for r in f["tried"] if r["used"])
    if "FFamSeedsTried" not in macros:
        put("FFamSeedsTried", f["seeds_tried"]); put("FFamSeedsNotUsed", len(f["seeds_not_used"]))


def learns_three(suffix):
    """Three outages: results of run_ol_final.py --stage learnsccg --k 3."""
    d = load("38_learning_operator", "lunar_k3_ccg_ds.json", suffix)
    if d is None:
        return
    r, s = d["runs"], d["summary"]
    assert all(s["checks"].values()), s["checks"]
    a, b = r["knows_scenario__column_and_constraint_generation"], r["learns_as_outages_happen__splus_set"]
    c, e = r["learns_as_outages_happen__thm1_set"], r["design_for_knowing__operated_learning__splus_set"]
    put("FKnowsCostThree", f"{a['total_cost']:,.0f}"); put("FLearnsCostThree", f"{b['total_cost']:,.0f}")
    put("FLearnsMinusKnowsThree", f"{s['multi_stage_minus_two_stage']:,.0f}")
    put("FLearnsMinusKnowsPctThree", f"{100 * s['multi_stage_minus_two_stage'] / a['total_cost']:.3f}")
    put("FLearnsOnEndMinusLearnsThree", f"{round(-s['thm1_set_understates_by'], 2) + 0.0:,.2f}")
    put("FKnowingDesignExtraThree", f"{s['two_stage_design_operated_multi_stage_extra_cost']:,.0f}")
    put("FLearningDesignExtraThree", f"{s['multi_stage_design_operated_two_stage_extra_cost']:,.0f}")
    put("FLearnsScenariosThree", f"{b['scenarios']:,}"); put("FLearnsNodesThree", f"{b['nodes']:,}")
    put("FLearnsOnEndNodesThree", f"{c['nodes']:,}")
    put("FLearnsVarsMillionsThree", f"{b['variables'] / 1e6:.1f}")
    put("FLearnsMinThree", f"{b['solve_s'] / 60:.0f}"); put("FLearnsOnEndMinThree", f"{c['solve_s'] / 60:.0f}")
    put("FLearnsSThree", secs(b["solve_s"])); put("FLearnsOnEndSThree", secs(c["solve_s"]))
    put("FKnowsSThree", secs(a["wall_s"])); put("FKnowsRoundsThree", a["rounds"])
    put("FLearnsCostMThree", f"{b['total_cost'] / 1e6:.3f}"); put("FKnowsCostMThree", f"{a['total_cost'] / 1e6:.3f}")
    put("FLearnsOnEndCostMThree", f"{c['total_cost'] / 1e6:.3f}")
    put("FLearnsOnEndScenariosThree", f"{c['scenarios']:,}")
    put("FLearnsOnEndVarsMillionsThree", f"{c['variables'] / 1e6:.1f}")
    put("FKnowsQThree", ", ".join(f"{v:.2f}" for v in a["q"])); put("FLearnsQThree", ", ".join(f"{v:.2f}" for v in b["q"]))
    per_node = (b["variables"] - 6) // b["nodes"]
    assert per_node * b["nodes"] + 6 == b["variables"]
    put("FVarsPerNode", per_node)


def tree_sizes(suffix):
    """Scenario-tree nodes (run_ol_final.py --stage treesize), checked against the
    node counts of the linear programs that were built."""
    d = load("34_scenario_counts", "tree_sizes.json", suffix)
    if d is None:
        return
    word = {1: "One", 2: "Two", 3: "Three", 4: "Four"}
    for row in d["rows"]:
        for kind, tag in (("all", "All"), ("splus", "Start"), ("thm1", "End")):
            if row[kind] is not None:
                put(f"FTree{tag}{word[row['k']]}", f"{row[kind]['tree_nodes']:,}")
    by_k = {row["k"]: row for row in d["rows"]}
    built = {"FLearnsNodes": by_k[2]["splus"], "FLearnsOnEndNodes": by_k[2]["thm1"], "FLearnsAllNodesDs": by_k[2]["all"],
             "FLearnsNodesThree": by_k[3]["splus"], "FLearnsOnEndNodesThree": by_k[3]["thm1"]}
    for name, rec in built.items():
        if name in macros:
            assert macros[name] == f"{rec['tree_nodes']:,}", (name, macros[name], rec["tree_nodes"])
    for k in (1, 2, 3):
        put(f"FTreeAllOverStart{word[k]}", f"{by_k[k]['all']['tree_nodes'] / by_k[k]['splus']['tree_nodes']:.0f}")
    if "FVarsPerNode" in macros:
        # variables of the tree linear program over every scenario with three outages, which is not built
        v = int(macros["FVarsPerNode"]) * by_k[3]["all"]["tree_nodes"] + 6
        put("FTreeAllVarsMillionsThree", f"{v / 1e6:.0f}")


def worst_floor(suffix):
    """Demand that is unmet in the worst scenario under every design (run_ol_final.py --stage worstfloor)."""
    d = load("35_milp_separation", "worst_scenario_unlimited_capacity.json", suffix)
    if d is None:
        return
    for r in d["rows"]:
        w = WORD[r["k"]]
        put(f"FWorstUnmet{w}", f"{r['unlimited_capacity']['unmet_tons']:,.0f}")
        assert abs(r["unlimited_capacity"]["unmet_tons"] - r["optimal_design"]["unmet_tons"]) < 1e-6
        put(f"FWorstOperating{w}M", f"{r['optimal_design']['operating_cost'] / 1e6:,.0f}")
        put(f"FWorstFloor{w}M", f"{r['unlimited_capacity']['operating_cost'] / 1e6:,.0f}")


def same_first(suffix):
    """Baseline that needs no proof (run_ol_final.py --stage samefirst): every outage replaced by the
    longest outage with the same first period, at most k of them."""
    d = load("42_longest_same_first_period", "lunar.json", suffix)
    if d is None:
        return
    for k, v in sorted(d["two_stage"].items()):
        a, b = v["summary"]["longest_any_start"], v["summary"]["longest_ending_inside"]
        w = WORD[int(k)]
        assert a["rounds"] == b["rounds"] and abs(a["total_cost"] - b["total_cost"]) <= 1.0
        put(f"FBaseMilp{w}S", secs(a["median_wall_s"])); put(f"FBaseMilpEnd{w}S", secs(b["median_wall_s"]))
        put(f"FBaseMilpRatio{w}", f"{a['median_wall_s'] / b['median_wall_s']:.2f}")
        if "FBaseSingle" not in macros:
            put("FBaseSingle", a["zero_one_variables"])
            put("FBaseSingleMinusEnd", a["zero_one_variables"] - b["zero_one_variables"])
    for k, v in sorted(d["multi_stage"].items()):
        a, b = v["at_most_k"], v["exactly_k"]
        w = WORD[int(k)]
        assert abs(a["total_cost"] - b["total_cost"]) <= 1e-6 * b["total_cost"]
        put(f"FBaseScenarios{w}", f"{a['scenarios']:,}"); put(f"FBaseTree{w}", f"{a['nodes']:,}")
        put(f"FBaseTree{w}S", secs(a["median_solve_s"])); put(f"FBaseTreeExact{w}S", secs(b["median_solve_s"]))
        put(f"FBaseTree{w}Min", f"{a['median_solve_s'] / 60:.0f}"); put(f"FBaseTreeExact{w}Min", f"{b['median_solve_s'] / 60:.0f}")
        put(f"FBaseTreeVarsMillions{w}", f"{a['variables'] / 1e6:.2f}" if a["variables"] < 1e6 else f"{a['variables'] / 1e6:.1f}")
        put(f"FBaseTreeExactVarsMillions{w}", f"{b['variables'] / 1e6:.2f}" if b["variables"] < 1e6 else f"{b['variables'] / 1e6:.1f}")
        put(f"FBaseTreeRuns{w}", len(a["solve_s_all_runs"]))
        if f"FBaseCount{w}" in macros:                      # the counts stage and this stage agree
            assert macros[f"FBaseCount{w}"] == f"{a['scenarios']:,}", (k, macros[f"FBaseCount{w}"], a["scenarios"])
        if f"FTreeStart{w}" in macros:
            assert macros[f"FTreeStart{w}"] == f"{b['nodes']:,}", (k, b["nodes"])
        if f"FTreeAll{w}" in macros:
            put(f"FTreeAllOverBase{w}", f"{int(macros[f'FTreeAll{w}'].replace(',', '')) / a['nodes']:.0f}")
        put(f"FBaseMinusStartScenarios{w}", f"{a['scenarios'] - b['scenarios']:,}")
        put(f"FBaseMinusStartNodes{w}", f"{a['nodes'] - b['nodes']:,}")


def fam_same_first(suffix):
    """The baseline on the random examples (run_ol_final.py --stage famsamefirst)."""
    d = load("42_longest_same_first_period", "family.json", suffix)
    if d is None:
        return
    rows = d["rows"]
    selected_seeds(suffix, [r["seed"] for r in rows], "42_longest_same_first_period")
    if len(rows) < 30:
        notes.append(f"unfinished: results/42_longest_same_first_period{suffix}/family.json has {len(rows)} rows")
        return
    fam = load("37_random_examples", "family.json", suffix)
    if fam is not None:                 # the same design and rounds as with every outage (other session)
        other = {r["seed"]: r["summary"]["all_single_outages"] for r in fam["rows"]}
        for r in rows:
            o, x = other[r["seed"]], r["two_stage"]["longest_any_start"]
            assert o["rounds"] == x["rounds"] and abs(o["total_cost"] - x["total_cost"]) <= 1.0, r["seed"]
    a = [r["two_stage"]["longest_any_start"]["median_wall_s"] for r in rows]
    b = [r["two_stage"]["longest_ending_inside"]["median_wall_s"] for r in rows]
    ratio = [x / y for x, y in zip(a, b)]
    assert all(r["two_stage"]["longest_any_start"]["zero_one_variables"] == 72 for r in rows)
    put("FFamBaseSMin", secs(min(a))); put("FFamBaseSMed", secs(st.median(a))); put("FFamBaseSMax", secs(max(a)))
    put("FFamBaseEndSMin", secs(min(b))); put("FFamBaseEndSMed", secs(st.median(b))); put("FFamBaseEndSMax", secs(max(b)))
    put("FFamBaseRatioMin", f"{min(ratio):.2f}"); put("FFamBaseRatioMed", f"{st.median(ratio):.2f}")
    put("FFamBaseRatioMax", f"{max(ratio):.2f}")
    put("FFamBaseSlowerWithProp", sum(1 for x in ratio if x < 1.0))
    ta = [r["multi_stage"]["at_most_k"]["solve_s"] for r in rows]
    tb = [r["multi_stage"]["exactly_k"]["solve_s"] for r in rows]
    put("FFamTreeBaseSMin", secs(min(ta))); put("FFamTreeBaseSMed", secs(st.median(ta))); put("FFamTreeBaseSMax", secs(max(ta)))
    put("FFamTreeStartSMin", secs(min(tb))); put("FFamTreeStartSMed", secs(st.median(tb))); put("FFamTreeStartSMax", secs(max(tb)))
    put("FFamTreeStartFaster", sum(1 for x, y in zip(ta, tb) if y < x))
    fallback = [r for r in rows if r.get("multi_stage_failed_attempts")]
    for r in rows:                       # the two trees of an instance always come from the same method
        assert r["multi_stage"]["at_most_k"]["method"] == r["multi_stage"]["exactly_k"]["method"], r["seed"]
    put("FFamTreeFallbackN", len(fallback))
    put("FFamTreeFallbackSentence", "" if not fallback else
        (f" In {len(fallback)} instance{'s' if len(fallback) > 1 else ''} the dual simplex method failed on one tree, "
         "and both trees were solved with the interior-point method."))
    na = [r["multi_stage"]["at_most_k"]["nodes"] for r in rows]
    nb = [r["multi_stage"]["exactly_k"]["nodes"] for r in rows]
    put("FFamTreeBaseNodesMin", f"{min(na):,}"); put("FFamTreeBaseNodesMax", f"{max(na):,}")
    put("FFamTreeStartNodesMin", f"{min(nb):,}"); put("FFamTreeStartNodesMax", f"{max(nb):,}")


def variants_regret(suffix):
    """The design found with the scenarios of Proposition 1, operated over those of Theorem 1
    (run_ol_final.py --stage variantsregret)."""
    v = load("40_late_demand_variants", "variants.json", suffix)
    if v is not None:
        short = [r for r in v["rows"] if r["thm1_set_understates_by"] > 1.0]
        if short:
            put("FVarShortMaxPctOne", f"{max(r['thm1_set_understates_by_pct'] for r in short):.1f}")
            put("FVarShortMin", f"{min(r['thm1_set_understates_by'] for r in short):,.0f}")
    d = load("40_late_demand_variants", "regret.json", suffix)
    if d is None:
        return
    rows = d["rows"]
    put("FVarRegretMin", f"{min(r['extra_cost_of_that_design'] for r in rows):,.0f}")
    put("FVarRegretMax", f"{max(r['extra_cost_of_that_design'] for r in rows):,.0f}")
    put("FVarRegretMaxPct", f"{max(r['extra_cost_pct'] for r in rows):.1f}")
    put("FVarRegretMaxPctTwo", f"{max(r['extra_cost_pct'] for r in rows):.2f}")


def trees_one_session(suffix):
    """Table 4: the three tree linear programs in one session (run_ol_final.py --stage trees)."""
    for directory, k in (("43_trees_one_session", 2), ("43_trees_one_session_k3", 3)):
        d = load(directory, "lunar.json", suffix)
        if d is None:
            continue
        rec = d["runs"][str(k)]
        w = WORD[k]
        for name, tag in (("at_most_k_longest", "Long"), ("theorem_1", "Beg"), ("proposition_1", "End")):
            t = rec["trees"][name]
            it = set(t["iterations_all_runs"])
            assert len(it) == 1, (k, name, it)          # the same iterations in every run
            put(f"FTree{tag}{w}Iter", f"{t['iterations_all_runs'][0]:,}")
            put(f"FTree{tag}{w}S", secs(t["median_solve_s"]))
            put(f"FTree{tag}{w}Min", f"{t['median_solve_s'] / 60:.0f}")
            put(f"FTree{tag}{w}Runs", len(t["solve_s_all_runs"]))
            put(f"FTree{tag}{w}CostM", f"{t['total_cost'] / 1e6:.3f}")
            put(f"FTree{tag}{w}Method", t["method"])
            if f"FTreeAll{w}" in macros and tag == "Long":
                assert macros[f"FBaseCount{w}"] == f"{t['scenarios']:,}"
            nodes_macro = {"Long": f"FBaseTree{w}", "Beg": f"FTreeStart{w}", "End": f"FTreeEnd{w}"}[tag]
            if nodes_macro in macros:
                assert macros[nodes_macro] == f"{t['nodes']:,}", (k, name, t["nodes"])
        a, b = rec["trees"]["at_most_k_longest"], rec["trees"]["theorem_1"]
        put(f"FTreeBegOverLongIter{w}", f"{b['iterations_all_runs'][0] / a['iterations_all_runs'][0]:.2f}")
        put(f"FTreeFailed{w}", len(rec["failed_attempts"]))


def fam_trees(suffix):
    """Table 5, multi-stage rows: the three trees on the random examples (stage famtrees)."""
    d = load("43_trees_one_session", "family.json", suffix)
    if d is None:
        return
    rows = d["rows"]
    selected_seeds(suffix, [r["seed"] for r in rows], "43_trees_one_session")
    if len(rows) < 30:
        notes.append(f"unfinished: results/43_trees_one_session{suffix}/family.json has {len(rows)} rows")
        return
    # Times and simplex iterations only over the instances whose three trees were all solved
    # with the dual simplex method: interior-point iterations (and crossover) are not simplex
    # iterations, so an instance that fell back to highs-ipm is reported in the note instead.
    names3 = ("at_most_k_longest", "theorem_1", "proposition_1")
    ds = [r for r in rows if all(r["trees"][n]["method"] == "highs-ds" for n in names3)]
    fallback = [r for r in rows if r not in ds]
    assert all(r["failed_attempts"] for r in fallback)
    put("FFamTreeDsN", len(ds))
    for name, tag in (("at_most_k_longest", "Long"), ("theorem_1", "Beg"), ("proposition_1", "End")):
        x = [r["trees"][name]["median_solve_s"] for r in ds]
        put(f"FFamTree{tag}SMin", secs(min(x))); put(f"FFamTree{tag}SMed", secs(st.median(x))); put(f"FFamTree{tag}SMax", secs(max(x)))
    ratio = [r["trees"]["theorem_1"]["iterations_all_runs"][0] / r["trees"]["at_most_k_longest"]["iterations_all_runs"][0]
             for r in ds]
    put("FFamTreeIterRatioMin", f"{min(ratio):.2f}"); put("FFamTreeIterRatioMed", f"{st.median(ratio):.2f}")
    put("FFamTreeIterRatioMax", f"{max(ratio):.2f}")
    put("FFamTreeFallbackN", len(fallback))
    if not fallback:
        put("FFamTreeFallbackSentence", "")
    elif len(fallback) == 1:
        t = fallback[0]["trees"]
        set_macro = {"at_most_k_longest": "$\\Ulong$", "theorem_1": "$\\Ubeg$", "proposition_1": "$\\Uend$"}
        failed = sorted({set_macro[a["tree"]] for a in fallback[0]["failed_attempts"]})
        s = ", ".join(secs(t[n]["median_solve_s"]) for n in names3[:2]) + " and " + secs(t[names3[2]]["median_solve_s"])
        put("FFamTreeFallbackSentence",
            f" In one instance the dual simplex method failed on the tree of {' and '.join(failed)}; the three trees "
            f"of that instance were solved with the interior-point method, in {s} seconds for $\\Ulong$, "
            "$\\Ubeg$ and $\\Uend$, and are left out of the multi-stage rows above.")
    else:
        put("FFamTreeFallbackSentence",
            f" In {len(fallback)} instances the dual simplex method failed on a tree; their trees were "
            "solved with the interior-point method and are left out of the rows above.")
    short = [r for r in rows if r["trees"]["theorem_1"]["total_cost"] - r["trees"]["proposition_1"]["total_cost"] > 1.0]
    put("FFamTreeEndShort", len(short))
    lj = load("39_learning_operator_random", "family.json", suffix)
    if lj is not None:                   # the same multi-stage optimum as in the earlier session
        other = {r["seed"]: r["learns_total_cost"] for r in lj["rows"]}
        for r in rows:
            assert abs(other[r["seed"]] - r["trees"]["theorem_1"]["total_cost"]) <= 1e-6 * other[r["seed"]], r["seed"]


def three_columns(suffix):
    """Tables 3 and 5, two-stage rows: all outages, the longest ones and those of Proposition 1
    in one session (stages milpsep and family with --out-suffix _three_columns)."""
    names = {"all_single_outages": "All", "longest_any_start": "Long", "longest_ending_inside": "End"}
    for k in (2, 3):
        d = load("35_milp_separation_three_columns", f"milp_separation_k{k}.json", suffix)
        if d is None:
            continue
        w = WORD[k]
        summ = d["summary"]
        assert len({summ[n]["rounds"] for n in names}) == 1
        assert max(summ[n]["total_cost"] for n in names) - min(summ[n]["total_cost"] for n in names) <= 1.0
        for n, tag in names.items():
            put(f"FCcg{tag}{w}S", secs(summ[n]["median_wall_s"]))
            put(f"FCcg{tag}Vars", summ[n]["zero_one_variables"]) if f"FCcg{tag}Vars" not in macros else None
        put(f"FCcgRounds{w}", summ["longest_ending_inside"]["rounds"])
        put(f"FCcgCost{w}M", f"{summ['longest_ending_inside']['total_cost'] / 1e6:.3f}")
        put(f"FCcgLongOverEnd{w}", f"{summ['longest_any_start']['median_wall_s'] / summ['longest_ending_inside']['median_wall_s']:.2f}")
        put(f"FCcgAllOverLong{w}", f"{summ['all_single_outages']['median_wall_s'] / summ['longest_any_start']['median_wall_s']:.0f}")
        spread = max(max(summ[n]["wall_s_all_runs"]) / min(summ[n]["wall_s_all_runs"]) - 1 for n in names)
        put(f"FCcgSpreadPct{w}", f"{100 * spread:.1f}")
    d = load("37_random_examples_three_columns", "family.json", suffix)
    if d is None:
        return
    rows = d["rows"]
    selected_seeds(suffix, [r["seed"] for r in rows], "37_random_examples_three_columns")
    if len(rows) < 30:
        notes.append(f"unfinished: results/37_random_examples_three_columns{suffix}/family.json has {len(rows)} rows")
        return
    assert all(r["same_cost"] and r["same_rounds"] for r in rows)
    for n, tag in names.items():
        x = [r["summary"][n]["median_wall_s"] for r in rows]
        put(f"FFamThree{tag}SMin", secs(min(x))); put(f"FFamThree{tag}SMed", secs(st.median(x))); put(f"FFamThree{tag}SMax", secs(max(x)))
        v = [r["summary"][n]["zero_one_variables"] for r in rows]
        put(f"FFamThree{tag}VarsMin", min(v)); put(f"FFamThree{tag}VarsMax", max(v))
    ratio = [r["summary"]["longest_any_start"]["median_wall_s"] / r["summary"]["longest_ending_inside"]["median_wall_s"] for r in rows]
    put("FFamThreeLongOverEndMin", f"{min(ratio):.2f}"); put("FFamThreeLongOverEndMed", f"{st.median(ratio):.2f}")
    put("FFamThreeLongOverEndMax", f"{max(ratio):.2f}")
    ratio2 = [r["summary"]["all_single_outages"]["median_wall_s"] / r["summary"]["longest_any_start"]["median_wall_s"] for r in rows]
    put("FFamThreeAllOverLongMin", f"{min(ratio2):.1f}"); put("FFamThreeAllOverLongMed", f"{st.median(ratio2):.1f}")
    put("FFamThreeAllOverLongMax", f"{max(ratio2):.1f}")
    rounds = [r["summary"]["longest_ending_inside"]["rounds"] for r in rows]
    put("FFamThreeRoundsMin", min(rounds)); put("FFamThreeRoundsMax", max(rounds))
    for n, tag in names.items():
        lo, hi = macros[f"FFamThree{tag}VarsMin"], macros[f"FFamThree{tag}VarsMax"]
        put(f"FFamThree{tag}VarsRange", lo if lo == hi else f"{lo}--{hi}")



def variants_table(suffix):
    """Rows of the table of late-demand variants in which the set of Proposition 1 understates the
    multi-stage cost, as the macro FVariantRows (with the extra cost of the design, regret.json)."""
    d = load("40_late_demand_variants", "variants.json", suffix)
    g = load("40_late_demand_variants", "regret.json", suffix)
    if d is None or g is None:
        return
    extra = {r["variant"]: r["extra_cost_of_that_design"] for r in g["rows"]}
    short = [r for r in d["rows"] if r["thm1_set_understates_by"] > 1.0]
    assert {r["variant"] for r in short} == set(extra)
    short.sort(key=lambda r: (r["first_demand_month"], r["demand_factor"], r["holding_factor"]))
    lines = []
    for r in short:
        change = max(100 * abs(x - y) / y for x, y in zip(r["q_multi_stage_on_thm1_set"], r["q_multi_stage"]))
        lines.append(f"{r['first_demand_month']} & {r['demand_factor']:g} & {r['holding_factor']:,.0f} & "
                     f"{r['multi_stage_total_cost'] / 1e6:,.1f} & {r['thm1_set_understates_by']:,.0f} & "
                     f"{extra[r['variant']]:,.0f} & {change:.0f}" + r" \\")
    put("FVariantRows", "\n".join(lines))


def variants(suffix):
    """Late demand: the multi-stage problem over the set of Theorem 2 and over that of Theorem 1."""
    d = load("40_late_demand_variants", "variants.json", suffix)
    if d is None:
        return
    rows = d["rows"]
    short = [r for r in rows if r["thm1_set_understates_by"] > 1.0]
    assert all(r["thm1_set_understates_by"] > -1.0 for r in rows)
    put("FVarN", len(rows)); put("FVarShort", len(short))
    put("FVarMultiMinusTwoMaxPct", f"{max(100 * r['multi_minus_two_stage'] / r['two_stage_total_cost'] for r in rows):.2f}")
    if short:
        worst = max(short, key=lambda r: r["thm1_set_understates_by"])
        put("FVarShortMax", f"{worst['thm1_set_understates_by']:,.0f}")
        put("FVarShortMaxPct", f"{max(r['thm1_set_understates_by_pct'] for r in short):.2f}")
        rel = [100 * abs(x - y) / y for r in short for x, y in zip(r["q_multi_stage_on_thm1_set"], r["q_multi_stage"])]
        put("FVarCapacityOffMaxPct", f"{max(rel):.0f}")
        put("FVarShortEarliestMonth", min(r["first_demand_month"] for r in short))
        put("FVarShortLeastHolding", f"{min(r['holding_factor'] for r in short):.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="", help="results directory suffix, e.g. _laptop for the provisional runs")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    for part in (counts, milp, worst_floor, verify, family, learns, learns_three, tree_sizes, learns_family, variants,
                 same_first, variants_regret, variants_table, trees_one_session, fam_trees,
                 three_columns):
        part(args.suffix)
    lines = ["% Generated by checks/make_numbers_final.py from results/34..42" + (f" (suffix {args.suffix})" if args.suffix else "")
             + " -- do not edit by hand."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    if not args.no_write:
        open(OUT, "w").write("\n".join(lines) + "\n")
    for k, v in macros.items():
        print(f"{k:28s} {v}")
    for n in notes:
        print("NOTE", n)
