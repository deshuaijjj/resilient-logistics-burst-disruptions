# Resilient Logistics under Burst Disruptions

This repository contains the code, input data, generated result tables, and
figures for a reproducible study of robust inventory--transportation co-design
for resilient logistics networks under sustained burst disruptions.

The computational case study uses a stylized cislunar logistics network as a
high-stress example: long lead times, limited emergency replenishment, sparse
depot infrastructure, and constrained destination reception capacity. The model
and scripts are written as a general time-expanded logistics design workflow.

## Version 2: outage scenarios that suffice

Version 2.0.0 adds the code, checks and recorded results of the manuscript
*Outage scenarios that suffice for two-stage and multi-stage robust capacity
design* (Y. Zhang). The manuscript shows which outage scenarios suffice to give
every capacity design its worst-case cost: for an operator who knows the
scenario before operating (two-stage), the scenarios with exactly k outages of
maximum duration that all end within the planning horizon; for an operator who
sees in each period only which lines are out (multi-stage), also those in
which an outage runs past the horizon.

- `src/ol_scenarios.py`: writes down and counts the scenario sets (all
  scenarios, at most k maximum-duration outages, the two-stage set, the
  multi-stage set).
- `src/run_ol_final.py`: every experiment of the manuscript, by stage
  (`counts`, `milpsep`, `verify`, `trees`, `variants`, `variantsregret`,
  `family`, `famtrees`, ...); the `run_ol_final_*.sh` scripts are the batches
  that were run. `src/run_ol_revision.py` and `src/run_ol_nonanticipative.py`
  hold the example, column-and-constraint generation and the scenario-tree
  linear program.
- `results/34_scenario_counts` to `results/43_trees_one_session_k3`: the
  recorded results behind every number of the manuscript (directories ending
  in `_laptop` are provisional runs on a second machine; `results/24` to
  `results/33` are earlier stages of the same study).
- `checks/optimization_letters/`: brute-force checks of the statements
  (`check_theorem_k.py` for the two-stage set and its count,
  `check_lemma2_construction.py` for the covering lemma,
  `check_nonanticipative_theorem.py` for the multi-stage set),
  `make_numbers_final.py`, which regenerates every number of the manuscript
  from `results/` into `numbers_final.tex`, and `make_figure1.py`.

In the result files the two-stage set is called `thm1` and the multi-stage set
`splus`; these were working names.

```bash
python checks/optimization_letters/check_theorem_k.py
python checks/optimization_letters/check_lemma2_construction.py
python checks/optimization_letters/check_nonanticipative_theorem.py cover counts values
python checks/optimization_letters/make_numbers_final.py
python src/run_ol_final.py --stage counts --out-suffix _rerun
```

The last command writes to `results/34_scenario_counts_rerun/`, so the recorded
results are not overwritten.

The sections below describe the material of version 1, which accompanied an
earlier manuscript on the same instance.

## Repository Contents

- `src/`: Python source code for scenario generation, sparse linear-programming
  model construction, baseline experiments, sensitivity analysis, and Applied
  Sciences revision evidence.
- `data/`: input JSON files for the stylized case-study instance and burst
  scenario data.
- `results/tables_for_paper/`: manuscript-facing CSV and LaTeX tables.
- `results/figures_for_paper/`: manuscript-facing PNG figures.
- `results/07_applied_sciences_evidence/`: raw outputs for random-seed
  robustness, holdout validation, deterministic larger-sample comparison, and
  stylized instance variants.

## Model Overview

The main optimization model is a two-stage robust linear program.

- First stage: choose depot inventory capacity and contracted transportation
  mode capacity.
- Second stage: after a disruption realization, route flows, manage depot
  inventory, deliver commodities to the destination, and assign penalized unmet
  demand.
- Uncertainty: sustained burst disruptions, where each event has a mode, start
  time, duration, and disruption-budget count.

The implementation builds sparse extensive-form linear programs and solves them
with SciPy's HiGHS interface.

## Environment

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproducing Key Results

The most important entry points are:

```bash
python src/generate_scenario_statistics.py
python src/scenario_sampling_stability.py
python src/run_baseline_sampled.py --sample-size 1000
python src/run_fair_baselines.py --sample-size 1000
python src/run_sensitivity_limited.py
python src/run_applied_sciences_evidence.py
python src/generate_paper_outputs.py
```

Some scripts solve large sparse linear programs and may require several
gigabytes of memory. Runtime depends on sample size and solver version.

## Key Generated Evidence

The Applied Sciences revision adds:

- random-seed robustness over 10 independent `N=500` sampled scenario sets;
- holdout validation of the `N=1000` sampled design on an `N=2000` scenario set;
- deterministic `N=1000` versus `N=2000` design comparison;
- small stylized instance variants for reception capacity, emergency-mode cost,
  and demand-spike timing.

These results are intentionally framed as sampled evidence, not as proof of
full robust optimality over the complete 29,916-scenario enumeration.

## Citation

If you use this repository, please cite the associated manuscript once it is
available. Until then, cite the repository URL and commit hash.

## License

MIT; see `LICENSE`.
