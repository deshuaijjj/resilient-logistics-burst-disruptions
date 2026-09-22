"""
Generate paper-facing figures and compact tables from experiment CSV outputs.
"""

import os
import shutil

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from PIL import Image
except ImportError:  # pragma: no cover - only affects PNG post-processing
    Image = None

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS = os.path.join(PROJECT_ROOT, 'results')
FIG_DIR = os.path.join(RESULTS, 'figures_for_paper')
TABLE_DIR = os.path.join(RESULTS, 'tables_for_paper')


def ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)


def save_figure(path: str):
    """Save publication figures with an opaque white background."""
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', transparent=False)
    plt.close()
    if Image is not None:
        with Image.open(path) as img:
            img.convert('RGB').save(path)


def save_rgb_copy(src: str, dst: str):
    """Copy an image while removing transparency when Pillow is available."""
    if Image is None:
        shutil.copy2(src, dst)
        return
    with Image.open(src) as img:
        if img.mode in ('RGBA', 'LA'):
            bg = Image.new('RGB', img.size, 'white')
            bg.paste(img, mask=img.getchannel('A'))
            bg.save(dst)
        else:
            img.convert('RGB').save(dst)


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        '\\': r'\textbackslash{}',
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    return ''.join(replacements.get(ch, ch) for ch in text)


def write_latex_table(
    df: pd.DataFrame,
    path: str,
    escape: bool = True,
    column_format: str | None = None,
):
    """Write a simple booktabs tabular without pandas' optional jinja2 dependency."""
    colspec = column_format or ('l' * len(df.columns))

    def cell(value: object) -> str:
        if pd.isna(value):
            text = ''
        else:
            text = str(value)
        return latex_escape(text) if escape else text

    with open(path, 'w') as f:
        f.write(f'\\begin{{tabular}}{{{colspec}}}\n')
        f.write('\\toprule\n')
        f.write(' & '.join(cell(col) for col in df.columns) + ' \\\\\n')
        f.write('\\midrule\n')
        for _, row in df.iterrows():
            f.write(' & '.join(cell(row[col]) for col in df.columns) + ' \\\\\n')
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')


def save_table(
    df: pd.DataFrame,
    name: str,
    latex_df: pd.DataFrame | None = None,
    escape: bool = True,
    column_format: str | None = None,
):
    csv_path = os.path.join(TABLE_DIR, f'{name}.csv')
    tex_path = os.path.join(TABLE_DIR, f'{name}.tex')
    df.to_csv(csv_path, index=False)
    out = latex_df if latex_df is not None else df
    write_latex_table(out, tex_path, escape=escape, column_format=column_format)


def clean_strategy(name: str) -> str:
    labels = {
        'nominal': 'Nominal',
        'independent': 'Independent',
        'burst': 'Burst robust',
        'joint': 'Joint',
        'burst_joint': 'Burst joint',
        'inventory_only': 'Inventory only',
        'redundancy_only': 'Redundancy only',
        'inventory_only_nominal_q': 'Inventory only\n(nominal q)',
        'inventory_only_balanced_q': 'Inventory only\n(balanced q)',
        'redundancy_only_nominal_C': 'Redundancy only\n(nominal C)',
        'budget_matched_inventory': 'Budget-matched\ninventory',
        'budget_matched_redundancy': 'Budget-matched\nredundancy',
    }
    return labels.get(name, name.replace('_', ' ').title())


def latex_strategy(name: str) -> str:
    labels = {
        'nominal': 'Nominal',
        'independent': 'Independent',
        'burst': 'Burst robust',
        'burst_joint': 'Burst joint',
        'inventory_only_nominal_q': r'\shortstack[l]{Inventory only\\(nominal $q$)}',
        'inventory_only_balanced_q': r'\shortstack[l]{Inventory only\\(balanced $q$)}',
        'redundancy_only_nominal_C': r'\shortstack[l]{Redundancy only\\(nominal $C$)}',
        'budget_matched_inventory': r'\shortstack[l]{Budget-matched\\inventory}',
        'budget_matched_redundancy': r'\shortstack[l]{Budget-matched\\redundancy}',
    }
    return labels.get(name, name.replace('_', ' ').title())


def fmt_money(x: float) -> str:
    return f'{x:,.2f}'


def fmt_capacity(value: str) -> str:
    parts = [p.strip() for p in str(value).split('/')]
    try:
        return ' / '.join(f'{float(p):,.0f}' for p in parts)
    except ValueError:
        return str(value)


def fmt_value(x: float, decimals: int = 2) -> str:
    if pd.isna(x):
        return ''
    value = float(x)
    if abs(value) < 0.5 * 10 ** (-decimals):
        value = 0.0
    return f'{value:,.{decimals}f}'


def combine_capacity(row: pd.Series, prefix: str, count: int) -> str:
    values = []
    for i in range(count):
        col = f'{prefix}_{i}'
        if col in row and not pd.isna(row[col]):
            values.append(f"{float(row[col]):,.0f}")
    return ' / '.join(values)


def paper_sampling_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df.copy()
    table = table.rename(columns={
        'sample_size': '$N$',
        'total_cost_M': r'Total cost (\$M)',
        'investment_cost_M': r'Investment (\$M)',
        'worst_operating_cost_M': r'Worst operating (\$M)',
        'C_depot_0': 'Depot 0 (tons)',
        'C_depot_1': 'Depot 1 (tons)',
        'q_mode_0': 'Mode 0 (t/mo)',
        'q_mode_1': 'Mode 1 (t/mo)',
        'q_mode_2': 'Mode 2 (t/mo)',
        'worst_scenario_id': 'Worst scenario',
        'num_worst_events': 'Worst events',
        'solve_time_s': 'Solve time (s)',
        'lp_status': 'Status',
    })
    money_cols = [r'Total cost (\$M)', r'Investment (\$M)', r'Worst operating (\$M)']
    capacity_cols = ['Depot 0 (tons)', 'Depot 1 (tons)', 'Mode 0 (t/mo)', 'Mode 1 (t/mo)', 'Mode 2 (t/mo)']
    for col in money_cols + capacity_cols + ['Solve time (s)']:
        table[col] = table[col].map(fmt_value)
    for col in ['$N$', 'Worst scenario', 'Worst events']:
        table[col] = table[col].map(lambda x: f'{int(x):,}')
    table['Status'] = table['Status'].str.capitalize()
    return table


def paper_gamma_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df.copy()
    table = table.rename(columns={
        'Gamma': r'$\Gamma$',
        'Total_Cost_M': r'Total cost (\$M)',
        'Investment_M': r'Investment (\$M)',
        'Operating_M': r'Worst operating (\$M)',
        'C_depot_total': 'Depot capacity (tons)',
        'q_mode_total': 'Mode capacity (t/mo)',
        'Num_Scenarios': 'Scenarios',
        'Solve_Time_s': 'Solve time (s)',
    })
    for col in [
        r'Total cost (\$M)', r'Investment (\$M)', r'Worst operating (\$M)',
        'Depot capacity (tons)', 'Mode capacity (t/mo)', 'Solve time (s)',
    ]:
        table[col] = table[col].map(fmt_value)
    for col in [r'$\Gamma$', 'Scenarios']:
        table[col] = table[col].map(lambda x: f'{int(x):,}')
    return table


def paper_rmax_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df.copy()
    table = table.rename(columns={
        'R_max_scale': 'Reception scale',
        'R_max_value': 'Months 1--6 cap. (t/mo)',
        'Total_Cost_M': r'Total cost (\$M)',
        'C_depot_0': 'Depot 0 (tons)',
        'C_depot_1': 'Depot 1 (tons)',
        'C_depot_total': 'Depot capacity (tons)',
        'q_mode_total': 'Mode capacity (t/mo)',
        'Solve_Time_s': 'Solve time (s)',
    })
    for col in table.columns:
        table[col] = table[col].map(fmt_value)
    return table


def paper_sinitial_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df.copy()
    table = table.rename(columns={
        's_initial_scale': 'Initial inventory scale',
        's_initial_value': 'Commodity 1 initial stock/depot (tons)',
        'Total_Cost_M': r'Total cost (\$M)',
        'C_depot_0': 'Depot 0 (tons)',
        'C_depot_1': 'Depot 1 (tons)',
        'C_depot_total': 'Depot capacity (tons)',
        'q_mode_total': 'Mode capacity (t/mo)',
        'Solve_Time_s': 'Solve time (s)',
    })
    for col in table.columns:
        table[col] = table[col].map(fmt_value)
    return table


def paper_full_certification_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df.copy()
    table['$C$ (tons)'] = table.apply(lambda r: combine_capacity(r, 'C_depot', 2), axis=1)
    table['$q$ (t/mo)'] = table.apply(lambda r: combine_capacity(r, 'q_mode', 3), axis=1)
    table = table.rename(columns={
        'design_label': 'Design',
        'optimization_total_cost_M': r'Training cost (\$M)',
        'certified_total_cost_M': r'Certified cost (\$M)',
        'cost_gap_M': r'Cert. increase (\$M)',
        'worst_scenario_id': 'Worst scen.',
        'worst_unmet_tons': 'Unmet tons',
    })
    keep = [
        'Design', r'Training cost (\$M)', r'Certified cost (\$M)',
        r'Cert. increase (\$M)', 'Worst scen.', 'Unmet tons',
        '$C$ (tons)', '$q$ (t/mo)',
    ]
    table = table[keep]
    table['Design'] = table['Design'].map(latex_strategy)
    for col in [
        r'Training cost (\$M)', r'Certified cost (\$M)',
        r'Cert. increase (\$M)', 'Unmet tons',
    ]:
        table[col] = table[col].map(fmt_value)
    for col in ['Worst scen.']:
        table[col] = table[col].map(lambda x: '' if pd.isna(x) else f'{int(x):,}')
    return table


def paper_full_worst_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df.copy()
    table = table.rename(columns={
        'design_label': 'Design',
        'scenario_id': 'Worst scenario',
        'num_events': 'Events',
        'operating_cost_M': r'Operating cost (\$M)',
        'total_cost_M': r'Total cost (\$M)',
        'unmet_tons': 'Unmet tons',
        'status': 'Status',
    })
    keep = [
        'Design', 'Worst scenario', 'Events', r'Operating cost (\$M)',
        r'Total cost (\$M)', 'Unmet tons', 'Status',
    ]
    table = table[keep]
    table['Design'] = table['Design'].map(latex_strategy)
    for col in [r'Operating cost (\$M)', r'Total cost (\$M)', 'Unmet tons']:
        table[col] = table[col].map(fmt_value)
    for col in ['Worst scenario', 'Events']:
        table[col] = table[col].map(lambda x: '' if pd.isna(x) else f'{int(x):,}')
    table['Status'] = table['Status'].map(lambda x: str(x).capitalize())
    return table


def paper_ccg_iterations_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df.copy()
    table = table.rename(columns={
        'iteration': 'Iteration',
        'master_scenarios': 'Master scenarios',
        'lower_bound_M': r'Lower bound (\$M)',
        'upper_bound_M': r'Upper bound (\$M)',
        'gap_pct': r'Opt. gap (\%)',
        'added_scenario_id': 'Added scenario',
        'master_solve_time_s': 'Master solve (s)',
        'separation_time_s': 'Separation (s)',
        'converged': 'Converged',
    })
    keep = [
        'Iteration', 'Master scenarios', r'Lower bound (\$M)',
        r'Upper bound (\$M)', r'Opt. gap (\%)', 'Added scenario',
        'Master solve (s)', 'Separation (s)', 'Converged',
    ]
    table = table[keep]
    for col in [r'Lower bound (\$M)', r'Upper bound (\$M)', r'Opt. gap (\%)', 'Master solve (s)', 'Separation (s)']:
        table[col] = table[col].map(fmt_value)
    for col in ['Iteration', 'Master scenarios', 'Added scenario']:
        table[col] = table[col].map(lambda x: '' if pd.isna(x) else f'{int(x):,}')
    table['Converged'] = table['Converged'].map(lambda x: 'Yes' if bool(x) else 'No')
    return table


def paper_ccg_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df.copy()
    table['Depot capacity'] = table.apply(lambda r: combine_capacity(r, 'C_depot', 2), axis=1)
    table['Mode capacity'] = table.apply(lambda r: combine_capacity(r, 'q_mode', 3), axis=1)
    table = table.rename(columns={
        'method': 'Method',
        'iterations': 'Iterations',
        'scenarios_used': 'Scenarios used',
        'master_objective_M': r'Master objective (\$M)',
        'certified_total_cost_M': r'Certified cost (\$M)',
        'relative_gap_pct': r'Opt. gap (\%)',
        'total_wall_time_s': 'Wall time (s)',
        'lp_status': 'Status',
        'converged': 'Converged',
    })
    keep = [
        'Method', 'Iterations', 'Scenarios used', r'Master objective (\$M)',
        r'Certified cost (\$M)', r'Opt. gap (\%)', 'Depot capacity', 'Mode capacity',
        'Wall time (s)', 'Status', 'Converged',
    ]
    table = table[keep]
    for col in [r'Master objective (\$M)', r'Certified cost (\$M)', r'Opt. gap (\%)', 'Wall time (s)']:
        table[col] = table[col].map(fmt_value)
    for col in ['Iterations', 'Scenarios used']:
        table[col] = table[col].map(lambda x: '' if pd.isna(x) else f'{int(x):,}')
    table['Method'] = table['Method'].map(lambda x: str(x).replace('_', ' ').title())
    table['Status'] = table['Status'].map(lambda x: str(x).capitalize())
    table['Converged'] = table['Converged'].map(lambda x: 'Yes' if bool(x) else 'No')
    return table


def scenario_statistics_table():
    path = os.path.join(
        RESULTS,
        '01_scenario_statistics',
        'small_T24_M3_J2_K2_Gamma2_full_enumeration.csv',
    )
    df = pd.read_csv(path)
    compact = pd.DataFrame([
        ['Total event-set scenarios', int(df.loc[0, 'total_scenarios'])],
        ['Nominal no-disruption scenario', 1],
        ['Single-event scenarios', int(df.loc[0, 'single_event_scenarios'])],
        ['Double-event scenarios', int(df.loc[0, 'double_event_scenarios'])],
        ['Total events', int(df.loc[0, 'total_events'])],
        ['Average duration', round(float(df.loc[0, 'avg_duration']), 2)],
        ['Mode 0 events', int(df.loc[0, 'mode_0_events'])],
        ['Mode 1 events', int(df.loc[0, 'mode_1_events'])],
        ['Mode 2 events', int(df.loc[0, 'mode_2_events'])],
    ], columns=['Statistic', 'Value'])
    latex = compact.copy()
    latex['Value'] = [
        f'{float(v):.2f}' if stat == 'Average duration' else f'{int(v):,}'
        for stat, v in zip(latex['Statistic'], latex['Value'])
    ]
    save_table(compact, 'scenario_statistics', latex_df=latex, column_format='lr')


def sampling_stability_figure():
    path = os.path.join(
        RESULTS,
        '04_sampling_stability',
        'small_T24_M3_J2_K2_Gamma2',
        'scenario_sample_stability_N100_N500_N1000.csv',
    )
    df = pd.read_csv(path)
    save_table(df, 'sampling_stability', latex_df=paper_sampling_table(df), escape=False)

    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    ax.plot(df['sample_size'], df['total_cost_M'], marker='o', linewidth=2)
    ax.set_xlabel('Number of sampled burst scenarios')
    ax.set_ylabel('Total cost ($M; zoomed scale)')
    ax.grid(alpha=0.3)
    ax.set_xticks(df['sample_size'])
    save_figure(os.path.join(FIG_DIR, 'sampling_stability_cost.png'))


def baseline_tables_and_figures():
    n1000_path = os.path.join(
        RESULTS,
        '02_baselines',
        'small_T24_M3_J2_K2_Gamma2_N1000_sampled',
        'summary.csv',
    )
    fair_path = os.path.join(
        RESULTS,
        '06_fair_baselines',
        'small_T24_M3_J2_K2_Gamma2_N1000_sampled',
        'summary.csv',
    )
    base = pd.read_csv(n1000_path)
    fair = pd.read_csv(fair_path)

    keep = [
        'Strategy', 'Total Cost ($M)', 'Investment ($M)', 'Worst Operating ($M)',
        'Depot Capacity (tons)', 'Mode Capacity (t/mo)',
    ]
    base_csv = base[base['Strategy'].isin(['nominal', 'independent', 'burst'])][keep].copy()
    fair_csv = fair[keep].copy()
    base_main = base_csv.copy()
    fair_main = fair_csv.copy()

    for table in [base_main, fair_main]:
        table['Strategy'] = table['Strategy'].map(latex_strategy)
        table['Total Cost ($M)'] = table['Total Cost ($M)'].map(fmt_money)
        table['Investment ($M)'] = table['Investment ($M)'].map(fmt_money)
        table['Worst Operating ($M)'] = table['Worst Operating ($M)'].map(fmt_money)
        table['Depot Capacity (tons)'] = table['Depot Capacity (tons)'].map(fmt_capacity)
        table['Mode Capacity (t/mo)'] = table['Mode Capacity (t/mo)'].map(fmt_capacity)

    base_main = base_main.rename(columns={
        'Total Cost ($M)': r'Total cost (\$M)',
        'Investment ($M)': r'Investment (\$M)',
        'Worst Operating ($M)': r'Worst operating (\$M)',
        'Depot Capacity (tons)': 'Depot capacity (tons)',
        'Mode Capacity (t/mo)': 'Mode capacity (t/mo)',
    })
    fair_main = fair_main.rename(columns={
        'Total Cost ($M)': r'Total cost (\$M)',
        'Investment ($M)': r'Investment (\$M)',
        'Worst Operating ($M)': r'Worst operating (\$M)',
        'Depot Capacity (tons)': 'Depot capacity (tons)',
        'Mode Capacity (t/mo)': 'Mode capacity (t/mo)',
    })

    save_table(base_csv, 'baseline_N1000', latex_df=base_main, escape=False)
    save_table(fair_csv, 'fair_baseline_N1000', latex_df=fair_main, escape=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    strategies = [clean_strategy(s).replace('\n', ' ') for s in fair['Strategy']]
    y = np.arange(len(strategies))
    values = fair['Total Cost ($M)']
    ax.barh(y, values, color='steelblue', edgecolor='black')
    ax.set_xscale('log')
    ax.set_xlim(values.min() * 0.75, values.max() * 2.3)
    ax.set_xlabel('Total cost ($M, log scale)')
    ax.set_yticks(y)
    ax.set_yticklabels(strategies, fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    for yi, value in zip(y, values):
        ax.text(value * 1.08, yi, f'{value:,.0f}', va='center', fontsize=7, clip_on=False)
    fig.subplots_adjust(left=0.35, right=0.88)
    save_figure(os.path.join(FIG_DIR, 'fair_baseline_costs_log.png'))

    focused = fair[fair['Strategy'].isin([
        'nominal',
        'independent',
        'burst_joint',
        'inventory_only_balanced_q',
        'redundancy_only_nominal_C',
        'budget_matched_redundancy',
    ])]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = np.arange(len(focused))
    ax.bar(x, focused['Total Cost ($M)'], color='seagreen', edgecolor='black')
    ax.set_ylabel('Total cost ($M)')
    ax.set_xticks(x)
    ax.set_xticklabels([clean_strategy(s) for s in focused['Strategy']], fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    save_figure(os.path.join(FIG_DIR, 'fair_baseline_costs_focused.png'))


def sensitivity_figures():
    sens_dir = os.path.join(
        RESULTS,
        '05_sensitivity',
        'small_T24_M3_J2_K2_Gamma2_N400_sampled',
    )

    gamma = pd.read_csv(os.path.join(sens_dir, 'sensitivity_gamma.csv'))
    rmax = pd.read_csv(os.path.join(sens_dir, 'sensitivity_r_max.csv'))
    sinit = pd.read_csv(os.path.join(sens_dir, 'sensitivity_s_initial.csv'))
    save_table(gamma, 'sensitivity_gamma', latex_df=paper_gamma_table(gamma), escape=False)
    save_table(rmax, 'sensitivity_r_max', latex_df=paper_rmax_table(rmax), escape=False)
    save_table(sinit, 'sensitivity_s_initial', latex_df=paper_sinitial_table(sinit), escape=False)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(gamma['Gamma'], gamma['Total_Cost_M'], marker='o', linewidth=2)
    ax.set_xlabel(r'Disruption budget $\Gamma$')
    ax.set_ylabel('Total cost ($M)')
    ax.grid(alpha=0.3)
    ax.set_xticks(gamma['Gamma'])
    save_figure(os.path.join(FIG_DIR, 'gamma_total_cost.png'))

    fig, (ax_cost, ax_cap) = plt.subplots(
        2, 1, figsize=(6.5, 5.0), sharex=True, constrained_layout=True
    )
    ax_cost.plot(rmax['R_max_scale'], rmax['Total_Cost_M'], marker='o', color='steelblue')
    ax_cost.set_ylabel('Total cost ($M)')
    ax_cost.grid(alpha=0.3)
    ax_cap.plot(rmax['R_max_scale'], rmax['C_depot_total'], marker='s', color='darkred')
    ax_cap.set_xlabel('Reception-capacity scale')
    ax_cap.set_ylabel('Depot capacity (tons)')
    ax_cap.grid(alpha=0.3)
    save_figure(os.path.join(FIG_DIR, 'rmax_bottleneck_cost_capacity.png'))

    fig, (ax_cost, ax_cap) = plt.subplots(
        2, 1, figsize=(6.5, 5.0), sharex=True, constrained_layout=True
    )
    ax_cost.plot(sinit['s_initial_scale'], sinit['Total_Cost_M'], marker='o', color='steelblue')
    ax_cost.set_ylabel('Total cost ($M)')
    ax_cost.grid(alpha=0.3)
    ax_cap.plot(sinit['s_initial_scale'], sinit['C_depot_total'], marker='s', color='darkred')
    ax_cap.set_xlabel('Initial inventory scale')
    ax_cap.set_ylabel('Depot capacity (tons)')
    ax_cap.grid(alpha=0.3)
    save_figure(os.path.join(FIG_DIR, 'sinitial_threshold_cost_capacity.png'))


def copy_existing_sensitivity_pngs():
    sens_dir = os.path.join(
        RESULTS,
        '05_sensitivity',
        'small_T24_M3_J2_K2_Gamma2_N400_sampled',
    )
    for name in ['sensitivity_gamma.png', 'sensitivity_r_max.png', 'sensitivity_s_initial.png']:
        src = os.path.join(sens_dir, name)
        if os.path.exists(src):
            save_rgb_copy(src, os.path.join(FIG_DIR, name))


def full_certification_outputs():
    cert_dir = os.path.join(RESULTS, '08_full_burst_certification')
    summary_path = os.path.join(cert_dir, 'full_certification_summary.csv')
    worst_path = os.path.join(cert_dir, 'full_certification_worst_scenarios.csv')
    if not os.path.exists(summary_path):
        print(f'Skipping full certification outputs; missing {summary_path}')
        return

    summary = pd.read_csv(summary_path)
    # The baseline and fair-baseline result files both contain the same sampled
    # burst joint design under different labels. Keep one paper-facing row.
    summary = summary[summary['design_label'] != 'burst_joint'].copy()
    save_table(
        summary,
        'full_certification_summary',
        latex_df=paper_full_certification_table(summary),
        escape=False,
    )
    if os.path.exists(worst_path):
        worst = pd.read_csv(worst_path)
        worst = worst[worst['design_label'] != 'burst_joint'].copy()
        save_table(
            worst,
            'full_certification_worst_scenarios',
            latex_df=paper_full_worst_table(worst),
            escape=False,
        )

    plot_df = summary[pd.notna(summary['certified_total_cost_M'])].copy()
    if not plot_df.empty:
        labels = [clean_strategy(s).replace('\n', ' ') for s in plot_df['design_label']]
        x = np.arange(len(plot_df))
        width = 0.38
        fig, ax = plt.subplots(figsize=(7.8, 4.2))
        if 'optimization_total_cost_M' in plot_df:
            ax.bar(
                x - width / 2,
                plot_df['optimization_total_cost_M'],
                width,
                label='Design/training objective',
                color='steelblue',
                edgecolor='black',
            )
        ax.bar(
            x + width / 2,
            plot_df['certified_total_cost_M'],
            width,
            label='Full-library certified cost',
            color='darkorange',
            edgecolor='black',
        )
        ax.set_ylabel('Total cost ($M)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        save_figure(os.path.join(FIG_DIR, 'full_certification_cost_gap.png'))

        fig, (ax_c, ax_q) = plt.subplots(2, 1, figsize=(7.6, 5.2), sharex=True)
        c_totals = plot_df[[c for c in ['C_depot_0', 'C_depot_1'] if c in plot_df]].sum(axis=1)
        q_totals = plot_df[[c for c in ['q_mode_0', 'q_mode_1', 'q_mode_2'] if c in plot_df]].sum(axis=1)
        ax_c.bar(x, c_totals, color='seagreen', edgecolor='black')
        ax_c.set_ylabel('Depot capacity')
        ax_c.grid(axis='y', alpha=0.3)
        ax_q.bar(x, q_totals, color='slateblue', edgecolor='black')
        ax_q.set_ylabel('Mode capacity')
        ax_q.set_xticks(x)
        ax_q.set_xticklabels(labels, rotation=25, ha='right', fontsize=8)
        ax_q.grid(axis='y', alpha=0.3)
        save_figure(os.path.join(FIG_DIR, 'full_certification_capacity_comparison.png'))


def ccg_outputs():
    ccg_dir = os.path.join(RESULTS, '09_burst_ccg')
    iterations_path = os.path.join(ccg_dir, 'ccg_iterations.csv')
    summary_path = os.path.join(ccg_dir, 'ccg_summary.csv')
    if not os.path.exists(iterations_path):
        print(f'Skipping CCG outputs; missing {iterations_path}')
        return

    iterations = pd.read_csv(iterations_path)
    save_table(
        iterations,
        'ccg_iterations',
        latex_df=paper_ccg_iterations_table(iterations),
        escape=False,
    )
    if os.path.exists(summary_path):
        summary = pd.read_csv(summary_path)
        save_table(
            summary,
            'ccg_summary',
            latex_df=paper_ccg_summary_table(summary),
            escape=False,
        )

    fig, (ax_bounds, ax_gap) = plt.subplots(2, 1, figsize=(6.8, 5.0), sharex=True)
    ax_bounds.plot(iterations['iteration'], iterations['lower_bound_M'], marker='o', label='Lower bound')
    ax_bounds.plot(iterations['iteration'], iterations['upper_bound_M'], marker='s', label='Upper bound')
    ax_bounds.set_ylabel('Cost ($M)')
    ax_bounds.legend(fontsize=8)
    ax_bounds.grid(alpha=0.3)
    ax_gap.plot(iterations['iteration'], iterations['gap_pct'], marker='o', color='darkred')
    ax_gap.set_xlabel('CCG iteration')
    ax_gap.set_ylabel('Gap (%)')
    ax_gap.grid(alpha=0.3)
    ax_gap.set_xticks(iterations['iteration'])
    save_figure(os.path.join(FIG_DIR, 'ccg_convergence.png'))


def main():
    ensure_dirs()
    scenario_statistics_table()
    sampling_stability_figure()
    baseline_tables_and_figures()
    sensitivity_figures()
    copy_existing_sensitivity_pngs()
    full_certification_outputs()
    ccg_outputs()
    print(f'Wrote figures to {FIG_DIR}')
    print(f'Wrote tables to {TABLE_DIR}')


if __name__ == '__main__':
    main()
