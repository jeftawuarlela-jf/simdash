"""
Supply Chain Simulation Engine — Plotly charts version
Identical simulation logic to simulation.py, but generates interactive
Plotly charts (.html + .json) instead of matplotlib PNGs.

This script is invoked as a subprocess by dash_app.py.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import json
import random
import os
from itertools import product

# ========================================
# CONFIGURATION
# ========================================
try:
    from config import *
    print("[OK] Configuration loaded from config.py")
except ImportError:
    print("[WARN] config.py not found, using default values")
    REORDER_THRESHOLD_RANGE = range(20, 21)
    TARGET_DOI_RANGE = range(35, 36)
    DAILY_SKU_CAPACITY = 360
    TOTAL_SKU_CAPACITY = 5100
    START_DATE = (2025, 7, 1)
    END_DATE = (2025, 12, 31)
    DATA_FILE = 'fulllead.csv'
    OUTPUT_DIR = 'simulation_results'
    SAVE_DETAILED_RESULTS = True
    SAVE_DAILY_SUMMARIES = True

if isinstance(START_DATE, tuple):
    START_DATE = datetime(*START_DATE)
if isinstance(END_DATE, tuple):
    END_DATE = datetime(*END_DATE)

# ========================================
# HELPER FUNCTIONS
# ========================================
def add_working_days(start_date, working_days):
    """Add working days to a date, skipping Sundays (weekday 6)."""
    current_date = start_date
    days_added = 0
    while days_added < working_days:
        current_date += timedelta(days=1)
        if current_date.weekday() < 6:
            days_added += 1
    return current_date


def run_single_simulation(sku_info, reorder_threshold, target_doi, date_range):
    """Run a single simulation with given parameters."""
    results = []

    for idx, sku_row in sku_info.iterrows():
        sku_code = sku_row['sku_code']
        product_name = sku_row['product_name']
        stock = sku_row['stock']
        qpd = sku_row['qpd']
        lead_time_days = int(sku_row['lead_time_days'])

        if qpd == 0 or pd.isna(qpd):
            continue

        orders_in_transit = []

        for date in date_range:
            stock_beginning = stock

            arriving_orders = [o for o in orders_in_transit if o[0] == date]
            stock_received = sum(o[1] for o in arriving_orders)
            stock += stock_received
            orders_in_transit = [o for o in orders_in_transit if o[0] != date]

            sales = qpd
            stock -= sales

            doi = stock / qpd if qpd > 0 else 999
            total_in_transit = sum(o[1] for o in orders_in_transit)

            reorder_trigger = (doi <= reorder_threshold) and (len(orders_in_transit) == 0)
            order_placed = False
            order_quantity = 0

            if reorder_trigger:
                estimated_calendar_days = lead_time_days * 1.17
                order_quantity = (target_doi + estimated_calendar_days) * qpd - stock
                if order_quantity > 0:
                    order_placed = True
                    arrival_date = add_working_days(date, lead_time_days)
                    orders_in_transit.append((arrival_date, order_quantity))

            results.append({
                'date': date,
                'sku_code': sku_code,
                'product_name': product_name,
                'lead_time_days': lead_time_days,
                'stock_beginning': stock_beginning,
                'sales': sales,
                'stock_received': stock_received,
                'stock_ending': stock,
                'doi': doi,
                'order_placed': order_placed,
                'order_quantity': order_quantity,
                'orders_in_transit_qty': total_in_transit,
                'orders_in_transit_count': len(orders_in_transit)
            })

    return pd.DataFrame(results)


def analyze_simulation(results_df, reorder_threshold, target_doi, date_range):
    """Analyze simulation results and return key metrics."""
    daily_arrivals = results_df[results_df['stock_received'] > 0].groupby('date').agg({
        'sku_code': 'count'
    }).reset_index()
    daily_arrivals.columns = ['date', 'unique_skus_arrived']

    all_dates = pd.DataFrame({'date': date_range})
    daily_arrivals = all_dates.merge(daily_arrivals, on='date', how='left').fillna(0)
    daily_arrivals['day_of_week'] = daily_arrivals['date'].dt.day_name()

    avg_daily_skus = daily_arrivals['unique_skus_arrived'].mean()
    max_daily_skus = daily_arrivals['unique_skus_arrived'].max()
    median_daily_skus = daily_arrivals['unique_skus_arrived'].median()
    std_daily_skus = daily_arrivals['unique_skus_arrived'].std()

    days_over_capacity = (daily_arrivals['unique_skus_arrived'] > DAILY_SKU_CAPACITY).sum()

    bins = [0, 30, 90, 180, 270, 360, 540, 720, float('inf')]
    bin_labels = ['0-30', '31-90', '91-180', '181-270', '271-360', '361-540', '541-720', '720+']

    daily_arrivals_no_sunday = daily_arrivals[daily_arrivals['day_of_week'] != 'Sunday'].copy()
    daily_arrivals_no_sunday['bin'] = pd.cut(
        daily_arrivals_no_sunday['unique_skus_arrived'],
        bins=bins, labels=bin_labels, include_lowest=True
    )
    bin_counts = daily_arrivals_no_sunday['bin'].value_counts().sort_index()
    bin_distribution = dict(zip(bin_labels, [bin_counts.get(label, 0) for label in bin_labels]))

    total_unique_skus_arrived = results_df[results_df['stock_received'] > 0]['sku_code'].nunique()
    avg_doi = results_df['doi'].mean()
    total_orders = results_df['order_placed'].sum()

    daily_arrivals['is_overload'] = daily_arrivals['unique_skus_arrived'] > DAILY_SKU_CAPACITY
    overload_by_day = daily_arrivals.groupby('day_of_week')['is_overload'].sum().to_dict()
    avg_arrivals_by_day = daily_arrivals.groupby('day_of_week')['unique_skus_arrived'].mean().to_dict()

    return {
        'reorder_threshold': reorder_threshold,
        'target_doi': target_doi,
        'avg_daily_skus': avg_daily_skus,
        'max_daily_skus': max_daily_skus,
        'median_daily_skus': median_daily_skus,
        'std_daily_skus': std_daily_skus,
        'days_over_capacity': days_over_capacity,
        'pct_days_over_capacity': (days_over_capacity / len(date_range) * 100),
        'capacity_utilization': (avg_daily_skus / DAILY_SKU_CAPACITY * 100),
        'total_unique_skus_arrived': total_unique_skus_arrived,
        'total_capacity_utilization': (total_unique_skus_arrived / TOTAL_SKU_CAPACITY * 100),
        'total_orders': total_orders,
        'avg_doi': avg_doi,
        'daily_arrivals': daily_arrivals,
        'overload_by_day': overload_by_day,
        'avg_arrivals_by_day': avg_arrivals_by_day,
        'bin_distribution': bin_distribution,
    }


# ========================================
# PLOTLY CHART GENERATORS
# ========================================

# Consistent color palettes
DAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
DAY_COLORS = px.colors.qualitative.Set2[:7]
BIN_LABELS = ['0-30', '31-90', '91-180', '181-270', '271-360', '361-540', '541-720', '720+']
BIN_COLORS = px.colors.qualitative.T10[:8]


def _save_fig(fig, output_dir, name):
    """Save a Plotly figure as both .html and .json."""
    fig.write_html(os.path.join(output_dir, f'{name}.html'), include_plotlyjs='cdn')
    with open(os.path.join(output_dir, f'{name}.json'), 'w') as f:
        f.write(fig.to_json())


def chart_overload_days_by_doi_grouped_by_rt(all_results, target_dois, reorder_thresholds, run_id):
    """Chart 1: Overload Days — X=DOI, bars=days, subplots=RT."""
    n_rt = len(reorder_thresholds)
    fig = make_subplots(rows=n_rt, cols=1,
                        subplot_titles=[f'Reorder Threshold: {rt}' for rt in reorder_thresholds],
                        shared_yaxes=True, vertical_spacing=0.08)

    # Compute global y-max for shared y-axis
    all_vals = [int(r['overload_by_day'].get(d, 0)) for r in all_results for d in DAY_ORDER]
    y_max = max(all_vals) * 1.20 if max(all_vals) > 0 else 10

    for row_idx, rt in enumerate(reorder_thresholds, 1):
        rt_scenarios = [r for r in all_results if r['reorder_threshold'] == rt]
        for i, day in enumerate(DAY_ORDER):
            vals = []
            for doi in target_dois:
                m = next((r for r in rt_scenarios if r['target_doi'] == doi), None)
                vals.append(int(m['overload_by_day'].get(day, 0)) if m else 0)
            fig.add_trace(go.Bar(
                name=day, x=[f'DOI {d}' for d in target_dois], y=vals,
                marker_color=DAY_COLORS[i], text=vals, textposition='outside',
                textfont_size=9,
                showlegend=(row_idx == 1),
                legendgroup=day,
            ), row=row_idx, col=1)

    fig.update_layout(
        barmode='group',
        title_text=f'Overload Days by Target DOI — Grouped by Reorder Threshold<br><sub>(Days Exceeding {DAILY_SKU_CAPACITY} SKU Capacity)</sub>',
        height=500 * n_rt,
        template='plotly_white',
        legend_title_text='Day of Week',
    )
    for i in range(1, n_rt + 1):
        fig.update_yaxes(title_text='Number of Overload Days', range=[0, y_max], row=i, col=1)
        fig.update_xaxes(title_text='Target DOI', row=i, col=1)
    _save_fig(fig, OUTPUT_DIR, f'comparison_overload_days_bydoi_grouped_by_rt_{run_id}')


def chart_avg_arrivals_by_doi_grouped_by_rt(all_results, target_dois, reorder_thresholds, run_id):
    """Chart 2: Avg Arrivals — X=DOI, bars=days, subplots=RT."""
    n_rt = len(reorder_thresholds)
    fig = make_subplots(rows=n_rt, cols=1,
                        subplot_titles=[f'Reorder Threshold: {rt}' for rt in reorder_thresholds],
                        shared_yaxes=True, vertical_spacing=0.08)

    # Compute global y-max for shared y-axis
    all_vals = [r['avg_arrivals_by_day'].get(d, 0) for r in all_results for d in DAY_ORDER]
    y_max = max(all_vals) * 1.20

    for row_idx, rt in enumerate(reorder_thresholds, 1):
        rt_scenarios = [r for r in all_results if r['reorder_threshold'] == rt]
        for i, day in enumerate(DAY_ORDER):
            vals = []
            for doi in target_dois:
                m = next((r for r in rt_scenarios if r['target_doi'] == doi), None)
                vals.append(round(m['avg_arrivals_by_day'].get(day, 0), 1) if m else 0)
            fig.add_trace(go.Bar(
                name=day, x=[f'DOI {d}' for d in target_dois], y=vals,
                marker_color=DAY_COLORS[i], text=vals, textposition='outside',
                textfont_size=9,
                showlegend=(row_idx == 1),
                legendgroup=day,
            ), row=row_idx, col=1)

        # Capacity line
        fig.add_hline(y=DAILY_SKU_CAPACITY, line_dash='dash', line_color='red',
                      annotation_text=f'Capacity ({DAILY_SKU_CAPACITY})',
                      row=row_idx, col=1)

    fig.update_layout(
        barmode='group',
        title_text='Average SKU Arrivals by Target DOI — Grouped by Reorder Threshold',
        height=500 * n_rt,
        template='plotly_white',
        legend_title_text='Day of Week',
    )
    for i in range(1, n_rt + 1):
        fig.update_yaxes(title_text='Average Unique SKUs Arrived', range=[0, y_max], row=i, col=1)
        fig.update_xaxes(title_text='Target DOI', row=i, col=1)
    _save_fig(fig, OUTPUT_DIR, f'comparison_avg_arrivals_bydoi_grouped_by_rt_{run_id}')


def chart_binning_by_doi_grouped_by_rt(all_results, target_dois, reorder_thresholds, run_id):
    """Chart 3: Binning Distribution — X=DOI, bars=bins, subplots=RT."""
    n_rt = len(reorder_thresholds)
    fig = make_subplots(rows=n_rt, cols=1,
                        subplot_titles=[f'Reorder Threshold: {rt}' for rt in reorder_thresholds],
                        shared_yaxes=True, vertical_spacing=0.08)

    # Compute global y-max for shared y-axis
    all_vals = [int(r['bin_distribution'].get(bl, 0)) for r in all_results for bl in BIN_LABELS]
    y_max = max(all_vals) * 1.20 if max(all_vals) > 0 else 10

    for row_idx, rt in enumerate(reorder_thresholds, 1):
        rt_scenarios = [r for r in all_results if r['reorder_threshold'] == rt]
        for bi, bl in enumerate(BIN_LABELS):
            vals = []
            for doi in target_dois:
                m = next((r for r in rt_scenarios if r['target_doi'] == doi), None)
                vals.append(int(m['bin_distribution'].get(bl, 0)) if m else 0)
            fig.add_trace(go.Bar(
                name=bl, x=[f'DOI {d}' for d in target_dois], y=vals,
                marker_color=BIN_COLORS[bi], text=vals, textposition='outside',
                textfont_size=8,
                showlegend=(row_idx == 1),
                legendgroup=bl,
            ), row=row_idx, col=1)

    fig.update_layout(
        barmode='group',
        title_text='Daily Arrivals Distribution by DOI — Grouped by Reorder Threshold<br><sub>(Excluding Sundays)</sub>',
        height=500 * n_rt,
        template='plotly_white',
        legend_title_text='Arrivals Range',
    )
    for i in range(1, n_rt + 1):
        fig.update_yaxes(title_text='Number of Days', range=[0, y_max], row=i, col=1)
        fig.update_xaxes(title_text='Target DOI', row=i, col=1)
    _save_fig(fig, OUTPUT_DIR, f'comparison_binning_distribution_byscenario_{run_id}')


def chart_avg_arrivals_by_rt_grouped_by_doi(all_results, target_dois, reorder_thresholds, run_id):
    """Chart 4: Avg Arrivals — X=RT, bars=days, subplots=DOI."""
    n_doi = len(target_dois)
    fig = make_subplots(rows=n_doi, cols=1,
                        subplot_titles=[f'Target DOI: {doi}' for doi in target_dois],
                        shared_yaxes=True, vertical_spacing=0.08)

    # Compute global y-max for shared y-axis
    all_vals = [r['avg_arrivals_by_day'].get(d, 0) for r in all_results for d in DAY_ORDER]
    y_max = max(all_vals) * 1.20

    for row_idx, doi in enumerate(target_dois, 1):
        doi_scenarios = [r for r in all_results if r['target_doi'] == doi]
        for i, day in enumerate(DAY_ORDER):
            vals = []
            for rt in reorder_thresholds:
                m = next((r for r in doi_scenarios if r['reorder_threshold'] == rt), None)
                vals.append(round(m['avg_arrivals_by_day'].get(day, 0), 1) if m else 0)
            fig.add_trace(go.Bar(
                name=day, x=[f'RT {rt}' for rt in reorder_thresholds], y=vals,
                marker_color=DAY_COLORS[i], text=vals, textposition='outside',
                textfont_size=9,
                showlegend=(row_idx == 1),
                legendgroup=day,
            ), row=row_idx, col=1)

        fig.add_hline(y=DAILY_SKU_CAPACITY, line_dash='dash', line_color='red',
                      annotation_text=f'Capacity ({DAILY_SKU_CAPACITY})',
                      row=row_idx, col=1)

    fig.update_layout(
        barmode='group',
        title_text='Average SKU Arrivals by Reorder Threshold — Grouped by Target DOI',
        height=500 * n_doi,
        template='plotly_white',
        legend_title_text='Day of Week',
    )
    for i in range(1, n_doi + 1):
        fig.update_yaxes(title_text='Average Unique SKUs Arrived', range=[0, y_max], row=i, col=1)
        fig.update_xaxes(title_text='Reorder Threshold', row=i, col=1)
    _save_fig(fig, OUTPUT_DIR, f'comparison_avg_arrivals_byrt_grouped_by_doi_{run_id}')


def chart_overload_days_by_rt_grouped_by_doi(all_results, target_dois, reorder_thresholds, run_id):
    """Chart 5: Overload Days — X=RT, bars=days, subplots=DOI."""
    n_doi = len(target_dois)
    fig = make_subplots(rows=n_doi, cols=1,
                        subplot_titles=[f'Target DOI: {doi}' for doi in target_dois],
                        shared_yaxes=True, vertical_spacing=0.08)

    # Compute global y-max for shared y-axis
    all_vals = [int(r['overload_by_day'].get(d, 0)) for r in all_results for d in DAY_ORDER]
    y_max = max(all_vals) * 1.20 if max(all_vals) > 0 else 10

    for row_idx, doi in enumerate(target_dois, 1):
        doi_scenarios = [r for r in all_results if r['target_doi'] == doi]
        for i, day in enumerate(DAY_ORDER):
            vals = []
            for rt in reorder_thresholds:
                m = next((r for r in doi_scenarios if r['reorder_threshold'] == rt), None)
                vals.append(int(m['overload_by_day'].get(day, 0)) if m else 0)
            fig.add_trace(go.Bar(
                name=day, x=[f'RT {rt}' for rt in reorder_thresholds], y=vals,
                marker_color=DAY_COLORS[i], text=vals, textposition='outside',
                textfont_size=9,
                showlegend=(row_idx == 1),
                legendgroup=day,
            ), row=row_idx, col=1)

    fig.update_layout(
        barmode='group',
        title_text=f'Overload Days by Reorder Threshold — Grouped by Target DOI<br><sub>(Days Exceeding {DAILY_SKU_CAPACITY} SKU Capacity)</sub>',
        height=500 * n_doi,
        template='plotly_white',
        legend_title_text='Day of Week',
    )
    for i in range(1, n_doi + 1):
        fig.update_yaxes(title_text='Number of Overload Days', range=[0, y_max], row=i, col=1)
        fig.update_xaxes(title_text='Reorder Threshold', row=i, col=1)
    _save_fig(fig, OUTPUT_DIR, f'comparison_overload_days_by_rt_grouped_by_doi_{run_id}')


def chart_binning_by_rt_grouped_by_doi(all_results, target_dois, reorder_thresholds, run_id):
    """Chart 6: Binning Distribution — X=RT, bars=bins, subplots=DOI."""
    n_doi = len(target_dois)
    fig = make_subplots(rows=n_doi, cols=1,
                        subplot_titles=[f'Target DOI: {doi}' for doi in target_dois],
                        shared_yaxes=True, vertical_spacing=0.08)

    # Compute global y-max for shared y-axis
    all_vals = [int(r['bin_distribution'].get(bl, 0)) for r in all_results for bl in BIN_LABELS]
    y_max = max(all_vals) * 1.20 if max(all_vals) > 0 else 10

    for row_idx, doi in enumerate(target_dois, 1):
        doi_scenarios = [r for r in all_results if r['target_doi'] == doi]
        for bi, bl in enumerate(BIN_LABELS):
            vals = []
            for rt in reorder_thresholds:
                m = next((r for r in doi_scenarios if r['reorder_threshold'] == rt), None)
                vals.append(int(m['bin_distribution'].get(bl, 0)) if m else 0)
            fig.add_trace(go.Bar(
                name=bl, x=[f'RT {rt}' for rt in reorder_thresholds], y=vals,
                marker_color=BIN_COLORS[bi], text=vals, textposition='outside',
                textfont_size=8,
                showlegend=(row_idx == 1),
                legendgroup=bl,
            ), row=row_idx, col=1)

    fig.update_layout(
        barmode='group',
        title_text='Daily Arrivals Distribution by Reorder Threshold — Grouped by Target DOI',
        height=500 * n_doi,
        template='plotly_white',
        legend_title_text='Arrivals Range',
    )
    for i in range(1, n_doi + 1):
        fig.update_yaxes(title_text='Number of Days', range=[0, y_max], row=i, col=1)
        fig.update_xaxes(title_text='Reorder Threshold', row=i, col=1)
    _save_fig(fig, OUTPUT_DIR, f'comparison_binning_distribution_by_rt_grouped_by_doi_{run_id}')


def chart_boxplot_arrivals(all_results, target_dois, reorder_thresholds, run_id):
    """Chart 7: Boxplot of Daily Arrivals — X=DOI, subplots=RT."""
    doi_colors = px.colors.qualitative.Set2[:len(target_dois)]
    doi_color_map = {doi: doi_colors[i] for i, doi in enumerate(target_dois)}

    n_rt = len(reorder_thresholds)
    fig = make_subplots(rows=n_rt, cols=1,
                        subplot_titles=[f'Reorder Threshold: {rt}' for rt in reorder_thresholds],
                        shared_yaxes=True, vertical_spacing=0.08)

    # Compute global y-max for shared y-axis
    all_max_vals = []
    for r in all_results:
        arr = r['daily_arrivals']
        filtered = arr[arr['day_of_week'] != 'Sunday']['unique_skus_arrived']
        if len(filtered) > 0:
            all_max_vals.append(filtered.max())
    y_max = max(all_max_vals) * 1.15 if all_max_vals else 10

    for row_idx, rt in enumerate(reorder_thresholds, 1):
        rt_scenarios = [r for r in all_results if r['reorder_threshold'] == rt]
        for doi in target_dois:
            m = next((r for r in rt_scenarios if r['target_doi'] == doi), None)
            if m:
                arr = m['daily_arrivals']
                filtered = arr[arr['day_of_week'] != 'Sunday']['unique_skus_arrived'].values
            else:
                filtered = []
            fig.add_trace(go.Box(
                y=filtered,
                name=f'DOI {doi}',
                marker_color=doi_color_map[doi],
                showlegend=(row_idx == 1),
                legendgroup=f'DOI {doi}',
            ), row=row_idx, col=1)

        fig.add_hline(y=DAILY_SKU_CAPACITY, line_dash='dash', line_color='red',
                      annotation_text=f'Capacity ({DAILY_SKU_CAPACITY})',
                      row=row_idx, col=1)

    fig.update_layout(
        title_text='Distribution of Daily SKU Arrivals by Target DOI — Grouped by Reorder Threshold<br><sub>(Excluding Sundays)</sub>',
        height=500 * n_rt,
        template='plotly_white',
        legend_title_text='Target DOI',
    )
    for i in range(1, n_rt + 1):
        fig.update_yaxes(title_text='Daily Unique SKUs Arrived', range=[0, y_max], row=i, col=1)
        fig.update_xaxes(title_text='Target DOI', row=i, col=1)
    _save_fig(fig, OUTPUT_DIR, f'comparison_boxplot_arrivals_{run_id}')


# ========================================
# MAIN EXECUTION
# ========================================
def main():
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"Run ID: {run_id}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    df['tanggal_update'] = pd.to_datetime(df['tanggal_update'])

    # Prepare starting inventory
    print("\nPreparing starting inventory (July 1, 2025)...")
    july_1 = datetime(2025, 7, 1)
    starting_data = df[df['tanggal_update'] == july_1].copy()

    if len(starting_data) == 0:
        print(f"Warning: No data for July 1, using first available date: {df['tanggal_update'].min()}")
        starting_data = df[df['tanggal_update'] == df['tanggal_update'].min()].copy()

    sku_info = starting_data.groupby('sku_code').agg({
        'product_name': 'first',
        'stock': 'first',
        'qpd': 'first',
        'doi': 'first',
        'lead_time_days': 'first'
    }).reset_index()

    print(f"Starting with {len(sku_info)} unique SKUs")
    print(f"Lead time range: {sku_info['lead_time_days'].min():.0f} to {sku_info['lead_time_days'].max():.0f} working days")

    date_range = pd.date_range(START_DATE, END_DATE, freq='D')

    param_combinations = list(product(REORDER_THRESHOLD_RANGE, TARGET_DOI_RANGE))
    total_scenarios = len(param_combinations)

    all_scenario_results = []
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Run simulations
    for scenario_num, (reorder_threshold, target_doi) in enumerate(param_combinations, 1):
        print(f"\nScenario {scenario_num}/{total_scenarios}: Reorder Threshold={reorder_threshold}, Target DOI={target_doi}")

        results_df = run_single_simulation(sku_info, reorder_threshold, target_doi, date_range)
        analysis = analyze_simulation(results_df, reorder_threshold, target_doi, date_range)
        all_scenario_results.append(analysis)

        # Save detailed results
        if SAVE_DETAILED_RESULTS:
            scenario_filename = f"scenario_RT{reorder_threshold}_DOI{target_doi}_detailed2.csv"
            results_df.to_csv(os.path.join(OUTPUT_DIR, scenario_filename), index=False)
            print(f"\n  [OK] Saved: {scenario_filename}")

        # Save daily arrivals
        if SAVE_DAILY_SUMMARIES:
            daily_filename = f"scenario_RT{reorder_threshold}_DOI{target_doi}_daily2.csv"
            analysis['daily_arrivals'].to_csv(os.path.join(OUTPUT_DIR, daily_filename), index=False)

    # ── Comparison summary ──
    print(f"\n{'='*60}")
    print("CREATING COMPARISON SUMMARY")
    print(f"{'='*60}")

    comparison_df = pd.DataFrame([
        {
            'Scenario': f"RT{r['reorder_threshold']}_DOI{r['target_doi']}",
            'Reorder_Threshold': r['reorder_threshold'],
            'Target_DOI': r['target_doi'],
            'Avg_Daily_SKUs': round(r['avg_daily_skus'], 2),
            'Max_Daily_SKUs': int(r['max_daily_skus']),
            'Days_Over_Capacity': int(r['days_over_capacity']),
            'Pct_Days_Over_Capacity': round(r['pct_days_over_capacity'], 2),
            'Capacity_Utilization_Pct': round(r['capacity_utilization'], 2),
            'Total_Orders': int(r['total_orders']),
            'StDev_Daily_SKUs': round(r['std_daily_skus'], 2),
            'Overload_Monday': int(r['overload_by_day'].get('Monday', 0)),
            'Overload_Tuesday': int(r['overload_by_day'].get('Tuesday', 0)),
            'Overload_Wednesday': int(r['overload_by_day'].get('Wednesday', 0)),
            'Overload_Thursday': int(r['overload_by_day'].get('Thursday', 0)),
            'Overload_Friday': int(r['overload_by_day'].get('Friday', 0)),
            'Overload_Saturday': int(r['overload_by_day'].get('Saturday', 0)),
            'Overload_Sunday': int(r['overload_by_day'].get('Sunday', 0)),
            'Avg_Monday': round(r['avg_arrivals_by_day'].get('Monday', 0), 2),
            'Avg_Tuesday': round(r['avg_arrivals_by_day'].get('Tuesday', 0), 2),
            'Avg_Wednesday': round(r['avg_arrivals_by_day'].get('Wednesday', 0), 2),
            'Avg_Thursday': round(r['avg_arrivals_by_day'].get('Thursday', 0), 2),
            'Avg_Friday': round(r['avg_arrivals_by_day'].get('Friday', 0), 2),
            'Avg_Saturday': round(r['avg_arrivals_by_day'].get('Saturday', 0), 2),
            'Avg_Sunday': round(r['avg_arrivals_by_day'].get('Sunday', 0), 2),
        }
        for r in all_scenario_results
    ])

    comparison_df = comparison_df.sort_values(['Reorder_Threshold', 'Target_DOI'])
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, f'scenario_comparison_summary_byday_{run_id}.csv'), index=False)

    print("\n" + "="*60)
    print("SCENARIO COMPARISON TABLE")
    print("="*60)
    print(comparison_df.to_string(index=False))

    best_capacity = comparison_df.loc[comparison_df['Days_Over_Capacity'].idxmin()]
    print(f"\n[BEST] Best for capacity (fewest days over limit):")
    print(f"  Scenario: {best_capacity['Scenario']}")
    print(f"  Days over capacity: {best_capacity['Days_Over_Capacity']}")
    print(f"  Capacity utilization: {best_capacity['Capacity_Utilization_Pct']:.1f}%")

    # ── Generate Plotly charts ──
    print("\n" + "="*60)
    print("CREATING PLOTLY VISUALIZATIONS")
    print("="*60)

    reorder_thresholds = sorted(set(r['reorder_threshold'] for r in all_scenario_results))
    target_dois = sorted(set(r['target_doi'] for r in all_scenario_results))

    chart_overload_days_by_doi_grouped_by_rt(all_scenario_results, target_dois, reorder_thresholds, run_id)
    print("  [OK] Chart 1: Overload Days (X=DOI, grouped by RT)")

    chart_avg_arrivals_by_doi_grouped_by_rt(all_scenario_results, target_dois, reorder_thresholds, run_id)
    print("  [OK] Chart 2: Avg Arrivals (X=DOI, grouped by RT)")

    chart_binning_by_doi_grouped_by_rt(all_scenario_results, target_dois, reorder_thresholds, run_id)
    print("  [OK] Chart 3: Binning Distribution (X=DOI, grouped by RT)")

    chart_avg_arrivals_by_rt_grouped_by_doi(all_scenario_results, target_dois, reorder_thresholds, run_id)
    print("  [OK] Chart 4: Avg Arrivals (X=RT, grouped by DOI)")

    chart_overload_days_by_rt_grouped_by_doi(all_scenario_results, target_dois, reorder_thresholds, run_id)
    print("  [OK] Chart 5: Overload Days (X=RT, grouped by DOI)")

    chart_binning_by_rt_grouped_by_doi(all_scenario_results, target_dois, reorder_thresholds, run_id)
    print("  [OK] Chart 6: Binning Distribution (X=RT, grouped by DOI)")

    chart_boxplot_arrivals(all_scenario_results, target_dois, reorder_thresholds, run_id)
    print("  [OK] Chart 7: Boxplot Daily Arrivals")

    print("\n" + "="*60)
    print("MULTI-SCENARIO ANALYSIS COMPLETE!")


if __name__ == "__main__":
    main()
