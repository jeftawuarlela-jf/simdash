"""
Supply Chain Simulation – Dash Web App
Run with:  python dash_app.py
Then open http://127.0.0.1:8050 in your browser.
"""

import dash
from dash import html, dcc, dash_table, Input, Output, State, callback
import dash_bootstrap_components as dbc
import subprocess
import sys
import os
import io
import base64
import zipfile
import tempfile
import textwrap
import glob
import json
import pandas as pd
from datetime import date, datetime
import plotly.io as pio

# ─────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    title="📦 Supply Chain Simulation",
)
server = app.server

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
sidebar = html.Div([
    html.Div([
        html.Img(src="https://img.icons8.com/fluency/96/supply-chain.png",
                 style={"width": "50px", "marginBottom": "10px"}),
        html.H3("Simulation Config", style={"color": "#7eb8f7"}),
    ], style={"textAlign": "center", "marginBottom": "20px"}),

    # Data file
    html.Label("📁 DATA FILE", className="sidebar-section-label"),
    dcc.Upload(
        id='upload-csv',
        children=html.Div([
            html.Span("Drag & Drop or "),
            html.A("Browse CSV", style={"color": "#7eb8f7", "fontWeight": "bold"})
        ]),
        style={
            "width": "100%", "padding": "12px", "borderWidth": "2px",
            "borderStyle": "dashed", "borderRadius": "8px", "borderColor": "#4a6785",
            "textAlign": "center", "cursor": "pointer", "marginBottom": "6px",
            "background": "#1a2636",
        },
        multiple=False,
    ),
    html.Div(id='upload-status', style={"fontSize": "0.8rem", "color": "#8fb8e0", "marginBottom": "16px"}),

    # RT
    html.Label("🔄 REORDER TRIGGER (RT)", className="sidebar-section-label"),
    dbc.Row([
        dbc.Col([
            html.Label("Start", style={"fontSize": "0.75rem"}),
            dbc.Input(id="rt-start", type="number", value=21, min=1, max=99, step=1, size="sm"),
        ]),
        dbc.Col([
            html.Label("Stop ①", style={"fontSize": "0.75rem"}),
            dbc.Input(id="rt-stop", type="number", value=22, min=2, max=100, step=1, size="sm"),
        ]),
    ], className="mb-3"),

    # DOI
    html.Label("🎯 TARGET DOI", className="sidebar-section-label"),
    dbc.Row([
        dbc.Col([
            html.Label("Start", style={"fontSize": "0.75rem"}),
            dbc.Input(id="doi-start", type="number", value=27, min=1, max=364, step=1, size="sm"),
        ]),
        dbc.Col([
            html.Label("Stop ②", style={"fontSize": "0.75rem"}),
            dbc.Input(id="doi-stop", type="number", value=30, min=2, max=365, step=1, size="sm"),
        ]),
    ], className="mb-3"),

    # Capacity
    html.Label("🏭 CAPACITY LIMITS", className="sidebar-section-label"),
    html.Label("Daily SKU Capacity", style={"fontSize": "0.75rem"}),
    dbc.Input(id="daily-cap", type="number", value=360, min=1, step=10, size="sm", className="mb-2"),
    html.Label("Total SKU Capacity", style={"fontSize": "0.75rem"}),
    dbc.Input(id="total-cap", type="number", value=5100, min=1, step=100, size="sm", className="mb-3"),

    # Dates
    html.Label("📆 SIMULATION PERIOD", className="sidebar-section-label"),
    html.Label("Start Date", style={"fontSize": "0.75rem"}),
    dcc.DatePickerSingle(id="start-date", date=date(2026, 2, 1),
                         display_format="YYYY-MM-DD",
                         style={"marginBottom": "8px", "width": "100%"}),
    html.Label("End Date", style={"fontSize": "0.75rem"}),
    dcc.DatePickerSingle(id="end-date", date=date(2026, 3, 31),
                         display_format="YYYY-MM-DD",
                         style={"marginBottom": "12px", "width": "100%"}),

    # Output options
    html.Label("💾 OUTPUT OPTIONS", className="sidebar-section-label"),
    dbc.Checklist(
        id="output-options",
        options=[
            {"label": " Save detailed per-SKU results", "value": "detailed"},
            {"label": " Save daily summaries", "value": "daily"},
        ],
        value=["detailed", "daily"],
        className="mb-3",
        style={"fontSize": "0.8rem"},
    ),

    html.Hr(style={"borderColor": "#3d5a80"}),
    html.Small("① ② Stop is exclusive (like Python's range())",
               style={"color": "#6a8cad", "fontSize": "0.7rem"}),

], style={
    "background": "#1e2a38", "color": "#e0e6ef", "padding": "20px",
    "height": "100vh", "overflowY": "auto", "position": "fixed",
    "top": 0, "left": 0, "width": "320px",
})

# ─────────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────────
main_content = html.Div([
    html.H2("📦 Supply Chain Simulation", style={"fontWeight": "bold"}),
    html.P("Configure parameters in the sidebar, then click Run Simulation below.",
           style={"color": "#9ab3cc"}),

    # Scenario preview metrics
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("RT values to test", className="text-muted"),
            html.H3(id="metric-rt", children="1"),
            html.Small(id="metric-rt-range", children="RT 21 → 21",
                       style={"color": "#6c8ebf"}),
        ]), className="metric-card"), width=4),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("DOI values to test", className="text-muted"),
            html.H3(id="metric-doi", children="3"),
            html.Small(id="metric-doi-range", children="DOI 27 → 29",
                       style={"color": "#6c8ebf"}),
        ]), className="metric-card"), width=4),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Total scenarios", className="text-muted"),
            html.H3(id="metric-total", children="3"),
            html.Small("combinations", style={"color": "#6c8ebf"}),
        ]), className="metric-card"), width=4),
    ], className="mb-4"),

    html.Hr(),

    # Run button
    dbc.Button("▶  Run Simulation", id="run-btn", color="primary", size="lg",
               className="w-100 mb-4", disabled=True),

    # Status
    html.Div(id="status-area"),

    # Log output
    html.Div(id="log-area"),

    # Results area
    html.Div(id="results-area"),

], style={
    "marginLeft": "340px", "padding": "30px 40px",
})

# ─────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────
app.layout = html.Div([
    sidebar,
    main_content,
    dcc.Store(id='csv-store'),        # stores uploaded CSV info
    dcc.Store(id='results-store'),    # stores output directory path
])

# ─────────────────────────────────────────────────────────────
# CSS overrides
# ─────────────────────────────────────────────────────────────
app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
        body { background: #0d1117; color: #c9d1d9; }
        .sidebar-section-label {
            color: #7eb8f7 !important;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 14px;
            margin-bottom: 6px;
            display: block;
            font-weight: 600;
        }
        .metric-card {
            background: #161b22 !important;
            border: 1px solid #21262d !important;
            border-radius: 10px !important;
        }
        .metric-card .card-body h3 { color: #58a6ff; }
        .metric-card .card-body h6 { color: #8b949e !important; }
        .log-box {
            background: #0d1117; color: #c9d1d9; font-family: 'Courier New', monospace;
            font-size: 0.78rem; padding: 16px; border-radius: 8px;
            max-height: 400px; overflow-y: auto; white-space: pre-wrap;
            border: 1px solid #21262d;
        }
        .dash-table-container .dash-spreadsheet {
            background: #161b22 !important;
        }
        .DateInput_input { background: #1a2636 !important; color: #e0e6ef !important; border: 1px solid #3d5a80 !important; }
        .SingleDatePickerInput { background: #1a2636 !important; }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>
'''

# ─────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────

# Update metrics preview
@callback(
    Output("metric-rt", "children"),
    Output("metric-rt-range", "children"),
    Output("metric-doi", "children"),
    Output("metric-doi-range", "children"),
    Output("metric-total", "children"),
    Input("rt-start", "value"),
    Input("rt-stop", "value"),
    Input("doi-start", "value"),
    Input("doi-stop", "value"),
)
def update_metrics(rt_start, rt_stop, doi_start, doi_stop):
    rt_start = rt_start or 21
    rt_stop = rt_stop or 22
    doi_start = doi_start or 27
    doi_stop = doi_stop or 30
    n_rt = max(0, rt_stop - rt_start)
    n_doi = max(0, doi_stop - doi_start)
    return (
        str(n_rt),
        f"RT {rt_start} → {rt_stop - 1}",
        str(n_doi),
        f"DOI {doi_start} → {doi_stop - 1}",
        str(n_rt * n_doi),
    )


# Handle file upload
@callback(
    Output("upload-status", "children"),
    Output("csv-store", "data"),
    Output("run-btn", "disabled"),
    Input("upload-csv", "contents"),
    State("upload-csv", "filename"),
)
def handle_upload(contents, filename):
    if contents is None:
        return "No file uploaded yet.", None, True
    return f"✓ {filename}", {"contents": contents, "filename": filename}, False


# Run simulation
@callback(
    Output("status-area", "children"),
    Output("log-area", "children"),
    Output("results-area", "children"),
    Input("run-btn", "n_clicks"),
    State("csv-store", "data"),
    State("rt-start", "value"),
    State("rt-stop", "value"),
    State("doi-start", "value"),
    State("doi-stop", "value"),
    State("daily-cap", "value"),
    State("total-cap", "value"),
    State("start-date", "date"),
    State("end-date", "date"),
    State("output-options", "value"),
    prevent_initial_call=True,
)
def run_simulation(n_clicks, csv_data, rt_start, rt_stop, doi_start, doi_stop,
                   daily_cap, total_cap, start_date_str, end_date_str, output_options):

    # ── Guard against None values ──
    rt_start = rt_start or 21
    rt_stop = rt_stop or 22
    doi_start = doi_start or 27
    doi_stop = doi_stop or 30
    daily_cap = daily_cap or 360
    total_cap = total_cap or 5100

    # ── Validation ──
    errors = []
    if csv_data is None:
        errors.append("Please upload your CSV data file.")
    if rt_stop <= rt_start:
        errors.append("RT Stop must be greater than RT Start.")
    if doi_stop <= doi_start:
        errors.append("DOI Stop must be greater than DOI Start.")

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date() if start_date_str else date(2026, 2, 1)
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date() if end_date_str else date(2026, 3, 31)
    if end_date <= start_date:
        errors.append("End Date must be after Start Date.")

    if errors:
        return (
            dbc.Alert([html.Li(e) for e in errors], color="danger"),
            None,
            None,
        )

    # ── Setup working directories ──
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = tempfile.mkdtemp(prefix="sim_work_")
    out_dir = tempfile.mkdtemp(prefix=f"sim_out_{run_id}_")

    # Save uploaded CSV
    content_type, content_string = csv_data["contents"].split(",")
    decoded = base64.b64decode(content_string)
    csv_path = os.path.join(work_dir, csv_data["filename"])
    with open(csv_path, "wb") as f:
        f.write(decoded)

    # Write config.py
    save_detailed = "detailed" in (output_options or [])
    save_daily = "daily" in (output_options or [])

    config_content = textwrap.dedent(f"""\
        REORDER_THRESHOLD_RANGE = range({rt_start}, {rt_stop})
        TARGET_DOI_RANGE        = range({doi_start}, {doi_stop})
        DAILY_SKU_CAPACITY      = {daily_cap}
        TOTAL_SKU_CAPACITY      = {total_cap}
        START_DATE = ({start_date.year}, {start_date.month}, {start_date.day})
        END_DATE   = ({end_date.year},   {end_date.month},   {end_date.day})
        DATA_FILE  = r'{csv_path}'
        OUTPUT_DIR = r'{out_dir}'
        SAVE_DETAILED_RESULTS = {save_detailed}
        SAVE_DAILY_SUMMARIES  = {save_daily}
    """)
    with open(os.path.join(work_dir, "config.py"), "w") as f:
        f.write(config_content)

    # Copy simulation_dash.py into work_dir
    import shutil
    sim_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation_dash.py")
    if not os.path.exists(sim_src):
        return (
            dbc.Alert("simulation_dash.py not found next to dash_app.py.", color="danger"),
            None, None,
        )
    shutil.copy(sim_src, os.path.join(work_dir, "simulation_dash.py"))

    # ── Run simulation ──
    proc = subprocess.Popen(
        [sys.executable, "simulation_dash.py"],
        cwd=work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    stdout, _ = proc.communicate()
    success = (proc.returncode == 0)

    # Status banner
    if success:
        status = dbc.Alert("✅  Simulation completed successfully!", color="success")
    else:
        status = dbc.Alert("❌  Simulation failed — see log below.", color="danger")

    # Log area
    log_section = html.Div([
        html.H5("🖥️ Simulation Log", style={"marginTop": "16px"}),
        html.Div(stdout, className="log-box"),
    ])

    if not success:
        return status, log_section, None

    # ── Build results area ──
    results_children = []
    results_children.append(html.Hr())
    results_children.append(html.H4("📊 Results", style={"marginTop": "20px"}))

    # Summary table
    csv_files = glob.glob(os.path.join(out_dir, "scenario_comparison_summary_byday_*.csv"))
    if csv_files:
        df = pd.read_csv(csv_files[0])
        results_children.append(html.H5("📋 Scenario Comparison Table"))
        results_children.append(
            dash_table.DataTable(
                data=df.to_dict("records"),
                columns=[{"name": c, "id": c} for c in df.columns],
                style_table={"overflowX": "auto"},
                style_header={
                    "backgroundColor": "#1a2636", "color": "#7eb8f7",
                    "fontWeight": "bold", "border": "1px solid #21262d",
                },
                style_cell={
                    "backgroundColor": "#161b22", "color": "#c9d1d9",
                    "border": "1px solid #21262d", "fontSize": "0.82rem",
                    "textAlign": "center", "padding": "6px 10px",
                },
                style_data_conditional=[
                    {"if": {"row_index": "odd"},
                     "backgroundColor": "#1a2130"},
                ],
                page_size=20,
            )
        )

        # Best scenario card
        if "Days_Over_Capacity" in df.columns:
            best = df.loc[df["Days_Over_Capacity"].idxmin()]
            results_children.append(html.H5("🏆 Best Scenario (fewest days over capacity)",
                                            style={"marginTop": "20px"}))
            results_children.append(dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("Scenario", className="text-muted"),
                    html.H4(str(best.get("Scenario", "—"))),
                ]), className="metric-card"), width=3),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("Days Over Capacity", className="text-muted"),
                    html.H4(str(int(best.get("Days_Over_Capacity", 0)))),
                ]), className="metric-card"), width=3),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("Capacity Util %", className="text-muted"),
                    html.H4(f"{best.get('Capacity_Utilization_Pct', 0):.1f}%"),
                ]), className="metric-card"), width=3),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("Avg DOI", className="text-muted"),
                    html.H4(f"{best.get('Avg_DOI', 0):.2f}"),
                ]), className="metric-card"), width=3),
            ], className="mb-4"))

    # Plotly charts
    chart_json_files = sorted(glob.glob(os.path.join(out_dir, "*.json")))
    if chart_json_files:
        results_children.append(html.H5("📈 Interactive Charts", style={"marginTop": "20px"}))
        for jf in chart_json_files:
            chart_name = os.path.basename(jf).replace("_", " ").replace(".json", "").title()
            with open(jf, 'r') as f:
                fig_dict = json.load(f)
            fig = pio.from_json(json.dumps(fig_dict))
            fig.update_layout(
                paper_bgcolor='#0d1117',
                plot_bgcolor='#161b22',
                font_color='#c9d1d9',
            )
            results_children.append(html.Details([
                html.Summary(f"📊 {chart_name}",
                             style={"cursor": "pointer", "padding": "10px",
                                    "background": "#161b22", "borderRadius": "8px",
                                    "marginBottom": "8px", "fontWeight": "bold",
                                    "border": "1px solid #21262d"}),
                dcc.Graph(figure=fig, style={"marginBottom": "20px"}),
            ], open=False))

    # ZIP download
    results_children.append(html.H5("⬇️ Download All Results", style={"marginTop": "20px"}))
    zip_buffer = io.BytesIO()
    all_output_files = glob.glob(os.path.join(out_dir, "*"))
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in all_output_files:
            zf.write(fpath, arcname=os.path.basename(fpath))
    zip_buffer.seek(0)
    b64_zip = base64.b64encode(zip_buffer.read()).decode()

    results_children.append(
        html.A(
            dbc.Button("📥 Download Results ZIP (CSVs + Charts)", color="primary",
                       size="lg", className="w-100"),
            href=f"data:application/zip;base64,{b64_zip}",
            download=f"simulation_results_{run_id}.zip",
        )
    )

    results_area = html.Div(results_children)
    return status, log_section, results_area


# ─────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, port=8050)
