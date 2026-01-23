from datetime import datetime, timedelta

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import Input, Output, callback, dash_table, dcc, html
from sqlalchemy import select

from punchbowl.pipe.control.db import Flow, Health
from punchbowl.pipe.monitor.app import get_database_session

REFRESH_RATE = 60  # seconds

column_names = [ "flow_level", "flow_type", "state", "priority",
                 "creation_time", "launch_time", "start_time", "end_time",
                 "flow_id", "flow_run_id",
                 "flow_run_name", "call_data"]
schedule_columns =[{"name": v.replace("_", " ").capitalize(), "id": v} for v in column_names]
PAGE_SIZE = 15

dash.register_page(__name__, path="/")

def layout():
    return html.Div([
        dcc.Graph(id="machine-graph"),
        dbc.Row(align="center", children=[
            dbc.Col(width="auto", children=[
                dcc.DatePickerRange(id="plot-range",
                                    min_date_allowed=datetime(2025, 3, 1),
                                    max_date_allowed=datetime.today() + timedelta(days=1),
                                    end_date=datetime.today() + timedelta(days=1),
                                    start_date=datetime.today() - timedelta(days=1),
                                    persistence=True, persistence_type="memory",
                                    ),
            ]),
            dbc.Col(width="auto", children=[
                "Averaging window (minutes)",
                dcc.Input(
                    id="machine-stat-smoothing",
                    type="number",
                    debounce=1,
                    min=1,
                    step=1,
                    value=5,
                    style={"display": "block"},
                    persistence=True, persistence_type="memory",
                ),
            ]),
            dbc.Col(width=True, children=[
                dcc.Dropdown(
                    id="machine-stat",
                    options=["cpu_usage", "memory_usage", "memory_percentage", "disk_usage", "disk_percentage", "num_pids"],
                    value="cpu_usage",
                    clearable=False,
                    style={"width": "50%"},
                    persistence=True, persistence_type="memory",
                ),
            ]),
        ]),
        html.Hr(),
        html.Div(
            id="status-cards",
        ),
        html.Div(
            id="file-status-cards",
        ),
        html.Div([
            html.Div(children=[dcc.Graph(id="flow-throughput")], style={"padding": 10, "flex": 1}),

            html.Div(children=[dcc.Graph(id="flow-duration")], style={"padding": 10, "flex": 1}),
        ], style={"display": "flex", "flexDirection": "row"}),
        dash_table.DataTable(id="flows-table",
                             data=pd.DataFrame({name: [] for name in column_names}).to_dict("records"),
                             columns=schedule_columns,
                             page_current=0,
                             page_size=PAGE_SIZE,
                             page_action="custom",

                             filter_action="custom",
                             filter_query="",

                             sort_action="custom",
                             sort_mode="multi",
                             sort_by=[],
                             style_table={"overflowX": "auto",
                                          "textAlign": "left"},
                             persistence=True, persistence_type="memory",
                             ),
        dcc.Interval(
            id="interval-component",
            interval=REFRESH_RATE * 1000,  # in milliseconds
            n_intervals=0),
    ], style={"margin": "10px"})

operators = [(["ge ", ">="], "__ge__"),
             (["le ", "<="], "__le__"),
             (["lt ", "<"], "__lt__"),
             (["gt ", ">"], "__gt__"),
             (["ne ", "!="], "__ne__"),
             (["eq ", "="], "__eq__"),
             (["contains "], "contains"),
             (["datestartswith "], None),
            ]

def split_filter_part(filter_part):
    for operator_type, py_method in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find("{") + 1: name_part.rfind("}")]

                value_part = value_part.strip()
                v0 = value_part[0]
                if (v0 == value_part[-1] and v0 in ("'", '"', "`")):
                    value = value_part[1: -1].replace("\\" + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                # word operators need spaces after them in the filter string,
                # but we don't want these later
                return name, operator_type[0].strip(), value, py_method

    return [None] * 4

@callback(
    Output("flows-table", "data"),
    Input("interval-component", "n_intervals"),
    Input("flows-table", "page_current"),
    Input("flows-table", "page_size"),
    Input("flows-table", "sort_by"),
    Input("flows-table", "filter_query"))
def update_flows(n, page_current, page_size, sort_by, filter):
    query = select(Flow)
    for filter_part in filter.split(" && "):
        col_name, operator, filter_value, py_method = split_filter_part(filter_part)
        if col_name is not None:
            query = query.where(getattr(getattr(Flow, col_name), py_method)(filter_value))
    for col in sort_by:
        sort_column = getattr(Flow, col["column_id"])
        if col["direction"] == "asc":
            sort_column = sort_column.asc()
        else:
            sort_column = sort_column.desc()
        query = query.order_by(sort_column)
    query = query.offset(page_current * page_size).limit(page_size)
    with get_database_session() as session:
        dff = pd.read_sql_query(query, session.connection())

    return dff.to_dict("records")


def create_card_content(level: int | str, type: str | None, status: str, message: str):
    if type == "levelq_CNN":
        type = "CNN"
    if type == "levelq_CTM":
        type = "CTM"
    type_insert = f" {type}" if type is not None else ""
    return [
        dbc.CardBody(
            [
                html.H5(f"Level {level}{type_insert} Status: {status}", className="card-title"),
                html.P(
                    message,
                    className="card-text",
                ),
            ],
        ),
    ]

@callback(
    Output("status-cards", "children"),
    Input("interval-component", "n_intervals"),
)
def update_cards(n):
    with get_database_session() as session:
        reference_time = datetime.now() - timedelta(hours=24)
        query = (f"SELECT flow_level AS level, flow_type, SUM(state = 'completed') AS n_good, "
                  "SUM(state = 'failed') AS n_bad, SUM(state = 'running') AS n_running,"
                 f"SUM(state = 'timed_out') AS n_timed_out "
                 f"FROM flows WHERE start_time > '{reference_time}' "
                  "GROUP BY level, flow_type;")
        df = pd.read_sql_query(query, session.connection())
        # These states don't have a start_time set
        query = ("SELECT flow_level AS level, flow_type, "
                 "SUM(state = 'launched') AS n_launched, SUM(state = 'planned') AS n_planned "
                 "FROM flows GROUP BY level, flow_type;")
        second_df = pd.read_sql_query(query, session.connection())
    df = df.dropna().merge(second_df.dropna().set_index("level"), on=["level", "flow_type"], how="outer")
    df = df.infer_objects()
    df.fillna(0, inplace=True)

    cards = []
    for level, type in zip(["0", "1", "2", "3", "S", "Q", "Q"],
                           [None, None, None, None, None, "levelq_CNN", "levelq_CTM"]):
        sub_df = df.loc[(df["level"] == level)]
        if type is not None:
            sub_df = sub_df.loc[(sub_df["flow_type"] == type)]

        if len(sub_df) == 0:
            cards.append(dbc.Col(dbc.Card(create_card_content(level,  type, "", "No activity"),
                                          color="light", inverse=False)))
            continue

        n_good, n_bad, n_running = sub_df["n_good"].sum(), sub_df["n_bad"].sum(), sub_df["n_running"].sum()
        n_launched, n_planned, n_timed_out = sub_df["n_launched"].sum(), sub_df["n_planned"].sum(), sub_df["n_timed_out"].sum()
        n_timed_out = sub_df["n_timed_out"].sum()

        n_planned = sub_df["n_planned"].sum()
        message = (f"{n_good:.0f} ✅     {n_bad + n_timed_out:.0f} ⛔     {n_launched:.0f} 🚀     {n_running:.0f} ⏳     "
                   f"{n_planned:.0f} 💭")
        if n_good == 0 and n_bad == 0 and n_planned == 0 and n_launched == 0 and n_running == 0:
            color = "light"
            status = ""
            message = "No activity"
        elif (n_good > 0 and n_bad / n_good > 0.95) or (n_good == 0 and n_bad):
            color = "danger"
            status = "Bad"
        else:
            color = "success"
            status = "Good"
        cards.append(dbc.Col(dbc.Card(create_card_content(level, type, status, message),
                                      color=color, inverse=color != "light",
                                      # This preserves the multiple spaces separating the status count indicators
                                      style={"white-space": "pre"})))

    return html.Div([dbc.Row(cards, className="mb-4")])


def create_file_card_content(level: int | str, type: str | None, status: str, message: str):
    if type == "CN":
        type = "CNN"
    if type == "CT":
        type = "CTM"
    type_insert = f" {type}" if type is not None else ""
    return [
        dbc.CardBody(
            [
                html.H5(f"Level {level}{type_insert} File Status: {status}", className="card-title"),
                html.P(
                    message,
                    className="card-text",
                ),
            ],
        ),
    ]

@callback(
    Output("file-status-cards", "children"),
    Input("interval-component", "n_intervals"),
)
def update_file_cards(n):
    query = ("SELECT level, file_type, SUM(state = 'created') AS n_created, "
             "SUM(state = 'failed') AS n_failed, SUM(state = 'planned') AS n_planned, "
             "SUM(state = 'creating') AS n_creating, SUM(state = 'progressed') AS n_progressed, "
             "SUM(state = 'timed_out') AS n_timed_out, "
             "SUM(state = 'quickpunched') AS n_quickpunched FROM files GROUP BY level, file_type;")
    with get_database_session() as session:
        df = pd.read_sql_query(query, session.connection())

    cards = []
    for level, type in zip(["0", "1", "2", "3", "S", "Q", "Q"], [None, None, None, None, None, "CN", "CT"]):
        sub_df = df.loc[(df["level"] == level)]
        if type is not None:
            sub_df = sub_df.loc[(sub_df["file_type"] == type)]

        if len(sub_df) == 0:
            cards.append(dbc.Col(dbc.Card(create_file_card_content(level, type, "", "No activity"),
                                          color="light", inverse=False)))
            continue

        n_created, n_failed = sub_df["n_created"].sum(), sub_df["n_failed"].sum()
        n_timed_out = sub_df["n_timed_out"].sum()
        n_creating, n_progressed = sub_df["n_creating"].sum(), sub_df["n_progressed"].sum()
        n_quickpunched, n_planned = sub_df["n_quickpunched"].sum(), sub_df["n_planned"].sum()

        n_good = n_created + n_quickpunched + n_progressed

        if level == "1" or (level == "Q" and type == "CN"):
            sub_status = f"({n_created:.0f} 🏁 + {n_quickpunched:.0f} ⚡ + {n_progressed:.0f} ➡️)"
        else:
            sub_status = f"({n_created:.0f} 🏁 + {n_progressed:.0f} ➡️)"
        message = (f"{n_good:.0f} ✅ {sub_status}\n{n_failed + n_timed_out:.0f} ⛔     {n_creating:.0f} ⏳     "
                   f"{n_planned:.0f} 💭️")
        if n_good == 0 and n_failed == 0:
            color = "light"
            status = ""
            message = "No activity"
        elif n_failed / n_good > 0.95:
            color = "danger"
            status = "Bad"
        else:
            color = "success"
            status = "Good"
        cards.append(dbc.Col(dbc.Card(create_file_card_content(level, type, status, message),
                                      color=color, inverse=color != "light",
                                      # This preserves the multiple spaces separating the status count indicators
                                      style={"white-space": "pre"})))

    return html.Div([dbc.Row(cards, className="mb-4")])

@callback(
    Output("machine-graph", "figure"),
    Input("interval-component", "n_intervals"),
    Input("machine-stat", "value"),
    Input("plot-range", "start_date"),
    Input("plot-range", "end_date"),
    Input("machine-stat-smoothing", "value"),
)
def update_machine_stats(n, machine_stat, start_date, end_date, smooth_window):
    axis_labels = {"cpu_usage": "CPU Usage %",
                   "memory_usage": "Memory Usage[GB]",
                   "memory_percentage": "Memory Usage %",
                   "disk_usage": "Disk Usage[GB]",
                   "disk_percentage": "Disk Usage %",
                   "num_pids": "Process Count"}

    query = select(Health).where(Health.datetime > start_date).where(Health.datetime < end_date)
    with get_database_session() as session:
        df = pd.read_sql_query(query, session.connection())

    if smooth_window is not None and smooth_window > 1:
        smooth_window = int(round(smooth_window))
        df = df.rolling(f"{smooth_window}min", on="datetime", center=True, min_periods=0).mean()

    fig = px.line(df, x="datetime", y=machine_stat, title="Machine stats")
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text=axis_labels[machine_stat])

    return fig

@callback(
    Output("flow-throughput", "figure"),
    Output("flow-duration", "figure"),
    Input("interval-component", "n_intervals"),
)
def update_flow_stats(n):
    now = datetime.now()
    reference_time = now - timedelta(hours=72)
    query = ("SELECT flow_type, end_time AS hour, AVG(TIMEDIFF(end_time, start_time)) AS duration, "
             "COUNT(*) AS count, state "
             f"FROM flows WHERE end_time > '{reference_time}' "
             "GROUP BY HOUR(end_time), DAY(end_time), MONTH(end_time), YEAR(end_time), flow_type, state;")
    with get_database_session() as session:
        df = pd.read_sql_query(query, session.connection())
    # Fill missing entries (for hours where nothing ran)
    df.hour = [ts.floor("h") for ts in df.hour]
    dates = pd.date_range(reference_time, now, freq=timedelta(hours=1)).floor("h")
    additions = []
    for flow_type in df.flow_type.unique():
        for d in dates:
            for state in ["failed", "completed"]:
                if len(df.query("hour == @d and state == @state and flow_type == @flow_type")) == 0:
                    additions.append([flow_type, d, None, 0, state])
    df = pd.concat([df, pd.DataFrame(additions, columns=df.columns)], ignore_index=True)
    df.sort_values(["state", "hour"], inplace=True)

    # Extrapolate the last hourly window
    now_index = pd.Timestamp(now).floor("h")
    seconds_into_hour = (now - now_index).total_seconds()
    # But don't do it if it's really a lot of extrapolation
    if seconds_into_hour > 120:
        df = df.astype({"count": "float"})
        df.loc[df["hour"] == now_index, "count"] *= 3600 / (now - now_index).total_seconds()
    else:
        # Don't show 0 or an un-extrapolable small number, instead just hide the current hour for the first few
        # minutes
        df.loc[df["hour"] == now_index, "count"] = None

    fig_throughput = px.line(df, x="hour", y="count", color="flow_type", line_dash="state",
                             title="Flow throughput (current hour's throughput is extrapolated)")
    fig_throughput.update_xaxes(title_text="Time")
    fig_throughput.update_yaxes(title_text="Flow runs per hour")
    fig_duration = px.line(df[df["state"] == "completed"], x="hour", y="duration", color="flow_type",
                           title="Flow duration")
    fig_duration.update_xaxes(title_text="Time")
    fig_duration.update_yaxes(title_text="Average flow duration (s)")

    return fig_throughput, fig_duration
