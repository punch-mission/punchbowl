from datetime import date, timedelta

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import Input, Output, State, callback, dash_table, dcc, html
from sqlalchemy import func, or_, select

from punchbowl.auto.control.db import File, Flow
from punchbowl.auto.monitor.app import get_database_session

REFRESH_RATE = 60  # seconds

USABLE_COLUMNS = ["Level", "File type", "Flow type", "Observatory", "File version", "Polarization", "State", "Outlier"]
PAGE_SIZE = 100

dash.register_page(__name__)

def layout():
    return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        "Show in Table: ",
                        dcc.Checklist(
                            USABLE_COLUMNS,
                            ["File type", "Observatory"],
                            inline=True,
                            id="show-in-table",
                            inputStyle={"margin-left": "10px", "margin-right": "3px"},
                            persistence=True, persistence_type="memory",
                        ),
                    ]),
                    html.Div([
                        "Group by: ",
                        dcc.Checklist(
                            USABLE_COLUMNS,
                            ["File type", "Observatory"],
                            inline=True,
                            id="group-by",
                            inputStyle={"margin-left": "10px", "margin-right": "3px"},
                            persistence=True, persistence_type="memory",
                        ),
                    ]),
                ], width="auto", className="gx-5"),
                dbc.Col([
                    "date_obs: ",
                    dcc.DatePickerRange(id="table-date-obs",
                                    min_date_allowed=date(2025, 3, 1),
                                    start_date=date.today() - timedelta(days=31),
                                    end_date=date.today() + timedelta(days=1),
                                    initial_visible_month=date.today(),
                                    clearable=True,
                                    persistence=True, persistence_type="memory",
                                    ),
                    html.Div([
                        "Extra filters/shortcuts: ",
                        dcc.Checklist(
                            ["Existing files", "Failed files"],
                            ["Existing files"],
                            inline=True,
                            id="extra-filters",
                            inputStyle={"margin-left": "10px", "margin-right": "3px"},
                            persistence=True, persistence_type="memory",
                        ),
                    ]),
                ], width="auto", className="gx-5"),
                dbc.Col([
                    "date_created: ",
                    dcc.DatePickerRange(id="table-date-created",
                                    min_date_allowed=date(2025, 3, 1),
                                    start_date=None,
                                    end_date=None,
                                    initial_visible_month=date.today(),
                                    clearable=True,
                                    persistence=True, persistence_type="memory",
                                    ),
                    html.Div([
                        "Extra filters/shortcuts: ",
                        dcc.Checklist(
                            ["L0", "L1", "L2", "L3", "LQ"],
                            ["L1"],
                            inline=True,
                            id="extra-filters2",
                            inputStyle={"margin-left": "10px", "margin-right": "3px"},
                            persistence=True, persistence_type="memory",
                        ),
                    ]),
                ], width="auto", className="gx-5"),
        ]),
        dash_table.DataTable(id="files-table",
                             data=pd.DataFrame({name: [] for name in USABLE_COLUMNS + ["Count"]}).to_dict("records"),
                             columns=[{"name": col, "id": col.lower().replace(" ", "_")} for col in USABLE_COLUMNS + ["Count"]],
                             page_current=0,
                             page_size=PAGE_SIZE,
                             page_action="custom",

                             filter_action="custom",
                             filter_query="{file_type} = CR,P*",

                             sort_action="custom",
                             sort_mode="multi",
                             sort_by=[dict(column_id="file_type", direction="asc"), dict(column_id="observatory", direction="asc")],
                             style_table={"overflowX": "auto",
                                          "textAlign": "left"},
                             fill_width=False,
                             persistence=True, persistence_type="memory",
                             ),
        html.Hr(),
        dbc.Row([
            dbc.Col(width="auto", align="center", children=[
                html.Div([
                    "Color points by: ",
                    dcc.Dropdown(
                        id="graph-color",
                        options=["Nothing"] + USABLE_COLUMNS,
                        value="Observatory",
                        clearable=False,
                        style={"width" :"400px"},
                        persistence=True, persistence_type="memory",
                    ),
                ]),
            ]),
            dbc.Col(width="auto", align="center", children=[
                html.Div([
                    "Set shapes by: ",
                    dcc.Dropdown(
                        id="graph-shape",
                        options=["Nothing"] + USABLE_COLUMNS,
                        value="Nothing",
                        clearable=False,
                        style={"width" :"400px"},
                        persistence=True, persistence_type="memory",
                    ),
                ]),
            ]),
            dbc.Col(width="auto", align="center", children=[
                html.Div([
                    "X axis: ",
                    dcc.RadioItems(
                        id="graph-x-axis",
                        options=["date_obs", "date_created"],
                        value="date_obs",
                        inputStyle={"margin-left": "10px", "margin-right": "3px"},
                        inline=True,
                        persistence=True, persistence_type="memory",
                    ),
                ]),
                html.Div([
                    "Auto-refresh: ",
                    dcc.Checklist(
                        ["Enabled"],
                        ["Enabled"],
                        inline=True,
                        id="auto-refresh",
                        inputStyle={"margin-left": "10px", "margin-right": "3px"},
                        style={"display": "inline"},
                        persistence=True, persistence_type="memory",
                    ),
                    dcc.Button(
                        'Refresh',
                        id='manual-refresh',
                    ),
                ],),
            ]),
            dbc.Col(width="auto", align="center", children=[
                html.Div([
                    "Show 1 point per row every",
                    dcc.Input(
                        id="graph_point_time_window",
                        type="number",
                        debounce=1,
                        min=0,
                        step=10,
                        value=60,
                        style={"display": "block"},
                        persistence=True, persistence_type="memory",
                    ),
                    "minutes (set to 0 for all points)",
                ]),
            ]),
        ]),
        dcc.Graph(id="file-graph", style={"height": "400"}),
        dcc.Interval(
            id="interval-component",
            interval=REFRESH_RATE * 1000,  # in milliseconds
            n_intervals=0),
    ], style={"margin": "10px"})


@callback(
    Output("interval-component", "interval"),
    Input("auto-refresh", "value"),
)
def toggle_auto_refresh(auto_refresh_settings):
    if "Enabled" in auto_refresh_settings:
        return REFRESH_RATE * 1000
    return 99999999999999999


# It's important that e.g. '>=' appears before '>' or '=', otherwise the latter ones will match '>='
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
    for input_operators, py_method in operators:
        for input_operator in input_operators:
            if input_operator in filter_part:
                name_part, value = filter_part.split(input_operator, 1)
                column_name = name_part[name_part.find("{") + 1: name_part.rfind("}")]

                value = value.strip()
                v0 = value[0]
                if (v0 == value[-1] and v0 in ("'", '"', "`")):
                    value = value[1: -1].replace("\\" + v0, v0)
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        if input_operator in ["=", "eq", "contains "]:
                            values = value.split(",")
                            py_method = None
                            parsed_values = []
                            for v in values:
                                v = v.strip()
                                if '*' in v:
                                    v = v.replace('*', '%')
                                    py_method = "like"
                                parsed_values.append(v)
                            if len(parsed_values) == 1:
                                parsed_values = parsed_values[0]
                            value = parsed_values
                # word operators need spaces after them in the filter string,
                # but we don't want these later
                return column_name, input_operator.strip(), value, py_method
    return [None] * 4


def construct_base_query(columns, filter, extra_filters, extra_filters2, include_count, date_obs_start,
                 date_obs_end, date_created_start, date_created_end):
    # Build the parts of a query common to the table and graph
    cols = []
    join_flow_type = False
    for col in columns:
        col = col.lower().replace(" ", "_")
        if col == "flow_type":
            cols.append(getattr(Flow, col))
            join_flow_type = True
        else:
            cols.append(getattr(File, col))
    if include_count:
        cols += [func.count(File.file_id).label("count")]

    query = select(*cols)

    if join_flow_type:
        query = query.join(Flow, Flow.flow_id == File.processing_flow)

    for filter_part in filter.split(" && "):
        col_name, operator, filter_value, py_method = split_filter_part(filter_part)
        if col_name is not None:
            column = getattr(File, col_name)
            method = getattr(column, py_method)
            if isinstance(filter_value, list):
                conditions = [method(value) for value in filter_value]
                query = query.where(or_(*conditions))
            else:
                query = query.where(method(filter_value))

    extra_filter_state = []
    if "Existing files" in extra_filters:
        extra_filter_state.extend(["created", "progressed"])
    if "Failed files" in extra_filters:
        extra_filter_state.append("failed")
    if extra_filter_state:
        query = query.where(File.state.in_(extra_filter_state))

    extra_filter_level = []
    for value in extra_filters2:
        if value[0] == "L":
            extra_filter_level.append(value[1:])
    if extra_filter_level:
        query = query.where(File.level.in_(extra_filter_level))

    if date_obs_start:
        query = query.where(File.date_obs >= date_obs_start)
    if date_obs_end:
        query = query.where(File.date_obs <= date_obs_end)
    if date_created_start:
        query = query.where(File.date_created >= date_created_start)
    if date_created_end:
        query = query.where(File.date_created <= date_created_end)
    return query


@callback(
    Output("files-table", "data"),
    Input("show-in-table", "value"),
    Input("group-by", "value"),
    Input("interval-component", "n_intervals"),
    Input("files-table", "page_current"),
    Input("files-table", "page_size"),
    Input("files-table", "sort_by"),
    Input("files-table", "filter_query"),
    Input("extra-filters", "value"),
    Input("extra-filters2", "value"),
    Input("table-date-obs", "start_date"),
    Input("table-date-obs", "end_date"),
    Input("table-date-created", "start_date"),
    Input("table-date-created", "end_date"),
    Input("manual-refresh", "n_clicks"),
)
def update_table(show_in_table, group_by, n, page_current, page_size, sort_by, filter, extra_filters, extra_filters2,
                 date_obs_start, date_obs_end, date_created_start, date_created_end, refresh_nclicks):
    query = construct_base_query(group_by, filter, extra_filters, extra_filters2, True, date_obs_start,
             date_obs_end, date_created_start, date_created_end)

    for col in sort_by:
        sort_column = getattr(File, col["column_id"])
        if col["direction"] == "asc":
            sort_column = sort_column.asc()
        else:
            sort_column = sort_column.desc()
        query = query.order_by(sort_column)

    for col in group_by:
        query = query.group_by(col.lower().replace(" ", "_"))

    with get_database_session() as session:
        dff = pd.read_sql_query(query, session.connection())

    # It turns out, surprisingly, that for at least some queries, if we put the limit in the MySQL query, a different
    # (worse) index gets chosen and the DB query takes orders of magnitude longer. Since we're GROUP BY-ing anyway
    # and so the number of rows is always small, we cut rows as needed in Python instead.
    if page_current * page_size < len(dff):
        dff = dff[page_current * page_size:]
    dff = dff[:page_size]

    if len(dff) == 0:
        dff.loc[-1] = ["-"] * (len(dff.columns) - 1) + ["No files found for these criteria"]

    return dff.to_dict("records")


@callback(
    Output("files-table", "columns"),
    Output("group-by", "value"),
    Output("group-by", "options"),
    Output("files-table", "sort_by"),
    Output("files-table", "filter_query"),
    Input("show-in-table", "value"),
    State("group-by", "value"),
    State("files-table", "sort_by"),
    State("files-table", "filter_query"),
)
def update_visible_columns(show_in_table, group_by_selection, current_sort, current_filter):
    # When the "show in table" checkboxes change, update the "group by" options and the table columns
    table_columns = [{"name": col, "id": col.lower().replace(" ", "_")} for col in USABLE_COLUMNS if col in show_in_table]
    table_columns.append({"name": "Count", "id": "count"})
    current_col_ids = [col["id"] for col in table_columns]

    group_by_options = [{"value": c, "label": c, "disabled": c not in show_in_table} for c in USABLE_COLUMNS]

    # Un-select any group-by checkboxes that won't be selectable anymore
    group_by_selection = [c for c in group_by_selection if c in show_in_table]

    # Clear sorts for non-shown columns
    current_sort = [col for col in current_sort if col["column_id"] in current_col_ids]

    new_filter_parts = []
    for filter_part in current_filter.split(" && "):
        col_name, operator, filter_value, py_method = split_filter_part(filter_part)
        if col_name in current_col_ids:
            new_filter_parts.append(filter_part)
    current_filter = " && ".join(new_filter_parts)

    return table_columns, group_by_selection, group_by_options, current_sort, current_filter


def make_y_axis_labels(dff):
    # Generate the label strings for the plot y axis
    joinables = []
    columns = list(dff.columns)
    if "file_type" in columns and "observatory" in columns:
        # If we're grouping by these columns, special-case it to show e.g. "CR2" instead of "CR 2" or "2 CR" or whatever
        joinables.append(dff["file_type"] + dff["observatory"])
        columns.remove("file_type")
        columns.remove("observatory")

    # These are x-axis values
    if "date_obs" in columns:
        columns.remove("date_obs")
    if "date_created" in columns:
        columns.remove("date_created")

    for column in columns:
        if len(dff[column].unique()) > 1:
            joinables.append(dff[column])

    if len(joinables) == 0:
        # We need something
        joinables = [dff[columns[0]]]

    keys = joinables[0].copy()
    for col in joinables[1:]:
        keys += " " + col.astype("string")
    return keys


@callback(
    Output("file-graph", "figure"),
    Output("file-graph", "style"),
    Input("interval-component", "n_intervals"),
    Input("group-by", "value"),
    Input("files-table", "filter_query"),
    Input("files-table", "sort_by"),
    Input("graph-color", "value"),
    Input("graph-shape", "value"),
    Input("extra-filters", "value"),
    Input("extra-filters2", "value"),
    Input("graph-x-axis", "value"),
    Input("table-date-obs", "start_date"),
    Input("table-date-obs", "end_date"),
    Input("table-date-created", "start_date"),
    Input("table-date-created", "end_date"),
    Input("graph_point_time_window", "value"),
    Input("manual-refresh", "n_clicks"),
)
def update_file_graph(n, group_by, filter, sort_by, color_key, shape_key, extra_filters, extra_filters2, graph_x_axis,
                      date_obs_start, date_obs_end, date_created_start, date_created_end, graph_point_time_window,
                      refresh_nclicks):
    group_by = [col.lower().replace(" ", "_") for col in group_by]
    color_key = color_key.lower().replace(" ", "_")
    shape_key = shape_key.lower().replace(" ", "_")

    query_cols = group_by + ["date_created", "date_obs"]

    # Make sure the color and shape columns are in the query. If they're not in the "group by" selection, track that so
    # we don't let these values become part of the y axis labels
    exclude_color_from_keys = False
    if color_key not in query_cols and color_key != "nothing":
        exclude_color_from_keys = True
        query_cols.append(color_key)

    exclude_shape_from_keys = False
    if shape_key not in query_cols and shape_key != "nothing":
        exclude_shape_from_keys = True
        query_cols.append(shape_key)

    query = construct_base_query(query_cols, filter, extra_filters, extra_filters2, False, date_obs_start,
                 date_obs_end, date_created_start, date_created_end)
    if graph_point_time_window > 0:
        col = getattr(File, graph_x_axis)
        y_axis_cols = [getattr(File, c) for c in group_by]
        query = query.order_by(col).group_by(*y_axis_cols, func.round(func.unix_timestamp(col) / (graph_point_time_window * 60)))
    with get_database_session() as session:
        dff = pd.read_sql_query(query, session.connection())

    # Extract the data that sets shape and color, and then remove if necessary
    if color_key != "nothing":
        color_data = dff[color_key]
        if exclude_color_from_keys:
            dff = dff.drop(color_key, axis=1)
    if shape_key != "nothing":
        shape_data = dff[shape_key]
        if exclude_shape_from_keys:
            dff = dff.drop(shape_key, axis=1)

    y_axis_labels = make_y_axis_labels(dff)

    # Generate a minimal dataframe to pass to the graph
    columns = [y_axis_labels, dff["date_created"], dff["date_obs"]]
    keys = ["name", "date_created", "date_obs"]
    if color_key != "nothing":
        columns.append(color_data)
        keys.append(color_key)
    if shape_key != "nothing":
        columns.append(shape_data)
        keys.append(shape_key)
    plot_df = pd.concat(columns, axis=1, keys=keys).dropna()

    # Make sure groups appear on the y axis in the same order as in the table
    category_orders = {}
    if sort_by:
        sort_by = [col for col in sort_by if col["column_id"] in dff.columns]
        sort_columns = [col["column_id"] for col in sort_by]
        sort_ascending = [col["direction"] == "asc" for col in sort_by]
        # Group the data and sort the group labels
        label_order_df = dff.groupby(group_by, as_index=False).first().sort_values(sort_columns, ascending=sort_ascending)
        # Make the corresponding axis labels, in the same order
        labels = list(make_y_axis_labels(label_order_df))
        category_orders["name"] = labels
    else:
        labels = plot_df["name"].unique()

    # Sort the legend entries
    if color_key != "nothing":
        category_orders[color_key] = sorted(plot_df[color_key].unique())
    if shape_key != "nothing":
        category_orders[shape_key] = sorted(plot_df[shape_key].unique())

    fig = px.scatter(plot_df, x=graph_x_axis, y="name",
                     color=color_key if color_key != "nothing" else None,
                     symbol=shape_key if shape_key != "nothing" else None,
                     category_orders=category_orders,
                     hover_data={"date_created": "|%Y-%m-%d %H:%M:%S", "date_obs": "|%Y-%m-%d %H:%M:%S"},
                     color_discrete_sequence=px.colors.qualitative.D3)
    fig.update_xaxes(title_text=graph_x_axis)

    # Adjust the plot height
    new_style = {"height": f"{150 + len(labels) * 30}px", "min-height": "400px"}

    return fig, new_style
