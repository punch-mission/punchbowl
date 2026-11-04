import dash
import pandas as pd
import plotly.express as px
from dash import Input, Output, State, callback, dcc, html
from dash_bootstrap_templates import load_figure_template
from sqlalchemy import select

from punchbowl.auto.control.db import Flow
from punchbowl.auto.control.util import load_pipeline_configuration
from punchbowl.auto.flows.util import get_max_flows_to_schedule
from punchbowl.auto.monitor.app import get_database_session

load_figure_template(["bootstrap", "bootstrap_dark"])

REFRESH_RATE = 60  # seconds

dash.register_page(__name__)

LIGHT_THEME = 'bootstrap'
DARK_THEME = 'bootstrap_dark'

def layout():
    return html.Div([
        html.Details([
           html.Summary("Help"),
            html.Div([
                "This page gives a summary of what the servers are up to now.",
                html.Br(),
                'The "Running weight" section shows what flows are actively running on each server, showing each flow '
                'according to how much of the total runnable weight that flow type is occupying.',
                html.Br(),
                'The "Queue fullness" section shows how full each flow type\'s queue is on each server. Every time a '
                'scheduler flow runs, its goal is to fill up that flow type\'s queue, so that there are a target '
                'number of scheduled flows. Launcher then launches flows from these queues according to their '
                'priorities. "Fullness" is shown as a percentage because we queue lots of flows for the smaller flow '
                'types, but only one or two flows for the calibration product flow types (e.g. starfields). Empty '
                'queues indicate there\'s nothing ready to run for that flow type (i.e. we\'re waiting for inputs to '
                'be generated.)'
            ])
        ]),
        html.Div(id="work-status-holder",
                 style={
                     'display': 'flex',
                     'flexDirection': 'row',
                     'justifyContent': 'flex-start',
                     'alignItems': 'end',
                     'gap': '20px',
                 }),

        html.Hr(),

        html.Div(id="queue-status-holder",
                 style={
                     'display': 'flex',
                     'flexDirection': 'row',
                     'justifyContent': 'flex-start',
                     'alignItems': 'end',
                     'gap': '20px',
                 }),

        dcc.Interval(
            id="interval",
            interval=REFRESH_RATE * 1000,  # convert seconds to milliseconds
            n_intervals=0  # starts at 0, which fires the callback immediately on page load
        ),

    ], style={"margin": "10px"})


@callback(
    Output("queue-status-holder", "children"),
    Input("interval", "n_intervals"),
    State("color-mode-switch", "value"),
)
def update_queue_status(_: int, light_mode: bool) -> list:
    """
    Generate a 'queue status' plot for each server.

    Parameters
    ----------
    _ : int
        Triggers automatic refreshes
    light_mode : bool
        Whether we're in light mode

    Returns
    -------
    graphs : list
        A list of graph elements
    """
    config = load_pipeline_configuration()
    flow_to_host = {}
    for flow in config['flows']:
        flow_to_host[flow] = config['flows'][flow]['run-on']

    query = select(Flow).where(Flow.state == 'planned')

    with get_database_session() as session:
        df = pd.read_sql_query(query, session.connection())

    counts = df.groupby('flow_type').size().reset_index(name='count')
    hosts = [flow_to_host[flow] for flow in counts['flow_type']]
    counts['host'] = hosts

    full_percentage = []
    max_counts = []
    for flow_type, count in zip(counts['flow_type'], counts['count']):
        max_scheduled = get_max_flows_to_schedule(config, flow_type)
        max_counts.append(max_scheduled)
        full_percentage.append(count / max_scheduled * 100)

    counts['full_percentage'] = full_percentage
    counts['max_counts'] = max_counts

    figs = []
    for host in sorted(set(hosts)):
        fig = px.bar(counts[counts['host'] == host], x='flow_type', y='full_percentage',
                     title=f"Queue fullness for {host}",
                     hover_data={"flow_type": True, "count": True, "full_percentage": True, "max_counts": True},
                     template=LIGHT_THEME if light_mode else DARK_THEME)
        fig.update_yaxes(ticksuffix="%", title="Scheduling queue fullness")
        fig.update_xaxes(title="Flow type")
        figs.append(dcc.Graph(figure=fig))
    return figs


@callback(
    Output("work-status-holder", "children"),
    Input("interval", "n_intervals"),
    State("color-mode-switch", "value"),
)
def update_work_status(_: int, light_mode: bool) -> list:
    """
    Generate a plot of what's running on each server.

    Parameters
    ----------
    _ : int
        Triggers automatic refreshes
    light_mode : bool
        Whether we're in light mode

    Returns
    -------
    graph : list
        A list containing a single graph
    """
    config = load_pipeline_configuration(raw=True)
    flow_to_weight = {}
    flow_to_host = {}
    for flow in config['flows']:
        flow_to_host[flow] = config['flows'][flow]['run-on']
        flow_to_weight[flow] = config['flows'][flow]['launch_weight']

    max_weights = {}
    section = config['control']['launcher']
    for host in flow_to_host.values():
        if f'max_weight_running-{host}' in section:
            max_weights[host] = section[f'max_weight_running-{host}']
        else:
            max_weights[host] = section[f'max_weight_running']


    query = select(Flow).where(Flow.state.in_(['launched', 'running']))

    with get_database_session() as session:
        df = pd.read_sql_query(query, session.connection())

    counts = df.groupby('flow_type').size().reset_index(name='count')
    hosts = [flow_to_host[flow] for flow in counts['flow_type']]
    counts['host'] = hosts

    weights = [flow_to_weight[flow] for flow in counts['flow_type']]
    counts['weight'] = [weight * count
                        for weight, count in zip(weights, counts['count'])]
    counts['fractional_weight'] = [weight * count / max_weights[host] * 100
                                   for weight, count, host in zip(weights, counts['count'], counts['host'])]

    fig = px.bar(counts, x='host', y='fractional_weight', color='flow_type',
                 title='Running weight',
                 hover_data={"flow_type": True, "count": True, "weight": True},
                 template=LIGHT_THEME if light_mode else DARK_THEME,
                 category_orders={"host": sorted(set(counts['host'])),
                                  "flow_type": sorted(set(counts['flow_type']))})
    fig.update_yaxes(ticksuffix="%", title="Percent of max weight running")
    fig.update_xaxes(title="Server")
    figs = [dcc.Graph(figure=fig)]
    return figs
