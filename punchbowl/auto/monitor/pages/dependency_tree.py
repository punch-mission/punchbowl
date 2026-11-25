import io
import base64
from contextlib import nullcontext

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dcc, html
from dash_bootstrap_templates import load_figure_template
from dateutil.parser import parse as parse_datetime_str
from matplotlib import pyplot as plt

from punchbowl.auto import pipeline_simulator
from punchbowl.auto.control.util import load_pipeline_configuration
from punchbowl.auto.pipeline_simulator import File, build_tree, mark_file_existence, plot_tree

load_figure_template(["bootstrap", "bootstrap_dark"])

REFRESH_RATE = 60  # seconds

dash.register_page(__name__)

LIGHT_THEME = "bootstrap"
DARK_THEME = "bootstrap_dark"

def layout():
    return html.Div([
        html.Details([
           html.Summary("Help"),
            html.Div([
                'This page generates a file dependency tree (the "Christmas tree" plot).',
                html.Br(),
                "You pick a file type and a date/time. The code, knowing our imaging schedule, finds the closest "
                "image to your time of that type, and then builds the full dependency tree. It uses the database to "
                "check which files exist. It loads the current pipeline configuration for all necessary details.",
                html.Br(),
                "Note that the tree is built using a quick-and-dirty imitation of the pipeline, which doesn't fully "
                "track reality. For instance, for background products, the real pipeline will widen its window if "
                "input products are missing, but this simulator doesn't match that behavior.",
            ]),
        ]),

        html.Div([
            "Target file type: ",
            dbc.RadioItems(options=["PAM", "PTM", "PIM", "CAM", "CTM", "CIM", "L2_PTM", "L2_CTM"],
                           value="CAM",
                           id="select-kind",
                           inline=True),
            "Comma-separated date-times: ",
            dbc.Input(
                id="dates",
                type="text",
                placeholder="Comma-separated date-times",
                debounce=True,
                style={"width": "250px"},
            ),
        ], className="d-flex align-items-center gap-3"),

        html.Div([
            dbc.Checkbox(id="dark-plot-mode", label="Plot in dark mode", value=True),
            dbc.Checkbox(id="mark-existence", label="Mark file existence", value=True),
        ], className="d-flex align-items-center gap-3"),

        html.Hr(),

        dcc.Loading(
            html.Div(id="plot-holder"),
            overlay_style={"visibility": "visible", "filter": "opacity(25%)"},
        ),

    ], style={"margin": "10px"})


# A dark style with the default colors adjusted
DARK_STYLE = {
    "lines.color": "white",
    "patch.edgecolor": "white",

    "text.color": "white",

    "axes.facecolor": "black",
    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "axes.prop_cycle": "cycler('color', ['1b5784', 'ff9a40', '0f8d0f', 'a81010'])",

    "xtick.color": "white",
    "ytick.color": "white",

    "grid.color": "white",

    "figure.facecolor": "black",
    "figure.edgecolor": "black",

    "savefig.facecolor": "black",
    "savefig.edgecolor": "black",
}


@callback(
    Output("plot-holder", "children"),
    Input("dates", "value"),
    Input("select-kind", "value"),
    Input("dark-plot-mode", "value"),
    Input("mark-existence", "value"),
    prevent_initial_call=True,
)
def generate_plot(dates: str, kind: str, dark_mode: bool, mark_existence: bool) -> html.Img:
    """
    Generate a dependency-tree plot for the dashboard.

    Parameters
    ----------
    dates : str
        Comma-separated date/times
    kind : str
        The kind of file for the tree top
    dark_mode : bool
        Whether to plot in dark mode
    mark_existence : bool
        Whether to mark files' existence

    Returns
    -------
    html.Img
        The <img> element to display

    """
    print(dates, kind)
    config = load_pipeline_configuration()
    File.set_pipeline_config(config)
    dates = [parse_datetime_str(date) for date in dates.split(",")]
    top_node_kind = getattr(pipeline_simulator, kind.upper())
    top = [top_node_kind().get_file_closest_to(date) for date in dates]

    tree_data = build_tree(top)

    if mark_existence:
        mark_file_existence(tree_data)

    with plt.style.context(DARK_STYLE) if dark_mode else nullcontext():
        plot_tree(tree_data, dark=dark_mode)

        buffer = io.BytesIO()
        plt.subplots_adjust(.08, .1, .99, .99)
        plt.savefig(buffer, format="png")
        plt.close()
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    src = f"data:image/png;base64,{encoded}"

    return html.Img(src=src, style={"max-width": "100%", "height": "auto", "display": "block"})
