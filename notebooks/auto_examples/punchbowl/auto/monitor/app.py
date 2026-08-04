from contextlib import contextmanager

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, clientside_callback, dcc, html, page_container, page_registry
from sqlalchemy.orm import Session

from punchbowl.auto.control.util import get_database_session as _get_database_session

# We'll keep and engine to keep a DB connection pool for the monitor, instead of making a new connection in each
# individual function every time the page loads or refreshes.
session, engine = _get_database_session(get_engine=True, engine_kwargs=dict(pool_recycle=6*3600))
session.close()


@contextmanager
def get_database_session():
    session = Session(engine)
    try:
        yield session
    finally:
        session.close()


def create_app():
    dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
    app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP, dbc_css, dbc.icons.FONT_AWESOME], use_pages=True, pages_folder="pages")

    # Dark mode toggle
    color_mode_switch = html.Div(
        [
            dbc.Label(className="fa fa-moon", html_for="color-mode-switch"),
            dbc.Switch(id="color-mode-switch", value=False, className="d-inline-block ms-1", persistence=True),
            dbc.Label(className="fa fa-sun", html_for="color-mode-switch"),
        ],
        style={"position": "absolute", "right": "15px", "top": "15px"}
    )

    # This re-themes almost everything when dark mode is toggled
    clientside_callback(
        """
        (switchOn) => {
           document.documentElement.setAttribute('data-bs-theme', switchOn ? 'light' : 'dark');
           return window.dash_clientside.no_update
        }
        """,
        Output("color-mode-switch", "id"),
        Input("color-mode-switch", "value"),
    )

    app.layout = html.Div([
        html.H1("PUNCHPipe dashboard"),
        color_mode_switch,
        html.Div([
            dcc.Link(f"{page['name']}", href=page["relative_path"], style={"margin": "10px"})
            for page in page_registry.values()
        ]),
        page_container,
    ], className="dbc")

    return app


if __name__ == "app":
    app = create_app()
    server = app.server
