from contextlib import contextmanager

import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, page_container, page_registry
from sqlalchemy.orm import Session

from punchbowl.pipe.control.util import get_database_session as _get_database_session

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
    app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], use_pages=True, pages_folder="pages")

    app.layout = html.Div([
        html.H1("PUNCHPipe dashboard"),
        html.Div([
            dcc.Link(f"{page['name']}", href=page["relative_path"], style={"margin": "10px"})
            for page in page_registry.values()
        ]),
        page_container,
    ])

    return app


if __name__ == "app":
    app = create_app()
    server = app.server
