from datetime import datetime

import pytest
from pytest_mock_resources import create_mysql_fixture

from punchbowl.auto.control.db import Base, Quicklook


def session_fn(session):
    row = Quicklook(
        day=datetime(2026, 1, 1),
        level="3",
        code="CAM",
        movie_made=True,
        image_made=True,
        movie_nfile=45)
    session.add(row)

    row = Quicklook(
        day=datetime(2026, 1, 1),
        level="3",
        code="PAM",
        movie_made=False,
        image_made=False,
        movie_nfile=45,
    )
    session.add(row)


db = create_mysql_fixture(Base, session_fn, session=True)
db_empty = create_mysql_fixture(Base, session=True)


def test_quicklook_query(db):
    result = db.query(Quicklook).filter_by(code="CAM").one()
    assert result.movie_made is True
    assert result.image_made is True
    assert result.movie_nfile == 45

    result = db.query(Quicklook).filter_by(code="PAM").one()
    assert result.movie_made is False
    assert result.image_made is False


def test_quicklook_insert(db_empty):
    row = Quicklook(day=datetime(2026, 1, 2), level="3", code="CAM")
    db_empty.add(row)
    db_empty.commit()
    assert db_empty.query(Quicklook).count() == 1
