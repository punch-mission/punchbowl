import pytest
from pytest_mock_resources import MysqlConfig


@pytest.fixture(scope='session')
def pmr_mysql_config():
    return MysqlConfig(image='mariadb:latest')
