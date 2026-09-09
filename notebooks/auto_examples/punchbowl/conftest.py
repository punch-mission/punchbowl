import pytest
from prefect.testing.utilities import prefect_test_harness


@pytest.fixture(autouse=True, scope="session")
def prefect_test_fixture() -> None:
    """Test harness for prefect."""
    with prefect_test_harness(server_startup_timeout=60):
        yield
