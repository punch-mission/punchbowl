import pytest
from ndcube import NDCube
from prefect.logging import disable_run_logger
from prefect.testing.utilities import prefect_test_harness

from punchbowl.level2.polarization import resolve_polarization, resolve_polarization_task
from punchbowl.tests.test_data import sample_punchdata, sample_punchdata_triplet


def test_resolve_polarization(sample_punchdata_triplet):
    """
        Test the resolve polarization function directly
    """

    output_punchdata_list = resolve_polarization(sample_punchdata_triplet)

    assert all(isinstance(output_punchdata, NDCube) for output_punchdata in output_punchdata_list)


@pytest.mark.prefect_test
def test_resolve_polarization_task(sample_punchdata_triplet):
    """
    Test the resolve polarization prefect flow using a test harness
    """

    with disable_run_logger():
        output_punchdata_list = resolve_polarization_task.fn(sample_punchdata_triplet)

    assert all(isinstance(output_punchdata, NDCube) for output_punchdata in output_punchdata_list)
