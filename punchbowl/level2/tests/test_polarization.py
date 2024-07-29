import pytest
from ndcube import NDCube
from prefect.logging import disable_run_logger
from prefect.testing.utilities import prefect_test_harness

from punchbowl.data.tests.test_io import sample_ndcube
from punchbowl.level2.polarization import resolve_polarization, resolve_polarization_task


@pytest.fixture
def sample_data_triplet(sample_ndcube):
    """
    Generate a list of sample PUNCHData objects for testing polarization resolving
    """

    sample_pd1 = sample_ndcube(shape=(50, 50))
    sample_pd2 = sample_ndcube(shape=(50, 50))
    sample_pd3 = sample_ndcube(shape=(50, 50))
    return [sample_pd1, sample_pd2, sample_pd3]


def test_resolve_polarization(sample_data_triplet):
    """
        Test the resolve polarization function directly
    """

    output_punchdata_list = resolve_polarization(sample_data_triplet)

    assert all(isinstance(output_punchdata, NDCube) for output_punchdata in output_punchdata_list)


@pytest.mark.prefect_test
def test_resolve_polarization_task(sample_data_triplet):
    """
    Test the resolve polarization prefect flow using a test harness
    """

    with disable_run_logger():
        output_punchdata_list = resolve_polarization_task.fn(sample_data_triplet)

    assert all(isinstance(output_punchdata, NDCube) for output_punchdata in output_punchdata_list)
