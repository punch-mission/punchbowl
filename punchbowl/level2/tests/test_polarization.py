import pytest
from ndcube import NDCube
from prefect.logging import disable_run_logger
from prefect.testing.utilities import prefect_test_harness

from punchbowl.data.tests.test_io import sample_ndcube
from punchbowl.level2.polarization import resolve_polarization, resolve_polarization_task


@pytest.fixture
def sample_data_triplet(sample_ndcube):
    """
    Generate a list of sample PUNCH data objects for testing polarization resolving
    """

    polar_angles = [-60, 0, 60]
    sample_data = []

    for angle in polar_angles:
        sample_pd = sample_ndcube(shape=(50, 50))
        sample_pd.meta['POLAR'] = angle
        sample_data.append(sample_pd)

    return sample_data


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
