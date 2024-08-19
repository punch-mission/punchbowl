from ndcube import NDCube
from prefect import get_run_logger, task
from thuban.pointing import refine_pointing

from punchbowl.data.wcs import calculate_celestial_wcs_from_helio, calculate_helio_wcs_from_celestial


@task
def align_task(data_object: NDCube) -> NDCube:
    """
    Determine the pointing of the image and updates the metadata appropriately.

    Parameters
    ----------
    data_object : PUNCHData
        data object to align

    Returns
    -------
    PUNCHData
        a modified version of the input with the WCS more accurately determined

    """
    logger = get_run_logger()
    logger.info("alignment started")
    celestial_input = calculate_celestial_wcs_from_helio(data_object.wcs,
                                                         data_object.meta.astropy_time,
                                                         data_object.data.shape)
    celestial_output, observed_coords, result, trial_num = refine_pointing(data_object.data, celestial_input,
                                                                        num_stars=10,
                                                                        detection_threshold=10,
                                                                        background_height=8, background_width=8,
                                                                        max_trials=32,
                                                                        dimmest_magnitude=5.0,
                                                                        chisqr_threshold=0.05,
                                                                        method="least_squares")
    recovered_wcs, _ = calculate_helio_wcs_from_celestial(celestial_output,
                                                       data_object.meta.astropy_time,
                                                       data_object.data.shape)
    logger.info("alignment finished")
    output = NDCube(data=data_object.data,
                    wcs=recovered_wcs,
                    uncertainty=data_object.uncertainty,
                    unit=data_object.unit,
                    meta=data_object.meta)
    output.meta.history.add_now("LEVEL1-Align", "alignment done")
    return output
