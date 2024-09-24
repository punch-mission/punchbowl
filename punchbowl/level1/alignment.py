import copy
from collections.abc import Callable

import numpy as np
from astropy.wcs import WCS
from ndcube import NDCube
from prefect import get_run_logger, task
from thuban.pointing import refine_pointing

from punchbowl.data.wcs import calculate_celestial_wcs_from_helio, calculate_helio_wcs_from_celestial


@task
def align_task(data_object: NDCube, mask: Callable | None = None) -> NDCube:
    """
    Determine the pointing of the image and updates the metadata appropriately.

    Parameters
    ----------
    data_object : PUNCHData
        data object to align
    mask : Callable | None
        function accepting coordinates and returning them only if they are not masked out

    Returns
    -------
    PUNCHData
        a modified version of the input with the WCS more accurately determined

    """
    logger = get_run_logger()
    logger.info("alignment started")
    saved_wcs = copy.deepcopy(data_object.wcs)
    celestial_input = calculate_celestial_wcs_from_helio(copy.deepcopy(data_object.wcs),
                                                         data_object.meta.astropy_time,
                                                         data_object.data.shape)
    refining_data = data_object.data.copy()
    refining_data[np.isinf(refining_data)] = 0
    refining_data[np.isnan(refining_data)] = 0
    celestial_output, observed_coords, result, trial_num = refine_pointing(refining_data, celestial_input,
                                                                        num_stars=10,
                                                                        detection_threshold=10,
                                                                        background_height=8, background_width=8,
                                                                        max_trials=32,
                                                                        dimmest_magnitude=6.0,
                                                                        chisqr_threshold=0.1,
                                                                        sigma=0.1,
                                                                        edge=100,
                                                                        mask=mask,
                                                                        method="least_squares")

    recovered_wcs, _ = calculate_helio_wcs_from_celestial(celestial_output,
                                                       data_object.meta.astropy_time,
                                                       data_object.data.shape)
    recovered_wcs.cpdis1 = data_object.wcs.cpdis1
    recovered_wcs.cpdis2 = data_object.wcs.cpdis2

    saved_wcs_fits = saved_wcs.to_fits()
    saved_header = saved_wcs_fits[0].header
    recovered_header = recovered_wcs.to_fits()[0].header
    for key in recovered_header:
        saved_header[key] = recovered_header[key]
    recovered_wcs = WCS(saved_header, saved_wcs_fits)

    logger.info("alignment finished")
    output = NDCube(data=data_object.data,
                    wcs=recovered_wcs,
                    uncertainty=data_object.uncertainty,
                    unit=data_object.unit,
                    meta=data_object.meta)
    output.meta.history.add_now("LEVEL1-Align", "alignment done")
    return output
