import copy
from collections.abc import Callable

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from ndcube import NDCube
from thuban.pointing import refine_pointing

from punchbowl.data.wcs import calculate_celestial_wcs_from_helio, calculate_helio_wcs_from_celestial
from punchbowl.prefect import punch_task


@punch_task
def align_task(data_object: NDCube, distortion_path: str | None, mask: Callable | None = None,
               pointing_shift: tuple[float, float] = (0, 0)) -> NDCube:
    """
    Determine the pointing of the image and updates the metadata appropriately.

    Parameters
    ----------
    data_object : NDCube
        data object to align
    mask : Callable | None
        function accepting coordinates and returning them only if they are not masked out
    distortion_path: str | None
        path to a distortion model
    pointing_shift: tuple[float, float]
        offset to pre-apply to the pointing to account for boresight to satellite-x shift

    Returns
    -------
    NDCube
        a modified version of the input with the WCS more accurately determined

    """
    celestial_input = calculate_celestial_wcs_from_helio(copy.deepcopy(data_object.wcs),
                                                         data_object.meta.astropy_time,
                                                         data_object.data.shape)
    refining_data = data_object.data.copy()
    refining_data[np.isinf(refining_data)] = 0
    refining_data[np.isnan(refining_data)] = 0

    if distortion_path:
        with fits.open(distortion_path) as distortion_hdul:
            distortion_wcs = WCS(distortion_hdul[0].header, distortion_hdul)
        celestial_input.cpdis1 = distortion_wcs.cpdis1
        celestial_input.cpdis2 = distortion_wcs.cpdis2
        celestial_input.wcs.set_pv([(2, 1, distortion_wcs.wcs.get_pv()[0][-1])])
        celestial_input.wcs.cdelt = distortion_wcs.wcs.cdelt

    celestial_input.wcs.crval = (celestial_input.wcs.crval[0] - pointing_shift[0] * celestial_input.wcs.cdelt[0],
                                 celestial_input.wcs.crval[1] - pointing_shift[1] * celestial_input.wcs.cdelt[1])

    celestial_output, observed_coords, result, trial_num = refine_pointing(refining_data, celestial_input,
                                                                        num_stars=10,
                                                                        detection_threshold=5,
                                                                        background_height=16,
                                                                        background_width=16,
                                                                        max_trials=5,
                                                                        dimmest_magnitude=6.5,
                                                                        chisqr_threshold=0.1,
                                                                        edge=100,
                                                                        mask=mask,
                                                                        max_error=150,
                                                                        method="least_squares")

    recovered_wcs, _ = calculate_helio_wcs_from_celestial(celestial_output,
                                                       data_object.meta.astropy_time,
                                                       data_object.data.shape)

    if distortion_path:
        with fits.open(distortion_path) as distortion_hdul:
            distortion_wcs = WCS(distortion_hdul[0].header, distortion_hdul)
        recovered_wcs.cpdis1 = distortion_wcs.cpdis1
        recovered_wcs.cpdis2 = distortion_wcs.cpdis2

    output = NDCube(data=data_object.data,
                    wcs=recovered_wcs,
                    uncertainty=data_object.uncertainty,
                    unit=data_object.unit,
                    meta=data_object.meta)
    output.meta.history.add_now("LEVEL1-Align", "alignment done")
    return output
