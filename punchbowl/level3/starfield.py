import remove_starfield
from remove_starfield.reducers import SkewGaussianReducer, GaussianReducer
import glob
import matplotlib.pyplot as plt
import os

import warnings
from typing import List, Optional
from datetime import datetime

import numpy as np
from prefect import get_run_logger, task

from punchbowl.data import PUNCHData
from punchbowl.exceptions import InvalidDataError


def starfield_background(
    data_list: List[str],
    outpath: str = ".") -> PUNCHData:

    """Creates a background starfield map from a series of PUNCH images over
    a long period of time.

    Creates a background starfield map


    Parameters
    ----------
    outpath
    data_list :
        list of filenames to use



    Returns
    -------
    return output_PUNCHobject : ['punchbowl.data.PUNCHData']
        returns an array of the same dimensions as the x and y dimensions of
        the input array
    """
    logger = get_run_logger()
    logger.info("construct_starfield_background started")

    # create an empty array to fill with data
    #   open the first file in the list to ge the shape of the file
    if len(data_list) == 0:
        raise ValueError("data_list cannot be empty")

    # # todo: replace in favor of using object directly
    # output = PUNCHData.from_fits(data_list[0])

    #TODO: get RA and DEC bounds from first and last files

    ifiles = sorted(glob.glob(data_list + '*.fits'))
    starfield = remove_starfield.build_starfield_estimate(
        ifiles, attribution=True, frame_count=True,
        reducer=GaussianReducer(n_sigma=5), map_scale=0.01,
        processor=remove_starfield.ImageProcessor(),  # wcs_key='A'),
        dec_bounds=(0, 35), ra_bounds=(100, 160), target_mem_usage=1000)

    #TODO: make something like if write = true
    plt.figure(figsize=(15, 5))
    starfield.plot(pmin=5)
    plt.savefig(outpath + 'generated_starfield.png', dpi=300)
    starfield.save(outpath + 'generated_starfield.h5')

    # create an output PUNCHdata object
    output = PUNCHData(starfield.starfield, wcs=starfield.wcs)

    logger.info("construct_starfield_background finished")
    output.meta.history.add_now("LEVEL3-starfield_background", "constructed starfield model")

    return output


## Input->NDcube of MZP
## Outputs of Level3 is BpB
# 2 funcs -> starfield generation; starfield subtraction

def subtract_starfield_background(data_object: PUNCHData, starfield_background_model_array: np.ndarray) -> PUNCHData:
    # check dimensions match
    if data_object.data.shape != starfield_background_model_array.shape:
        raise InvalidDataError(
            "starfield_background_subtraction expects the data_object and"
            "starfield_background arrays to have the same dimensions."
            f"data_array dims: {data_object.data.shape} and f_background_model dims: {starfield_background_model_array.shape}"
        )

    starfield_subtracted_data = data_object.data - starfield_background_model_array

    return data_object.duplicate_with_updates(data=starfield_subtracted_data)

# # Here we use different options for WISPRImageProcessor---this is the second use case our processor supports
# for ifile in [ifiles[500]]:
#     subtracted = starfield.subtract_from_image(ifile, processor=remove_starfield.ImageProcessor())
#     subtracted.save(outpath+f'{os.path.splitext(os.path.basename(ifile))[0]}_starfield_removed.fits', overwrite=True)