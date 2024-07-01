import solpolpy as spp
import numpy as np

from prefect import get_run_logger, task
from ndcube import NDCollection
from punchbowl.data import PUNCHData, NormalizedMetadata


def convert2bpb(
        input_data: PUNCHData) -> PUNCHData:
    """Creates a brightness (B) and polarized brightness (pB)
    from an input of MZP triplet.

    Parameters
    ----------
    input_data


    Returns
    -------
    return output_PUNCHobject : ['punchbowl.data.PUNCHData']
        returns an array of the same dimensions as the x and y dimensions of
        the input array
    """

    logger = get_run_logger()
    logger.info("convert2bpb started")

    # Unpack data into a NDCollection object
    data_collection = NDCollection([("M", input_data[0, :, :]), ("Z", input_data[1, :, :]), ("P", input_data[2, :, :])], aligned_axes='all')

    resolved_data_collection = spp.resolve(data_collection, "BpB", imax_effect=False)

    # Repack data
    data_list = []
    wcs_list = []

    for key in resolved_data_collection:
        data_list.append(resolved_data_collection[key].data)
        wcs_list.append(resolved_data_collection[key].wcs)

    # Remove alpha channel
    data_list.pop()
    wcs_list.pop()

    # Repack into a PUNCHData object
    new_data = np.stack(data_list, axis=0)
    new_wcs = input_data.wcs.copy()

    output = PUNCHData(data=new_data, wcs=new_wcs, meta=input_data.meta)

    logger.info("convert2bpb finished")
    output.meta.history.add_now("LEVEL3-convert2bpb", "Convert MZP to BpB")

    return output

#
# @task
# def convert2bpb_task(data_object: PUNCHData) -> PUNCHData:
#     """Creates a brightness (B) and polarized brightness (pB)
#     from an input of MZP triplet.
#
#     Parameters
#     ----------
#     data_object
#
#
#     Returns
#     -------
#     return output_PUNCHobject : ['punchbowl.data.PUNCHData']
#         returns an array of the same dimensions as the x and y dimensions of
#         the input array
#     """
#
#     logger = get_run_logger()
#     logger.info("subtract_starfield_background started")
#
#     output = subtract_starfield_background(data_object, star_data_array)
#
#     logger.info("subtract_f_corona_background finished")
#     output.meta.history.add_now("LEVEL3-subtract_starfield_background", "subtracted starfield background")
#
#     return output
#
#
