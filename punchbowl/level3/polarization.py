import numpy as np
from ndcube import NDCollection, NDCube
from prefect import get_run_logger, task
from solpolpy import resolve


@task
def convert_polarization(
        input_data: NDCube) -> NDCube:
    """Convert polarization from MZP to BpB."""
    logger = get_run_logger()
    logger.info("convert2bpb started")

    data_collection = NDCollection(
        [("M", input_data[0, :, :]),
         ("Z", input_data[1, :, :]),
         ("P", input_data[2, :, :])],
        aligned_axes="all")

    resolved_data_collection = resolve(data_collection, "BpB", imax_effect=False)

    new_data = np.stack([resolved_data_collection["B"].data, resolved_data_collection["pB"].data], axis=0)
    new_wcs = input_data.wcs.copy()

    output = NDCube(data=new_data, wcs=new_wcs, meta=input_data.meta)

    logger.info("convert2bpb finished")
    output.meta.history.add_now("LEVEL3-convert2bpb", "Convert MZP to BpB")

    return output
