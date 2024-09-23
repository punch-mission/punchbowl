import astropy.units as u
import numpy as np
from ndcube import NDCollection, NDCube
from prefect import get_run_logger, task
from solpolpy import resolve

from punchbowl.data.meta import NormalizedMetadata


@task
def convert_polarization(
        input_data: NDCube) -> NDCube:
    """Convert polarization from MZP to BpB."""
    logger = get_run_logger()
    logger.info("convert2bpb started")

    collection_contents = [(label, NDCube(data=input_data[i].data,
                                 wcs=input_data.wcs.dropaxis(2),
                                 meta={"POLAR": angle}))
                       for (label, i, angle) in zip(["M", "Z", "P"],
                                                    [0, 1, 2],
                                                    [-60*u.deg, 0*u.deg, 60*u.deg], strict=False)]
    data_collection = NDCollection(collection_contents, aligned_axes="all")

    resolved_data_collection = resolve(data_collection, "BpB", imax_effect=False)

    new_data = np.stack([resolved_data_collection["B"].data, resolved_data_collection["pB"].data], axis=0)
    new_wcs = input_data.wcs.copy()

    output_meta = NormalizedMetadata.load_template("PTM", "3")
    output_meta["DATE-OBS"] = input_data.meta["DATE-OBS"].value
    output = NDCube(data=new_data, wcs=new_wcs, meta=input_data.meta)

    logger.info("convert2bpb finished")
    output.meta.history.add_now("LEVEL3-convert2bpb", "Convert MZP to BpB")

    return output
