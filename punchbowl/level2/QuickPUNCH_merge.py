# Core Python imports
from typing import List

# Third party imports
from prefect import flow, get_run_logger

# Punchbowl imports
from punchbowl.data import PUNCHData
from punchbowl.level2.image_merge import generate_wcs, mosaic


# core module flow
@flow
def quickpunch_merge_flow(data: List) -> PUNCHData:
    logger = get_run_logger()
    logger.info("QuickPUNCH_merge module started")

    # Define output WCS from file
    (trefoil_wcs, trefoil_shape) = generate_wcs("level2/data/trefoil_hdr.fits")

    # Generate a mosaic to these specifications from input data
    data_object = mosaic(data, trefoil_wcs, trefoil_shape)

    logger.info("QuickPUNCH_merge flow finished")
    data_object.meta.history.add_now("LEVEL2-module", "QuickPUNCH_merge ran")
    return data_object
