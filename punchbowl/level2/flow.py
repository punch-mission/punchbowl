from typing import List, Union

from prefect import flow, get_run_logger

from punchbowl.data import PUNCHData, load_trefoil_wcs
from punchbowl.level2.bright_structure import identify_bright_structures_task
from punchbowl.level2.merge import merge_many_task
from punchbowl.level2.polarization import resolve_polarization_task
from punchbowl.level2.quality import quality_flag_task
from punchbowl.level2.resample import reproject_many_flow
from punchbowl.util import load_image_task


@flow(validate_parameters=False)
def level2_core_flow(data_list: Union[List[str], List[PUNCHData]]) -> List[PUNCHData]:
    logger = get_run_logger()

    logger.info("beginning level 2 core flow")
    trefoil_wcs, trefoil_shape = load_trefoil_wcs()

    data_list = [load_image_task(d) if isinstance(d, str) else d for d in data_list]
    data_list = resolve_polarization_task(data_list)
    data_list = reproject_many_flow(data_list, trefoil_wcs, trefoil_shape)
    data_list = identify_bright_structures_task(
        data_list
    )  # make sure we have the same polarization states going into each brightfeature run. Needs to be run for all polarization states.
    data_list = quality_flag_task(data_list)
    # TODO: merge only similar polarizations together
    data_list = [merge_many_task(data_list, trefoil_wcs)]
    logger.info("ending level 2 core flow")
    return data_list
