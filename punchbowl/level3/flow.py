from typing import List, Union, Optional

from prefect import flow, get_run_logger

from punchbowl.data import PUNCHData
from punchbowl.level3.f_corona_model import subtract_f_corona_background_task
from punchbowl.util import load_image_task


@flow(validate_parameters=False)
def level3_core_flow(data_list: Union[List[str], List[PUNCHData]],
                     f_corona_model_path: Optional[str]) -> List[PUNCHData]:
    logger = get_run_logger()

    logger.info("beginning level 3 core flow")
    data_list = [load_image_task(d) if isinstance(d, str) else d for d in data_list]
    data_list = [subtract_f_corona_background_task(d, f_corona_model_path) for d in data_list]
    logger.info("ending level 3 core flow")
    return data_list
