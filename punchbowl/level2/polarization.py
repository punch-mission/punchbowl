
import solpolpy
from ndcube import NDCollection, NDCube
from prefect import get_run_logger, task


def resolve_polarization(data_list: list[NDCube]) -> list[NDCube]:
    """
    Take a set of input data in the camera MZP frame and convert to the solar MZP frame.

    Parameters
    ----------
    data_list : List[PUNCHData]
        List of PUNCHData objects on which to resolve polarization

    Returns
    -------
    List[PUNCHData]
        modified version of the input with polarization resolved

    """
    # Unpack data into a NDCollection object
    data_dictionary = dict(zip(["Bm", "Bz", "Bp"], data_list, strict=False))
    data_collection = NDCollection(data_dictionary)

    resolved_data_collection = solpolpy.resolve(data_collection, "MZP", imax_effect=True)

    # Repack data
    data_list = []
    for key in resolved_data_collection:
        data_list.append(resolved_data_collection[key])

    return data_list


@task
def resolve_polarization_task(data_list: list[NDCube]) -> list[NDCube]:
    """
    Prefect task for polarization resolving.

    Parameters
    ----------
    data_list : List[PUNCHData]
        List of PUNCHData objects on which to resolve polarization

    Returns
    -------
    List[PUNCHData]
        modified version of the input with polarization resolved

    """
    logger = get_run_logger()
    logger.info("resolve_polarization started")

    # Resolve polarization
    data_list = resolve_polarization(data_list)

    logger.info("resolve_polarization ended")

    for data_object in data_list:
        data_object.meta.history.add_now("LEVEL2-resolve_polarization", "polarization resolved")
    return data_list
