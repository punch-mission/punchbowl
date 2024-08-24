
import astropy.units as u
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
    # TODO: don't assume the order and that there are only 3... that is not the case
    data_dictionary = dict(zip(["M", "Z", "P"], data_list, strict=False))
    input_collection = NDCollection(data_dictionary)
    data_collection = NDCollection({k: NDCube(data=input_collection[k].data,
                                 wcs=input_collection[k].wcs,
                                 meta={"POLAR": input_collection[k].meta["POLAR"].value * u.degree})
                       for k in ["M", "Z", "P"]})

    out = []
    resolved_data_collection = solpolpy.resolve(data_collection, "MZP", imax_effect=True)
    for key in resolved_data_collection:
        resolved_data_collection[key].meta = input_collection[key].meta
        resolved_data_collection[key].uncertainty = input_collection[key].uncertainty
        out.append(resolved_data_collection[key])

    return out


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
