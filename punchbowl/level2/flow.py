
from ndcube import NDCube
from prefect import flow, get_run_logger

from punchbowl.data import load_trefoil_wcs
from punchbowl.level2.bright_structure import identify_bright_structures_task
from punchbowl.level2.merge import merge_many_task
from punchbowl.level2.polarization import resolve_polarization_task
from punchbowl.level2.resample import reproject_many_flow
from punchbowl.util import load_image_task, output_image_task


@flow(validate_parameters=False)
def level2_core_flow(data_list: list[str] | list[NDCube],
                     voter_filenames: list[list[str]],
                     output_filename: str | None) -> list[NDCube]:
    """Level 2 core flow."""
    logger = get_run_logger()

    logger.info("beginning level 2 core flow")
    trefoil_wcs, trefoil_shape = load_trefoil_wcs()

    data_list = [load_image_task(d) if isinstance(d, str) else d for d in data_list]
    data_list = resolve_polarization_task(data_list[:3]) + resolve_polarization_task(data_list[3:6]) + resolve_polarization_task(data_list[6:9]) + resolve_polarization_task(data_list[9:])
    data_list = reproject_many_flow(data_list, trefoil_wcs, trefoil_shape)
    data_list = [identify_bright_structures_task(cube, voter_filenames)
                 for cube, voter_filenames in zip(data_list, voter_filenames)]
    # TODO: merge only similar polarizations together
    output_data = merge_many_task(data_list, trefoil_wcs)

    if output_filename is not None:
        output_image_task(output_data, output_filename)

    logger.info("ending level 2 core flow")
    return [output_data]


if __name__ == "__main__":
    level2_core_flow(["/Users/jhughes/Desktop/repos/punchbowl/test_run/test_PM1.fits",
                                  "/Users/jhughes/Desktop/repos/punchbowl/test_run/test_PZ1.fits",
                                  "/Users/jhughes/Desktop/repos/punchbowl/test_run/test_PP1.fits",
                                  "/Users/jhughes/Desktop/repos/punchbowl/test_run/test_PM2.fits",
                                  "/Users/jhughes/Desktop/repos/punchbowl/test_run/test_PZ2.fits",
                                  "/Users/jhughes/Desktop/repos/punchbowl/test_run/test_PP2.fits",
                                  "/Users/jhughes/Desktop/repos/punchbowl/test_run/test_PM3.fits",
                                  "/Users/jhughes/Desktop/repos/punchbowl/test_run/test_PZ3.fits",
                                  "/Users/jhughes/Desktop/repos/punchbowl/test_run/test_PP3.fits",
                                  "/Users/jhughes/Desktop/repos/punchbowl/test_run/test_PM4.fits",
                                  "/Users/jhughes/Desktop/repos/punchbowl/test_run/test_PZ4.fits",
                                  "/Users/jhughes/Desktop/repos/punchbowl/test_run/test_PP4.fits"
                      ],
                               voter_filenames=[[], [], [],
                                                [], [], [],
                                                [], [], [],
                                                [], [], []],
                               output_filename="/Users/jhughes/Desktop/repos/punchbowl/test_run/test_l2_v12.fits")
