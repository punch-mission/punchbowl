from datetime import UTC, datetime

from ndcube import NDCube

from punchbowl.data.meta import check_moon_in_fov, set_spacecraft_location_to_earth
from punchbowl.util import find_first_existing_file

SPACECRAFT_OBSCODE = {"1": "WFI1",
                      "2": "WFI2",
                      "3": "WFI3",
                      "4": "NFI4"}


def finalize_output(output_cube: NDCube, input_cubes: list[NDCube]) -> None:
    """Do metadata updates common to L2 *TM merged files and L3 *IM re-merged files."""
    # Put cubes in a deterministic order
    input_cubes.sort(key=lambda cube: cube.meta.datetime)

    sample_cube = find_first_existing_file(input_cubes)

    output_cube.meta["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    set_spacecraft_location_to_earth(output_cube)

    output_cube.meta.provenance = [fname for d in input_cubes
        if d is not None and (fname := d.meta.get("FILENAME").value)]

    output_cube.meta["FILEVRSN"] = sample_cube.meta["FILEVRSN"].value

    _, angle_sun, _, _, _, xpix, ypix = check_moon_in_fov(
        output_cube.meta["DATE-OBS"].value, wcs=output_cube.wcs,
        image_shape=output_cube.data.shape)
    output_cube.meta["MOONDIST"] = angle_sun[0]
    output_cube.meta["MOON_X"] = xpix[0]
    output_cube.meta["MOON_Y"] = ypix[0]

    for d in filter(None, input_cubes):
        spacecraft = SPACECRAFT_OBSCODE[d.meta["OBSCODE"].value]
        output_cube.meta[f"HAS_{spacecraft}"] = 1

    output_cube.meta["ALL_INPT"] = {output_cube.meta["HAS_WFI1"].value,
                                    output_cube.meta["HAS_WFI2"].value,
                                    output_cube.meta["HAS_WFI3"].value,
                                    output_cube.meta["HAS_NFI4"].value} == {1}
