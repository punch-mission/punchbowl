import os
from pathlib import Path
from collections.abc import Callable

import numpy as np
import reproject
from astropy.wcs import WCS
from ndcube import NDCube
from prefect import get_run_logger
from regularizepsf import ArrayPSF, ArrayPSFBuilder, ArrayPSFTransform, simple_functional_psf, varied_functional_psf
from regularizepsf.util import calculate_covering

from punchbowl.data.punch_io import load_ndcube_from_fits
from punchbowl.prefect import punch_task


def build_psf_transform(image_paths: list[str] | list[Path],
                        alpha: float = 0.7,
                        epsilon: float = 0.515,
                        target_sigma: float = 3.3,
                        psf_size: int = 64) -> ArrayPSFTransform:
    """
    Build the PSF transform for Level 1 processing from a list of images.

    Parameters
    ----------
    image_paths : List[str] | List[Path]
        images to use in building the PSF model
    alpha : float
        alpha parameter of the PSF transform, see Coma Off It paper
    epsilon : float
        epsilon parameter of the PSF transform, see Coma Off It paper
    target_sigma : float
        the target sigma of the PSF model in pixels
    psf_size : int
        size of the grid in the PSF model in pixels

    Returns
    -------
    ArrayPSFTransform
        the PSF transform that corresponds to the input images

    """
    b = ArrayPSFBuilder(psf_size)
    model, counts = b.build(image_paths)

    first_cube = load_ndcube_from_fits(image_paths[0], key="A")

    corrected_psf = generate_projected_psf(
            first_cube.wcs, psf_width=psf_size, star_gaussian_sigma=target_sigma)
    return ArrayPSFTransform.construct(model, corrected_psf, alpha=alpha, epsilon=epsilon)

def generate_projected_psf(
        source_wcs: WCS,
        psf_width: int = 64,
        star_gaussian_sigma: float = 3.3 / 2.355) -> ArrayPSF:
    """Create a varying PSF reflecting how a true circle looks in the mosaic image projection."""
    # Create a Gaussian star
    coords = np.arange(psf_width) - psf_width / 2 + .5
    xx, yy = np.meshgrid(coords, coords)
    perfect_star = np.exp(-(xx ** 2 + yy ** 2) / (2 * star_gaussian_sigma ** 2))

    star_wcs = WCS(naxis=2)
    star_wcs.wcs.ctype = "RA---ARC", "DEC--ARC"
    star_wcs.wcs.crpix = psf_width / 2 + .5, psf_width / 2 + .5
    star_wcs.wcs.cdelt = source_wcs.wcs.cdelt

    @simple_functional_psf
    def projected_psf(row: np.ndarray,  # noqa: ARG001
                      col: np.ndarray,  # noqa: ARG001
                      i: int = 0,
                      j: int = 0) -> np.ndarray:
        # Work out the center of this PSF patch
        ic = i + psf_width / 2 - .5
        jc = j + psf_width / 2 - .5
        ra, dec = source_wcs.array_index_to_world_values(ic, jc)

        # Create a WCS that places a star at that exact location
        swcs = star_wcs.deepcopy()
        swcs.wcs.crval = ra, dec

        # Project the star into this patch of the full image, telling us what a round
        # star looks like in this projection, distortion, etc.
        psf = reproject.reproject_adaptive(
            (perfect_star, swcs),
            source_wcs[i:i + psf_width, j:j + psf_width],
            (psf_width, psf_width),
            roundtrip_coords=False, return_footprint=False,
            boundary_mode="grid-constant", boundary_fill_value=0)
        return psf / np.sum(psf)

    @varied_functional_psf(projected_psf)
    def varying_projected_psf(row: int, col: int) -> dict:
        # row and col seem to be the upper-left corner of the image patch we're to describe
        return {"i": row, "j": col}

    coords = calculate_covering(source_wcs.array_shape, psf_width)
    return varying_projected_psf.as_array_psf(coords, psf_width)


def correct_psf(
    data: NDCube,
    psf_transform: ArrayPSFTransform,
    max_workers: int | None = None,
) -> NDCube:
    """Correct PSF."""
    new_data = psf_transform.apply(data.data, workers=max_workers)

    data.data[...] = new_data[...]
    # TODO: uncertainty propagation
    return data

@punch_task
def correct_psf_task(
    data_object: NDCube,
    model_path: str | Callable | None = None,
    max_workers: int | None = None,
) -> NDCube:
    """
    Prefect Task to correct the PSF of an image.

    Parameters
    ----------
    data_object : NDCube
        data to operate on
    model_path : str
        path to the PSF model to use in the correction
    max_workers : int
        the maximum number of worker threads to use

    Returns
    -------
    NDCube
        modified version of the input with the PSF corrected

    """
    if model_path is not None:
        if isinstance(model_path, Callable):
            corrector, model_path = model_path()
        else:
            corrector = ArrayPSFTransform.load(Path(model_path))
        data_object = correct_psf(data_object, corrector, max_workers)
        data_object.meta.history.add_now("LEVEL1-correct_psf",
                                         f"PSF corrected with {os.path.basename(model_path)} model")
    else:
        data_object.meta.history.add_now("LEVEL1-correct_psf", "Empty model path so no correction applied")
        logger = get_run_logger()
        logger.info("No model path so PSF correction is skipped")
    return data_object
