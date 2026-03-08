from functools import lru_cache

import numpy as np
import scipy
from ndcube import NDCube
from prefect import get_run_logger

from punchbowl.util import load_mask_file


def trim_edges(data_list: list[NDCube], trim_edge_px: int | list[int] = 0) -> None:
    """Trim the edges of the image, expanding the mask the same amount. Sets masked pixels to nan."""
    for cube in data_list:
        if cube is None:
            continue
        if isinstance(trim_edge_px, list):
            trim_amount = trim_edge_px[int(cube.meta["OBSCODE"].value) - 1]
        else:
            trim_amount = trim_edge_px
        mask = (np.isnan(cube.data) + (cube.data == 0)) * (~np.isfinite(cube.uncertainty.array))
        mask = ~scipy.ndimage.binary_fill_holes(~mask)
        if trim_amount:
            mask = scipy.ndimage.binary_dilation(mask, iterations=trim_amount)
            mask[:trim_amount] = 1
            mask[-trim_amount:] = 1
            mask[:, :trim_amount] = 1
            mask[:, -trim_amount:] = 1
            cube.meta.history.add_now("LEVEL2-preprocess", f"Edges pulled in by {trim_amount} pixels")
        cube.data[mask] = np.nan
        cube.uncertainty.array[mask] = np.inf


@lru_cache
def _load_mask_wrapper(mask_path: str) -> np.ndarray:
    mask = load_mask_file(mask_path)
    mask = np.where(mask, 1, np.nan)
    return mask


def apply_masks(data_list: list[NDCube], mask_list: list[str | None]) -> None:
    """Apply image masks. Sets masked pixels to nan."""
    for cube, mask_path in zip(data_list, mask_list, strict=True):
        if cube is None or mask_path is None:
            continue

        mask = _load_mask_wrapper(mask_path)
        cube.data *= mask
        cube.uncertainty.array[np.isnan(mask)] = np.inf


def apply_alpha(data_list: list[NDCube], alphas_file: str | None = None) -> None:
    """Apply alpha scalings."""
    logger = get_run_logger()
    if alphas_file is not None:
        alpha_data = np.loadtxt(alphas_file, delimiter=",", skiprows=1, dtype=str)
        alphas = {code[1:]: float(alpha) for code, alpha in alpha_data}
        for cube in data_list:
            if cube is None:
                continue
            code = cube.meta["TYPECODE"].value[1:] + cube.meta["OBSCODE"].value
            try:
                alpha = alphas[code]
                cube.data[:] /= alpha
                cube.uncertainty.array[:] /= alpha
                cube.meta.history.add_now("LEVEL2-preprocess", f"Image scaled by factor of {alpha}")
            except KeyError:
                logger.warning(f"Did not find alpha value for {cube.meta['FILENAME'].value}")


def preprocess_trefoil_inputs(data_list: list[NDCube], mask_list: list[str | None],
                              trim_edge_px: int = 0, alphas_file: str | None = None) -> None:
    """Preprocess trefoil inputs."""
    trim_edges(data_list, trim_edge_px)

    apply_alpha(data_list, alphas_file)

    apply_masks(data_list, mask_list)
