import numpy as np
import scipy
from ndcube import NDCube
from prefect import get_run_logger


def trim_edges(data_list: list[NDCube], trim_edge_px: int = 0) -> None:
    """Trim the edges of the image, expanding the mask the same amount. Sets masked pixels to nan."""
    for cube in data_list:
        mask = (cube.data == 0) * (np.isinf(cube.uncertainty.array))
        if trim_edge_px:
            mask[:trim_edge_px] = 1
            mask[-trim_edge_px:] = 1
            mask[:, :trim_edge_px] = 1
            mask[:, -trim_edge_px:] = 1
            mask = scipy.ndimage.binary_dilation(mask, iterations=trim_edge_px)
        cube.data[mask] = np.nan
        cube.uncertainty.array[mask] = np.inf


def apply_alpha(data_list: list[NDCube], alphas_file: str | None = None) -> None:
    """Apply alpha scalings."""
    logger = get_run_logger()
    if alphas_file is not None:
        alpha_data = np.loadtxt(alphas_file, delimiter=",", skiprows=1, dtype=str)
        alphas = {code[1:]: float(alpha) for code, alpha in alpha_data}
        for cube in data_list:
            code = cube.meta["TYPECODE"].value[1:] + cube.meta["OBSCODE"].value
            try:
                alpha = alphas[code]
                cube.data /= alpha
                cube.uncertainty.array /= alpha
            except KeyError:
                logger.warning(f"Did not find alpha value for {cube.meta['FILENAME'].value}")


def preprocess_trefoil_inputs(data_list: list[NDCube], trim_edge_px: int = 0, alphas_file: str | None = None) -> None:
    """Preprocess trefoil inputs."""
    trim_edges(data_list, trim_edge_px)

    apply_alpha(data_list, alphas_file)
