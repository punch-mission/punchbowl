from collections.abc import Callable

import numpy as np
import scipy.signal
from ndcube import NDCube
from prefect import get_run_logger
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from punchbowl.prefect import punch_task
from punchbowl.util import load_image_task


@punch_task()
def pca_filter(input_cube: NDCube, files_to_fit: list[NDCube | Callable | str],
               n_components: int=50, med_filt: int=5) -> None:
    """Run PCA-based filtering."""
    logger = get_run_logger()
    all_files_to_fit = np.empty((len(files_to_fit), *input_cube.data.shape), dtype=input_cube.data.dtype)
    for i, input_file in enumerate(files_to_fit):
        if isinstance(input_file, NDCube):
            all_files_to_fit[i] = input_file.data
        elif isinstance(input_file, str):
            all_files_to_fit[i] = load_image_task(input_file, include_provenance=False, include_uncertainty=False).data
        elif isinstance(input_file, Callable):
            input_file(all_files_to_fit[i])
        else:
            raise TypeError(f"Invalid type {type(input_file)} for input file")
    logger.info(
        f"Loaded {len(all_files_to_fit)} images to fit; filling {all_files_to_fit.nbytes/1024**3:.2f} GB")

    with threadpool_limits(30):
        pca = PCA(n_components=n_components)
        pca.fit(all_files_to_fit.reshape((len(all_files_to_fit), -1)))
        logger.info("Fitting finished")

        transformed = pca.transform(input_cube.data.reshape((1, -1)))

        if med_filt:
            for i in range(len(pca.components_)):
                comp = pca.components_[i].reshape(input_cube.data.shape)
                comp = scipy.signal.medfilt2d(comp, med_filt)
                pca.components_[i] = comp.ravel()
            logger.info("Median smoothing finished")

        reconstructed = pca.inverse_transform(transformed).reshape(input_cube.data.shape)
        input_cube.data -= reconstructed
