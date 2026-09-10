import os
import multiprocessing as mp
from datetime import UTC, datetime
from concurrent.futures import ProcessPoolExecutor

import numba
import numpy as np
from astropy.nddata import StdDevUncertainty

from punchbowl.data import NormalizedMetadata, load_ndcube_from_fits
from punchbowl.data.meta import check_moon_in_fov
from punchbowl.data.punchcube import PUNCHCube
from punchbowl.level1.dynamic_stray_light import phase_in_day
from punchbowl.level2 import nfi_pca
from punchbowl.prefect import get_logger, punch_task
from punchbowl.util import ShmPickleableNDArray, interpolate_data, load_mask_file, nan_percentile_window, stack_images


@punch_task
def quickpunch_pca_filter(input_files,
                          context_files,
                          nfi_mask,
                          pca_components,
                          instrument_frame_background,
                          first_helio_frame_background,
                          second_helio_frame_background,
                          n_workers,
                          n_loaders,
                          downsample_factor):
    logger = get_logger()
    logger.info("Starting PCA flow")

    instrument_frame_background = load_ndcube_from_fits(
        instrument_frame_background, include_uncertainty=False, include_provenance=False)
    first_helio_frame_background = load_ndcube_from_fits(
        first_helio_frame_background, include_uncertainty=False, include_provenance=False)
    second_helio_frame_background = load_ndcube_from_fits(
        second_helio_frame_background, include_uncertainty=False, include_provenance=False)

    numba.set_num_threads(n_workers)
    context = mp.get_context("forkserver")
    with (ProcessPoolExecutor(n_workers, mp_context=context) as process_pool):
        file_list = sorted(input_files + context_files)
        logger.info(f"Loading {len(input_files)} to filter and {len(context_files)} context files")
        x_cube, metas, wcses, cwcses, loaded_files, sat_mask_cube = nfi_pca.load_files(
            file_list, n_workers=n_loaders, downsample_factor=downsample_factor)

        phases = np.array([phase_in_day(path) for path in loaded_files])

        nfi_mask = load_mask_file(nfi_mask)
        nfi_mask_ds = nfi_mask.reshape((x_cube.shape[1] // downsample_factor, downsample_factor,
                                        x_cube.shape[2] // downsample_factor, downsample_factor)).all(axis=(1, 3))

        x_cube_downsampled = ShmPickleableNDArray((x_cube.shape[0],
                                                   x_cube.shape[1] // downsample_factor,
                                                   x_cube.shape[1] // downsample_factor), dtype=x_cube.dtype)
        for i in range(len(x_cube)):
            x_cube_downsampled[i] = nfi_pca.downsample(x_cube[i], downsample_factor)

        logger.info("Images loaded")

        x_cube_ds_filled, plot_masks = nfi_pca.fill_problem_regions(
            x_cube_downsampled, metas, cwcses, sat_mask_cube, nfi_mask_ds, downsample_factor, process_pool)

        x_cube_downsampled.free()
        del x_cube_downsampled

        logger.info("Problem regions filled")

        good_mask = nfi_pca.find_outliers_with_PCA(x_cube_ds_filled, np.ones(len(x_cube)), nfi_mask_ds, n_workers)

        dsl_models = nfi_pca.build_models_with_existing_components(
            x_cube_ds_filled, nfi_mask_ds, pca_components, phases, process_pool)

        logger.info("PCA filtering complete")

        filtered_images, filtered_filled_images = nfi_pca.subtract_models_from_data(
            x_cube, x_cube_ds_filled, dsl_models, downsample_factor, process_pool)

        x_cube.free()
        x_cube_ds_filled.free()
        dsl_models.free()
        del x_cube, x_cube_ds_filled, dsl_models

        logger.info("Components upsampled and subtracted")

        post_filtered_images, _ = nfi_pca.inst_frame_filter(
            filtered_images, filtered_filled_images, downsample_factor, process_pool, instrument_frame_background)

        filtered_images.free()
        filtered_filled_images.free()
        del filtered_images, filtered_filled_images

        logger.info("Post-filtering complete")

        oriented_images, masks, target_frame = nfi_pca.reproject_images(
            post_filtered_images, plot_masks, wcses, process_pool, downsample_factor)

        post_filtered_images.free()
        plot_masks.free()
        del post_filtered_images, plot_masks

        logger.info("Images reprojected")

        subtract_fcorona_models(oriented_images, metas, first_helio_frame_background, second_helio_frame_background)

        logger.info('"F corona" models subtracted')

        circular_mask = nfi_pca.make_circular_mask(oriented_images.shape[1:])
        oriented_images *= circular_mask[None, :, :]

        nan_percentile_window(oriented_images, percentile=50, window_size=7)

        output_array = ShmPickleableNDArray.empty_like(oriented_images)
        vmin = 3e-15
        vmax = 6e-13
        value_masks = (oriented_images > vmin / 10) * (oriented_images < vmax * 10)
        stack_images(oriented_images, value_masks, z_filter_index=0.5, output_array=output_array)

        output_cubes = []
        for i, path in enumerate(loaded_files):
            if path in input_files:
                new_meta = NormalizedMetadata.load_template("QNN", "Q")
                new_meta["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
                for key in metas[i].keys():
                    if ((key in ["DATE-OBS", "DATE-BEG", "DATE-AVG", "DATE-END", "FILEVRSN", "OUTLIER", "BADPKTS",
                                 "XACTTIME", "GEOD_LON", "GEOD_LAT", "GEOD_ALT", "LOS_ALT"]
                            or key[-4:] in ["_OBS", "_VOB"])
                            and key in new_meta):
                        new_meta[key] = metas[i][key].value
                _, _, _, _, moondist, xpix, ypix = check_moon_in_fov(
                    metas[i]["DATE-OBS"].value, wcs=wcses[i], image_shape=output_array[i].shape)
                new_meta["MOONDIST"] = moondist[0]
                new_meta["MOON_X"] = xpix[0]
                new_meta["MOON_Y"] = ypix[0]
                new_meta["OUTLIER"] = not good_mask[i] or metas[i]["OUTLIER"].value

                new_meta.provenance = [os.path.basename(path)]

                uncertainty = np.full(output_array[i].shape, 1e-13)
                uncertainty[masks[i] < 0.5] = np.inf
                cube = PUNCHCube(data=output_array[i], meta=new_meta, wcs=target_frame,
                                 uncertainty=StdDevUncertainty(uncertainty))
                output_cubes.append(cube)

        print("PCA flow done!")
        return output_cubes


def subtract_fcorona_models(data_cube, metas, first_helio_frame_background, second_helio_frame_background):
    for i in range(len(data_cube)):
        interpolated_model = interpolate_data(
            first_helio_frame_background,
            second_helio_frame_background,
            metas[i].datetime,
            time_key="DATE-END",
            allow_extrapolation=True)
        data_cube[i] -= interpolated_model
