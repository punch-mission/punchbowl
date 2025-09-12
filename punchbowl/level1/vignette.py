import os
import time
import pathlib
import warnings
import multiprocessing as mp

import numpy as np
import pandas as pd
import regularizepsf
from astropy.io import fits
from astropy.wcs import WCS
from ndcube import NDCube
from prefect.logging import disable_run_logger
from reproject import reproject_adaptive
from scipy.ndimage import binary_dilation, binary_erosion, grey_closing
from scipy.spatial import KDTree

from punchbowl.data import load_ndcube_from_fits
from punchbowl.data.meta import NormalizedMetadata
from punchbowl.data.punch_io import get_base_file_name
from punchbowl.exceptions import (
    IncorrectPolarizationStateWarning,
    IncorrectTelescopeWarning,
    InvalidDataError,
    LargeTimeDeltaWarning,
    NoCalibrationDataWarning,
)
from punchbowl.level1.alignment import (
    filter_for_visible_stars,
    find_catalog_in_image,
    load_hipparcos_catalog,
    solve_pointing,
)
from punchbowl.level1.sqrt import decode_sqrt_data
from punchbowl.prefect import punch_task
from punchbowl.util import DataLoader, load_spacecraft_mask


@punch_task
def correct_vignetting_task(data_object: NDCube, vignetting_path: str | pathlib.Path | DataLoader | None) -> NDCube:
    """
    Prefect task to correct the vignetting of an image.

    Vignetting is a reduction of an image's brightness or saturation toward the
    periphery compared to the image center, created by the optical path. The
    Vignetting Module will transform the data through a flat-field correction
    map, to cancel out the effects of optical vignetting created by distortions
    in the optical path. This module also corrects detector gain variation and
    offset.

    Correction maps will be 2048*2048 arrays, to match the input data, and
    built using the starfield brightness pattern. Mathematical Operation:

        I'_{i,j} = I_i,j / FF_{i,j}

    Where I_{i,j} is the number of counts in pixel i, j. I'_{i,j} refers to the
    modified value. FF_{i,j} is the small-scale flat field factor for pixel i,
    j. The correction mapping will take into account the orientation of the
    spacecraft and its position in the orbit.

    Uncertainty across the image plane is calculated using the modelled
    flat-field correction with stim lamp calibration data. Deviations from the
    known flat-field are used to calculate the uncertainty in a given pixel.
    The uncertainty is convolved with the input uncertainty layer to produce
    the output uncertainty layer.


    Parameters
    ----------
    data_object : PUNCHData
        data on which to operate

    vignetting_path : pathlib
        path to vignetting function to apply to input data

    Returns
    -------
    PUNCHData
        modified version of the input with the vignetting corrected

    """
    if vignetting_path is None:
        data_object.meta.history.add_now("LEVEL1-correct_vignetting", "Vignetting skipped")
        msg=f"Calibration file {vignetting_path} is unavailable, vignetting correction not applied"
        warnings.warn(msg, NoCalibrationDataWarning)
    else:
        if isinstance(vignetting_path, DataLoader):
            vignetting_function = vignetting_path.load()
            vignetting_path = vignetting_path.src_repr()
        else:
            if isinstance(vignetting_path, str):
                vignetting_path = pathlib.Path(vignetting_path)
            if not vignetting_path.exists():
                msg = f"File {vignetting_path} does not exist."
                raise InvalidDataError(msg)
            vignetting_function = load_ndcube_from_fits(vignetting_path, include_provenance=False)
        vignetting_function_date = vignetting_function.meta.astropy_time
        observation_date = data_object.meta.astropy_time
        if abs((vignetting_function_date - observation_date).to("day").value) > 14:
            msg = f"Calibration file {vignetting_path} contains data created greater than 2 weeks from the observation"
            warnings.warn(msg, LargeTimeDeltaWarning)
        if vignetting_function.meta["TELESCOP"].value != data_object.meta["TELESCOP"].value:
            msg = f"Incorrect TELESCOP value within {vignetting_path}"
            warnings.warn(msg, IncorrectTelescopeWarning)
        if vignetting_function.meta["OBSLAYR1"].value != data_object.meta["OBSLAYR1"].value:
            msg = f"Incorrect polarization state within {vignetting_path}"
            warnings.warn(msg, IncorrectPolarizationStateWarning)
        if vignetting_function.data.shape != data_object.data.shape:
            msg = f"Incorrect vignetting function shape within {vignetting_path}"
            raise InvalidDataError(msg)

        data_object.data[:, :] /= vignetting_function.data[:, :]
        data_object.uncertainty.array[:, :] /= vignetting_function.data[:, :]
        data_object.meta.history.add_now("LEVEL1-correct_vignetting",
                                         f"Vignetting corrected using {os.path.basename(str(vignetting_path))}")
    return data_object


def generate_vignetting_calibration_wfi(path_vignetting: str,
                                        path_mask: str,
                                        spacecraft: str,
                                        vignetting_threshold: float = 1.2,
                                        rows_ignore: tuple = (13,15),
                                        rows_adjust: tuple = (15,16),
                                        rows_adjust_source: tuple = (16,20),
                                        mask_erosion: tuple = (6,6)) -> np.ndarray:
    """
    Create calibration data for vignetting.

    Parameters
    ----------
    path_vignetting : str
        path to raw input vignetting function
    path_mask : str
        path to spacecraft mask function
    spacecraft : str
        spacecraft number
    vignetting_threshold : float, optional
        threshold for bad vignetting pixels, by default 1.2
    rows_ignore : tuple, optional
        rows to exclude entirely from original vignetting data, by default (13,15) for 128x128 input
    rows_adjust : tuple, optional
        rows to adjust to the minimum of a set of rows above (per column), by default (15,16) for 128x128 input
    rows_adjust_source : tuple, optional
        rows to use for statistics to adjust vignetting rows as above, by default (16,20) for 128x128 input
    mask_erosion: tuple, optional
        kernel to use in erosion operation to reduce the mask applied to the vignetting function, by default (6,6)

    Returns
    -------
    np.ndarray
        vignetting function array

    """
    if spacecraft in ["1", "2", "3"]:
        if not os.path.exists(path_vignetting):
            return np.ones((2048,2048))

        with open(path_vignetting) as f:
            lines = f.readlines()

        with open(path_mask, "rb") as f:
            byte_array = f.read()
        mask = np.unpackbits(np.frombuffer(byte_array, dtype=np.uint8)).reshape(2048,2048)
        mask = mask.T

        num_bins, bin_size = lines[0].split()
        num_bins = int(num_bins)
        bin_size = int(bin_size)

        values = np.array([float(v) for line in lines[1:] for v in line.split()])
        vignetting = values[:num_bins**2].reshape((num_bins, num_bins))

        vignetting[vignetting > vignetting_threshold] = np.nan

        vignetting[rows_ignore[0]:rows_ignore[1],:] = np.nan
        vignetting[rows_adjust[0]:rows_adjust[1],:] = np.min(vignetting[rows_adjust_source[0]:rows_adjust_source[1],:],
                                                             axis=0)

        wcs_vignetting = WCS(naxis=2)

        wcs_wfi = WCS(naxis=2)
        wcs_wfi.wcs.cdelt = wcs_wfi.wcs.cdelt * vignetting.shape[0] / 2048.

        vignetting_reprojected = reproject_adaptive((vignetting, wcs_vignetting),
                                                shape_out=(2048,2048),
                                                output_projection=wcs_wfi,
                                                boundary_mode="ignore",
                                                bad_value_mode="ignore",
                                                return_footprint=False)

        mask = binary_erosion(mask, structure=np.ones(mask_erosion))

        vignetting_reprojected = vignetting_reprojected * mask

        vignetting_reprojected[mask == 0] = 1

        return vignetting_reprojected
    if spacecraft=="4":
        raise RuntimeError("Please use the NFI vignetting generator function.")
    raise RuntimeError(f"Unknown spacecraft {spacecraft}")


def generate_vignetting_calibration_nfi(input_files: list[str],
                                        dark_path: str,
                                        path_mask: str,
                                        polarizer: str,
                                        dateobs: str,
                                        version: str,
                                        output_path: str | None = None) -> np.ndarray | None:
    """
    Create calibration data for vignetting for the NFI spacecraft.

    Parameters
    ----------
    input_files : list[str]
        Paths to input NFI files for processing
    dark_path : str
        Path to the dark frame FITS file
    path_mask : str
        Path to the speckle mask FITS file
    polarizer : str
        Polarizer name
    dateobs : str
        Timestamp for calibration file
    version : str
        File version
    output_path : str | None
        Path to calibration file output


    Returns
    -------
    np.ndarray | None
        vignetting function array

    """
    if input_files is None:
        return np.ones((2048,2048))

    # Load speckle mask and dark frame
    with fits.open(path_mask) as hdul:
        specklemask = np.fliplr(hdul[0].data)

    with fits.open(dark_path) as hdul:
        nfidark = hdul[1].data

    # Load a WCS to use later on
    with fits.open(input_files[0]) as hdul:
        cube_wcs = WCS(hdul[1].header)

    # Load and square root decode input data
    cubes = [
        decode_sqrt_data.fn(cube)
        for cube in (load_ndcube_from_fits(file) for file in input_files)
        if 490 <= cube.meta["DATAMDN"].value <= 655 and cube.meta["DATAP99"].value != 4095
           and not cube.meta.__setitem__("OFFSET", 400)
    ]

    # Subtract dark frame
    for cube in cubes:
        cube.data[...] -= nfidark

    # Build speckle boundary mask
    inverted_mask = 1 - specklemask
    dilated = binary_dilation(inverted_mask, structure=np.ones((3, 3)))
    boundary_mask = dilated & (~inverted_mask)

    # Stack image data
    images = np.array([cube.data for cube in cubes])
    applied_images = images * boundary_mask
    applied_speck = images * specklemask

    # Compute averages and construct flatfield
    avg_images = np.nanmean(applied_images, axis=0)
    avg_img_darkremoved = np.nanmean(images, axis=0)
    avg_speck = np.nanmean(applied_speck, axis=0)
    avg_speckfilled = grey_closing(avg_images, structure=np.ones((7, 7)))

    nficlean = avg_speckfilled * inverted_mask + avg_speck
    nfiflat = avg_img_darkremoved / nficlean

    # Load spacecraft mask
    mask_nfi = load_spacecraft_mask(path_mask)

    # Deal with infs and remask
    nfiflat[np.isinf(nfiflat)] = 1.
    nfiflat[mask_nfi == 0] = 1

    # Generate an output metadata and NDCube
    m = NormalizedMetadata.load_template(f"G{polarizer}4", "1")
    m["DATE-OBS"] = dateobs
    m["FILEVRSN"] = version

    cube = NDCube(data=nfiflat.astype("float32"), wcs=cube_wcs, meta=m)

    if output_path is not None:
        filename = f"{output_path}/{get_base_file_name(cube)}.fits"

        full_header = cube.meta.to_fits_header(wcs=cube.wcs)
        full_header["FILENAME"] = os.path.basename(filename)

        hdu_data = fits.ImageHDU(data=cube.data,
                                     header=full_header,
                                     name="Primary data array")
        hdu_provenance = fits.BinTableHDU.from_columns(fits.ColDefs([fits.Column(
            name="provenance", format="A40", array=np.char.array(cube.meta.provenance))]))
        hdu_provenance.name = "File provenance"

        hdul = cube.wcs.to_fits()
        hdul[0] = fits.PrimaryHDU()
        hdul.insert(1, hdu_data)

        hdul.append(hdu_provenance)
        hdul.writeto(filename, overwrite=True, checksum=True)
        hdul.close()

        return None
    return cube.data

def measure_single_star(x: int, y: int, image: np.ndarray, width: int = 7) -> np.ndarray:
    x, y = int(x), int(y)
    half_width = width // 2
    patch = image[x-half_width:x+half_width+1, y-half_width:y+half_width+1]
    border = np.concatenate([patch[0, :], patch[-1, :], patch[:, 0], patch[:, -1]])
    border_median = np.median(border)
    return np.sum(patch - border_median)

def measure_stars_in_one_image(cube, distortion_wcs, psf_model, catalog=None, width: int = 7) -> pd.DataFrame:
    d = cube.data.astype(np.float64).copy()
    d = d**2 / cube.meta["SCALE"].value  # TODO: do proper sqrt decoding
    d = psf_model.apply(d, saturation_threshold=55_000, saturation_dilation=3)
    d = d.copy()
    w = solve_pointing(d, cube.wcs, distortion_wcs)
    if catalog is None:
        catalog = filter_for_visible_stars(load_hipparcos_catalog(), dimmest_magnitude=8)
    stars_found = find_catalog_in_image(catalog, w, d.shape)

    results = []
    for _, star in stars_found.iterrows():
        try:
            measurement = measure_single_star(star["y_pix"], star["x_pix"], d, width)
            results.append({"hip": star["HIP"],
                            "aperture_sum_bkgsub": measurement,
                            "xcenter": star["x_pix"],
                            "ycenter": star["y_pix"],
                            "Vmag": star["Vmag"]})
        except:  # if it fails for any reason, just move onto the next star
            pass
    return pd.DataFrame(results)

def single_image_helper(path, distortion_wcs, psf_model):
    try:
        catalog = filter_for_visible_stars(load_hipparcos_catalog(), dimmest_magnitude=6)
        with disable_run_logger():
            cube = load_ndcube_from_fits(path, key="A")
            return measure_stars_in_one_image(cube, distortion_wcs, psf_model, catalog)
    except Exception as e:
        print(e)
        return None

def measure_stars_for_vignetting(level0_paths: list[str], distortion_path: str, psf_path: str, num_workers: int = -1) -> list[pd.DataFrame]:
    with fits.open(distortion_path) as distortion:
        distortion_wcs = WCS(distortion[0].header, distortion, key="A")
    psf_transform = regularizepsf.ArrayPSFTransform.load(psf_path)

    paths = [(p, distortion_wcs, psf_transform) for p in level0_paths]
    with mp.Pool(num_workers) as pool:
        tables = pool.starmap(single_image_helper, paths)
    return tables

def convert_star_measurements_to_vignetting(tables: list[pd.DataFrame], image_mask: np.ndarray) -> np.ndarray:
    # image_mask = fits.open(paths[0])[1].data == 0
    tables = [t for t in tables if t is not None]
    all_hip = set()
    for table in tables:
        all_hip = all_hip.union(set([int(i) for i in np.array(table["hip"])]))

    xcenters = {h: np.zeros(len(tables)) + np.nan for h in all_hip}
    ycenters = {h: np.zeros(len(tables)) + np.nan for h in all_hip}
    measurement = {h: np.zeros(len(tables)) + np.nan for h in all_hip}
    mags = dict.fromkeys(all_hip, np.nan)
    for i, table in enumerate(tables):
        for _, row in table.iterrows():
            h = row["hip"]
            xcenters[h][i] = row["xcenter"]
            ycenters[h][i] = row["ycenter"]
            measurement[h][i] = row["aperture_sum_bkgsub"]
            mags[h] = row["Vmag"]

    num_nan = {h: np.sum(np.isnan(v)) for h, v in measurement.items()}
    used_hip = [h for h, v in num_nan.items() if v < len(tables) - 500]
    used_hip = np.array(used_hip)

    rows = []
    for h in used_hip:
        for i in range(len(tables)):
            if not np.isnan(measurement[h][i]):
                this_dict = {"hip": h,
                             "mag": mags[h],
                             "x_center": xcenters[h][i],
                             "y_center": ycenters[h][i],
                             "measurement": measurement[h][i]}
                rows.append(this_dict)
    df = pd.DataFrame(rows)

    neighborhood_size = 500

    ls_used_hip = np.random.choice(np.array([h for h in used_hip if 6 > mags[h] > 3]), 500)
    v_size = 10

    ls_used_hip_mapping = {h: i for i, h in enumerate(ls_used_hip)}

    subdf = df[df["hip"].isin(ls_used_hip)]
    subdf = subdf[~image_mask[subdf["y_center"].values.astype(int), subdf["x_center"].values.astype(int)]]
    tree = KDTree(np.stack([subdf["x_center"], subdf["y_center"]], axis=1))

    window = 25
    mask = (subdf["x_center"] > 1024 - window) * (subdf["x_center"] < 1024 + window) * (
                subdf["y_center"] > 1024 - window) * (subdf["y_center"] < 1024 + window) * (
                       subdf["measurement"] > 0)
    centerdf = subdf[mask]
    poly = np.polyfit(centerdf["mag"], np.log10(centerdf["measurement"]), 1)

    initial_brightnesses = []
    for h in ls_used_hip:
        this_hip_df = df[df["hip"] == h]
        mag = this_hip_df["mag"].iloc[0]
        initial_brightnesses.append(10 ** np.polyval(poly, mag))

    current_brightnesses = np.array(initial_brightnesses)

    for iteration in range(1, 6): # TODO: make this not a hard coded value
        v_size = 30 * iteration  # TODO: make this not a hard coded value
        v_step = 2048 / v_size

        from skimage.transform import resize
        image_mask_resized = resize(image_mask, (v_size, v_size))

        print(f"ITERATION {iteration}")
        updated_vignetting = np.ones((v_size, v_size))
        samples = np.zeros((v_size, v_size))

        for i in range(v_size):
            for j in range(v_size):
                ii, jj = i * v_step, j * v_step
                out = np.array(tree.query_ball_point(np.array([[ii, jj]]), neighborhood_size)[0])
                if len(out):
                    d = np.sqrt(
                        (subdf["x_center"].values[out] - ii) ** 2 + (subdf["y_center"].values[out] - jj) ** 2)
                    vignette_amount = np.clip(np.array([v / current_brightnesses[ls_used_hip_mapping[h]]
                                                        for h, v in zip(subdf["hip"].values[out],
                                                                        subdf["measurement"].values[out], strict=False)]), 1E-6,
                                              1)
                    low, high = np.nanpercentile(vignette_amount, (15, 95))  # TODO: make this not hard coed
                    mask = (vignette_amount > low) * (vignette_amount < high)
                    if np.sum(mask) < 500:  # TODO: make this not hard coded
                        mask = np.ones(len(mask), dtype=bool)
                    w = 1 / (d + neighborhood_size) ** 2
                    w[d > neighborhood_size] = 0
                    if np.sum(w) == 0:
                        w = 1 / (d + neighborhood_size) ** 2
                    try:
                        new_value = np.average(vignette_amount[mask], weights=w[mask])
                        samples[j, i] = len(vignette_amount[mask])
                    except ZeroDivisionError:
                        new_value = np.average(vignette_amount, weights=w)
                        samples[j, i] = len(vignette_amount)
                else:
                    new_value = 0
                updated_vignetting[j, i] = new_value
        updated_vignetting[image_mask_resized] = 0

        start = time.time()
        for i, h in enumerate(ls_used_hip):
            this_hip_df = subdf[subdf["hip"] == h]
            xx = np.round(this_hip_df["x_center"].values / v_step).astype(int).clip(0, v_size - 1)
            yy = np.round(this_hip_df["y_center"].values / v_step).astype(int).clip(0, v_size - 1)
            mm = this_hip_df["measurement"] / updated_vignetting[yy, xx]
            new_brightness = np.nanpercentile(mm[mm > 0], 50)
            current_brightnesses[i] = new_brightness

        return updated_vignetting
