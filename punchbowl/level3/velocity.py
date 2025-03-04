from datetime import datetime

import astropy.units as u
import cv2 as cv
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCube
from sunpy.sun import constants

from punchbowl.data import load_ndcube_from_fits
from punchbowl.data.meta import NormalizedMetadata


def calc_ylims(ycen_band_rs: np.ndarray, r_band_width: float, arcsec_per_px: float) -> tuple[int, int]:
    """
    Convert y-coordinates of lower and upper row of bands to array indices for slicing.

    Parameters
    ----------
    ycen_band_rs : np.ndarray
        y-coordinates of center of band in solar radii

    r_band_width : float
        Half-width of each radial band in solar radii

    arcsec_per_px : float
        Radial pixel scale in arcsec/px in the polar-remapped images

    Returns
    -------
    list
        Lower and upper Numpy array indices of the radial band

    """
    # Unless we have a cropped image, bottom axis of the polar transform should be at 0 Rs.
    origin_rs = 0
    origin_arcsec = origin_rs * arcsec_per_px
    ycen_band_arcsec = ycen_band_rs * constants.average_angular_size.value   # center of the radial band in arcsec
    rband_width_arcsec = r_band_width * constants.average_angular_size.value  # width of the radial band in arcsec
    ylo_band_idx = ((ycen_band_arcsec - rband_width_arcsec) - origin_arcsec) / arcsec_per_px   # lower index of the band
    yhi_band_idx = ((ycen_band_arcsec + rband_width_arcsec) - origin_arcsec) / arcsec_per_px   # upper index of the band
    return [ylo_band_idx, yhi_band_idx]


def preprocess_image(data: NDCube, max_radius_px: int, num_azimuth_bins: int, az_bin: int) -> np.ndarray:
    """
    Normalize and preprocess FITS image by removing bad values and scaling.

    Parameters
    ----------
    data: NDCube
        Input data NDCube

    max_radius_px : int
        Maximum radius to include for polar remapping

    num_azimuth_bins : int
        Number of azimuthal samples to use in polar remapping

    az_bin: int
        Binning factor for binning the polar remapped image over the azimuth. The binning rule is currently numpy.mean()

    Returns
    -------
    np.ndarray, dict
        - Preprocecess polar-remapped image
        - associated metadata

    """
    # Replace with appropriate preprocessing needed to clean-up. We need to have finite values for the polar remap
    image = data.data[0,...]
    header = data.meta.to_fits_header(wcs=data.wcs)

    image[~np.isfinite(image)] = 0

    polar_image = cv.warpPolar(image.astype(np.float64), [int(max_radius_px), int(num_azimuth_bins)],
                               [header["CRPIX1"], header["CRPIX2"]], max_radius_px, cv.INTER_CUBIC)

    polar_image_binned = polar_image.T.reshape([polar_image.shape[1],
                                                polar_image.shape[0] // az_bin, az_bin]).mean(axis=2)

    # Remove background radially by taking the mean over the radial axis (From Craig's original implementation)
    polar_image_binned_radial_bkg = polar_image_binned - np.mean(polar_image_binned, axis=0)
    # Flat-fielding further: dividing by RMS value along the radial axis
    ff_rms = np.sqrt(np.mean(polar_image_binned_radial_bkg ** 2, axis=0))
    processed_image = polar_image_binned_radial_bkg / ff_rms
    # Clean-up, as divide by zero will occur
    processed_image[~np.isfinite(processed_image)] = 0

    polar_header = {
        "NAXIS": 2,
        "CTYPE1": "HPLN-CAR",
        "NAXIS1": processed_image.shape[1],
        "CDELT1": 360 / processed_image.shape[1],
        "CUNIT1": "deg",
        "CRPIX1": 0.5,
        "CRVAL1": 0,
        "CTYPE2": "HPLT-CAR",
        "NAXIS2": processed_image.shape[0],
        "CDELT2": 45 * 3600 / max_radius_px,
        "CUNIT2": "arcsec",
        "CRPIX2": processed_image.shape[0]//2 + 0.5,
        "CRVAL2": (processed_image.shape[0]//2 + 0.5) * (45 * 3600 / max_radius_px),
        "DATE-OBS": header["DATE-OBS"],
    }

    return processed_image, polar_header


def calculate_cross_correlation(image1: np.ndarray, image2: np.ndarray,
                                offsets: np.ndarray, delta_px: int,
                                central_offset: int) -> np.ndarray:
    """
    Perform cross-correlation for a range of offsets.

    Parameters
    ----------
    image1 : np.ndarray
        First image array for correlation

    image2 : np.ndarray
        Second, time-offset image array for correlation

    offsets : np.ndarray
        Array of pixel offsets to iterate over for cross-correlation

    delta_px : int
        Pixel offset increment between samples

    central_offset : int
        Central offset from which to start correlation

    Returns
    -------
    np.ndarray
        Accumulated cross-correlation array over all offsets

    """
    # Initialize accumulator array
    acc = np.zeros((len(offsets), image1.shape[0], image1.shape[1]), dtype=float)
    for jj, offset_index in enumerate(offsets):
        # The two images need to be shifted from each other at each iteration.
        # We first calculate the overall shift, then divide it in two parts, each part being used to shift each image
        this_of = int(delta_px * (offset_index - (len(offsets) - 1) / 2)) + central_offset
        # Amount of shift for image
        offset_1 = int(this_of / 2)
        offset_2 = int(this_of) - offset_1

        # Padding the images symmetrically according to the direction of the shift. The padding rule is to replicate the
        # values at the nearest edge
        if offset_1 < 0:
            padded_image1 = np.pad(image1, ((0, -offset_1), (0, 0)), mode="edge")[abs(offset_1):image1.shape[0] +
                                                                                  abs(offset_1), :]
        else:
            padded_image1 = np.pad(image1, ((offset_1, 0), (0, 0)), mode="edge")[:image1.shape[0], :]

        if offset_2 < 0:
            padded_image2 = np.pad(image2, ((-offset_2, 0), (0, 0)), mode="edge")[:image2.shape[0], :]
        else:
            padded_image2 = np.pad(image2, ((0, offset_2), (0, 0)), mode="edge")[offset_2:image2.shape[0] + offset_2, :]

        acc[jj, :, :] += padded_image1 * padded_image2

    return acc


def accumulate_cross_correlation_across_frames(files: list, delta_t: int, sparsity: int, n_ofs: int,
                                               max_radius_deg: int, num_azimuth_bins: int, az_bin: int,
                                               delta_px: int, central_offset: int) -> np.ndarray:
    """
    Accumulate cross-correlation across frames in a list of FITS files.

    Parameters
    ----------
    files : list
        List of file paths to FITS files

    delta_t : int
        Frame offset (in frames) between time-offset image pairs

    sparsity : int
        Interval between frames to skip when accumulating cross-correlation

    n_ofs : int
        Number of pixel offsets to use in cross-correlation

    max_radius_deg : int
        Maximum radius in degrees to include for polar remapping

    num_azimuth_bins : int
        Number of azimuthal samples to use in polar remapping

    az_bin: int
        Binning factor for binning the polar remapped image over the azimuth. The binning rule is currently numpy.mean()

    delta_px : int
        Pixel offset increment between samples

    central_offset : int
        Central offset from which to start correlation


    Returns
    -------
    np.ndarray
        Accumulated cross-correlation array over all frames and offsets

    """
    data = load_ndcube_from_fits(files[0])
    header = data.meta.to_fits_header(wcs=data.wcs)


    max_radius_px = max_radius_deg / header["CDELT1"]
    polar_sample, _ = preprocess_image(data, max_radius_px, num_azimuth_bins, az_bin)

    acc = np.zeros((n_ofs, polar_sample.shape[0], polar_sample.shape[1]), dtype=float)
    n = 0
    for i in range(0, len(files) - delta_t, delta_t * sparsity):
        data1 = load_ndcube_from_fits(files[i])

        data2 = load_ndcube_from_fits(files[i + delta_t])

        prepped_image1, _ = preprocess_image(data1, max_radius_px, num_azimuth_bins, az_bin)
        prepped_image2, _ = preprocess_image(data2, max_radius_px, num_azimuth_bins, az_bin)

        acc += calculate_cross_correlation(prepped_image1, prepped_image2, np.arange(n_ofs), delta_px, central_offset)

        n += 1

    acc /= n

    return acc


def compute_all_bands(acc: np.ndarray, ycen_band_rs: np.ndarray, r_band_half_width: float,
                      arcsec_per_px: float, velocity_azimuth_bins: int,
                      x_kps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute speed and sigma for all radial bands.

    Parameters
    ----------
    acc : np.ndarray
        Cross-correlation array accumulated across frames

    ycen_band_rs : np.ndarray
        y-coordinates of band centers in solar radii

    r_band_half_width : float
        Half-width of each radial band in solar radii

    arcsec_per_px : float
        Radial pixel scale in arcsec/px in the polar-remapped images

    velocity_azimuth_bins : int
        Number of azimuthal bins in the output flow maps

    x_kps : np.ndarray
        Array mapping pixel offsets to speed in km/s

    Returns
    -------
    tuple
        Tuple containing:
        - np.ndarray : Average speed per angular bin for each radial band
        - np.ndarray : Sigma (standard deviation) of speed per angular bin for each radial band

    """
    ylohi = calc_ylims(ycen_band_rs, r_band_half_width, arcsec_per_px)
    # Determine spike location (index of the correlation peak) in the cross-correlation array
    spike_location = np.where(x_kps < 0)[0].max() + 2

    avg_speeds = []
    sigmas = []
    for (ylo, yhi) in zip(*ylohi, strict=False):
        acc_k = acc[:, int(ylo):int(yhi) + 1, ...].mean(axis=1)
        # The modulus must be zero
        azimuth_bin_size = acc_k.shape[1] // velocity_azimuth_bins
        avcor_rbins_theta = acc_k.reshape(acc_k.shape[0], azimuth_bin_size, velocity_azimuth_bins)

        speedmax_idx_per_thbin = np.array(
            [avcor_rbins_theta[spike_location:, :, i].argmax(axis=0) +
             spike_location for i in range(velocity_azimuth_bins)])
        speedmax_per_theta = x_kps[speedmax_idx_per_thbin]
        avg_speeds.append(speedmax_per_theta.mean(axis=1))
        sigmas.append(speedmax_per_theta.std(axis=1) / np.sqrt(azimuth_bin_size))

    return np.array(avg_speeds), np.array(sigmas)


def process_corr(files: list, arcsec_per_px:float, expected_kps_windspeed: float, delta_t: int, sparsity: int,
                 delta_px: int, ycens: np.ndarray, r_band_half_width: float, n_ofs: int, max_radius_deg: int,
                 num_azimuth_bins: int, az_bin: int, velocity_azimuth_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Process the cross-correlation across frames  in a list of FITS files with associated average speeds.

    Parameters
    ----------
    files : list
        List of file paths to FITS files

    arcsec_per_px: float
        pixel scale in arcsec over the radial axis in the polar-remapped image

    expected_kps_windspeed: float
        Expected Wind Speed in km/s for narrowing the cross-correlation

    delta_t : float
        Time offset (in nb of frames) between for an image pair

    sparsity : int
        Interval between frames to skip when accumulating cross-correlation

    delta_px : int
        Pixel offset increment between samples

    ycens: np.ndarray
        y-coordinates of center of bands in solar radii

    r_band_half_width : float
        Half-width of each radial band in solar radii

    n_ofs : int
        Number of pixel offsets to use in cross-correlation

    max_radius_deg : int
        Maximum radius in degrees to include for polar remapping

    num_azimuth_bins : int
        Number of azimuthal samples to use in polar remapping

    az_bin: int
        Binning factor for binning the polar remapped image over the azimuth. The binning rule is currently numpy.mean()

    velocity_azimuth_bins : int
        Number of azimuthal bins in the output flow maps


    Returns
    -------
    [np.ndarray, np.ndarray]
        Average speed and 1-sigma uncertainty over radius and angular bins

    """
    # Expected windspeed in pixels
    time_cadence_sec = 4 * 60  # time cadence of punch images in seconds (4 minutes)
    arcsec_km = (1*u.arcsec).to(u.rad).value * 1*u.au.to(u.km)
    expected_px_windspeed =  expected_kps_windspeed / (arcsec_per_px * arcsec_km ) * time_cadence_sec
    # Central offset to start correlation from.
    central_offset = int(delta_t * expected_px_windspeed)
    # Calculate speed mapping for offsets in km/s
    x_pix = delta_px * (np.arange(n_ofs) - (n_ofs - 1) / 2) + central_offset
    x_kps = x_pix / central_offset * expected_kps_windspeed
    # Accumulate cross-correlation across frames
    acc = accumulate_cross_correlation_across_frames(files, delta_t, sparsity, n_ofs, max_radius_deg, num_azimuth_bins,
                                                     az_bin, delta_px, central_offset)
    # Compute average speeds and sigma for each radial band and latitudinal bin
    avg_speeds, sigmas = compute_all_bands(acc, ycens, r_band_half_width, arcsec_per_px, velocity_azimuth_bins, x_kps)

    return avg_speeds, sigmas


def plot_flow_map(filename: str | None, data: NDCube, cmap: str = "magma") -> None:
    """
    Plot polar maps of the radial flows.

    Parameters
    ----------
    filename: str
        Output plot filename. If None, the figure is not saved out.

    data: NDCube
        Flow tracking data NDCube

    cmap : str, optional
        Colormap for the plot (default is 'magma')

    Returns
    -------
    fig
        The generated Matplotlib Figure

    """
    speeds = data.data
    sigmas = data.uncertainty.array

    ycen_band_rs = np.fromstring(data.meta["YCENS"].value[1:-1], dtype=float, sep=",")
    rbands = np.fromstring(data.meta["RBANDS"].value[1:-1], dtype=int, sep=",")
    velocity_azimuth_bins = data.meta["PLTBINS"].value

    thetas = np.linspace(0, 2 * np.pi, velocity_azimuth_bins + 1)

    plt.close("all")
    fig = plt.figure(figsize=(20, 8))

    vmin = speeds.min()
    vmax = speeds.max()
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    for i, ridx in enumerate(rbands):

        signal = np.append(speeds[ridx], speeds[ridx][0])
        error = np.append(sigmas[ridx], sigmas[ridx][0])

        ax = fig.add_subplot(1, len(rbands), i + 1, projection="polar")
        ax.plot(thetas, signal, "k-")
        ax.fill_between(thetas, signal - error, signal + error, alpha=0.3, color="gray")

        colors = np.array([mapper.to_rgba(v) for v in signal])
        for theta, value, err, color in zip(thetas, signal, error, colors, strict=False):
            ax.plot(theta, value, "o", color=color, ms=4)
            ax.errorbar(theta, value, yerr=err, lw=2, capsize=3, color=color)

        ax.set_title(f"Altitude = {ycen_band_rs[ridx]} Rs")
        ax.set_ylim(50, 1.05 * vmax)
        ax.set_rlabel_position(270)

    cbar_ax = fig.add_axes([0.11, 0.2, 0.8, 0.03])
    plt.colorbar(mapper, cax=cbar_ax, orientation="horizontal").ax.set_xlabel("Speed (km/s)")
    if filename is not None:
        plt.savefig(filename)
    return fig


def track_velocity(files: list[str],
                   delta_t: int = 12,
                   sparsity: int = 2,
                   n_ofs: int = 151,
                   delta_px: int = 2,
                   expected_kps_windspeed: int = 300,
                   r_band_half_width: float = 0.5,
                   max_radius_deg: int = 45,
                   num_azimuth_bins: int = 1440*8,
                   az_bin: int = 4,
                   velocity_azimuth_bins: int = 36,
                   ycens: np.ndarray | None = None,
                   rbands: list[int] | None = None) -> NDCube:
    """
    Generate velocity map using flow tracking.

    Parameters
    ----------
    files : list[str]
        List of file paths for input data

    delta_t : int, optional
        Time offset in frames between images

    sparsity : int, optional
        Frame skip interval for averaging

    n_ofs : int, optional
        Number of spatial offsets for cross-correlation

    delta_px : int, optional
        Pixel offset increment per sample

    expected_kps_windspeed : int, optional
        Expected wind speed in km/s

    r_band_half_width : float, optional
        Half-width of each radial band in solar radii

    max_radius_deg : int, optional
        The maximum radius in degrees

    num_azimuth_bins : int, optional
        Number of azimuthal bins in the polar remapped images

    az_bin : int, optional
        Binning factor for binning the polar remapped image over the azimuth

    velocity_azimuth_bins : int, optional
        Number of azimuthal bins in the output flow maps

    ycens : numpy.ndarray, optional
        Radial band centers in solar radii

    rbands : list[int], optional
        Indices of radial bands to visualize

    Returns
    -------
    ndcube.NDCube
        The generated velocity map

    """
    # Set defaults for missing input parameters
    if ycens is None:
        ycens = np.arange(7, 14.5, 0.5)
    if rbands is None:
        rbands = [0, 4, 8, 14]

    files.sort()

    if len(files) < 2:
        msg = "At least to input files must be provided for flow tracking"
        raise ValueError(msg)

    # Data preprocessing
    data0 = load_ndcube_from_fits(files[0])
    header1 = data0.meta.to_fits_header(wcs=data0.wcs)
    _, polar_header1 = preprocess_image(data0, max_radius_deg/header1["CDELT1"], num_azimuth_bins, az_bin)

    avg_speeds, sigmas = process_corr(files, polar_header1["CDELT2"], expected_kps_windspeed,
                                      delta_t, sparsity, delta_px, ycens, r_band_half_width, n_ofs,
                                      max_radius_deg, num_azimuth_bins, az_bin, velocity_azimuth_bins)

    output_meta = NormalizedMetadata.load_template("VAM", "3")

    with fits.open(files[0]) as hdul:
        output_meta["DATE-BEG"] = hdul[1].header["DATE-BEG"]

    with fits.open(files[-1]) as hdul:
        output_meta["DATE-END"] = hdul[1].header["DATE-END"]

    date_beg = datetime.strptime(output_meta["DATE-BEG"].value, "%Y-%m-%dT%H:%M:%S").astimezone()
    date_end = datetime.strptime(output_meta["DATE-END"].value, "%Y-%m-%dT%H:%M:%S").astimezone()
    date_avg = (date_beg + (date_end - date_beg) / 2).strftime("%Y-%m-%dT%H:%M:%S")
    output_meta["DATE-AVG"] = date_avg
    output_meta["DATE-OBS"] = date_avg

    output_meta["DELTAT"] = delta_t
    output_meta["SPARSITY"] = sparsity
    output_meta["N_OFS"] = n_ofs
    output_meta["DELTA_PX"] = delta_px
    output_meta["KPSEXP"] = expected_kps_windspeed
    output_meta["BANDWDTH"] = r_band_half_width
    output_meta["MAXRAD"] = max_radius_deg
    output_meta["AZMBINS"] = num_azimuth_bins
    output_meta["AZMBINF"] = az_bin
    output_meta["PLTBINS"] = velocity_azimuth_bins
    output_meta["YCENS"] = np.array2string(ycens, separator=",", max_line_width=10_000)
    output_meta["RBANDS"] = np.array2string(np.array(rbands), separator=",", max_line_width=10_000)

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "radius","azimuth"
    wcs.wcs.cunit = "solRad","rad"
    wcs.wcs.cdelt = 1, 0.5
    wcs.wcs.crpix = 0, 0
    wcs.wcs.crval = 0, 0
    wcs.wcs.cname = "solar radii", "azimuth"
    wcs.array_shape = avg_speeds.shape

    return NDCube(data = avg_speeds,
                  uncertainty=StdDevUncertainty(sigmas),
                  meta = output_meta,
                  wcs = wcs)
