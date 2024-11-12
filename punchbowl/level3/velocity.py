import os
import glob

import cv2 as cv
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

cv.setNumThreads(0)  # Disable OpenCV multithreading
cv.ocl.setUseOpenCL(False)

# Solar radius in arcsec missing from PUNCH PTM header: RSUN_ARC=  / [arcsec] photospheric solar radius
RS_ARCSEC = 959.90  # As seen from Earth
ARCSEC_RAD = 4.85e-6  # rad
AU_KM = 150e6  # Astronomical unit in km
ARCSEC_KM = ARCSEC_RAD * AU_KM  # 1 arcsec in km at 1 a.u
TIME_CADENCE_SEC = 4 * 60  # time cadence of punch images in seconds (4 minutes)

def read_fits(filepath: str, hdu: int = 1) -> np.ndarray:
    """
    Read FITS image array.

    Parameters
    ----------
    filepath : str
        Path to the FITS file.
    hdu : int, optional
        Index of the header data unit storing the data array, by default 1.

    Returns
    -------
    np.ndarray, fits.header.Header
        2D image array from the specified FITS file.

    """
    with fits.open(filepath) as hdul:
        hdul.verify("fix")
        data = hdul[hdu].data
        header = hdul[hdu].header

    return data, header


def read_header(filepath: str, hdu: int = 1) -> fits.header.Header:
    """
    Read the header from a FITS file.

    Parameters
    ----------
    filepath : str
        Path to the FITS file.
    hdu : int, optional
        Index of the header data unit containing the header information, by default 1.

    Returns
    -------
    fits.header.Header
        Header object from the specified FITS file.

    """
    with fits.open(filepath) as hdul:
        hdul.verify("fix")
        header = hdul[hdu].header
    return header


def calc_ylims(ycen_band_rs: np.ndarray, r_band_width: float, arcsec_per_px: float) -> [int, int]:
    """
    Convert y-coordinates of lower and upper row of bands to array indices for slicing.

    Parameters
    ----------
    ycen_band_rs : np.ndarray
        y-coordinates of center of band in solar radii.

    r_band_width : float
        Half-width of each radial band in solar radii.

    arcsec_per_px : float
        Radial pixel scale in arcsec/px in the polar-remapped images

    Returns
    -------
    list
        Lower and upper Numpy array indices of the radial band.

    """
    # Unless we have a cropped image, bottom axis of the polar transform should be at 0 Rs.
    origin_rs = 0
    origin_arcsec = origin_rs * arcsec_per_px
    ycen_band_arcsec = ycen_band_rs * RS_ARCSEC   # center of the radial band in arcsec
    rband_width_arcsec = r_band_width * RS_ARCSEC  # width of the radial band in arcsec
    ylo_band_idx = ((ycen_band_arcsec - rband_width_arcsec) - origin_arcsec) / arcsec_per_px   # lower index of the band
    yhi_band_idx = ((ycen_band_arcsec + rband_width_arcsec) - origin_arcsec) / arcsec_per_px   # upper index of the band
    return [ylo_band_idx, yhi_band_idx]


def preprocess_image(image: np.ndarray, header: fits.header.Header, max_radius_px: int, num_azimuth_bins: int, az_bin: int) -> np.ndarray:
    """
    Normalize and preprocess FITS image by removing bad values and scaling.

    Parameters
    ----------
    image : np.ndarray
        Input FITS image array.

    header : fits.header.Header
        FITS header

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
    # image[np.abs(image) > 100] = 0
    image[~np.isfinite(image)] = 0

    polar_image = cv.warpPolar(image.astype(np.float64), [int(max_radius_px), int(num_azimuth_bins)], [header["CRPIX1"], header["CRPIX2"]], max_radius_px, cv.INTER_CUBIC)

    polar_image_binned = polar_image.T.reshape([polar_image.shape[1], polar_image.shape[0] // az_bin, az_bin]).mean(axis=2)
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


def calculate_cross_correlation(image1: np.ndarray, image2: np.ndarray, offsets: np.ndarray, delta_px: int, central_offset: int) -> np.ndarray:
    """
    Perform cross-correlation for a range of offsets.

    Parameters
    ----------
    image1 : np.ndarray
        First image array for correlation.

    image2 : np.ndarray
        Second, time-offseted image array for correlation.

    offsets : np.ndarray
        Array of pixel offsets to iterate over for cross-correlation.

    delta_px : int
        Pixel offset increment between samples.

    central_offset : int
        Central offset to start correlation from.

    Returns
    -------
    np.ndarray
        Accumulated cross-correlation array over all offsets.

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

        # Padding the images symmetrically according to the direction of the shift. The padding rule is to replicate the values at the nearest edge
        if offset_1 < 0:
            padded_image1 = np.pad(image1, ((0, -offset_1), (0, 0)), mode="edge")[abs(offset_1):image1.shape[0] + abs(offset_1), :]
        else:
            padded_image1 = np.pad(image1, ((offset_1, 0), (0, 0)), mode="edge")[:image1.shape[0], :]

        if offset_2 < 0:
            padded_image2 = np.pad(image2, ((-offset_2, 0), (0, 0)), mode="edge")[:image2.shape[0], :]
        else:
            padded_image2 = np.pad(image2, ((0, offset_2), (0, 0)), mode="edge")[offset_2:image2.shape[0] + offset_2, :]

        acc[jj, :, :] += padded_image1 * padded_image2

    return acc



def accumulate_cross_correlation_across_frames(files: list, hdu: int, delta_t: int, sparsity: int, n_ofs: int,
                                               max_radius_deg: int, num_azimuth_bins: int, az_bin: int,
                                               delta_px: int, central_offset: int) -> np.ndarray:
    """
    Accumulate cross-correlation across frames in a list of FITS files.

    Parameters
    ----------
    files : list
        List of file paths to FITS files.

    hdu : int
        Index of the header data unit storing the data array.

    delta_t : int
        Frame offset (in frames) between time-offset image pairs.

    sparsity : int
        Interval between frames to skip when accumulating cross-correlation.

    n_ofs : int
        Number of pixel offsets to use in cross-correlation.

    max_radius_deg : int
        Maximum radius in degrees to include for polar remapping

    num_azimuth_bins : int
        Number of azimuthal samples to use in polar remapping

    az_bin: int
        Binning factor for binning the polar remapped image over the azimuth. The binning rule is currently numpy.mean()

    delta_px : int
        Pixel offset increment between samples.

    central_offset : int
        Central offset to start correlation from.


    Returns
    -------
    np.ndarray
        Accumulated cross-correlation array over all frames and offsets.

    """
    sample, header = read_fits(files[0], hdu=hdu)
    max_radius_px = max_radius_deg / header["CDELT1"]
    polar_sample, _ = preprocess_image(sample, header, max_radius_px, num_azimuth_bins, az_bin)

    acc = np.zeros((n_ofs, polar_sample.shape[0], polar_sample.shape[1]), dtype=float)
    n = 0
    for i in range(0, len(files) - delta_t, delta_t * sparsity):
        print(f"Frame {i} vs frame {i + delta_t}")

        image1, header1 = read_fits(files[i], hdu=hdu)
        image2, header2 = read_fits(files[i + delta_t], hdu=hdu)
        prepped_image1, _ = preprocess_image(image1, header1, max_radius_px, num_azimuth_bins, az_bin)
        prepped_image2, _ = preprocess_image(image2, header2, max_radius_px, num_azimuth_bins, az_bin)

        acc += calculate_cross_correlation(prepped_image1, prepped_image2, np.arange(n_ofs), delta_px, central_offset)

        n += 1

    acc /= n

    return acc


def compute_all_bands(acc: np.ndarray, ycen_band_rs: np.ndarray, r_band_half_width: float, arcsec_per_px: float, velocity_azimuth_bins: int,
                      x_kps: np.ndarray):
    """
    Compute speed and sigma for all radial bands.

    Parameters
    ----------
    acc : np.ndarray
        Cross-correlation array accumulated across frames.

    ycen_band_rs : np.ndarray
        y-coordinates of band centers in solar radii.

    r_band_half_width : float
        Half-width of each radial band in solar radii.

    arcsec_per_px : float
        Radial pixel scale in arcsec/px in the polar-remapped images

    velocity_azimuth_bins : int
        Number of azimuthal bins in the output flow maps

    x_kps : np.ndarray
        Array mapping pixel offsets to speed in km/s.

    Returns
    -------
    tuple
        Tuple containing:
        - np.ndarray : Average speed per angular bin for each radial band.
        - np.ndarray : Sigma (standard deviation) of speed per angular bin for each radial band.

    """
    ylohi = calc_ylims(ycen_band_rs, r_band_half_width, arcsec_per_px)
    # Determine spike location (index of the correlation peak) in the cross-correlation array
    spike_location = np.where(x_kps < 0)[0].max() + 2

    avg_speeds = []
    sigmas = []
    for kk, (ylo, yhi) in enumerate(zip(*ylohi, strict=False)):
        acc_k = acc[:, int(ylo):int(yhi) + 1, ...].mean(axis=1)
        # The modulus must be zero
        azimuth_bin_size = acc_k.shape[1] // velocity_azimuth_bins
        avcor_rbins_theta = acc_k.reshape(acc_k.shape[0], azimuth_bin_size, velocity_azimuth_bins)

        speedmax_idx_per_thbin = np.array(
            [avcor_rbins_theta[spike_location:, :, i].argmax(axis=0) + spike_location for i in range(velocity_azimuth_bins)])
        speedmax_per_theta = x_kps[speedmax_idx_per_thbin]
        avg_speeds.append(speedmax_per_theta.mean(axis=1))
        sigmas.append(speedmax_per_theta.std(axis=1) / np.sqrt(azimuth_bin_size))

    return np.array(avg_speeds), np.array(sigmas)

def process_corr(files: list, hdu: int, arcsec_per_px:float, expected_kps_windspeed: float, delta_t: int, sparsity: int,
                 delta_px: int, r_band_half_width: float, n_ofs: int, max_radius_deg: int, num_azimuth_bins: int,
                 az_bin: int, velocity_azimuth_bins: int):
    """
    Process the cross-correlation across frames  in a list of FITS files with associated average speeds

    Parameters
    ----------
    files : list
        List of file paths to FITS files.

    hdu : int
        Index of the header data unit storing the data array.

    arcsec_per_px: float
        pixel scale in arcsec over the radial axis in the polar-remapped image

    expected_kps_windspeed: float
        Expected Wind Speed in km/s for narrowing the cross-correlation

    delta_t : float
        Time offset (in nb of frames) between for an image pair

    sparsity : int
        Interval between frames to skip when accumulating cross-correlation.

    delta_px : int
        Pixel offset increment between samples.

    r_band_half_width : float
        Half-width of each radial band in solar radii.

    n_ofs : int
        Number of pixel offsets to use in cross-correlation.

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
    expected_px_windspeed =  expected_kps_windspeed / (arcsec_per_px * ARCSEC_KM ) * TIME_CADENCE_SEC
    # Central offset to start correlation from.
    central_offset = int(delta_t * expected_px_windspeed)
    # Calculate speed mapping for offsets in km/s
    x_pix = delta_px * (np.arange(n_ofs) - (n_ofs - 1) / 2) + central_offset
    x_kps = x_pix / central_offset * expected_kps_windspeed
    # Accumulate cross-correlation across frames
    acc = accumulate_cross_correlation_across_frames(files, hdu, delta_t, sparsity, n_ofs, max_radius_deg, num_azimuth_bins, az_bin, delta_px, central_offset)
    # Compute average speeds and sigma for each radial band and latitudinal bin
    avg_speeds, sigmas = compute_all_bands(acc, ycens, r_band_half_width, arcsec_per_px, velocity_azimuth_bins, x_kps)

    return avg_speeds, sigmas


def plot_flow_map(speeds: np.ndarray, sigmas: np.ndarray, ycen_band_rs: np.ndarray, rbands: list[int], velocity_azimuth_bins: int,
                  cmap: str = "inferno"):
    """
    Plot polar maps of the radial flows.

    Parameters
    ----------
    speeds : np.ndarray
        Averaged speed over each radial band and latitudinal bin.
    sigmas : np.ndarray
        1-sigma uncertainty associated with each binned speed.
    ycen_band_rs : np.ndarray
        y-coordinates of center of bands in solar radii.
    rbands : list[int]
        Indices of the radial bands to visualize.
    velocity_azimuth_bins : int
        Number of angular bins in the velocity map.
    cmap : str, optional
        Colormap for the plot (default is 'inferno').

    """
    thetas = np.linspace(0, 2 * np.pi, velocity_azimuth_bins + 1)

    plt.close("all")
    fig = plt.figure(figsize=(20, 8))

    vmin = speeds.min()
    vmax = speeds.max()
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
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
    plt.savefig("Radial_Speed_Map.png")


if __name__ == "__main__":
    # Input parameters and configuration
    files = sorted(glob.glob(os.path.join(os.environ["PUNCHDATA"], "*")))
    hdu = 1  # Index of the header data unit (change if necessary)
    delta_t = 12  # Time offset in frames between images
    sparsity = 2  # Frame skip interval for averaging
    n_ofs = 151  # Number of spatial offsets for cross-correlation
    delta_px = 2  # Pixel offset increment per sample
    expected_kps_windspeed = 300  # Expected wind speed in km/s
    r_band_half_width = 0.5  # Half-width of each radial band in solar radii
    max_radius_deg = 45  # The maximum radius is 45 degrees
    num_azimuth_bins = int(1440 * 8)  # Number of azimuthal bins in the polar remapped images
    az_bin = 4  # Binning factor for binning the polar remapped image over the azimuth.
    velocity_azimuth_bins = 36  # Number of azimuthal bins in the output flow maps
    # Define radial band centers in solar radii
    ycens = np.array([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14])
    rbands = [0, 4, 8, 14]  # Indices of radial bands to visualize
    # End of user input

    # Testing preprocessing
    image1, header1 = read_fits(files[0], hdu=hdu)
    prepped_image1, polar_header1 = preprocess_image(image1, header1, max_radius_deg/header1["CDELT1"], num_azimuth_bins, az_bin)

    avg_speeds, sigmas = process_corr(files, hdu, polar_header1["CDELT2"], expected_kps_windspeed, delta_t, sparsity, delta_px, r_band_half_width, n_ofs, max_radius_deg,
                                      num_azimuth_bins, az_bin, velocity_azimuth_bins)
    #
    # # Save the speed and sigma data in a FITS file
    data_cube = np.stack((avg_speeds, sigmas), axis=0)
    hdu = fits.PrimaryHDU(data_cube)
    hdu.writeto("speeds_sigmas.fits", overwrite=True)

    # Generate and save the radial flow map plot
    plot_flow_map(avg_speeds, sigmas, ycens, rbands, velocity_azimuth_bins)
