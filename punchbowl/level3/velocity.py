import math
import warnings
from bisect import bisect_left
from pathlib import Path
from datetime import datetime, timedelta

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sunpy.coordinates import sun

from punchbowl.data import load_ndcube_from_fits
from punchbowl.data.meta import NormalizedMetadata
from punchbowl.data.punchcube import PUNCHCube
from punchbowl.prefect import get_logger, punch_flow

# Time interval (minutes) between consecutive frames for each data product usable for flow tracking
_TCADENCE_MIN = {
    "CAM": 32,
    "CTM": 8,
    "PTM": 4,
    "PAM": 32,
}

# Tolerated missing frames per ~6-hour span, which are filled by linear interpolation: longest run of consecutive
# missing frames, maximum total of missing frames, and an alternative accepted pattern of runs.
_MISSING_FRAME_RULES = {
    "CAM": {"max_run": 1, "max_total": 2, "accepted_runs": None},
    "PAM": {"max_run": 1, "max_total": 2, "accepted_runs": None},
    "CTM": {"max_run": 2, "max_total": 3, "accepted_runs": [2, 2]},
    "PTM": {"max_run": 2, "max_total": 3, "accepted_runs": [2, 2]},
}


def get_cadence_min(product_code: str) -> int:
    """
    Get the time interval in minutes between consecutive frames of a data product usable for flow tracking.

    Parameters
    ----------
    product_code : str
        Either 'CAM', 'PAM', 'CTM', or 'PTM'

    Returns
    -------
    int
        Cadence of the data product in minutes

    """
    if product_code not in _TCADENCE_MIN:
        msg = f"Unsupported product code {product_code!r} for flow tracking, expected one of {list(_TCADENCE_MIN)}"
        raise ValueError(msg)
    return _TCADENCE_MIN[product_code]


def get_time_avg_size(product_code: str, time_avg_hours: float) -> int:
    """
    Get the size in frames of the running time-average kernel spanning about ``time_avg_hours``.

    The size is odd so that the kernel is centered on each frame.

    Parameters
    ----------
    product_code : str
        Either 'CAM', 'PAM', 'CTM', or 'PTM'
    time_avg_hours : float
        Duration of the running time-average kernel in hours

    Returns
    -------
    int
        Odd number of frames of the running time-average kernel

    """
    n_frames = round(time_avg_hours * 60 / get_cadence_min(product_code))
    return 2 * (n_frames // 2) + 1


def get_delta_t_frames(product_code: str, delta_t_minutes: float) -> int:
    """
    Get the time lag in frames between the two frames of each correlated pair, nearest to ``delta_t_minutes``.

    Parameters
    ----------
    product_code : str
        Either 'CAM', 'PAM', 'CTM', or 'PTM'
    delta_t_minutes : float
        Time lag in minutes between the two frames of each correlated pair

    Returns
    -------
    int
        Time lag in frames, at least 1

    """
    n_frames = round(delta_t_minutes / get_cadence_min(product_code))
    if n_frames < 1:
        msg = (f"delta_t_minutes={delta_t_minutes} is shorter than half the {product_code} cadence of "
               f"{get_cadence_min(product_code)} min, so the correlated frames would have no time lag")
        raise ValueError(msg)
    return n_frames


def get_frames_per_window(product_code: str, flow_window_hours: float) -> int:
    """
    Get the number of frames of the flow map integration window spanning at least ``flow_window_hours``.

    Parameters
    ----------
    product_code : str
        Either 'CAM', 'PAM', 'CTM', or 'PTM'
    flow_window_hours : float
        Minimum duration of the flow map integration window in hours

    Returns
    -------
    int
        Number of frames in the flow map integration window

    """
    # Round first so that floating point errors do not add a frame, e.g. 0.4 h at 4 min -> 6.000000000000001
    return math.ceil(round(flow_window_hours * 60 / get_cadence_min(product_code), 6))


def get_required_frame_count(product_code: str,
                             flow_window_hours: float,
                             time_avg_hours: float) -> tuple[int, int, int]:
    """
    Get the number of input frames required by flow tracking and the position of the flow window among them.

    The flow map integration window is centered in the timeline of the input files, with a temporal-average
    buffer on either side:
    [ta start......ft start......ft end.......ta end]
    The flow window [tstart, tend] is inclusive and holds exactly ``get_frames_per_window`` frames.
    Each side of the buffer is half of the running time-average kernel (see :func:`get_time_avg_size`),
    so that the kernel centered on any frame of the flow window only covers actual frames, without edge effects.

    Parameters
    ----------
    product_code : str
        Either 'CAM', 'PAM', 'CTM', or 'PTM'
    flow_window_hours : float
        Minimum duration of the flow map integration window in hours
    time_avg_hours : float
        Duration of the running time-average kernel in hours

    Returns
    -------
    tuple[int, int, int]
        Number of required files, and the start and end indices (inclusive) of the flow window among them

    """
    frames_per_window = get_frames_per_window(product_code, flow_window_hours)
    # The time-average kernel has an odd size, so it slides evenly over the flow window
    half_buffer = get_time_avg_size(product_code, time_avg_hours) // 2
    tstart = half_buffer
    tend = tstart + frames_per_window - 1
    n_files = tend + half_buffer + 1
    return n_files, tstart, tend


def get_flow_window_start(obs_times: list[datetime], reference_time: datetime, frames_per_window: int) -> int:
    """
    Get the index of the first frame of the flow window centered on the reference time.

    The flow window [tstart, tend] (inclusive) holds ``frames_per_window`` frames, with
    ``frames_per_window // 2`` frames on either side of its middle:

    - odd ``frames_per_window``: the middle frame is the one nearest to the reference time.
    - even ``frames_per_window``: the reference time falls between the two middle frames, i.e. the
      second middle frame is the first one observed at or after the reference time.

    Parameters
    ----------
    obs_times : list[datetime]
        Observation times of the frames, in chronological order
    reference_time : datetime
        Authoritative reference time on which to center the flow window
    frames_per_window : int
        Number of frames in the flow window

    Returns
    -------
    int
        Index of the first frame of the flow window in ``obs_times``. It may be out of the range of
        ``obs_times`` if there are not enough frames before the reference time.

    """
    i_mid = bisect_left(obs_times, reference_time)
    if frames_per_window % 2 == 1 and (
            i_mid == len(obs_times)
            or (i_mid > 0 and reference_time - obs_times[i_mid - 1] <= obs_times[i_mid] - reference_time)):
        # The previous frame is the nearest one (ties go to the earlier frame)
        i_mid -= 1
    return i_mid - frames_per_window // 2


def get_frame_slots(obs_times: list[datetime], cadence_min: float) -> tuple[list[int | None], list[datetime]]:
    """
    Lay out chronological frames on a regular sequence of slots at the product cadence.

    Missing frames get empty slots. Their number between two consecutive frames is inferred from the time
    difference of these frames, which is robust to small timing jitter and slow drifts of the cadence.

    Parameters
    ----------
    obs_times : list[datetime]
        Observation times of the frames, in chronological order
    cadence_min : float
        Time interval between consecutive frames in minutes

    Returns
    -------
    tuple[list[int | None], list[datetime]]
        For each slot, the index of its frame in ``obs_times`` or None for a missing frame, and the
        time of each slot, linearly interpolated between the neighboring frames for missing ones.

    """
    cadence = timedelta(minutes=cadence_min)
    slots, slot_times = [], []
    for i, obs_time in enumerate(obs_times):
        if i > 0:
            n_missing = max(round((obs_time - obs_times[i - 1]) / cadence) - 1, 0)
            step = (obs_time - obs_times[i - 1]) / (n_missing + 1)
            slots.extend([None] * n_missing)
            slot_times.extend(obs_times[i - 1] + k * step for k in range(1, n_missing + 1))
        slots.append(i)
        slot_times.append(obs_time)
    return slots, slot_times


def _missing_runs(slots: list[int | None], start: int, stop: int) -> list[int]:
    """Get the full lengths of the runs of missing slots overlapping slots[start:stop]."""
    runs = []
    i = start
    while i < stop:
        if slots[i] is None:
            lo = hi = i
            while lo > 0 and slots[lo - 1] is None:
                lo -= 1
            while hi + 1 < len(slots) and slots[hi + 1] is None:
                hi += 1
            runs.append(hi - lo + 1)
            i = hi + 1
        else:
            i += 1
    return runs


def check_missing_runs(runs: list[int], product_code: str) -> bool:
    """
    Check whether runs of consecutive missing frames over a ~6-hour span can be filled by interpolation.

    - CAM, PAM: runs of 1 frame, and at most 2 missing frames.
    - CTM, PTM: runs of at most 2 frames, and at most 3 missing frames, or exactly two runs of 2 frames.

    Parameters
    ----------
    runs : list[int]
        Lengths of the runs of consecutive missing frames
    product_code : str
        Either 'CAM', 'PAM', 'CTM', or 'PTM'

    Returns
    -------
    bool
        True if the missing frames are tolerated

    """
    # Raises a ValueError for products unsupported by flow tracking
    get_cadence_min(product_code)
    rules = _MISSING_FRAME_RULES[product_code]
    if any(run > rules["max_run"] for run in runs):
        return False
    return sum(runs) <= rules["max_total"] or sorted(runs) == rules["accepted_runs"]


def select_frame_slots(slots: list[int | None],
                       slot_times: list[datetime],
                       product_code: str,
                       flow_window_hours: float,
                       time_avg_hours: float,
                       reference_time: datetime | None = None) -> tuple[int, int, int, int]:
    """
    Select the slots required by flow tracking, with the flow window centered on the reference time.

    The missing frames are checked separately in the flow window and in the temporal-average buffer
    (see :func:`check_missing_runs`). When the first or last required slot is missing, the selection is
    extended to the nearest actual frame so that the missing ones can be interpolated.

    Parameters
    ----------
    slots : list[int | None]
        Slots from :func:`get_frame_slots`
    slot_times : list[datetime]
        Slot times from :func:`get_frame_slots`
    product_code : str
        Either 'CAM', 'PAM', 'CTM', or 'PTM'
    flow_window_hours : float
        Minimum duration of the flow map integration window in hours
    time_avg_hours : float
        Duration of the running time-average kernel in hours
    reference_time : datetime, optional
        Time on which to center the flow window (see :func:`get_flow_window_start`). If None, the
        selection starts at the first slot.

    Returns
    -------
    tuple[int, int, int, int]
        First and last (inclusive) selected slots, and the start and end indices (inclusive) of the flow
        window relative to the first selected slot.

    """
    n_required, tstart, tend = get_required_frame_count(product_code, flow_window_hours, time_avg_hours)
    if reference_time is None:
        first = 0
    else:
        first = get_flow_window_start(slot_times, reference_time, tend - tstart + 1) - tstart
    last = first + n_required - 1
    if first < 0 or last >= len(slots):
        msg = f"Not enough frames for flow tracking: {n_required} frames are required"
        if reference_time is not None:
            msg += f" centered on reference time {reference_time.isoformat()}"
        raise ValueError(msg)

    window_runs = _missing_runs(slots, first + tstart, first + tend + 1)
    buffer_runs = _missing_runs(slots, first, first + tstart) + _missing_runs(slots, first + tend + 1, last + 1)
    for name, runs in (("flow window", window_runs), ("temporal-average buffer", buffer_runs)):
        if not check_missing_runs(runs, product_code):
            msg = f"Too many missing {product_code} frames in the {name} for flow tracking: runs of {runs} frames"
            raise ValueError(msg)

    # Extend to the nearest actual frames to interpolate the missing ones at the edges
    while slots[first] is None:
        first -= 1
        tstart += 1
        tend += 1
        if first < 0:
            msg = "No frame before the first missing frames to interpolate them"
            raise ValueError(msg)
    while slots[last] is None:
        last += 1
        if last >= len(slots):
            msg = "No frame after the last missing frames to interpolate them"
            raise ValueError(msg)
    return first, last, tstart, tend


def infer_product_code(files: list[str] | list[Path], hdu: int = 1) -> str:
    """
    Infer the PUNCH product code shared by all the input files from their headers.

    Parameters
    ----------
    files : list[str]
        Input file paths
    hdu : int, optional
        Position of the header data unit in the FITS files holding the data

    Returns
    -------
    str
        Product code of the files, e.g. 'PTM'

    """
    product_codes = set()
    for file in files:
        header = fits.getheader(file, hdu)
        product_codes.add(f"{header['TYPECODE']}{header['OBSCODE']}")
    if len(product_codes) != 1:
        msg = f"Flow tracking input files must all be of the same product type, got {sorted(product_codes)}"
        raise ValueError(msg)
    product_code = product_codes.pop()
    # Raises a ValueError for products unsupported by flow tracking
    get_cadence_min(product_code)
    return product_code


# ---------------------------------------------------------------------------- #
# Coordinate utilities                                                          #
# ---------------------------------------------------------------------------- #

def get_annulus(ycen_band_rs: float, r_band_width: float,
                arcsec_per_px: float, rs_arcsec: float) -> np.ndarray:
    """
    Convert an annulus, defined by its center and radial width, to pixel indices of its inner and outer edges.

    Works only for polar-remapped image with center of transformation at sun center, with its origin at the
    bottom of the horizontal axis.

    Parameters
    ----------
    ycen_band_rs : float
        Center of the radial band in solar radii.
    r_band_width : float
        width of the radial band in solar radii.
    arcsec_per_px : float
        Radial pixel scale of the polar-remapped image in arcsec/pixel.
    rs_arcsec : float
        angular radius of the sun (arcsec) as seen from Earth

    Returns
    -------
    ndarray
        ``[lower_index, upper_index]`` pixel coordinates of lower and outer edge of the annuli.
        Guaranteed to span at least one pixel.

    """
    ycen_arcsec = ycen_band_rs * rs_arcsec
    half_width_arcsec = r_band_width * rs_arcsec / 2

    ylow = int(np.round((ycen_arcsec - half_width_arcsec) / arcsec_per_px))
    yhigh = int(np.round((ycen_arcsec + half_width_arcsec) / arcsec_per_px))

    if ylow == yhigh:
        yhigh += 1

    return np.array([ylow, yhigh])


def build_velocity_axis(
        n_ofs: int,
        central_offset: int,
        expected_wind_kps: float,
        delta_px: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct the pixel-offset and equivalent km/s velocity axes.

    Parameters
    ----------
    n_ofs : int
        Number of offset samples.
    central_offset : int
        Pixel offset corresponding to the expected wind speed.
    expected_wind_kps : float
        Wind speed used to calibrate the velocity axis (km/s).
    delta_px : int, optional
        Pixel increment per sample.

    Returns
    -------
    x_pix : np.ndarray
        Pixel offset at each sample.
    x_kps : np.ndarray
        Corresponding projected speed in km/s.

    """
    x_pix = delta_px * (np.arange(n_ofs) - (n_ofs - 1) / 2) + central_offset
    x_kps = x_pix / central_offset * expected_wind_kps
    return x_pix, x_kps


# ---------------------------------------------------------------------------- #
# Image preprocessing                                                           #
# ---------------------------------------------------------------------------- #


def polar_remap(
        image: np.ndarray,
        header: fits.header.Header,
        polar_nr: int,
        num_azimuth_bins: int,
        az_bin: int,
        rotate90: bool = False,
        crop: list | None = None,
        polar_header: bool = False,
) -> np.ndarray | tuple:
    """
    Remap a PUNCH WFI image from Cartesian to polar coordinates.

    The OpenCV ``warpPolar`` routine is centred on ``(CRPIX1, CRPIX2)`` from
    the FITS header.  The image is then low-pass filtered along the azimuthal
    axis (anti-aliasing) and decimated by a factor of ``az_bin``.

    Parameters
    ----------
    image : np.ndarray
        Input PUNCH WFI image array.  Non-finite values are replaced with 0
        in-place before remapping.
    header : fits.header.Header
        FITS header providing the image reference pixel (CRPIX1, CRPIX2).
    polar_nr : int
        Maximum radial extent for the polar remap in pixels.
    num_azimuth_bins : int
        Number of azimuthal samples in the polar-remapped image before binning.
        Must be divisible by ``az_bin``.
    az_bin : int
        Azimuthal binning factor; columns are averaged in groups of ``az_bin``.
    rotate90: bool
        Rotate the angular origin by 90 degrees counter-clockwise with respect to unit circle origin.
    crop : list of int, optional
        ``[row_low, row_high]`` radial rows retained after remapping and
        binning.  If ``None``, the full radial range is retained.
    polar_header : bool, optional
        If ``True``, also return a WCS-like metadata dictionary describing the
        remapped grid.  Default is ``False``.

    Returns
    -------
    polar_image_binned : np.ndarray
        Polar-remapped and azimuthally binned image,
        shape ``(polar_nr, num_azimuth_bins // az_bin)``.
    polar_meta : dict
        WCS-like metadata for the remapped grid.  Only returned when
        ``polar_header=True``.

    """
    # OpenCV is an optional (super_user) dependency, only needed for flow tracking
    import cv2 as cv  # noqa: PLC0415

    image[~np.isfinite(image)] = 0

    polar_image = cv.warpPolar(
        image.astype(np.float64),
        [int(polar_nr), int(num_azimuth_bins)],
        [header["CRPIX1"], header["CRPIX2"]],
        polar_nr,
        cv.INTER_CUBIC,
    )

    if rotate90:
        # Rotate by a quarter circle counter-clockwise so the segment pointing up becomes the angular origin
        polar_image = np.roll(polar_image, shift=-int(num_azimuth_bins) // 4, axis=0)

    # Anti-Aliasing Filter (Low-pass before downsampling/binning) along the azimuth axis (axis=0)
    # mode='wrap' ensures 0 degrees smoothly blurs into 360 degrees
    sigma = az_bin / 2.0
    filtered_polar = gaussian_filter1d(polar_image, sigma=sigma, axis=0, mode="wrap")

    # Decimate by binning, then transpose
    # Resulting Shape: (polar_nr, num_azimuth_bins // az_bin)
    polar_image_binned = filtered_polar[::az_bin, :].T

    if crop is not None:
        # Crop only along the rows
        polar_image_binned = polar_image_binned[crop[0]:crop[1], :]

    if not polar_header:
        return polar_image_binned

    # NOTE: polar WCS is provisional and pending SOC review.
    arcsec_per_radial_px = header["CDELT1"] * 3600
    polar_meta = {
        "NAXIS": 2,
        "NAXIS1": polar_image_binned.shape[0],
        "CTYPE1": "ELONG",
        "CDELT1": arcsec_per_radial_px,
        "CUNIT1": "arcsec",
        "CRPIX1": polar_image_binned.shape[0] // 2 + 0.5,
        "CRVAL1": (polar_image_binned.shape[0] // 2 + 0.5) * arcsec_per_radial_px,
        "NAXIS2": polar_image_binned.shape[1],
        "CTYPE2": "POS_ANGLE",
        "CDELT2": 360 / polar_image_binned.shape[1],
        "CUNIT2": "deg",
        "CRPIX2": 0.5,
        "CRVAL2": 0,
        "DATE-OBS": header["DATE-OBS"],
    }

    return polar_image_binned, polar_meta


def remove_az_gain(cube: np.ndarray, ref: np.ndarray, az_smooth: float = 0.0) -> np.ndarray:
    """
    Remove per-frame, per-azimuth gain flicker g(t, theta), assumed constant along radius.

    The gain is estimated as the median over the radial axis of the ratio
    ``cube / ref``, optionally smoothed along azimuth, then divided out of
    the cube.

    Parameters
    ----------
    cube : np.ndarray
        Polar-remapped frames of shape (n_t, n_r, n_az), NOT standardized.
    ref : np.ndarray
        Reference cube or image against which the gain g(t, theta) is
        estimated (e.g. a temporal-median reference cube).
    az_smooth : float, optional
        Gaussian sigma (in azimuth bins) used to smooth g along azimuth.
        Default 0 (no smoothing).

    Returns
    -------
    np.ndarray
        Gain-corrected cube, same shape as ``cube``.

    """
    with np.errstate(divide="ignore", invalid="ignore"):
        q = cube / ref
    q = np.where(np.isfinite(q), q, np.nan)  # masks 0/0 outside the FOV
    g = np.nanmedian(q, axis=1)  # (n_t, n_az)
    if az_smooth:
        g = gaussian_filter1d(g, az_smooth, axis=1, mode="wrap")

    return cube / g[:, None, :]


def _interpolate_missing_frames(cube: list, headers: list) -> None:
    """
    Fill in place the missing frames (None) by linear interpolation in time between the nearest frames on either side.

    The headers of missing frames are copies of the previous frame's header with an interpolated DATE-OBS.
    """
    present = [i for i, image in enumerate(cube) if image is not None]
    for i in range(len(cube)):
        if cube[i] is not None:
            continue
        k = bisect_left(present, i)
        if k in (0, len(present)):
            msg = f"Missing frame {i} cannot be interpolated without frames on both sides"
            raise ValueError(msg)
        lo, hi = present[k - 1], present[k]
        weight = (i - lo) / (hi - lo)
        cube[i] = (1 - weight) * cube[lo] + weight * cube[hi]
        obs_lo = datetime.fromisoformat(headers[lo]["DATE-OBS"])
        obs_hi = datetime.fromisoformat(headers[hi]["DATE-OBS"])
        headers[i] = headers[lo].copy()
        headers[i]["DATE-OBS"] = (obs_lo + weight * (obs_hi - obs_lo)).isoformat(timespec="milliseconds")


def preprocess_cube(
        files: list,
        product: str,
        polar_nr: int | None = None,
        num_azimuth_bins: int | None = None,
        az_bin: int = 1,
        rotate90: bool = False,
        crop: list | None = None,
        do_polar_remap: bool = True,
        deflicker: bool = False,
        az_smooth: float = 0,
        remove_temporal_median: bool = False,
        time_win: list | None = None,
        time_avg_size: int | None = None,
) -> tuple:
    """
    Load a time series of FITS frames into a single preprocessed cube.

    Each frame is read and, optionally, polar-remapped. Frames are stacked into a cube
    of shape (n_t, n_rows, n_cols) and corrected for the per-frame,
    per-azimuth gain flicker g(t, theta) (assumed constant along radius)
    estimated against the temporal-median reference via :func:`remove_az_gain`.
    Frames are NOT standardized here; per-pair standardization is left to the
    caller.

    This function is meant to have some buffer frames that are cropped out to have
    the temporal median applied evenly on both ends without any duplicated edge frames

    Parameters
    ----------
    files : list of Path or str or None
        Sorted list of FITS file paths to process, one per frame at the product cadence. None marks a missing
        frame, which is linearly interpolated in time between the nearest frames on either side.
    product : str
        PUNCH data product code of the input files (``'CAM'``, ``'PAM'``,
        ``'CTM'`` or ``'PTM'``).  For 3-D polarized/total-brightness cubes
        (PAM/PTM), only the first (total brightness) layer is used.
    polar_nr : int, optional
        Maximum radial extent of the polar remap in pixels.  Required when
        ``do_polar_remap=True``; ignored otherwise.
    num_azimuth_bins : int, optional
        Number of azimuthal samples before binning.  Required when
        ``do_polar_remap=True``; ignored otherwise.
    az_bin : int, optional
        Azimuthal binning factor applied by :func:`polar_remap`.  Default 1.
    rotate90 : bool, optional
        Rotate the angular origin by 90 degrees counter-clockwise.  Default False.
    crop : list of int, optional
        ``[row_low, row_high]`` radial crop applied after remapping, e.g. the
        margin-padded ``effective_crop`` from :func:`accumulate_cross_correlation_across_frames`.
        Default None.
    do_polar_remap : bool, optional
        If True, polar-remap each frame; set False for already-remapped files.
    deflicker: bool, optional
        If True, will compensate for time-azimuthal gain changes by dividing an estimate based on
        base difference and smoothed radial average.
    az_smooth : float, optional
        Gaussian sigma (azimuth bins) regularizing the gain g(t, theta).
        Default 10.
    remove_temporal_median: bool, optional
        If True: will subtract the median of the cube. If Deflicker is applied, the median runs
        after it.
        Default False
    time_win: tuple, optional
        (start, end) indices (inclusive) of the time slice of interest in the input time series.
        This is useful to not have edge effects from the sliding temporal median
    time_avg_size: int, optional
        Size of the running time averaging kernel in number of frames. Default is the number of files.
        Ignored if both deflicker and remove_temporal_median are False.

    Returns
    -------
    cube : np.ndarray
        Gain-corrected, non-standardized frames, shape (n_t, n_rows, n_cols).
    headers : list of fits.header.Header
        FITS headers in the order of ``files``.

    """
    if time_win is None:
        time_win = (0, len(files) - 1)

    if time_avg_size is None:
        time_avg_size = len(files)

    cube = [None] * len(files)
    headers = [None] * len(files)
    for i, file in enumerate(files):
        if file is None:
            continue
        data = load_ndcube_from_fits(file)
        image = data.data[0, :, :] if product in ("PAM", "PTM") and data.data.ndim >= 3 else data.data
        header = data.meta.to_fits_header(wcs=data.wcs)
        headers[i] = header
        if do_polar_remap:
            image = polar_remap(
                image, header, polar_nr, num_azimuth_bins, az_bin,
                crop=crop, rotate90=rotate90)
        cube[i] = image

    # Interpolating after the polar remap is equivalent to before it, as the remap is linear in pixel values.
    _interpolate_missing_frames(cube, headers)

    cube = np.array(cube)

    if deflicker:
        ref = median_filter(cube, size=(time_avg_size, 1, 1), mode="nearest")
        cube = remove_az_gain(cube, ref, az_smooth=az_smooth)

    if remove_temporal_median:
        # The median is computed after the deflicker, so that it is cleaner
        cube -= median_filter(cube, size=(time_avg_size, 1, 1), mode="nearest")

    # Time-crop to the time window of interest, which removes temporal edge effects, symmetrically
    cube = cube[time_win[0]:time_win[1] + 1]
    headers = headers[time_win[0]:time_win[1] + 1]

    return cube, headers


def standardize(image: np.ndarray, use_median: bool = True) -> np.ndarray:
    """
    Standardize each azimuthal row of a polar-remapped image.

    Each row (radial bin) is background-subtracted along the azimuthal axis
    and divided by its spread, so that every row has comparable zero-centered,
    unit-scale fluctuations before cross-correlation.

    Parameters
    ----------
    image : np.ndarray
        Polar-remapped image of shape (n_rows, n_cols), where rows are radial
        bins and columns are azimuth bins.
    use_median : bool, optional
        If ``True``, subtract the per-row median and divide by the scaled
        median absolute deviation (MAD * 1.4826), which is robust to outliers
        (stars, spikes).  If ``False``, use the mean and standard deviation.
        Default is ``True``.

    Returns
    -------
    np.ndarray
        Standardized image, same shape as ``image``.  Rows with zero spread
        are set to NaN.

    """
    if use_median:
        bkg = np.median(image, axis=1, keepdims=True)
        processed = image - bkg

        # MAD = median(|x - median(x)|)
        spread = np.median(np.abs(processed), axis=1, keepdims=True)

        # Scale factor (1.4826) makes MAD consistent with standard deviation for normal distributions
        spread = spread * 1.482602
    else:
        bkg = np.mean(image, axis=1, keepdims=True)
        processed = image - bkg
        spread = np.std(processed, axis=1, keepdims=True)

    # Avoid zero-division warnings by using np.divide with the 'where' mask
    return np.divide(
        processed, spread, out=np.full_like(processed, np.nan), where=(spread != 0),
    )


def max_single_image_shift(n_ofs: int, delta_px: int, central_offset: int) -> int:
    """
    Maximum number of rows a single image can be shifted during correlation.

    This helper returns a rounded-up bound on the per-image shift, for getting a crop margin
    so that no cross-correlated row is ever filled by edge replication.

    Parameters
    ----------
    n_ofs : int
        Number of offset samples.
    delta_px : int
        Pixel increment between successive offset samples.
    central_offset : int
        Central pixel offset corresponding to the expected feature displacement.

    Returns
    -------
    int
        Conservative upper bound on the per-image radial shift, in pixels.

    """
    max_total_shift = abs(delta_px) * (n_ofs - 1) // 2 + abs(int(central_offset))
    # The total is split in two (shift1 = total // 2, shift2 = total - shift1);
    # take the larger half and add 1 px of safety.
    return max_total_shift - max_total_shift // 2 + 1


# ---------------------------------------------------------------------------- #
# Cross-correlation                                                             #
# ---------------------------------------------------------------------------- #

def _shift_rows(arr: np.ndarray, shift: int, fill: bool | float | None = None) -> np.ndarray:
    """
    Shift an array along axis=0 such that ``out[i] == arr[i - shift]``.

    Reproduces the original inline pad/slice logic exactly (verified against it
    for shifts in [-9, 9]), but in one place so that an image and its mask can
    never drift apart.

    Parameters
    ----------
    arr : np.ndarray
        Array to shift, shape ``(n_rows, n_cols)``.
    shift : int
        Rows to shift by.  Positive moves content to higher row indices
        (outward in elongation).
    fill : scalar or None, optional
        Boundary treatment.  ``None`` replicates the edge row, matching the
        original ``mode="edge"``.  Pass ``False`` for boolean masks so that
        edge-replicated rows are flagged invalid.

    Returns
    -------
    np.ndarray
        Shifted array, same shape as ``arr``.

    """
    n = arr.shape[0]
    if shift == 0:
        return arr

    pad_kw = {"mode": "edge"} if fill is None else {"mode": "constant", "constant_values": fill}
    if shift > 0:
        return np.pad(arr, ((shift, 0), (0, 0)), **pad_kw)[:n, :]
    return np.pad(arr, ((0, -shift), (0, 0)), **pad_kw)[-shift:n - shift, :]


def frame_mask(image: np.ndarray, k: float = 4.0) -> np.ndarray:
    """
    Per-pixel validity mask for one preprocessed frame.

    Flags non-finite pixels (the all-NaN annuli that ``standardize`` emits when
    ``spread == 0``) and per-column radial outliers beyond ``k`` robust sigma,
    i.e. star residuals and cosmic-ray spikes.

    Computed in the unshifted frame, so the retained sample is a fixed property
    of the data and does not co-vary with the displacement being measured.

    Parameters
    ----------
    image : np.ndarray
        Preprocessed (standardized) polar image, shape ``(n_rows, n_cols)``.
    k : float, optional
        Clip threshold in robust sigma.  Default 4.0.  After ``standardize``
        the radial columns are ~unit scale, so 4 sigma removes stars while
        leaving the ~1 sigma extended features untouched.

    Returns
    -------
    np.ndarray of bool
        True where the pixel is finite and not an outlier.

    """
    finite = np.isfinite(image)
    x = np.where(finite, image, np.nan)

    med = np.nanmedian(x, axis=0)
    mad = np.nanmedian(np.abs(x - med), axis=0) * 1.4826

    # A zero or non-finite MAD must not invalidate the whole column.
    thr = np.where(np.isfinite(mad) & (mad > 0), k * mad, np.inf)

    with np.errstate(invalid="ignore"):
        return finite & (np.abs(image - med) < thr)


def calculate_cross_correlation(
        image1: np.ndarray,
        image2: np.ndarray,
        offsets: np.ndarray,
        central_offset: int,
        margin: tuple,
        delta_px: int = 1,
        k: float = 4.0,
) -> tuple:
    """
    Centered pairwise cross-correlation between two preprocessed polar images over a range of radial pixel offsets.

    At each offset the total displacement is split symmetrically: ``image1`` is
    shifted outward by half and ``image2`` inward by the other half.  Each
    column is then centered on the mean of its retained pixels within the
    summation window, so the ``N * mu_a * mu_b`` pedestal cancels exactly.

    Parameters
    ----------
    image1, image2 : np.ndarray
        Earlier and later preprocessed images, shape ``(n_rows, n_cols)``.
    offsets : np.ndarray
        Integer offset indices, e.g. ``np.arange(n_ofs)``.
    central_offset : int
        Central pixel offset corresponding to the expected displacement.
    margin : tuple of int
        ``(lo_pad, hi_pad)`` safety rows trimmed after shifting.
    delta_px : int, optional
        Pixel increment between successive offset samples.
    k : float, optional
        Outlier clip threshold in robust sigma, passed to :func:`frame_mask`.

    Returns
    -------
    crossp, auto1_acc, auto2_acc : np.ndarray
        Per-offset, per-pixel cross product and the two auto-correlation terms,
        each of shape ``(len(offsets), n_rows - lo_pad - hi_pad, n_cols)``.

    """
    lo_pad, hi_pad = margin
    n_rows, n_cols = image1.shape
    out_rows = n_rows - lo_pad - hi_pad

    crossp = np.zeros((len(offsets), out_rows, n_cols), dtype=float)
    auto1_acc = np.zeros_like(crossp)
    auto2_acc = np.zeros_like(crossp)

    # Create outlier masks ONCE per pair using some reasonable MAD clipping in the unshifted frame, then shift the masks
    m1 = frame_mask(image1, k=k)
    m2 = frame_mask(image2, k=k)
    a1 = np.where(m1, image1, 0.0)
    a2 = np.where(m2, image2, 0.0)

    for jj, offset_index in enumerate(offsets):
        total_shift = (
                int(delta_px * (offset_index - (len(offsets) - 1) / 2)) + central_offset
        )
        shift1 = total_shift // 2
        shift2 = total_shift - shift1

        # image1 moves outward by shift1, image2 inward by shift2.
        # Masks pad with False so edge-replicated rows are never correlated.
        pa1 = _shift_rows(a1, shift1)
        pm1 = _shift_rows(m1, shift1, fill=False)
        pa2 = _shift_rows(a2, -shift2)
        pm2 = _shift_rows(m2, -shift2, fill=False)

        # Trim the safety margin so returned rows map onto [crop[0], crop[1]].
        if lo_pad or hi_pad:
            hi_end = pa1.shape[0] - hi_pad
            pa1, pm1 = pa1[lo_pad:hi_end, :], pm1[lo_pad:hi_end, :]
            pa2, pm2 = pa2[lo_pad:hi_end, :], pm2[lo_pad:hi_end, :]

        # Only the O(N) moments stay in the loop: the summation window slides
        # with the lag, so the centering constant must be recomputed per lag.
        valid = pm1 & pm2
        n = valid.sum(axis=0)

        mean_a = np.where(valid, pa1, 0.0).sum(axis=0) / np.maximum(n, 1)
        mean_b = np.where(valid, pa2, 0.0).sum(axis=0) / np.maximum(n, 1)

        centered_a = np.where(valid, pa1 - mean_a, 0.0)
        centered_b = np.where(valid, pa2 - mean_b, 0.0)

        crossp[jj] = centered_a * centered_b
        auto1_acc[jj] = centered_a * centered_a
        auto2_acc[jj] = centered_b * centered_b

    return crossp, auto1_acc, auto2_acc


def accumulate_cross_correlation_across_frames(
        files: list,
        delta_t: int,
        sparsity: int,
        n_ofs: int,
        polar_nr: float,
        num_azimuth_bins: int,
        az_bin: int,
        central_offset: int,
        product: str,
        delta_px: int = 1,
        time_win: list | None = None,
        use_median: bool = True,
        crop: list | None = None,
        crop_margin: int = 0,
        rotate90: bool = False,
        az_smooth: int = 10,
        do_polar_remap: bool = True,
        deflicker: bool = False,
        remove_temporal_median: bool = False,
        time_avg_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Accumulate pairwise cross-correlations over a sequence of FITS image frames.

    Each consecutive frame pair is independently preprocessed and
    cross-correlated; the results are averaged to produce a single correlation
    array representative of the entire time window.

    Parameters
    ----------
    files : list of Path or str or None
        Sorted list of FITS file paths to process, None for missing frames (see :func:`preprocess_cube`).
    delta_t : int
        Frame offset (in frames) between time-offset image pairs. Must be at least 1
    sparsity : int
        Step size when iterating over frame pairs.  A value of 1 processes
        every consecutive pair; 2 processes alternating pairs, etc.
    n_ofs : int
        Number of pixel offsets to sample in the cross-correlation.
    polar_nr : int
        Pixels along elongation axis the polar-remapped image.
    num_azimuth_bins : int
        Number of azimuthal samples before binning.
    az_bin : int
        Azimuthal binning factor applied by :func:`polar_remap` (anti-alias
        filter and decimation).
    central_offset : int
        Central pixel offset corresponding to the expected feature displacement.
    product : str
        PUNCH data product code of the input files (``'CAM'``, ``'PAM'``,
        ``'CTM'`` or ``'PTM'``).  For 3-D polarized/total-brightness cubes
        (PAM/PTM), only the first (total brightness) layer is used.
    delta_px : int, optional
        Pixel increment between successive offset samples of the
        cross-correlation.  Must match the ``delta_px`` used to build the
        velocity axis.  Default 1.
    time_win: tuple, optional
        start and end (inclusive) of the time slice of interest in the input time series.
        This is useful to not have edge effects from the sliding temporal median
    use_median : bool, optional
        Use the median (``True``) or mean (``False``) for background estimation when
        applying row-wise standardization.
        Default is ``True``.
    crop : list of int, optional
        ``[row_low, row_high]`` pixel crop applied to the radial axis.
        Default is ``None``.
    crop_margin : int, optional
        Number of extra rows to retain on each side of ``crop`` while
        correlating, then trimmed off before returning.  This guarantees that
        every row inside ``[crop[0], crop[1]]`` is correlated against real data
        at every offset (no edge-replication contamination).  Ignored when
        ``crop`` is ``None``.  Default is 0.
    rotate90: bool
        During preprocessing, rotate the angular origin by 90 degrees counter-clockwise with respect to the
        unit circle origin.
    do_polar_remap: bool
        If True, will apply polar remapping before tracking. If the images are already polar-remapped, set it to False.
    deflicker: bool, optional
        If True, will compensate for time-azimuthal gain changes by dividing an estimate based on
        base difference and smoothed radial average.
    remove_temporal_median: bool, optional
        If True: will subtract the median of the cube. If Deflicker is applied, the median runs
        after it.
        Default False
    time_avg_size: int, optional
        Size of the running time averaging kernel in number of frames, see :func:`preprocess_cube`.
    az_smooth: int
        Will apply some smoothing over the azimuth gain.

    Returns
    -------
    acc_crossp : np.ndarray
        Time-averaged per-pixel cross product, shape
        ``(n_ofs, n_rows, n_cols)``, where ``n_rows``/``n_cols`` are the
        dimensions of the preprocessed (and margin-trimmed) polar images.
    acc_auto1 : np.ndarray or None
        Time-averaged per-pixel auto-correlation of the earlier image residuals
        (``centered_a ** 2``), same shape as ``acc_crossp``.
    acc_auto2 : np.ndarray or None
        Time-averaged per-pixel auto-correlation of the later image residuals
        (``centered_b ** 2``), same shape as ``acc_crossp``.

    """
    logger = get_logger()

    if crop is not None and crop_margin > 0:
        if crop[0] < crop_margin or crop[1] + crop_margin > round(polar_nr):
            raise ValueError(
                f"crop {crop} with crop_margin {crop_margin} exceeds the "
                f"available radial extent [0, {polar_nr - 1}]; "
                f"cannot guarantee an edge-free correlation window.",
            )
        lo_pad = hi_pad = crop_margin
        effective_crop = [crop[0] - crop_margin, crop[1] + crop_margin]
    else:
        lo_pad = hi_pad = 0
        effective_crop = crop

    # Load the full cube once: polar remap + azimuthal gain (flicker) removal
    cube, headers = preprocess_cube(
        files, product=product,
        polar_nr=polar_nr, num_azimuth_bins=num_azimuth_bins, az_bin=az_bin,
        rotate90=rotate90, crop=effective_crop,
        do_polar_remap=do_polar_remap, az_smooth=az_smooth, time_win=time_win,
        deflicker=deflicker, remove_temporal_median=remove_temporal_median, time_avg_size=time_avg_size,
    )

    n_rows, n_cols = cube.shape[1:]
    acc_rows = n_rows - lo_pad - hi_pad
    acc_crossp = np.zeros((n_ofs, acc_rows, n_cols), dtype=float)

    # Pearson normalization denominator terms
    acc_auto1 = np.zeros_like(acc_crossp)
    acc_auto2 = np.zeros_like(acc_crossp)

    n_pairs = 0

    # If time_win not None, i=0 is relative to whatever time_win is.
    for i in range(0, len(cube) - delta_t, sparsity):
        logger.info(f"Frame {i} vs frame {i + delta_t}")
        prepped1 = standardize(cube[i], use_median=use_median)
        prepped2 = standardize(cube[i + delta_t], use_median=use_median)

        try:
            # Escalate ONLY the target warning to an exception so the except
            # clause can attribute it to this pair index. The 'error' action
            # fires on every occurrence (no once-per-location dedup).
            with warnings.catch_warnings():
                warnings.filterwarnings("error", message="All-NaN slice")
                crossp, auto1, auto2 = calculate_cross_correlation(
                    prepped1, prepped2, np.arange(n_ofs), central_offset, (lo_pad, hi_pad), delta_px=delta_px)
        except RuntimeWarning as e:
            # prepped1/prepped2 are already bound here: count the columns that
            # are NaN at every row (exactly what makes nanmedian warn).
            bad1 = np.flatnonzero(np.isnan(prepped1).all(axis=0))
            bad2 = np.flatnonzero(np.isnan(prepped2).all(axis=0))
            logger.warning(f"Pair i={i} (cube frames {i} vs {i + delta_t}: "
                           f"{headers[i]['DATE-OBS']} -> {headers[i + delta_t]['DATE-OBS']}) -> {e}\n"
                           f"    all-NaN columns: frame {i}: {bad1.size}, frame {i + delta_t}: {bad2.size}")

            # The escalation aborted this pair's computation, so redo it with
            # the warning suppressed to keep the accumulator correct.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                crossp, auto1, auto2 = calculate_cross_correlation(
                    prepped1, prepped2, np.arange(n_ofs), central_offset, (lo_pad, hi_pad), delta_px=delta_px)

        acc_crossp += crossp
        acc_auto1 += auto1
        acc_auto2 += auto2

        n_pairs += 1

    acc_crossp /= n_pairs

    acc_auto1 /= n_pairs
    acc_auto2 /= n_pairs
    return acc_crossp, acc_auto1, acc_auto2


def pearson_from_acc(acc_crossp: np.ndarray, acc_auto1: np.ndarray, acc_auto2: np.ndarray,
                     rows: slice | tuple | None = None, cols: slice | tuple | None = None) -> np.ndarray:
    """
    Combine time-averaged per-pixel cross/auto products into per-column Pearson correlation coefficients.

    The output has shape (n_ofs, n_cols).

    Parameters
    ----------
    acc_crossp, acc_auto1, acc_auto2 : np.ndarray
        Accumulators from accumulate_cross_correlation_across_frames,
        shape (n_ofs, n_rows, n_cols).
    rows : None, slice, or (start, stop[, step]) sequence
        Radial row window over which to sum before normalizing:
          - None          -> all rows
          - (start, stop) -> equivalent to acc[:, start:stop, :]
          - slice(a, b)   -> used as-is (supports negative indices / None bounds)
    cols: None, slice, or (start, stop[, step]) sequence
        Same as rows, for slicing columns

    Returns
    -------
    np.ndarray, shape (n_ofs, n_cols)
        Pearson r per offset and per column. NaN where the denominator is 0
        (zero-variance or fully invalid window).

    """
    if rows is None:
        row_slice = slice(None)
    elif isinstance(rows, slice):
        row_slice = rows
    else:
        row_slice = slice(*rows)  # accepts (start, stop) and (start, stop, step)

    if cols is None:
        num = acc_crossp[:, row_slice, :].sum(axis=1)
        den = np.sqrt(acc_auto1[:, row_slice, :].sum(axis=1)
                      * acc_auto2[:, row_slice, :].sum(axis=1))

        return np.divide(num, den, out=np.full_like(num, np.nan), where=(den > 0))

    cols_slice = cols if isinstance(cols, slice) else slice(*cols)

    num = acc_crossp[:, row_slice, cols_slice].sum(axis=2).sum(axis=1)
    den = np.sqrt(acc_auto1[:, row_slice, cols_slice].sum(axis=2).sum(axis=1)
                  * acc_auto2[:, row_slice, cols_slice].sum(axis=2).sum(axis=1))

    return np.divide(num, den, out=np.full_like(num, np.nan), where=(den > 0))


def process_corr_vel(files: list, preprocess_opts: dict, delta_t: int, sparsity: int, n_ofs: int, polar_nr: int,
                     azimuth_bins_remap: int, az_bin: int, central_offset: int, x_kps: np.ndarray,
                     offset_speed_kps: float, vel_bin_width: int, annuli_crop: list,
                     speed_max: float, delta_px: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Accumulate cross-correlations over a list of FITS files and fit peak speeds per annulus.

    Parameters
    ----------
    files : list of Path or str
        Sorted list of FITS file paths to process.
    preprocess_opts : dict
        Keyword options forwarded to
        :func:`accumulate_cross_correlation_across_frames` (product, time_win,
        use_median, crop, rotate90, remove_temporal_median, deflicker).
    delta_t : int
        Frame offset (in frames) between time-offset image pairs.
    sparsity : int
        Step size when iterating over frame pairs.
    n_ofs : int
        Number of pixel offsets to sample in the cross-correlation.
    polar_nr : int
        Radial size (pixels) of the polar-remapped image.
    azimuth_bins_remap : int
        Number of azimuthal samples in the polar-remapped images before binning.
    az_bin : int
        Azimuthal binning factor applied by :func:`polar_remap`.
    central_offset : int
        Central pixel offset corresponding to the expected displacement.
    x_kps : np.ndarray
        Velocity axis in km/s matching the sampled offsets.
    offset_speed_kps : float
        Speed below which peaks are ignored when searching for the correlation peak.
    vel_bin_width : int
        Bin size over azimuth of the output flow maps.
    annuli_crop : list
        Radial row windows (relative to the crop) of each annulus.
    speed_max : float
        Maximum velocity to consider in the peak calculation.
    delta_px : int, optional
        Pixel increment between successive offset samples of the
        cross-correlation.  Must match the ``delta_px`` used to build
        ``x_kps``.  Default 1.

    Returns
    -------
    speeds : np.ndarray
        Per-annulus, per-azimuth-bin speeds, shape (n_annuli, flow_az_bins).
    sigmas : np.ndarray
        Matching speed uncertainties (robust sigma), same shape as ``speeds``.

    """
    acc = accumulate_cross_correlation_across_frames(
        files, delta_t, sparsity, n_ofs, polar_nr,
        azimuth_bins_remap, az_bin, central_offset,
        delta_px=delta_px, **preprocess_opts)

    # Loop over the annuli
    speeds, sigmas = zip(*[correl_peak_speed(acc, a_crop, x_kps, offset_speed_kps, vel_bin_width,
                                             speed_max=speed_max) for a_crop in annuli_crop],
                         strict=True)

    return np.stack(speeds), np.stack(sigmas)


def correl_peak_speed(acc: tuple, rows: slice | tuple, x_speed: np.ndarray, offset_speed: float, vel_bin_width: int,
                      speed_max: float = 1000, debug: bool = False) -> tuple:
    """
    Create the speed map from the time-averaged correlation array for one annulus.

    Parameters
    ----------
    acc : tuple of np.ndarray
        Accumulators (cross and auto terms) from
        :func:`accumulate_cross_correlation_across_frames`.
    rows : slice or (start, stop)
        Radial row window of the annulus.
    x_speed : np.ndarray
        Velocity axis in km/s matching the sampled offsets.
    offset_speed : float
        Speed below which peaks are ignored when searching for the correlation peak.
    vel_bin_width : int
        Bin size over azimuth; must divide evenly the azimuthal axis.
    speed_max : float, optional
        Maximum velocity considered in the peak calculation.  Default 1000.
    debug : bool, optional
        If ``True``, also return the binned average correlation array.
        Default False.

    Returns
    -------
    rspeed_per_theta : np.ndarray
        Peak speed per azimuth bin.
    sigma_per_theta : np.ndarray
        Matching speed uncertainty per azimuth bin.
    avcor_rbins_theta : np.ndarray, optional
        Binned average correlation array, shape (n_ofs, n_az_bins).  Only
        returned when ``debug=True``.

    """
    row_slice = rows if isinstance(rows, slice) else slice(*rows)

    # Average correlation signal over the selected annuli
    acc_k = pearson_from_acc(*acc, rows=row_slice)
    if acc_k.shape[1] % vel_bin_width != 0:
        raise ValueError("Bin width must divide evenly the azimuthal axis")
    n_az_bins = acc_k.shape[1] // vel_bin_width
    # Reshape the average correlation array for max efficiency of the per-bin velocity measurement
    avcor_rbins_theta = acc_k.reshape(acc_k.shape[0], n_az_bins, vel_bin_width).mean(axis=2)
    # Get the 1st correlation peak past the non-physical ones (detector, flat-field pattern, stars etc...)
    rspeed_per_theta = []
    sigma_per_theta = []
    for i in range(n_az_bins):
        acc = avcor_rbins_theta[:, i]
        # V2
        rbest, sigma, *_ = find_best_bump_v2b(x_speed, acc, x_min=offset_speed, x_max=speed_max, chi2_max=3.5)

        rspeed_per_theta.append(rbest)
        sigma_per_theta.append(sigma)

    if debug:
        return np.array(rspeed_per_theta), np.array(sigma_per_theta), avcor_rbins_theta

    return np.array(rspeed_per_theta), np.array(sigma_per_theta)


def rebin_speeds_sigmas(speeds: np.ndarray, new_bin_width: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Rebin the speeds along azimuth and compute a MAD-based uncertainty.

    Parameters
    ----------
    speeds : np.ndarray
        Speeds over one annulus, shape (n_az_bins,), with ``n_az_bins``
        divisible by ``new_bin_width``.
    new_bin_width : int
        Number of original azimuth bins merged into each new bin.

    Returns
    -------
    median_speed : np.ndarray
        Median speed per rebinned azimuth bin, shape (n_az_bins / new_bin_width,).
    sigmas : np.ndarray
        Standard error of the median per rebinned bin, estimated from the
        robust (MAD) spread of the original bins.

    """
    rspeeds = speeds.reshape(-1, new_bin_width)
    median_speed = np.median(rspeeds, axis=1)
    spread = 1.4826 * np.median(np.abs(rspeeds - median_speed[:, None]), axis=1)  # robust sigma (MAD)
    sigmas = 1.2533 * spread / np.sqrt(new_bin_width)

    return median_speed, sigmas


def find_best_bump1(xspeed: np.ndarray, corr: np.ndarray, x_min: float = 100, x_max: float = 1000,
                    power: float = 2, n_sigma: float = 3.0, trend_smoothness: float = 0.1) -> tuple[float, float]:
    """
    Locate the most prominent bump of a correlation profile over a speed range.

    A large-sigma Gaussian filter estimates the macro-trend of the profile;
    subtracting it flattens the slope so that small bumps stand out.  Peaks
    are found on the flattened signal with a prominence of 1.5 robust-noise
    sigma, then re-evaluated on the original signal to pick the true maximum.
    A global speed uncertainty is computed as the weighted spread of the
    profile above ``n_sigma`` times the noise level.

    Parameters
    ----------
    xspeed : np.ndarray
        Velocity axis in km/s.
    corr : np.ndarray
        Correlation profile sampled on ``xspeed``.
    x_min, x_max : float, optional
        Speed range over which to search for the bump.  Defaults 100, 1000.
    power : float, optional
        Exponent applied to the above-cutoff profile when computing the
        weighted moments.  Default 2.
    n_sigma : float, optional
        Cutoff level, in robust-noise sigma above the 10th-percentile
        baseline, for the global sigma estimate.  Default 3.0.
    trend_smoothness : float, optional
        Fraction of the profile length used as the Gaussian sigma of the
        macro-trend estimate.  Default 0.1.

    Returns
    -------
    best_peak_x : float
        Speed of the most prominent bump, or NaN when no bump is found.
    global_sigma : float
        Global uncertainty (weighted spread) of the bump, or NaN when no
        signal rises above the cutoff.

    """
    # 1. Crop to the broad range of interest
    broad_mask = (xspeed >= x_min) & (xspeed <= x_max)
    x_broad = xspeed[broad_mask]
    corr_broad = corr[broad_mask]

    n_points = len(corr_broad)
    if n_points < 2:
        return np.nan, np.nan

    # 2. NOISE ESTIMATION (Unaffected by slope)
    diffs = np.diff(corr_broad)
    noise_sigma = (np.median(np.abs(diffs - np.median(diffs))) * 1.4826) / np.sqrt(2)
    if noise_sigma == 0:
        noise_sigma = 1e-6

    # We use a Gaussian filter with a large sigma to estimate the macro-trend.
    # It acts as a steamroller: perfectly tracking the slope while flattening the bumps.
    # trend_smoothness is a fraction of your window size (10% is usually perfect).
    sigma_bins = max(5, int(n_points * trend_smoothness))
    background_trend = gaussian_filter1d(corr_broad, sigma=sigma_bins)

    # Subtracting the background flattens the slope without tilting local Gaussians
    flat_signal = corr_broad - background_trend

    # Find peaks on the flattened signal.
    # Because the macro-slope is gone, bump_prominence can be small and sensitive.
    bump_prominence = 1.5 * noise_sigma
    peaks, _ = find_peaks(flat_signal, prominence=bump_prominence)

    # 3. Calculate Global Sigma (Uncertainty of the whole mass) on ORIGINAL signal
    baseline = np.percentile(corr_broad, 10)
    cutoff = baseline + (n_sigma * noise_sigma)

    corr_above = np.maximum(0.0, corr_broad - cutoff)
    weights = corr_above ** power
    total_weight = np.sum(weights)

    if total_weight > 0:
        com_global = np.sum(x_broad * weights) / total_weight
        variance = np.sum(weights * (x_broad - com_global) ** 2) / total_weight
        global_sigma = np.sqrt(variance)
    else:
        global_sigma = np.nan

    # 4. Edge Case: Pure negative slope (no bumps/shoulders found)
    if len(peaks) == 0:
        return np.nan, global_sigma

    # 5. Normal Case: Bumps found.
    # We found the peak coordinates using the flattened signal, but we evaluate
    # their actual height using the ORIGINAL raw signal to find the true max.
    peak_intensities = corr_broad[peaks]
    best_peak_idx = peaks[np.argmax(peak_intensities)]
    best_peak_x = x_broad[best_peak_idx]

    return best_peak_x, global_sigma


def _gauss(x: np.ndarray, amp: float, mu: float, s: float, c: float) -> np.ndarray:
    """Gaussian bump with amplitude ``amp``, center ``mu``, width ``s`` and offset ``c``."""
    return amp * np.exp(-0.5 * ((x - mu) / s) ** 2) + c


def find_best_bump_v2b(xspeed: np.ndarray, corr: np.ndarray, x_min: float = 100, x_max: float = 1000,
                       chi2_max: float = 2.0,  # calibrate on your data (see note)
                       sigma_range: tuple[float, float] = (100, 800),  # generous around your 200-300
                       min_snr_amp: float = 5.0,
                       **kwargs: float) -> tuple[float, float, str, float]:
    """
    Refine the bump estimate with a Gaussian fit, falling back to the bump finder.

    A Gaussian plus a constant is fitted to the profile over [x_min, x_max],
    with the noise estimated from the profile differences.  The fit is
    accepted only if the reduced chi-square is below ``chi2_max``, the fitted
    center lies within the search range, the fitted width lies within
    ``sigma_range``, and the amplitude is detected above ``min_snr_amp``
    significance; otherwise the :func:`find_best_bump1` estimate is returned.

    Parameters
    ----------
    xspeed : np.ndarray
        Velocity axis in km/s.
    corr : np.ndarray
        Correlation profile sampled on ``xspeed``.
    x_min, x_max : float, optional
        Speed range over which to fit.  Defaults 100, 1000.
    chi2_max : float, optional
        Maximum reduced chi-square for the fit to be accepted.  Default 2.0.
    sigma_range : tuple of float, optional
        Acceptable range for the fitted Gaussian width, in km/s.
        Default (100, 800).
    min_snr_amp : float, optional
        Minimum amplitude signal-to-noise ratio for the fit to be accepted.
        Default 5.0.
    **kwargs
        Extra keyword arguments forwarded to :func:`find_best_bump1`.

    Returns
    -------
    speed : float
        Fitted Gaussian center when the fit is accepted, otherwise the
        bump-finder peak (NaN when no bump is found).
    sigma : float
        Fitted Gaussian width when the fit is accepted, otherwise the
        bump-finder global sigma (NaN when no bump is found).
    kind : str
        ``"gaussian"`` when the fit is accepted, ``"bump"`` otherwise.
    chi2_red : float
        Reduced chi-square of the fit, or NaN when the fit could not be
        performed.

    """
    bump_x, bump_sig = find_best_bump1(xspeed, corr, x_min, x_max, **kwargs)

    m = (xspeed >= x_min) & (xspeed <= x_max)
    x, y = xspeed[m], corr[m]
    if len(x) < 10:
        return bump_x, bump_sig, "bump", np.nan

    d = np.diff(y)
    noise = max(np.median(np.abs(d - np.median(d))) * 1.4826 / np.sqrt(2), 1e-6)

    # Initial guesses from the bump finder, with fallbacks
    mu0 = bump_x if np.isfinite(bump_x) else x[np.argmax(y)]
    s0 = bump_sig if np.isfinite(bump_sig) else 250 / 2.355
    p0 = [y.max() - y.min(), mu0, s0, y.min()]
    bounds = ([0, x.min(), 20, -np.inf], [np.inf, x.max(), 600, np.inf])

    try:
        popt, pcov = curve_fit(_gauss, x, y, p0=p0, bounds=bounds,
                               sigma=np.full_like(y, noise),
                               absolute_sigma=True, maxfev=10000)
    except (RuntimeError, ValueError):
        return bump_x, bump_sig, "bump", np.nan

    amp, mu, s, _ = popt
    amp_err = np.sqrt(pcov[0, 0])
    chi2_red = np.sum(((y - _gauss(x, *popt)) / noise) ** 2) / (len(x) - 4)

    is_gauss = (chi2_red < chi2_max
                and x_min <= mu <= x_max
                and sigma_range[0] <= s <= sigma_range[1]
                and amp / amp_err > min_snr_amp)

    if is_gauss:
        return mu, s, "gaussian", chi2_red  # or np.sqrt(pcov[1,1]) if you want the error on mu
    return bump_x, bump_sig, "bump", chi2_red



def circle_results(vel: np.ndarray, sig: np.ndarray, thetas: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Close azimuthal circles by appending the first sample at the end.

    Parameters
    ----------
    vel, sig : np.ndarray
        Speeds and uncertainties sampled on the azimuthal axis.
    thetas : np.ndarray
        Azimuthal angles (radians) sampled on the same grid.

    Returns
    -------
    vel_circle, sig_circle, thetas_circle : np.ndarray
        The input arrays closed into full circles (first value repeated at
        the end).  ``thetas`` is only appended when its last value differs
        from its first.

    """
    vel_circle = np.append(vel, vel[0])
    sig_circle = np.append(sig, sig[0])
    thetas_circle = np.append(thetas, thetas[0]) if thetas[0] != thetas[-1] else thetas
    return vel_circle, sig_circle, thetas_circle


def plot_flow_map(data: PUNCHCube, rebin: int = 10, plot_errors: bool = True, vmax: float = 800,
                  theme: str = "dark_background", filename: str | None = None) -> plt.Figure:
    """
    Plot polar maps of the radial flows.

    Parameters
    ----------
    data: PUNCHCube
        Flow tracking data PUNCHCube

    rebin: int, optional
        Number of bins to rebin the data by
        Must divide evenly the input azimuth size

    plot_errors: bool, optional
        Whether to plot the error bars on the flow map

    vmax: float, optional
        Maximum velocity to plot on the flow map

    theme: str, optional
        Matplotlib style sheet applied to the figure.
        Default 'dark_background'

    filename: str, optional
        Output plot filename. If None, the figure is not saved out.

    Returns
    -------
    fig
        The generated Matplotlib Figure

    """
    speeds = data.data
    sigmas = data.uncertainty.array

    if rebin > 1:
        if speeds.shape[1] % rebin != 0:
            raise ValueError("rebin must divide evenly the input azimuth size")
        speeds, sigmas = zip(*[rebin_speeds_sigmas(s, rebin) for s in data.data], strict=True)
        speeds = np.array(speeds)
        sigmas = np.array(sigmas)

    nbins = speeds.shape[1]
    thetas = np.linspace(0, 2 * np.pi, nbins)
    dthetas = thetas[1] - thetas[0]
    thetas += dthetas / 2

    band_centers_rs = np.fromstring(data.meta["YCENS"].value[1:-1], dtype=float, sep=",")
    band_width_rs = int(data.meta["BANDWDTH"].value)

    # Use context manager here to isolate the theme to this plot without affect default theme globally.
    with plt.style.context(theme):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1, projection="polar")
        colors = ["cyan", "orange", "yellow", "magenta"]

        for i, center in enumerate(band_centers_rs[0:4]):  # up to 4 bands. Ignore the rest if more
            speed_annulus, sigma_annulus, thetas = circle_results(speeds[i], sigmas[i], thetas)
            annulus = [int(center - band_width_rs / 2), int(center + band_width_rs / 2)]
            ax.plot(thetas, speed_annulus, color=colors[i], ls="-", label=f"{annulus[0]} --> {annulus[1]} Rs")
            if plot_errors:
                ax.fill_between(thetas, speed_annulus - sigma_annulus, speed_annulus + sigma_annulus, alpha=0.2,
                                color=colors[i])

        date_start = data.meta["DATE-BEG"].value[:-7]
        date_end = data.meta["DATE-END"].value[:-7]
        ax.set_title(f"{date_start} -> {date_end}")
        ax.set_ylim(0, vmax)

        ax.set_theta_zero_location("N")
        ax.set_rlabel_position(180)
        ax.set_xlabel("Radial velocity [km/s]")
        ax.legend(loc="lower right", bbox_to_anchor=[1.05, -0.1])

        ax.set_xticks(np.deg2rad(np.arange(0, 360, 30)))  # labeled every 30 deg
        ax.set_xticks(np.deg2rad(np.arange(0, 360, 10)), minor=True)  # ticks every 10 deg
        ax.tick_params(axis="x", which="minor", length=3)
        ax.grid(which="minor", axis="x", linestyle=":", linewidth=1, alpha=0.4)

        if filename is not None:
            plt.savefig(filename)

    # Return the Figure outside the 'with' block.
    # The figure object has already absorbed the style internally.
    return fig


@punch_flow(log_prints=True, timeout_seconds=21_600)
def track_velocity(files: list[str] | list[Path],
                   product_code: str,
                   flow_window_hours: float = 6,
                   delta_t_minutes: float = 64,
                   sparsity: int = 1,
                   n_ofs: int = 101,
                   delta_px: int = 1,
                   expected_wind_kps: int = 300,
                   offset_speed_kps: int = 50,
                   annuli_centers_rs: list[float | int] | np.ndarray = (33, 53),
                   annuli_width_rs: float = 20,
                   azimuth_bins_remap: int = 5760,
                   az_bin: int = 4,
                   vel_bin_width: int=4,
                   speed_max: float = 1000,
                   time_avg_hours: float = 6,
                   use_median: bool = True,
                   rotate90: bool = True,
                   remove_temporal_median: bool = True,
                   deflicker: bool = True,
                   hdu: int = 1,
                   reference_time: datetime | None = None) -> PUNCHCube:
    """
    Generate velocity map using flow tracking.

    Parameters
    ----------
    files : list[str]
        List of file paths for input data, in chronological order.  The number of
        required frames is given by :func:`get_required_frame_count`, from the product
        cadence, ``flow_window_hours`` and ``time_avg_hours``.
        Missing frames are detected from the observation times and filled by linear interpolation
        within the limits of :func:`check_missing_runs`.

    product_code: str
        Either 'CAM', 'PAM', 'CTM', or 'PTM'

    flow_window_hours: float, optional
        Minimum duration of the flow map integration window in hours, see :func:`get_frames_per_window`

    delta_t_minutes : float, optional
        Time lag in minutes between the two frames of each correlated pair, converted to the nearest
        whole number of frames of the product, see :func:`get_delta_t_frames`

    sparsity : int, optional
        Frame skip interval for averaging

    n_ofs : int, optional
        Number of spatial offsets for cross-correlation

    delta_px : int, optional
        Pixel offset increment between successive correlation samples; also
        used to build the velocity axis

    expected_wind_kps : int, optional
        Expected wind speed in km/s

    offset_speed_kps: int, optional
        Offset speed before which speeds are ignored when searching for the peak speed

    annuli_centers_rs: sequence(float | int), optional
        Centers of the annuli, need at east two for measuring any potential acceleration

    annuli_width_rs : float, optional
        Width of the annuli in solar radii

    azimuth_bins_remap : int, optional
        Number of azimuthal bins in the polar-remapped images before binning

    az_bin : int, optional
        Binning factor for binning the polar-remapped image over the azimuth
        azimuth_bins_remap / az_bin sets the size of the output flow map over the azimuth axis

    vel_bin_width: int, optional
        bin size over azimuth of the output flow maps

    speed_max: int, optional
        Maximum velocity to consider in the moments and peak calculation.

    time_avg_hours: float, optional
        Duration in hours of the running time-average kernel used by the deflicker and the temporal median removal.
        It also sets the buffer of frames on either side of the flow map integration window, so that the kernel
        centered on any frame of the window only covers actual frames.

    use_median: bool, optional
        Whether to use the median instead of the mean for the temporal averages

    rotate90: bool, optional
        Whether to rotate the output flow map by 90 degrees to get solar north as origin of position angle

    remove_temporal_median: bool, optional
        Whether to remove the temporal median from the images

    deflicker: bool, optional
        Whether to apply an azimuthal filter to attenuate the time- and azimuthal-dependent flickering

    hdu: int, optional
        Position of the header data unit in the FITS files holding the data.
        Default is 1 for RICE-compressed FITS (use 0 for uncompressed FITS).

    reference_time: datetime, optional
        Authoritative time of the velocity map, written to DATE-OBS. When set, the flow map integration
        window is centered on it (see :func:`get_flow_window_start`): for an odd number of frames
        the middle frame is the one nearest to it, for an even number of frames it falls between
        the two middle frames. ``files`` may cover a wider time range than required. When None, the
        leading files are used and the reference time is the middle of the flow map integration window.

    Returns
    -------
    ndcube.PUNCHCube
        The generated velocity map

    """
    # ---------------------------------------------------------------------------- #
    # Instrument constants                                                          #
    # ---------------------------------------------------------------------------- #
    # Some astronomical constants
    arcsec_rad = 4.84814e-6  # 1 arcsec in rad
    au_km = 150e6  # Astronomical unit in km
    arcsec_km = arcsec_rad * au_km  # 1 arcsec in km at 1 AU

    # Checking input comply with requirements

    delta_t = get_delta_t_frames(product_code, delta_t_minutes)

    if azimuth_bins_remap % az_bin != 0:
        raise ValueError("AZIMUTH_BINS_REMAP must be divisible by AZ_BIN")

    polar_naz = azimuth_bins_remap // az_bin  # azimuthal size after binning

    if polar_naz % vel_bin_width != 0:
        raise ValueError("POLAR_NAZ % VEL_BIN_WIDTH !=0 --- Bin width must divide evenly the azimuthal axis")

    flow_az_bins = polar_naz // vel_bin_width

    # Accept tuples (the default), lists (e.g. from pipeline configs), or arrays
    annuli_centers_rs = np.atleast_1d(np.asarray(annuli_centers_rs, dtype=float))
    # At least two annuli are needed to measure any potential acceleration
    if len(annuli_centers_rs) < 2:
        raise ValueError("annuli_centers_rs must have at least two elements")

    # We need to get a number of files sufficient to cover the integration time of the flow map + required buffer for
    # an evenly sliding temporal mean of either side of the first and last frame of interest.
    # The flow map integration window [tstart, tend] is centered in that timeline. To illustrate, with
    # ta = temporal average of background and ft = flow time window:
    # [ta start......ft start......ft end.......ta end]
    obs_times = [datetime.fromisoformat(fits.getheader(f, hdu)["DATE-OBS"]) for f in files]
    slots, slot_times = get_frame_slots(obs_times, get_cadence_min(product_code))
    if reference_time is not None:
        reference_time = reference_time.replace(tzinfo=None)
    # Missing frames are None in subset_files, and are interpolated during pre-processing
    first, last, tstart, tend = select_frame_slots(slots, slot_times, product_code, flow_window_hours,
                                                   time_avg_hours, reference_time)
    subset_files = [None if slot is None else files[slot] for slot in slots[first:last + 1]]
    if None in subset_files:
        get_logger().info(f"Interpolating {subset_files.count(None)} missing {product_code} frames")

    # Get metadata for scaling units of solar radius
    data = load_ndcube_from_fits(subset_files[0])  # the first selected slot is always an actual frame

    # Get annuli pixel limits, with associated crop coordinates for slicing the radial range of interest
    cdelt1 = data.wcs.wcs.cdelt[0]  # punch wfi native pixel scale in deg/pixel - should be 0.0225 deg/pixel
    max_elong_deg = cdelt1 * 2048  # maximum elongation (deg) = 46.08 deg.
    polar_nr = round(max_elong_deg / cdelt1)  # radial axis size (pixels)
    arcsec_per_px = max_elong_deg * 3600 / polar_nr
    km_per_px = arcsec_per_px * arcsec_km

    def slot_date(slot: int, key: str) -> datetime:
        """Get a date keyword of a slot, offset from the nearest actual frame for a missing frame."""
        nearest = min((i for i in range(first, last + 1) if slots[i] is not None), key=lambda i: abs(i - slot))
        header = fits.getheader(files[slots[nearest]], hdu)
        return datetime.fromisoformat(header[key]) + (slot_times[slot] - slot_times[nearest])

    date_beg = slot_date(first + tstart, "DATE-BEG")
    date_end = slot_date(first + tend, "DATE-END")
    date_avg = date_beg + (date_end - date_beg) / 2
    if reference_time is None:
        # Deterministic fallback: the middle of the flow map integration window
        reference_time = date_avg

    rs_arcsec = sun.angular_radius(reference_time).to_value(u.arcsec)
    annuli = [get_annulus(center, annuli_width_rs, arcsec_per_px, rs_arcsec) for center in annuli_centers_rs]
    annuli_crop = [a - annuli[0][0] for a in annuli]

    # For the polar-remapped cropped image
    r_low = min(a[0] for a in annuli)
    r_high = max(a[1] for a in annuli)

    # Cross-correlation velocity offset based on expected speed, and velocity lookup axis
    effective_cadence_sec = get_cadence_min(product_code) * delta_t * 60  # effective time step between frame pairs (s)
    expected_displacement_px = expected_wind_kps / km_per_px * effective_cadence_sec
    central_offset = int(expected_displacement_px)
    _, x_kps = build_velocity_axis(n_ofs, central_offset, expected_wind_kps, delta_px=delta_px)

    # Wrap up inputs for preprocessing
    preprocess_opts = {
        "product": product_code,
        "time_win": (tstart, tend),
        "use_median": use_median,
        "crop": [r_low, r_high],
        "rotate90": rotate90,
        "remove_temporal_median": remove_temporal_median,
        "deflicker": deflicker,
        "time_avg_size": get_time_avg_size(product_code, time_avg_hours),
    }

    speeds, sigmas = process_corr_vel(subset_files, preprocess_opts, delta_t, sparsity, n_ofs, polar_nr,
                                         azimuth_bins_remap, az_bin, central_offset, x_kps, offset_speed_kps,
                                         vel_bin_width, annuli_crop, speed_max, delta_px=delta_px)


    # ------------------------------------------------------------------ #
    # Output the results                                                 #
    # ------------------------------------------------------------------ #
    output_meta = NormalizedMetadata.load_template("VAM", "3")

    output_meta["DATE-BEG"] = date_beg.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    output_meta["DATE-END"] = date_end.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    output_meta["DATE-AVG"] = date_avg.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    output_meta["DATE-OBS"] = reference_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

    output_meta["DELTAT"] = delta_t
    output_meta["SPARSITY"] = sparsity
    output_meta["N_OFS"] = n_ofs
    output_meta["DELTA_PX"] = delta_px
    output_meta["KPSEXP"] = expected_wind_kps
    output_meta["BANDWDTH"] = annuli_width_rs
    output_meta["MAXRAD"] = round(max_elong_deg)
    output_meta["AZMBINS"] = azimuth_bins_remap
    output_meta["AZMBINF"] = az_bin
    output_meta["PLTBINS"] = flow_az_bins
    output_meta["YCENS"] = np.array2string(annuli_centers_rs, separator=",", max_line_width=10_000)
    output_meta["RBANDS"] = str(vel_bin_width)

    # avg_speeds has shape (n_annuli, flow_az_bins): numpy axis 0 = radius (annulus
    # index), axis 1 = azimuth.  FITS/WCS axes are reversed relative to numpy axes,
    # so WCS axis 1 (fastest-varying) = azimuth, WCS axis 2 = radius.
    cdelt_radius = float(np.mean(np.diff(annuli_centers_rs))) if len(annuli_centers_rs) > 1 else annuli_width_rs

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "azimuth", "radius"
    wcs.wcs.cunit = "deg", "solRad"
    wcs.wcs.cdelt = 360 / flow_az_bins, cdelt_radius
    wcs.wcs.crpix = 1, 1
    wcs.wcs.crval = 0, float(annuli_centers_rs[0])
    wcs.wcs.cname = "azimuth", "solar radii"
    wcs.array_shape = speeds.shape

    return PUNCHCube(data = speeds,
                  uncertainty=StdDevUncertainty(sigmas),
                  meta = output_meta,
                  wcs = wcs)
