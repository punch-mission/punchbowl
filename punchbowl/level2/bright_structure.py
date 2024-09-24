
import numpy as np
from ndcube import NDCube
from prefect import get_run_logger, task
from skimage.morphology import binary_dilation

from punchbowl.data import load_ndcube_from_fits


def run_zspike(
    data: np.ndarray,
    uncertainty: np.ndarray,
    threshold: float = 4,
    required_yes: int = 6,
    veto_limit: int = 2,
    diff_method: str = "sigma",
    dilation: int = 0,
    index_of_interest: int | None = None,
) -> np.ndarray:
    """
    Identify bright structures in temporal series of images.

    Given a time sequence of images, identify "spikes" that exceed a
    threshold in a single frame.

    Diffuse bright structures over the background F-corona are
    identified and marked in the data using in-band "bad value" marking
    (which is supported by the FITS standard). Data marking follows the
    existing ZSPIKE temporal despiking algorithm to identify auroral
    transients.

    This is a voting algorithm based on my ZSPIKE.PRO from the 1990s,
    which is available in the solarsoft distribution from Lockheed Martin.

    ZSPIKE was originally used to identify cosmic rays, and
    was adopted on STEREO for on-board despiking during exposure accumulation.
    For this application ZSPIKE is ideal because it does not rely on the
    spatial structure of spikes, only their temporal structure. Both cosmic
    rays and, if present, high-altitude aurora are transient and easily
    detected with ZSPIKE.

    The algorithm assembles "votes" from the images
    surrounding each one in a stream, to determine whether a particular pixel
    is a good candidate for a temporal spike. If the pixel is sufficiently
    bright compared to its neighbors in time, it is marked bad. "Bad values"
    are stored in the DRP for file quality marking outlined in Module Quality
    Marking.

    There are two methods of identifying if a pixel is above a given threshold,
    and therefore considered a spike. diff_method='abs' or 'sigma'.

    diff_method 'abs' represents the absolute difference, and is the default.
    If set, this is an absolute difference, in DN, required for a pixel to
    'vote' its central value.  If the central value is this much higher than a
    given voting value, then the central value is voted to be a spike.  If
    it's this much lower, the veto count is incremented.

    If diff_method 'sigma' is set if, then each pixel is treated as a
    time series and the calculated sigma (RMS variation from the mean) of
    the timeseries is used to calculate a difference threshold at each
    location.

    The threshold is the value over which a pixel is voted as a spike. The threshold
    should be different depending diff_method.

    Parameters
    ----------
    data : np.ndarray
        data to operate on - this is expected to have an odd number of frames, or
        a frame_of_interest should be inserted.

    uncertainty: np.ndarray
        data to operate on - this is expected to have an odd number of frames, or
        a frame_of_interest should be inserted.

    threshold : float
        This is the threshold over which a pixel is voted as a spike.

    required_yes: int
        (default is 4) - number of 'voting' frames that must vote the
        value is a spike, for it to be marked as such.

    veto_limit: int
        (default is 2) - number of 'voting' frames that must vote NO to veto
        marking the central value as a spike.

    diff_method: str
        This is the method by which the threshold is set. 'abs' treats each pixel
        independently and finds the absolute difference, and 'sigma' treats each
        corresponding pixel as a time series and the calculated RMS variation
        from the mean of the timeseries is used to calculate a difference at each
        location.

    dilation: int
        If nonzero, this is the number of times to morphologically dilate pixels marked as bright structures.

    index_of_interest : int
        if you have an even number of frames, or you don't want to find spikes
        in the center frame, the frame_of_interest will be used. index_of_interest
        starts from 0, therefore the center frame from 21 frames will be 10.

    Returns
    -------
    np.ndarray
        a frame matching the dimensions of the frame of interest with True flagging
        bad pixels, and False flagging good data points

    """
    if len(data.shape) != 3:
        msg = "`data` must be a 3-D array"
        raise ValueError(msg)

    # test if odd number of frames, or if a frame of interest has been included
    z_shape = np.shape(data[:, 0, 0])

    if z_shape[0] % 2 == 0:
        if index_of_interest is None:
            msg = "Number of frames in `data` must be odd or have `frame_of_interest` set."
            raise ValueError(msg)
    elif index_of_interest is None:
        index_of_interest = z_shape[0] // 2

    frame_of_interest = data[index_of_interest, :, :]
    frame_of_interest_uncertainty = uncertainty[index_of_interest, :, :]

    voters_array = np.delete(data, index_of_interest, 0)
    voters_array_uncertainty = np.delete(uncertainty, index_of_interest, 0)

    # create a reference cube the same dimensions as the voter cube, filled with frame of interest
    reference_cube = np.stack([frame_of_interest for _ in range(voters_array.shape[0])], axis=0)
    difference_array = np.abs(reference_cube - voters_array)

    if diff_method == "abs":
        threshold_array = np.full(voters_array.shape, threshold)
    elif diff_method == "sigma":
        threshold_array = threshold * np.nanstd(voters_array, axis=0, where=voters_array_uncertainty < 1.0)
    else:
        msg = f"A `diff_method` of `sigma` or `abs` is expected. Found diff_method={diff_method}."
        raise ValueError(msg)

    yes_vote_count = np.sum(difference_array > threshold_array, axis=0)
    no_vote_count = np.full(frame_of_interest.shape, voters_array.shape[0]) - yes_vote_count
    flagged_features_array = yes_vote_count > required_yes

    # if the number of no votes exceeds a veto limit, then veto
    veto_mask = no_vote_count > veto_limit

    flagged_features_array[veto_mask] = False

    # if input data already has an unbounded uncertainty, create a True flag
    flagged_features_array = np.where(np.isinf(frame_of_interest_uncertainty), True, flagged_features_array)

    # expand flags by dilating to remove holes or fuzzy edges
    for _ in range(dilation):
        binary_dilation(flagged_features_array, out=flagged_features_array)

    return flagged_features_array


@task
def identify_bright_structures_task(
    data: NDCube,
    voter_filenames: list[str],
    threshold: float = 4,
    required_yes: int = 6,
    veto_limit: int = 2,
    diff_method: str = "sigma",
    dilation: int = 0,
) -> NDCube:
    """Prefect task to perform bright structure identification."""
    logger = get_run_logger()
    logger.info("identify_bright_structures_task started")

    if len(voter_filenames) == 0:
        data.meta.history.add_now("LEVEL2-bright_structure",
                                  "Identify bright structures skipped since no voters provided")
        return data

    # construct voter cube
    voters, voters_uncertainty = [], []
    for voter_filename in voter_filenames:
        this_punchdata = load_ndcube_from_fits(voter_filename)
        voters.append(this_punchdata.data)
        voters_uncertainty.append(this_punchdata.uncertainty)

    # add frame of interest
    voters.append(data.data)
    voters_uncertainty.append(data.uncertainty)

    # apply find spikes
    spike_mask = run_zspike(
        data=np.array(voters),
        uncertainty=np.array(voters_uncertainty),
        threshold=threshold,
        required_yes=required_yes,
        veto_limit=veto_limit,
        diff_method=diff_method,
        dilation=dilation,
        index_of_interest=-1)

    # add the uncertainty to the output punch data object
    data.uncertainty.array[spike_mask] = np.inf

    logger.info("identify_bright_structures_task ended")
    data.meta.history.add_now("LEVEL2-bright_structures",
                              "bright structure identification completed")

    return data
