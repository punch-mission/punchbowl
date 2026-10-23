import os
import pathlib
from datetime import datetime, timedelta

import numpy as np
import pytest
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS

import punchbowl.level3.velocity as velocity_module
from punchbowl.data import NormalizedMetadata, write_ndcube_to_fits
from punchbowl.data.punchcube import PUNCHCube
from punchbowl.level3.flow import generate_level3_velocity_flow
from punchbowl.level3.velocity import (
    check_missing_runs,
    get_delta_t_frames,
    get_flow_window_start,
    get_frame_slots,
    get_frames_per_window,
    get_required_frame_count,
    get_time_avg_size,
    infer_product_code,
    preprocess_cube,
    track_velocity,
)

THIS_DIRECTORY = pathlib.Path(__file__).parent.resolve()
OBS_DAY = datetime(2026, 1, 1, 0, 0, 0)

# --------------------------------------------------------------------------- #
# Small-scale versions of the production parameters, so the tests stay fast. #
# --------------------------------------------------------------------------- #
PRODUCT_CODE = "PTM"
CADENCE_MIN = 4  # minutes between consecutive PTM frames, as used by track_velocity
# With fewer frame pairs, the sub-pixel peak fit of the synthetic drift is biased beyond the test tolerance
FLOW_WINDOW_HOURS = 0.4
FRAMES_PER_WINDOW = 6  # FLOW_WINDOW_HOURS at the PTM cadence
DELTA_T_MINUTES = CADENCE_MIN  # a 1-frame lag, over which the synthetic ring drifts by RING_DRIFT_PX
SPARSITY = 1
N_OFS = 21
AZIMUTH_BINS_REMAP = 360
AZ_BIN = 4
VEL_BIN_WIDTH = 10
TIME_AVG_HOURS = 0.3  # 18 min: a 5-frame kernel, so a 2-frame temporal-average buffer on either side
ANNULI_CENTERS_RS = (3.0, 4.0)
ANNULI_WIDTH_RS = 1.0

# Synthetic outflow: a ring of emission drifting radially outward by
# RING_DRIFT_PX pixels per frame, azimuthally modulated so every radial row has
# azimuthal structure for the standardization step.
RING_START_PX = 32.0
RING_DRIFT_PX = 2.0
RING_SIGMA_PX = 1.5
RING_AMPLITUDE = 10.0
NOISE_LEVEL = 0.2
IMAGE_SHAPE = (128, 128)

# PUNCH WFI native pixel scale (deg/pixel). Written to the synthetic WCS and read back by track_velocity.
CDELT_DEG = 0.0225

# track_velocity measures radial speeds in km/s. Its internal conversion is
# CDELT_DEG * 3600 = 81 arcsec/polar-pixel * ~727 km/arcsec = ~58,900 km per polar pixel, and the
# effective time step between correlated frames is DELTA_T_MINUTES.
ARCSEC_PER_POLAR_PX = CDELT_DEG * 3600
KM_PER_POLAR_PX = ARCSEC_PER_POLAR_PX * (4.84814e-6 * 150e6)
EFFECTIVE_CADENCE_SEC = DELTA_T_MINUTES * 60
EXPECTED_WIND_KPS = int(round(RING_DRIFT_PX * KM_PER_POLAR_PX / EFFECTIVE_CADENCE_SEC))

TEST_PARAMS = {
    "flow_window_hours": FLOW_WINDOW_HOURS,
    "delta_t_minutes": DELTA_T_MINUTES,
    "sparsity": SPARSITY,
    "n_ofs": N_OFS,
    "expected_wind_kps": EXPECTED_WIND_KPS,
    "offset_speed_kps": 200,
    "speed_max": 3000,
    "annuli_centers_rs": ANNULI_CENTERS_RS,
    "annuli_width_rs": ANNULI_WIDTH_RS,
    "azimuth_bins_remap": AZIMUTH_BINS_REMAP,
    "az_bin": AZ_BIN,
    "vel_bin_width": VEL_BIN_WIDTH,
    "time_avg_hours": TIME_AVG_HOURS,
    "remove_temporal_median": False,
}


def _num_files_required() -> int:
    """Number of input files track_velocity requires for the test parameters."""
    # The odd-sized time-average kernel spans 5 frames, half of it on either side of the flow window
    return FRAMES_PER_WINDOW + 2 * 2


def _synthetic_frame(index: int) -> np.ndarray:
    """One synthetic image: an azimuthally-modulated ring drifting radially outward."""
    y, x = np.mgrid[0:IMAGE_SHAPE[0], 0:IMAGE_SHAPE[1]]
    center = (IMAGE_SHAPE[0] // 2 - 0.5, IMAGE_SHAPE[1] // 2 - 0.5)
    radius = np.hypot(x - center[1], y - center[0])
    theta = np.arctan2(y - center[0], x - center[1])
    ring_radius = RING_START_PX + RING_DRIFT_PX * index
    ring = np.exp(-((radius - ring_radius) ** 2) / (2 * RING_SIGMA_PX ** 2))
    azimuthal_modulation = 1 + 0.8 * np.cos(5 * theta + np.pi / 3)
    rng = np.random.default_rng(index)
    return RING_AMPLITUDE * ring * azimuthal_modulation + NOISE_LEVEL * rng.standard_normal(IMAGE_SHAPE)


def _write_synthetic_cube(file_path: str, frame_index: int, obs_time: datetime,
                          product_code: str = PRODUCT_CODE) -> None:
    """Write one synthetic frame, PTM by default, to a FITS file."""
    data = _synthetic_frame(frame_index)

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ("HPLN-AZP", "HPLT-AZP")
    wcs.wcs.cunit = ("deg", "deg")
    wcs.wcs.cdelt = (CDELT_DEG, CDELT_DEG)
    wcs.wcs.crpix = (IMAGE_SHAPE[1] // 2, IMAGE_SHAPE[0] // 2)
    wcs.wcs.crval = (0, 24.75)
    wcs.array_shape = data.shape

    meta = NormalizedMetadata.load_template(product_code, "3")
    meta["DATE-OBS"] = obs_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    meta["DATE-BEG"] = obs_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    meta["DATE-END"] = (obs_time + timedelta(minutes=CADENCE_MIN)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    meta["DATE-AVG"] = obs_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

    uncertainty = StdDevUncertainty(np.zeros_like(data))
    cube = PUNCHCube(data=data, wcs=wcs, meta=meta, uncertainty=uncertainty)
    write_ndcube_to_fits(cube, file_path)


@pytest.fixture
def synthetic_data(tmpdir):
    """
    Create a synthetic time series of PTM frames containing a ring of emission
    drifting radially outward at a known speed.

    Returns
    -------
    list of str
        Paths to the generated FITS files, spaced at the PTM cadence.

    """
    # Provide a few more files than strictly required; track_velocity only uses
    # the leading files it needs for the window plus its temporal-average buffer.
    num_files = _num_files_required() + 3
    spacing = timedelta(minutes=CADENCE_MIN)

    files = []
    for i in range(num_files):
        file_path = os.path.join(str(tmpdir), f"file_{i}.fits")
        _write_synthetic_cube(file_path, i, OBS_DAY + i * spacing)
        files.append(str(file_path))
    return files


def test_shape_matching(synthetic_data):
    """Test that the output shape matches the expected configuration."""
    files = synthetic_data
    result = track_velocity(files, product_code=PRODUCT_CODE, **TEST_PARAMS)

    assert isinstance(result, PUNCHCube)
    n_annuli = len(ANNULI_CENTERS_RS)
    flow_az_bins = (AZIMUTH_BINS_REMAP // AZ_BIN) // VEL_BIN_WIDTH
    assert result.data.shape == (n_annuli, flow_az_bins)
    assert result.uncertainty.array.shape == result.data.shape


def test_recovers_outflow_speed(synthetic_data):
    """Test that the recovered speed matches the synthetic ring's drift speed."""
    files = synthetic_data
    result = track_velocity(files, product_code=PRODUCT_CODE, **TEST_PARAMS)

    speeds = result.data
    assert np.isfinite(speeds).any(), "Data contains no valid speed measurement"
    # The ring is present at all azimuths, so nearly all of them should yield a speed
    assert np.isfinite(speeds).mean() > 0.9, "Most azimuths should yield a speed"
    assert np.median(speeds[np.isfinite(speeds)]) == pytest.approx(EXPECTED_WIND_KPS, rel=0.2)


def test_recovers_outflow_speed_with_delta_px_two(synthetic_data):
    """Test that delta_px=2 samples every other offset and still recovers the drift speed.

    The ring drifts RING_DRIFT_PX (even) pixels per frame and the calibrated
    central_offset is even, so the delta_px=2 offset grid contains the true
    displacement exactly.  The coarse grid spaces the speed samples ~500 km/s
    apart, so the peak-search range is adjusted for the peak finder: it needs
    at least 10 in-range samples for its Gaussian fit to run, the peak must
    sit inside the range rather than on its edge, and the range must stay
    below the spurious far-lag correlations of the margin-free correlation
    window.  offset_speed_kps=-5000 includes the sub-peak flank and
    speed_max=2000 satisfies all three conditions.
    """
    files = synthetic_data
    result = track_velocity(files, product_code=PRODUCT_CODE,
                            **dict(TEST_PARAMS, delta_px=2, offset_speed_kps=-5000, speed_max=2000))

    speeds = result.data
    assert np.isfinite(speeds).any(), "Data contains no valid speed measurement"
    assert np.isfinite(speeds).mean() > 0.9, "Most azimuths should yield a speed"
    assert np.median(speeds[np.isfinite(speeds)]) == pytest.approx(EXPECTED_WIND_KPS, rel=0.2)


def test_delta_px_is_forwarded_to_correlation_sampling(synthetic_data, monkeypatch):
    """Test that delta_px reaches the correlation sampling and the velocity axis.

    Regression test: track_velocity used to accept delta_px but silently drop
    it, always correlating with 1-px steps and building the velocity axis with
    a hardcoded delta_px=1, so only the DELTA_PX metadata keyword reflected the
    caller's setting.
    """
    seen_correlation_delta_px = []
    seen_axis_delta_px = []
    original_correlation = velocity_module.calculate_cross_correlation
    original_axis = velocity_module.build_velocity_axis

    def recording_correlation(image1, image2, offsets, central_offset, margin, delta_px=1, k=4.0):
        seen_correlation_delta_px.append(delta_px)
        return original_correlation(image1, image2, offsets, central_offset, margin,
                                     delta_px=delta_px, k=k)

    def recording_axis(n_ofs, central_offset, expected_wind_kps, delta_px=1):
        seen_axis_delta_px.append(delta_px)
        return original_axis(n_ofs, central_offset, expected_wind_kps, delta_px=delta_px)

    monkeypatch.setattr(velocity_module, "calculate_cross_correlation", recording_correlation)
    monkeypatch.setattr(velocity_module, "build_velocity_axis", recording_axis)

    files = synthetic_data
    track_velocity(files, product_code=PRODUCT_CODE, **dict(TEST_PARAMS, delta_px=2))

    # Every correlated pair, including any All-NaN retry, must use delta_px
    assert seen_correlation_delta_px, "calculate_cross_correlation was never called"
    assert set(seen_correlation_delta_px) == {2}
    assert seen_axis_delta_px == [2]


def test_insufficient_files_raises_value_error(synthetic_data):
    """Test that too few input files raise a ValueError."""
    files = synthetic_data
    # The temporal-average buffer requires more than a handful of frames
    with pytest.raises(ValueError):
        track_velocity(files[: max(_num_files_required() - 1, 1)], product_code=PRODUCT_CODE, **TEST_PARAMS)


def test_incompatible_geometry_raises_value_error(synthetic_data):
    """Test that azimuthal geometry that cannot be evenly divided raises a ValueError."""
    files = synthetic_data

    # azimuth_bins_remap must be divisible by az_bin
    bad_remap = dict(TEST_PARAMS, azimuth_bins_remap=AZIMUTH_BINS_REMAP + 1)
    with pytest.raises(ValueError):
        track_velocity(files, product_code=PRODUCT_CODE, **bad_remap)

    # the binned azimuthal axis must be divisible by vel_bin_width
    bad_bin_width = dict(TEST_PARAMS, vel_bin_width=AZIMUTH_BINS_REMAP // AZ_BIN + 1)
    with pytest.raises(ValueError):
        track_velocity(files, product_code=PRODUCT_CODE, **bad_bin_width)


def test_required_frame_count_matches_time_avg_kernel():
    """Test that the buffer on either side of the flow window is half of the time-average kernel."""
    n_files, tstart, tend = get_required_frame_count(PRODUCT_CODE, FLOW_WINDOW_HOURS, TIME_AVG_HOURS)
    assert n_files == _num_files_required()
    assert tstart == get_time_avg_size(PRODUCT_CODE, TIME_AVG_HOURS) // 2
    # The flow window [tstart, tend] is inclusive and holds exactly frames_per_window frames
    assert tend - tstart + 1 == FRAMES_PER_WINDOW


@pytest.mark.parametrize("product_code", ["CAM", "PAM", "CTM", "PTM"])
@pytest.mark.parametrize("time_avg_hours", [0.2, 3, 6, 12])
def test_required_frame_count_fits_time_avg_kernel(product_code, time_avg_hours):
    """Test that the time-average kernel centered on any frame of the flow window only covers actual frames."""
    n_files, tstart, tend = get_required_frame_count(product_code, 6, time_avg_hours)
    time_avg_size = get_time_avg_size(product_code, time_avg_hours)
    assert time_avg_size % 2 == 1
    assert tstart - time_avg_size // 2 >= 0
    assert tend + time_avg_size // 2 <= n_files - 1


def test_infer_product_code(synthetic_data, tmpdir):
    """Test that the product code is read from the files and that mixed product types are rejected."""
    assert infer_product_code(synthetic_data) == PRODUCT_CODE

    ctm_file = os.path.join(str(tmpdir), "ctm_file.fits")
    _write_synthetic_cube(ctm_file, 0, OBS_DAY, product_code="CTM")
    with pytest.raises(ValueError, match="same product type"):
        infer_product_code([*synthetic_data, ctm_file])


def test_product_code_mismatch_stops_velocity_flow(synthetic_data):
    """Test that the velocity flow stops when the files are not the expected product type."""
    with pytest.raises(ValueError, match="Expected CAM input files"):
        generate_level3_velocity_flow.fn(synthetic_data, product_code="CAM", **TEST_PARAMS)


@pytest.mark.parametrize("frames_per_window", [4, 5, 6, 7])
@pytest.mark.parametrize("offset_min", [-1.5, 0, 1.5])
def test_flow_window_is_centered_on_reference_time(frames_per_window, offset_min):
    """Test that the flow window holds frames_per_window frames centered on the reference time, for odd and even sizes."""
    cadence = timedelta(minutes=CADENCE_MIN)
    obs_times = [OBS_DAY + i * cadence for i in range(30)]
    reference_time = obs_times[15] + timedelta(minutes=offset_min)

    tstart = get_flow_window_start(obs_times, reference_time, frames_per_window)
    tend = tstart + frames_per_window - 1
    window = obs_times[tstart:tend + 1]
    assert len(window) == frames_per_window

    half = frames_per_window // 2
    if frames_per_window % 2 == 1:
        # The middle frame is the one nearest to the reference time
        middle = window[half]
        assert all(abs(middle - reference_time) <= abs(t - reference_time) for t in obs_times)
    else:
        # The reference time falls between the two middle frames: half the frames on either side
        assert sum(t < reference_time for t in window) == half
        assert sum(t >= reference_time for t in window) == half


def test_reference_time_centers_window_and_sets_date_obs(synthetic_data):
    """Test that the flow window is centered on the given reference time, which becomes DATE-OBS."""
    _, tstart, tend = get_required_frame_count(PRODUCT_CODE, FLOW_WINDOW_HOURS, TIME_AVG_HOURS)
    cadence = timedelta(minutes=CADENCE_MIN)
    file_first = 2  # the fixture has 3 spare files
    # FRAMES_PER_WINDOW is even, so the reference time falls between the two middle frames
    i_mid = file_first + tstart + FRAMES_PER_WINDOW // 2
    reference_time = OBS_DAY + i_mid * cadence - cadence / 2

    result = track_velocity(synthetic_data, product_code=PRODUCT_CODE, reference_time=reference_time, **TEST_PARAMS)

    assert datetime.fromisoformat(result.meta["DATE-OBS"].value) == reference_time
    # DATE-BEG and DATE-END are those of the first and last frames of the flow window
    assert datetime.fromisoformat(result.meta["DATE-BEG"].value) == OBS_DAY + (file_first + tstart) * cadence
    assert datetime.fromisoformat(result.meta["DATE-END"].value) == OBS_DAY + (file_first + tend + 1) * cadence


def _reference_time_with_spare_files() -> tuple[datetime, int, int]:
    """Reference time centering the flow window two files after the first one, and flow window start and end."""
    _, tstart, tend = get_required_frame_count(PRODUCT_CODE, FLOW_WINDOW_HOURS, TIME_AVG_HOURS)
    file_first = 2
    i_ref = file_first + tstart + FRAMES_PER_WINDOW // 2
    return OBS_DAY + i_ref * timedelta(minutes=CADENCE_MIN), file_first + tstart, file_first + tend


def test_missing_frame_is_interpolated(synthetic_data):
    """Test that a missing frame in the flow window is interpolated, keeping the window centered."""
    reference_time, i_start, i_end = _reference_time_with_spare_files()
    cadence = timedelta(minutes=CADENCE_MIN)
    files = [f for i, f in enumerate(synthetic_data) if i != i_start + 2]

    result = track_velocity(files, product_code=PRODUCT_CODE, reference_time=reference_time, **TEST_PARAMS)

    assert datetime.fromisoformat(result.meta["DATE-OBS"].value) == reference_time
    assert datetime.fromisoformat(result.meta["DATE-BEG"].value) == OBS_DAY + i_start * cadence
    assert datetime.fromisoformat(result.meta["DATE-END"].value) == OBS_DAY + (i_end + 1) * cadence
    # The synthetic ring is thinner than its drift between frames, so an interpolated frame holds two faint
    # rings rather than one in between, which biases the speed: only check that speeds are measured.
    assert np.isfinite(result.data).mean() > 0.9


def test_missing_first_frame_of_flow_window_is_interpolated(synthetic_data):
    """Test that a missing frame at the start of the selection is interpolated from the frame before it."""
    reference_time, i_start, _ = _reference_time_with_spare_files()
    _, tstart, _ = get_required_frame_count(PRODUCT_CODE, FLOW_WINDOW_HOURS, TIME_AVG_HOURS)
    # The first required frame is missing, and the frame before it is used for the interpolation
    files = [f for i, f in enumerate(synthetic_data) if i != i_start - tstart]
    result = track_velocity(files, product_code=PRODUCT_CODE, reference_time=reference_time, **TEST_PARAMS)
    assert datetime.fromisoformat(result.meta["DATE-OBS"].value) == reference_time


def test_too_many_missing_frames_raises(synthetic_data):
    """Test that a run of 3 missing PTM frames in the flow window is rejected."""
    reference_time, i_start, _ = _reference_time_with_spare_files()
    files = [f for i, f in enumerate(synthetic_data) if i not in (i_start + 1, i_start + 2, i_start + 3)]
    with pytest.raises(ValueError, match="Too many missing PTM frames in the flow window"):
        track_velocity(files, product_code=PRODUCT_CODE, reference_time=reference_time, **TEST_PARAMS)


@pytest.mark.parametrize(("product_code", "runs", "accepted"), [
    ("CAM", [], True),
    ("CAM", [1], True),
    ("CAM", [1, 1], True),
    ("CAM", [2], False),
    ("CAM", [1, 1, 1], False),
    ("PAM", [1, 1], True),
    ("PAM", [2], False),
    ("PTM", [2], True),
    ("PTM", [2, 1], True),
    ("PTM", [1, 1, 1], True),
    ("PTM", [2, 2], True),
    ("PTM", [3], False),
    ("PTM", [1, 1, 1, 1], False),
    ("PTM", [2, 2, 1], False),
    ("CTM", [2, 2], True),
    ("CTM", [2, 1, 1], False),
])
def test_missing_frame_rules(product_code, runs, accepted):
    """Test the tolerated patterns of missing frames for each product."""
    assert check_missing_runs(runs, product_code) is accepted


@pytest.mark.parametrize(("product_code", "expected"), [("CAM", 12), ("PAM", 12), ("CTM", 45), ("PTM", 90)])
def test_frames_per_window_spans_flow_window_hours(product_code, expected):
    """Test that the flow window spans at least 6 hours, with one more frame when just below."""
    assert get_frames_per_window(product_code, 6) == expected
    assert get_frames_per_window(PRODUCT_CODE, FLOW_WINDOW_HOURS) == FRAMES_PER_WINDOW


@pytest.mark.parametrize(("product_code", "expected"), [("CAM", 2), ("PAM", 2), ("CTM", 8), ("PTM", 16)])
def test_delta_t_frames_spans_delta_t_minutes(product_code, expected):
    """Test that the default 64-min lag is the same physical time lag for every product."""
    assert get_delta_t_frames(product_code, 64) == expected


def test_delta_t_shorter_than_half_a_frame_raises():
    with pytest.raises(ValueError, match="delta_t_minutes"):
        get_delta_t_frames("CAM", 10)


def test_frame_slots_with_gaps_and_jitter():
    """Test that missing frames get empty slots with interpolated times, despite timing jitter."""
    cadence = timedelta(minutes=CADENCE_MIN)
    jitter = timedelta(seconds=7)
    obs_times = [OBS_DAY, OBS_DAY + cadence + jitter, OBS_DAY + 4 * cadence - jitter, OBS_DAY + 5 * cadence]
    slots, slot_times = get_frame_slots(obs_times, CADENCE_MIN)
    assert slots == [0, 1, None, None, 2, 3]
    assert slot_times[2] == obs_times[1] + (obs_times[2] - obs_times[1]) / 3


def test_preprocess_cube_interpolates_missing_frame(synthetic_data):
    """Test that a missing frame is the linear interpolation of its neighbors."""
    files = synthetic_data[:4]
    cube, headers = preprocess_cube(files, product=PRODUCT_CODE, do_polar_remap=False)
    gappy_cube, gappy_headers = preprocess_cube([files[0], None, None, files[3]], product=PRODUCT_CODE,
                                                do_polar_remap=False)
    np.testing.assert_allclose(gappy_cube[1], (2 * cube[0] + cube[3]) / 3)
    np.testing.assert_allclose(gappy_cube[2], (cube[0] + 2 * cube[3]) / 3)
    assert datetime.fromisoformat(gappy_headers[1]["DATE-OBS"]) == datetime.fromisoformat(headers[1]["DATE-OBS"])


def test_reference_time_without_enough_files_raises(synthetic_data):
    """Test that a reference time too close to the end of the files raises a ValueError."""
    with pytest.raises(ValueError, match="Not enough frames for flow tracking"):
        track_velocity(synthetic_data, product_code=PRODUCT_CODE, reference_time=OBS_DAY, **TEST_PARAMS)


def test_default_reference_time_is_window_midpoint(synthetic_data):
    """Test that without a reference time, DATE-OBS is the middle of the flow window, not the first file time."""
    result = track_velocity(synthetic_data, product_code=PRODUCT_CODE, **TEST_PARAMS)
    assert result.meta["DATE-OBS"].value == result.meta["DATE-AVG"].value
    assert datetime.fromisoformat(result.meta["DATE-OBS"].value) > OBS_DAY
