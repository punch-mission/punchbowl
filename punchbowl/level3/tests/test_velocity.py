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
from punchbowl.level3.velocity import get_buffer, track_velocity

THIS_DIRECTORY = pathlib.Path(__file__).parent.resolve()

# --------------------------------------------------------------------------- #
# Small-scale versions of the production parameters, so the tests stay fast. #
# --------------------------------------------------------------------------- #
PRODUCT_CODE = "PTM"
CADENCE_MIN = 4  # minutes between consecutive PTM frames, as used by track_velocity
FRAMES_PER_WINDOW = 4
DELTA_T = 1
SPARSITY = 1
N_OFS = 21
AZIMUTH_BINS_REMAP = 360
AZ_BIN = 4
VEL_BIN_WIDTH = 10
BUFFER_TARGET_HOURS = 0.2  # 12 min, exactly the time spanned by the window: buffer == frames_per_window
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
# effective time step between correlated frames is CADENCE_MIN * DELTA_T minutes.
ARCSEC_PER_POLAR_PX = CDELT_DEG * 3600
KM_PER_POLAR_PX = ARCSEC_PER_POLAR_PX * (4.84814e-6 * 150e6)
EFFECTIVE_CADENCE_SEC = CADENCE_MIN * DELTA_T * 60
EXPECTED_WIND_KPS = int(round(RING_DRIFT_PX * KM_PER_POLAR_PX / EFFECTIVE_CADENCE_SEC))

TEST_PARAMS = {
    "frames_per_window": FRAMES_PER_WINDOW,
    "delta_t": DELTA_T,
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
    "buffer_target_hours": BUFFER_TARGET_HOURS,
    "remove_temporal_median": False,
}


def _num_files_required() -> int:
    """Number of input files track_velocity requires for the test parameters."""
    buffer = get_buffer(FRAMES_PER_WINDOW, DELTA_T, CADENCE_MIN, target_hours=BUFFER_TARGET_HOURS)
    return FRAMES_PER_WINDOW + 2 * (buffer // 2) + 1


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


def _write_synthetic_cube(file_path: str, frame_index: int, obs_time: datetime) -> None:
    """Write one synthetic PTM frame to a FITS file."""
    data = _synthetic_frame(frame_index)

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ("HPLN-AZP", "HPLT-AZP")
    wcs.wcs.cunit = ("deg", "deg")
    wcs.wcs.cdelt = (CDELT_DEG, CDELT_DEG)
    wcs.wcs.crpix = (IMAGE_SHAPE[1] // 2, IMAGE_SHAPE[0] // 2)
    wcs.wcs.crval = (0, 24.75)
    wcs.array_shape = data.shape

    meta = NormalizedMetadata.load_template("PTM", "3")
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
    obs_day = datetime(2026, 1, 1, 0, 0, 0)
    spacing = timedelta(minutes=CADENCE_MIN)

    files = []
    for i in range(num_files):
        file_path = os.path.join(str(tmpdir), f"file_{i}.fits")
        _write_synthetic_cube(file_path, i, obs_day + i * spacing)
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
