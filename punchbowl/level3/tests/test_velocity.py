import os
import pathlib

import numpy as np
import pytest
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCube

from punchbowl.data import NormalizedMetadata, write_ndcube_to_fits
from punchbowl.level3.velocity import plot_flow_map, track_velocity

THIS_DIRECTORY = pathlib.Path(__file__).parent.resolve()


@pytest.fixture
def synthetic_data(tmpdir):
    """
    Create synthetic compressed FITS data from NDCube instances for testing.

    This fixture generates a list of file paths for FITS files containing random
    NDCube data. These files are written to a temporary directory and removed
    after the test session. Each file includes:
    - A 128x128 array of random data.
    - WCS metadata specifying helioprojective coordinates.
    - Normalized metadata according to the specified schema.
    - Uncertainty data initialized to zero.

    Returns:
        list of str: Paths to the generated FITS files.
    """
    files = []
    num_files = 5  # You can parameterize this if needed

    for i in range(num_files):
        # Generate random data
        data = np.random.rand(128, 128)

        # Define WCS for the NDCube
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ("HPLN-AZP", "HPLT-AZP")
        wcs.wcs.cunit = ("deg", "deg")
        wcs.wcs.cdelt = (0.02, 0.02)
        wcs.wcs.crpix = (64, 64)
        wcs.wcs.crval = (0, 24.75)
        wcs.array_shape = data.shape

        # Define metadata for the NDCube
        meta = NormalizedMetadata.load_template('PTM', '3')
        meta['DATE-OBS'] = "2024-01-01T00:00:00"
        meta['DATE-BEG'] = "2024-01-01T00:00:00"
        meta['DATE-END'] = "2024-01-01T00:00:00"
        meta['DATE-AVG'] = "2024-01-01T00:00:00"

        # Create NDCube
        uncertainty = StdDevUncertainty(np.zeros_like(data))
        cube = NDCube(data=data, wcs=wcs, meta=meta, uncertainty=uncertainty)

        # Write NDCube to a compressed FITS file
        file_path = os.path.join(str(tmpdir), f"file_{i}.fits")
        write_ndcube_to_fits(cube, str(file_path))
        files.append(str(file_path))

    return files


def test_shape_matching(synthetic_data):
    """Test that the output shape matches the expected configuration."""
    files = synthetic_data
    ycens = np.arange(7, 14.5, 0.5)
    result = track_velocity(files, ycens=ycens)

    assert isinstance(result, NDCube)
    assert result.data.shape[0] == len(ycens)


@pytest.mark.mpl_image_compare(style="default")
def test_wind_plot(synthetic_data):
    """Tests that wind plots are generated and  output to file."""
    files = synthetic_data
    ycens = np.arange(7, 14.5, 0.5)
    result = track_velocity(files, ycens=ycens)

    return plot_flow_map(None, result)


def test_no_nans_or_negatives(synthetic_data):
    """Test that the output does not contain NaNs or negative values."""
    files = synthetic_data
    result = track_velocity(files)

    assert not np.isnan(result.data).any(), "Data contains NaNs"
    assert (result.data >= 0).all(), "Data contains negative values"


def test_with_bad_data(tmpdir):
    """Test the function with intentionally bad data."""
    # Generate bad data (all NaNs)
    data = np.full((128, 128), np.nan)

    # Define WCS for the NDCube
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "HPLN-AZP", "HPLT-AZP"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.cdelt = 0.02, 0.02
    wcs.wcs.crpix = 64, 64
    wcs.wcs.crval = 0, 24.75

    # Define metadata for the NDCube
    meta = NormalizedMetadata.load_template('PTM', '3')
    meta['DATE-OBS'] = "2024-01-01T00:00:00"
    meta['DATE-BEG'] = "2024-01-01T00:00:00"
    meta['DATE-END'] = "2024-01-01T00:00:00"
    meta['DATE-AVG'] = "2024-01-01T00:00:00"

    # Create NDCube
    uncertainty = StdDevUncertainty(np.zeros_like(data))
    cube = NDCube(data=data, wcs=wcs, meta=meta, uncertainty=uncertainty)

    # Write NDCube to a compressed FITS file using your custom function
    file_path = os.path.join(str(tmpdir), "bad_file.fits")
    write_ndcube_to_fits(cube, file_path)

    with pytest.raises(ValueError):
        _ = track_velocity([str(file_path)])


def test_sample_radial_outflows(tmpdir):
    """Test the function with sample radial outflows."""
    files = []
    for i in range(5):
        radial_outflow_data = np.linspace(0, 1, 128)[:, None] * np.linspace(1, 0, 128)

        # Define WCS for the NDCube
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = "HPLN-AZP", "HPLT-AZP"
        wcs.wcs.cunit = "deg", "deg"
        wcs.wcs.cdelt = 0.02, 0.02
        wcs.wcs.crpix = 64, 64
        wcs.wcs.crval = 0, 24.75

        # Define metadata for the NDCube
        meta = NormalizedMetadata.load_template('PTM', '3')
        meta['DATE-OBS'] = "2024-01-01T00:00:00"
        meta['DATE-BEG'] = "2024-01-01T00:00:00"
        meta['DATE-END'] = "2024-01-01T00:00:00"
        meta['DATE-AVG'] = "2024-01-01T00:00:00"

        # Create NDCube
        uncertainty = StdDevUncertainty(np.zeros_like(radial_outflow_data))
        cube = NDCube(data=radial_outflow_data, wcs=wcs, meta=meta, uncertainty=uncertainty)

        # Write NDCube to a compressed FITS file
        file_path = os.path.join(str(tmpdir), f"radial_outflow_file_{i}.fits")
        write_ndcube_to_fits(cube, file_path)
        files.append(str(file_path))

    result = track_velocity(files)

    assert isinstance(result, NDCube)
    assert result.data.mean() > 0  # Verify that there is a positive outflow signal
