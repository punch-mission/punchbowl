import os
from datetime import UTC, datetime

import numpy as np
import pytest
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from astropy.wcs.utils import add_stokes_axis_to_wcs
from ndcube import NDCube

from punchbowl.data.meta import NormalizedMetadata
from punchbowl.level1.despike import despike_polseq


@pytest.fixture
def sample_ndcube_for_despike():
    def _sample_ndcube(data, code="PM1", level="0"):
        uncertainty = StdDevUncertainty(np.sqrt(np.abs(data)))
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"
        wcs.wcs.cunit = "deg", "deg"
        wcs.wcs.cdelt = 0.1, 0.1
        wcs.wcs.crpix = 0, 0
        wcs.wcs.crval = 1, 1
        wcs.wcs.cname = "HPC lon", "HPC lat"

        if level in ["2", "3"] and code[0] == "P":
            wcs = add_stokes_axis_to_wcs(wcs, 2)

        meta = NormalizedMetadata.load_template(code, level)
        meta['DATE-OBS'] = str(datetime(2024, 2, 22, 16, 0, 1))
        meta['FILEVRSN'] = "1"
        meta["RAWBITS"] = 16
        meta["COMPBITS"] = 10
        meta["GAINBTM"] = 4.9
        meta["GAINTOP"] = 4.9
        meta["OFFSET"] = 100
        meta["EXPTIME"] = 49
        meta["DSATVAL"] = 100
        return NDCube(data=data, uncertainty=uncertainty, wcs=wcs, meta=meta)
    return _sample_ndcube

@pytest.mark.parametrize("num_neighbors", [2, 3, 4, 5, 6])
def test_spikes_are_removed(num_neighbors, sample_ndcube_for_despike):
    reference = np.random.random((300, 300))
    reference[30, 20] = 10

    neighbors = [np.random.random((300, 300)) for _ in range(num_neighbors)]

    sample_data = sample_ndcube_for_despike(reference, code="PM1", level="0")
    sample_neighbors = [sample_ndcube_for_despike(n, code="PM1", level="0") for n in neighbors]

    despiked, spike_map = despike_polseq(sample_data, sample_neighbors)

    print("MAX", np.max(despiked.data))
    assert np.all(despiked.data <= 10)
    assert spike_map[-1][30, 20] == 1
