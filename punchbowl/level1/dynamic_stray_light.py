import re
import pathlib
import warnings
from datetime import UTC, datetime
from itertools import pairwise

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from ndcube import NDCube
from prefect import get_run_logger

from punchbowl.data import NormalizedMetadata
from punchbowl.data.punch_io import load_many_cubes, load_ndcube_from_fits
from punchbowl.exceptions import IncorrectPolarizationStateError, IncorrectTelescopeError, InvalidDataError
from punchbowl.prefect import punch_flow, punch_task
from punchbowl.util import DataLoader, average_datetime, nan_percentile

fiducial_utime = datetime(2025, 1, 1,  tzinfo=UTC).timestamp() - 4 * 60


def fname_date_to_utime(timestamp: str) -> int:
    """Get a timestamp."""
    dt = datetime.strptime(timestamp, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    return int(dt.timestamp())


def fname_to_utime(fname: str) -> int:
    """Get a timestamp."""
    tstr = re.findall(r"_(\d{14})_", fname)[0]
    if len(tstr):
        return fname_date_to_utime(tstr)
    raise ValueError


def cube_to_utime(cube: NDCube) -> int:
    """Get a timestamp."""
    t = cube.meta.datetime.replace(tzinfo=UTC)
    return t.timestamp()


def phase_in_window(fname: str) -> int:
    """Get roll position phase."""
    utime = fname_to_utime(fname)
    return int(((utime - fiducial_utime))/60) % 8


def phase_in_window_from_cube(cube: NDCube) -> int:
    """Get roll position phase."""
    utime = cube_to_utime(cube)
    return int(((utime - fiducial_utime))/60) % 8


def make_phases(file_list: list[str]) -> list[list[str]]:
    """Group files by phase within roll position."""
    phases = [[],[],[],[],[],[],[],[]]
    for fname in file_list:
        phase = phase_in_window(fname)
        phases[phase].append(fname)
    return phases


def collect_pairs_by_phase(phases: list[list[str]], phase1: int, phase2: int) -> list[tuple[str, str]]:
    """Match pairs of files across two phases."""
    pairs = []
    j = 0
    for i in range(len(phases[phase1])):
        ti = fname_to_utime(phases[phase1][i])
        tj = fname_to_utime(phases[phase2][j])
        while tj > ti and j > 0:
            j = j - 1
            tj = fname_to_utime(phases[phase2][j])
        while tj <= ti and j < len(phases[phase2]) - 1:
            j = j + 1
            tj = fname_to_utime(phases[phase2][j])
        dt_min = (tj - ti) / 60
        if dt_min > 0 and dt_min < 8:
            pairs.append((phases[phase1][i], phases[phase2][j]))
    return pairs


@punch_flow
def construct_dynamic_stray_light_model(filepaths: list[str], reference_time: datetime | str, #noqa: C901
                                        pol_state: str, n_crota_bins: int = 24, n_loaders: int = 5) -> list[NDCube]:
    """Estimate time- and orbital-anomaly-dependent stray light."""
    logger = get_run_logger()

    if isinstance(reference_time, str):
        reference_time = datetime.strptime(reference_time, "%Y-%m-%d %H:%M:%S") # noqa: DTZ007

    phases = make_phases(filepaths)
    if pol_state == "P":
        i1, i2 = 1, 5
    elif pol_state == "Z":
        i1, i2 = 2, 6
    elif pol_state == "M":
        i1, i2 = 3, 7
    else:
        raise ValueError("Unrecognized polarization state")

    logger.info(f"Phase bin sizes: {[len(p) for p in phases]}")

    pairs = collect_pairs_by_phase(phases, i1, i2)

    logger.info(f"{len(pairs)} pairs of files found")

    bin_edges = np.linspace(-180, 180, n_crota_bins + 1)
    binned_pairs = [[] for _ in range(n_crota_bins)]
    for pair in pairs:
        try:
            header = fits.getheader(pair[0], 1)
            crota = header["CROTA"]
        except: # noqa: E722, S112
            continue
        for i, (start, stop) in enumerate(pairwise(bin_edges)):
            if start <= crota <= stop:
                binned_pairs[i].append(pair)
                break

    logger.info(f"Pairs per CROTA bin: {[len(bin) for bin in binned_pairs]}") # noqa: A001

    diff_maps = []
    date_obses = []
    for i, this_bin in enumerate(binned_pairs):
        first_half = load_many_cubes([pair[0] for pair in this_bin], include_uncertainty=False,
                                     include_provenance=False, n_workers=n_loaders, allow_errors=True)
        second_half = load_many_cubes([pair[1] for pair in this_bin], include_uncertainty=False,
                                      include_provenance=False, n_workers=n_loaders, allow_errors=True)
        this_bin = list(zip(first_half, second_half, strict=False)) # noqa: PLW2901
        # Exclude any pairs that raised an exception during loading
        this_bin = [p for p in this_bin if not isinstance(p[0], str) and not isinstance(p[1], str)] # noqa: PLW2901

        date_obses.extend(c.meta.datetime for pair in this_bin for c in pair)

        if len(this_bin) < 7:
            logger.info(f"Bin {i + 1} only loaded {len(this_bin)} valid pairs---recording zero dynamic stray light")
            diff_maps.append(np.zeros((2048, 2048)))
            continue

        diffs = np.empty((len(this_bin), *this_bin[0][0].data.shape))
        for j, pair in enumerate(this_bin):
            diffs[j] = pair[1].data - pair[0].data

        diff_map = nan_percentile(diffs, 50)

        diff_maps.append(diff_map)
        logger.info(f"Finished bin {i+1}")
    diff_maps = np.array(diff_maps)

    meta = NormalizedMetadata.load_template(
        "T" + header["TYPECODE"][1] + header["OBSCODE"], "1")
    meta["DATE-AVG"] = average_datetime(date_obses).strftime("%Y-%m-%dT%H:%M:%S")
    meta["DATE-OBS"] = reference_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    meta["DATE-BEG"] = min(date_obses).strftime("%Y-%m-%dT%H:%M:%S")
    meta["DATE-END"] = max(date_obses).strftime("%Y-%m-%dT%H:%M:%S")
    meta["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    meta.history.add_now("dynamic stray light",
                         f"Generated with {len(date_obses)} files running from "
                         f"{min(date_obses).strftime('%Y-%m-%dT%H:%M:%S')} to "
                         f"{max(date_obses).strftime('%Y-%m-%dT%H:%M:%S')}")
    meta["FILEVRSN"] = header["FILEVRSN"]

    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message=".*CROTA.*Human-readable solar north pole angle.*")
        wcs = WCS(header)
    wcs.wcs.pc = np.eye(2)
    return [NDCube(data=diff_maps, meta=meta, wcs=wcs)]


@punch_task
def remove_dynamic_stray_light_task(cube: NDCube, # noqa: C901
                                    before_cube: NDCube | str | DataLoader,
                                    after_cube: NDCube | str | DataLoader,
                                    ) -> None:
    """Interpolate and remove time- and orbital-anomaly-dependent stray light."""
    if before_cube is None or after_cube is None:
        cube.meta.history.add_now("LEVEL1-remove_stray_light", "Stray light correction skipped")

    if isinstance(before_cube, DataLoader):
        before_cube = before_cube.load()
    elif not isinstance(before_cube, NDCube):
        stray_light_before_path = pathlib.Path(before_cube)
        if not stray_light_before_path.exists():
            msg = f"File {stray_light_before_path} does not exist."
            raise InvalidDataError(msg)
        before_cube = load_ndcube_from_fits(stray_light_before_path)

    if isinstance(after_cube, DataLoader):
        after_cube = after_cube.load()
    elif not isinstance(after_cube, NDCube):
        stray_light_after_path = pathlib.Path(after_cube)
        if not stray_light_after_path.exists():
            msg = f"File {stray_light_after_path} does not exist."
            raise InvalidDataError(msg)
        after_cube = load_ndcube_from_fits(stray_light_after_path)

    for model in before_cube, after_cube:
        if model.meta["TELESCOP"].value != cube.meta["TELESCOP"].value:
            msg=f"Incorrect TELESCOP value within {model['FILENAME'].value}"
            raise IncorrectTelescopeError(msg)
        if model.meta["OBSLAYR1"].value != cube.meta["OBSLAYR1"].value:
            msg=f"Incorrect polarization state within {model['FILENAME'].value}"
            raise IncorrectPolarizationStateError(msg)
        if model.data.shape[1:] != cube.data.shape:
            msg = f"Incorrect stray light function shape within {model['FILENAME'].value}"
            raise InvalidDataError(msg)

    bin_edges = np.linspace(-180, 180, before_cube.shape[0] + 1)
    bin_centers = bin_edges[:-1] / 2 + bin_edges[1:] / 2
    bin_width = 360 / len(bin_centers)
    crota = cube.meta["CROTA"].value
    # CROTA falls within [-180, 180]

    for bin_idx, (start, stop) in enumerate(pairwise(bin_edges)): # noqa: B007
        if start <= crota <= stop:
            break

    if bin_idx == 0 and crota < bin_centers[0]:
        before_bin = -1
        after_bin = 0
        fpos = 1 - (bin_centers[0] - crota) / bin_width
    elif bin_idx == len(bin_centers) - 1 and crota > bin_centers[-1]:
        before_bin = -1
        after_bin = 0
        fpos = (crota - bin_centers[-1]) / bin_width
    else:
        if crota > bin_centers[bin_idx]:
            before_bin = bin_idx
            after_bin = bin_idx + 1
        else:
            before_bin = bin_idx - 1
            after_bin = bin_idx
        fpos = (crota - bin_centers[before_bin]) / bin_width

    before_at_orbit_pos = before_cube.data[before_bin] * (1 - fpos) + before_cube.data[after_bin] * fpos
    after_at_orbit_pos = after_cube.data[before_bin] * (1 - fpos) + after_cube.data[after_bin] * fpos

    before_date = before_cube.meta.datetime
    after_date = after_cube.meta.datetime
    date = cube.meta.datetime

    fpos = (date - before_date) / (after_date - before_date)

    data_interpolated = before_at_orbit_pos * (1 - fpos) + after_at_orbit_pos * fpos

    phase = phase_in_window_from_cube(cube)
    # The difference map is (later image) - (earlier image)
    # Pos is SL in the later image
    # Neg is SL in the earlier image
    # Zero is equal (and hopefully small) SL in both images
    if phase < 4:
        data_interpolated[data_interpolated > 0] = 0
        cube.data[...] += data_interpolated
    else:
        data_interpolated[data_interpolated < 0] = 0
        cube.data[...] -= data_interpolated

    cube.uncertainty.array[...] = np.sqrt(cube.uncertainty.array**2 + np.sqrt(np.abs(data_interpolated))**2)
    cube.meta.history.add_now("LEVEL1-remove_dynamic_stray_light_task",
                                     f"stray light removed with {before_cube.meta['FILENAME'].value} "
                                     f"and {after_cube.meta['FILENAME'].value}")
